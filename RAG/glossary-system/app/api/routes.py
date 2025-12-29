import logging
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from app.models.schemas import (
    TranslationRequest, 
    TranslationResponse,
    PDFUploadResponse,
    SessionStatusResponse,
    PDFExtractionResponse,
    SentenceProcessRequest,
    SentenceProcessResponse,
    BatchProcessResponse,
    SessionCleanupResponse,
    PDFGlossaryTermsResponse
)
from app.services.optimized_glossary_service import OptimizedGlossaryService
from app.services.pdf_service import PDFService
from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services with lazy loading for better startup performance
_glossary_service = None
_pdf_service = None


def get_glossary_service() -> OptimizedGlossaryService:
    """Lazy initialization of glossary service."""
    global _glossary_service
    if _glossary_service is None:
        _glossary_service = OptimizedGlossaryService()
    return _glossary_service


def get_pdf_service() -> PDFService:
    """Lazy initialization of PDF service."""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service


# Backward compatible aliases
def get_glossary_service_instance() -> OptimizedGlossaryService:
    return get_glossary_service()

def get_pdf_service_instance() -> PDFService:
    return get_pdf_service()


# =====================================================
# SENTENCE TRANSLATION ENDPOINT
# =====================================================

@router.post("/translate/sentence", response_model=TranslationResponse)
async def translate_sentence(
    request: TranslationRequest,
    clear_cache: bool = Query(False, description="Clear cache for this domain before processing")
):
    """
    Translate a single sentence using glossary lookup.
    
    - **text**: Source sentence to process
    - **source_lang**: Source language code (en, ar, fr)
    - **target_lang**: Target language code (en, ar, fr)
    - **domain**: Domain for glossary lookup (health, agriculture, etc.)
    - **clear_cache**: If true, clears the cache for this domain before processing
    """
    # Validation: Source and target languages must be different
    if request.source_lang == request.target_lang:
        raise HTTPException(
            status_code=400, 
            detail="Source and target languages must be different"
        )

    try:
        service = get_glossary_service()
        
        # Clear cache if requested
        if clear_cache:
            service.invalidate_domain_cache(request.domain)
            logger.info(f"Cache cleared for domain: {request.domain}")
        
        result = await service.process_request_async(
            request.text,
            request.source_lang,
            request.target_lang,
            request.domain
        )
        return result
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error processing sentence: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Internal server error processing glossary"
        )


# =====================================================
# PDF UPLOAD & SESSION INITIALIZATION
# =====================================================

@router.post("/translate/pdf", response_model=PDFUploadResponse, status_code=201)
async def upload_pdf_session(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
    source_lang: str = Form(..., description="Source language code"),
    target_lang: str = Form(..., description="Target language code"),
    domain: str = Form(..., description="Domain for glossary lookup"),
    auto_process: bool = Form(False, description="Start processing immediately in background")
):
    """
    Upload a PDF and initialize a translation session.
    
    This endpoint:
    1. Validates the uploaded PDF file
    2. Stores the file securely
    3. Loads domain-specific glossary into Redis cache
    4. Creates a session for subsequent processing
    
    - **file**: PDF file (max 50MB)
    - **source_lang**: Source language (en, ar, fr)
    - **target_lang**: Target language (en, ar, fr)
    - **domain**: Domain (health, agriculture, history, finance, legal, technology)
    - **auto_process**: If true, starts background processing immediately
    """
    # Debug logging
    logger.info(f"PDF Upload - filename: {file.filename}, content_type: {file.content_type}")
    
    # 1. Validate file - check extension OR content-type
    filename = file.filename or ""
    is_pdf_extension = filename.lower().endswith('.pdf')
    is_pdf_content_type = file.content_type == "application/pdf"
    
    if not is_pdf_extension and not is_pdf_content_type:
        raise HTTPException(
            status_code=400, 
            detail=f"File must be a PDF document. Received: filename='{filename}', content_type='{file.content_type}'"
        )
    
    # 2. Validate domain
    if domain not in settings.ALLOWED_DOMAINS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid domain. Allowed domains: {settings.ALLOWED_DOMAINS}"
        )
    
    # 3. Validate languages
    if source_lang not in settings.ALLOWED_LANGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid source language. Allowed: {settings.ALLOWED_LANGS}"
        )
    
    if target_lang not in settings.ALLOWED_LANGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid target language. Allowed: {settings.ALLOWED_LANGS}"
        )
    
    if source_lang == target_lang:
        raise HTTPException(
            status_code=400, 
            detail="Source and target languages must be different"
        )

    # 4. Initialize session
    try:
        response = await get_pdf_service().initialize_session(
            file, source_lang, target_lang, domain
        )
        
        # 5. Optionally trigger background processing
        if auto_process:
            background_tasks.add_task(
                _process_pdf_background,
                response['session_id']
            )
            response['auto_processing'] = True
            response['status'] = 'processing'
        else:
            response['auto_processing'] = False
        
        logger.info(f"PDF session initialized: {response['session_id']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session initialization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize session: {str(e)}"
        )


async def _process_pdf_background(session_id: str) -> None:
    """Background task to process PDF after upload."""
    try:
        logger.info(f"Starting background processing for session: {session_id}")
        result = get_pdf_service().process_pdf_batch(session_id)
        logger.info(
            f"Background processing completed for {session_id}: "
            f"{result.get('total_sentences', 0)} sentences, "
            f"{result.get('total_matches', 0)} matches"
        )
    except Exception as e:
        logger.error(f"Background processing failed for {session_id}: {e}", exc_info=True)


# =====================================================
# SESSION STATUS
# =====================================================

@router.get("/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Get the current status of a PDF processing session.
    
    - **session_id**: Session identifier returned from PDF upload
    """
    try:
        status = get_pdf_service().get_session_status(session_id)
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve session status"
        )


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

@router.get("/session/{session_id}/extract", response_model=PDFExtractionResponse)
async def extract_pdf_text(
    session_id: str,
    pages: Optional[str] = Query(
        None, 
        description="Comma-separated page numbers (1-indexed), e.g., '1,2,5'"
    )
):
    """
    Extract text from the uploaded PDF.
    
    - **session_id**: Active session ID
    - **pages**: Optional specific pages to extract (comma-separated)
    """
    try:
        # Parse page numbers if provided
        page_numbers = None
        if pages:
            try:
                page_numbers = [int(p.strip()) for p in pages.split(',') if p.strip()]
                if any(p < 1 for p in page_numbers):
                    raise ValueError("Page numbers must be positive")
            except ValueError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid page numbers format: {str(e)}"
                )
        
        result = get_pdf_service().extract_text_from_pdf(session_id, page_numbers)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to extract text from PDF"
        )


# =====================================================
# GET ALL GLOSSARY TERMS FROM PDF
# =====================================================

@router.get("/session/{session_id}/glossary-terms", response_model=PDFGlossaryTermsResponse)
async def get_pdf_glossary_terms(
    session_id: str,
    pages: Optional[str] = Query(
        None, 
        description="Comma-separated page numbers (1-indexed), e.g., '1,2,5'"
    ),
    include_context: bool = Query(
        False,
        description="Include example sentences where each term appears"
    )
):
    """
    Get ALL glossary terms found in the uploaded PDF.
    
    This endpoint scans the entire PDF (or specific pages) and returns:
    - All unique glossary terms found
    - Their translations from the glossary
    - Number of occurrences in the PDF
    - Optionally, example sentences containing each term
    
    - **session_id**: Active session ID from PDF upload
    - **pages**: Optional specific pages to scan (comma-separated)
    - **include_context**: If true, includes up to 3 example sentences per term
    """
    try:
        # Parse page numbers if provided
        page_numbers = None
        if pages:
            try:
                page_numbers = [int(p.strip()) for p in pages.split(',') if p.strip()]
                if any(p < 1 for p in page_numbers):
                    raise ValueError("Page numbers must be positive")
            except ValueError as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid page numbers format: {str(e)}"
                )
        
        result = get_pdf_service().get_all_glossary_terms_from_pdf(
            session_id=session_id,
            page_numbers=page_numbers,
            include_context=include_context
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract glossary terms: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to extract glossary terms from PDF"
        )


# =====================================================
# SINGLE SENTENCE PROCESSING (with session)
# =====================================================

@router.post("/session/{session_id}/process/sentence", response_model=SentenceProcessResponse)
async def process_session_sentence(session_id: str, request: SentenceProcessRequest):
    """
    Process a single sentence within an active session.
    
    Uses cached glossary for fast lookup.
    
    - **session_id**: Active session ID
    - **sentence**: Sentence to process
    """
    try:
        # Get session to retrieve domain and language settings
        session = get_pdf_service().cache.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404, 
                detail="Session not found or expired"
            )
        
        result = get_pdf_service().process_sentence(
            sentence=request.sentence,
            session_id=session_id,
            domain=session.get('domain'),
            src=session.get('source_lang'),
            tgt=session.get('target_lang')
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentence processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to process sentence"
        )


# =====================================================
# BATCH PDF PROCESSING
# =====================================================

@router.post("/session/{session_id}/process/batch", response_model=BatchProcessResponse)
async def process_pdf_batch(
    session_id: str,
    background_tasks: BackgroundTasks,
    pages: Optional[str] = Query(
        None, 
        description="Comma-separated page numbers to process"
    ),
    batch_size: int = Query(10, ge=1, le=100, description="Sentences per batch"),
    async_mode: bool = Query(False, description="Process in background")
):
    """
    Process the entire PDF or specific pages for glossary matches.
    
    - **session_id**: Active session ID
    - **pages**: Optional specific pages (comma-separated)
    - **batch_size**: Number of sentences to process per batch
    - **async_mode**: If true, processes in background and returns immediately
    """
    try:
        # Parse page numbers
        page_numbers = None
        if pages:
            try:
                page_numbers = [int(p.strip()) for p in pages.split(',') if p.strip()]
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid page numbers format"
                )
        
        if async_mode:
            # Start background processing
            background_tasks.add_task(
                _batch_process_background,
                session_id,
                page_numbers,
                batch_size
            )
            return {
                "session_id": session_id,
                "status": "processing",
                "message": "Processing started in background. Check session status for updates.",
                "total_pages": 0,
                "total_sentences": 0,
                "total_matches": 0,
                "results": [],
                "processing_time_ms": 0
            }
        
        # Synchronous processing
        result = get_pdf_service().process_pdf_batch(
            session_id=session_id,
            page_numbers=page_numbers,
            batch_size=batch_size
        )
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to process PDF batch"
        )


async def _batch_process_background(
    session_id: str, 
    page_numbers: Optional[List[int]], 
    batch_size: int
) -> None:
    """Background task for batch PDF processing."""
    try:
        logger.info(f"Starting batch processing for session: {session_id}")
        result = get_pdf_service().process_pdf_batch(
            session_id=session_id,
            page_numbers=page_numbers,
            batch_size=batch_size
        )
        logger.info(
            f"Batch processing completed for {session_id}: "
            f"{result.get('total_matches', 0)} total matches"
        )
    except Exception as e:
        logger.error(f"Batch processing failed for {session_id}: {e}", exc_info=True)


# =====================================================
# SESSION CLEANUP
# =====================================================

@router.delete("/session/{session_id}", response_model=SessionCleanupResponse)
async def cleanup_session(session_id: str):
    """
    Clean up a session and release resources.
    
    This removes:
    - Cached glossary data
    - Session metadata
    - Uploaded PDF file
    
    - **session_id**: Session to clean up
    """
    try:
        result = get_pdf_service().cleanup_session(session_id)
        
        if result.get('status') == 'not_found':
            raise HTTPException(
                status_code=404, 
                detail="Session not found"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to cleanup session"
        )


# =====================================================
# GLOSSARY TERMS QUERY BY DOMAIN
# =====================================================

@router.get("/glossary/terms/{domain}")
async def get_glossary_terms_by_domain(
    domain: str,
    source_lang: str = Query(..., description="Source language (en, ar, fr)"),
    target_lang: str = Query(..., description="Target language (en, ar, fr)"),
    limit: Optional[int] = Query(None, ge=1, le=10000, description="Max number of terms to return"),
    offset: int = Query(0, ge=0, description="Number of terms to skip")
):
    """
    Get all glossary terms for a specific domain and language pair.
    
    - **domain**: Domain name (health, agriculture, history, finance, legal, technology)
    - **source_lang**: Source language code
    - **target_lang**: Target language code
    - **limit**: Maximum number of terms to return (optional)
    - **offset**: Pagination offset
    """
    # Validate domain
    if domain not in settings.ALLOWED_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain. Allowed: {list(settings.ALLOWED_DOMAINS)}"
        )
    
    # Validate languages
    if source_lang not in settings.ALLOWED_LANGS or target_lang not in settings.ALLOWED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Allowed: {list(settings.ALLOWED_LANGS)}"
        )
    
    try:
        from app.core.database import get_db_connection
        
        query = """
            SELECT source_term, target_term, n_gram_size
            FROM glossary_terms
            WHERE domain = ? AND source_lang = ? AND target_lang = ?
            ORDER BY n_gram_size DESC
        """
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (domain, source_lang, target_lang))
            rows = cursor.fetchall()
            
            # Get total count
            count_query = """
                SELECT COUNT(*) as total
                FROM glossary_terms
                WHERE domain = ? AND source_lang = ? AND target_lang = ?
            """
            cursor.execute(count_query, (domain, source_lang, target_lang))
            total = cursor.fetchone()['total']
        
        terms = [
            {
                "source_term": r['source_term'],
                "target_term": r['target_term'],
                "n_gram_size": r['n_gram_size']
            }
            for r in rows
        ]
        
        return {
            "domain": domain,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "total_terms": total,
            "returned_terms": len(terms),
            "offset": offset,
            "terms": terms
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve glossary terms: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve glossary terms"
        )


# =====================================================
# CACHE STATISTICS (Admin/Debug)
# =====================================================

@router.get("/session/{session_id}/stats")
async def get_cache_stats(session_id: str):
    """
    Get cache statistics for a session (admin/debug endpoint).
    
    - **session_id**: Session to get stats for
    """
    try:
        session = get_pdf_service().cache.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404, 
                detail="Session not found"
            )
        
        domain = session.get('domain')
        stats = get_pdf_service().cache.get_session_stats(session_id, domain)
        
        return {
            "session": session,
            "cache_stats": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve cache statistics"
        )


# =====================================================
# SESSION MANAGEMENT (Admin)
# =====================================================

@router.get("/sessions")
async def list_all_sessions():
    """
    List ALL active sessions (admin endpoint).
    
    Returns all sessions currently stored in Redis with their metadata.
    """
    try:
        sessions = get_pdf_service().cache.list_all_sessions()
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to list sessions"
        )


@router.delete("/sessions")
async def close_all_sessions():
    """
    Close ALL active sessions and cleanup resources (admin endpoint).
    
    WARNING: This will:
    - Delete all cached glossary data
    - Remove all session metadata
    - Delete all uploaded PDF files
    
    Use with caution!
    """
    try:
        # Get all session IDs first to cleanup files
        session_ids = get_pdf_service().cache.get_all_session_ids()
        
        # Cleanup PDF files for each session
        files_deleted = 0
        for session_id in session_ids:
            session = get_pdf_service().cache.get_session(session_id)
            if session:
                file_path = session.get('pdf_path')
                if file_path:
                    get_pdf_service()._cleanup_file(file_path)
                    files_deleted += 1
        
        # Clear all sessions from Redis
        success = get_pdf_service().cache.clear_all_sessions()
        
        if success:
            logger.info(f"All sessions cleared: {len(session_ids)} sessions, {files_deleted} files")
            return {
                "status": "success",
                "message": "All sessions closed",
                "sessions_closed": len(session_ids),
                "files_deleted": files_deleted
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to clear sessions from cache"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close all sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to close all sessions"
        )


# =====================================================
# HEALTH CHECK ENDPOINTS
# =====================================================

@router.get("/health/services")
async def check_services_health():
    """
    Check health of all dependent services.
    """
    redis_healthy = get_pdf_service().cache.is_connected
    
    return {
        "status": "healthy" if redis_healthy else "degraded",
        "services": {
            "redis": {
                "status": "connected" if redis_healthy else "disconnected",
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT
            }
        },
        "config": {
            "max_pdf_size_mb": settings.MAX_PDF_SIZE_MB,
            "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
            "allowed_domains": list(settings.ALLOWED_DOMAINS),
            "allowed_langs": list(settings.ALLOWED_LANGS)
        }
    }


# =====================================================
# DATABASE MANAGEMENT ENDPOINTS
# =====================================================

@router.get("/database/stats")
async def get_database_stats():
    """
    Get database statistics including term counts by domain and language.
    """
    try:
        from app.core.database import get_db_connection
        
        with get_db_connection() as conn:
            # Get total term count
            cursor = conn.execute("SELECT COUNT(*) FROM glossary_terms")
            total_terms = cursor.fetchone()[0]
            
            # Get counts by domain
            cursor = conn.execute("""
                SELECT domain, COUNT(*) as count 
                FROM glossary_terms 
                GROUP BY domain 
                ORDER BY count DESC
            """)
            by_domain = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get counts by language pair
            cursor = conn.execute("""
                SELECT source_lang, target_lang, COUNT(*) as count 
                FROM glossary_terms 
                GROUP BY source_lang, target_lang
            """)
            by_lang_pair = [
                {"source": row[0], "target": row[1], "count": row[2]} 
                for row in cursor.fetchall()
            ]
            
            # Get database file size
            import os
            db_path = settings.DATABASE_URL.replace("file:", "")
            db_size_mb = 0.0
            if os.path.exists(db_path):
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            
            return {
                "total_terms": total_terms,
                "by_domain": by_domain,
                "by_language_pair": by_lang_pair,
                "database_size_mb": round(db_size_mb, 2),
                "database_path": db_path
            }
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve database statistics"
        )


@router.post("/database/terms")
async def add_term_to_database(
    source_term: str = Query(..., description="Source term"),
    target_term: str = Query(..., description="Target term"),
    source_lang: str = Query("en", description="Source language"),
    target_lang: str = Query("ar", description="Target language"),
    domain: str = Query("general", description="Domain"),
):
    """
    Add a new term to the glossary database.
    """
    try:
        from app.core.database import get_db_connection
        
        # Calculate n-gram size
        n_gram_size = len(source_term.split())
        
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO glossary_terms 
                (source_term, target_term, source_lang, target_lang, domain, n_gram_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source_term, target_term, source_lang, target_lang, domain, n_gram_size))
            conn.commit()
            
            # Get the inserted term ID
            cursor = conn.execute("SELECT last_insert_rowid()")
            term_id = cursor.fetchone()[0]
            
        logger.info(f"Added term: {source_term} -> {target_term} (ID: {term_id})")
        
        return {
            "status": "success",
            "message": "Term added successfully",
            "term": {
                "id": term_id,
                "source_term": source_term,
                "target_term": target_term,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domain": domain,
                "n_gram_size": n_gram_size
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to add term: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add term: {str(e)}"
        )


@router.get("/database/search")
async def search_terms(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=1000, description="Max results")
):
    """
    Search for terms in the database using FTS5 full-text search.
    """
    try:
        from app.core.database import get_db_connection
        
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT g.source_term, g.target_term, g.source_lang, g.target_lang, 
                       g.domain, g.n_gram_size
                FROM glossary_fts 
                JOIN glossary_terms g ON glossary_fts.rowid = g.id
                WHERE glossary_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            results = [
                {
                    "source_term": row[0],
                    "target_term": row[1],
                    "source_lang": row[2],
                    "target_lang": row[3],
                    "domain": row[4],
                    "n_gram_size": row[5]
                }
                for row in cursor.fetchall()
            ]
            
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/database/reset")
async def reset_database():
    """
    Reset the database by recreating the schema.
    WARNING: This will delete all existing terms!
    """
    try:
        from app.core.database import init_database_schema
        
        db_path = settings.DATABASE_URL.replace("file:", "")
        
        # Recreate schema (removes existing database file)
        init_database_schema(db_path)
        
        logger.info("Database reset completed successfully")
        
        return {
            "status": "success",
            "message": "Database reset successfully",
            "database_path": db_path
        }
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database reset failed: {str(e)}"
        )


@router.delete("/database")
async def delete_database():
    """
    Delete the entire database file.
    WARNING: This is irreversible! All data will be lost!
    """
    try:
        import os
        # from app.core.database import get_db_connection
        
        db_path = settings.DATABASE_URL.replace("file:", "")
        
        # Close all connections
        # get_db_connection.cache_clear()
        
        # Delete the file if it exists
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.warning(f"Database file deleted: {db_path}")
            
            return {
                "status": "success",
                "message": "Database deleted successfully",
                "deleted_path": db_path
            }
        else:
            return {
                "status": "success",
                "message": "Database file does not exist",
                "path": db_path
            }
            
    except Exception as e:
        logger.error(f"Database deletion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Database deletion failed: {str(e)}"
        )
