import uuid
import shutil
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import UploadFile, HTTPException

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from app.core.config import settings
from app.core.database import get_db_connection
from app.services.cache_service import CacheService
from app.utils.text_processor import normalize_text, tokenize, generate_ngrams

# Setup logging
logger = logging.getLogger(__name__)


class PDFService:
    """
    Production-ready PDF processing service for glossary lookups.
    
    Handles:
    - PDF upload and validation
    - Session management with Redis caching
    - Per-sentence glossary term extraction
    - N-gram based term matching with overlap removal
    """
    
    ALLOWED_EXTENSIONS = {'pdf'}
    
    def __init__(self):
        self.cache = CacheService()
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate required dependencies are available."""
        if pdfplumber is None:
            logger.warning("pdfplumber not installed. PDF processing will be unavailable.")

    # =====================================================
    # PHASE 1: SESSION INITIALIZATION
    # =====================================================
    
    async def initialize_session(
        self, 
        file: UploadFile, 
        src: str, 
        tgt: str, 
        domain: str
    ) -> Dict[str, Any]:
        """
        Initialize a PDF processing session.
        
        Steps:
        1. Validate file (size, type)
        2. Save file securely
        3. Load domain glossary from SQLite
        4. Cache glossary in Redis
        5. Create session metadata
        
        Args:
            file: Uploaded PDF file
            src: Source language code
            tgt: Target language code
            domain: Domain for glossary lookup
            
        Returns:
            Session initialization response with session_id
            
        Raises:
            HTTPException: For validation or processing errors
        """
        # 1. Validate file
        self._validate_file(file)
        
        # 2. Generate session ID and save file
        session_id = f"pdf_{uuid.uuid4().hex}"
        file_path = await self._save_file(file, session_id)
        
        try:
            # 3. Load domain glossary from SQLite
            glossary_terms = self._load_full_domain_glossary(domain, src, tgt)
            
            if not glossary_terms:
                logger.warning(f"No glossary terms found for domain={domain}, {src}->{tgt}")
            
            # 4. Bulk cache glossary in Redis
            cache_success = self.cache.bulk_cache_glossary(session_id, domain, glossary_terms)
            
            if not cache_success:
                logger.warning(f"Failed to cache glossary for session {session_id}")
            
            # 5. Create session metadata
            meta = {
                'session_id': session_id,
                'domain': domain,
                'source_lang': src,
                'target_lang': tgt,
                'created_at': datetime.utcnow().isoformat(),
                'pdf_path': file_path,
                'status': 'ready',
                'glossary_count': len(glossary_terms),
                'total_pages': self._get_pdf_page_count(file_path),
                'processed_sentences': 0
            }
            self.cache.create_session(session_id, meta)
            
            logger.info(f"Session {session_id} initialized with {len(glossary_terms)} glossary terms")
            
            return {
                "session_id": session_id,
                "status": "initialized",
                "domain": domain,
                "source_lang": src,
                "target_lang": tgt,
                "glossary_terms_loaded": len(glossary_terms),
                "total_pages": meta['total_pages'],
                "cache_expires_in_seconds": settings.CACHE_TTL_SECONDS
            }
            
        except Exception as e:
            # Cleanup on failure
            self._cleanup_file(file_path)
            logger.error(f"Session initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Session initialization failed: {str(e)}")

    def _validate_file(self, file: UploadFile) -> None:
        """
        Validate uploaded file.
        
        Checks:
        - File extension is PDF
        - File size within limits
        - File is not empty
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check extension
        file_ext = file.filename.rsplit('.', 1)[-1].lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {self.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size (read first chunk to estimate)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        max_size_bytes = settings.MAX_PDF_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_PDF_SIZE_MB}MB"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

    async def _save_file(self, file: UploadFile, session_id: str) -> str:
        """
        Save uploaded file securely.
        
        Returns:
            Path to saved file
        """
        file_ext = file.filename.rsplit('.', 1)[-1].lower()
        safe_filename = f"{session_id}.{file_ext}"
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        
        try:
            with open(file_path, "wb") as buffer:
                # Read and write in chunks for memory efficiency
                chunk_size = 1024 * 1024  # 1MB chunks
                while chunk := file.file.read(chunk_size):
                    buffer.write(chunk)
            
            logger.debug(f"File saved: {file_path}")
            return file_path
            
        except IOError as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    def _cleanup_file(self, file_path: str) -> None:
        """Remove file if it exists."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

    def _get_pdf_page_count(self, file_path: str) -> int:
        """Get total number of pages in PDF."""
        if pdfplumber is None:
            return 0
        try:
            with pdfplumber.open(file_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            logger.warning(f"Failed to get page count: {str(e)}")
            return 0

    def _load_full_domain_glossary(
        self, 
        domain: str, 
        src: str, 
        tgt: str
    ) -> List[Dict[str, Any]]:
        """
        Load ALL terms for a domain to prime the Redis cache.
        
        Args:
            domain: Domain filter
            src: Source language
            tgt: Target language
            
        Returns:
            List of glossary term dictionaries
        """
        query = """
            SELECT source_term, target_term, n_gram_size
            FROM glossary_terms
            WHERE domain = ? AND source_lang = ? AND target_lang = ?
            ORDER BY n_gram_size DESC
        """
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (domain, src, tgt))
                rows = cursor.fetchall()
                
            return [
                {
                    'source': r['source_term'], 
                    'target': r['target_term'], 
                    'n_size': r['n_gram_size']
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to load glossary: {str(e)}")
            return []

    # =====================================================
    # PHASE 2: PDF TEXT EXTRACTION
    # =====================================================
    
    def extract_text_from_pdf(
        self, 
        session_id: str, 
        page_numbers: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Extract text from PDF file.
        
        Args:
            session_id: Active session ID
            page_numbers: Optional list of specific pages to extract (1-indexed)
            
        Returns:
            Extracted text with page information
        """
        if pdfplumber is None:
            raise HTTPException(
                status_code=503, 
                detail="PDF processing unavailable. Install pdfplumber."
            )
        
        # Get session metadata
        session = self.cache.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        file_path = session.get('pdf_path')
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        try:
            extracted_pages = []
            full_text = []
            
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Determine which pages to process
                if page_numbers:
                    pages_to_process = [p - 1 for p in page_numbers if 0 < p <= total_pages]
                else:
                    pages_to_process = range(total_pages)
                
                for page_idx in pages_to_process:
                    page = pdf.pages[page_idx]
                    text = page.extract_text() or ""
                    
                    # Clean extracted text
                    text = self._clean_extracted_text(text)
                    
                    extracted_pages.append({
                        'page_number': page_idx + 1,
                        'text': text,
                        'char_count': len(text)
                    })
                    full_text.append(text)
            
            combined_text = "\n\n".join(full_text)
            
            # Update session with extraction status
            self.cache.update_session(session_id, {
                'status': 'text_extracted',
                'extracted_char_count': len(combined_text)
            })
            
            return {
                "session_id": session_id,
                "total_pages": total_pages,
                "pages_extracted": len(extracted_pages),
                "total_characters": len(combined_text),
                "pages": extracted_pages,
                "full_text": combined_text
            }
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted PDF text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()

    def split_into_sentences(self, text: str, lang: str = 'en') -> List[str]:
        """
        Split text into sentences for processing.
        
        Args:
            text: Full text to split
            lang: Language code for sentence splitting rules
            
        Returns:
            List of sentences
        """
        import re
        
        if not text:
            return []
        
        # Basic sentence splitting (can be enhanced with spaCy for better results)
        if lang == 'ar':
            # Arabic sentence delimiters
            sentences = re.split(r'[.؟!،]\s*', text)
        else:
            # English/French sentence delimiters
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences and normalize
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]
        
        return sentences

    # =====================================================
    # PHASE 3: PER-SENTENCE PROCESSING
    # =====================================================
    
    def process_sentence(
        self, 
        sentence: str, 
        session_id: str, 
        domain: str, 
        src: str, 
        tgt: str
    ) -> Dict[str, Any]:
        """
        Process a single sentence for glossary term extraction.
        
        Implements:
        - N-gram generation
        - Redis cache lookup
        - SQLite fallback for cache misses
        - Overlap removal and ranking
        
        Args:
            sentence: Input sentence to process
            session_id: Active session ID
            domain: Domain for glossary lookup
            src: Source language
            tgt: Target language
            
        Returns:
            Processing result with matched terms
        """
        import time
        start_time = time.time()
        
        # 1. Preprocessing
        clean_text = normalize_text(sentence, src)
        tokens = tokenize(clean_text, src)
        
        if len(tokens) < 1:
            return self._build_sentence_response([], sentence, domain, start_time)
        
        # 2. N-gram Generation
        ngram_data = generate_ngrams(tokens)
        
        if not ngram_data:
            return self._build_sentence_response([], sentence, domain, start_time)
        
        # 3. Prepare lookup keys
        lookup_keys = [(item['text'], item['n_size']) for item in ngram_data]
        
        # 4. Redis Cache Lookup
        matched_terms = []
        missed_ngrams = []
        
        try:
            redis_results = self.cache.lookup_ngrams(session_id, domain, lookup_keys)
            
            for i, res in enumerate(redis_results):
                ng_item = ngram_data[i]
                
                if res:
                    matched_terms.append({
                        'source': ng_item['text'],
                        'target': res.decode('utf-8') if isinstance(res, bytes) else res,
                        'n_size': ng_item['n_size'],
                        'freq': 9999,  # Cache hits get high priority
                        'start_idx': ng_item['start_idx'],
                        'cache_hit': True
                    })
                else:
                    missed_ngrams.append(ng_item)
                    
        except Exception as e:
            logger.warning(f"Redis lookup failed, falling back to SQLite: {str(e)}")
            missed_ngrams = ngram_data
        
        # 5. SQLite Fallback for Cache Misses
        if missed_ngrams:
            missed_texts = [ng['text'] for ng in missed_ngrams]
            fallback_matches = self._sqlite_fallback(missed_texts, domain, src, tgt)
            
            # Map fallback results back to ngram data for indices
            for fb_match in fallback_matches:
                for ng_item in missed_ngrams:
                    if ng_item['text'] == fb_match['source']:
                        fb_match['start_idx'] = ng_item['start_idx']
                        fb_match['cache_hit'] = False
                        break
                matched_terms.append(fb_match)
            
            # Update cache with fallback results (cache-aside pattern)
            if fallback_matches:
                self._update_cache_with_fallback(session_id, domain, fallback_matches)
        
        # 6. Overlap Removal and Ranking
        final_terms = self._remove_overlaps_and_rank(matched_terms, tokens)
        
        # 7. Build response
        return self._build_sentence_response(final_terms, sentence, domain, start_time)

    def _sqlite_fallback(
        self, 
        terms: List[str], 
        domain: str, 
        src: str, 
        tgt: str
    ) -> List[Dict[str, Any]]:
        """
        Fallback to SQLite FTS5 for terms not found in cache.
        
        Args:
            terms: List of terms to search
            domain: Domain filter
            src: Source language
            tgt: Target language
            
        Returns:
            List of matched terms from database
        """
        if not terms:
            return []
        
        try:
            all_matches = []
            batch_size = 50
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for i in range(0, len(terms), batch_size):
                    batch = terms[i:i + batch_size]
                    
                    # Build FTS5 MATCH query
                    escaped_terms = []
                    for term in batch:
                        escaped = term.replace('"', '""')
                        escaped_terms.append(f'"{escaped}"')
                    
                    fts_query = ' OR '.join(escaped_terms)
                    
                    query = """
                        SELECT g.source_term, g.target_term, g.n_gram_size
                        FROM glossary_terms g
                        INNER JOIN glossary_fts fts ON g.id = fts.rowid
                        WHERE g.domain = ? 
                        AND g.source_lang = ? 
                        AND g.target_lang = ?
                        AND fts.source_term MATCH ?
                        ORDER BY g.n_gram_size DESC
                    """
                    
                    cursor.execute(query, (domain, src, tgt, fts_query))
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        all_matches.append({
                            'source': row['source_term'],
                            'target': row['target_term'],
                            'n_size': row['n_gram_size']
                        })
            
            return all_matches
            
        except Exception as e:
            logger.error(f"SQLite fallback failed: {str(e)}")
            return []

    def _update_cache_with_fallback(
        self, 
        session_id: str, 
        domain: str, 
        matches: List[Dict[str, Any]]
    ) -> None:
        """Update Redis cache with fallback results (cache-aside pattern)."""
        try:
            for match in matches:
                self.cache.cache_single_term(
                    session_id, 
                    domain, 
                    match['source'], 
                    match['target'],
                    match.get('n_size', 1)
                )
        except Exception as e:
            logger.warning(f"Failed to update cache: {str(e)}")

    def _remove_overlaps_and_rank(
        self, 
        matched_terms: List[Dict[str, Any]], 
        tokens: List[str],
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping matches and rank by n-gram size 
        
        Prioritizes:
        1. Larger n-grams (more specific terms)
        2. Non-overlapping coverage
        
        Args:
            matched_terms: List of matched terms with indices
            tokens: Original tokenized sentence
            max_results: Maximum number of results to return
            
        Returns:
            Filtered and ranked list of terms
        """
        if not matched_terms:
            return []
        
        # Sort by n-gram size (desc)
        matched_terms.sort(key=lambda x: x.get('n_size', 1), reverse=True)
        
        selected_terms = []
        covered_indices = set()
        
        for match in matched_terms:
            n_size = match.get('n_size', 1)
            start_idx = match.get('start_idx')
            
            if start_idx is None:
                # Try to find the term in tokens
                source_tokens = match['source'].split()
                for i in range(len(tokens) - len(source_tokens) + 1):
                    if ' '.join(tokens[i:i + len(source_tokens)]) == match['source']:
                        start_idx = i
                        break
            
            if start_idx is not None:
                term_indices = set(range(start_idx, start_idx + n_size))
                
                # Check for overlap
                if not term_indices.intersection(covered_indices):
                    selected_terms.append({
                        'source_term': match['source'],
                        'target_term': match['target'],
                        'n_gram_size': n_size,
                        'cache_hit': match.get('cache_hit', False)
                    })
                    covered_indices.update(term_indices)
                    
                    if len(selected_terms) >= max_results:
                        break
        
        return selected_terms

    def _build_sentence_response(
        self, 
        matches: List[Dict[str, Any]], 
        sentence: str, 
        domain: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Build standardized response for sentence processing."""
        import time
        
        return {
            "source_sentence": sentence,
            "domain": domain,
            "glossary_matches": matches,
            "match_count": len(matches),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    # =====================================================
    # PHASE 4: BATCH PROCESSING
    # =====================================================
    
    def process_pdf_batch(
        self, 
        session_id: str,
        page_numbers: Optional[List[int]] = None,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Process entire PDF or specific pages in batches.
        
        Args:
            session_id: Active session ID
            page_numbers: Optional specific pages to process
            batch_size: Number of sentences to process per batch
            
        Returns:
            Complete processing results
        """
        import time
        start_time = time.time()
        
        # Get session
        session = self.cache.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        domain = session.get('domain')
        src = session.get('source_lang')
        tgt = session.get('target_lang')
        
        # Extract text
        extraction_result = self.extract_text_from_pdf(session_id, page_numbers)
        full_text = extraction_result.get('full_text', '')
        
        # Split into sentences
        sentences = self.split_into_sentences(full_text, src)
        
        if not sentences:
            return {
                "session_id": session_id,
                "status": "completed",
                "total_sentences": 0,
                "results": [],
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # Process sentences in batches
        all_results = []
        total_matches = 0
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            for sentence in batch:
                result = self.process_sentence(sentence, session_id, domain, src, tgt)
                all_results.append(result)
                total_matches += result.get('match_count', 0)
        
        # Update session
        self.cache.update_session(session_id, {
            'status': 'completed',
            'processed_sentences': len(sentences),
            'total_matches': total_matches
        })
        
        return {
            "session_id": session_id,
            "status": "completed",
            "total_pages": extraction_result.get('total_pages', 0),
            "total_sentences": len(sentences),
            "total_matches": total_matches,
            "results": all_results,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }

        return persisted_count

    # =====================================================
    # SESSION MANAGEMENT
    # =====================================================
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status and metadata."""
        session = self.cache.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        return {
            "session_id": session_id,
            "status": session.get('status', 'unknown'),
            "domain": session.get('domain'),
            "source_lang": session.get('source_lang'),
            "target_lang": session.get('target_lang'),
            "glossary_count": session.get('glossary_count', 0),
            "total_pages": session.get('total_pages', 0),
            "processed_sentences": session.get('processed_sentences', 0),
            "created_at": session.get('created_at')
        }

    def cleanup_session(self, session_id: str) -> Dict[str, str]:
        """
        Clean up session resources.
        
        Removes:
        - Cached glossary data
        - Session metadata
        - Uploaded PDF file
        """
        session = self.cache.get_session(session_id)
        
        if session:
            # Remove PDF file
            file_path = session.get('pdf_path')
            if file_path:
                self._cleanup_file(file_path)
            
            # Clear cache
            self.cache.delete_session(session_id)
            
            logger.info(f"Session {session_id} cleaned up")
            return {"status": "deleted", "session_id": session_id}
        
        return {"status": "not_found", "session_id": session_id}

    # =====================================================
    # GET ALL GLOSSARY TERMS FROM PDF
    # =====================================================
    
    def get_all_glossary_terms_from_pdf(
        self, 
        session_id: str,
        page_numbers: Optional[List[int]] = None,
        include_context: bool = False
    ) -> Dict[str, Any]:
        """
        Extract PDF text and find ALL matching glossary terms.
        
        This method:
        1. Extracts text from the PDF
        2. Scans the entire text for glossary term matches
        3. Returns unique terms with their translations and occurrence counts
        
        Args:
            session_id: Active session ID
            page_numbers: Optional specific pages to process
            include_context: If True, include sentence context for each term
            
        Returns:
            Dictionary with all found glossary terms and metadata
        """
        import time
        start_time = time.time()
        
        # Get session metadata
        session = self.cache.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        domain = session.get('domain')
        src = session.get('source_lang')
        tgt = session.get('target_lang')
        
        # Extract text from PDF
        extraction_result = self.extract_text_from_pdf(session_id, page_numbers)
        full_text = extraction_result.get('full_text', '')
        
        if not full_text.strip():
            return {
                "session_id": session_id,
                "status": "completed",
                "domain": domain,
                "source_lang": src,
                "target_lang": tgt,
                "total_pages": extraction_result.get('total_pages', 0),
                "unique_terms_found": 0,
                "total_occurrences": 0,
                "glossary_terms": [],
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
        
        # Split into sentences for context
        sentences = self.split_into_sentences(full_text, src)
        
        # Track found terms with occurrences and context
        found_terms = {}  # key: source_term, value: term info dict
        
        # Process each sentence
        for sentence in sentences:
            # Normalize and tokenize
            clean_text = normalize_text(sentence, src)
            tokens = tokenize(clean_text, src)
            
            if len(tokens) < 1:
                continue
            
            # Generate n-grams
            ngram_data = generate_ngrams(tokens)
            
            if not ngram_data:
                continue
            
            # Prepare lookup keys
            lookup_keys = [(item['text'], item['n_size']) for item in ngram_data]
            
            # Look up in cache
            try:
                redis_results = self.cache.lookup_ngrams(session_id, domain, lookup_keys)
                
                for i, res in enumerate(redis_results):
                    if res:
                        ng_item = ngram_data[i]
                        source_term = ng_item['text']
                        target_term = res.decode('utf-8') if isinstance(res, bytes) else res
                        
                        if source_term in found_terms:
                            # Update occurrence count
                            found_terms[source_term]['occurrences'] += 1
                            if include_context and sentence not in found_terms[source_term]['contexts']:
                                found_terms[source_term]['contexts'].append(sentence)
                        else:
                            # Add new term
                            found_terms[source_term] = {
                                'source_term': source_term,
                                'target_term': target_term,
                                'n_gram_size': ng_item['n_size'],
                                'occurrences': 1,
                                'contexts': [sentence] if include_context else []
                            }
            except Exception as e:
                logger.warning(f"Cache lookup failed, trying SQLite: {str(e)}")
                # Fallback to SQLite if cache fails
                ngram_texts = [ng['text'] for ng in ngram_data]
                fallback_matches = self._sqlite_fallback(ngram_texts, domain, src, tgt)
                
                for match in fallback_matches:
                    source_term = match['source']
                    if source_term in found_terms:
                        found_terms[source_term]['occurrences'] += 1
                    else:
                        found_terms[source_term] = {
                            'source_term': source_term,
                            'target_term': match['target'],
                            'n_gram_size': match.get('n_size', 1),
                            'occurrences': 1,
                            'contexts': [sentence] if include_context else []
                        }
        
        # Convert to list and sort by occurrences (most frequent first)
        terms_list = list(found_terms.values())
        terms_list.sort(key=lambda x: (-x['occurrences'], -x['n_gram_size']))
        
        # Limit contexts to prevent huge responses
        if include_context:
            for term in terms_list:
                term['contexts'] = term['contexts'][:3]  # Max 3 context sentences
        
        total_occurrences = sum(t['occurrences'] for t in terms_list)
        
        # Update session with results
        self.cache.update_session(session_id, {
            'status': 'terms_extracted',
            'unique_terms_found': len(terms_list),
            'total_occurrences': total_occurrences
        })
        
        return {
            "session_id": session_id,
            "status": "completed",
            "domain": domain,
            "source_lang": src,
            "target_lang": tgt,
            "total_pages": extraction_result.get('total_pages', 0),
            "total_sentences_scanned": len(sentences),
            "unique_terms_found": len(terms_list),
            "total_occurrences": total_occurrences,
            "glossary_terms": terms_list,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }