"""
API Routes for RAG System
FastAPI endpoints for semantic translation memory
"""

import asyncio
import logging
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks

from app.models.schemas import (
    SemanticSearchRequest,
    HybridSearchRequest,
    Phase1IntegrationRequest,
    SearchResponse,
    Phase2Output,
    MatchResult,
    CollectionInfo,
    HealthResponse,
    StatsResponse,
    DomainPairsResponse,
    SetupResponse
)
from app.core.config import settings
from app.services.setup_qdrant import get_qdrant_client, get_collection_info, verify_qdrant_connection
from app.services.optimized_retrieval_service import OptimizedRetriever
from app.services.pipeline import retrieve_fuzzy_matches
from app.services.integration import integrate_with_phase1, format_for_prompt
from app.services.caching import get_cache, retrieve_fuzzy_matches_cached
from app.utils.error_handling import validate_query, validate_domain, validate_language
from app.utils.monitoring import get_monitor, get_query_logger

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (initialized on startup)
_retriever: Optional[OptimizedRetriever] = None
_retriever_lock = asyncio.Lock()


async def get_retriever() -> OptimizedRetriever:
    """Get or create retriever instance (thread-safe)."""
    global _retriever
    
    if _retriever is None:
        async with _retriever_lock:
            if _retriever is None:
                client = get_qdrant_client()
                _retriever = OptimizedRetriever(client)
    
    return _retriever


# =====================================================
# SEMANTIC SEARCH ENDPOINTS
# =====================================================

@router.post("/search/semantic", response_model=SearchResponse)
async def search_semantic(request: SemanticSearchRequest):
    """
    Semantic search for similar translations.
    
    Finds translations with similar MEANING using cross-lingual embeddings.
    Best for finding conceptually similar translations.
    """
    start_time = time.time()
    
    # Validate inputs
    is_valid, error = validate_query(request.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    is_valid, error = validate_domain(request.domain)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        retriever = await get_retriever()
        cache = get_cache()
        
        # Try cache first
        cached_results = cache.get(
            request.query, request.domain, 
            request.source_lang, request.target_lang, request.top_k
        )
        
        if cached_results is not None:
            elapsed = (time.time() - start_time) * 1000
            return SearchResponse(
                query=request.query,
                domain=request.domain,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                total_results=len(cached_results),
                results=cached_results,
                cached=True,
                elapsed_ms=round(elapsed, 2)
            )
        
        # Perform async search
        results = await retriever.search_semantic_async(
            query=request.query,
            top_k=request.top_k,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        
        # Cache results
        cache.set(
            request.query, results, request.domain,
            request.source_lang, request.target_lang, request.top_k
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        # Log query
        get_query_logger().log_query(
            query=request.query,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            result_count=len(results),
            elapsed_time=elapsed / 1000,
            cache_hit=False
        )
        
        return SearchResponse(
            query=request.query,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            total_results=len(results),
            results=results,
            cached=False,
            elapsed_ms=round(elapsed, 2)
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(request: HybridSearchRequest):
    """
    Hybrid search combining semantic and wording similarity.
    
    Combines both meaning and wording matches for best results.
    Uses weighted combination of scores.
    """
    start_time = time.time()
    
    # Validate inputs
    is_valid, error = validate_query(request.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        retriever = await get_retriever()
        
        # Use pipeline for full retrieval
        results = retrieve_fuzzy_matches(
            retriever=retriever,
            query=request.query,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            top_k=request.top_k,
            enable_hybrid=True,
            enable_diversity=request.enable_diversity
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            domain=request.domain,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            total_results=len(results),
            results=results,
            cached=False,
            elapsed_ms=round(elapsed, 2)
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_get(
    query: str = Query(..., min_length=3, description="Source sentence to search"),
    domain: Optional[str] = Query(None, description="Domain filter"),
    source_lang: str = Query("en", description="Source language"),
    target_lang: str = Query("ar", description="Target language"),
    top_k: int = Query(7, ge=1, le=100, description="Number of results")
):
    """
    GET endpoint for semantic search (convenience endpoint).
    
    Same as POST /search/semantic but accessible via GET.
    """
    request = SemanticSearchRequest(
        query=query,
        domain=domain,
        source_lang=source_lang,
        target_lang=target_lang,
        top_k=top_k
    )
    return await search_semantic(request)


# =====================================================
# PHASE 1 INTEGRATION
# =====================================================

@router.post("/integrate", response_model=Phase2Output)
async def integrate_phase1(request: Phase1IntegrationRequest):
    """
    Integrate with Phase 1 (Glossary System).
    
    Takes glossary matches from Phase 1 and enriches with semantic matches.
    Returns combined output ready for Phase 3 (prompt construction).
    """
    start_time = time.time()
    
    # Validate
    is_valid, error = validate_query(request.source_sentence)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        retriever = await get_retriever()
        
        # Build Phase 1 output format
        phase1_output = {
            'source_sentence': request.source_sentence,
            'glossary_matches': request.glossary_matches,
            'domain': request.domain,
            'source_lang': request.source_lang,
            'target_lang': request.target_lang
        }
        
        # Execute Phase 2 integration
        result = integrate_with_phase1(phase1_output, retriever)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Phase 2 integration completed in {elapsed:.2f}ms")
        
        return Phase2Output(**result)
        
    except Exception as e:
        logger.error(f"Phase 1 integration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")


@router.post("/integrate/format")
async def integrate_and_format(request: Phase1IntegrationRequest):
    """
    Integrate with Phase 1 and return formatted prompt text.
    
    Returns the integration result formatted for LLM prompt inclusion.
    """
    try:
        retriever = await get_retriever()
        
        phase1_output = {
            'source_sentence': request.source_sentence,
            'glossary_matches': request.glossary_matches,
            'domain': request.domain,
            'source_lang': request.source_lang,
            'target_lang': request.target_lang
        }
        
        result = integrate_with_phase1(phase1_output, retriever)
        formatted = format_for_prompt(result)
        
        return {
            "formatted_prompt": formatted,
            "glossary_count": result['glossary_count'],
            "fuzzy_count": result['fuzzy_count']
        }
        
    except Exception as e:
        logger.error(f"Format integration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")


# =====================================================
# HEALTH & STATUS ENDPOINTS
# =====================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all system components."""
    try:
        # Check Qdrant
        client = get_qdrant_client()
        qdrant_healthy = verify_qdrant_connection(client)
        qdrant_info = get_collection_info(client) if qdrant_healthy else None
        
        # Check Redis
        cache = get_cache()
        redis_stats = cache.get_stats()
        redis_healthy = redis_stats.get("status") == "connected"
        
        # Check Model
        from app.services.embedding_service import verify_model
        try:
            model_info = verify_model()
            model_healthy = model_info.get("status") == "ok"
        except Exception as e:
            model_info = {"status": "error", "error": str(e)}
            model_healthy = False
        
        overall_status = "healthy" if (qdrant_healthy and model_healthy) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            qdrant={
                "connected": qdrant_healthy,
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
                "collection": qdrant_info
            },
            redis={
                "connected": redis_healthy,
                **redis_stats
            },
            model=model_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            qdrant={"connected": False, "error": str(e)},
            redis={"connected": False},
            model={"status": "unknown"}
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        client = get_qdrant_client()
        collection_info = get_collection_info(client)
        
        cache = get_cache()
        cache_stats = cache.get_stats()
        
        monitor = get_monitor()
        perf_stats = monitor.get_stats()
        
        return StatsResponse(
            collection_info=CollectionInfo(**collection_info) if collection_info else None,
            cache_stats=cache_stats,
            performance_stats=perf_stats
        )
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/pairs/domain", response_model=DomainPairsResponse)
async def get_pairs_by_domain(domain: Optional[str] = Query(None, description="Domain to filter by (leave empty for all domains)")):
    """Get all translation pairs grouped by domain."""
    try:
        retriever = await get_retriever()
        pairs = retriever.get_all_pairs_by_domain(domain)
        
        return DomainPairsResponse(
            domain=domain,
            total_pairs=len(pairs),
            pairs=pairs
        )
        
    except Exception as e:
        logger.error(f"Failed to get pairs by domain: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pairs: {str(e)}")


# =====================================================
# CACHE MANAGEMENT
# =====================================================

@router.delete("/cache")
async def clear_cache():
    """Clear all cached search results."""
    try:
        cache = get_cache()
        success = cache.clear_all()
        
        return {
            "status": "success" if success else "failed",
            "message": "Cache cleared" if success else "Failed to clear cache"
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    return cache.get_stats()


# =====================================================
# QUERY LOGGING
# =====================================================

@router.get("/queries/recent")
async def get_recent_queries(count: int = Query(100, ge=1, le=1000)):
    """Get recent queries for analysis."""
    query_logger = get_query_logger()
    return {
        "queries": query_logger.get_recent(count),
        "stats": query_logger.get_stats()
    }
