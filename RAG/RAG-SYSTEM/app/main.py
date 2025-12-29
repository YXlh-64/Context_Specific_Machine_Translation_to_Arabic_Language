"""
Phase 2: Semantic Translation Memory System
FastAPI Application Entry Point
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Starting RAG System (Phase 2)")
    logger.info("=" * 50)
    logger.info(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"Collection: {settings.QDRANT_COLLECTION}")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info(f"Allowed domains: {settings.ALLOWED_DOMAINS}")
    logger.info(f"Allowed languages: {settings.ALLOWED_LANGS}")
    
    # Pre-load model (optional, for faster first request)
    try:
        from app.services.embedding_service import get_model
        logger.info("Pre-loading LaBSE model...")
        model = get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model pre-loading failed: {e}")
    
    # Initialize Qdrant connection
    try:
        from app.services.setup_qdrant import get_qdrant_client, verify_qdrant_connection
        client = get_qdrant_client()
        if verify_qdrant_connection(client):
            logger.info("Qdrant connection verified")
        else:
            logger.warning("Qdrant connection could not be verified")
    except Exception as e:
        logger.warning(f"Qdrant initialization warning: {e}")
    
    # Initialize Redis cache
    try:
        from app.services.caching import get_cache
        cache = get_cache()
        if cache.is_connected:
            logger.info("Redis cache connected")
        else:
            logger.warning("Redis cache not connected - caching disabled")
    except Exception as e:
        logger.warning(f"Redis initialization warning: {e}")
    
    logger.info("RAG System startup complete")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System")


# Create FastAPI application
app = FastAPI(
    title="RAG System - Semantic Translation Memory",
    description="""
## Phase 2 - RAG System (Port 8002)

Semantic Translation Memory using LaBSE embeddings and Qdrant vector database.

### Features
- LaBSE multilingual embeddings for cross-lingual search
- Qdrant vector database for fast similarity search
- Multi-vector search (semantic, wording, hybrid)
- Diversity re-ranking (MMR)
- Redis caching for performance

### Endpoints (prefix: /api/v1)

**Search:**
- `POST /api/v1/search/semantic` - Semantic search for similar translations
- `POST /api/v1/search/hybrid` - Hybrid search combining semantic and keyword
- `GET /api/v1/search` - Simple GET search endpoint

**Integration:**
- `POST /api/v1/integrate` - Integration endpoint for Phase 1 data
- `POST /api/v1/integrate/format` - Format integration results

**Health & Stats:**
- `GET /api/v1/health` - Service health with Qdrant/Redis status
- `GET /api/v1/stats` - Collection statistics
- `GET /api/v1/pairs/domain` - All translation pairs by domain
- `GET /health` - Quick health check

**Cache:**
- `DELETE /api/v1/cache` - Clear cache
- `GET /api/v1/cache/stats` - Cache statistics

**Monitoring:**
- `GET /api/v1/queries/recent` - Recent query log
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["RAG API"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "RAG System - Semantic Translation Memory",
        "version": "1.0.0",
        "phase": 2,
        "description": "Semantic search for translation memory",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "search": "/api/v1/search",
            "search_semantic": "/api/v1/search/semantic",
            "search_hybrid": "/api/v1/search/hybrid",
            "integrate": "/api/v1/integrate",
            "health": "/api/v1/health",
            "stats": "/api/v1/stats"
        }
    }


# Health check at root level
@app.get("/health")
async def root_health():
    """Quick health check"""
    return {"status": "ok", "service": "rag-system"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
