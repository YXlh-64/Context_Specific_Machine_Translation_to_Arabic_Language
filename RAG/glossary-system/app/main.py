import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings

# =====================================================
# LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Add file handler for production
        # logging.FileHandler('logs/app.log')
    ]
)

logger = logging.getLogger(__name__)


# =====================================================
# LIFESPAN MANAGEMENT
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.PROJECT_NAME}")
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Allowed domains: {settings.ALLOWED_DOMAINS}")
    logger.info(f"Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    
    # Initialize connection pool
    try:
        from app.core.connection_pool import get_connection_pool
        pool = get_connection_pool()
        logger.info(f"Connection pool initialized: {pool.get_stats()}")
    except Exception as e:
        logger.warning(f"Connection pool initialization warning: {e}")
    
    # Initialize cache
    try:
        from app.core.lru_cache import get_glossary_cache
        cache = get_glossary_cache()
        logger.info("LRU cache initialized")
    except Exception as e:
        logger.warning(f"LRU cache initialization warning: {e}")
    
    yield
    
    # Shutdown - cleanup resources
    logger.info("Shutting down application")
    
    try:
        from app.core.connection_pool import close_connection_pool
        close_connection_pool()
        logger.info("Connection pool closed")
    except Exception as e:
        logger.warning(f"Connection pool cleanup warning: {e}")


# =====================================================
# APPLICATION FACTORY
# =====================================================

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    application = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
## Phase 1 - Glossary Lookup API (Port 8001)

A production-ready API for glossary-based translation assistance.

### Features
- **Sentence Translation**: Process individual sentences for glossary term matches
- **PDF Processing**: Upload and process PDF documents for batch glossary lookups
- **Session Management**: Efficient caching with Redis for repeated lookups
- **Multi-domain Support**: Health, agriculture, history, finance, legal, technology

### Endpoints (prefix: /api/v1)

**Sentence Translation:**
- `POST /api/v1/translate/sentence` - Translate a single sentence using glossary lookup

**PDF Processing:**
- `POST /api/v1/translate/pdf` - Upload PDF and create session
- `GET /api/v1/session/{session_id}` - Get session status
- `GET /api/v1/session/{session_id}/extract` - Extract text from PDF pages
- `GET /api/v1/session/{session_id}/glossary-terms` - Get all glossary terms for session
- `POST /api/v1/session/{session_id}/process/sentence` - Process single sentence in session
- `POST /api/v1/session/{session_id}/process/batch` - Process multiple sentences
- `DELETE /api/v1/session/{session_id}` - Delete session and cleanup

**Glossary:**
- `GET /api/v1/glossary/terms/{domain}` - Get all glossary terms by domain

**Session Management:**
- `GET /api/v1/session/{session_id}/stats` - Get session statistics
- `GET /api/v1/sessions` - List all active sessions
- `DELETE /api/v1/sessions` - Delete all sessions

**Health:**
- `GET /api/v1/health/services` - Check all service health
- `GET /health` - Basic health check
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    @application.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        """Log all incoming requests with timing."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request (skip health checks to reduce noise)
        if not request.url.path.startswith("/health"):
            logger.info(
                f"{request.method} {request.url.path} "
                f"- Status: {response.status_code} "
                f"- Duration: {duration_ms:.2f}ms"
            )
        
        # Add timing header
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        
        return response
    
    # Global exception handler
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred",
                "type": "internal_error"
            }
        )
    
    # Include routers
    application.include_router(router, prefix="/api/v1", tags=["Translation"])
    
    return application


# Create application instance
app = create_application()


# =====================================================
# ROOT ENDPOINTS
# =====================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api_prefix": "/api/v1"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}