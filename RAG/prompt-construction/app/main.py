"""
Phase 3 - Prompt Construction Service
Main FastAPI Application
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Phase 3 - Prompt Construction Service Starting")
    logger.info("=" * 60)
    logger.info(f"Service running on port: {settings.PORT}")
    logger.info(f"Default prompt format: {settings.DEFAULT_PROMPT_FORMAT}")
    logger.info(f"Default domain: {settings.DEFAULT_DOMAIN}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("Phase 3 - Prompt Construction Service Shutting Down")


# Create FastAPI application
app = FastAPI(
    title="Phase 3 - Prompt Construction Service",
    description="""
## Phase 3 - Prompt Construction (Port 8003)

Constructs optimized prompts for LLM-based translation.

**NOTE**: This service ONLY constructs prompts. The actual LLM translation
is handled by another team/service.

### Features
- Prompt Construction from glossary and RAG results
- Multiple Formats: XML, JSON, Markdown, Plain text
- Domain Support: Medical, legal, education, technology, economic, general
- Token Management for model limits
- System message generation for LLM configuration

### Endpoints (prefix: /api/v1)

**Prompt Construction:**
- `POST /api/v1/prompt/construct` - Construct a translation prompt
- `POST /api/v1/prompt/preview` - Preview prompt in all formats

**Info:**
- `GET /api/v1/health` - Health check
- `GET /api/v1/info` - Service information
- `GET /api/v1/domains` - List available domains
- `GET /api/v1/formats` - List available prompt formats

**Root:**
- `GET /` - Service status
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router with prefix
app.include_router(router, prefix="/api/v1", tags=["Prompt Construction"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Phase 3 - Prompt Construction",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "info": "/info"
    }


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )
