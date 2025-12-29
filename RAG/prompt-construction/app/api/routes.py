"""
API routes for Phase 3 Prompt Construction
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    PromptConstructionRequest,
    PromptConstructionResponse,
    PromptFormat,
    DomainType,
    GlossaryTerm,
    FuzzyMatch
)
from app.services.optimized_prompt_service import get_prompt_constructor
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Prompt Construction Endpoints
# ============================================================================

@router.post("/prompt/construct", response_model=PromptConstructionResponse)
async def construct_prompt(request: PromptConstructionRequest):
    """
    Construct a translation prompt from glossary matches and fuzzy matches.
    
    This endpoint takes pre-fetched glossary and RAG results and constructs
    an optimized prompt for LLM translation.
    
    The constructed prompt can be passed to your team's LLM service.
    """
    try:
        logger.info(f"Constructing prompt for: '{request.source_sentence[:50]}...'")
        
        constructor = get_prompt_constructor()
        response = await constructor.construct_async(request)
        
        logger.info(f"Prompt constructed: {response.token_count} tokens")
        return response
        
    except Exception as e:
        logger.error(f"Prompt construction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompt/preview")
async def preview_prompts(request: PromptConstructionRequest):
    """
    Preview prompt in all available formats.
    
    Returns the same prompt formatted as XML, JSON, Markdown, and Plain text
    for comparison and debugging.
    """
    try:
        constructor = get_prompt_constructor()
        
        # Create versions in all formats
        previews = {}
        
        for fmt in PromptFormat:
            modified_request = PromptConstructionRequest(
                source_sentence=request.source_sentence,
                glossary_matches=request.glossary_matches,
                fuzzy_matches=request.fuzzy_matches,
                domain=request.domain,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                prompt_format=fmt,
                include_system_message=request.include_system_message,
                style=request.style,
                formality=request.formality,
                custom_instructions=request.custom_instructions
            )
            response = constructor.construct(modified_request, use_cache=False)
            previews[fmt.value] = {
                "prompt": response.prompt,
                "system_message": response.system_message,
                "token_count": response.token_count
            }
        
        return {
            "source_sentence": request.source_sentence,
            "domain": request.domain,
            "previews": previews
        }
        
    except Exception as e:
        logger.error(f"Preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "prompt-construction",
        "phase": 3
    }


@router.get("/info")
async def service_info():
    """Get service information and configuration."""
    return {
        "service": "Phase 3 - Prompt Construction",
        "version": "1.0.0",
        "description": "Constructs optimized prompts from glossary and RAG results for LLM translation",
        "endpoints": {
            "prompt_construction": "/prompt/construct",
            "prompt_preview": "/prompt/preview"
        },
        "available_formats": [f.value for f in PromptFormat],
        "available_domains": [d.value for d in DomainType],
        "settings": {
            "max_glossary_terms": settings.MAX_GLOSSARY_TERMS,
            "max_fuzzy_matches": settings.MAX_FUZZY_MATCHES,
            "default_prompt_format": settings.DEFAULT_PROMPT_FORMAT
        }
    }


@router.get("/domains")
async def list_domains():
    """List available domains for translation."""
    return {
        "domains": [
            {"value": d.value, "name": d.name}
            for d in DomainType
        ]
    }


@router.get("/formats")
async def list_formats():
    """List available prompt formats."""
    return {
        "formats": [
            {"value": f.value, "name": f.name}
            for f in PromptFormat
        ]
    }
