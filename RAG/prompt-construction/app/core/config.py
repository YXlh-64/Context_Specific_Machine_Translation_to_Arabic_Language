"""
Configuration for Phase 3: Prompt Construction
"""

from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    APP_NAME: str = "Prompt Construction Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    LOG_LEVEL: str = "INFO"
    
    # Prompt Configuration
    MAX_GLOSSARY_TERMS: int = 5
    MAX_FUZZY_MATCHES: int = 7
    MAX_PROMPT_TOKENS: int = 4096
    DEFAULT_DOMAIN: str = "general"
    DEFAULT_PROMPT_FORMAT: str = "xml"
    
    # Language settings
    DEFAULT_SOURCE_LANG: str = "en"
    DEFAULT_TARGET_LANG: str = "ar"
    
    # Domain-specific tone settings
    DOMAIN_TONES: dict = {
        "medical": "Formal, precise, clinical terminology",
        "health": "Formal, precise, clinical terminology",
        "legal": "Formal, exact, standardized legal terms",
        "education": "Clear, accessible, pedagogical language",
        "technology": "Technical, consistent, standard tech terminology",
        "economic": "Professional, precise, financial terminology",
        "general": "Clear, natural, contextually appropriate"
    }
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()


# Language name mappings
LANGUAGE_NAMES = {
    "en": "English",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese"
}


def get_language_name(code: str) -> str:
    """Get full language name from code"""
    return LANGUAGE_NAMES.get(code, code.capitalize())
