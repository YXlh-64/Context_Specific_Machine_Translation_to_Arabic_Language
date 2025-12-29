"""
Configuration settings for the RAG System (Phase 2)
Semantic Translation Memory System
"""

import os
from pathlib import Path
from typing import Set
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "translation_memory")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # For cloud deployment
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")  # For cloud deployment
    
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "1"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "sentence-transformers/LaBSE")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "7"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # HNSW Index Configuration
    HNSW_M: int = 16  # Number of edges per node
    HNSW_EF_CONSTRUCT: int = 100  # Construction time accuracy
    
    # Data Paths
    TRANSLATION_DATA_DIR: Path = Path(os.getenv(
        "TRANSLATION_DATA_DIR", 
        str(BASE_DIR.parent.parent / "AYA'sDATA")
    ))
    
    # Allowed Domains
    ALLOWED_DOMAINS: Set[str] = set(
        os.getenv("ALLOWED_DOMAINS", "health,agriculture,history,finance,legal,technology").split(",")
    )
    
    # Allowed Languages
    ALLOWED_LANGS: Set[str] = set(
        os.getenv("ALLOWED_LANGS", "en,ar,fr").split(",")
    )
    
    # Server Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8002"))
    
    # Retrieval Weights
    SEMANTIC_WEIGHT: float = 0.6
    WORDING_WEIGHT: float = 0.4
    
    # Diversity Configuration (MMR)
    DIVERSITY_LAMBDA: float = 0.7  # Balance between relevance and diversity
    
    # Domain Boosting
    DOMAIN_BOOST_FACTOR: float = 1.2  # Boost for same-domain matches
    
    # Complexity Filtering
    MIN_SENTENCE_LENGTH: int = 3  # Minimum words in sentence
    MAX_LENGTH_RATIO: float = 2.0  # Max ratio between query and result length
    
    @property
    def qdrant_connection_params(self) -> dict:
        """Get Qdrant connection parameters"""
        if self.QDRANT_URL:
            return {
                "url": self.QDRANT_URL,
                "api_key": self.QDRANT_API_KEY if self.QDRANT_API_KEY else None
            }
        return {
            "host": self.QDRANT_HOST,
            "port": self.QDRANT_PORT
        }
    
    @property
    def redis_connection_params(self) -> dict:
        """Get Redis connection parameters"""
        params = {
            "host": self.REDIS_HOST,
            "port": self.REDIS_PORT,
            "db": self.REDIS_DB,
            "decode_responses": True
        }
        if self.REDIS_PASSWORD:
            params["password"] = self.REDIS_PASSWORD
        return params


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
