import os
from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    # ===========================================
    # Project Settings
    # ===========================================
    PROJECT_NAME: str = "Glossary Lookup System"
    
    # ===========================================
    # Database Configuration
    # ===========================================
    DATABASE_URL: str = "file:data/glossary.db"  # FTS5 enabled database
    DATABASE_URL_READONLY: str = "file:data/glossary.db?mode=ro"  # Read-only mode for production
    
    # ===========================================
    # Redis Configuration
    # ===========================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL_SECONDS: int = 7200  # 2 hours

    # ===========================================
    # File Storage
    # ===========================================
    UPLOAD_DIR: str = "uploads"
    MAX_PDF_SIZE_MB: int = 50
    
    # ===========================================
    # Validation Constraints
    # ===========================================
    ALLOWED_DOMAINS: set = {'health', 'agriculture', 'history', 'finance', 'media', 'technology'}
    ALLOWED_LANGS: set = {'en', 'ar', 'fr'}
    MAX_TEXT_LENGTH: int = 500  # Words

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


settings = Settings()

# Ensure dirs exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
