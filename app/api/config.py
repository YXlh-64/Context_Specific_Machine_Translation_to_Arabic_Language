import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    # Default port changed to 5002 to avoid common macOS service conflicts
    PORT = int(os.getenv('FLASK_PORT', 5002))
    # Allow local dev servers (Vite on 8080 and other ports)
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:8080,http://localhost:5173').split(',')

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}