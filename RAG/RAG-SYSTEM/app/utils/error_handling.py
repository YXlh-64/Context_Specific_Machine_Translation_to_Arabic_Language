"""
Error Handling Module
Robust error handling with retries and connection management
"""

import logging
import time
from typing import Callable, Optional, Any
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


# Custom Exceptions
class RAGSystemError(Exception):
    """Base exception for RAG system errors"""
    pass


class QdrantConnectionError(RAGSystemError):
    """Qdrant connection error"""
    pass


class EmbeddingError(RAGSystemError):
    """Embedding generation error"""
    pass


class RetrievalError(RAGSystemError):
    """Retrieval operation error"""
    pass


class CacheError(RAGSystemError):
    """Cache operation error"""
    pass


# Retry Decorators
def retry_on_connection_error(max_attempts: int = 3):
    """Decorator to retry on connection errors"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, UnexpectedResponse)),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def search_with_retry(client: QdrantClient, *args, **kwargs):
    """Execute Qdrant search with automatic retry on transient failures"""
    return client.search(*args, **kwargs)


def safe_operation(func: Callable):
    """
    Decorator for safe operation execution with error handling
    
    Catches exceptions and returns None/empty result instead of propagating
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RAGSystemError as e:
            logger.error(f"RAG system error in {func.__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return None
    
    return wrapper


def safe_async_operation(func: Callable):
    """Async version of safe_operation decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RAGSystemError as e:
            logger.error(f"RAG system error in {func.__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return None
    
    return wrapper


class QdrantConnectionManager:
    """
    Manage Qdrant connection with health checks and auto-reconnection
    """
    
    def __init__(self, max_retries: int = 3, health_check_interval: int = 60):
        """
        Initialize connection manager
        
        Args:
            max_retries: Maximum connection retry attempts
            health_check_interval: Seconds between health checks
        """
        self._client: Optional[QdrantClient] = None
        self._max_retries = max_retries
        self._health_check_interval = health_check_interval
        self._last_health_check = 0
        self._is_healthy = False
    
    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client, reconnecting if necessary"""
        if self._client is None or not self._check_health():
            self._connect()
        return self._client
    
    def _connect(self):
        """Establish connection to Qdrant"""
        for attempt in range(self._max_retries):
            try:
                self._client = QdrantClient(**settings.qdrant_connection_params)
                # Test connection
                self._client.get_collections()
                self._is_healthy = True
                self._last_health_check = time.time()
                logger.info("Connected to Qdrant successfully")
                return
            except Exception as e:
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise QdrantConnectionError(f"Failed to connect to Qdrant after {self._max_retries} attempts")
    
    def _check_health(self) -> bool:
        """Check if connection is healthy"""
        current_time = time.time()
        
        # Only check periodically
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_healthy
        
        try:
            self._client.get_collections()
            self._is_healthy = True
            self._last_health_check = current_time
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            self._is_healthy = False
            return False
    
    def get_health_status(self) -> dict:
        """Get current health status"""
        return {
            "connected": self._is_healthy,
            "last_check": self._last_health_check,
            "client_initialized": self._client is not None
        }
    
    def close(self):
        """Close the connection"""
        if self._client:
            try:
                self._client.close()
            except:
                pass
            self._client = None
            self._is_healthy = False


def retrieve_fuzzy_matches_safe(
    retriever,  # SemanticRetriever
    query: str,
    domain: str = None,
    source_lang: str = "en",
    target_lang: str = "ar",
    top_k: int = None
) -> list:
    """
    Safe wrapper for fuzzy match retrieval
    
    Returns empty list on error instead of raising exception.
    
    Args:
        retriever: SemanticRetriever instance
        query: Search query
        domain: Domain filter
        source_lang: Source language
        target_lang: Target language
        top_k: Number of results
        
    Returns:
        List of results (empty on error)
    """
    from app.services.pipeline import retrieve_fuzzy_matches
    
    if top_k is None:
        top_k = settings.DEFAULT_TOP_K
    
    try:
        return retrieve_fuzzy_matches(
            retriever=retriever,
            query=query,
            domain=domain,
            source_lang=source_lang,
            target_lang=target_lang,
            top_k=top_k
        )
    except QdrantConnectionError as e:
        logger.error(f"Qdrant connection error: {e}")
        return []
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        return []
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        return []


def validate_query(query: str) -> tuple[bool, str]:
    """
    Validate a query string
    
    Args:
        query: Query to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    if not isinstance(query, str):
        return False, "Query must be a string"
    
    query = query.strip()
    
    if len(query) < 3:
        return False, "Query must be at least 3 characters"
    
    if len(query) > 10000:
        return False, "Query must be less than 10000 characters"
    
    word_count = len(query.split())
    if word_count < 1:
        return False, "Query must contain at least one word"
    
    return True, ""


def validate_domain(domain: str) -> tuple[bool, str]:
    """Validate domain parameter"""
    if domain is None:
        return True, ""  # Domain is optional
    
    if domain not in settings.ALLOWED_DOMAINS:
        return False, f"Invalid domain. Allowed: {list(settings.ALLOWED_DOMAINS)}"
    
    return True, ""


def validate_language(lang: str, param_name: str) -> tuple[bool, str]:
    """Validate language parameter"""
    if lang not in settings.ALLOWED_LANGS:
        return False, f"Invalid {param_name}. Allowed: {list(settings.ALLOWED_LANGS)}"
    
    return True, ""


# Global connection manager instance
_connection_manager: Optional[QdrantConnectionManager] = None


def get_connection_manager() -> QdrantConnectionManager:
    """Get global connection manager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = QdrantConnectionManager()
    return _connection_manager


if __name__ == "__main__":
    # Test error handling
    logging.basicConfig(level=logging.INFO)
    
    # Test validation
    print("Query validation:")
    print(validate_query(""))
    print(validate_query("ok"))
    print(validate_query("This is a valid query"))
    
    print("\nDomain validation:")
    print(validate_domain("health"))
    print(validate_domain("invalid"))
    
    print("\nLanguage validation:")
    print(validate_language("en", "source_lang"))
    print(validate_language("xx", "source_lang"))
