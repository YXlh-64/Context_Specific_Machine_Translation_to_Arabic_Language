"""
Caching Module
Redis-based caching for retrieval results
"""

import logging
import hashlib
import json
from typing import Optional, List, Dict, Any
import redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class ResultCache:
    """Redis-based cache for retrieval results"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        ttl_seconds: int = None
    ):
        """
        Initialize Redis cache connection
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl_seconds: Cache TTL in seconds
        """
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db or settings.REDIS_DB
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        
        self._client = None
        self._connected = False
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self._client = redis.Redis(
                **settings.redis_connection_params,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Redis cache connected: {self.host}:{self.port}")
        except redis.ConnectionError as e:
            self._connected = False
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
        except Exception as e:
            self._connected = False
            logger.warning(f"Redis initialization failed: {e}. Caching disabled.")
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        if not self._connected:
            return False
        try:
            self._client.ping()
            return True
        except:
            self._connected = False
            return False
    
    def _generate_key(
        self,
        query: str,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        top_k: int = None
    ) -> str:
        """Generate cache key from search parameters"""
        key_parts = [
            "rag",
            query,
            domain or "all",
            source_lang or "any",
            target_lang or "any",
            str(top_k or settings.DEFAULT_TOP_K)
        ]
        
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"rag:search:{key_hash}"
    
    def get(
        self,
        query: str,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        top_k: int = None
    ) -> Optional[List[Dict]]:
        """
        Get cached results for a query
        
        Args:
            query: Search query
            domain: Domain filter
            source_lang: Source language
            target_lang: Target language
            top_k: Number of results
            
        Returns:
            Cached results or None if not found
        """
        if not self.is_connected:
            return None
        
        try:
            key = self._generate_key(query, domain, source_lang, target_lang, top_k)
            cached = self._client.get(key)
            
            if cached:
                logger.debug(f"Cache hit for: {query[:30]}...")
                return json.loads(cached)
            
            logger.debug(f"Cache miss for: {query[:30]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(
        self,
        query: str,
        results: List[Dict],
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        top_k: int = None,
        ttl: int = None
    ) -> bool:
        """
        Cache results for a query
        
        Args:
            query: Search query
            results: Results to cache
            domain: Domain filter
            source_lang: Source language
            target_lang: Target language
            top_k: Number of results
            ttl: Cache TTL (overrides default)
            
        Returns:
            True if cached successfully
        """
        if not self.is_connected:
            return False
        
        try:
            key = self._generate_key(query, domain, source_lang, target_lang, top_k)
            value = json.dumps(results)
            ttl = ttl or self.ttl
            
            self._client.setex(key, ttl, value)
            logger.debug(f"Cached results for: {query[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(
        self,
        query: str,
        domain: str = None,
        source_lang: str = None,
        target_lang: str = None,
        top_k: int = None
    ) -> bool:
        """Delete cached results for a query"""
        if not self.is_connected:
            return False
        
        try:
            key = self._generate_key(query, domain, source_lang, target_lang, top_k)
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all RAG cache entries"""
        if not self.is_connected:
            return False
        
        try:
            pattern = "rag:search:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.is_connected:
            return {"status": "disconnected"}
        
        try:
            pattern = "rag:search:*"
            keys = self._client.keys(pattern)
            
            return {
                "status": "connected",
                "host": self.host,
                "port": self.port,
                "db": self.db,
                "cache_entries": len(keys),
                "ttl_seconds": self.ttl
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def retrieve_fuzzy_matches_cached(
    cache: ResultCache,
    retriever,  # SemanticRetriever
    query: str,
    domain: str = None,
    source_lang: str = "en",
    target_lang: str = "ar",
    top_k: int = None
) -> List[Dict]:
    """
    Retrieve with caching
    
    Checks cache first, then falls back to actual retrieval.
    
    Args:
        cache: ResultCache instance
        retriever: SemanticRetriever instance
        query: Search query
        domain: Domain filter
        source_lang: Source language
        target_lang: Target language
        top_k: Number of results
        
    Returns:
        Retrieval results (from cache or fresh)
    """
    from app.services.pipeline import retrieve_fuzzy_matches
    
    if top_k is None:
        top_k = settings.DEFAULT_TOP_K
    
    # Try cache first
    cached_results = cache.get(query, domain, source_lang, target_lang, top_k)
    
    if cached_results is not None:
        logger.info(f"Returning cached results for: {query[:30]}...")
        return cached_results
    
    # Cache miss - perform actual retrieval
    results = retrieve_fuzzy_matches(
        retriever=retriever,
        query=query,
        domain=domain,
        source_lang=source_lang,
        target_lang=target_lang,
        top_k=top_k
    )
    
    # Cache the results
    cache.set(query, results, domain, source_lang, target_lang, top_k)
    
    return results


# Global cache instance
_cache_instance: Optional[ResultCache] = None


def get_cache() -> ResultCache:
    """Get or create global cache instance"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = ResultCache()
    
    return _cache_instance


if __name__ == "__main__":
    # Test caching
    logging.basicConfig(level=logging.INFO)
    
    cache = ResultCache()
    print(f"Cache stats: {cache.get_stats()}")
    
    # Test set/get
    test_query = "Test query for caching"
    test_results = [{"id": 1, "score": 0.95, "source": "test"}]
    
    cache.set(test_query, test_results)
    retrieved = cache.get(test_query)
    
    print(f"Set/Get test: {retrieved == test_results}")
