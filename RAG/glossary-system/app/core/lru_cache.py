"""
LRU Cache Module for Glossary Terms

Implements a multi-layer caching strategy:
1. L1 Cache: In-memory LRU (fastest, limited size)
2. L2 Cache: Redis (distributed, larger capacity)

Features:
- Thread-safe operations
- Automatic cache invalidation
- TTL-based expiration
- Cache statistics and monitoring
- Write-through caching strategy

Performance Impact:
- L1 hit: ~0.01ms (10,000x faster than DB)
- L2 hit: ~1ms (100x faster than DB)
- DB query: ~10-50ms
"""

import threading
import time
import logging
import hashlib
import json
from collections import OrderedDict
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_requests: int = 0
    
    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate."""
        total = self.l1_hits + self.l1_misses
        return (self.l1_hits / total * 100) if total > 0 else 0.0
    
    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 cache hit rate."""
        total = self.l2_hits + self.l2_misses
        return (self.l2_hits / total * 100) if total > 0 else 0.0
    
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        hits = self.l1_hits + self.l2_hits
        return (hits / self.total_requests * 100)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: int
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def touch(self):
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class LRUCache:
    """
    Thread-safe Least Recently Used (LRU) cache.
    
    Uses OrderedDict for O(1) operations:
    - get: O(1)
    - put: O(1)
    - delete: O(1)
    
    Memory bounded with automatic eviction of least recently used items.
    """
    
    def __init__(
        self, 
        max_size: int = 10000, 
        default_ttl: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            cleanup_interval: Seconds between cleanup runs
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._stats = CacheStats()
        self._last_cleanup = time.time()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _maybe_cleanup(self):
        """Run cleanup if interval has passed."""
        if (time.time() - self._last_cleanup) > self._cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = time.time()
    
    def _cleanup_expired(self):
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1
            
            if expired_keys:
                logger.debug(f"LRU cleanup: removed {len(expired_keys)} expired entries")
    
    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            self._stats.total_requests += 1
            self._maybe_cleanup()
            
            if key not in self._cache:
                self._stats.l1_misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._stats.l1_misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.l1_hits += 1
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        with self._lock:
            # Update existing entry
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl or self._default_ttl
                )
                return
            
            # Evict if necessary
            self._evict_if_needed()
            
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl or self._default_ttl
            )
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.invalidations += 1
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Key pattern to match (simple contains check)
            
        Returns:
            Number of invalidated entries
        """
        with self._lock:
            matching_keys = [
                key for key in self._cache.keys()
                if pattern in key
            ]
            for key in matching_keys:
                del self._cache[key]
                self._stats.invalidations += 1
            return len(matching_keys)
    
    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.invalidations += count
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "l1_hits": self._stats.l1_hits,
                "l1_misses": self._stats.l1_misses,
                "l1_hit_rate": round(self._stats.l1_hit_rate, 2),
                "evictions": self._stats.evictions,
                "invalidations": self._stats.invalidations,
                "total_requests": self._stats.total_requests
            }


class GlossaryTermCache:
    """
    Specialized cache for glossary term lookups.
    
    Implements multi-layer caching:
    - L1: In-memory LRU cache (fast, limited)
    - L2: Redis cache (distributed, larger)
    
    Cache key structure: glossary:{domain}:{src_lang}:{tgt_lang}:{term_hash}
    """
    
    def __init__(
        self,
        l1_max_size: int = 50000,
        l1_ttl: int = 1800,  # 30 minutes for L1
        l2_ttl: int = 7200   # 2 hours for L2
    ):
        """
        Initialize glossary term cache.
        
        Args:
            l1_max_size: Maximum L1 cache entries
            l1_ttl: L1 cache TTL in seconds
            l2_ttl: L2 (Redis) cache TTL in seconds
        """
        self._l1 = LRUCache(max_size=l1_max_size, default_ttl=l1_ttl)
        self._l2_ttl = l2_ttl
        self._redis_client = None
        self._stats = CacheStats()
    
    def set_redis_client(self, client):
        """Set Redis client for L2 cache."""
        self._redis_client = client
    
    def _make_key(
        self, 
        domain: str, 
        src_lang: str, 
        tgt_lang: str, 
        terms: List[str]
    ) -> str:
        """Generate cache key for glossary lookup."""
        terms_hash = hashlib.md5(
            ":".join(sorted(terms)).encode()
        ).hexdigest()[:16]
        return f"glossary:{domain}:{src_lang}:{tgt_lang}:{terms_hash}"
    
    def get(
        self, 
        domain: str, 
        src_lang: str, 
        tgt_lang: str, 
        terms: List[str]
    ) -> Optional[List[Dict]]:
        """
        Get glossary matches from cache.
        
        Checks L1 first, then L2.
        
        Args:
            domain: Domain filter
            src_lang: Source language
            tgt_lang: Target language
            terms: List of terms to look up
            
        Returns:
            Cached matches or None if not found
        """
        key = self._make_key(domain, src_lang, tgt_lang, terms)
        
        # Try L1 first
        result = self._l1.get(key)
        if result is not None:
            logger.debug(f"L1 cache hit for key: {key[:50]}...")
            return result
        
        # Try L2 (Redis)
        if self._redis_client:
            try:
                cached = self._redis_client.get(key)
                if cached:
                    result = json.loads(cached)
                    # Promote to L1
                    self._l1.put(key, result)
                    self._stats.l2_hits += 1
                    logger.debug(f"L2 cache hit for key: {key[:50]}...")
                    return result
                self._stats.l2_misses += 1
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        return None
    
    def put(
        self, 
        domain: str, 
        src_lang: str, 
        tgt_lang: str, 
        terms: List[str],
        matches: List[Dict],
        ttl: Optional[int] = None
    ) -> None:
        """
        Store glossary matches in cache.
        
        Writes to both L1 and L2 (write-through).
        
        Args:
            domain: Domain filter
            src_lang: Source language
            tgt_lang: Target language
            terms: List of terms
            matches: Glossary matches to cache
            ttl: Optional TTL override
        """
        key = self._make_key(domain, src_lang, tgt_lang, terms)
        
        # Write to L1
        self._l1.put(key, matches, ttl)
        
        # Write to L2 (Redis)
        if self._redis_client:
            try:
                self._redis_client.setex(
                    key,
                    ttl or self._l2_ttl,
                    json.dumps(matches)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
    
    def invalidate_domain(self, domain: str) -> int:
        """
        Invalidate all cached entries for a domain.
        
        Use when domain glossary is updated.
        
        Args:
            domain: Domain to invalidate
            
        Returns:
            Number of invalidated entries
        """
        count = self._l1.invalidate_pattern(f"glossary:{domain}")
        
        if self._redis_client:
            try:
                pattern = f"glossary:{domain}:*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                    count += len(keys)
            except Exception as e:
                logger.warning(f"Redis invalidation failed: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        l1_stats = self._l1.get_stats()
        return {
            "l1": l1_stats,
            "l2_enabled": self._redis_client is not None,
            "l2_hits": self._stats.l2_hits,
            "l2_misses": self._stats.l2_misses
        }


# Global cache instance
_glossary_cache: Optional[GlossaryTermCache] = None
_cache_lock = threading.Lock()


def get_glossary_cache() -> GlossaryTermCache:
    """Get or create the global glossary cache."""
    global _glossary_cache
    
    if _glossary_cache is None:
        with _cache_lock:
            if _glossary_cache is None:
                _glossary_cache = GlossaryTermCache(
                    l1_max_size=50000,
                    l1_ttl=1800,
                    l2_ttl=7200
                )
    
    return _glossary_cache


def memoize(ttl: int = 300, max_size: int = 1000):
    """
    Decorator for memoizing function results.
    
    Args:
        ttl: Cache TTL in seconds
        max_size: Maximum cache entries
        
    Example:
        @memoize(ttl=600)
        def expensive_computation(arg1, arg2):
            ...
    """
    cache = LRUCache(max_size=max_size, default_ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        
        return wrapper
    
    return decorator
