import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import redis
from redis.exceptions import RedisError

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)


class CacheService:
    """
    Production-ready Redis caching service for glossary lookups.
    
    Features:
    - Session-based glossary caching
    - Bulk operations for efficiency
    - N-gram lookup with pipeline
    - Automatic TTL management
    - Graceful error handling
    """
    
    # Key prefixes for organization
    PREFIX_SESSION = "session"
    PREFIX_GLOSSARY = "glossary"
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with retry logic."""
        try:
            self._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False,  # Handle bytes manually for flexibility
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info("Redis connection established")
        except RedisError as e:
            logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
            self._connected = False
    
    @property
    def redis(self) -> Optional[redis.Redis]:
        """Backward compatibility property for redis client."""
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is available."""
        if not self._connected or not self._client:
            return False
        try:
            self._client.ping()
            return True
        except RedisError:
            self._connected = False
            return False
    
    def _ensure_connection(self) -> bool:
        """Ensure Redis connection is available, attempt reconnect if needed."""
        if self.is_connected:
            return True
        
        # Attempt reconnection
        self._connect()
        return self._connected

    # =====================================================
    # SESSION MANAGEMENT
    # =====================================================
    
    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID is not empty or None."""
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty or None")
    
    def _session_key(self, session_id: str) -> str:
        """Generate session metadata key."""
        self._validate_session_id(session_id)
        return f"{self.PREFIX_SESSION}:{session_id}:meta"
    
    def _glossary_key(self, session_id: str, domain: str) -> str:
        """Generate glossary hash key for a session."""
        self._validate_session_id(session_id)
        if not domain or not domain.strip():
            raise ValueError("Domain cannot be empty or None")
        return f"{self.PREFIX_GLOSSARY}:{session_id}:{domain}"
    
    def create_session(self, session_id: str, meta: Dict[str, Any]) -> bool:
        """
        Create a new session with metadata.
        
        Args:
            session_id: Unique session identifier
            meta: Session metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_connection():
            logger.warning("Redis unavailable, session created without cache")
            return False
        
        try:
            key = self._session_key(session_id)
            
            # Add expiration info
            expires_at = datetime.utcnow() + timedelta(seconds=settings.CACHE_TTL_SECONDS)
            meta['cache_expires_at'] = expires_at.isoformat()
            
            # Convert dict values to strings for Redis Hash
            formatted_meta = {k: str(v) if not isinstance(v, str) else v for k, v in meta.items()}
            
            self._client.hset(key, mapping=formatted_meta)
            self._client.expire(key, settings.CACHE_TTL_SECONDS)
            
            logger.debug(f"Session created: {session_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to create session: {str(e)}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metadata dict or None if not found
        """
        if not self._ensure_connection():
            return None
        
        try:
            key = self._session_key(session_id)
            data = self._client.hgetall(key)
            if data:
                # Decode bytes to strings
                return {
                    k.decode('utf-8') if isinstance(k, bytes) else k: 
                    v.decode('utf-8') if isinstance(v, bytes) else v 
                    for k, v in data.items()
                }
            return None
        except RedisError as e:
            logger.error(f"Failed to get session: {str(e)}")
            return None
    
    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get_session for backward compatibility."""
        return self.get_session(session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session metadata (partial update).
        
        Args:
            session_id: Session identifier
            updates: Fields to update
            
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            key = self._session_key(session_id)
            
            # Check if session exists
            if not self._client.exists(key):
                return False
            
            # Convert values to strings
            formatted_updates = {k: str(v) if not isinstance(v, str) else v for k, v in updates.items()}
            
            self._client.hset(key, mapping=formatted_updates)
            return True
        except RedisError as e:
            logger.error(f"Failed to update session: {str(e)}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session and all associated data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            # Find all keys for this session
            pattern = f"*:{session_id}:*"
            keys = list(self._client.scan_iter(match=pattern, count=100))
            
            if keys:
                self._client.delete(*keys)
            
            logger.debug(f"Session deleted: {session_id}")
            return True
        except RedisError as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False

    # =====================================================
    # GLOSSARY CACHING
    # =====================================================
    
    def bulk_cache_glossary(
        self, 
        session_id: str, 
        domain: str, 
        terms: List[Dict[str, Any]]
    ) -> bool:
        """
        Bulk cache glossary terms for a session using Redis Hash.
        
        Uses HSET for efficient bulk insertion.
        Key format: glossary:{session_id}:{domain}
        Field format: {source_term}|{n_size}
        Value format: target_term
        
        Args:
            session_id: Session identifier
            domain: Domain name
            terms: List of term dictionaries with 'source', 'target', 'n_size', 'freq'
            
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            logger.warning("Redis unavailable, glossary not cached")
            return False
        
        if not terms:
            logger.debug("No terms to cache")
            return True
        
        try:
            cache_key = self._glossary_key(session_id, domain)
            
            # Prepare mapping: "source_term|n_size" -> "target_term"
            mapping = {}
            for term in terms:
                field = f"{term['source']}|{term.get('n_size', 1)}"
                mapping[field] = term['target']
            
            # Pipeline execution for atomicity
            pipe = self._client.pipeline()
            pipe.hset(cache_key, mapping=mapping)
            pipe.expire(cache_key, settings.CACHE_TTL_SECONDS)
            pipe.execute()
            
            logger.info(f"Cached {len(terms)} glossary terms for session {session_id}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to bulk cache glossary: {str(e)}")
            return False
    
    def cache_single_term(
        self, 
        session_id: str, 
        domain: str, 
        source: str, 
        target: str,
        n_size: int = 1
    ) -> bool:
        """
        Cache a single glossary term (cache-aside pattern).
        
        Args:
            session_id: Session identifier
            domain: Domain name
            source: Source term
            target: Target translation
            n_size: N-gram size
            
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            cache_key = self._glossary_key(session_id, domain)
            field = f"{source}|{n_size}"
            self._client.hset(cache_key, field, target)
            return True
        except RedisError as e:
            logger.error(f"Failed to cache term: {str(e)}")
            return False

    # =====================================================
    # N-GRAM LOOKUP
    # =====================================================
    
    def lookup_ngrams(
        self, 
        session_id: str, 
        domain: str, 
        ngrams_with_size: List[Tuple[str, int]]
    ) -> List[Optional[bytes]]:
        """
        Look up multiple n-grams efficiently using HMGET.
        
        Args:
            session_id: Session identifier
            domain: Domain name
            ngrams_with_size: List of (term_text, n_size) tuples
            
        Returns:
            List of target translations (None for misses)
        """
        if not self._ensure_connection():
            return [None] * len(ngrams_with_size)
        
        if not ngrams_with_size:
            return []
        
        try:
            cache_key = self._glossary_key(session_id, domain)
            
            # Prepare fields: "source_term|n_size"
            fields = [f"{text}|{size}" for text, size in ngrams_with_size]
            
            # Execute batch query
            results = self._client.hmget(cache_key, fields)
            
            # Log cache stats
            hits = sum(1 for r in results if r is not None)
            logger.debug(f"Cache lookup: {hits}/{len(ngrams_with_size)} hits")
            
            return results
            
        except RedisError as e:
            logger.error(f"Failed to lookup ngrams: {str(e)}")
            return [None] * len(ngrams_with_size)
    
    def lookup_single_term(
        self, 
        session_id: str, 
        domain: str, 
        source_term: str,
        n_size: int = 1
    ) -> Optional[str]:
        """
        Look up a single term from cache.
        
        Args:
            session_id: Session identifier
            domain: Domain name
            source_term: Term to look up
            n_size: N-gram size
            
        Returns:
            Target translation or None if not found
        """
        if not self._ensure_connection():
            return None
        
        try:
            cache_key = self._glossary_key(session_id, domain)
            field = f"{source_term}|{n_size}"
            result = self._client.hget(cache_key, field)
            
            if result:
                return result.decode('utf-8') if isinstance(result, bytes) else result
            return None
            
        except RedisError as e:
            logger.error(f"Failed to lookup term: {str(e)}")
            return None

    # =====================================================
    # CACHE STATISTICS & MANAGEMENT
    # =====================================================
    
    def get_session_stats(self, session_id: str, domain: str) -> Dict[str, Any]:
        """
        Get cache statistics for a session.
        
        Args:
            session_id: Session identifier
            domain: Domain name
            
        Returns:
            Statistics dictionary
        """
        if not self._ensure_connection():
            return {"error": "Redis unavailable"}
        
        try:
            cache_key = self._glossary_key(session_id, domain)
            
            term_count = self._client.hlen(cache_key)
            ttl = self._client.ttl(cache_key)
            memory = self._client.memory_usage(cache_key) or 0
            
            return {
                "session_id": session_id,
                "domain": domain,
                "cached_terms": term_count,
                "ttl_seconds": ttl,
                "memory_bytes": memory
            }
        except RedisError as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_all_sessions(self) -> bool:
        """
        Clear all session data (use with caution).
        
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            # Find all session/glossary keys
            patterns = [f"{self.PREFIX_SESSION}:*", f"{self.PREFIX_GLOSSARY}:*"]
            
            for pattern in patterns:
                keys = list(self._client.scan_iter(match=pattern, count=100))
                if keys:
                    self._client.delete(*keys)
            
            logger.info("All sessions cleared")
            return True
        except RedisError as e:
            logger.error(f"Failed to clear sessions: {str(e)}")
            return False
    
    def list_all_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions with their metadata.
        
        Returns:
            List of session metadata dictionaries
        """
        if not self._ensure_connection():
            return []
        
        try:
            sessions = []
            pattern = f"{self.PREFIX_SESSION}:*:meta"
            
            for key in self._client.scan_iter(match=pattern, count=100):
                # Extract session_id from key
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 2:
                    session_id = parts[1]
                    
                    # Get session metadata
                    session_data = self.get_session(session_id)
                    if session_data:
                        # Add TTL info
                        ttl = self._client.ttl(key)
                        session_data['ttl_seconds'] = ttl
                        sessions.append(session_data)
            
            return sessions
        except RedisError as e:
            logger.error(f"Failed to list sessions: {str(e)}")
            return []
    
    def get_all_session_ids(self) -> List[str]:
        """
        Get all active session IDs.
        
        Returns:
            List of session IDs
        """
        if not self._ensure_connection():
            return []
        
        try:
            session_ids = []
            pattern = f"{self.PREFIX_SESSION}:*:meta"
            
            for key in self._client.scan_iter(match=pattern, count=100):
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 2:
                    session_ids.append(parts[1])
            
            return session_ids
        except RedisError as e:
            logger.error(f"Failed to get session IDs: {str(e)}")
            return []

    def extend_session_ttl(self, session_id: str, domain: str) -> bool:
        """
        Extend TTL for session and glossary cache.
        
        Args:
            session_id: Session identifier
            domain: Domain name
            
        Returns:
            True if successful
        """
        if not self._ensure_connection():
            return False
        
        try:
            session_key = self._session_key(session_id)
            glossary_key = self._glossary_key(session_id, domain)
            
            pipe = self._client.pipeline()
            pipe.expire(session_key, settings.CACHE_TTL_SECONDS)
            pipe.expire(glossary_key, settings.CACHE_TTL_SECONDS)
            pipe.execute()
            
            return True
        except RedisError as e:
            logger.error(f"Failed to extend TTL: {str(e)}")
            return False
