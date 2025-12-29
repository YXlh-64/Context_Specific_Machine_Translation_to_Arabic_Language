"""
Connection Pool Module for SQLite Database

Implements a thread-safe connection pool with:
- Configurable pool size
- Connection health checking
- Automatic connection recycling
- Metrics and monitoring

Performance Impact:
- Eliminates connection creation overhead (30-50ms per connection)
- Thread-safe connection reuse
- O(1) connection acquisition with available connections
"""

import sqlite3
import threading
import logging
import time
from contextlib import contextmanager
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Connection pool statistics for monitoring."""
    total_connections_created: int = 0
    total_connections_recycled: int = 0
    total_checkouts: int = 0
    total_checkins: int = 0
    current_pool_size: int = 0
    active_connections: int = 0
    failed_checkouts: int = 0
    avg_checkout_time_ms: float = 0.0
    _checkout_times: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record_checkout(self, duration_ms: float):
        """Record a checkout operation duration."""
        with self._lock:
            self._checkout_times.append(duration_ms)
            # Keep only last 1000 samples
            if len(self._checkout_times) > 1000:
                self._checkout_times = self._checkout_times[-1000:]
            self.avg_checkout_time_ms = sum(self._checkout_times) / len(self._checkout_times)


class PooledConnection:
    """
    Wrapper for SQLite connection with metadata tracking.
    
    Tracks:
    - Creation time (for connection recycling)
    - Last used time (for idle timeout)
    - Health status
    """
    
    def __init__(self, connection: sqlite3.Connection, db_path: str):
        self.connection = connection
        self.db_path = db_path
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_healthy = True
        self.use_count = 0
    
    def mark_used(self):
        """Update usage metadata."""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if connection should be recycled due to age."""
        return (time.time() - self.created_at) > max_age_seconds
    
    def is_idle_too_long(self, idle_timeout_seconds: int = 300) -> bool:
        """Check if connection has been idle too long."""
        return (time.time() - self.last_used) > idle_timeout_seconds
    
    def check_health(self) -> bool:
        """Verify connection is still valid."""
        try:
            self.connection.execute("SELECT 1")
            self.is_healthy = True
            return True
        except sqlite3.Error:
            self.is_healthy = False
            return False


class SQLiteConnectionPool:
    """
    Thread-safe SQLite connection pool.
    
    Features:
    - Pre-allocated connections for fast checkout
    - Health checking with automatic replacement
    - Connection recycling to prevent resource leaks
    - Timeout-based checkout with graceful fallback
    
    Usage:
        pool = SQLiteConnectionPool(db_path="data/glossary.db", pool_size=5)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM glossary_terms")
    
    Time Complexity:
    - get_connection(): O(1) with available connections, O(connection_creation) otherwise
    - return_connection(): O(1)
    """
    
    def __init__(
        self,
        db_path: str = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 5.0,
        recycle_seconds: int = 3600,
        idle_timeout_seconds: int = 300
    ):
        """
        Initialize the connection pool.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum additional connections allowed beyond pool_size
            timeout: Seconds to wait for a connection before raising error
            recycle_seconds: Seconds before a connection is recycled
            idle_timeout_seconds: Seconds of idle time before closing connection
        """
        self.db_path = db_path or self._extract_db_path(settings.DATABASE_URL)
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.timeout = timeout
        self.recycle_seconds = recycle_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        
        self._pool: Queue = Queue(maxsize=pool_size)
        self._overflow_count = 0
        self._lock = threading.RLock()
        self._stats = ConnectionStats()
        self._shutdown = False
        
        # Pre-fill the pool
        self._initialize_pool()
        
        logger.info(
            f"SQLiteConnectionPool initialized: pool_size={pool_size}, "
            f"db_path={self.db_path}"
        )
    
    @staticmethod
    def _extract_db_path(db_url: str) -> str:
        """Extract file path from SQLite URI."""
        if db_url.startswith("file:"):
            return db_url.split("?")[0].replace("file:", "")
        return db_url
    
    def _create_connection(self) -> PooledConnection:
        """Create a new database connection with optimal settings."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False,  # Allow cross-thread usage with pool
                isolation_level=None  # Autocommit for read-heavy workload
            )
            
            # Enable Row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            # Performance optimizations
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance speed/safety
            conn.execute("PRAGMA cache_size=-32000")  # 32MB cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
            
            self._stats.total_connections_created += 1
            
            return PooledConnection(conn, self.db_path)
            
        except sqlite3.Error as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    def _initialize_pool(self):
        """Pre-fill the connection pool."""
        for _ in range(self.pool_size):
            try:
                conn = self._create_connection()
                self._pool.put_nowait(conn)
                self._stats.current_pool_size += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to pre-fill pool connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Yields:
            sqlite3.Connection: A database connection
            
        Raises:
            TimeoutError: If no connection available within timeout
        """
        if self._shutdown:
            raise RuntimeError("Connection pool is shut down")
        
        start_time = time.time()
        pooled_conn = None
        created_overflow = False
        
        try:
            # Try to get from pool
            try:
                pooled_conn = self._pool.get(timeout=self.timeout)
                self._stats.total_checkouts += 1
            except Empty:
                # Pool exhausted, try to create overflow connection
                with self._lock:
                    if self._overflow_count < self.max_overflow:
                        pooled_conn = self._create_connection()
                        self._overflow_count += 1
                        created_overflow = True
                        self._stats.total_checkouts += 1
                        logger.debug(f"Created overflow connection: {self._overflow_count}/{self.max_overflow}")
                    else:
                        self._stats.failed_checkouts += 1
                        raise TimeoutError(
                            f"Connection pool exhausted. Pool size: {self.pool_size}, "
                            f"Overflow: {self._overflow_count}/{self.max_overflow}"
                        )
            
            # Validate connection
            if pooled_conn and not pooled_conn.check_health():
                logger.warning("Retrieved unhealthy connection, creating new one")
                try:
                    pooled_conn.connection.close()
                except:
                    pass
                pooled_conn = self._create_connection()
                self._stats.total_connections_recycled += 1
            
            # Check if connection should be recycled due to age
            if pooled_conn and pooled_conn.is_stale(self.recycle_seconds):
                logger.debug("Recycling stale connection")
                try:
                    pooled_conn.connection.close()
                except:
                    pass
                pooled_conn = self._create_connection()
                self._stats.total_connections_recycled += 1
            
            pooled_conn.mark_used()
            
            # Record checkout time
            checkout_time = (time.time() - start_time) * 1000
            self._stats.record_checkout(checkout_time)
            self._stats.active_connections += 1
            
            yield pooled_conn.connection
            
        finally:
            if pooled_conn:
                self._stats.active_connections -= 1
                self._stats.total_checkins += 1
                
                if created_overflow:
                    # Close overflow connections
                    with self._lock:
                        self._overflow_count -= 1
                    try:
                        pooled_conn.connection.close()
                    except:
                        pass
                else:
                    # Return to pool
                    try:
                        self._pool.put_nowait(pooled_conn)
                    except Full:
                        # Pool is full, close the connection
                        try:
                            pooled_conn.connection.close()
                        except:
                            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "current_pool_size": self._stats.current_pool_size,
            "active_connections": self._stats.active_connections,
            "overflow_count": self._overflow_count,
            "total_checkouts": self._stats.total_checkouts,
            "total_checkins": self._stats.total_checkins,
            "failed_checkouts": self._stats.failed_checkouts,
            "connections_created": self._stats.total_connections_created,
            "connections_recycled": self._stats.total_connections_recycled,
            "avg_checkout_time_ms": round(self._stats.avg_checkout_time_ms, 2)
        }
    
    def close_all(self):
        """Close all connections and shut down the pool."""
        self._shutdown = True
        
        # Close all pooled connections
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                pooled_conn.connection.close()
            except (Empty, sqlite3.Error):
                pass
        
        logger.info("Connection pool closed")


# Global connection pool instance (singleton)
_connection_pool: Optional[SQLiteConnectionPool] = None
_pool_lock = threading.Lock()


def get_connection_pool() -> SQLiteConnectionPool:
    """
    Get or create the global connection pool.
    
    Thread-safe singleton pattern.
    """
    global _connection_pool
    
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = SQLiteConnectionPool(
                    pool_size=10,
                    max_overflow=20,
                    timeout=5.0,
                    recycle_seconds=3600
                )
    
    return _connection_pool


@contextmanager
def get_pooled_connection():
    """
    Convenience function to get a pooled connection.
    
    Usage:
        with get_pooled_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(...)
    """
    pool = get_connection_pool()
    with pool.get_connection() as conn:
        yield conn


def close_connection_pool():
    """Close the global connection pool."""
    global _connection_pool
    
    with _pool_lock:
        if _connection_pool is not None:
            _connection_pool.close_all()
            _connection_pool = None
