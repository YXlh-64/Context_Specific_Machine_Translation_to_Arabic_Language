"""
Monitoring Module
Performance tracking and logging for the RAG system
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Any, List
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


def monitor_retrieval(func: Callable):
    """
    Decorator to monitor retrieval performance
    
    Tracks:
    - Execution time
    - Number of results
    - Errors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Log performance
            result_count = len(result) if isinstance(result, list) else 1
            
            logger.info(
                f"[PERF] {func.__name__}: "
                f"time={elapsed:.3f}s, "
                f"results={result_count}"
            )
            
            # Add timing to result if it's a list of dicts
            if isinstance(result, list) and result:
                # Don't modify original results
                pass
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[PERF] {func.__name__}: "
                f"time={elapsed:.3f}s, "
                f"error={str(e)}"
            )
            raise
    
    return wrapper


def monitor_async_retrieval(func: Callable):
    """Async version of monitor_retrieval decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            result_count = len(result) if isinstance(result, list) else 1
            
            logger.info(
                f"[PERF] {func.__name__}: "
                f"time={elapsed:.3f}s, "
                f"results={result_count}"
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[PERF] {func.__name__}: "
                f"time={elapsed:.3f}s, "
                f"error={str(e)}"
            )
            raise
    
    return wrapper


class PerformanceMonitor:
    """Track system performance metrics over time"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history = 1000  # Keep last 1000 records per metric
    
    def record(
        self,
        metric_name: str,
        value: float,
        metadata: Dict = None
    ):
        """Record a metric value"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "metadata": metadata or {}
        }
        
        self.metrics[metric_name].append(record)
        
        # Trim history if needed
        if len(self.metrics[metric_name]) > self.max_history:
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_history:]
    
    def record_retrieval(
        self,
        query: str,
        elapsed_time: float,
        result_count: int,
        cache_hit: bool = False,
        domain: str = None
    ):
        """Record retrieval performance"""
        self.record(
            "retrieval",
            elapsed_time,
            {
                "query_length": len(query),
                "result_count": result_count,
                "cache_hit": cache_hit,
                "domain": domain
            }
        )
    
    def get_stats(self, metric_name: str = None) -> Dict:
        """Get statistics for metrics"""
        if metric_name:
            records = self.metrics.get(metric_name, [])
            return self._calculate_stats(metric_name, records)
        
        # Return stats for all metrics
        all_stats = {}
        for name, records in self.metrics.items():
            all_stats[name] = self._calculate_stats(name, records)
        
        return all_stats
    
    def _calculate_stats(self, metric_name: str, records: List[Dict]) -> Dict:
        """Calculate statistics from records"""
        if not records:
            return {
                "metric": metric_name,
                "count": 0
            }
        
        values = [r["value"] for r in records]
        
        return {
            "metric": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "latest_timestamp": records[-1]["timestamp"]
        }
    
    def get_recent(self, metric_name: str, count: int = 10) -> List[Dict]:
        """Get recent records for a metric"""
        records = self.metrics.get(metric_name, [])
        return records[-count:]
    
    def clear(self, metric_name: str = None):
        """Clear metrics"""
        if metric_name:
            self.metrics[metric_name] = []
        else:
            self.metrics.clear()


class QueryLogger:
    """Log queries for analysis"""
    
    def __init__(self, max_queries: int = 10000):
        self.queries: List[Dict] = []
        self.max_queries = max_queries
    
    def log_query(
        self,
        query: str,
        domain: str,
        source_lang: str,
        target_lang: str,
        result_count: int,
        elapsed_time: float,
        cache_hit: bool = False
    ):
        """Log a query"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "domain": domain,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "result_count": result_count,
            "elapsed_time": elapsed_time,
            "cache_hit": cache_hit
        }
        
        self.queries.append(record)
        
        # Trim if needed
        if len(self.queries) > self.max_queries:
            self.queries = self.queries[-self.max_queries:]
    
    def get_recent(self, count: int = 100) -> List[Dict]:
        """Get recent queries"""
        return self.queries[-count:]
    
    def get_stats(self) -> Dict:
        """Get query statistics"""
        if not self.queries:
            return {"total_queries": 0}
        
        cache_hits = sum(1 for q in self.queries if q.get("cache_hit"))
        domains = defaultdict(int)
        for q in self.queries:
            domains[q.get("domain", "unknown")] += 1
        
        times = [q["elapsed_time"] for q in self.queries]
        
        return {
            "total_queries": len(self.queries),
            "cache_hit_rate": cache_hits / len(self.queries) if self.queries else 0,
            "avg_response_time": sum(times) / len(times),
            "queries_by_domain": dict(domains)
        }


# Global instances
_monitor: PerformanceMonitor = None
_query_logger: QueryLogger = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def get_query_logger() -> QueryLogger:
    """Get global query logger"""
    global _query_logger
    if _query_logger is None:
        _query_logger = QueryLogger()
    return _query_logger


if __name__ == "__main__":
    # Test monitoring
    logging.basicConfig(level=logging.INFO)
    
    monitor = PerformanceMonitor()
    
    # Record some test metrics
    for i in range(10):
        monitor.record_retrieval(
            query=f"Test query {i}",
            elapsed_time=0.1 * (i + 1),
            result_count=5,
            cache_hit=(i % 2 == 0),
            domain="health"
        )
    
    print("Stats:", monitor.get_stats())
    print("Recent:", monitor.get_recent("retrieval", 3))
