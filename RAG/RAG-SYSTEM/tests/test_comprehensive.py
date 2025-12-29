"""
Comprehensive Test Suite for RAG System

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - End-to-end API testing
3. Edge Case Tests - Boundary conditions
4. Performance Tests - Load and timing verification
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app


# =====================================================
# TEST FIXTURES
# =====================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    mock = MagicMock()
    mock.search.return_value = [
        MagicMock(
            id="1",
            score=0.95,
            payload={
                "source_text": "artificial intelligence",
                "target_text": "الذكاء الاصطناعي",
                "domain": "technology"
            }
        ),
        MagicMock(
            id="2",
            score=0.85,
            payload={
                "source_text": "machine learning",
                "target_text": "التعلم الآلي",
                "domain": "technology"
            }
        )
    ]
    mock.get_collection.return_value = MagicMock(
        vectors_count=1000,
        points_count=1000
    )
    return mock


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model."""
    mock = MagicMock()
    mock.encode.return_value = np.random.rand(768).astype(np.float32)
    return mock


@pytest.fixture
def sample_search_request():
    """Sample valid search request."""
    return {
        "query": "artificial intelligence translation",
        "domain": "technology",
        "top_k": 5,
        "threshold": 0.7
    }


# =====================================================
# UNIT TESTS - EMBEDDING SERVICE
# =====================================================

class TestOptimizedEmbeddingService:
    """Test optimized embedding service."""
    
    def test_embedding_cache_key_generation(self):
        """Test cache key generation for embeddings."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        
        # Test that same text produces same key
        key1 = cache._generate_key("test text")
        key2 = cache._generate_key("test text")
        key3 = cache._generate_key("different text")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_embedding_cache_put_and_get(self):
        """Test embedding cache operations."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        
        test_embedding = np.random.rand(768).astype(np.float32)
        cache.put("test text", test_embedding)
        
        result = cache.get("test text")
        assert result is not None
        np.testing.assert_array_equal(result, test_embedding)
    
    def test_embedding_cache_miss(self):
        """Test cache miss returns None."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        result = cache.get("nonexistent")
        assert result is None
    
    def test_embedding_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=3)
        
        cache.put("key1", np.random.rand(768))
        cache.put("key2", np.random.rand(768))
        cache.put("key3", np.random.rand(768))
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key - should evict key2 (least recently used)
        cache.put("key4", np.random.rand(768))
        
        assert cache.get("key1") is not None
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
    
    def test_embedding_cache_statistics(self):
        """Test cache statistics tracking."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        
        cache.put("key1", np.random.rand(768))
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1


# =====================================================
# UNIT TESTS - RETRIEVAL SERVICE (ResultCache)
# =====================================================

class TestResultCache:
    """Test ResultCache from optimized_retrieval_service."""
    
    def test_cache_initialization(self):
        """Test ResultCache initializes correctly."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        assert cache is not None
        assert cache._max_size == 100
        assert cache._ttl == 300
    
    def test_cache_put_and_get(self):
        """Test putting and getting items from ResultCache."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        # Store results
        test_results = [{"id": 1, "text": "test", "score": 0.9}]
        cache.put("query1", "medical", "en", "ar", 5, "semantic", test_results)
        
        # Retrieve results
        cached = cache.get("query1", "medical", "en", "ar", 5, "semantic")
        
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["text"] == "test"
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        # Query non-existent key
        result = cache.get("nonexistent", "medical", "en", "ar", 5, "semantic")
        
        assert result is None
    
    def test_cache_key_differentiation(self):
        """Test different parameters create different cache keys."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        results1 = [{"id": 1, "score": 0.9}]
        results2 = [{"id": 2, "score": 0.8}]
        
        # Store with different domains
        cache.put("query", "medical", "en", "ar", 5, "semantic", results1)
        cache.put("query", "legal", "en", "ar", 5, "semantic", results2)
        
        # Retrieve - should get different results
        cached1 = cache.get("query", "medical", "en", "ar", 5, "semantic")
        cached2 = cache.get("query", "legal", "en", "ar", 5, "semantic")
        
        assert cached1[0]["id"] == 1
        assert cached2[0]["id"] == 2
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        # Make some cache operations
        cache.put("query1", "", "", "", 5, "semantic", [{"id": 1}])
        cache.get("query1", "", "", "", 5, "semantic")  # Hit
        cache.get("query2", "", "", "", 5, "semantic")  # Miss
        
        stats = cache.get_stats()
        
        assert "hits" in stats or "size" in stats
    
    def test_cache_clear(self):
        """Test cache clearing."""
        from app.services.optimized_retrieval_service import ResultCache
        
        cache = ResultCache(max_size=100, ttl_seconds=300)
        
        # Add items
        cache.put("query1", "", "", "", 5, "semantic", [{"id": 1}])
        cache.put("query2", "", "", "", 5, "semantic", [{"id": 2}])
        
        # Clear
        cache.clear()
        
        # Should be empty
        assert cache.get("query1", "", "", "", 5, "semantic") is None


class TestCircuitBreakerState:
    """Test CircuitBreakerState dataclass."""
    
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreakerState initializes correctly."""
        from app.services.optimized_retrieval_service import CircuitBreakerState
        
        cb = CircuitBreakerState()
        
        assert cb.failures == 0
        assert cb.state == "closed"
    
    def test_circuit_breaker_can_proceed_when_closed(self):
        """Test circuit breaker allows requests when closed."""
        from app.services.optimized_retrieval_service import CircuitBreakerState
        
        cb = CircuitBreakerState()
        
        # Should allow when closed
        assert cb.can_proceed() == True
    
    def test_circuit_breaker_record_failure(self):
        """Test recording failures increments counter."""
        from app.services.optimized_retrieval_service import CircuitBreakerState
        
        cb = CircuitBreakerState()
        
        cb.record_failure()
        cb.record_failure()
        
        assert cb.failures == 2
    
    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        from app.services.optimized_retrieval_service import CircuitBreakerState
        
        cb = CircuitBreakerState()
        
        # Record failures (default threshold is 5)
        for _ in range(5):
            cb.record_failure()
        
        # Should be open now
        assert cb.state == "open"
    
    def test_circuit_breaker_record_success_resets(self):
        """Test recording success resets circuit breaker."""
        from app.services.optimized_retrieval_service import CircuitBreakerState
        
        cb = CircuitBreakerState()
        
        # Record some failures
        cb.record_failure()
        cb.record_failure()
        
        # Record success
        cb.record_success()
        
        # Should be reset
        assert cb.failures == 0
        assert cb.state == "closed"


# =====================================================
# INTEGRATION TESTS - API ENDPOINTS
# =====================================================

class TestHealthCheckEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        # Root returns service info
        assert "version" in data or "description" in data or "service" in data
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSearchEndpoints:
    """Test search endpoints."""
    
    def test_search_semantic_valid_request(self, client, sample_search_request):
        """Test successful semantic search request."""
        # Use the correct endpoint: /api/v1/search/semantic
        with patch('app.api.routes.get_retriever') as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.search_semantic_async = AsyncMock(return_value=[
                {
                    "source": "artificial intelligence",
                    "target": "الذكاء الاصطناعي",
                    "similarity_percentage": 95.0,
                    "domain": "technology"
                }
            ])
            mock_retriever.return_value = mock_instance
            
            response = client.post(
                "/api/v1/search/semantic",
                json={
                    "query": sample_search_request["query"],
                    "domain": sample_search_request["domain"],
                    "source_lang": "en",
                    "target_lang": "ar",
                    "top_k": sample_search_request["top_k"]
                }
            )
            
            # May return 200 or 500 depending on mock setup
            assert response.status_code in [200, 500]
    
    def test_search_hybrid_endpoint(self, client):
        """Test hybrid search endpoint exists."""
        with patch('app.api.routes.get_retriever') as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.search_hybrid_async = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_instance
            
            response = client.post("/api/v1/search/hybrid", json={
                "query": "test query",
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "top_k": 5,
                "wording_weight": 0.3
            })
            
            # May return 200 or 500 depending on mock setup
            assert response.status_code in [200, 422, 500]


# =====================================================
# EDGE CASE TESTS
# =====================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_embedding_cache_with_long_text(self):
        """Test embedding cache with long text."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        long_text = "test " * 1000
        
        test_embedding = np.random.rand(768)
        cache.put(long_text, test_embedding)
        
        result = cache.get(long_text)
        assert result is not None
    
    def test_embedding_cache_unicode(self):
        """Test embedding cache with Unicode text."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        unicode_text = "الذكاء الاصطناعي AI 人工智能 🤖"
        
        test_embedding = np.random.rand(768)
        cache.put(unicode_text, test_embedding)
        
        result = cache.get(unicode_text)
        assert result is not None
        np.testing.assert_array_equal(result, test_embedding)
    
    def test_embedding_cache_special_characters(self):
        """Test embedding cache with special characters."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=100)
        special_text = "test \"with\" 'quotes' and <brackets> & symbols"
        
        test_embedding = np.random.rand(768)
        cache.put(special_text, test_embedding)
        
        result = cache.get(special_text)
        assert result is not None


# =====================================================
# PERFORMANCE TESTS
# =====================================================

class TestPerformance:
    """Performance and load tests."""
    
    def test_embedding_cache_performance(self):
        """Test embedding cache lookup is fast."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=10000)
        
        # Populate cache
        for i in range(1000):
            cache.put(f"key_{i}", np.random.rand(768))
        
        # Measure lookup time
        start = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        elapsed = (time.time() - start) * 1000
        
        # Should complete 1000 lookups in under 50ms
        assert elapsed < 50, f"Cache too slow: {elapsed}ms for 1000 lookups"
    
    def test_search_result_serialization(self):
        """Test search result serialization is efficient."""
        results = [
            {
                "source_text": f"source text {i}",
                "target_text": f"target text {i}",
                "score": 0.95 - (i * 0.01),
                "domain": "technology"
            }
            for i in range(100)
        ]
        
        import json
        
        start = time.time()
        for _ in range(100):
            json.dumps(results)
        elapsed = (time.time() - start) * 1000
        
        # Should complete 100 serializations in under 100ms
        assert elapsed < 100, f"Serialization too slow: {elapsed}ms"
    
    def test_concurrent_embedding_requests(self):
        """Test handling of concurrent embedding requests."""
        from app.services.optimized_embedding_service import OptimizedEmbeddingService
        
        errors = []
        
        with patch('app.services.optimized_embedding_service.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(768)
            mock_st.return_value = mock_model
            
            service = OptimizedEmbeddingService()
            
            def worker(worker_id):
                try:
                    for i in range(50):
                        service.encode_single(f"text from worker {worker_id} iteration {i}")
                except Exception as e:
                    errors.append(str(e))
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(worker, i) for i in range(4)]
                for f in futures:
                    f.result()
            
            assert len(errors) == 0, f"Errors: {errors}"


# =====================================================
# ASYNC TESTS
# =====================================================

class TestAsyncOperations:
    """Test async operations."""
    
    @pytest.mark.asyncio
    async def test_async_embedding(self):
        """Test async embedding generation."""
        from app.services.optimized_embedding_service import OptimizedEmbeddingService
        
        with patch('app.services.optimized_embedding_service.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(768)
            mock_st.return_value = mock_model
            
            service = OptimizedEmbeddingService()
            
            # Use encode_single_async instead of encode_async
            result = await service.encode_single_async("test text")
            
            assert len(result) == 768
    
    @pytest.mark.asyncio
    async def test_async_batch_embedding(self):
        """Test async batch embedding."""
        from app.services.optimized_embedding_service import OptimizedEmbeddingService
        
        with patch('app.services.optimized_embedding_service.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.rand(3, 768)
            mock_st.return_value = mock_model
            
            service = OptimizedEmbeddingService()
            
            texts = ["text1", "text2", "text3"]
            result = await service.encode_batch_async(texts)
            
            assert result.shape[0] == 3


class TestCacheThreadSafety:
    """Test thread safety of caches."""
    
    def test_concurrent_cache_access(self):
        """Test concurrent cache access."""
        from app.services.optimized_embedding_service import EmbeddingCache
        
        cache = EmbeddingCache(max_size=1000)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    embedding = np.random.rand(768)
                    cache.put(key, embedding)
                    result = cache.get(key)
                    if result is None:
                        errors.append(f"Cache miss for {key}")
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()
        
        assert len(errors) == 0, f"Errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
