"""
Comprehensive Test Suite for Glossary System

Test Categories:
1. Unit Tests - Individual component testing with mocks
2. Integration Tests - End-to-end API testing
3. Edge Case Tests - Boundary conditions and error handling
4. Performance Tests - Load and timing verification
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import io
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import (
    TranslationRequest,
    TranslationResponse,
    GlossaryMatch,
    PDFUploadResponse,
    SessionStatusResponse,
    SentenceProcessRequest
)


# =====================================================
# TEST FIXTURES
# =====================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"source_term": "test", "target_term": "اختبار", "n_gram_size": 1, "frequency": 10}
    ]
    mock_cursor.fetchone.return_value = None
    return mock_conn


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.hgetall.return_value = {}
    mock.hset.return_value = True
    return mock


@pytest.fixture
def sample_translation_request():
    """Sample valid translation request."""
    return {
        "text": "This is a test sentence for glossary lookup",
        "source_lang": "en",
        "target_lang": "ar",
        "domain": "technology"
    }


# =====================================================
# UNIT TESTS - LRU CACHE
# =====================================================

class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60)
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
    
    def test_cache_miss_returns_none(self):
        """Test cache miss returns None."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60)
        assert cache.get("nonexistent") is None
    
    def test_cache_eviction_on_max_size(self):
        """Test LRU eviction when cache is full."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=3, default_ttl=60)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key - should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_cache_expiration(self):
        """Test TTL-based expiration."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=1)  # 1 second TTL
        cache.put("key1", "value1")
        
        assert cache.get("key1") == "value1"
        
        time.sleep(1.1)  # Wait for expiration
        
        assert cache.get("key1") is None
    
    def test_cache_thread_safety(self):
        """Test concurrent access to cache."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=1000, default_ttl=60)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    cache.put(key, f"value_{i}")
                    result = cache.get(key)
                    if result != f"value_{i}":
                        errors.append(f"Mismatch: expected value_{i}, got {result}")
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()
        
        assert len(errors) == 0, f"Errors: {errors}"
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=100, default_ttl=60)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["l1_hits"] == 2
        assert stats["l1_misses"] == 1
        assert stats["size"] == 1


# =====================================================
# UNIT TESTS - CONNECTION POOL
# =====================================================

class TestConnectionPool:
    """Test SQLite connection pool."""
    
    def test_pool_initialization(self):
        """Test pool initializes with correct size."""
        from app.core.connection_pool import SQLiteConnectionPool
        
        # Skip if no database file
        try:
            pool = SQLiteConnectionPool(
                db_path="data/glossary.db",
                pool_size=3
            )
            stats = pool.get_stats()
            assert stats["pool_size"] == 3
            pool.close_all()
        except FileNotFoundError:
            pytest.skip("Database file not found")
    
    def test_pool_get_stats(self):
        """Test pool statistics reporting."""
        from app.core.connection_pool import SQLiteConnectionPool
        
        try:
            pool = SQLiteConnectionPool(
                db_path="data/glossary.db",
                pool_size=2
            )
            
            with pool.get_connection() as conn:
                conn.execute("SELECT 1")
            
            stats = pool.get_stats()
            assert stats["total_checkouts"] >= 1
            assert stats["total_checkins"] >= 1
            
            pool.close_all()
        except FileNotFoundError:
            pytest.skip("Database file not found")


# =====================================================
# UNIT TESTS - GLOSSARY SERVICE
# =====================================================

class TestOptimizedGlossaryService:
    """Test optimized glossary service."""
    
    def test_service_initialization(self):
        """Test service initializes correctly."""
        from app.services.optimized_glossary_service import OptimizedGlossaryService
        
        with patch('app.services.optimized_glossary_service.CacheService') as mock_cache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.is_connected = True
            mock_cache_instance.redis = MagicMock()
            mock_cache.return_value = mock_cache_instance
            
            service = OptimizedGlossaryService()
            assert service is not None
    
    def test_build_response_structure(self):
        """Test response structure is correct."""
        from app.services.optimized_glossary_service import OptimizedGlossaryService, LookupMetrics
        from app.models.schemas import GlossaryMatch
        
        with patch('app.services.optimized_glossary_service.CacheService'):
            service = OptimizedGlossaryService()
                
                metrics = LookupMetrics()
                matches = [GlossaryMatch(source_term="test", target_term="اختبار", n_gram_size=1)]
                
                response = service._build_response(
                    matches=matches,
                    text="test sentence",
                    domain="technology",
                    metrics=metrics,
                    start_time=time.time() - 0.01
                )
                
                assert "glossary_matches" in response
                assert "match_count" in response
                assert "source_sentence" in response
                assert "domain" in response
                assert "processing_time_ms" in response
                assert "metrics" in response


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
        assert "name" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_services_health(self, client):
        """Test services health check."""
        response = client.get("/api/v1/health/services")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data


class TestTranslateEndpoint:
    """Test translation endpoint."""
    
    def test_translate_valid_request(self, client, sample_translation_request):
        """Test successful translation request."""
        with patch('app.api.routes.get_glossary_service') as mock_service:
            mock_instance = MagicMock()
            mock_instance.process_request_async = AsyncMock(return_value={
                "glossary_matches": [
                    {"source_term": "test", "target_term": "اختبار", "n_gram_size": 1}
                ],
                "match_count": 1,
                "source_sentence": sample_translation_request["text"],
                "domain": "technology",
                "processing_time_ms": 10.5,
                "metrics": {}
            })
            mock_service.return_value = mock_instance
            
            response = client.post(
                "/api/v1/translate/sentence",
                json=sample_translation_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "glossary_matches" in data
            assert "match_count" in data
    
    def test_translate_same_language_error(self, client):
        """Test error when source and target languages are the same."""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "Test sentence",
            "source_lang": "en",
            "target_lang": "en",
            "domain": "technology"
        })
        
        assert response.status_code == 400
        assert "different" in response.json()["detail"].lower()
    
    def test_translate_invalid_domain(self, client):
        """Test error when domain is invalid."""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "Test sentence",
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "invalid_domain_xyz"
        })
        
        assert response.status_code == 422
    
    def test_translate_empty_text(self, client):
        """Test error when text is empty."""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "",
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "technology"
        })
        
        assert response.status_code == 422


# =====================================================
# EDGE CASE TESTS
# =====================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_text(self, client):
        """Test handling of very long text."""
        long_text = "test " * 1000  # 5000 characters
        
        with patch('app.api.routes.get_glossary_service') as mock_service:
            mock_instance = MagicMock()
            mock_instance.process_request_async = AsyncMock(return_value={
                "glossary_matches": [],
                "match_count": 0,
                "source_sentence": long_text[:100] + "...",
                "domain": "technology",
                "processing_time_ms": 50.0,
                "metrics": {}
            })
            mock_service.return_value = mock_instance
            
            response = client.post("/api/v1/translate/sentence", json={
                "text": long_text,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            })
            
            assert response.status_code == 200
    
    def test_unicode_text(self, client):
        """Test handling of Unicode text."""
        unicode_text = "مرحبا بالعالم Hello World 你好世界 🌍"
        
        with patch('app.api.routes.get_glossary_service') as mock_service:
            mock_instance = MagicMock()
            mock_instance.process_request_async = AsyncMock(return_value={
                "glossary_matches": [],
                "match_count": 0,
                "source_sentence": unicode_text,
                "domain": "technology",
                "processing_time_ms": 5.0,
                "metrics": {}
            })
            mock_service.return_value = mock_instance
            
            response = client.post("/api/v1/translate/sentence", json={
                "text": unicode_text,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            })
            
            assert response.status_code == 200
    
    def test_special_characters(self, client):
        """Test handling of special characters."""
        special_text = "Test with \"quotes\" and 'apostrophes' and <brackets> and &ampersand"
        
        with patch('app.api.routes.get_glossary_service') as mock_service:
            mock_instance = MagicMock()
            mock_instance.process_request_async = AsyncMock(return_value={
                "glossary_matches": [],
                "match_count": 0,
                "source_sentence": special_text,
                "domain": "technology",
                "processing_time_ms": 5.0,
                "metrics": {}
            })
            mock_service.return_value = mock_instance
            
            response = client.post("/api/v1/translate/sentence", json={
                "text": special_text,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            })
            
            assert response.status_code == 200
    
    def test_null_handling(self, client):
        """Test handling of null/None values."""
        response = client.post("/api/v1/translate/sentence", json={
            "text": None,
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "technology"
        })
        
        assert response.status_code == 422
    
    def test_max_text_boundary(self, client):
        """Test text at maximum allowed length."""
        # Assuming max is 5000 characters
        max_text = "a" * 5000
        
        with patch('app.api.routes.get_glossary_service') as mock_service:
            mock_instance = MagicMock()
            mock_instance.process_request_async = AsyncMock(return_value={
                "glossary_matches": [],
                "match_count": 0,
                "source_sentence": max_text,
                "domain": "technology",
                "processing_time_ms": 50.0,
                "metrics": {}
            })
            mock_service.return_value = mock_instance
            
            response = client.post("/api/v1/translate/sentence", json={
                "text": max_text,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            })
            
            assert response.status_code == 200


# =====================================================
# PERFORMANCE TESTS
# =====================================================

class TestPerformance:
    """Performance and load tests."""
    
    def test_cache_performance(self):
        """Test cache lookup is fast."""
        from app.core.lru_cache import LRUCache
        
        cache = LRUCache(max_size=10000, default_ttl=60)
        
        # Populate cache
        for i in range(1000):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Measure lookup time
        start = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        elapsed = (time.time() - start) * 1000
        
        # Should complete 1000 lookups in under 10ms
        assert elapsed < 10, f"Cache too slow: {elapsed}ms for 1000 lookups"
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            with patch('app.api.routes.get_glossary_service') as mock_service:
                mock_instance = MagicMock()
                mock_instance.process_request_async = AsyncMock(return_value={
                    "glossary_matches": [],
                    "match_count": 0,
                    "source_sentence": "test",
                    "domain": "technology",
                    "processing_time_ms": 1.0,
                    "metrics": {}
                })
                mock_service.return_value = mock_instance
                
                return client.get("/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)


# =====================================================
# PDF UPLOAD TESTS
# =====================================================

class TestPDFUpload:
    """Test PDF upload endpoints."""
    
    def test_upload_non_pdf_file(self, client):
        """Test error when uploading non-PDF file."""
        files = {"file": ("test.txt", io.BytesIO(b"text content"), "text/plain")}
        data = {
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "technology"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower()
    
    def test_upload_invalid_domain(self, client):
        """Test error when domain is invalid."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "invalid_domain"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400
    
    def test_upload_same_language(self, client):
        """Test error when source and target languages are same."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {
            "source_lang": "en",
            "target_lang": "en",
            "domain": "technology"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
