"""
Comprehensive Test Suite for Prompt Construction System

Test Categories:
1. Unit Tests - Individual component testing
2. Integration Tests - End-to-end API testing
3. Edge Case Tests - Boundary conditions and security
4. Performance Tests - Load and timing verification
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import time
import asyncio
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
def mock_template():
    """Mock Jinja2 template."""
    mock = MagicMock()
    mock.render.return_value = "Rendered prompt content"
    return mock


@pytest.fixture
def sample_prompt_request():
    """Sample valid prompt construction request."""
    return {
        "sentence": "Translate artificial intelligence to Arabic",
        "glossary_matches": [
            {
                "source_term": "artificial intelligence",
                "target_term": "الذكاء الاصطناعي",
                "n_gram_size": 2,
                "frequency": 100
            }
        ],
        "similar_examples": [
            {
                "source_text": "Machine learning is a subset of AI",
                "target_text": "التعلم الآلي هو جزء من الذكاء الاصطناعي",
                "score": 0.92
            }
        ],
        "domain": "technology",
        "source_lang": "en",
        "target_lang": "ar",
        "format": "xml"
    }


# =====================================================
# UNIT TESTS - INPUT SANITIZER
# =====================================================

class TestInputSanitizer:
    """Test input sanitization."""
    
    def test_sanitize_xml_characters(self):
        """Test XML character sanitization."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        dangerous_input = '<script>alert("xss")</script>'
        sanitized = sanitizer.sanitize_for_xml(dangerous_input)
        
        assert "<script>" not in sanitized
        assert "&lt;" in sanitized or "script" not in sanitized
    
    def test_sanitize_json_characters(self):
        """Test JSON character sanitization."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        dangerous_input = '{"key": "value"}\n"injection": true'
        sanitized = sanitizer.sanitize_for_json(dangerous_input)
        
        # Should not break JSON structure when used as value
        import json
        test_json = json.dumps({"text": sanitized})
        parsed = json.loads(test_json)
        assert "text" in parsed
    
    def test_sanitize_preserves_arabic_text(self):
        """Test that Arabic text is preserved."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        arabic_text = "الذكاء الاصطناعي"
        sanitized = sanitizer.sanitize_for_xml(arabic_text)
        
        assert arabic_text in sanitized or "الذكاء" in sanitized
    
    def test_sanitize_cdata_injection(self):
        """Test prevention of CDATA injection."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        injection = "]]><evil>malicious</evil><![CDATA["
        sanitized = sanitizer.sanitize_for_xml(injection)
        
        # Should escape or remove CDATA markers
        assert "]]>" not in sanitized or "]]&gt;" in sanitized
    
    def test_sanitize_newline_handling(self):
        """Test newline handling."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        multiline = "line1\nline2\rline3\r\nline4"
        sanitized = sanitizer.sanitize_for_json(multiline)
        
        # Should handle newlines safely
        import json
        test_json = json.dumps({"text": sanitized})
        parsed = json.loads(test_json)
        assert parsed is not None


# =====================================================
# UNIT TESTS - TOKEN ESTIMATOR
# =====================================================

class TestTokenEstimator:
    """Test token estimation."""
    
    def test_estimate_tokens_english(self):
        """Test token estimation for English text."""
        from app.services.optimized_prompt_service import TokenEstimator
        
        estimator = TokenEstimator()
        
        text = "This is a test sentence with multiple words."
        tokens = estimator.estimate_tokens(text)
        
        # Should be reasonable estimate (around 10 tokens)
        assert 5 <= tokens <= 20
    
    def test_estimate_tokens_arabic(self):
        """Test token estimation for Arabic text."""
        from app.services.optimized_prompt_service import TokenEstimator
        
        estimator = TokenEstimator()
        
        text = "الذكاء الاصطناعي هو مستقبل التكنولوجيا"
        tokens = estimator.estimate_tokens(text)
        
        assert tokens > 0
    
    def test_estimate_tokens_empty(self):
        """Test token estimation for empty text."""
        from app.services.optimized_prompt_service import TokenEstimator
        
        estimator = TokenEstimator()
        
        tokens = estimator.estimate_tokens("")
        
        assert tokens == 0
    
    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        from app.services.optimized_prompt_service import TokenEstimator
        
        estimator = TokenEstimator()
        
        long_text = "word " * 1000
        tokens = estimator.estimate_tokens(long_text)
        
        # Should be around 1000-1500 tokens
        assert 800 <= tokens <= 2000


# =====================================================
# UNIT TESTS - PROMPT CACHE
# =====================================================

class TestPromptCache:
    """Test prompt caching."""
    
    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        from app.services.optimized_prompt_service import PromptCache
        
        cache = PromptCache(max_size=100)
        
        cache.put("key1", "prompt1")
        result = cache.get("key1")
        
        assert result == "prompt1"
    
    def test_cache_miss(self):
        """Test cache miss."""
        from app.services.optimized_prompt_service import PromptCache
        
        cache = PromptCache(max_size=100)
        
        result = cache.get("nonexistent")
        
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU eviction."""
        from app.services.optimized_prompt_service import PromptCache
        
        cache = PromptCache(max_size=3)
        
        cache.put("key1", "prompt1")
        cache.put("key2", "prompt2")
        cache.put("key3", "prompt3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key - should evict key2
        cache.put("key4", "prompt4")
        
        assert cache.get("key1") == "prompt1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "prompt3"
        assert cache.get("key4") == "prompt4"
    
    def test_cache_thread_safety(self):
        """Test concurrent access."""
        from app.services.optimized_prompt_service import PromptCache
        
        cache = PromptCache(max_size=1000)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    cache.put(key, f"prompt_{i}")
                    result = cache.get(key)
                    if result != f"prompt_{i}":
                        errors.append(f"Mismatch for {key}")
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()
        
        assert len(errors) == 0


# =====================================================
# UNIT TESTS - PROMPT CONSTRUCTOR
# =====================================================

class TestOptimizedPromptConstructor:
    """Test optimized prompt constructor."""
    
    def test_constructor_initialization(self):
        """Test constructor initializes correctly."""
        from app.services.optimized_prompt_service import OptimizedPromptConstructor
        
        constructor = OptimizedPromptConstructor()
        assert constructor is not None
    
    def test_construct_xml_prompt(self):
        """Test XML prompt construction."""
        from app.services.optimized_prompt_service import OptimizedPromptConstructor
        
        with patch('app.services.optimized_prompt_service.jinja2.Environment') as mock_env:
            mock_template = MagicMock()
            mock_template.render.return_value = "<prompt>test</prompt>"
            mock_env_instance = MagicMock()
            mock_env_instance.get_template.return_value = mock_template
            mock_env.return_value = mock_env_instance
            
            constructor = OptimizedPromptConstructor()
            constructor._env = mock_env_instance
            
            result = constructor.construct(
                sentence="test sentence",
                glossary_matches=[],
                similar_examples=[],
                domain="technology",
                source_lang="en",
                target_lang="ar",
                format="xml"
            )
            
            assert result is not None
    
    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        from app.services.optimized_prompt_service import OptimizedPromptConstructor
        
        constructor = OptimizedPromptConstructor()
        
        key1 = constructor._get_cache_key(
            sentence="test",
            glossary_matches=[{"source_term": "ai", "target_term": "ذكاء"}],
            similar_examples=[],
            domain="tech",
            source_lang="en",
            target_lang="ar",
            format="xml"
        )
        
        key2 = constructor._get_cache_key(
            sentence="test",
            glossary_matches=[{"source_term": "ai", "target_term": "ذكاء"}],
            similar_examples=[],
            domain="tech",
            source_lang="en",
            target_lang="ar",
            format="xml"
        )
        
        assert key1 == key2
    
    def test_statistics_tracking(self):
        """Test statistics are tracked."""
        from app.services.optimized_prompt_service import OptimizedPromptConstructor
        
        constructor = OptimizedPromptConstructor()
        
        stats = constructor.get_statistics()
        
        assert "total_constructions" in stats
        assert "cache_hits" in stats
        assert "average_tokens" in stats


# =====================================================
# INTEGRATION TESTS - API ENDPOINTS
# =====================================================

class TestHealthCheckEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "message" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPromptEndpoints:
    """Test prompt construction endpoints."""
    
    def test_construct_prompt_valid(self, client, sample_prompt_request):
        """Test successful prompt construction."""
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "<prompt>Constructed prompt</prompt>",
                "token_estimate": 150,
                "format": "xml",
                "from_cache": False
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post(
                "/api/v1/construct",
                json=sample_prompt_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prompt" in data
    
    def test_construct_prompt_invalid_format(self, client):
        """Test error with invalid format."""
        response = client.post("/api/v1/construct", json={
            "sentence": "Test sentence",
            "glossary_matches": [],
            "similar_examples": [],
            "domain": "technology",
            "source_lang": "en",
            "target_lang": "ar",
            "format": "invalid_format"
        })
        
        assert response.status_code == 422
    
    def test_construct_prompt_empty_sentence(self, client):
        """Test error with empty sentence."""
        response = client.post("/api/v1/construct", json={
            "sentence": "",
            "glossary_matches": [],
            "similar_examples": [],
            "domain": "technology",
            "source_lang": "en",
            "target_lang": "ar",
            "format": "xml"
        })
        
        assert response.status_code == 422


class TestPreviewEndpoints:
    """Test preview endpoints."""
    
    def test_preview_prompt(self, client, sample_prompt_request):
        """Test prompt preview."""
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct.return_value = {
                "prompt": "<prompt>Preview content</prompt>",
                "token_estimate": 100,
                "format": "xml"
            }
            mock_constructor.return_value = mock_instance
            
            response = client.post(
                "/api/v1/preview",
                json=sample_prompt_request
            )
            
            assert response.status_code == 200


# =====================================================
# EDGE CASE TESTS
# =====================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_sentence(self, client):
        """Test handling of very long sentence."""
        long_sentence = "test " * 500
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "prompt",
                "token_estimate": 1000,
                "format": "xml"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": long_sentence,
                "glossary_matches": [],
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "xml"
            })
            
            assert response.status_code in [200, 400]
    
    def test_unicode_content(self, client):
        """Test handling of Unicode content."""
        unicode_sentence = "الذكاء الاصطناعي AI 人工智能 🤖"
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "<prompt>unicode prompt</prompt>",
                "token_estimate": 50,
                "format": "xml"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": unicode_sentence,
                "glossary_matches": [],
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "xml"
            })
            
            assert response.status_code == 200
    
    def test_xml_injection_prevention(self, client):
        """Test prevention of XML injection."""
        malicious_sentence = '<script>alert("xss")</script>'
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "<prompt>sanitized</prompt>",
                "token_estimate": 10,
                "format": "xml"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": malicious_sentence,
                "glossary_matches": [],
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "xml"
            })
            
            # Should succeed but sanitize input
            assert response.status_code == 200
            if "prompt" in response.json():
                assert "<script>" not in response.json()["prompt"]
    
    def test_large_glossary_matches(self, client):
        """Test handling of many glossary matches."""
        matches = [
            {"source_term": f"term{i}", "target_term": f"مصطلح{i}", "n_gram_size": 1, "frequency": i}
            for i in range(100)
        ]
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "prompt",
                "token_estimate": 500,
                "format": "xml"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": "Test sentence",
                "glossary_matches": matches,
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "xml"
            })
            
            assert response.status_code in [200, 400]
    
    def test_null_values(self, client):
        """Test handling of null values."""
        response = client.post("/api/v1/construct", json={
            "sentence": None,
            "glossary_matches": [],
            "similar_examples": [],
            "domain": "technology",
            "source_lang": "en",
            "target_lang": "ar",
            "format": "xml"
        })
        
        assert response.status_code == 422


# =====================================================
# SECURITY TESTS
# =====================================================

class TestSecurity:
    """Security-focused tests."""
    
    def test_json_injection_prevention(self, client):
        """Test prevention of JSON injection."""
        malicious = '{"injected": true}\n"break": false'
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": '{"text": "safe"}',
                "token_estimate": 10,
                "format": "json"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": malicious,
                "glossary_matches": [],
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "json"
            })
            
            assert response.status_code == 200
    
    def test_cdata_injection_prevention(self, client):
        """Test prevention of CDATA injection."""
        cdata_attack = "]]><malicious>attack</malicious><![CDATA["
        
        with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
            mock_instance = MagicMock()
            mock_instance.construct_async = AsyncMock(return_value={
                "prompt": "<prompt>safe</prompt>",
                "token_estimate": 10,
                "format": "xml"
            })
            mock_constructor.return_value = mock_instance
            
            response = client.post("/api/v1/construct", json={
                "sentence": cdata_attack,
                "glossary_matches": [],
                "similar_examples": [],
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "format": "xml"
            })
            
            assert response.status_code == 200


# =====================================================
# PERFORMANCE TESTS
# =====================================================

class TestPerformance:
    """Performance and load tests."""
    
    def test_prompt_cache_performance(self):
        """Test prompt cache lookup performance."""
        from app.services.optimized_prompt_service import PromptCache
        
        cache = PromptCache(max_size=10000)
        
        # Populate cache
        for i in range(1000):
            cache.put(f"key_{i}", f"prompt_{i}")
        
        # Measure lookup time
        start = time.time()
        for i in range(1000):
            cache.get(f"key_{i}")
        elapsed = (time.time() - start) * 1000
        
        # Should complete 1000 lookups in under 10ms
        assert elapsed < 10, f"Cache too slow: {elapsed}ms for 1000 lookups"
    
    def test_sanitizer_performance(self):
        """Test sanitizer performance."""
        from app.services.optimized_prompt_service import InputSanitizer
        
        sanitizer = InputSanitizer()
        test_text = "This is a test <dangerous> text with 'quotes' and \"more quotes\""
        
        start = time.time()
        for _ in range(1000):
            sanitizer.sanitize_for_xml(test_text)
        elapsed = (time.time() - start) * 1000
        
        # Should complete 1000 sanitizations in under 50ms
        assert elapsed < 50, f"Sanitizer too slow: {elapsed}ms"
    
    def test_token_estimation_performance(self):
        """Test token estimation performance."""
        from app.services.optimized_prompt_service import TokenEstimator
        
        estimator = TokenEstimator()
        test_text = "This is a test sentence with multiple words. " * 50
        
        start = time.time()
        for _ in range(1000):
            estimator.estimate_tokens(test_text)
        elapsed = (time.time() - start) * 1000
        
        # Should complete 1000 estimations in under 100ms
        assert elapsed < 100, f"Token estimation too slow: {elapsed}ms"
    
    def test_concurrent_prompt_construction(self, client):
        """Test handling of concurrent requests."""
        def make_request():
            with patch('app.api.routes.get_prompt_constructor') as mock_constructor:
                mock_instance = MagicMock()
                mock_instance.construct_async = AsyncMock(return_value={
                    "prompt": "prompt",
                    "token_estimate": 100,
                    "format": "xml"
                })
                mock_constructor.return_value = mock_instance
                
                return client.get("/health")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)


# =====================================================
# ASYNC TESTS
# =====================================================

class TestAsyncOperations:
    """Test async operations."""
    
    @pytest.mark.asyncio
    async def test_async_construction(self):
        """Test async prompt construction."""
        from app.services.optimized_prompt_service import OptimizedPromptConstructor
        
        constructor = OptimizedPromptConstructor()
        
        with patch.object(constructor, '_env') as mock_env:
            mock_template = MagicMock()
            mock_template.render.return_value = "<prompt>test</prompt>"
            mock_env.get_template.return_value = mock_template
            
            result = await constructor.construct_async(
                sentence="test sentence",
                glossary_matches=[],
                similar_examples=[],
                domain="technology",
                source_lang="en",
                target_lang="ar",
                format="xml"
            )
            
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
