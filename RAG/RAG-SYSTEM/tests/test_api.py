"""
Integration Tests for RAG System API
Test complete API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, AsyncMock
import numpy as np


# =====================================================
# FIXTURES
# =====================================================

@pytest.fixture
def mock_retriever():
    """Mock SemanticRetriever"""
    mock = Mock()
    # Regular methods
    mock.search_semantic.return_value = [
        {
            "rank": 1,
            "id": 1,
            "score": 0.95,
            "similarity_percentage": 95.0,
            "source": "Patients with critical symptoms need urgent care",
            "target": "المرضى ذوي الأعراض الحرجة يحتاجون رعاية عاجلة",
            "domain": "health",
            "language_pair": "en-ar",
            "source_lang": "en",
            "target_lang": "ar",
            "source_length": 6,
            "target_length": 6,
            "search_type": "semantic"
        }
    ]
    mock.search_hybrid.return_value = mock.search_semantic.return_value
    mock.search_with_diversity.return_value = mock.search_semantic.return_value
    
    # New method for pairs by domain
    mock.get_all_pairs_by_domain.return_value = [
        {
            "id": 1,
            "source": "Patients with critical symptoms need urgent care",
            "target": "المرضى ذوي الأعراض الحرجة يحتاجون رعاية عاجلة",
            "domain": "health",
            "language_pair": "en-ar",
            "source_lang": "en",
            "target_lang": "ar",
            "source_length": 6,
            "target_length": 6
        },
        {
            "id": 2,
            "source": "Regular checkups prevent diseases",
            "target": "الفحوصات الدورية تمنع الأمراض",
            "domain": "health",
            "language_pair": "en-ar",
            "source_lang": "en",
            "target_lang": "ar",
            "source_length": 4,
            "target_length": 4
        }
    ]
    
    # Async methods
    mock.search_semantic_async = AsyncMock(return_value=mock.search_semantic.return_value)
    
    mock.get_stats.return_value = {
        "collection": "translation_memory",
        "points_count": 1000,
        "vectors_count": 3000
    }
    return mock


@pytest.fixture
def mock_cache():
    """Mock ResultCache"""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.is_connected = True
    mock.get_stats.return_value = {
        "status": "connected",
        "cache_entries": 10
    }
    return mock


@pytest.fixture
def client(mock_retriever, mock_cache):
    """Create test client with mocked dependencies"""
    # get_retriever is async, so we need to patch it to return the mock_retriever when awaited
    async def async_get_retriever():
        return mock_retriever
        
    with patch('app.api.routes.get_retriever', side_effect=async_get_retriever):
        with patch('app.api.routes.get_cache', return_value=mock_cache):
            with patch('app.services.embedding_service.get_model') as mock_model:
                mock_model.return_value = Mock()
                mock_model.return_value.encode.return_value = np.random.rand(768)
                
                from app.main import app
                yield TestClient(app)


# =====================================================
# ROOT ENDPOINT TESTS
# =====================================================

class TestRootEndpoints:
    """Test root endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "RAG System" in data["service"]
    
    def test_root_health(self, client):
        """Test root health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# =====================================================
# SEARCH ENDPOINT TESTS
# =====================================================

class TestSearchEndpoints:
    """Test search endpoints"""
    
    def test_search_semantic_post(self, client):
        """Test POST /api/v1/search/semantic"""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "Patients with severe symptoms require immediate care",
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert "elapsed_ms" in data
    
    def test_search_semantic_invalid_query(self, client):
        """Test search with invalid query"""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "ab",  # Too short
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar"
            }
        )
        
        # Should fail validation
        assert response.status_code == 422 or response.status_code == 400
    
    def test_search_get_endpoint(self, client):
        """Test GET /api/v1/search"""
        response = client.get(
            "/api/v1/search",
            params={
                "query": "Test search query",
                "domain": "health",
                "top_k": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
    
    def test_search_hybrid_post(self, client):
        """Test POST /api/v1/search/hybrid"""
        response = client.post(
            "/api/v1/search/hybrid",
            json={
                "query": "Patients with severe symptoms require immediate care",
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar",
                "top_k": 5,
                "semantic_weight": 0.6,
                "wording_weight": 0.4,
                "enable_diversity": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


# =====================================================
# INTEGRATION ENDPOINT TESTS
# =====================================================

class TestIntegrationEndpoints:
    """Test Phase 1-2 integration endpoints"""
    
    def test_integrate_endpoint(self, client):
        """Test POST /api/v1/integrate"""
        response = client.post(
            "/api/v1/integrate",
            json={
                "source_sentence": "Patients with severe symptoms require immediate care",
                "glossary_matches": [
                    {"source_term": "patients", "target_term": "المرضى"}
                ],
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "glossary_matches" in data
        assert "fuzzy_matches" in data
        assert "glossary_count" in data
        assert "fuzzy_count" in data
    
    def test_integrate_format_endpoint(self, client):
        """Test POST /api/v1/integrate/format"""
        response = client.post(
            "/api/v1/integrate/format",
            json={
                "source_sentence": "Patients with severe symptoms require immediate care",
                "glossary_matches": [
                    {"source_term": "patients", "target_term": "المرضى"}
                ],
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "formatted_prompt" in data


# =====================================================
# HEALTH & STATS ENDPOINT TESTS
# =====================================================

class TestHealthEndpoints:
    """Test health and stats endpoints"""
    
    def test_health_endpoint(self, client):
        """Test GET /api/v1/health"""
        with patch('app.api.routes.get_qdrant_client') as mock_client:
            with patch('app.api.routes.verify_qdrant_connection', return_value=True):
                with patch('app.api.routes.get_collection_info', return_value={
                    "name": "test",
                    "points_count": 1000,
                    "vectors_count": 3000,
                    "indexed_vectors_count": 3000,
                    "status": "green"
                }):
                    with patch('app.services.embedding_service.verify_model', return_value={
                        "status": "ok",
                        "model": "LaBSE"
                    }):
                        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "qdrant" in data
        assert "redis" in data
    
    def test_stats_endpoint(self, client):
        """Test GET /api/v1/stats"""
        with patch('app.api.routes.get_qdrant_client') as mock_client:
            with patch('app.api.routes.get_collection_info', return_value={
                "name": "test",
                "points_count": 1000,
                "vectors_count": 3000,
                "indexed_vectors_count": 3000,
                "status": "green"
            }):
                response = client.get("/api/v1/stats")
        
        assert response.status_code == 200


# =====================================================
# CACHE ENDPOINT TESTS
# =====================================================

class TestCacheEndpoints:
    """Test cache management endpoints"""
    
    def test_cache_stats_endpoint(self, client):
        """Test GET /api/v1/cache/stats"""
        response = client.get("/api/v1/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_cache_clear_endpoint(self, client):
        """Test DELETE /api/v1/cache"""
        response = client.delete("/api/v1/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


# =====================================================
# QUERY LOGGING ENDPOINT TESTS
# =====================================================

class TestQueryLoggingEndpoints:
    """Test query logging endpoints"""
    
    def test_recent_queries_endpoint(self, client):
        """Test GET /api/v1/queries/recent"""
        response = client.get("/api/v1/queries/recent", params={"count": 10})
        
        assert response.status_code == 200
        data = response.json()
        assert "queries" in data
        assert "stats" in data


class TestPairsEndpoints:
    """Test pairs by domain endpoints"""
    
    def test_get_pairs_by_domain_all(self, client):
        """Test getting all pairs without domain filter"""
        response = client.get("/api/v1/pairs/domain")
        assert response.status_code == 200
        data = response.json()
        assert "domain" in data
        assert "total_pairs" in data
        assert "pairs" in data
        assert data["domain"] is None
        assert data["total_pairs"] == 2
        assert len(data["pairs"]) == 2
    
    def test_get_pairs_by_domain_filtered(self, client):
        """Test getting pairs filtered by domain"""
        response = client.get("/api/v1/pairs/domain?domain=health")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "health"
        assert data["total_pairs"] == 2
        assert len(data["pairs"]) == 2
        # Check that all pairs have the correct domain
        for pair in data["pairs"]:
            assert pair["domain"] == "health"


# =====================================================
# ERROR HANDLING TESTS
# =====================================================

class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_domain(self, client):
        """Test search with invalid domain"""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "Valid query text here",
                "domain": "invalid_domain_xyz",
                "source_lang": "en",
                "target_lang": "ar"
            }
        )
        
        # Should return 400 for invalid domain
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    def test_missing_required_field(self, client):
        """Test search with missing required field"""
        response = client.post(
            "/api/v1/search/semantic",
            json={
                # Missing 'query' field
                "domain": "health"
            }
        )
        
        assert response.status_code == 422


# =====================================================
# RUN TESTS
# =====================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
