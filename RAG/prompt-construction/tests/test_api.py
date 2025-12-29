"""
API Tests for Phase 3 Prompt Construction

Tests the FastAPI endpoints with TestClient.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

# API prefix
API_PREFIX = "/api/v1"


# ============================================================================
# Sample Test Data
# ============================================================================

SAMPLE_PROMPT_REQUEST = {
    "source_sentence": "Machine learning is transforming the technology industry.",
    "glossary_matches": [
        {
            "source_term": "machine learning",
            "target_term": "التعلم الآلي",
            "domain": "technology"
        }
    ],
    "fuzzy_matches": [
        {
            "source": "Deep learning transforms industries.",
            "target": "التعلم العميق يحول الصناعات.",
            "similarity_percentage": 85.0
        }
    ],
    "domain": "technology",
    "source_lang": "en",
    "target_lang": "ar",
    "prompt_format": "xml"
}


# ============================================================================
# Health & Info Tests
# ============================================================================

class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get(f"{API_PREFIX}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["phase"] == 3
    
    def test_info_endpoint(self):
        """Test service info endpoint."""
        response = client.get(f"{API_PREFIX}/info")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "available_formats" in data
        assert "available_domains" in data
    
    def test_domains_endpoint(self):
        """Test domains list endpoint."""
        response = client.get(f"{API_PREFIX}/domains")
        assert response.status_code == 200
        data = response.json()
        assert "domains" in data
        assert len(data["domains"]) > 0
    
    def test_formats_endpoint(self):
        """Test formats list endpoint."""
        response = client.get(f"{API_PREFIX}/formats")
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert len(data["formats"]) > 0


# ============================================================================
# Prompt Construction Tests
# ============================================================================

class TestPromptConstructionAPI:
    """Test prompt construction API endpoints."""
    
    def test_construct_prompt_xml(self):
        """Test prompt construction with XML format."""
        response = client.post(f"{API_PREFIX}/prompt/construct", json=SAMPLE_PROMPT_REQUEST)
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "token_count" in data
        assert "<translation_task>" in data["prompt"]
    
    def test_construct_prompt_json_format(self):
        """Test prompt construction with JSON format."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["prompt_format"] = "json"
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert '"glossary"' in data["prompt"]
    
    def test_construct_prompt_markdown_format(self):
        """Test prompt construction with Markdown format."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["prompt_format"] = "markdown"
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "##" in data["prompt"]
    
    def test_construct_prompt_plain_format(self):
        """Test prompt construction with Plain format."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["prompt_format"] = "plain"
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
    
    def test_construct_prompt_with_system_message(self):
        """Test that system message is included when requested."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["include_system_message"] = True
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "system_message" in data
        assert data["system_message"] is not None
    
    def test_construct_prompt_empty_matches(self):
        """Test prompt construction with empty matches."""
        request = {
            "source_sentence": "Test sentence",
            "glossary_matches": [],
            "fuzzy_matches": [],
            "domain": "general"
        }
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
    
    def test_preview_prompts(self):
        """Test prompt preview in all formats."""
        response = client.post(f"{API_PREFIX}/prompt/preview", json=SAMPLE_PROMPT_REQUEST)
        assert response.status_code == 200
        data = response.json()
        assert "previews" in data
        assert "xml" in data["previews"]
        assert "json" in data["previews"]
        assert "markdown" in data["previews"]
        assert "plain" in data["previews"]


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Test input validation."""
    
    def test_missing_source_sentence(self):
        """Test validation error for missing source sentence."""
        request = {
            "glossary_matches": [],
            "fuzzy_matches": []
        }
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_domain(self):
        """Test that invalid domain still works (domain is a string field, not strictly validated)."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["domain"] = "invalid_domain_xyz"
        
        # Since domain is a string field (not enum-validated), this should succeed
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 200  # Domain is a string, accepts any value
    
    def test_invalid_prompt_format(self):
        """Test validation error for invalid prompt format."""
        request = SAMPLE_PROMPT_REQUEST.copy()
        request["prompt_format"] = "invalid_format"
        
        response = client.post(f"{API_PREFIX}/prompt/construct", json=request)
        assert response.status_code == 422  # Validation error


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
