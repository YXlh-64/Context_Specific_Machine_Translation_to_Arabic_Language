import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from app.main import app
from app.models.schemas import (
    TranslationRequest, 
    TranslationResponse, 
    GlossaryMatch,
    PDFUploadResponse,
    SessionStatusResponse,
    SentenceProcessRequest
)
from app.services.optimized_glossary_service import OptimizedGlossaryService
from app.services.pdf_service import PDFService

client = TestClient(app)


# =====================================================
# HEALTH CHECK TESTS
# =====================================================

class TestHealthCheck:
    """Test the health check endpoints"""
    
    def test_root_returns_api_info(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_check_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_services_health_check(self):
        response = client.get("/api/v1/health/services")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "config" in data


# =====================================================
# SENTENCE TRANSLATION TESTS
# =====================================================

class TestTranslateEndpoint:
    """Test the /api/v1/translate/sentence endpoint"""
    
    def test_translate_valid_request(self):
        """Test successful translation request"""
        with patch.object(OptimizedGlossaryService, 'process_request_async') as mock_process:
            mock_process.return_value = {
                "glossary_matches": [
                    {"source_term": "test", "target_term": "اختبار", "n_gram_size": 1}
                ],
                "match_count": 1,
                "source_sentence": "This is a test",
                "domain": "technology",
                "processing_time_ms": 10.5
            }
            
            response = client.post("/api/v1/translate/sentence", json={
                "text": "This is a test",
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["match_count"] == 1
            assert data["domain"] == "technology"
    
    def test_translate_same_language_error(self):
        """Test error when source and target languages are the same"""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "This is a test",
            "source_lang": "en",
            "target_lang": "en",
            "domain": "technology"
        })
        
        assert response.status_code == 400
        assert "different" in response.json()["detail"].lower()
    
    def test_translate_invalid_domain(self):
        """Test error when domain is invalid"""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "This is a test",
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "invalid_domain"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_translate_invalid_language(self):
        """Test error when language is invalid"""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "This is a test",
            "source_lang": "invalid",
            "target_lang": "ar",
            "domain": "technology"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_translate_empty_text(self):
        """Test error when text is empty"""
        response = client.post("/api/v1/translate/sentence", json={
            "text": "",
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "technology"
        })
        
        assert response.status_code == 422  # Validation error


# =====================================================
# PDF UPLOAD TESTS
# =====================================================

class TestPDFUploadEndpoint:
    """Test the /api/v1/translate/pdf endpoint"""
    
    def test_upload_pdf_valid_request(self):
        """Test successful PDF upload"""
        with patch.object(PDFService, 'initialize_session', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {
                "session_id": "pdf_test123",
                "status": "initialized",
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "glossary_terms_loaded": 100,
                "total_pages": 5,
                "cache_expires_in_seconds": 7200
            }
            
            # Create a fake PDF file
            pdf_content = b"%PDF-1.4 fake pdf content"
            files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
            data = {
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology",
                "auto_process": "false"
            }
            
            response = client.post("/api/v1/translate/pdf", files=files, data=data)
            
            assert response.status_code == 201
            result = response.json()
            assert result["session_id"] == "pdf_test123"
            assert result["status"] == "initialized"
    
    def test_upload_non_pdf_file_error(self):
        """Test error when uploading non-PDF file"""
        files = {"file": ("test.txt", io.BytesIO(b"text content"), "text/plain")}
        data = {
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "technology"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower()
    
    def test_upload_invalid_domain_error(self):
        """Test error when domain is invalid"""
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {
            "source_lang": "en",
            "target_lang": "ar",
            "domain": "invalid_domain"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400
        assert "domain" in response.json()["detail"].lower()
    
    def test_upload_same_language_error(self):
        """Test error when source and target are same"""
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        data = {
            "source_lang": "en",
            "target_lang": "en",
            "domain": "technology"
        }
        
        response = client.post("/api/v1/translate/pdf", files=files, data=data)
        
        assert response.status_code == 400
        assert "different" in response.json()["detail"].lower()


# =====================================================
# SESSION STATUS TESTS
# =====================================================

class TestSessionEndpoints:
    """Test session management endpoints"""
    
    def test_get_session_status_valid(self):
        """Test getting session status"""
        with patch.object(PDFService, 'get_session_status') as mock_status:
            mock_status.return_value = {
                "session_id": "pdf_test123",
                "status": "ready",
                "domain": "technology",
                "source_lang": "en",
                "target_lang": "ar",
                "glossary_count": 100,
                "total_pages": 5,
                "processed_sentences": 0,
                "created_at": "2025-01-01T00:00:00"
            }
            
            response = client.get("/api/v1/session/pdf_test123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "pdf_test123"
            assert data["status"] == "ready"
    
    def test_delete_session_valid(self):
        """Test session cleanup"""
        with patch.object(PDFService, 'cleanup_session') as mock_cleanup:
            mock_cleanup.return_value = {
                "status": "deleted",
                "session_id": "pdf_test123"
            }
            
            response = client.delete("/api/v1/session/pdf_test123")
            
            assert response.status_code == 200
            assert response.json()["status"] == "deleted"
    
    def test_delete_session_not_found(self):
        """Test cleanup for non-existent session"""
        with patch.object(PDFService, 'cleanup_session') as mock_cleanup:
            mock_cleanup.return_value = {
                "status": "not_found",
                "session_id": "nonexistent"
            }
            
            response = client.delete("/api/v1/session/nonexistent")
            
            assert response.status_code == 404


# =====================================================
# PDF EXTRACTION TESTS
# =====================================================

class TestPDFExtractionEndpoint:
    """Test PDF text extraction endpoint"""
    
    def test_extract_text_valid(self):
        """Test successful text extraction"""
        with patch.object(PDFService, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                "session_id": "pdf_test123",
                "total_pages": 5,
                "pages_extracted": 5,
                "total_characters": 1000,
                "pages": [
                    {"page_number": 1, "text": "Page 1 content", "char_count": 200}
                ],
                "full_text": "Page 1 content..."
            }
            
            response = client.get("/api/v1/session/pdf_test123/extract")
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "pdf_test123"
            assert data["total_pages"] == 5
    
    def test_extract_text_specific_pages(self):
        """Test extraction of specific pages"""
        with patch.object(PDFService, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = {
                "session_id": "pdf_test123",
                "total_pages": 5,
                "pages_extracted": 2,
                "total_characters": 400,
                "pages": [
                    {"page_number": 1, "text": "Page 1", "char_count": 200},
                    {"page_number": 3, "text": "Page 3", "char_count": 200}
                ],
                "full_text": "Page 1\n\nPage 3"
            }
            
            response = client.get("/api/v1/session/pdf_test123/extract?pages=1,3")
            
            assert response.status_code == 200
            mock_extract.assert_called_with("pdf_test123", [1, 3])


# =====================================================
# BATCH PROCESSING TESTS
# =====================================================

class TestBatchProcessingEndpoint:
    """Test batch PDF processing endpoint"""
    
    def test_batch_process_sync(self):
        """Test synchronous batch processing"""
        with patch.object(PDFService, 'process_pdf_batch') as mock_batch:
            mock_batch.return_value = {
                "session_id": "pdf_test123",
                "status": "completed",
                "total_pages": 5,
                "total_sentences": 50,
                "total_matches": 25,
                "results": [],
                "processing_time_ms": 1500.0
            }
            
            response = client.post("/api/v1/session/pdf_test123/process/batch")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["total_sentences"] == 50
    
    def test_batch_process_async(self):
        """Test asynchronous batch processing"""
        response = client.post("/api/v1/session/pdf_test123/process/batch?async_mode=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"


# =====================================================
# SERVICE TESTS
# =====================================================

class TestOptimizedGlossaryService:
    """Test the OptimizedGlossaryService class"""
    
    def test_process_request_returns_dict(self):
        """Test that process_request returns proper structure"""
        service = OptimizedGlossaryService()
        
        with patch.object(service, '_query_database_optimized', return_value=[]):
            result = service.process_request(
                "test sentence",
                "en",
                "ar",
                "technology"
            )
            
            assert isinstance(result, dict)
            assert "glossary_matches" in result
            assert "match_count" in result
            assert "source_sentence" in result
            assert "domain" in result
            assert "processing_time_ms" in result


class TestPDFService:
    """Test the PDFService class"""
    
    def test_validate_file_non_pdf(self):
        """Test file validation rejects non-PDF"""
        from fastapi import HTTPException
        
        service = PDFService()
        mock_file = MagicMock()
        mock_file.filename = "test.txt"
        
        with pytest.raises(HTTPException) as exc_info:
            service._validate_file(mock_file)
        
        assert exc_info.value.status_code == 400


# =====================================================
# TEXT PROCESSOR TESTS
# =====================================================

class TestTextProcessor:
    """Test text processing utilities"""
    
    def test_normalize_text_english(self):
        from app.utils.text_processor import normalize_text
        
        result = normalize_text("  Hello   World  ", "en")
        assert result == "hello world"
    
    def test_normalize_text_arabic(self):
        from app.utils.text_processor import normalize_text
        
        # Test Arabic alef normalization
        result = normalize_text("أحمد", "ar")
        assert "ا" in result  # Alef should be normalized
    
    def test_tokenize_english(self):
        from app.utils.text_processor import tokenize
        
        result = tokenize("hello world", "en")
        assert isinstance(result, list)
        assert len(result) >= 2
    
    def test_generate_ngrams(self):
        from app.utils.text_processor import generate_ngrams
        
        tokens = ["hello", "world", "test"]
        result = generate_ngrams(tokens, min_n=1, max_n=2)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('text' in item for item in result)
        assert all('n_size' in item for item in result)


# =====================================================
# SCHEMA TESTS
# =====================================================

class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_translation_request_valid(self):
        request = TranslationRequest(
            text="Hello world",
            source_lang="en",
            target_lang="ar",
            domain="technology"
        )
        assert request.text == "Hello world"
        assert request.domain == "technology"
    
    def test_translation_request_invalid_domain(self):
        with pytest.raises(ValueError):
            TranslationRequest(
                text="Hello world",
                source_lang="en",
                target_lang="ar",
                domain="invalid"
            )
    
    def test_glossary_match_creation(self):
        match = GlossaryMatch(
            source_term="test",
            target_term="اختبار",
            n_gram_size=1
        )
        assert match.source_term == "test"
        assert match.n_gram_size == 1
    
    def test_pdf_upload_response(self):
        response = PDFUploadResponse(
            session_id="pdf_test123",
            status="initialized",
            domain="technology",
            source_lang="en",
            target_lang="ar",
            glossary_terms_loaded=100,
            total_pages=5,
            cache_expires_in_seconds=7200
        )
        assert response.session_id == "pdf_test123"
        assert response.glossary_terms_loaded == 100
    
    def test_session_status_response(self):
        response = SessionStatusResponse(
            session_id="pdf_test123",
            status="ready",
            domain="technology",
            source_lang="en",
            target_lang="ar",
            glossary_count=100,
            total_pages=5,
            processed_sentences=0
        )
        assert response.status == "ready"


# =====================================================
# INTEGRATION TESTS
# =====================================================

class TestIntegration:
    """Integration tests for the full workflow"""
    
    def test_full_pdf_workflow(self):
        """Test the complete PDF processing workflow"""
        # This would be a full integration test
        # In production, you'd mock the database and Redis
        pass
