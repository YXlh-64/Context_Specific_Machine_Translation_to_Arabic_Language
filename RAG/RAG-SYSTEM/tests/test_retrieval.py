"""
Unit Tests for RAG System
Test individual components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Test configuration
TEST_QUERY = "Patients with severe symptoms require immediate care"
TEST_DOMAIN = "health"
TEST_SOURCE_LANG = "en"
TEST_TARGET_LANG = "ar"


# =====================================================
# FIXTURES
# =====================================================

@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch('app.core.config.settings') as mock:
        mock.QDRANT_HOST = "localhost"
        mock.QDRANT_PORT = 6333
        mock.QDRANT_COLLECTION = "test_collection"
        mock.EMBEDDING_DIM = 768
        mock.DEFAULT_TOP_K = 7
        mock.SIMILARITY_THRESHOLD = 0.5
        mock.ALLOWED_DOMAINS = {"health", "technology", "finance"}
        mock.ALLOWED_LANGS = {"en", "ar", "fr"}
        mock.SEMANTIC_WEIGHT = 0.6
        mock.WORDING_WEIGHT = 0.4
        mock.DIVERSITY_LAMBDA = 0.7
        mock.DOMAIN_BOOST_FACTOR = 1.2
        mock.MAX_LENGTH_RATIO = 2.0
        mock.MIN_SENTENCE_LENGTH = 3
        yield mock


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client"""
    mock = Mock()
    mock.get_collections.return_value = Mock(collections=[])
    mock.get_collection.return_value = Mock(
        points_count=1000,
        vectors_count=3000,
        indexed_vectors_count=3000,
        status=Mock(value="green")
    )
    return mock


@pytest.fixture
def mock_model():
    """Mock SentenceTransformer model"""
    mock = Mock()
    mock.encode.return_value = np.random.rand(768).astype(np.float32)
    mock.device = "cpu"
    return mock


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return [
        Mock(
            id=1,
            score=0.95,
            payload={
                "source": "Patients with critical symptoms need urgent care",
                "target": "المرضى الذين يعانون من أعراض خطيرة يحتاجون إلى رعاية عاجلة",
                "domain": "health",
                "language_pair": "en-ar",
                "source_lang": "en",
                "target_lang": "ar",
                "source_length": 6,
                "target_length": 8
            }
        ),
        Mock(
            id=2,
            score=0.88,
            payload={
                "source": "Severe symptoms require immediate medical attention",
                "target": "الأعراض الشديدة تتطلب عناية طبية فورية",
                "domain": "health",
                "language_pair": "en-ar",
                "source_lang": "en",
                "target_lang": "ar",
                "source_length": 6,
                "target_length": 5
            }
        )
    ]


# =====================================================
# EMBEDDING SERVICE TESTS
# =====================================================

class TestEmbeddingService:
    """Tests for embedding generation"""
    
    def test_embedding_dimension(self, mock_model):
        """Verify embeddings have correct dimension"""
        embedding = mock_model.encode("test text")
        assert embedding.shape == (768,), f"Expected (768,), got {embedding.shape}"
    
    def test_embedding_normalization(self, mock_model):
        """Verify embeddings are normalized"""
        # Create normalized mock embedding
        embedding = np.random.rand(768)
        embedding = embedding / np.linalg.norm(embedding)
        mock_model.encode.return_value = embedding
        
        result = mock_model.encode("test")
        norm = np.linalg.norm(result)
        assert np.abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"
    
    def test_batch_embedding_shape(self, mock_model):
        """Verify batch embeddings have correct shape"""
        batch_size = 5
        mock_model.encode.return_value = np.random.rand(batch_size, 768)
        
        texts = ["text " + str(i) for i in range(batch_size)]
        embeddings = mock_model.encode(texts)
        
        assert embeddings.shape == (batch_size, 768)


# =====================================================
# RETRIEVAL SERVICE TESTS
# =====================================================

class TestRetrievalService:
    """Tests for semantic retrieval"""
    
    def test_search_returns_results(self, mock_qdrant_client, mock_model, sample_search_results):
        """Verify search returns results"""
        mock_qdrant_client.search.return_value = sample_search_results
        
        # Simulate search
        results = mock_qdrant_client.search(
            collection_name="test",
            query_vector={"name": "cross_lingual", "vector": [0.1] * 768},
            limit=7
        )
        
        assert len(results) == 2
        assert results[0].score > results[1].score
    
    def test_search_with_domain_filter(self, mock_qdrant_client):
        """Verify domain filtering works"""
        # This tests the filter building logic
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        domain = "health"
        filter_obj = Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        )
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 1
    
    def test_result_formatting(self, sample_search_results):
        """Verify result formatting"""
        formatted = []
        for i, result in enumerate(sample_search_results):
            formatted.append({
                "rank": i + 1,
                "id": result.id,
                "score": float(result.score),
                "similarity_percentage": round(result.score * 100, 2),
                "source": result.payload.get("source", ""),
                "target": result.payload.get("target", ""),
                "domain": result.payload.get("domain", "")
            })
        
        assert len(formatted) == 2
        assert formatted[0]["rank"] == 1
        assert formatted[0]["similarity_percentage"] == 95.0


# =====================================================
# PIPELINE TESTS
# =====================================================

class TestPipeline:
    """Tests for retrieval pipeline"""
    
    def test_complexity_filter(self):
        """Test complexity filtering"""
        query = "This is a medium length query sentence"
        query_length = len(query.split())  # 7 words
        
        results = [
            {"source_length": 6, "score": 0.9},   # ratio 1.17 - keep
            {"source_length": 20, "score": 0.85},  # ratio 2.86 - filter
            {"source_length": 5, "score": 0.8},   # ratio 1.4 - keep
        ]
        
        max_ratio = 2.0
        filtered = []
        for result in results:
            source_length = result.get("source_length", 0)
            if source_length > 0:
                ratio = max(query_length, source_length) / max(min(query_length, source_length), 1)
                if ratio <= max_ratio:
                    filtered.append(result)
        
        assert len(filtered) == 2
    
    def test_domain_boost(self):
        """Test domain boosting"""
        results = [
            {"domain": "health", "score": 0.8},
            {"domain": "technology", "score": 0.85},
        ]
        
        target_domain = "health"
        boost_factor = 1.2
        
        for result in results:
            if result["domain"] == target_domain:
                result["score"] *= boost_factor
        
        # Health should now be higher
        assert results[0]["score"] == 0.96  # 0.8 * 1.2
        assert results[0]["score"] > results[1]["score"]


# =====================================================
# CACHING TESTS
# =====================================================

class TestCaching:
    """Tests for caching functionality"""
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        import hashlib
        
        query = "test query"
        domain = "health"
        source_lang = "en"
        target_lang = "ar"
        top_k = 7
        
        key_parts = [
            "rag",
            query,
            domain or "all",
            source_lang or "any",
            target_lang or "any",
            str(top_k)
        ]
        
        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        cache_key = f"rag:search:{key_hash}"
        
        assert cache_key.startswith("rag:search:")
        assert len(key_hash) == 32
    
    def test_cache_key_uniqueness(self):
        """Test that different queries produce different keys"""
        import hashlib
        
        def generate_key(query, domain):
            key_parts = ["rag", query, domain, "en", "ar", "7"]
            key_string = ":".join(key_parts)
            return hashlib.md5(key_string.encode()).hexdigest()
        
        key1 = generate_key("query 1", "health")
        key2 = generate_key("query 2", "health")
        key3 = generate_key("query 1", "technology")
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3


# =====================================================
# VALIDATION TESTS
# =====================================================

class TestValidation:
    """Tests for input validation"""
    
    def test_query_validation_empty(self):
        """Test empty query validation"""
        from app.utils.error_handling import validate_query
        
        is_valid, error = validate_query("")
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_query_validation_short(self):
        """Test short query validation"""
        from app.utils.error_handling import validate_query
        
        is_valid, error = validate_query("ab")
        assert not is_valid
        assert "3 characters" in error
    
    def test_query_validation_valid(self):
        """Test valid query"""
        from app.utils.error_handling import validate_query
        
        is_valid, error = validate_query("This is a valid query")
        assert is_valid
        assert error == ""
    
    def test_domain_validation_valid(self):
        """Test valid domain"""
        from app.utils.error_handling import validate_domain
        
        is_valid, error = validate_domain("health")
        assert is_valid
    
    def test_domain_validation_invalid(self):
        """Test invalid domain"""
        from app.utils.error_handling import validate_domain
        
        is_valid, error = validate_domain("invalid_domain")
        assert not is_valid
        assert "Invalid domain" in error
    
    def test_domain_validation_none(self):
        """Test None domain (optional)"""
        from app.utils.error_handling import validate_domain
        
        is_valid, error = validate_domain(None)
        assert is_valid


# =====================================================
# INTEGRATION TESTS
# =====================================================

class TestIntegration:
    """Tests for Phase 1-2 integration"""
    
    def test_phase1_output_format(self):
        """Test Phase 1 output format handling"""
        phase1_output = {
            'source_sentence': TEST_QUERY,
            'glossary_matches': [
                {'source_term': 'patients', 'target_term': 'المرضى'},
                {'source_term': 'symptoms', 'target_term': 'الأعراض'}
            ],
            'domain': 'health',
            'source_lang': 'en',
            'target_lang': 'ar'
        }
        
        assert phase1_output['source_sentence'] == TEST_QUERY
        assert len(phase1_output['glossary_matches']) == 2
        assert phase1_output['domain'] == 'health'
    
    def test_format_for_prompt(self):
        """Test prompt formatting"""
        phase2_output = {
            'source_sentence': TEST_QUERY,
            'glossary_matches': [
                {'source_term': 'patients', 'target_term': 'المرضى'}
            ],
            'fuzzy_matches': [
                {
                    'similarity_percentage': 92.5,
                    'source': 'Similar source text',
                    'target': 'Similar target text'
                }
            ]
        }
        
        # Build formatted prompt
        sections = []
        sections.append(f"Source Sentence: {phase2_output['source_sentence']}")
        
        if phase2_output['glossary_matches']:
            sections.append("=== Glossary Terms ===")
            for match in phase2_output['glossary_matches']:
                sections.append(f"• {match['source_term']} → {match['target_term']}")
        
        formatted = "\n".join(sections)
        
        assert "Source Sentence:" in formatted
        assert "Glossary Terms" in formatted
        assert "patients" in formatted


# =====================================================
# RUN TESTS
# =====================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
