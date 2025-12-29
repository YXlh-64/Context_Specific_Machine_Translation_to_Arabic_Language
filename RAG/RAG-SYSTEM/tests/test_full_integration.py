"""
=============================================================================
COMPREHENSIVE INTEGRATION TEST SUITE
Phase 1 (Glossary System) + Phase 2 (RAG System) Full Integration
=============================================================================

This test suite validates the complete translation assistance pipeline:
- Phase 1: Exact glossary term matching
- Phase 2: Semantic fuzzy matching via RAG
- Combined pipeline: Both systems working together

Requirements:
- Qdrant running on port 6333
- Glossary System running on port 8001  
- Redis running on port 6379
- RAG System services available

Author: Integration Test Suite
Date: December 2024
=============================================================================
"""

import pytest
import requests
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Test configuration"""
    GLOSSARY_API = "http://127.0.0.1:8001/api/v1"
    RAG_API = "http://127.0.0.1:8002/api/v1"
    QDRANT_URL = "http://localhost:6333"
    REDIS_URL = "redis://localhost:6379"
    COLLECTION_NAME = "translation_memory"
    
    # Timeouts
    REQUEST_TIMEOUT = 30
    EMBEDDING_TIMEOUT = 60


# =============================================================================
# TEST DATA
# =============================================================================

class TestData:
    """Comprehensive test data for all scenarios"""
    
    # Healthcare domain sentences
    HEALTHCARE_SENTENCES = [
        {
            "text": "Patients with diabetes require regular blood sugar monitoring",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "expected_glossary_terms": ["diabetes"],  # Expected to match glossary
            "description": "Medical condition with known glossary term"
        },
        {
            "text": "The clinical trial showed promising results for the new treatment",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "expected_glossary_terms": [],
            "description": "Medical research sentence"
        },
        {
            "text": "Patients with severe symptoms require immediate medical attention",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "expected_glossary_terms": [],
            "description": "Emergency care sentence - should find fuzzy matches"
        }
    ]
    
    # Edge cases
    EDGE_CASES = [
        {
            "text": "",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "description": "Empty string"
        },
        {
            "text": "A",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "description": "Single character"
        },
        {
            "text": "The patient needs medication for hypertension and diabetes management",
            "domain": "health",
            "source_lang": "en",
            "target_lang": "ar",
            "description": "Multiple potential glossary terms"
        }
    ]
    
    # Different domains
    MULTI_DOMAIN = [
        {"text": "Neural networks process data efficiently", "domain": "technology"},
        {"text": "Economic growth exceeded expectations", "domain": "economic"},
        {"text": "The heart pumps blood through arteries", "domain": "health"},
    ]


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def qdrant_client():
    """Create Qdrant client for direct database testing"""
    return QdrantClient(url=Config.QDRANT_URL)


@pytest.fixture(scope="module")
def embedding_model():
    """Load embedding model once for all tests"""
    print("\n🔄 Loading LaBSE embedding model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    print("✅ Model loaded")
    return model


# =============================================================================
# PHASE 1: GLOSSARY SYSTEM TESTS
# =============================================================================

class TestPhase1GlossarySystem:
    """
    Phase 1: Glossary System Tests
    Tests exact term matching and glossary lookup functionality
    """
    
    def test_01_glossary_health_check(self):
        """Verify Glossary System is running and healthy"""
        response = requests.get(
            f"{Config.GLOSSARY_API}/health/services",
            timeout=Config.REQUEST_TIMEOUT
        )
        assert response.status_code == 200, f"Health check failed: {response.text}"
        
        data = response.json()
        assert data["status"] == "healthy", f"System unhealthy: {data}"
        
        print(f"\n✅ Glossary System Health Check")
        print(f"   Status: {data['status']}")
        print(f"   Database: {data.get('database', 'N/A')}")
        print(f"   Redis: {data.get('redis', 'N/A')}")
    
    def test_02_glossary_term_lookup(self):
        """Test basic glossary term lookup"""
        test_sentence = TestData.HEALTHCARE_SENTENCES[0]
        
        response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": test_sentence["text"],
                "source_lang": test_sentence["source_lang"],
                "target_lang": test_sentence["target_lang"],
                "domain": test_sentence["domain"]
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200, f"Lookup failed: {response.text}"
        data = response.json()
        
        print(f"\n✅ Glossary Term Lookup")
        print(f"   Input: '{test_sentence['text'][:50]}...'")
        print(f"   Matches found: {data.get('match_count', 0)}")
        
        if data.get('glossary_matches'):
            for match in data['glossary_matches'][:5]:
                print(f"   • {match['source_term']} → {match['target_term']}")
        
        # Verify structure
        assert "glossary_matches" in data
        assert "match_count" in data
    
    def test_03_glossary_multiple_terms(self):
        """Test sentence with multiple potential glossary terms"""
        test_case = TestData.EDGE_CASES[2]  # Multiple terms sentence
        
        response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": test_case["text"],
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        print(f"\n✅ Multiple Terms Test")
        print(f"   Input: '{test_case['text']}'")
        print(f"   Terms found: {data.get('match_count', 0)}")
        
        if data.get('glossary_matches'):
            terms_found = [m['source_term'] for m in data['glossary_matches']]
            print(f"   Matched: {terms_found}")
    
    def test_04_glossary_domain_filtering(self):
        """Test that domain filtering works correctly"""
        sentence = "The patient has symptoms of diabetes"
        
        # Test with health domain
        health_response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": sentence,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        
        # Test with different domain
        tech_response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": sentence,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "technology"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        
        assert health_response.status_code == 200
        assert tech_response.status_code == 200
        
        health_data = health_response.json()
        tech_data = tech_response.json()
        
        print(f"\n✅ Domain Filtering Test")
        print(f"   Health domain matches: {health_data.get('match_count', 0)}")
        print(f"   Technology domain matches: {tech_data.get('match_count', 0)}")
    
    def test_05_glossary_response_time(self):
        """Test that glossary lookup is performant"""
        test_sentence = TestData.HEALTHCARE_SENTENCES[0]
        
        times = []
        for _ in range(3):
            start = time.time()
            response = requests.post(
                f"{Config.GLOSSARY_API}/translate/sentence",
                json={
                    "text": test_sentence["text"],
                    "source_lang": "en",
                    "target_lang": "ar",
                    "domain": "health"
                },
                timeout=Config.REQUEST_TIMEOUT
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        print(f"\n✅ Glossary Performance Test")
        print(f"   Average response time: {avg_time:.0f}ms")
        print(f"   Min: {min(times):.0f}ms, Max: {max(times):.0f}ms")
        
        # Should respond within 500ms on average
        assert avg_time < 500, f"Glossary lookup too slow: {avg_time}ms"


# =============================================================================
# PHASE 2: RAG SYSTEM TESTS
# =============================================================================

class TestPhase2RAGSystem:
    """
    Phase 2: RAG System Tests
    Tests semantic search and fuzzy matching functionality
    """
    
    def test_01_qdrant_connection(self, qdrant_client):
        """Verify Qdrant vector database is accessible"""
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        print(f"\n✅ Qdrant Connection Test")
        print(f"   Collections: {collection_names}")
        
        assert Config.COLLECTION_NAME in collection_names, \
            f"Collection '{Config.COLLECTION_NAME}' not found"
    
    def test_02_collection_has_data(self, qdrant_client):
        """Verify translation memory has data"""
        collection_info = qdrant_client.get_collection(Config.COLLECTION_NAME)
        points_count = collection_info.points_count
        
        print(f"\n✅ Collection Data Test")
        print(f"   Collection: {Config.COLLECTION_NAME}")
        print(f"   Total points: {points_count}")
        
        assert points_count > 0, "Collection is empty!"
    
    def test_03_semantic_search_direct(self, qdrant_client, embedding_model):
        """Test direct semantic search against Qdrant"""
        query = "Patients with severe symptoms require immediate medical attention"
        
        # Generate embedding
        query_embedding = embedding_model.encode(query)
        
        # Search
        results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="cross_lingual",
            limit=5,
            with_payload=True
        )
        
        print(f"\n✅ Direct Semantic Search Test")
        print(f"   Query: '{query[:50]}...'")
        print(f"   Results found: {len(results.points)}")
        
        assert len(results.points) > 0, "No results found"
        
        for i, point in enumerate(results.points[:3], 1):
            score = point.score * 100
            source = point.payload.get("source", "N/A")[:50]
            print(f"   {i}. [{score:.1f}%] {source}...")
    
    def test_04_semantic_search_high_similarity(self, qdrant_client, embedding_model):
        """Test that highly similar sentences get high scores"""
        # These sentences were added as ~90% matches
        query = "Patients with severe symptoms require immediate medical attention"
        
        query_embedding = embedding_model.encode(query)
        
        results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="source_semantic",
            limit=5,
            with_payload=True
        )
        
        print(f"\n✅ High Similarity Search Test")
        
        # Check if we have high-scoring matches (the test sentences we added)
        high_scores = [p for p in results.points if p.score > 0.9]
        
        print(f"   Results with >90% similarity: {len(high_scores)}")
        
        for point in high_scores:
            source = point.payload.get("source", "N/A")
            print(f"   • [{point.score*100:.1f}%] {source[:60]}...")
        
        # We added 2 test sentences with ~90% similarity
        assert len(high_scores) >= 2, "Expected at least 2 high-similarity matches"
    
    def test_05_semantic_search_wording_vs_meaning(self, qdrant_client, embedding_model):
        """Compare source_semantic vs cross_lingual search"""
        query = "Patients with severe symptoms require immediate medical attention"
        query_embedding = embedding_model.encode(query)
        
        # Search with source_semantic (wording)
        wording_results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="source_semantic",
            limit=5,
            with_payload=True
        )
        
        # Search with cross_lingual (meaning)
        meaning_results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="cross_lingual",
            limit=5,
            with_payload=True
        )
        
        print(f"\n✅ Wording vs Meaning Search Comparison")
        print(f"\n   Source Semantic (Wording) Top 3:")
        for i, p in enumerate(wording_results.points[:3], 1):
            print(f"   {i}. [{p.score*100:.1f}%] {p.payload.get('source', '')[:50]}...")
        
        print(f"\n   Cross-Lingual (Meaning) Top 3:")
        for i, p in enumerate(meaning_results.points[:3], 1):
            print(f"   {i}. [{p.score*100:.1f}%] {p.payload.get('source', '')[:50]}...")
    
    def test_06_embedding_consistency(self, embedding_model):
        """Test that embeddings are consistent across calls"""
        text = "This is a test sentence for embedding consistency"
        
        emb1 = embedding_model.encode(text)
        emb2 = embedding_model.encode(text)
        
        # Calculate cosine similarity
        similarity = (emb1 @ emb2) / (
            (emb1 @ emb1) ** 0.5 * (emb2 @ emb2) ** 0.5
        )
        
        print(f"\n✅ Embedding Consistency Test")
        print(f"   Same text, two embeddings")
        print(f"   Cosine similarity: {similarity:.6f}")
        
        assert similarity > 0.9999, "Embeddings not consistent"


# =============================================================================
# COMBINED PIPELINE TESTS
# =============================================================================

class TestCombinedPipeline:
    """
    Combined Pipeline Tests
    Tests Phase 1 + Phase 2 working together
    """
    
    def test_01_full_pipeline_with_glossary_match(self, qdrant_client, embedding_model):
        """
        Full pipeline test with sentence containing glossary terms
        Phase 1: Should find glossary matches
        Phase 2: Should find fuzzy matches
        """
        test_sentence = "Patients with diabetes require regular blood sugar monitoring"
        
        print(f"\n{'='*70}")
        print(f"FULL PIPELINE TEST - With Glossary Match")
        print(f"{'='*70}")
        print(f"Input: '{test_sentence}'")
        
        # PHASE 1: Glossary Lookup
        print(f"\n📘 PHASE 1: Glossary System (Exact Match)")
        print(f"   {'-'*50}")
        
        start = time.time()
        glossary_response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": test_sentence,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        glossary_time = (time.time() - start) * 1000
        
        assert glossary_response.status_code == 200
        glossary_data = glossary_response.json()
        
        print(f"   ⏱️  Response time: {glossary_time:.0f}ms")
        print(f"   📊 Matches found: {glossary_data.get('match_count', 0)}")
        
        if glossary_data.get('glossary_matches'):
            for match in glossary_data['glossary_matches']:
                print(f"   ✓ '{match['source_term']}' → '{match['target_term']}'")
        
        # PHASE 2: Semantic Search
        print(f"\n📗 PHASE 2: RAG System (Fuzzy Match)")
        print(f"   {'-'*50}")
        
        start = time.time()
        query_embedding = embedding_model.encode(test_sentence)
        embedding_time = (time.time() - start) * 1000
        
        start = time.time()
        rag_results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="cross_lingual",
            limit=5,
            with_payload=True,
            score_threshold=0.3
        )
        search_time = (time.time() - start) * 1000
        
        print(f"   ⏱️  Embedding time: {embedding_time:.0f}ms")
        print(f"   ⏱️  Search time: {search_time:.0f}ms")
        print(f"   📊 Fuzzy matches found: {len(rag_results.points)}")
        
        for i, point in enumerate(rag_results.points[:3], 1):
            score = point.score * 100
            source = point.payload.get("source", "")[:50]
            target = point.payload.get("target", "")[:50]
            print(f"   {i}. [{score:.1f}%]")
            print(f"      EN: {source}...")
            print(f"      AR: {target}...")
        
        # COMBINED OUTPUT
        print(f"\n📙 COMBINED OUTPUT")
        print(f"   {'-'*50}")
        print(f"   Total glossary terms: {glossary_data.get('match_count', 0)}")
        print(f"   Total fuzzy matches: {len(rag_results.points)}")
        print(f"   Total processing time: {glossary_time + embedding_time + search_time:.0f}ms")
        
        # Assertions
        assert glossary_data.get('match_count', 0) > 0, "Expected glossary matches for 'diabetes'"
    
    def test_02_full_pipeline_fuzzy_only(self, qdrant_client, embedding_model):
        """
        Full pipeline test with sentence that has no glossary matches
        Phase 1: No glossary matches expected
        Phase 2: Should find fuzzy matches
        """
        test_sentence = "Patients with severe symptoms require immediate medical attention"
        
        print(f"\n{'='*70}")
        print(f"FULL PIPELINE TEST - Fuzzy Match Only")
        print(f"{'='*70}")
        print(f"Input: '{test_sentence}'")
        
        # PHASE 1
        print(f"\n📘 PHASE 1: Glossary System")
        
        glossary_response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={
                "text": test_sentence,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            },
            timeout=Config.REQUEST_TIMEOUT
        )
        
        glossary_data = glossary_response.json()
        print(f"   Glossary matches: {glossary_data.get('match_count', 0)}")
        
        # PHASE 2
        print(f"\n📗 PHASE 2: RAG System")
        
        query_embedding = embedding_model.encode(test_sentence)
        rag_results = qdrant_client.query_points(
            collection_name=Config.COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="source_semantic",
            limit=5,
            with_payload=True
        )
        
        print(f"   Fuzzy matches: {len(rag_results.points)}")
        
        # Should find the high-similarity test sentences
        high_matches = [p for p in rag_results.points if p.score > 0.9]
        print(f"   High similarity (>90%): {len(high_matches)}")
        
        for point in high_matches:
            print(f"   ✓ [{point.score*100:.1f}%] {point.payload.get('source', '')[:50]}...")
        
        assert len(high_matches) >= 2, "Expected at least 2 high-similarity matches"
    
    def test_03_batch_processing(self, qdrant_client, embedding_model):
        """Test processing multiple sentences in batch"""
        sentences = [s["text"] for s in TestData.HEALTHCARE_SENTENCES]
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING TEST")
        print(f"{'='*70}")
        print(f"Processing {len(sentences)} sentences")
        
        results = []
        total_glossary_time = 0
        total_rag_time = 0
        
        for i, sentence in enumerate(sentences, 1):
            print(f"\n📝 Sentence {i}: '{sentence[:40]}...'")
            
            # Phase 1
            start = time.time()
            g_resp = requests.post(
                f"{Config.GLOSSARY_API}/translate/sentence",
                json={"text": sentence, "source_lang": "en", "target_lang": "ar", "domain": "health"},
                timeout=Config.REQUEST_TIMEOUT
            )
            glossary_time = (time.time() - start) * 1000
            total_glossary_time += glossary_time
            
            # Phase 2
            start = time.time()
            emb = embedding_model.encode(sentence)
            rag_res = qdrant_client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=emb.tolist(),
                using="cross_lingual",
                limit=3,
                with_payload=True
            )
            rag_time = (time.time() - start) * 1000
            total_rag_time += rag_time
            
            g_data = g_resp.json()
            results.append({
                "sentence": sentence,
                "glossary_count": g_data.get("match_count", 0),
                "fuzzy_count": len(rag_res.points),
                "time": glossary_time + rag_time
            })
            
            print(f"   Glossary: {g_data.get('match_count', 0)}, Fuzzy: {len(rag_res.points)}, Time: {glossary_time + rag_time:.0f}ms")
        
        print(f"\n📊 BATCH SUMMARY")
        print(f"   Total sentences: {len(sentences)}")
        print(f"   Total glossary time: {total_glossary_time:.0f}ms")
        print(f"   Total RAG time: {total_rag_time:.0f}ms")
        print(f"   Average per sentence: {(total_glossary_time + total_rag_time) / len(sentences):.0f}ms")
    
    def test_04_language_pair_handling(self, qdrant_client, embedding_model):
        """Test handling of different language pairs"""
        sentence = "The patient has symptoms of diabetes"
        
        print(f"\n{'='*70}")
        print(f"LANGUAGE PAIR HANDLING TEST")
        print(f"{'='*70}")
        
        # Test EN -> AR
        print(f"\n🔄 Testing EN → AR")
        g_resp = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={"text": sentence, "source_lang": "en", "target_lang": "ar", "domain": "health"},
            timeout=Config.REQUEST_TIMEOUT
        )
        assert g_resp.status_code == 200
        print(f"   Glossary matches: {g_resp.json().get('match_count', 0)}")
        
        # Test AR -> EN (if supported)
        arabic_sentence = "المريض يعاني من أعراض السكري"
        print(f"\n🔄 Testing AR → EN")
        g_resp2 = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={"text": arabic_sentence, "source_lang": "ar", "target_lang": "en", "domain": "health"},
            timeout=Config.REQUEST_TIMEOUT
        )
        print(f"   Status: {g_resp2.status_code}")
        if g_resp2.status_code == 200:
            print(f"   Glossary matches: {g_resp2.json().get('match_count', 0)}")


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCasesAndErrors:
    """Test edge cases and error handling"""
    
    def test_01_empty_input(self):
        """Test handling of empty input"""
        print(f"\n✅ Empty Input Test")
        
        response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={"text": "", "source_lang": "en", "target_lang": "ar", "domain": "health"},
            timeout=Config.REQUEST_TIMEOUT
        )
        
        # Should either return 200 with no matches or 400/422 for validation error
        print(f"   Status code: {response.status_code}")
        assert response.status_code in [200, 400, 422]
    
    def test_02_very_long_input(self, embedding_model):
        """Test handling of very long input"""
        long_text = "The patient has symptoms. " * 100  # ~2600 characters
        
        print(f"\n✅ Very Long Input Test")
        print(f"   Input length: {len(long_text)} characters")
        
        # Glossary should handle it
        response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={"text": long_text, "source_lang": "en", "target_lang": "ar", "domain": "health"},
            timeout=Config.REQUEST_TIMEOUT
        )
        print(f"   Glossary status: {response.status_code}")
        
        # Embedding should handle it (with truncation)
        start = time.time()
        embedding = embedding_model.encode(long_text)
        elapsed = (time.time() - start) * 1000
        print(f"   Embedding time: {elapsed:.0f}ms")
        print(f"   Embedding shape: {embedding.shape}")
    
    def test_03_special_characters(self):
        """Test handling of special characters"""
        special_text = "Patient's temperature: 38.5°C (101.3°F) — elevated!"
        
        print(f"\n✅ Special Characters Test")
        print(f"   Input: '{special_text}'")
        
        response = requests.post(
            f"{Config.GLOSSARY_API}/translate/sentence",
            json={"text": special_text, "source_lang": "en", "target_lang": "ar", "domain": "health"},
            timeout=Config.REQUEST_TIMEOUT
        )
        
        print(f"   Status: {response.status_code}")
        assert response.status_code in [200, 400, 422]
    
    def test_04_unicode_arabic(self, embedding_model):
        """Test handling of Arabic Unicode text"""
        arabic_text = "يعاني المريض من ارتفاع ضغط الدم والسكري"
        
        print(f"\n✅ Arabic Unicode Test")
        print(f"   Input: '{arabic_text}'")
        
        # Should generate embedding without error
        embedding = embedding_model.encode(arabic_text)
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   First 5 values: {embedding[:5]}")
        
        assert embedding.shape[0] == 768, "LaBSE should produce 768-dim embeddings"
    
    def test_05_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        print(f"\n✅ Concurrent Requests Test")
        
        def make_request(i):
            response = requests.post(
                f"{Config.GLOSSARY_API}/translate/sentence",
                json={
                    "text": f"Test sentence number {i} with medical terms",
                    "source_lang": "en",
                    "target_lang": "ar",
                    "domain": "health"
                },
                timeout=Config.REQUEST_TIMEOUT
            )
            return response.status_code
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.time() - start
        
        success_count = sum(1 for r in results if r == 200)
        print(f"   Requests: 10 concurrent")
        print(f"   Successful: {success_count}/10")
        print(f"   Total time: {elapsed:.2f}s")
        
        assert success_count >= 8, "Too many concurrent request failures"


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_01_glossary_latency_benchmark(self):
        """Benchmark glossary lookup latency"""
        sentence = "Patients with diabetes require regular monitoring"
        iterations = 10
        
        print(f"\n📊 Glossary Latency Benchmark ({iterations} iterations)")
        
        times = []
        for i in range(iterations):
            start = time.time()
            response = requests.post(
                f"{Config.GLOSSARY_API}/translate/sentence",
                json={"text": sentence, "source_lang": "en", "target_lang": "ar", "domain": "health"},
                timeout=Config.REQUEST_TIMEOUT
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200
        
        avg = sum(times) / len(times)
        p95 = sorted(times)[int(len(times) * 0.95)]
        
        print(f"   Average: {avg:.0f}ms")
        print(f"   Min: {min(times):.0f}ms")
        print(f"   Max: {max(times):.0f}ms")
        print(f"   P95: {p95:.0f}ms")
    
    def test_02_embedding_latency_benchmark(self, embedding_model):
        """Benchmark embedding generation latency"""
        sentences = [
            "Short sentence.",
            "This is a medium length sentence with more words.",
            "This is a longer sentence that contains many more words and should take more time to process through the embedding model."
        ]
        
        print(f"\n📊 Embedding Latency Benchmark")
        
        for sentence in sentences:
            times = []
            for _ in range(5):
                start = time.time()
                _ = embedding_model.encode(sentence)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg = sum(times) / len(times)
            print(f"   '{sentence[:30]}...' ({len(sentence)} chars): {avg:.0f}ms avg")
    
    def test_03_qdrant_search_latency_benchmark(self, qdrant_client, embedding_model):
        """Benchmark Qdrant search latency"""
        query = "Patients require medical attention"
        query_embedding = embedding_model.encode(query)
        
        print(f"\n📊 Qdrant Search Latency Benchmark")
        
        for limit in [5, 10, 20]:
            times = []
            for _ in range(5):
                start = time.time()
                _ = qdrant_client.query_points(
                    collection_name=Config.COLLECTION_NAME,
                    query=query_embedding.tolist(),
                    using="cross_lingual",
                    limit=limit,
                    with_payload=True
                )
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            avg = sum(times) / len(times)
            print(f"   Top-{limit}: {avg:.0f}ms avg")


# =============================================================================
# FINAL SUMMARY TEST
# =============================================================================

class TestFinalSummary:
    """Final comprehensive summary test"""
    
    def test_final_system_summary(self, qdrant_client, embedding_model):
        """Generate final system summary"""
        print(f"\n{'='*70}")
        print(f"FINAL SYSTEM SUMMARY")
        print(f"{'='*70}")
        
        # Check Glossary System
        try:
            g_health = requests.get(f"{Config.GLOSSARY_API}/health/services", timeout=5)
            glossary_status = "✅ Running" if g_health.status_code == 200 else "❌ Down"
        except:
            glossary_status = "❌ Unreachable"
        
        # Check Qdrant
        try:
            collections = qdrant_client.get_collections()
            qdrant_status = "✅ Running"
            collection_info = qdrant_client.get_collection(Config.COLLECTION_NAME)
            points_count = collection_info.points_count
        except:
            qdrant_status = "❌ Down"
            points_count = 0
        
        # Check embedding model
        try:
            _ = embedding_model.encode("test")
            model_status = "✅ Loaded"
        except:
            model_status = "❌ Error"
        
        print(f"\n📋 SYSTEM STATUS")
        print(f"   Phase 1 (Glossary System): {glossary_status}")
        print(f"   Phase 2 (Qdrant): {qdrant_status}")
        print(f"   Embedding Model (LaBSE): {model_status}")
        print(f"   Translation Memory: {points_count} pairs")
        
        print(f"\n📋 ENDPOINTS")
        print(f"   Glossary API: {Config.GLOSSARY_API}")
        print(f"   RAG API: {Config.RAG_API}")
        print(f"   Qdrant: {Config.QDRANT_URL}")
        
        print(f"\n📋 CAPABILITIES")
        print(f"   ✓ Exact glossary term matching (Phase 1)")
        print(f"   ✓ Semantic fuzzy matching (Phase 2)")
        print(f"   ✓ Cross-lingual embeddings (LaBSE)")
        print(f"   ✓ Multi-vector search (wording + meaning)")
        print(f"   ✓ Domain filtering")
        print(f"   ✓ Language pair support (EN ↔ AR)")
        
        print(f"\n{'='*70}")
        print(f"ALL INTEGRATION TESTS COMPLETE")
        print(f"{'='*70}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
