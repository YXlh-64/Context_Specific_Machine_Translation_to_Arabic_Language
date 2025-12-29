"""
REAL Integration Tests - Uses actual Qdrant, embeddings, and Glossary System
Requires: Qdrant running on 6333, Glossary System on 8001, Redis on 6379
"""

import pytest
import requests
import time

# Configuration
GLOSSARY_API = "http://127.0.0.1:8001/api/v1"
RAG_API = "http://127.0.0.1:8002/api/v1"
QDRANT_URL = "http://localhost:6333"

# Test data
TEST_QUERIES = [
    {
        "query": "Patients with severe symptoms require immediate medical attention",
        "domain": "health",
        "source_lang": "en",
        "target_lang": "ar"
    },
    {
        "query": "The clinical trial showed promising results for the new treatment",
        "domain": "health", 
        "source_lang": "en",
        "target_lang": "ar"
    }
]


class TestRealQdrantConnection:
    """Test real Qdrant database connection"""
    
    def test_qdrant_is_running(self):
        """Verify Qdrant is accessible"""
        response = requests.get(f"{QDRANT_URL}/collections")
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        print(f"✅ Qdrant connected - Collections: {[c['name'] for c in data['result']['collections']]}")
    
    def test_collection_has_data(self):
        """Verify translation_memory collection has data"""
        response = requests.get(f"{QDRANT_URL}/collections/translation_memory")
        assert response.status_code == 200
        data = response.json()
        points_count = data["result"]["points_count"]
        assert points_count > 0, "Collection is empty!"
        print(f"✅ Collection has {points_count} translation pairs")


class TestRealGlossarySystem:
    """Test real Glossary System (Phase 1) integration"""
    
    def test_glossary_system_running(self):
        """Verify Glossary System API is accessible"""
        response = requests.get(f"{GLOSSARY_API}/health/services")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Glossary System status: {data['status']}")
    
    def test_glossary_lookup(self):
        """Test real glossary term lookup"""
        response = requests.post(
            f"{GLOSSARY_API}/translate/sentence",
            json={
                "text": "The patient has symptoms of diabetes",
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            }
        )
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Glossary matches found: {data.get('match_count', 0)}")
        if data.get('glossary_matches'):
            for match in data['glossary_matches'][:3]:
                print(f"   • {match['source_term']} → {match['target_term']}")


class TestRealRAGSystem:
    """Test real RAG System (Phase 2) with Qdrant"""
    
    def test_rag_health(self):
        """Verify RAG System is running (optional - may not be deployed)"""
        try:
            response = requests.get(f"{RAG_API}/health", timeout=3)
            assert response.status_code == 200
            data = response.json()
            assert data["qdrant"]["connected"] == True
            print(f"✅ RAG System healthy - Qdrant connected: {data['qdrant']['connected']}")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pytest.skip("RAG System server not running on port 8002 - testing library code only")
    
    def test_semantic_search_real(self):
        """Test real semantic search against Qdrant"""
        for test in TEST_QUERIES:
            start = time.time()
            response = requests.post(
                f"{RAG_API}/search/semantic",
                json={
                    "query": test["query"],
                    "domain": test["domain"],
                    "source_lang": test["source_lang"],
                    "target_lang": test["target_lang"],
                    "top_k": 5
                }
            )
            elapsed = (time.time() - start) * 1000
            
            assert response.status_code == 200, f"Failed: {response.text}"
            data = response.json()
            
            print(f"\n🔍 Query: '{test['query'][:50]}...'")
            print(f"   Found: {data['total_results']} results in {elapsed:.0f}ms")
            
            if data["results"]:
                top = data["results"][0]
                print(f"   Top match ({top['similarity_percentage']:.1f}%):")
                print(f"   Source: {top['source'][:60]}...")
                print(f"   Target: {top['target'][:60]}...")
    
    def test_hybrid_search_real(self):
        """Test real hybrid search"""
        response = requests.post(
            f"{RAG_API}/search/hybrid",
            json={
                "query": TEST_QUERIES[0]["query"],
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar",
                "top_k": 7,
                "semantic_weight": 0.6,
                "wording_weight": 0.4,
                "enable_diversity": True
            }
        )
        
        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        print(f"\n✅ Hybrid search returned {data['total_results']} diverse results")
        
        # Print detailed results
        if data.get("results"):
            print(f"\n📊 Top Results:")
            for i, result in enumerate(data["results"][:5], 1):
                print(f"\n   {i}. Score: {result.get('similarity_percentage', 0):.1f}%")
                print(f"      Source: {result.get('source', 'N/A')[:80]}...")
                print(f"      Target: {result.get('target', 'N/A')[:80]}...")
                print(f"      Domain: {result.get('domain', 'N/A')}")


class TestRealEndToEndIntegration:
    """Test complete Phase 1 + Phase 2 pipeline"""
    
    def test_full_pipeline(self):
        """Test complete translation assistance pipeline"""
        test_sentence = "Patients with diabetes require regular blood sugar monitoring"
        
        # Step 1: Call Glossary System (Phase 1)
        print("\n📘 STEP 1: Glossary Lookup (Phase 1)")
        glossary_response = requests.post(
            f"{GLOSSARY_API}/translate/sentence",
            json={
                "text": test_sentence,
                "source_lang": "en",
                "target_lang": "ar",
                "domain": "health"
            }
        )
        assert glossary_response.status_code == 200
        glossary_data = glossary_response.json()
        print(f"   Glossary matches: {glossary_data.get('match_count', 0)}")
        
        # Step 2: Call RAG System (Phase 2) integration endpoint
        print("\n📗 STEP 2: Semantic Search (Phase 2)")
        
        # Convert glossary matches to expected format
        glossary_matches = []
        for match in glossary_data.get("glossary_matches", []):
            glossary_matches.append({
                "source_term": match.get("source_term", ""),
                "target_term": match.get("target_term", ""),
                "domain": match.get("domain", "health"),
                "n_gram_size": match.get("n_gram_size", 1),
                "frequency": match.get("frequency", 1)
            })
        
        rag_response = requests.post(
            f"{RAG_API}/integrate",
            json={
                "source_sentence": test_sentence,
                "glossary_matches": glossary_matches,
                "domain": "health",
                "source_lang": "en",
                "target_lang": "ar"
            }
        )
        assert rag_response.status_code == 200, f"Failed: {rag_response.text}"
        rag_data = rag_response.json()
        
        print(f"   Fuzzy matches: {rag_data.get('fuzzy_count', 0)}")
        
        # Step 3: Verify combined output
        print("\n📙 STEP 3: Combined Output")
        print(f"   Source: {rag_data['source_sentence']}")
        print(f"   Glossary terms: {rag_data.get('glossary_count', 0)}")
        print(f"   Fuzzy matches: {rag_data.get('fuzzy_count', 0)}")
        
        if rag_data.get('fuzzy_matches'):
            print("\n   Top fuzzy matches:")
            for i, match in enumerate(rag_data['fuzzy_matches'][:3], 1):
                print(f"   {i}. [{match['similarity_percentage']:.1f}%] {match['source'][:50]}...")
        
        # Assertions
        assert "source_sentence" in rag_data
        assert "glossary_matches" in rag_data
        assert "fuzzy_matches" in rag_data
        
        print("\n✅ Full pipeline test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])