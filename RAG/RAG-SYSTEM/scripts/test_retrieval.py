"""
Direct test to verify DIABETES sentences are retrievable
"""

import sys
import os
# Adjust path to find your app modules if needed, or keep standard imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "translation_memory"

# UPDATED QUERY for the diabetes test case
QUERY = "Treatment options for Type 2 Diabetes"

def test_retrieval():
    print("🔄 Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("🔗 Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Generate query embedding
    print(f"\n🔍 Query: '{QUERY}'")
    query_embedding = model.encode(QUERY)
    
    # ---------------------------------------------------------
    # 1. Test Semantic/Cross-Lingual Retrieval
    # ---------------------------------------------------------
    print("\n📊 1. Searching 'cross_lingual' (Semantic View)...")
    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="cross_lingual",
            limit=10,
            with_payload=True
        )
        
        print(f"✅ Found {len(results.points)} results:\n")
        for i, point in enumerate(results.points, 1):
            score = point.score
            source = point.payload.get("source", "N/A")
            target = point.payload.get("target", "N/A")
            
            # Check for our specific tag "diabetes_rrf_test"
            test_tag = point.payload.get("test_category", "")
            is_new = test_tag == "diabetes_rrf_test"
            
            marker = "🆕 [TEST DATA]" if is_new else "   [EXISTING] "
            print(f"{i}. {marker} Score: {score:.4f}")
            print(f"       Source: {source[:80]}...")
            print(f"       Target: {target[:80]}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")

    # ---------------------------------------------------------
    # 2. Test Wording/Source Retrieval
    # ---------------------------------------------------------
    print("\n📊 2. Searching 'source_semantic' (Wording View)...")
    try:
        results2 = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="source_semantic",
            limit=10,
            with_payload=True
        )
        
        print(f"✅ Found {len(results2.points)} results:\n")
        for i, point in enumerate(results2.points, 1):
            score = point.score
            source = point.payload.get("source", "N/A")
            
            # Check for our specific tag
            test_tag = point.payload.get("test_category", "")
            is_new = test_tag == "diabetes_rrf_test"
            
            marker = "🆕 [TEST DATA]" if is_new else "   [EXISTING] "
            print(f"{i}. {marker} Score: {score:.4f} - {source[:80]}...")
            
    except Exception as e:
        print(f"❌ Wording search failed: {e}")

if __name__ == "__main__":
    test_retrieval()