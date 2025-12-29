"""
Add DIABETES test sentences to Qdrant for RRF Ranking Test
"""

import sys
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "translation_memory"

# Query to test later: "Treatment options for Type 2 Diabetes"

TEST_SENTENCES = [
    {
        # RANK 1 TARGET: Perfect Hybrid Match
        # Contains exact keywords: "Type 2 diabetes", "treatment", "medications"
        "source": "Common treatment options for Type 2 diabetes include Metformin, lifestyle changes, and insulin therapy.",
        "target": "تشمل خيارات علاج مرض السكري من النوع 2 الميتفورمين، وتغيير نمط الحياة، والعلاج بالأنسولين.",
        "domain": "health",
        "source_lang": "en",
        "target_lang": "ar"
    },
    {
        # RANK 2 TARGET: High Semantic, Specific Medical Term
        # Contains "Type 2", but uses more technical language ("GLP-1")
        "source": "GLP-1 receptor agonists are effective injectable medications for managing Type 2 diabetes.",
        "target": "منبهات مستقبلات GLP-1 هي أدوية فعالة عن طريق الحقن لإدارة مرض السكري من النوع 2.",
        "domain": "health",
        "source_lang": "en",
        "target_lang": "ar"
    },
    {
        # RANK 3 TARGET: Pure Semantic Match (The "RRF Test")
        # DOES NOT say "Type 2" or "Diabetes" explicitly, but talks about "Adult-onset" and "blood sugar".
        # A keyword search might miss this, but Semantic search should find it.
        "source": "Managing adult-onset high blood sugar often requires weight loss and oral antihyperglycemic drugs.",
        "target": "غالبًا ما تتطلب إدارة ارتفاع نسبة السكر في الدم عند البالغين فقدان الوزن وتناول أدوية خفض السكر عن طريق الفم.",
        "domain": "health",
        "source_lang": "en",
        "target_lang": "ar"
    },
    {
        # RANK 4 TARGET: The Distractor (Type 1)
        # Contains keywords "Diabetes" and "Insulin", but is logically WRONG for the query.
        # Simple keyword search might rank this high. Hybrid should push it down.
        "source": "Type 1 diabetes is an autoimmune condition where the pancreas produces little to no insulin.",
        "target": "مرض السكري من النوع 1 هو حالة من أمراض المناعة الذاتية حيث ينتج البنكرياس كمية قليلة جدًا من الأنسولين أو لا ينتجه على الإطلاق.",
        "domain": "health",
        "source_lang": "en",
        "target_lang": "ar"
    }
]


def add_sentences_to_qdrant():
    """Add test sentences to Qdrant vector database"""
    
    print("🔄 Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print("🔗 Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Verify collection exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ Collection '{COLLECTION_NAME}' not found!")
        return
    
    print(f"✅ Connected to collection: {COLLECTION_NAME}")
    
    points = []
    print("\n📝 Preparing Data:")
    
    for i, sentence in enumerate(TEST_SENTENCES):
        # Generate embeddings
        source_embedding = model.encode(sentence["source"])
        target_embedding = model.encode(sentence["target"])
        
        # Cross-lingual embedding (average)
        cross_lingual_embedding = (source_embedding + target_embedding) / 2
        
        # Create point with UUID
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,
            vector={
                "source_semantic": source_embedding.tolist(),
                "cross_lingual": cross_lingual_embedding.tolist()
            },
            payload={
                "source": sentence["source"],
                "target": sentence["target"],
                "domain": sentence["domain"],
                "source_lang": sentence["source_lang"],
                "target_lang": sentence["target_lang"],
                "language_pair": f"{sentence['source_lang']}-{sentence['target_lang']}",
                "source_length": len(sentence["source"]),
                "target_length": len(sentence["target"]),
                "test_category": "diabetes_rrf_test" # Tag for easy deletion later
            }
        )
        points.append(point)
        print(f"   [{i+1}] {sentence['source'][:50]}...")

    # Upload
    print("\n🚀 Uploading to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    print(f"✅ Successfully added {len(points)} diabetes test sentences.")
    print("   Run your hybrid search now with query: 'Treatment options for Type 2 Diabetes'")

if __name__ == "__main__":
    add_sentences_to_qdrant()