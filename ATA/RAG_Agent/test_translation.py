"""
Quick test for Local Llama RAG Translation

Tests basic functionality:
1. Vector DB loading
2. Local Llama model loading
3. Translation with context
4. Translation without context
"""

import os
from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import load_config, setup_logging

def main():
    print("🔬 Testing Local Llama RAG Translation")
    print("="*60)
    
    # Load config
    config = load_config("config.ini")
    
    # Initialize Vector Database
    print("\n📚 Loading Vector Database...")
    vector_db = VectorDBManager(
        db_type=config.get("VECTOR_DB", "db_type", fallback="faiss"),
        embedding_model=config.get("EMBEDDINGS", "model_name"),
        db_path=config.get("GENERAL", "vector_db_dir", fallback="./vector_db"),
        collection_name=config.get("VECTOR_DB", "collection_name", fallback="translation_corpus"),
        device=config.get("EMBEDDINGS", "device", fallback="cuda")
    )
    
    stats = vector_db.get_stats()
    print(f"   ✓ Loaded {stats['total_entries']} translation examples")
    
    # Initialize Local Llama Agent
    print("\n🤖 Initializing Local Llama Translation Agent...")
    print("   (This may take a minute to load and quantize the model...)")
    agent = RAGTranslationAgent(
        model_name=config.get("TRANSLATION", "model_name"),
        vector_db_manager=vector_db,
        device=config.get("TRANSLATION", "device", fallback="cuda"),
        max_length=config.getint("TRANSLATION", "max_length", fallback=256),
        temperature=config.getfloat("TRANSLATION", "temperature", fallback=0.3),
        top_p=config.getfloat("TRANSLATION", "top_p", fallback=0.9),
        top_k_retrieval=3,
        use_4bit=config.getboolean("TRANSLATION", "use_4bit", fallback=True)
    )
    print("   ✓ Local Llama agent ready")
    
    # Display model info
    model_info = agent.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   Model: {model_info['model_name']}")
    print(f"   Device: {model_info['device']}")
    print(f"   VRAM Allocated: {model_info['vram_allocated_gb']:.2f} GB")
    print(f"   VRAM Reserved: {model_info['vram_reserved_gb']:.2f} GB")
    print(f"   Parameters: {model_info['parameters']/1e9:.2f}B")
    
    # Test sentences (economic domain)
    test_sentences = [
        "The economic growth rate increased significantly.",
        "Inflation remains a major concern for policymakers.",
        "The central bank announced new monetary policies."
    ]
    
    print("\n" + "="*60)
    print("🧪 TEST 1: Translation WITH RAG Context")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Example {i} ---")
        print(f"Source: {sentence}")
        
        try:
            result = agent.translate(
                sentence,
                use_context=True,
                return_context=True
            )
            
            print(f"Translation: {result['translation']}")
            print(f"Retrieved: {result['num_retrieved']} similar examples")
            
            # Show retrieved examples
            if 'retrieved_examples' in result:
                print("\nRetrieved Context:")
                for j, ex in enumerate(result['retrieved_examples'], 1):
                    print(f"  {j}. EN: {ex['en'][:60]}...")
                    print(f"     AR: {ex['ar'][:60]}...")
                    print(f"     Similarity: {ex['similarity']:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("🧪 TEST 2: Translation WITHOUT Context (Baseline)")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n--- Example {i} ---")
        print(f"Source: {sentence}")
        
        try:
            result = agent.translate(
                sentence,
                use_context=False
            )
            
            print(f"Translation: {result['translation']}")
            print(f"Context used: {result['used_context']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ Test Complete!")
    print("="*60)
    print("\n💡 Next Steps:")
    print("   1. Compare translations with/without context above")
    print("   2. Run full evaluation: python compare_rag.py --test_file test_data.csv")
    print("   3. Check README.md for detailed usage")


if __name__ == "__main__":
    main()
