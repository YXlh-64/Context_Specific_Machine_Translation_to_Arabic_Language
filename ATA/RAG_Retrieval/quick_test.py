"""
Quick Test Script for RAG Translation Agent

Tests basic functionality with minimal setup using local Llama model.
"""

from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import setup_logging, load_config

def main():
    # Setup logging
    setup_logging("./logs")
    
    # Load config
    config = load_config("config.ini")
    
    print("🚀 Quick Test - RAG Translation Agent (Local Llama)")
    print("=" * 60)
    
    # Initialize Vector Database Manager
    print("\n1️⃣ Loading Vector Database...")
    vector_db = VectorDBManager(
        db_type=config.get("VECTOR_DB", "db_type", fallback="faiss"),
        embedding_model=config.get("EMBEDDINGS", "model_name"),
        db_path=config.get("GENERAL", "vector_db_dir", fallback="./vector_db"),
        collection_name=config.get("VECTOR_DB", "collection_name", fallback="translation_corpus"),
        device=config.get("EMBEDDINGS", "device", fallback="cuda")
    )
    
    stats = vector_db.get_stats()
    print(f"   ✓ Loaded {stats['total_entries']} translation examples")
    
    # Initialize Translation Agent
    print("\n2️⃣ Loading Local Llama Model (4-bit quantized)...")
    print("   (This may take a minute...)")
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
    print("   ✓ Model loaded successfully")
    
    # Show model info
    model_info = agent.get_model_info()
    print(f"\n   📊 Model: {model_info['model_name']}")
    print(f"   📊 VRAM: {model_info['vram_allocated_gb']:.2f} GB")
    
    # Test translation
    print("\n3️⃣ Testing Translation...")
    print("=" * 60)
    
    test_text = "The economic growth rate increased significantly this year."
    print(f"\n📝 Input (English):")
    print(f"   {test_text}")
    
    # Translate WITH context (RAG-enabled)
    print("\n🔍 Translating with RAG context...")
    result_with_rag = agent.translate(
        test_text,
        use_context=True,
        return_context=True
    )
    
    print(f"\n✨ Translation (with RAG):")
    print(f"   {result_with_rag['translation']}")
    
    if 'retrieved_examples' in result_with_rag:
        print(f"\n📚 Retrieved Context Examples ({len(result_with_rag['retrieved_examples'])}):")
        for i, ctx in enumerate(result_with_rag['retrieved_examples'][:2], 1):  # Show first 2
            print(f"\n   Example {i} (similarity: {ctx['similarity']:.3f}):")
            print(f"   EN: {ctx['en'][:80]}...")
            print(f"   AR: {ctx['ar'][:80]}...")
    
    # Translate WITHOUT context (baseline)
    print("\n\n🔄 Translating without RAG context (baseline)...")
    result_without_rag = agent.translate(
        test_text,
        use_context=False
    )
    
    print(f"\n✨ Translation (baseline - no RAG):")
    print(f"   {result_without_rag['translation']}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
