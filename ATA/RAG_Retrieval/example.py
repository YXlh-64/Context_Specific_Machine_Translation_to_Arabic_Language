"""
Simple Example: Using RAG Translation Agent with Local Llama

This example demonstrates how to use the RAG Translation Agent
for basic translation tasks using a local Llama model.
"""

from rag_agent import RAGTranslationAgent
from vector_db import VectorDBManager
from utils import setup_logging, load_config

def main():
    # Setup logging
    setup_logging("./logs")
    
    # Load config
    config = load_config("config.ini")
    
    print("🌐 RAG Translation Agent - Simple Example (Local Llama)")
    print("=" * 50)
    
    # Initialize Vector Database Manager
    print("\n1️⃣ Initializing Vector Database...")
    vector_db = VectorDBManager(
        db_type=config.get("VECTOR_DB", "db_type", fallback="faiss"),
        embedding_model=config.get("EMBEDDINGS", "model_name"),
        db_path=config.get("GENERAL", "vector_db_dir", fallback="./vector_db"),
        collection_name=config.get("VECTOR_DB", "collection_name", fallback="translation_corpus"),
        device=config.get("EMBEDDINGS", "device", fallback="cuda")
    )
    
    # Show database statistics
    stats = vector_db.get_stats()
    print(f"   ✓ Database loaded with {stats['total_entries']} entries")
    
    # Initialize RAG Translation Agent with Local Llama
    print("\n2️⃣ Initializing Translation Agent (Local Llama)...")
    print("   (Loading 4-bit quantized model, this may take a minute...)")
    agent = RAGTranslationAgent(
        model_name=config.get("TRANSLATION", "model_name"),
        vector_db_manager=vector_db,
        device=config.get("TRANSLATION", "device", fallback="cuda"),
        max_length=config.getint("TRANSLATION", "max_length", fallback=256),
        temperature=config.getfloat("TRANSLATION", "temperature", fallback=0.3),
        top_p=config.getfloat("TRANSLATION", "top_p", fallback=0.9),
        top_k_retrieval=5,
        use_4bit=config.getboolean("TRANSLATION", "use_4bit", fallback=True)
    )
    print("   ✓ Agent initialized")
    
    # Show model info
    model_info = agent.get_model_info()
    print(f"   📊 Model: {model_info['model_name']}")
    print(f"   📊 VRAM: {model_info['vram_allocated_gb']:.2f} GB")
    
    # Example translations
    print("\n3️⃣ Translating Examples...")
    print("=" * 50)
    
    examples = [
        "The economic growth rate increased by 3.5% this quarter.",
        "The central bank announced new monetary policies.",
        "Foreign direct investment reached record levels."
    ]
    
    for i, source_text in enumerate(examples, 1):
        print(f"\n📝 Example {i}:")
        print(f"   EN: {source_text}")
        
        # Translate with context
        result = agent.translate(
            source_text=source_text,
            use_context=True,
            return_context=True
        )
        
        print(f"   AR: {result['translation']}")
        print(f"   📊 Retrieved {result.get('num_retrieved', 0)} similar examples")
        
        # Show top retrieved example
        if result.get('retrieved_examples'):
            top_example = result['retrieved_examples'][0]
            print(f"\n   🔍 Most similar example (similarity: {top_example['similarity']:.3f}):")
            print(f"      EN: {top_example['en'][:80]}...")
            print(f"      AR: {top_example['ar'][:80]}...")
    
    # Compare with and without context
    print("\n" + "=" * 50)
    print("4️⃣ Comparing With/Without Context")
    print("=" * 50)
    
    test_text = "The GDP growth accelerated to 4.2% annually."
    
    print(f"\n📝 Test text: {test_text}\n")
    
    # Without context
    print("🚫 Without RAG context:")
    result_without = agent.translate(test_text, use_context=False)
    print(f"   {result_without['translation']}")
    
    # With context
    print("\n✅ With RAG context:")
    result_with = agent.translate(test_text, use_context=True, return_context=True)
    print(f"   {result_with['translation']}")
    
    print(f"\n   📊 Used {result_with.get('num_retrieved', 0)} examples for context")
    
    # Batch translation
    print("\n" + "=" * 50)
    print("5️⃣ Batch Translation")
    print("=" * 50)
    
    batch_texts = [
        "Exports increased significantly.",
        "The unemployment rate decreased.",
        "Industrial production expanded."
    ]
    
    print(f"\n📦 Translating {len(batch_texts)} texts...\n")
    results = agent.translate_batch(batch_texts, domain="economic", show_progress=True)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. EN: {result['source']}")
        print(f"   AR: {result['translation']}")
    
    print("\n" + "=" * 50)
    print("✅ Example completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("  - Try with your own text")
    print("  - Adjust top_k parameter for more/less context")
    print("  - Use domain filtering for better results")
    print("  - Launch the web UI: streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
