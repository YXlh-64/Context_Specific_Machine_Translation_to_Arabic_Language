"""
GPU Diagnostic Script for RAG Translation Agent

This script checks your CUDA/GPU setup and provides recommendations.

Usage:
    python check_gpu.py
"""

import sys

print("🔍 Checking GPU/CUDA Setup for RAG Translation Agent")
print("=" * 60)

# Check Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Check CUDA/PyTorch
print("\n📦 Checking PyTorch and CUDA...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # Test GPU
            try:
                x = torch.randn(100, 100).cuda(i)
                y = x @ x
                print(f"    ✓ GPU {i} is working!")
                del x, y
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ✗ GPU {i} test failed: {e}")
    else:
        print("⚠️  CUDA is not available. You'll be using CPU.")
        print("   To use GPU, make sure you have:")
        print("   1. NVIDIA GPU installed")
        print("   2. CUDA toolkit installed")
        print("   3. PyTorch with CUDA support installed")
        
except ImportError:
    print("✗ PyTorch not installed")
    print("  Install with: pip install torch")
    sys.exit(1)

# Check Transformers
print("\n📦 Checking Transformers...")
try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError:
    print("✗ Transformers not installed")
    print("  Install with: pip install transformers")

# Check Sentence Transformers
print("\n📦 Checking Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    print(f"✓ Sentence Transformers installed")
    
    # Test loading a small model
    print("  Testing model loading...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device=device)
        print(f"  ✓ Test model loaded on {device}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        
except ImportError:
    print("✗ Sentence Transformers not installed")
    print("  Install with: pip install sentence-transformers")

# Check FAISS
print("\n📦 Checking FAISS...")
try:
    import faiss
    print(f"✓ FAISS installed")
    
    # Check GPU support
    try:
        gpu_res = faiss.StandardGpuResources()
        print("✓ FAISS GPU support available")
        
        # Test GPU index
        try:
            d = 128
            index = faiss.IndexFlatIP(d)
            index_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            print("✓ FAISS GPU test successful")
            del index, index_gpu
        except Exception as e:
            print(f"⚠️  FAISS GPU test failed: {e}")
            
    except Exception:
        print("⚠️  FAISS GPU support not available (CPU version installed)")
        print("   For GPU acceleration, install: pip install faiss-gpu")
        
except ImportError:
    print("✗ FAISS not installed")
    print("  Install with: pip install faiss-gpu (for GPU) or faiss-cpu")

# Check other dependencies
print("\n📦 Checking other dependencies...")
packages = [
    ('chromadb', 'ChromaDB'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('streamlit', 'Streamlit'),
    ('sacrebleu', 'SacreBLEU'),
    ('tqdm', 'tqdm'),
    ('loguru', 'Loguru'),
]

for module, name in packages:
    try:
        __import__(module)
        print(f"✓ {name} installed")
    except ImportError:
        print(f"✗ {name} not installed")

# Check bitsandbytes for quantization
print("\n📦 Checking quantization support...")
try:
    import bitsandbytes
    print(f"✓ bitsandbytes installed (enables 4-bit/8-bit quantization)")
except ImportError:
    print("⚠️  bitsandbytes not installed")
    print("   For memory-efficient models, install: pip install bitsandbytes")

# Memory recommendations
print("\n" + "=" * 60)
print("💡 Recommendations")
print("=" * 60)

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n✓ GPU detected with {total_memory:.1f}GB memory")
    
    if total_memory >= 24:
        print("\n🚀 Excellent! You can run:")
        print("  - Large models without quantization")
        print("  - Multiple models simultaneously")
        print("  - Large batch sizes")
        config = "High Performance"
    elif total_memory >= 16:
        print("\n✓ Good! Recommended settings:")
        print("  - Use 8-bit quantization for models >7B")
        print("  - Batch size: 16-32")
        print("  - Top-K retrieval: 5-7")
        config = "Balanced"
    elif total_memory >= 8:
        print("\n⚠️  Limited memory. Recommended settings:")
        print("  - Use 4-bit quantization")
        print("  - Batch size: 8-16")
        print("  - Top-K retrieval: 3-5")
        config = "Memory Efficient"
    else:
        print("\n⚠️  Very limited memory. Consider:")
        print("  - Using smaller models")
        print("  - CPU inference for embeddings")
        print("  - Batch size: 4-8")
        config = "Low Memory"
    
    print(f"\n📝 Suggested config.ini settings ({config}):")
    print("\n[EMBEDDINGS]")
    print("device = cuda")
    if total_memory < 16:
        print("batch_size = 16")
    else:
        print("batch_size = 32")
    
    print("\n[VECTOR_DB]")
    print("db_type = faiss  # GPU-accelerated")
    if total_memory < 12:
        print("top_k = 3")
    else:
        print("top_k = 5")
    
    print("\n[TRANSLATION]")
    print("device = cuda")
    if total_memory < 12:
        print("use_4bit = true")
        print("max_length = 256")
    elif total_memory < 16:
        print("use_8bit = true")
        print("max_length = 512")
    else:
        print("use_8bit = false")
        print("max_length = 512")
    
else:
    print("\n⚠️  No GPU detected. CPU settings recommended:")
    print("\n[EMBEDDINGS]")
    print("device = cpu")
    print("batch_size = 16")
    
    print("\n[VECTOR_DB]")
    print("db_type = chromadb  # or faiss-cpu")
    print("top_k = 3")
    
    print("\n[TRANSLATION]")
    print("device = cpu")
    print("# Use smaller models for CPU")
    print("model_name = Helsinki-NLP/opus-mt-en-ar")

print("\n" + "=" * 60)
print("✅ Diagnostic complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Update config.ini with recommended settings")
print("2. Run: python build_vector_db.py")
print("3. Test with: python example.py")
print("\n")
