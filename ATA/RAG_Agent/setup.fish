#!/usr/bin/env fish
# Setup script for Local Llama RAG Translation System
# Run this after cloning the repository

echo "🚀 Setting up Local Llama RAG Translation System"
echo "================================================"

# Check if we're in the right directory
if not test -f "config.ini"
    echo "❌ Error: Please run this script from the RAG_Agent directory"
    exit 1
end

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p vector_db
mkdir -p evaluation_results
mkdir -p outputs
echo "✅ Directories created"

# Check Python version
echo ""
echo "🐍 Checking Python version..."
python3 --version
if test $status -ne 0
    echo "❌ Python 3 not found. Please install Python 3.8 or higher"
    exit 1
end

# Check for CUDA
echo ""
echo "🎮 Checking for CUDA..."
if command -v nvidia-smi >/dev/null 2>&1
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ CUDA detected"
else
    echo "⚠️  Warning: CUDA not detected. Install CUDA for GPU acceleration"
    echo "   CPU mode will be very slow for Llama models"
end

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
echo "   This will install PyTorch, Transformers, and other packages"
read -P "Continue? (y/n): " confirm

if test "$confirm" = "y"
    pip install -r requirements.txt
    if test $status -eq 0
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Installation failed. Check error messages above"
        exit 1
    end
else
    echo "⏭️  Skipping installation"
end

# Check if vector database exists
echo ""
if test -d "vector_db/translation_corpus"
    echo "✅ Vector database found"
else
    echo "⚠️  Vector database not found"
    echo ""
    echo "To build the vector database, run:"
    echo "  python build_vector_db.py --sample 10000"
    echo ""
end

# Check if test data exists
echo ""
if test -f "test_samples.csv"
    echo "✅ Test dataset found"
else
    echo "⚠️  Test dataset not found"
    echo ""
    echo "To create a test dataset, run:"
    echo "  python extract_test_samples.py --num_samples 100"
    echo ""
end

echo ""
echo "================================================"
echo "✅ Setup complete!"
echo "================================================"
echo ""
echo "🎯 Next steps:"
echo ""
echo "1. Build vector database (if not done):"
echo "   python build_vector_db.py --sample 10000"
echo ""
echo "2. Run quick test:"
echo "   python test_gemini.py"
echo ""
echo "3. Extract test samples:"
echo "   python extract_test_samples.py --num_samples 100"
echo ""
echo "4. Compare RAG vs Non-RAG:"
echo "   python compare_rag.py --test_file test_samples.csv --sample 50"
echo ""
echo "5. Launch Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "📖 Read QUICKSTART_LOCAL_LLAMA.md for detailed instructions"
echo ""
