#!/bin/bash
# Setup script for RAG Translation Agent (Bash)
# Usage: ./setup.sh

echo "🌐 RAG Translation Agent Setup"
echo "================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists"
else
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p outputs
mkdir -p vector_db
mkdir -p data
echo "   ✓ Directories created"

# Check configuration
echo ""
echo "⚙️  Checking configuration..."
if [ -f "config.ini" ]; then
    echo "   ✓ config.ini found"
else
    echo "   ⚠️  config.ini not found - using defaults"
fi

# Check for corpus file
echo ""
echo "📄 Checking for corpus file..."
corpus_file=$(grep "corpus_file" config.ini | cut -d "=" -f 2 | xargs)
if [ -f "$corpus_file" ]; then
    echo "   ✓ Corpus file found: $corpus_file"
else
    echo "   ⚠️  Corpus file not found: $corpus_file"
    echo "   Please update config.ini with correct path"
fi

echo ""
echo "================================"
echo "✅ Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Build vector database:"
echo "   python build_vector_db.py"
echo ""
echo "3. Launch demo app:"
echo "   streamlit run app.py"
echo ""
echo "For more information, see README.md"
echo ""
