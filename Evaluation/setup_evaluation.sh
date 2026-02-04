#!/bin/bash
# Quick Setup Script for Semantic Evaluation
# This script helps you set up and test the semantic evaluation system

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Semantic Evaluation System - Quick Setup                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "evaluate_semantic.py" ]; then
    print_error "Please run this script from the Evaluation directory"
    exit 1
fi

print_info "Starting setup process..."
echo ""

# Step 1: Check Python version
echo "═══════════════════════════════════════════════════════════════"
echo "Step 1: Checking Python version..."
echo "═══════════════════════════════════════════════════════════════"

python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python version is compatible (3.8+)"
else
    print_error "Python 3.8 or higher is required"
    exit 1
fi
echo ""

# Step 2: Create virtual environment (optional)
echo "═══════════════════════════════════════════════════════════════"
echo "Step 2: Virtual Environment (Optional)"
echo "═══════════════════════════════════════════════════════════════"

read -p "Do you want to create a virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
else
    print_warning "Skipping virtual environment creation"
fi
echo ""

# Step 3: Choose installation type
echo "═══════════════════════════════════════════════════════════════"
echo "Step 3: Choose Installation Type"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1) Full Installation (RECOMMENDED)"
echo "   - All semantic metrics (COMET, BERTScore, Embeddings)"
echo "   - Best accuracy and quality assessment"
echo "   - ~2GB downloads on first use"
echo ""
echo "2) Partial Installation"
echo "   - Traditional metrics + BERTScore"
echo "   - Good balance of speed and accuracy"
echo "   - ~500MB downloads"
echo ""
echo "3) Minimal Installation"
echo "   - Traditional metrics only (BLEU, CHRF)"
echo "   - Fastest, minimal downloads"
echo "   - Less accurate"
echo ""

read -p "Enter choice [1/2/3]: " install_choice

case $install_choice in
    1)
        print_info "Installing full dependencies..."
        pip install -r requirements.txt
        print_success "Full installation complete"
        ;;
    2)
        print_info "Installing partial dependencies..."
        pip install pandas requests sacrebleu bert-score torch transformers
        print_success "Partial installation complete"
        ;;
    3)
        print_info "Installing minimal dependencies..."
        pip install pandas requests sacrebleu
        print_success "Minimal installation complete"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac
echo ""

# Step 4: Check API connection
echo "═══════════════════════════════════════════════════════════════"
echo "Step 4: Checking Translation API"
echo "═══════════════════════════════════════════════════════════════"

api_url="http://localhost:5002"
print_info "Checking API at $api_url/api/health..."

if curl -s --max-time 5 "$api_url/api/health" > /dev/null 2>&1; then
    print_success "Translation API is running"
    api_running=true
else
    print_warning "Translation API is not running"
    print_info "The API needs to be running to perform evaluation"
    print_info "Start it with: cd ../app && ./run_backend.sh"
    api_running=false
fi
echo ""

# Step 5: Verify installation
echo "═══════════════════════════════════════════════════════════════"
echo "Step 5: Verifying Installation"
echo "═══════════════════════════════════════════════════════════════"

# Check which libraries are available
print_info "Checking installed packages..."

check_package() {
    if python3 -c "import $1" 2>/dev/null; then
        print_success "$2 is available"
        return 0
    else
        print_warning "$2 is not available"
        return 1
    fi
}

check_package "pandas" "Pandas"
check_package "sacrebleu" "SacreBleu"
check_package "bert_score" "BERTScore"
check_package "comet" "COMET"
check_package "sentence_transformers" "Sentence Transformers"

echo ""

# Step 6: Test evaluation (if API is running)
if [ "$api_running" = true ]; then
    echo "═══════════════════════════════════════════════════════════════"
    echo "Step 6: Test Run (Optional)"
    echo "═══════════════════════════════════════════════════════════════"
    
    read -p "Do you want to run a test evaluation (2 samples)? [y/N]: " run_test
    if [[ $run_test =~ ^[Yy]$ ]]; then
        print_info "Running test evaluation..."
        echo ""
        
        case $install_choice in
            1)
                python3 evaluate_semantic.py --sample-size 2 --english-only
                ;;
            2)
                python3 evaluate_enhanced.py --sample-size 2 --english-only --use-bertscore
                ;;
            3)
                python3 evaluate_translation.py --sample-size 2 --english-only
                ;;
        esac
        
        echo ""
        print_success "Test evaluation completed"
    fi
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

print_info "Next Steps:"
echo ""

if [ "$api_running" = false ]; then
    echo "1. Start the translation API:"
    echo "   cd ../app && ./run_backend.sh"
    echo ""
fi

echo "2. Run evaluation:"
case $install_choice in
    1)
        echo "   # Full semantic evaluation (recommended)"
        echo "   python evaluate_semantic.py --sample-size 50"
        ;;
    2)
        echo "   # Enhanced evaluation with BERTScore"
        echo "   python evaluate_enhanced.py --sample-size 50 --use-bertscore"
        ;;
    3)
        echo "   # Basic evaluation"
        echo "   python evaluate_translation.py --sample-size 50"
        ;;
esac

echo ""
echo "3. Check results in Results/ directory"
echo ""

print_info "Documentation:"
echo "   - README.md              - Comprehensive guide"
echo "   - EVALUATION_GUIDE.md    - Script comparison"
echo "   - CHANGES_SUMMARY.md     - What's new"
echo ""

print_success "Setup completed successfully!"
echo ""

# Offer to show usage examples
read -p "Show usage examples? [y/N]: " show_examples
if [[ $show_examples =~ ^[Yy]$ ]]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    Usage Examples                              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "# Quick test (20 samples)"
    echo "python evaluate_semantic.py --sample-size 20"
    echo ""
    echo "# English only"
    echo "python evaluate_semantic.py --sample-size 50 --english-only"
    echo ""
    echo "# Faster (skip COMET)"
    echo "python evaluate_semantic.py --sample-size 50 --no-comet"
    echo ""
    echo "# All samples, all metrics"
    echo "python evaluate_semantic.py"
    echo ""
    echo "# Compare with basic evaluation"
    echo "python evaluate_translation.py --sample-size 50"
    echo ""
fi

echo "Done! 🎉"
