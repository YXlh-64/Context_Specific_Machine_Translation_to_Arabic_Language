#!/usr/bin/env fish
# Quick setup script for QWEN fine-tuning pipeline (Fish shell)

echo "================================================"
echo "QWEN Translation Fine-tuning Setup"
echo "================================================"
echo ""

# Check Python version
set python_version (python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if CUDA is available
if command -v nvidia-smi > /dev/null
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ WARNING: CUDA not detected. Training will be very slow on CPU."
end

echo ""
echo "================================================"
echo "Step 1: Installing Dependencies"
echo "================================================"
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Step 2: Preparing Data"
echo "================================================"
python prepare_data.py

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. To train a single domain:"
echo "   python train_qwen.py --domain technology"
echo ""
echo "2. To train all domains:"
echo "   python train_all.py"
echo ""
echo "3. To evaluate a model:"
echo "   python evaluate.py --model_path models/technology_model --domain technology"
echo ""
echo "4. To launch the UI:"
echo "   streamlit run app.py"
echo ""
