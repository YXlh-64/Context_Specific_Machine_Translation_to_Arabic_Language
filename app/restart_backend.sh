#!/bin/bash

# Restart Backend Script
# Stops any running Flask instances and starts a fresh one

cd "$(dirname "$0")/api"

echo "============================================================"
echo "🔄 RESTARTING BACKEND..."
echo "============================================================"

# Kill any existing Flask processes
echo "Stopping existing Flask processes..."
pkill -f "python app.py" || true
sleep 2

# Clear GPU memory (optional - only if you have nvidia-smi)
if command -v nvidia-smi &> /dev/null; then
    echo "Clearing GPU memory..."
    nvidia-smi --gpu-reset || true
fi

echo ""
echo "Starting fresh Flask backend..."
echo ""

# Start Flask
python app.py
