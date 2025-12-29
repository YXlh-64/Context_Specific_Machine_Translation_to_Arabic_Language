#!/bin/bash

# Run Flask Backend
# This script runs the Flask backend server

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Flask Backend"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating parent virtual environment..."
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found"
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

# Run Flask app
python -m api.app
