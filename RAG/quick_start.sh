#!/bin/bash

# Quick Start Script for NLP Translation Pipeline
# This script helps you set up and run all services

set -e  # Exit on error

echo "=========================================="
echo "NLP Translation Pipeline - Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
echo "Installing Glossary System dependencies..."
pip install -q -r glossary-system/requirements.txt

echo "Installing RAG System dependencies..."
pip install -q -r RAG-SYSTEM/requirements.txt

echo "Installing Prompt Construction dependencies..."
pip install -q -r prompt-construction/requirements.txt

# Check for spaCy models
echo -e "${YELLOW}Checking spaCy models...${NC}"
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
    echo "Downloading spaCy English model..."
    python -m spacy download en_core_web_sm
}

# Check Redis
echo -e "${YELLOW}Checking Redis...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}Redis is running${NC}"
    else
        echo -e "${YELLOW}Redis is not running. Starting Redis...${NC}"
        redis-server --daemonize yes 2>/dev/null || echo -e "${RED}Could not start Redis. Please start it manually.${NC}"
    fi
else
    echo -e "${YELLOW}Redis not found. It's optional but recommended.${NC}"
fi

# Check Qdrant
echo -e "${YELLOW}Checking Qdrant...${NC}"
if curl -s http://localhost:6333/health &> /dev/null; then
    echo -e "${GREEN}Qdrant is running${NC}"
else
    echo -e "${YELLOW}Qdrant is not running.${NC}"
    if command -v docker &> /dev/null; then
        echo "Starting Qdrant with Docker..."
        docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant:latest 2>/dev/null || {
            echo -e "${GREEN}Qdrant container already exists, starting it...${NC}"
            docker start qdrant
        }
        sleep 2
        if curl -s http://localhost:6333/health &> /dev/null; then
            echo -e "${GREEN}Qdrant started successfully${NC}"
        else
            echo -e "${RED}Failed to start Qdrant. Please start it manually.${NC}"
        fi
    else
        echo -e "${RED}Docker not found. Please install Docker and start Qdrant manually.${NC}"
        echo "Run: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest"
    fi
fi

# Initialize Glossary Database
echo -e "${YELLOW}Initializing Glossary Database...${NC}"
cd glossary-system
if [ ! -f "data/glossary.db" ]; then
    echo "Creating database..."
    python scripts/init_db.py
    if [ -f "scripts/seed_glossary.py" ]; then
        echo "Seeding database..."
        python scripts/seed_glossary.py || echo -e "${YELLOW}Note: Database seeding may require CSV files${NC}"
    fi
else
    echo -e "${GREEN}Database already exists${NC}"
fi
cd ..

# Setup Qdrant Collection (optional)
echo -e "${YELLOW}Setting up Qdrant collection...${NC}"
cd RAG-SYSTEM
if [ -f "scripts/setup.py" ]; then
    echo "Running Qdrant setup..."
    python scripts/setup.py || echo -e "${YELLOW}Note: Qdrant setup may require translation data files${NC}"
fi
cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start the services, run:"
echo ""
echo "  Terminal 1 - Glossary System:"
echo "    cd glossary-system"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
echo ""
echo "  Terminal 2 - RAG System:"
echo "    cd RAG-SYSTEM"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload"
echo ""
echo "  Terminal 3 - Prompt Construction:"
echo "    cd prompt-construction"
echo "    uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload"
echo ""
echo "Or use the start_all_services.sh script (if available)"
echo ""
echo "API Documentation will be available at:"
echo "  - Glossary: http://localhost:8001/docs"
echo "  - RAG: http://localhost:8002/docs"
echo "  - Prompt: http://localhost:8003/docs"
echo ""
