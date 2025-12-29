#!/bin/bash

# Start All Services Script
# This script starts all three services in separate background processes

set -e

echo "=========================================="
echo "Starting All NLP Translation Services"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Activating virtual environment...${NC}"
        source venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found. Please run quick_start.sh first.${NC}"
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Check ports
if check_port 8001; then
    echo -e "${RED}Port 8001 is already in use (Glossary System)${NC}"
    exit 1
fi

if check_port 8002; then
    echo -e "${RED}Port 8002 is already in use (RAG System)${NC}"
    exit 1
fi

if check_port 8003; then
    echo -e "${RED}Port 8003 is already in use (Prompt Construction)${NC}"
    exit 1
fi

# Start Glossary System
echo -e "${GREEN}Starting Glossary System on port 8001...${NC}"
cd glossary-system
nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload > ../logs/glossary.log 2>&1 &
GLOSSARY_PID=$!
echo "Glossary System PID: $GLOSSARY_PID"
cd ..

# Wait a bit
sleep 2

# Start RAG System
echo -e "${GREEN}Starting RAG System on port 8002...${NC}"
cd RAG-SYSTEM
nohup uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload > ../logs/rag.log 2>&1 &
RAG_PID=$!
echo "RAG System PID: $RAG_PID"
cd ..

# Wait a bit
sleep 2

# Start Prompt Construction
echo -e "${GREEN}Starting Prompt Construction on port 8003...${NC}"
cd prompt-construction
nohup uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload > ../logs/prompt.log 2>&1 &
PROMPT_PID=$!
echo "Prompt Construction PID: $PROMPT_PID"
cd ..

# Wait for services to start
echo ""
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Check if services are running
echo ""
echo "Checking service health..."

# Check Glossary
if curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✓ Glossary System is running${NC}"
else
    echo -e "${RED}✗ Glossary System failed to start${NC}"
fi

# Check RAG
if curl -s http://localhost:8002/health > /dev/null; then
    echo -e "${GREEN}✓ RAG System is running${NC}"
else
    echo -e "${RED}✗ RAG System failed to start${NC}"
fi

# Check Prompt
if curl -s http://localhost:8003/health > /dev/null; then
    echo -e "${GREEN}✓ Prompt Construction is running${NC}"
else
    echo -e "${RED}✗ Prompt Construction failed to start${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "All Services Started!"
echo "==========================================${NC}"
echo ""
echo "Service URLs:"
echo "  Glossary System:     http://localhost:8001"
echo "  RAG System:          http://localhost:8002"
echo "  Prompt Construction: http://localhost:8003"
echo ""
echo "API Documentation:"
echo "  Glossary: http://localhost:8001/docs"
echo "  RAG:      http://localhost:8002/docs"
echo "  Prompt:   http://localhost:8003/docs"
echo ""
echo "Logs are in the 'logs/' directory:"
echo "  - logs/glossary.log"
echo "  - logs/rag.log"
echo "  - logs/prompt.log"
echo ""
echo "To stop all services, run:"
echo "  ./stop_all_services.sh"
echo ""
echo "Or manually kill processes:"
echo "  kill $GLOSSARY_PID $RAG_PID $PROMPT_PID"
echo ""

# Save PIDs to file
echo "$GLOSSARY_PID $RAG_PID $PROMPT_PID" > .service_pids
