#!/bin/bash

# Start Integrated Translation System
# Starts RAG services and Flask backend together

set -e

echo "=========================================="
echo "Starting Integrated Translation System"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -d "RAG" ] || [ ! -d "app" ]; then
    echo -e "${RED}Error: Must run from Context_Specific_Machine_Translation_to_Arabic_Language directory${NC}"
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Check all required ports
PORTS=(5000 8001 8002 8003)
for port in "${PORTS[@]}"; do
    if check_port $port; then
        echo -e "${RED}Port $port is already in use${NC}"
        exit 1
    fi
done

# Start RAG services
echo -e "${GREEN}Starting RAG middleware services...${NC}"
cd RAG
if [ -f "start_all_services.sh" ]; then
    ./start_all_services.sh
    echo ""
else
    echo -e "${RED}RAG start script not found${NC}"
    exit 1
fi
cd ..

# Wait for RAG services to be ready
echo -e "${YELLOW}Waiting for RAG services to initialize...${NC}"
sleep 5

# Check RAG services
echo "Checking RAG service health..."
RAG_SERVICES_OK=true

if ! curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${RED}✗ Glossary System (8001) not responding${NC}"
    RAG_SERVICES_OK=false
else
    echo -e "${GREEN}✓ Glossary System (8001) is running${NC}"
fi

if ! curl -s http://localhost:8002/health > /dev/null; then
    echo -e "${RED}✗ RAG System (8002) not responding${NC}"
    RAG_SERVICES_OK=false
else
    echo -e "${GREEN}✓ RAG System (8002) is running${NC}"
fi

if ! curl -s http://localhost:8003/health > /dev/null; then
    echo -e "${RED}✗ Prompt Construction (8003) not responding${NC}"
    RAG_SERVICES_OK=false
else
    echo -e "${GREEN}✓ Prompt Construction (8003) is running${NC}"
fi

if [ "$RAG_SERVICES_OK" = false ]; then
    echo -e "${YELLOW}Warning: Some RAG services are not responding. Continuing anyway...${NC}"
fi

# Start Flask backend
echo ""
echo -e "${GREEN}Starting Flask backend...${NC}"
cd app

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    echo -e "${YELLOW}No virtual environment found. Creating one...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        source ../venv/bin/activate
    fi
fi

# Create logs directory
mkdir -p logs

# Start Flask in background
echo "Starting Flask backend on port 5000..."
nohup python -m api.app > logs/flask.log 2>&1 &
FLASK_PID=$!
echo "Flask backend PID: $FLASK_PID"

cd ..

# Wait for Flask to start
sleep 3

# Check Flask backend
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo -e "${GREEN}✓ Flask backend (5000) is running${NC}"
else
    echo -e "${RED}✗ Flask backend (5000) not responding${NC}"
    echo "Check logs/app/flask.log for errors"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "System Started Successfully!"
echo "==========================================${NC}"
echo ""
echo "Service URLs:"
echo "  Flask Backend:        http://localhost:5000"
echo "  Glossary System:      http://localhost:8001"
echo "  RAG System:           http://localhost:8002"
echo "  Prompt Construction:  http://localhost:8003"
echo ""
echo "API Documentation:"
echo "  Flask:    http://localhost:5000/api"
echo "  Glossary: http://localhost:8001/docs"
echo "  RAG:      http://localhost:8002/docs"
echo "  Prompt:   http://localhost:8003/docs"
echo ""
echo "Logs:"
echo "  Flask:    app/logs/flask.log"
echo "  RAG:      RAG/logs/"
echo ""
echo "To stop all services:"
echo "  pkill -f 'uvicorn.*8001'"
echo "  pkill -f 'uvicorn.*8002'"
echo "  pkill -f 'uvicorn.*8003'"
echo "  kill $FLASK_PID"
echo ""
echo "Or run: ./stop_integrated_system.sh"
echo ""

# Save PIDs
echo "$FLASK_PID" > .flask_pid
