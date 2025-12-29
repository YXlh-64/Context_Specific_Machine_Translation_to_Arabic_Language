#!/bin/bash

# Helper script to run individual services with the virtual environment
# Usage: ./run_service.sh [glossary|rag|prompt] [port]

set -e

SERVICE=$1
PORT=${2:-8001}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Please run quick_start.sh first.${NC}"
    exit 1
fi

# Prefer venv Python + module invocation so we don't depend on a uvicorn entrypoint script
if [ -n "$VIRTUAL_ENV" ]; then
    # venv already activated in the shell
    UVICORN_CMD="python -m uvicorn"
else
    # run explicitly via venv Python
    UVICORN_CMD="venv/bin/python -m uvicorn"
fi

case $SERVICE in
    glossary)
        echo -e "${GREEN}Starting Glossary System on port $PORT...${NC}"
        cd glossary-system
        $UVICORN_CMD app.main:app --host 0.0.0.0 --port $PORT --reload
        ;;
    rag)
        echo -e "${GREEN}Starting RAG System on port $PORT...${NC}"
        cd RAG-SYSTEM
        $UVICORN_CMD app.main:app --host 0.0.0.0 --port $PORT --reload
        ;;
    prompt)
        echo -e "${GREEN}Starting Prompt Construction on port $PORT...${NC}"
        cd prompt-construction
        $UVICORN_CMD app.main:app --host 0.0.0.0 --port $PORT --reload
        ;;
    *)
        echo -e "${RED}Usage: $0 [glossary|rag|prompt] [port]${NC}"
        echo ""
        echo "Examples:"
        echo "  $0 glossary 8001"
        echo "  $0 rag 8002"
        echo "  $0 prompt 8003"
        exit 1
        ;;
esac
