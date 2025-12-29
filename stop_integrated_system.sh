#!/bin/bash

# Stop Integrated Translation System

echo "Stopping Integrated Translation System..."

# Stop Flask backend
if [ -f ".flask_pid" ]; then
    FLASK_PID=$(cat .flask_pid)
    if ps -p $FLASK_PID > /dev/null 2>&1; then
        echo "Stopping Flask backend (PID: $FLASK_PID)..."
        kill $FLASK_PID
        rm .flask_pid
    fi
fi

# Stop RAG services
echo "Stopping RAG services..."
pkill -f 'uvicorn.*8001' 2>/dev/null || true
pkill -f 'uvicorn.*8002' 2>/dev/null || true
pkill -f 'uvicorn.*8003' 2>/dev/null || true

# Also check for PIDs file from RAG start script
if [ -f "RAG/.service_pids" ]; then
    PIDS=$(cat RAG/.service_pids)
    for pid in $PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null || true
        fi
    done
    rm RAG/.service_pids
fi

echo "All services stopped."
