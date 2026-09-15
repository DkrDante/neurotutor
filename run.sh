#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# Setup virtual environment if missing
if [ ! -d ".venv" ]; then
    echo "⚡ Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "⚡ Installing dependencies..."
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Check if server is already running on port 8000
if lsof -i :8000 > /dev/null 2>&1; then
    echo "✅ NeuroTutor server is already running!"
    echo "🌐 Opening http://localhost:8000..."
    open "http://localhost:8000" 2>/dev/null || true
else
    echo "🚀 Starting NeuroTutor server on http://localhost:8000..."
    open "http://localhost:8000" 2>/dev/null || true
    exec uvicorn web.server:app --host 127.0.0.1 --port 8000 --reload
fi
