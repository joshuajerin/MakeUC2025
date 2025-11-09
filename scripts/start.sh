#!/bin/bash

# EmpathLens Quick Start Script
# This script checks dependencies and starts the service

set -e

echo "================================================"
echo "EmpathLens - AI Social Companion"
echo "Quick Start Script"
echo "================================================"
echo ""

# Check Python virtual environment
echo "[1/3] Checking Python environment..."
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "[2/3] Installing dependencies..."
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check .env file
echo "[3/3] Checking configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "   ⚠️  IMPORTANT: Please edit .env and add your GEMINI_API_KEY!"
    echo "   Get your free API key at: https://makersuite.google.com/app/apikey"
    echo ""
    read -p "Press Enter after you've added your API key to .env..."
fi
echo "✅ Configuration ready"

echo ""
echo "================================================"
echo "Starting EmpathLens Provider..."
echo "================================================"
echo ""
echo "Service will be available at: http://localhost:8000"
echo ""
echo "Endpoints:"
echo "  - POST /distress/infer       (Panic attack support)"
echo "  - POST /conversation/assist  (Social conversation help)"
echo "  - GET  /health               (Health check)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the service using uvicorn directly
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
