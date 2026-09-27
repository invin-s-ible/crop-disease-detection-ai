#!/bin/bash
# Smart Plant Health Monitoring System - Linux/Mac Startup Script

echo "===================================="
echo "Smart Plant Health Monitoring System"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version

echo "[2/4] Installing/checking dependencies..."
pip3 install -r requirements.txt

echo "[3/4] Checking for .env file..."
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your Groq API key!"
    read -p "Press Enter to continue..."
fi

echo "[4/4] Starting application..."
echo ""
echo "===================================="
echo "Server starting on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "===================================="
echo ""

python3 app.py
