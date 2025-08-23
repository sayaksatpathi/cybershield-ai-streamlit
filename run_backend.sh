#!/bin/bash

echo "🛡️ CyberShield AI Backend Server"
echo "================================"

# Kill any existing backend processes
pkill -f "python.*backend" 2>/dev/null || true
pkill -f "flask" 2>/dev/null || true

sleep 2

# Navigate to project directory
cd /home/sayak/coding/fraud-detection-streamlit

echo "📂 Project directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

# Check if files exist
if [ -f "simple_backend.py" ]; then
    echo "✅ Found simple_backend.py"
    echo "🚀 Starting simple backend server..."
    echo "   URL: http://localhost:5000"
    echo "   Health: http://localhost:5000/api/health"
    echo ""
    exec python simple_backend.py
elif [ -f "backend_api.py" ]; then
    echo "✅ Found backend_api.py"
    echo "🚀 Starting main backend server..."
    echo "   URL: http://localhost:5000"
    echo "   Health: http://localhost:5000/api/health"
    echo ""
    exec python backend_api.py
else
    echo "❌ No backend files found!"
    echo "   Please ensure you're in the correct directory"
    exit 1
fi
