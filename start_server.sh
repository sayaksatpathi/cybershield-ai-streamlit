#!/bin/bash

echo "🛡️ CyberShield AI - Backend Server Startup Script"
echo "================================================="

# Change to the correct directory
cd /home/sayak/coding/fraud-detection-streamlit

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Check if backend files exist
if [ ! -f "simple_backend.py" ]; then
    echo "❌ Error: simple_backend.py not found"
    exit 1
fi

if [ ! -f "backend_api.py" ]; then
    echo "❌ Error: backend_api.py not found"
    exit 1
fi

echo "📁 Available backend files:"
ls -la *.py | grep backend

echo ""
echo "🚀 Starting CyberShield AI Backend Server..."
echo "   Access: http://localhost:5000/api/health"
echo "   Press Ctrl+C to stop"
echo ""

# Try to start simple backend first, fallback to main backend
if python simple_backend.py; then
    echo "✅ Simple backend started successfully"
else
    echo "⚠️  Simple backend failed, trying main backend..."
    python backend_api.py
fi
