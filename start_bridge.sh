#!/bin/bash

# 🛡️ CyberShield AI - Streamlit Frontend Bridge
# This script starts the API bridge that connects the frontend to Streamlit

echo "🛡️ CyberShield AI - Starting Streamlit Integration Bridge"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import flask, flask_cors, requests, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip install flask flask-cors requests pandas numpy
fi

# Create the bridge configuration
echo "🔧 Setting up Streamlit bridge..."

# Stop any existing bridge process
pkill -f "streamlit_api_backend.py" 2>/dev/null || true

# Start the API bridge
echo "🚀 Starting API bridge on http://localhost:5000"
echo "📡 Connecting to Streamlit: https://cybershield-ai-app-vmbevtd5fcdrjfexcthga5.streamlit.app"
echo ""
echo "✅ Frontend (Netlify): https://rad-donut-a8e264.netlify.app/"
echo "✅ Streamlit App: https://cybershield-ai-app-vmbevtd5fcdrjfexcthga5.streamlit.app/"
echo "🔗 API Bridge: http://localhost:5000"
echo ""
echo "To stop the bridge, press Ctrl+C"
echo ""

# Start the bridge server
python3 streamlit_api_backend.py
