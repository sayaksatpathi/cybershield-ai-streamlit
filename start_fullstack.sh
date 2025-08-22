#!/bin/bash
# CyberShield AI - Full Stack Startup Script
# Starts both backend API and serves frontend

echo "🛡️ Starting CyberShield AI Full Stack System..."
echo "======================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "📦 Installing Flask..."
    pip install flask flask-cors
fi

# Check if sklearn is installed
if ! python3 -c "import sklearn" &> /dev/null; then
    echo "📦 Installing scikit-learn..."
    pip install scikit-learn pandas numpy
fi

# Kill any existing processes on ports 5000 and 8080
echo "🔄 Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Start backend API server
echo "🚀 Starting Backend API Server (Port 5000)..."
python3 backend_api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start simple HTTP server for frontend
echo "🌐 Starting Frontend Server (Port 8080)..."
cd frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 2

echo "======================================================="
echo "✅ CyberShield AI Full Stack System is now running!"
echo ""
echo "🔗 Access Points:"
echo "   Frontend:  http://localhost:8080/index_backend.html"
echo "   Backend:   http://localhost:5000/api/health"
echo ""
echo "🛡️ Features Available:"
echo "   • 1GB File Upload Support"
echo "   • 4 Machine Learning Models"
echo "   • Real-time Fraud Prediction"
echo "   • Synthetic Data Generation"
echo "   • Custom Dataset Processing"
echo ""
echo "📊 System Status:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "🛑 To stop the system, press Ctrl+C"
echo "======================================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down CyberShield AI..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "✅ System shutdown complete."
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Keep script running
wait
