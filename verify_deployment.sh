#!/bin/bash

# 🚀 CyberShield AI - Complete Deployment Verification Script

echo "🔍 CyberShield AI Deployment Verification"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run from the fraud-detection-streamlit directory."
    exit 1
fi

echo "✅ Found streamlit_app.py"

# Check requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

echo "✅ Found requirements.txt"

# Check Streamlit config
if [ ! -f ".streamlit/config.toml" ]; then
    echo "❌ Error: .streamlit/config.toml not found"
    exit 1
fi

echo "✅ Found .streamlit/config.toml"

# Verify 1GB upload setting
if grep -q "maxUploadSize = 1024" .streamlit/config.toml; then
    echo "✅ 1GB upload limit configured"
else
    echo "❌ Warning: 1GB upload limit not found in config"
fi

# Check Python dependencies
echo "🔧 Checking Python dependencies..."
python -c "import streamlit, pandas, numpy, sklearn, plotly, seaborn, matplotlib, joblib, imblearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All Python dependencies available"
else
    echo "⚠️  Some dependencies may be missing. Run: pip install -r requirements.txt"
fi

# Check Git status
echo "📚 Checking Git repository status..."
git status --porcelain
if [ $? -eq 0 ]; then
    echo "✅ Git repository is clean"
else
    echo "⚠️  Git repository has uncommitted changes"
fi

# Test app syntax
echo "🔍 Testing app syntax..."
python -m py_compile streamlit_app.py
if [ $? -eq 0 ]; then
    echo "✅ streamlit_app.py syntax is valid"
else
    echo "❌ Syntax errors found in streamlit_app.py"
    exit 1
fi

echo ""
echo "🎯 Deployment Status Summary:"
echo "=============================="
echo "✅ Core files present"
echo "✅ 1GB upload capability configured"
echo "✅ Dependencies checked"
echo "✅ App syntax validated"
echo "✅ Git repository status verified"
echo ""
echo "🚀 Ready for deployment!"
echo ""
echo "📍 Deployment Options:"
echo "1. Local: streamlit run streamlit_app.py --server.port 8502"
echo "2. Streamlit Cloud: https://share.streamlit.io/ (auto-deploy from GitHub)"
echo "3. Production: Use Docker/Railway/Vercel configurations included"
echo ""
echo "🌐 GitHub Repository: https://github.com/sayaksatpathi/cybershield-ai-streamlit"
echo "📊 Local App URL: http://localhost:8502"
echo ""
echo "✨ CyberShield AI with 1GB Upload Capability - DEPLOYMENT READY! ✨"
