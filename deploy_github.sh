#!/bin/bash

# 🚀 GitHub Deployment Script for CyberShield AI Streamlit

echo "🚀 CyberShield AI - GitHub Deployment"
echo "===================================="

# Check if GitHub repo URL is provided
if [ -z "$1" ]; then
    echo "❌ Please provide your GitHub repository URL"
    echo ""
    echo "📋 Steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: cybershield-ai-streamlit"
    echo "3. Make it Public (for free Streamlit hosting)"
    echo "4. DON'T initialize with README"
    echo "5. Click 'Create repository'"
    echo "6. Copy the HTTPS URL"
    echo ""
    echo "Usage: ./deploy_github.sh https://github.com/yourusername/cybershield-ai-streamlit.git"
    exit 1
fi

GITHUB_URL=$1

echo "📊 Project Stats:"
echo "- Size: $(du -sh . | cut -f1)"
echo "- Files: $(find . -type f | wc -l) files"
echo "- No large CSV/PKL files ✅"
echo ""

echo "🔗 Setting up GitHub remote..."
git remote add origin $GITHUB_URL

echo "📤 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Your CyberShield AI is now on GitHub!"
    echo ""
    echo "🚀 Next Steps - Deploy to Streamlit Cloud:"
    echo "1. Go to https://share.streamlit.io"
    echo "2. Sign in with GitHub"
    echo "3. Click 'New app'"
    echo "4. Select repository: cybershield-ai-streamlit"
    echo "5. Main file: simple_model_trainer.py"
    echo "6. Click 'Deploy!'"
    echo ""
    echo "🎯 Your app will be live at:"
    echo "https://yourusername-cybershield-ai-streamlit-simple-model-trainer-xyz.streamlit.app"
    echo ""
    echo "📱 Features included:"
    echo "- ✅ Interactive Model Trainer"
    echo "- ✅ Real-time Fraud Detection"
    echo "- ✅ Performance Analytics"
    echo "- ✅ Data Visualization"
    echo "- ✅ Multiple ML Algorithms"
else
    echo "❌ Push failed. Please check your GitHub URL and try again."
fi
