#!/bin/bash

# 🌐 CyberShield AI - Create Netlify Full-Stack Package

echo "🛡️ CyberShield AI - Creating Netlify Full-Stack Package"
echo "=================================================="

# Create the deployment package
echo "📦 Creating deployment package..."

# Create a clean package directory
rm -rf netlify-package
mkdir netlify-package

# Copy frontend files
echo "📱 Copying frontend files..."
cp -r frontend/* netlify-package/

# Copy Netlify functions
echo "⚡ Copying serverless functions..."
cp -r netlify-functions netlify-package/

# Copy configuration
echo "🔧 Copying configuration..."
cp netlify.toml netlify-package/

# Create the zip package
echo "📦 Creating deployment package..."
cd netlify-package
zip -r ../cybershield-fullstack-netlify.zip .
cd ..

# Show package contents
echo ""
echo "✅ Package created: cybershield-fullstack-netlify.zip"
echo "📄 Package contents:"
unzip -l cybershield-fullstack-netlify.zip

echo ""
echo "🚀 DEPLOYMENT INSTRUCTIONS:"
echo "1. Visit: https://app.netlify.com/drop"
echo "2. Drag & drop: cybershield-fullstack-netlify.zip"
echo "3. Wait for deployment (2-3 minutes)"
echo "4. Your full-stack app will be live!"
echo ""
echo "🎯 Features included:"
echo "   ✅ Frontend web interface"
echo "   ✅ Serverless Python backend"
echo "   ✅ API endpoints (/api/health, /api/predict, etc.)"
echo "   ✅ Streamlit integration"
echo "   ✅ Complete fraud detection system"
echo ""
echo "🌐 After deployment, update your Netlify site at:"
echo "   https://rad-donut-a8e264.netlify.app/"
echo ""
echo "Package ready for deployment! 🎉"
