#!/bin/bash

# 🚀 GitLab Migration Script for Fraud Detection AI
# Run this script after creating your GitLab repository

echo "🚀 CyberShield AI - GitLab Migration Script"
echo "==========================================="

# Check if GitLab URL is provided
if [ -z "$1" ]; then
    echo "❌ Please provide your GitLab repository URL"
    echo "Usage: ./migrate_to_gitlab.sh https://gitlab.com/yourusername/fraud-detection-ai.git"
    exit 1
fi

GITLAB_URL=$1

echo "📋 Migration Steps:"
echo "1. Adding GitLab as remote..."
git remote add gitlab $GITLAB_URL

echo "2. Pushing all branches and tags..."
git push --all gitlab
git push --tags gitlab

echo "3. Setting GitLab as default remote..."
git remote set-url origin $GITLAB_URL

echo "✅ Migration Complete!"
echo "🎯 Your fraud detection AI is now hosted on GitLab!"
echo "📊 Large files (CSV, PKL) are handled by Git LFS"
echo "🌐 Repository URL: $GITLAB_URL"

echo ""
echo "📝 Next Steps:"
echo "- Visit your GitLab repository"
echo "- Enable GitLab Pages for web demos"
echo "- Set up CI/CD pipelines"
echo "- Configure issue tracking"

echo ""
echo "🔗 GitLab Features Available:"
echo "- ✅ 10GB repository storage"
echo "- ✅ Git LFS for large files"
echo "- ✅ CI/CD pipelines"
echo "- ✅ Container registry"
echo "- ✅ Issue tracking"
echo "- ✅ Wiki and documentation"
