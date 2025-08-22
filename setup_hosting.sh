#!/bin/bash

# 🚀 CyberShield AI - Production Deployment Script

echo "🚀 Deploying CyberShield AI to Production..."
echo "============================================"

# Check if we're in the right directory
if [ ! -f "simple_model_trainer.py" ]; then
    echo "❌ Error: Please run this script from the fraud-detection-ai directory"
    exit 1
fi

# Create production requirements
echo "📦 Creating production requirements..."
cat > requirements_production.txt << EOF
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.15.0
gunicorn==21.2.0
flask==2.3.2
joblib==1.3.2
seaborn==0.12.2
matplotlib==3.7.2
EOF

# Create Docker configuration
echo "🐳 Creating Docker configuration..."
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

COPY . .

EXPOSE 8501 5000

CMD ["streamlit", "run", "simple_model_trainer.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Create docker-compose for local testing
cat > docker-compose.yml << EOF
version: '3.8'
services:
  cybershield-ai:
    build: .
    ports:
      - "8501:8501"
      - "5000:5000"
    volumes:
      - .:/app
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
EOF

# Create Railway deployment config
cat > railway.json << EOF
{
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "startCommand": "streamlit run simple_model_trainer.py --server.port=\$PORT --server.address=0.0.0.0",
    "healthcheckPath": "/",
    "healthcheckTimeout": 300
  }
}
EOF

# Create Render deployment config
cat > render.yaml << EOF
services:
  - type: web
    name: cybershield-ai
    env: python
    buildCommand: pip install -r requirements_production.txt
    startCommand: streamlit run simple_model_trainer.py --server.port=\$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.16
EOF

# Create Vercel deployment config
cat > vercel.json << EOF
{
  "builds": [
    {
      "src": "api_server.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api_server.py"
    }
  ]
}
EOF

echo "✅ Production deployment files created!"
echo ""
echo "🚀 Available deployment options:"
echo "1. GitLab Pages: git push gitlab main"
echo "2. Docker: docker-compose up"
echo "3. Railway: railway deploy"
echo "4. Render: Connect GitLab repo"
echo "5. Vercel: vercel --prod"
echo ""
echo "📖 See HOSTING_GUIDE.md for detailed instructions"
