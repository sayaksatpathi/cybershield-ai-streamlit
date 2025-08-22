# 🛡️ CyberShield AI - Project Structure

## 📁 Core Files

### 🧠 Machine Learning Components
- `enhanced_prediction_interface.py` - Main ML prediction interface
- `data_generator.py` - Synthetic transaction data generation
- `model_training.py` - Model training pipeline
- `prediction_interface.py` - Basic prediction interface
- `feature_engineering.py` - Feature extraction utilities
- `main_pipeline.py` - Complete ML pipeline
- `simple_pipeline.py` - Simplified training pipeline

### 🌐 Web Applications
- `api_server.py` - Flask API backend server
- `streamlit_app/cybershield_app.py` - Streamlit interactive interface
- `frontend/` - HTML/CSS/JS standalone frontend
  - `index.html` - Main HTML interface
  - `styles.css` - Cybersecurity-themed styling
  - `script.js` - Interactive JavaScript

### 📊 Data Files
- `transaction_data.csv` - Generated transaction dataset
- `customer_profiles_generated.csv` - Customer data
- `enhanced_customer_profiles.csv` - Enhanced customer profiles
- `customer_profiles.csv` - Base customer data

### 🤖 Model Files
- `enhanced_fraud_detection_model.pkl` - Trained ML model
- `enhanced_fraud_detection_scaler.pkl` - Feature scaler
- `enhanced_fraud_detection_model_metadata.pkl` - Model metadata
- `fraud_detection_model.pkl` - Basic model
- `fraud_detection_model_metadata.pkl` - Basic model metadata

### 🔧 Configuration & Deployment
- `requirements.txt` - Python dependencies
- `gunicorn.conf.py` - Production server configuration
- `wsgi.py` - WSGI application entry point
- `setup.sh` - Environment setup script
- `startup.sh` - Application startup script
- `start_production.sh` - Production deployment script

### 📚 Documentation
- `README.md` - Main project documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `LICENSE` - Project license
- `PROJECT_STRUCTURE.md` - This file

### ⚙️ Configuration
- `.gitignore` - Git ignore rules
- `.streamlit/` - Streamlit configuration

## 🚀 Quick Start Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start Streamlit app
streamlit run streamlit_app/cybershield_app.py

# Start API server
python api_server.py
```

### Production
```bash
# Setup environment
chmod +x setup.sh && ./setup.sh

# Start production server
chmod +x start_production.sh && ./start_production.sh
```

## 🎯 File Purposes

### Essential Files (Keep)
- All ML components
- Web applications
- Data files (CSV)
- Model files (PKL)
- Configuration files
- Documentation

### Removed Files (Cleaned up)
- Duplicate HTML files
- Redundant API implementations
- Old testing files
- Duplicate README files
- Temporary log directories
- Cache directories
