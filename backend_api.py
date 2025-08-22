"""
🛡️ CyberShield AI - Enhanced Backend API Server
Connects Streamlit fraud detection engine with frontend interface
Supports 1GB file uploads and advanced fraud detection models
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import os
import json
import logging
from datetime import datetime
import warnings
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure for 1GB uploads
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
scaler = None
feature_columns = []

class FraudDetectionEngine:
    """Advanced fraud detection engine matching Streamlit functionality"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        
    def generate_synthetic_data(self, num_transactions=10000):
        """Generate synthetic fraud data matching Streamlit functionality"""
        np.random.seed(42)
        
        # Enhanced fraud patterns
        data = []
        for i in range(num_transactions):
            is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate
            
            if is_fraud:
                # Fraudulent transaction patterns
                transaction = {
                    'amount': np.random.exponential(500) + np.random.uniform(50, 5000),
                    'hour': np.random.choice([0, 1, 2, 3, 22, 23], p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice([1, 7, 8, 9, 10]),
                    'transaction_count_1h': np.random.poisson(8),
                    'transaction_count_24h': np.random.poisson(50),
                    'avg_amount_30d': np.random.uniform(50, 200),
                    'card_present': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'country_risk_score': np.random.uniform(0.6, 1.0),
                    'velocity_score': np.random.uniform(0.7, 1.0),
                    'is_fraud': 1
                }
            else:
                # Legitimate transaction patterns
                transaction = {
                    'amount': np.random.lognormal(3, 1.5),
                    'hour': np.random.choice(range(24), p=[0.01]*6 + [0.05]*12 + [0.08]*6),
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice(range(1, 15)),
                    'transaction_count_1h': np.random.poisson(2),
                    'transaction_count_24h': np.random.poisson(15),
                    'avg_amount_30d': np.random.uniform(100, 500),
                    'card_present': np.random.choice([0, 1], p=[0.3, 0.7]),
                    'country_risk_score': np.random.uniform(0.0, 0.3),
                    'velocity_score': np.random.uniform(0.0, 0.4),
                    'is_fraud': 0
                }
            
            data.append(transaction)
        
        return pd.DataFrame(data)
    
    def train_models(self, df):
        """Train all fraud detection models"""
        try:
            # Prepare features
            feature_cols = [col for col in df.columns if col != 'is_fraud']
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df['is_fraud']
            
            # Store feature columns
            self.feature_columns = feature_cols
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train all models
            results = {}
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[name] = {
                    'auc': float(auc_score),
                    'precision': float(report['1']['precision']),
                    'recall': float(report['1']['recall']),
                    'f1_score': float(report['1']['f1-score']),
                    'accuracy': float(report['accuracy'])
                }
            
            self.is_trained = True
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return None
    
    def predict_single(self, transaction_data, model_name='Random Forest'):
        """Predict fraud probability for single transaction"""
        if not self.is_trained:
            return None
            
        try:
            # Prepare features
            features = []
            for col in self.feature_columns:
                features.append(transaction_data.get(col, 0))
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            model = self.models[model_name]
            probability = model.predict_proba(features_scaled)[0][1]
            prediction = model.predict(features_scaled)[0]
            
            return {
                'fraud_probability': float(probability),
                'is_fraud_predicted': bool(prediction),
                'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            return None

# Global fraud detection engine
fraud_engine = FraudDetectionEngine()

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files"""
    return send_from_directory('frontend', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1.0',
        'features': {
            'max_upload_size': '1GB',
            'models_trained': fraud_engine.is_trained,
            'available_models': list(fraud_engine.models.keys())
        }
    })

@app.route('/api/generate-data', methods=['POST'])
def generate_synthetic_data():
    """Generate synthetic fraud detection data"""
    try:
        data = request.json or {}
        num_transactions = data.get('num_transactions', 10000)
        
        # Generate data
        df = fraud_engine.generate_synthetic_data(num_transactions)
        
        # Train models
        results = fraud_engine.train_models(df)
        
        if results:
            return jsonify({
                'success': True,
                'message': f'Generated {len(df)} transactions and trained models',
                'data_stats': {
                    'total_transactions': len(df),
                    'fraud_count': int(df['is_fraud'].sum()),
                    'fraud_rate': float(df['is_fraud'].mean()),
                    'features': df.columns.tolist()
                },
                'model_results': results
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to train models'
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and process custom dataset (supports up to 1GB)"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                'success': False,
                'message': 'Only CSV files are supported'
            }), 400
        
        # Read CSV directly from memory
        try:
            df = pd.read_csv(file.stream)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error reading CSV: {str(e)}'
            }), 400
        
        # Validate dataset
        if 'is_fraud' not in df.columns:
            return jsonify({
                'success': False,
                'message': 'Dataset must contain "is_fraud" column'
            }), 400
        
        # Train models
        results = fraud_engine.train_models(df)
        
        if results:
            return jsonify({
                'success': True,
                'message': f'Successfully processed dataset with {len(df)} transactions',
                'data_stats': {
                    'total_transactions': len(df),
                    'fraud_count': int(df['is_fraud'].sum()),
                    'fraud_rate': float(df['is_fraud'].mean()),
                    'features': df.columns.tolist(),
                    'file_size_mb': round(file.content_length / (1024 * 1024), 2) if file.content_length else 'Unknown'
                },
                'model_results': results
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to train models on uploaded dataset'
            }), 500
            
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'message': 'File too large. Maximum size is 1GB.'
        }), 413
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for single transaction"""
    try:
        if not fraud_engine.is_trained:
            return jsonify({
                'success': False,
                'message': 'Models not trained. Please generate data or upload dataset first.'
            }), 400
        
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': 'No transaction data provided'
            }), 400
        
        model_name = data.get('model', 'Random Forest')
        transaction_data = data.get('transaction', {})
        
        # Predict
        result = fraud_engine.predict_single(transaction_data, model_name)
        
        if result:
            return jsonify({
                'success': True,
                'prediction': result,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Prediction failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Error predicting: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Get status of all models"""
    return jsonify({
        'trained': fraud_engine.is_trained,
        'available_models': list(fraud_engine.models.keys()),
        'feature_columns': fraud_engine.feature_columns,
        'total_features': len(fraud_engine.feature_columns)
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum upload size is 1GB.'
    }), 413

if __name__ == '__main__':
    logger.info("🛡️ Starting CyberShield AI Backend Server...")
    logger.info("Features: 1GB file upload, 4 ML models, real-time prediction")
    app.run(host='0.0.0.0', port=5000, debug=True)
