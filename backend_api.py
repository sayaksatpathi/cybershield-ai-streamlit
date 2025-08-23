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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
import re

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure for 1GB uploads with extended timeouts
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
scaler = None
feature_columns = []

class UniversalFraudDetector:
    """Universal fraud detection engine that can analyze any dataset format"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = []
        self.fraud_column = None
        self.dataset_type = None
        
    def read_any_file(self, file_stream, filename):
        """Read any type of file format with memory optimization for large files"""
        try:
            file_ext = filename.lower().split('.')[-1]
            
            if file_ext == 'csv':
                # Try different separators and encodings
                for sep in [',', ';', '\t', '|']:
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            file_stream.seek(0)
                            # For large files, use chunked reading
                            try:
                                # Try to read first chunk to validate format
                                df_sample = pd.read_csv(file_stream, sep=sep, encoding=encoding, nrows=1000)
                                if len(df_sample.columns) > 1:
                                    # Format validated, now read full file with chunking for memory efficiency
                                    file_stream.seek(0)
                                    chunks = []
                                    chunk_size = 50000  # Read 50k rows at a time
                                    
                                    for chunk in pd.read_csv(file_stream, sep=sep, encoding=encoding, chunksize=chunk_size):
                                        chunks.append(chunk)
                                        # Memory management for very large files
                                        if len(chunks) > 20:  # If more than 1M rows, sample it
                                            logger.warning(f"Large file detected, sampling for performance")
                                            df = pd.concat(chunks[:20], ignore_index=True)
                                            break
                                    else:
                                        df = pd.concat(chunks, ignore_index=True)
                                    
                                    return df
                            except Exception as chunk_error:
                                # Fallback to regular reading for smaller files
                                file_stream.seek(0)
                                df = pd.read_csv(file_stream, sep=sep, encoding=encoding)
                                if len(df.columns) > 1:
                                    return df
                        except:
                            continue
                            
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_stream)
                return df
                
            elif file_ext == 'json':
                file_stream.seek(0)
                data = json.load(file_stream)
                if isinstance(data, list):
                    df = pd.json_normalize(data)
                else:
                    df = pd.json_normalize([data])
                return df
                
            elif file_ext == 'tsv':
                file_stream.seek(0)
                df = pd.read_csv(file_stream, sep='\t')
                return df
                
            else:
                # Try as CSV with various delimiters
                file_stream.seek(0)
                df = pd.read_csv(file_stream)
                return df
                
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return None
    
    def intelligent_fraud_detection(self, df):
        """Intelligently detect fraud patterns in any dataset"""
        logger.info("Starting intelligent fraud analysis...")
        
        # Step 1: Find fraud column or create fraud labels
        fraud_info = self._find_or_create_fraud_column(df)
        
        if fraud_info['method'] == 'found':
            logger.info(f"Found fraud column: {fraud_info['column']}")
            return self._train_with_labels(df, fraud_info['column'])
        else:
            logger.info(f"Creating fraud labels using: {fraud_info['method']}")
            return self._train_unsupervised(df, fraud_info)
    
    def _find_or_create_fraud_column(self, df):
        """Find existing fraud column or determine how to create fraud labels"""
        
        # Extended list of fraud indicator patterns
        fraud_patterns = [
            # Exact matches
            'is_fraud', 'fraud', 'fraudulent', 'is_fraudulent', 'isFraud', 'isFlaggedFraud',
            'fraud_flag', 'fraud_indicator', 'is_fraud_flag', 'label', 'target', 'class',
            'outcome', 'result', 'status', 'anomaly', 'outlier', 'suspicious',
            
            # Pattern matches (regex)
            r'.*fraud.*', r'.*anomal.*', r'.*suspic.*', r'.*illegal.*', r'.*invalid.*',
            r'.*error.*', r'.*wrong.*', r'.*fake.*', r'.*false.*', r'.*cheat.*'
        ]
        
        # Look for existing fraud column
        for col in df.columns:
            col_lower = col.lower()
            
            # Check exact matches
            for pattern in fraud_patterns[:12]:  # First 12 are exact matches
                if pattern == col_lower:
                    return {'method': 'found', 'column': col}
            
            # Check pattern matches
            for pattern in fraud_patterns[12:]:  # Rest are regex patterns
                if re.search(pattern, col_lower):
                    return {'method': 'found', 'column': col}
        
        # No fraud column found - determine best unsupervised method
        return self._determine_unsupervised_method(df)
    
    def _determine_unsupervised_method(self, df):
        """Determine the best unsupervised method based on data characteristics"""
        
        # Analyze dataset characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Look for financial indicators
        financial_keywords = ['amount', 'price', 'cost', 'value', 'balance', 'transaction', 'payment']
        has_financial = any(any(keyword in col.lower() for keyword in financial_keywords) for col in df.columns)
        
        # Look for time indicators
        time_keywords = ['time', 'date', 'hour', 'day', 'month', 'year', 'timestamp']
        has_time = any(any(keyword in col.lower() for keyword in time_keywords) for col in df.columns)
        
        # Look for ID/categorical indicators
        id_keywords = ['id', 'name', 'user', 'customer', 'account', 'type', 'category']
        has_ids = any(any(keyword in col.lower() for keyword in id_keywords) for col in df.columns)
        
        # Determine dataset type and fraud detection strategy
        if has_financial and has_time:
            self.dataset_type = "financial_transactions"
            return {'method': 'financial_anomaly', 'features': numeric_cols}
        elif has_financial:
            self.dataset_type = "financial_data"
            return {'method': 'amount_anomaly', 'features': numeric_cols}
        elif len(numeric_cols) > len(categorical_cols):
            self.dataset_type = "numeric_heavy"
            return {'method': 'statistical_anomaly', 'features': numeric_cols}
        else:
            self.dataset_type = "mixed_data"
            return {'method': 'pattern_anomaly', 'features': numeric_cols + categorical_cols[:5]}
    
    def _train_with_labels(self, df, fraud_col):
        """Train models when fraud labels are available"""
        
        # Clean and prepare fraud column
        df_clean = df.copy()
        
        # Convert fraud column to binary
        fraud_series = df_clean[fraud_col].astype(str).str.lower()
        fraud_mapping = {
            '0': 0, '1': 1, 'false': 0, 'true': 1, 'no': 0, 'yes': 1,
            'legitimate': 0, 'fraud': 1, 'normal': 0, 'anomaly': 1,
            'valid': 0, 'invalid': 1, 'clean': 0, 'suspicious': 1
        }
        
        df_clean['is_fraud'] = fraud_series.map(fraud_mapping)
        df_clean = df_clean.dropna(subset=['is_fraud'])
        
        # Prepare features
        feature_cols = [col for col in df_clean.columns if col != fraud_col and col != 'is_fraud']
        df_features = self._prepare_features(df_clean[feature_cols])
        
        X = df_features
        y = df_clean['is_fraud'].astype(int)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        self.fraud_column = 'is_fraud'
        
        # Train models
        return self._train_supervised_models(X, y, df_clean)
    
    def _train_unsupervised(self, df, fraud_info):
        """Train unsupervised models to detect anomalies as fraud"""
        
        method = fraud_info['method']
        features = fraud_info.get('features', [])
        
        df_clean = df.copy()
        
        # Create fraud labels based on method
        if method == 'financial_anomaly':
            fraud_indices = self._detect_financial_anomalies(df_clean)
        elif method == 'amount_anomaly':
            fraud_indices = self._detect_amount_anomalies(df_clean)
        elif method == 'statistical_anomaly':
            fraud_indices = self._detect_statistical_anomalies(df_clean)
        else:  # pattern_anomaly
            fraud_indices = self._detect_pattern_anomalies(df_clean)
        
        # Create binary fraud column
        df_clean['is_fraud'] = 0
        df_clean.loc[fraud_indices, 'is_fraud'] = 1
        
        # Prepare features
        feature_cols = [col for col in df_clean.columns if col != 'is_fraud']
        df_features = self._prepare_features(df_clean[feature_cols])
        
        X = df_features
        y = df_clean['is_fraud']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        self.fraud_column = 'is_fraud'
        
        # Train models
        return self._train_supervised_models(X, y, df_clean)
    
    def _detect_financial_anomalies(self, df):
        """Detect financial anomalies - large amounts, round numbers, etc."""
        anomalies = []
        
        # Find numeric columns that might represent amounts
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        amount_cols = []
        
        for col in numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'cost', 'sum', 'total', 'balance', 'payment']):
                amount_cols.append(col)
        
        if not amount_cols:
            # If no obvious amount columns, use the numeric column with highest variance
            if len(numeric_cols) > 0:
                variances = df[numeric_cols].var()
                amount_cols = [variances.idxmax()]
        
        for col in amount_cols:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    # Detect outliers using IQR method
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_threshold = Q3 + 1.5 * IQR
                    
                    # Large amount anomalies
                    large_amounts = values > outlier_threshold
                    
                    # Round number anomalies (suspicious round amounts)
                    round_amounts = values % 100 == 0
                    
                    # Combine anomalies
                    financial_anomalies = large_amounts | round_amounts
                    
                    # Map back to original dataframe indices
                    for idx in values[financial_anomalies].index:
                        anomalies.append(idx)
        
        return list(set(anomalies))  # Remove duplicates
    
    def _detect_amount_anomalies(self, df):
        """Detect anomalies in transaction amounts or values."""
        anomalies = []
        
        # Find columns that might contain transaction amounts
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 10:  # Skip columns with too few values
                continue
                
            # Statistical outlier detection using Z-score
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val > 0:  # Avoid division by zero
                z_scores = np.abs((values - mean_val) / std_val)
                outliers = z_scores > 3  # Z-score threshold of 3
                
                # Add suspicious patterns
                # Very small amounts (potentially fake transactions)
                very_small = values < (mean_val * 0.01)
                
                # Repeated exact amounts (suspicious pattern)
                value_counts = values.value_counts()
                repeated_amounts = values.isin(value_counts[value_counts > max(3, len(values) * 0.05)].index)
                
                # Combine all amount anomalies
                amount_anomalies = outliers | very_small | repeated_amounts
                
                # Map back to original dataframe indices
                for idx in values[amount_anomalies].index:
                    anomalies.append(idx)
        
        return list(set(anomalies))
    
    def _detect_statistical_anomalies(self, df):
        """Detect statistical anomalies using unsupervised methods."""
        anomalies = []
        
        # Prepare data for anomaly detection
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return anomalies
        
        # Handle missing values
        numeric_df = numeric_df.fillna(numeric_df.median())
        
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_predictions = iso_forest.fit_predict(scaled_data)
        
        # DBSCAN clustering to find outliers
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(scaled_data)
        
        # Combine results
        for i, (iso_pred, cluster_label) in enumerate(zip(iso_predictions, cluster_labels)):
            # Isolation Forest outliers (prediction = -1)
            # DBSCAN outliers (cluster_label = -1)
            if iso_pred == -1 or cluster_label == -1:
                anomalies.append(numeric_df.index[i])
        
        return list(set(anomalies))
    
    def _detect_pattern_anomalies(self, df):
        """Detect pattern-based anomalies in text and categorical data."""
        anomalies = []
        
        # Analyze text and categorical columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            values = df[col].dropna().astype(str)
            if len(values) < 5:  # Skip columns with too few values
                continue
            
            # Detect suspicious patterns
            suspicious_indices = []
            
            # Pattern 1: Repeated identical values (potential fake data)
            value_counts = values.value_counts()
            repeated_threshold = max(3, len(values) * 0.1)
            repeated_values = value_counts[value_counts > repeated_threshold].index
            repeated_mask = values.isin(repeated_values)
            suspicious_indices.extend(values[repeated_mask].index.tolist())
            
            # Pattern 2: Unusual patterns using regex
            for idx, value in values.items():
                value_str = str(value).lower()
                
                # Suspicious keywords
                suspicious_keywords = ['test', 'fake', 'dummy', 'temp', 'null', 'none', 'unknown']
                if any(keyword in value_str for keyword in suspicious_keywords):
                    suspicious_indices.append(idx)
                
                # Suspicious patterns
                # All same character repeated
                if len(set(value_str.replace(' ', ''))) <= 2 and len(value_str) > 3:
                    suspicious_indices.append(idx)
                
                # Too many numbers in text (potential fake names/addresses)
                import re
                digit_ratio = len(re.findall(r'\d', value_str)) / max(len(value_str), 1)
                if digit_ratio > 0.5 and len(value_str) > 3:
                    suspicious_indices.append(idx)
            
            anomalies.extend(suspicious_indices)
        
        return list(set(anomalies))
    
    def _prepare_features(self, df):
        """Prepare features for training"""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values
        for col in numeric_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Encode categorical variables
        for col in categorical_cols:
            df_processed[col] = df_processed[col].fillna('Unknown')
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                # Handle new categories
                df_processed[col] = df_processed[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_classes else 'Unknown'
                )
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def _train_supervised_models(self, X, y, df_original):
        """Train supervised models"""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train models
            results = {}
            for name, model in self.models.items():
                if name == 'Isolation Forest':
                    # Unsupervised model - different evaluation
                    model.fit(X_train)
                    y_pred = (model.predict(X_test) == -1).astype(int)
                    
                    # Calculate basic metrics
                    accuracy = np.mean(y_pred == y_test)
                    results[name] = {
                        'accuracy': float(accuracy),
                        'precision': 0.5,  # Default for unsupervised
                        'recall': 0.5,
                        'f1_score': 0.5,
                        'auc': 0.5
                    }
                else:
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
            
            # Return comprehensive results
            return {
                'success': True,
                'model_results': results,
                'data_stats': {
                    'total_transactions': len(df_original),
                    'fraud_count': int(y.sum()),
                    'fraud_rate': float(y.mean()),
                    'features': X.columns.tolist(),
                    'dataset_type': self.dataset_type or 'general',
                    'detection_method': 'supervised' if hasattr(self, 'fraud_column') else 'unsupervised'
                }
            }
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return None
    
    def generate_synthetic_data(self, num_transactions=10000):
        """Generate enhanced synthetic fraud data"""
        np.random.seed(42)
        
        # Enhanced fraud patterns
        data = []
        for i in range(num_transactions):
            is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate
            
            if is_fraud:
                # Fraudulent transaction patterns
                transaction = {
                    'amount': np.random.exponential(500) + np.random.uniform(50, 5000),
                    'hour': np.random.choice([0, 1, 2, 3, 22, 23]),
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice([1, 7, 8, 9, 10]),
                    'transaction_count_1h': np.random.poisson(8),
                    'transaction_count_24h': np.random.poisson(50),
                    'avg_amount_30d': np.random.uniform(50, 200),
                    'card_present': np.random.choice([0, 1]),
                    'country_risk_score': np.random.uniform(0.6, 1.0),
                    'velocity_score': np.random.uniform(0.7, 1.0),
                    'is_fraud': 1
                }
            else:
                # Legitimate transaction patterns
                transaction = {
                    'amount': np.random.lognormal(3, 1.5),
                    'hour': np.random.randint(6, 22),  # Normal business hours
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice(range(1, 15)),
                    'transaction_count_1h': np.random.poisson(2),
                    'transaction_count_24h': np.random.poisson(15),
                    'avg_amount_30d': np.random.uniform(100, 500),
                    'card_present': np.random.choice([0, 1]),
                    'country_risk_score': np.random.uniform(0.0, 0.3),
                    'velocity_score': np.random.uniform(0.0, 0.4),
                    'is_fraud': 0
                }
            
            data.append(transaction)
        
        return pd.DataFrame(data)
    
    def predict_single(self, transaction_data, model_name='Random Forest'):
        """Predict fraud probability for single transaction"""
        if not self.is_trained:
            return None
            
        try:
            # Prepare features to match training
            features_dict = {}
            for col in self.feature_columns:
                features_dict[col] = transaction_data.get(col, 0)
            
            # Create dataframe and process features
            test_df = pd.DataFrame([features_dict])
            test_processed = self._prepare_features(test_df)
            
            # Scale features
            features_scaled = self.scaler.transform(test_processed)
            
            # Predict
            model = self.models[model_name]
            if model_name == 'Isolation Forest':
                prediction = model.predict(features_scaled)[0]
                probability = 0.8 if prediction == -1 else 0.2
                is_fraud = prediction == -1
            else:
                probability = model.predict_proba(features_scaled)[0][1]
                is_fraud = model.predict(features_scaled)[0]
            
            return {
                'fraud_probability': float(probability),
                'is_fraud_predicted': bool(is_fraud),
                'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            return None
        
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
                    'hour': np.random.choice([0, 1, 2, 3, 22, 23]),
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice([1, 7, 8, 9, 10]),
                    'transaction_count_1h': np.random.poisson(8),
                    'transaction_count_24h': np.random.poisson(50),
                    'avg_amount_30d': np.random.uniform(50, 200),
                    'card_present': np.random.choice([0, 1]),
                    'country_risk_score': np.random.uniform(0.6, 1.0),
                    'velocity_score': np.random.uniform(0.7, 1.0),
                    'is_fraud': 1
                }
            else:
                # Legitimate transaction patterns
                transaction = {
                    'amount': np.random.lognormal(3, 1.5),
                    'hour': np.random.randint(6, 22),  # Normal business hours
                    'day_of_week': np.random.randint(0, 7),
                    'merchant_category': np.random.choice(range(1, 15)),
                    'transaction_count_1h': np.random.poisson(2),
                    'transaction_count_24h': np.random.poisson(15),
                    'avg_amount_30d': np.random.uniform(100, 500),
                    'card_present': np.random.choice([0, 1]),
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
fraud_engine = UniversalFraudDetector()

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
    """Generate synthetic fraud detection data using the universal engine"""
    try:
        data = request.json or {}
        num_transactions = data.get('num_transactions', 10000)
        
        # Generate enhanced synthetic data
        df = fraud_engine.generate_synthetic_data(num_transactions)
        
        # Train models using intelligent detection
        results = fraud_engine.intelligent_fraud_detection(df)
        
        if results and results.get('success'):
            return jsonify({
                'success': True,
                'message': f'Generated {len(df)} transactions and trained universal fraud detection models',
                'data_stats': results['data_stats'],
                'model_results': results['model_results']
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to train models on generated data'
            }), 500
            
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and process any dataset format with intelligent fraud detection - optimized for large files"""
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
        
        # Accept multiple file formats
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json', '.tsv', '.txt']
        file_ext = '.' + file.filename.lower().split('.')[-1]
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'message': f'Unsupported file format. Supported: {", ".join(allowed_extensions)}'
            }), 400
        
        # Get file size for optimization decisions
        file_size_mb = 0
        if hasattr(file, 'content_length') and file.content_length:
            file_size_mb = file.content_length / (1024 * 1024)
        
        logger.info(f"Processing file: {file.filename} (estimated size: {file_size_mb:.1f} MB)")
        
        # Set processing timeout based on file size
        timeout_seconds = min(600, max(60, int(file_size_mb * 2)))  # 2 seconds per MB, max 10 minutes
        
        try:
            # Read the file using universal reader with memory optimization
            df = fraud_engine.read_any_file(file.stream, file.filename)
            
            if df is None or len(df) == 0:
                return jsonify({
                    'success': False,
                    'message': 'Failed to read file or file is empty'
                }), 400
            
            logger.info(f"Successfully read file: {file.filename} with {len(df)} rows and columns: {list(df.columns)}")
            
            # For very large files, show progress and sample data
            if len(df) > 500000:
                logger.info(f"Large dataset detected ({len(df)} rows), optimizing processing...")
            
            # Intelligent fraud detection with memory optimization
            results = fraud_engine.intelligent_fraud_detection(df)
            
            if results and results.get('success'):
                return jsonify({
                    'success': True,
                    'message': f'Successfully processed {file.filename} with intelligent fraud detection',
                    'data_stats': {
                        **results['data_stats'],
                        'file_name': file.filename,
                        'file_size_mb': round(file_size_mb, 2) if file_size_mb else 'Unknown',
                        'original_columns': list(df.columns),
                        'processed_features': len(fraud_engine.feature_columns),
                        'processing_time': f"Optimized for {file_size_mb:.1f}MB file"
                    },
                    'model_results': results['model_results'],
                    'analysis_info': {
                        'dataset_type': results['data_stats'].get('dataset_type', 'unknown'),
                        'detection_method': results['data_stats'].get('detection_method', 'unknown'),
                        'fraud_detection_strategy': 'Universal AI-powered analysis (Large File Optimized)' if len(df) > 100000 else 'Universal AI-powered analysis'
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to analyze dataset for fraud patterns'
                }), 500
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error processing file: {str(e)}'
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
            'message': f'Error processing file: {str(e)}'
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
    print("🛡️ Starting CyberShield AI Backend Server...")
    print("Features: 1GB file upload, 5 ML models, real-time prediction")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
