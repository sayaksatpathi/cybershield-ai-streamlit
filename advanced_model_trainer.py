import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFraudDetector:
    """
    Advanced fraud detection system with multiple models, feature engineering,
    and comprehensive evaluation capabilities.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.best_model = None
        self.best_model_name = None
        
        # Define model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
    
    def load_and_prepare_data(self, transaction_file: str, customer_file: str = None) -> pd.DataFrame:
        """Load and prepare transaction data for training."""
        print(f"📊 Loading data from {transaction_file}...")
        
        # Load transaction data
        df = pd.read_csv(transaction_file)
        
        # Load customer data if provided
        if customer_file and pd.io.common.file_exists(customer_file):
            customers = pd.read_csv(customer_file)
            print(f"👥 Merging with customer data from {customer_file}...")
            df = df.merge(customers, on='customer_id', how='left')
        
        print(f"✅ Loaded {len(df):,} transactions")
        print(f"🚨 Fraud rate: {df['is_fraud'].mean():.2%}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for fraud detection."""
        print("🔧 Engineering features...")
        
        df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
            df['is_holiday_season'] = ((df['timestamp'].dt.month == 12) | 
                                     (df['timestamp'].dt.month == 1)).astype(int)
        
        # Amount-based features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_squared'] = df['amount'] ** 2
            df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
            df['is_very_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        
        # Customer behavior features (if customer data available)
        if 'avg_transaction_amount' in df.columns:
            df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_transaction_amount'] + 1)
            df['amount_deviation'] = abs(df['amount'] - df['avg_transaction_amount'])
            df['is_high_amount'] = (df['amount'] > df['avg_transaction_amount'] * 3).astype(int)
            df['is_low_amount'] = (df['amount'] < df['avg_transaction_amount'] * 0.1).astype(int)
        
        # Location-based features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Simple distance calculation (could be enhanced with actual geographic distance)
            if 'home_latitude' in df.columns:
                df['distance_from_home'] = np.sqrt(
                    (df['latitude'] - df['home_latitude'])**2 + 
                    (df['longitude'] - df['home_longitude'])**2
                )
                df['is_far_from_home'] = (df['distance_from_home'] > 0.1).astype(int)
            else:
                # Use simple location clusters
                df['location_cluster'] = (df['latitude'].round(1).astype(str) + '_' + 
                                        df['longitude'].round(1).astype(str))
        
        # Risk score features
        risk_columns = [col for col in df.columns if 'risk_score' in col]
        if risk_columns:
            df['composite_risk_score'] = df[risk_columns].mean(axis=1)
            df['max_risk_score'] = df[risk_columns].max(axis=1)
            df['high_risk_flag'] = (df['composite_risk_score'] > 0.7).astype(int)
        
        # Merchant category features
        if 'merchant_category' in df.columns:
            # Create binary features for high-risk categories
            high_risk_categories = ['online', 'atm', 'travel']
            df['is_high_risk_category'] = df['merchant_category'].isin(high_risk_categories).astype(int)
            
            # Category frequency (how common this category is for the customer)
            category_counts = df.groupby(['customer_id', 'merchant_category']).size()
            df['category_frequency'] = df.apply(
                lambda row: category_counts.get((row['customer_id'], row['merchant_category']), 0),
                axis=1
            )
        
        # Time-based risk features
        if 'transaction_hour' in df.columns:
            df['is_night_transaction'] = ((df['transaction_hour'] < 6) | 
                                        (df['transaction_hour'] > 22)).astype(int)
            df['is_business_hours'] = ((df['transaction_hour'] >= 9) & 
                                     (df['transaction_hour'] <= 17)).astype(int)
        
        # Account age features
        if 'account_age_days' in df.columns:
            df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
            df['is_very_new_account'] = (df['account_age_days'] < 30).astype(int)
            df['account_age_months'] = df['account_age_days'] / 30
        
        # Time since last transaction features
        if 'time_since_last_transaction' in df.columns:
            df['is_rapid_transaction'] = (df['time_since_last_transaction'] < 0.5).astype(int)
            df['is_very_rapid_transaction'] = (df['time_since_last_transaction'] < 0.1).astype(int)
            df['time_gap_log'] = np.log1p(df['time_since_last_transaction'])
        
        # Device and security features
        if 'device_trust_score' in df.columns:
            df['is_low_trust_device'] = (df['device_trust_score'] < 0.3).astype(int)
            df['is_high_trust_device'] = (df['device_trust_score'] > 0.8).astype(int)
        
        print(f"✅ Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training."""
        print("🎯 Preparing features for training...")
        
        # Define feature columns (exclude target and ID columns)
        exclude_columns = [
            'transaction_id', 'customer_id', 'timestamp', 'is_fraud',
            'fraud_type'  # If present from enhanced generator
        ]
        
        # Also exclude text/object columns that need encoding
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        exclude_columns.extend([col for col in categorical_columns if col not in ['merchant_category']])
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle categorical variables
        df_processed = df.copy()
        
        # Encode merchant category if present
        if 'merchant_category' in df_processed.columns:
            if 'merchant_category' not in self.feature_encoders:
                self.feature_encoders['merchant_category'] = LabelEncoder()
                df_processed['merchant_category_encoded'] = self.feature_encoders['merchant_category'].fit_transform(
                    df_processed['merchant_category']
                )
            else:
                df_processed['merchant_category_encoded'] = self.feature_encoders['merchant_category'].transform(
                    df_processed['merchant_category']
                )
            feature_columns.append('merchant_category_encoded')
            feature_columns.remove('merchant_category')
        
        # Handle other categorical columns
        for col in categorical_columns:
            if col in df_processed.columns and col != 'merchant_category':
                if col not in self.feature_encoders:
                    self.feature_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.feature_encoders[col].fit_transform(
                        df_processed[col].astype(str)
                    )
                else:
                    df_processed[f'{col}_encoded'] = self.feature_encoders[col].transform(
                        df_processed[col].astype(str)
                    )
                feature_columns.append(f'{col}_encoded')
        
        # Select final features
        X = df_processed[feature_columns].copy()
        y = df_processed['is_fraud'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"✅ Prepared {len(feature_columns)} features for {len(X)} samples")
        print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns
    
    def train_multiple_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Train multiple models and find the best performer."""
        print("🚀 Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")
        
        best_score = 0
        
        for name, config in self.model_configs.items():
            print(f"\n🎯 Training {name}...")
            
            try:
                # Scale features for models that need it
                if name in ['logistic_regression', 'svm', 'neural_network']:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    self.scalers[name] = scaler
                else:
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                
                # Grid search for best parameters
                print(f"   🔍 Performing grid search...")
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=3, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                self.models[name] = best_model
                
                # Evaluate model
                y_pred = best_model.predict(X_test_scaled)
                y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                          cv=5, scoring='roc_auc', n_jobs=-1)
                
                self.performance_metrics[name] = {
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': grid_search.best_params_,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"   ✅ AUC Score: {auc_score:.4f}")
                print(f"   📊 CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"   🎯 Best params: {grid_search.best_params_}")
                
                # Track best model
                if auc_score > best_score:
                    best_score = auc_score
                    self.best_model = best_model
                    self.best_model_name = name
                
                # Feature importance (if available)
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, best_model.feature_importances_))
                elif hasattr(best_model, 'coef_'):
                    self.feature_importance[name] = dict(zip(X.columns, abs(best_model.coef_[0])))
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
                continue
        
        print(f"\n🏆 Best model: {self.best_model_name} (AUC: {best_score:.4f})")
        
        return X_test, y_test
    
    def evaluate_model_performance(self):
        """Generate comprehensive performance evaluation."""
        print("\n📊 Model Performance Summary:")
        print("=" * 60)
        
        for name, metrics in self.performance_metrics.items():
            print(f"\n🎯 {name.upper()}:")
            print(f"   AUC Score: {metrics['auc_score']:.4f}")
            print(f"   CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            print(f"   Best Parameters: {metrics['best_params']}")
            
            # Confusion Matrix
            cm = metrics['confusion_matrix']
            print(f"   Confusion Matrix:")
            print(f"   True Negative:  {cm[0][0]:,}")
            print(f"   False Positive: {cm[0][1]:,}")
            print(f"   False Negative: {cm[1][0]:,}")
            print(f"   True Positive:  {cm[1][1]:,}")
            
            # Calculate precision, recall, F1
            precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
            recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
    
    def save_best_model(self, filepath: str = 'best_fraud_model.pkl'):
        """Save the best performing model."""
        if self.best_model is None:
            print("❌ No model trained yet!")
            return
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scalers.get(self.best_model_name),
            'feature_encoders': self.feature_encoders,
            'feature_importance': self.feature_importance.get(self.best_model_name, {}),
            'performance_metrics': self.performance_metrics[self.best_model_name],
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Best model ({self.best_model_name}) saved to {filepath}")
        
        # Save feature importance separately
        if self.feature_importance.get(self.best_model_name):
            feature_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in self.feature_importance[self.best_model_name].items()
            ]).sort_values('importance', ascending=False)
            
            feature_df.to_csv(filepath.replace('.pkl', '_feature_importance.csv'), index=False)
            print(f"📊 Feature importance saved to {filepath.replace('.pkl', '_feature_importance.csv')}")
    
    def predict_transaction(self, transaction_data: dict) -> dict:
        """Predict fraud probability for a single transaction."""
        if self.best_model is None:
            return {"error": "No model trained"}
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features
        X, _, _ = self.prepare_features(df)
        
        # Apply scaling if needed
        if self.best_model_name in self.scalers:
            X = self.scalers[self.best_model_name].transform(X)
        
        # Make prediction
        fraud_probability = self.best_model.predict_proba(X)[0][1]
        is_fraud = fraud_probability > 0.5
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = "LOW"
        elif fraud_probability < 0.6:
            risk_level = "MEDIUM"
        elif fraud_probability < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'fraud_probability': round(fraud_probability, 4),
            'is_fraud': bool(is_fraud),
            'risk_level': risk_level,
            'model_used': self.best_model_name,
            'confidence': round(max(fraud_probability, 1 - fraud_probability), 4)
        }

# Training script
if __name__ == "__main__":
    # Initialize detector
    detector = AdvancedFraudDetector()
    
    # Load and prepare data
    print("🚀 Starting Advanced Fraud Detection Training...")
    
    # Try to load enhanced dataset first, then fall back to regular dataset
    try:
        df = detector.load_and_prepare_data(
            'enhanced_transactions_medium.csv', 
            'enhanced_customers_medium.csv'
        )
    except FileNotFoundError:
        try:
            df = detector.load_and_prepare_data(
                'transaction_data.csv', 
                'customer_profiles_generated.csv'
            )
        except FileNotFoundError:
            print("❌ No dataset found! Please generate data first.")
            exit(1)
    
    # Engineer features
    df = detector.engineer_features(df)
    
    # Prepare features
    X, y, feature_columns = detector.prepare_features(df)
    
    # Train models
    X_test, y_test = detector.train_multiple_models(X, y)
    
    # Evaluate performance
    detector.evaluate_model_performance()
    
    # Save best model
    detector.save_best_model('advanced_fraud_model.pkl')
    
    print("\n🎉 Training complete!")
    print(f"🏆 Best model: {detector.best_model_name}")
    print(f"📊 AUC Score: {detector.performance_metrics[detector.best_model_name]['auc_score']:.4f}")
    
    # Test with sample transaction
    print("\n🧪 Testing with sample transaction...")
    sample_transaction = {
        'amount': 1500.0,
        'merchant_category': 'online',
        'transaction_hour': 2,
        'day_of_week': 6,
        'is_weekend': True,
        'account_age_days': 30,
        'location_risk_score': 0.8,
        'device_trust_score': 0.3,
        'time_since_last_transaction': 0.1
    }
    
    result = detector.predict_transaction(sample_transaction)
    print(f"🎯 Prediction result: {result}")
