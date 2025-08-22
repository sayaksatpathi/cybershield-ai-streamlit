#!/usr/bin/env python3
"""
Quick Model Training Script for CyberShield AI
Uses existing transaction data to train a fresh model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_fraud_model():
    print("🚀 Starting CyberShield AI Model Training...")
    
    # Load transaction data
    print("📊 Loading transaction data...")
    df = pd.read_csv('transaction_data.csv')
    print(f"✅ Loaded {len(df)} transactions")
    
    # Feature engineering
    print("🔧 Engineering features...")
    
    # Convert categorical variables
    le_merchant = LabelEncoder()
    df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
    
    # Time-based features
    df['transaction_hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Select features for training (using only available columns)
    feature_columns = [
        'amount', 'merchant_category_encoded', 'transaction_hour', 
        'day_of_week', 'is_weekend', 'latitude', 'longitude'
    ]
    
    # Prepare data
    X = df[feature_columns]
    y = df['is_fraud']
    
    print(f"📈 Features: {len(feature_columns)}")
    print(f"🎯 Fraud rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("🧠 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.3f}")
    
    # Save model and components
    print("💾 Saving model files...")
    
    # Save model
    joblib.dump(model, 'enhanced_fraud_detection_model.pkl')
    print("✅ Saved enhanced_fraud_detection_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'enhanced_fraud_detection_scaler.pkl')
    print("✅ Saved enhanced_fraud_detection_scaler.pkl")
    
    # Save metadata (in expected format for enhanced_prediction_interface)
    metadata = {
        'model_name': 'Enhanced Random Forest Fraud Detector',
        'feature_columns': feature_columns,  # This is the key the interface expects
        'features': feature_columns,  # Keep backward compatibility
        'accuracy': accuracy,
        'training_samples': len(X_train),
        'feature_count': len(feature_columns),
        'fraud_rate': float(y.mean()),
        'model_type': 'RandomForest',
        'version': '1.0'
    }
    
    joblib.dump(metadata, 'enhanced_fraud_detection_model_metadata.pkl')
    print("✅ Saved enhanced_fraud_detection_model_metadata.pkl")
    
    print(f"\n🎉 Model training completed!")
    print(f"📊 Model accuracy: {accuracy:.3f}")
    print(f"🎯 Features used: {len(feature_columns)}")
    print(f"💾 All model files saved successfully")
    
    return model, scaler, metadata

if __name__ == "__main__":
    try:
        model, scaler, metadata = train_fraud_model()
        print("\n✅ Training successful!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
