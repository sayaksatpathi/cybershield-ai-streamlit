
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import json
import time
import os

def load_datasets():
    """Loads and combines all generated transaction datasets."""
    print("🔄 Loading and combining datasets...")
    
    datasets = []
    
    # Load all enhanced dataset files
    dataset_files = [
        'enhanced_transactions_small.csv',
        'enhanced_transactions_medium.csv', 
        'enhanced_transactions_large.csv',
        'enhanced_transactions.csv'
    ]
    
    for filename in dataset_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                datasets.append(df)
                print(f"✅ Loaded {filename} with {len(df)} records.")
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
            
    if not datasets:
        print("❌ No datasets found. Please run the enhanced_data_generator.py first.")
        return None

    # Combine all loaded datasets
    df = pd.concat(datasets, ignore_index=True)
    print(f"📊 Combined dataset has {len(df)} total records.")
    
    return df

def engineer_features(df):
    """Creates advanced features for the model."""
    print("🛠️  Engineering advanced features...")
    
    # Clean data first - handle missing values and infinites
    df = df.fillna(0)
    
    # Time-based features (using the correct column name 'timestamp')
    df['transaction_date'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['transaction_hour']  # Already provided
    df['day_of_week'] = df['day_of_week']       # Already provided
    df['is_weekend'] = df['is_weekend']         # Already provided
    df['is_night'] = ((df['hour_of_day'] <= 6) | (df['hour_of_day'] >= 22)).astype(int)

    # Amount-based features  
    df['amount_log'] = np.log1p(np.abs(df['amount']))
    df['amount_sqrt'] = np.sqrt(np.abs(df['amount']))
    
    # Risk composite features
    df['location_device_risk'] = (df['location_risk_score'] + (1 - df['device_trust_score'])) / 2
    df['total_risk_score'] = (df['location_risk_score'] + df['time_risk_score'] + df['amount_risk_score']) / 3
    
    # Time since last transaction features (handle zero and negative values)
    df['time_since_last_transaction'] = np.maximum(df['time_since_last_transaction'], 0.001)
    df['time_since_last_log'] = np.log1p(df['time_since_last_transaction'])
    df['transaction_velocity'] = 1 / (df['time_since_last_transaction'] + 0.1)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)
    
    # Final cleanup - replace any remaining inf or nan values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    print("✅ Feature engineering complete.")
    return df

def train_model():
    """Loads data, engineers features, and trains an XGBoost model."""
    
    df = load_datasets()
    if df is None:
        return

    df = engineer_features(df)

    # Define features and target
    target = 'is_fraud'
    exclude_cols = ['transaction_id', 'customer_id', 'timestamp', 'transaction_date', 'fraud_type', 'profile_type', target]
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure all feature columns are numeric
    for col in features:
        if df[col].dtype == 'object':
            print(f"Warning: Feature '{col}' has non-numeric data. Attempting to convert.")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Split data into {len(X_train)} training and {len(X_test)} testing records.")

    # Train XGBoost model
    print("\n🚀 Training XGBoost model...")
    start_time = time.time()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"✅ Model training completed in {training_time:.2f} seconds.")

    # Evaluate model
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"🎯 Recall: {recall:.4f}")
    print(f"🎯 F1-Score: {f1:.4f}")
    print(f"🎯 ROC AUC Score: {roc_auc:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("📋 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and metadata
    print("\n💾 Saving model and metadata...")
    model_filename = 'enhanced_fraud_model.joblib'
    metadata_filename = 'enhanced_model_metadata.json'

    joblib.dump(model, model_filename)
    print(f"✅ Model saved to {model_filename}")

    metadata = {
        'model_type': 'XGBoost',
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': round(training_time, 2),
        'performance_metrics': {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'roc_auc_score': round(roc_auc, 4)
        },
        'features': features,
        'total_records_trained': len(df)
    }

    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to {metadata_filename}")
    
    print("\n🎉 Enhanced model training process complete!")

if __name__ == "__main__":
    train_model()
