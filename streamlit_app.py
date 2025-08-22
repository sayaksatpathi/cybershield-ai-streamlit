import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="🛡️ CyberShield AI - Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# IMPROVED Self-contained data generator
def generate_transaction_data(n_transactions=10000):
    """Generate realistic transaction data for fraud detection training with proper fraud balance"""
    np.random.seed(42)
    
    # Generate basic transaction features
    data = {
        'transaction_amount': np.random.lognormal(3, 1.5, n_transactions),
        'account_age_days': np.random.gamma(2, 200, n_transactions),
        'previous_transactions': np.random.poisson(50, n_transactions),
        'transaction_hour': np.random.randint(0, 24, n_transactions),
        'days_since_last_transaction': np.random.exponential(2, n_transactions),
        'merchant_risk_score': np.random.beta(2, 5, n_transactions),
        'transaction_country_risk': np.random.gamma(1, 0.3, n_transactions),
        'payment_method_risk': np.random.choice([0.1, 0.3, 0.5, 0.8], n_transactions, 
                                               p=[0.4, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to reasonable ranges
    df['transaction_amount'] = np.clip(df['transaction_amount'], 1, 10000)
    df['account_age_days'] = np.clip(df['account_age_days'], 1, 3650)
    df['merchant_risk_score'] = np.clip(df['merchant_risk_score'], 0, 1)
    df['transaction_country_risk'] = np.clip(df['transaction_country_risk'], 0, 1)
    
    # IMPROVED fraud generation logic
    # Calculate risk factors
    high_amount = (df['transaction_amount'] > 2000).astype(int)
    new_account = (df['account_age_days'] < 90).astype(int)
    night_transaction = (df['transaction_hour'].isin([1, 2, 3, 4, 5])).astype(int)
    long_gap = (df['days_since_last_transaction'] > 30).astype(int)
    high_merchant_risk = (df['merchant_risk_score'] > 0.6).astype(int)
    high_country_risk = (df['transaction_country_risk'] > 0.5).astype(int)
    high_payment_risk = (df['payment_method_risk'] > 0.4).astype(int)
    
    # Create fraud probability with multiple patterns
    fraud_score = (
        high_amount * 0.25 +
        new_account * 0.2 +
        night_transaction * 0.15 +
        long_gap * 0.1 +
        high_merchant_risk * 0.3 +
        high_country_risk * 0.2 +
        high_payment_risk * 0.15
    )
    
    # Add some additional fraud patterns
    # Pattern 1: High amount + new account
    pattern1 = (high_amount & new_account).astype(int)
    # Pattern 2: Night transaction + high payment risk
    pattern2 = (night_transaction & high_payment_risk).astype(int)
    # Pattern 3: High merchant risk + high country risk
    pattern3 = (high_merchant_risk & high_country_risk).astype(int)
    
    fraud_score += pattern1 * 0.4 + pattern2 * 0.3 + pattern3 * 0.35
    
    # Normalize fraud score
    fraud_score = np.clip(fraud_score, 0, 1)
    
    # Create fraud labels with adjusted probability
    # Ensure we get roughly 5-15% fraud cases
    base_fraud_rate = 0.08  # 8% base fraud rate
    fraud_probability = base_fraud_rate + (fraud_score * 0.6)
    
    # Generate fraud labels
    df['is_fraud'] = np.random.binomial(1, fraud_probability)
    
    # Ensure minimum fraud cases (at least 3% of total)
    current_fraud_rate = df['is_fraud'].mean()
    if current_fraud_rate < 0.03:
        # Force some high-risk transactions to be fraud
        high_risk_indices = df[fraud_score > 0.7].index
        additional_fraud_needed = int(0.05 * len(df)) - df['is_fraud'].sum()
        if len(high_risk_indices) > 0 and additional_fraud_needed > 0:
            selected_indices = np.random.choice(
                high_risk_indices, 
                min(additional_fraud_needed, len(high_risk_indices)), 
                replace=False
            )
            df.loc[selected_indices, 'is_fraud'] = 1
    
    return df

# Data validation function
def validate_uploaded_data(df):
    """Validate uploaded dataset for fraud detection"""
    issues = []
    suggestions = []
    
    # Check for minimum rows
    if len(df) < 100:
        issues.append(f"Dataset too small ({len(df)} rows). Minimum 100 rows recommended.")
    
    # Check for fraud column
    fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
    if not fraud_cols:
        issues.append("No obvious fraud indicator column found.")
        suggestions.append("Ensure you have a column indicating fraud (0/1 values)")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 3:
        issues.append(f"Only {len(numeric_cols)} numeric columns found. More features recommended.")
    
    return issues, suggestions

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ CyberShield AI - Fraud Detection System</h1>
    <p>Advanced Machine Learning for Financial Security</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset & Training", "🔍 Model Testing", "📈 Analytics", "💾 Export"])

with tab1:
    st.header("📊 Dataset Generation & Model Training")
    
    # Dataset source selection
    st.subheader("📁 Choose Dataset Source")
    dataset_source = st.radio(
        "Select how you want to provide the dataset:",
        ["🎲 Generate Synthetic Data", "📤 Upload Your Own Dataset"],
        horizontal=True
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if dataset_source == "🎲 Generate Synthetic Data":
            st.subheader("Dataset Configuration")
            n_transactions = st.slider("Number of transactions", 1000, 50000, 10000, step=1000)
            
            if st.button("🎲 Generate Dataset", type="primary"):
                with st.spinner("Generating transaction data..."):
                    df = generate_transaction_data(n_transactions)
                    st.session_state['dataset'] = df
                    st.session_state['data_source'] = 'generated'
                
                # Display dataset info
                fraud_count = df['is_fraud'].sum()
                fraud_rate = df['is_fraud'].mean()
                
                st.success(f"✅ Generated {len(df)} transactions with {fraud_count} fraudulent cases ({fraud_rate:.1%} fraud rate)")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Transactions", len(df))
                with col_b:
                    st.metric("Fraud Cases", fraud_count)
                with col_c:
                    st.metric("Fraud Rate", f"{fraud_rate:.2%}")
                
                # Show fraud distribution
                if fraud_count > 0:
                    st.subheader("Fraud Pattern Analysis")
                    fraud_data = df[df['is_fraud'] == 1]
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        fig = px.histogram(fraud_data, x='transaction_amount', 
                                         title='Fraud Transaction Amounts',
                                         color_discrete_sequence=['red'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_y:
                        fig = px.histogram(fraud_data, x='transaction_hour', 
                                         title='Fraud by Hour of Day',
                                         color_discrete_sequence=['red'])
                        st.plotly_chart(fig, use_container_width=True)
        
        else:  # Upload Dataset
            st.subheader("📤 Upload Your Dataset")
            
            # File upload with drag and drop
            st.markdown("""
            <div class="upload-box">
                <h3>📁 Drag and drop your CSV file here</h3>
                <p>Or click to browse files</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type=['csv'],
                help="Upload a CSV file with transaction data and fraud indicators."
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    df = pd.read_csv(uploaded_file)
                    
                    st.success(f"✅ Successfully loaded {len(df)} transactions from {uploaded_file.name}")
                    
                    # Validate the data
                    issues, suggestions = validate_uploaded_data(df)
                    
                    if issues:
                        st.warning("⚠️ Data Validation Issues Found:")
                        for issue in issues:
                            st.write(f"• {issue}")
                        
                        if suggestions:
                            st.info("💡 Suggestions:")
                            for suggestion in suggestions:
                                st.write(f"• {suggestion}")
                    
                    # Display basic info
                    st.subheader("📊 Dataset Overview")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Rows", len(df))
                    with col_b:
                        st.metric("Total Columns", len(df.columns))
                    with col_c:
                        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    # Show column information
                    st.subheader("📋 Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Null Count': df.isnull().sum(),
                        'Sample Values': [str(df[col].head(3).tolist()) for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                    # Check for fraud column
                    fraud_columns = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower() or 'target' in col.lower()]
                    
                    if fraud_columns:
                        st.success(f"🎯 Potential fraud columns detected: {', '.join(fraud_columns)}")
                        
                        fraud_col = st.selectbox(
                            "Select the fraud indicator column:",
                            fraud_columns,
                            help="Choose the column that indicates whether a transaction is fraudulent (1) or legitimate (0)"
                        )
                    else:
                        st.warning("⚠️ No obvious fraud column detected.")
                        fraud_col = st.selectbox(
                            "Select the fraud indicator column:",
                            df.columns.tolist(),
                            help="Choose the column that indicates fraud"
                        )
                    
                    if fraud_col in df.columns:
                        # Analyze fraud column
                        unique_values = sorted(df[fraud_col].unique())
                        st.write(f"**Unique values in '{fraud_col}':** {unique_values}")
                        
                        # Try to convert to binary
                        if len(unique_values) == 2:
                            # Binary column - map to 0/1
                            val0, val1 = unique_values
                            
                            st.subheader("🔄 Binary Mapping")
                            fraud_value = st.selectbox(
                                f"Which value represents FRAUD?",
                                unique_values,
                                index=1,  # Default to second value
                                help="Select the value that indicates a fraudulent transaction"
                            )
                            
                            if st.button("✅ Process Dataset"):
                                # Create processed dataset
                                df_processed = df.copy()
                                df_processed['is_fraud'] = (df_processed[fraud_col] == fraud_value).astype(int)
                                
                                # Remove original fraud column if different name
                                if fraud_col != 'is_fraud':
                                    df_processed = df_processed.drop(columns=[fraud_col])
                                
                                # Handle missing values
                                df_processed = df_processed.fillna(df_processed.mean())
                                
                                # Store processed dataset
                                st.session_state['dataset'] = df_processed
                                st.session_state['data_source'] = 'uploaded'
                                
                                fraud_count = df_processed['is_fraud'].sum()
                                fraud_rate = fraud_count / len(df_processed)
                                
                                st.success(f"🎯 Dataset processed successfully!")
                                
                                col_x, col_y, col_z = st.columns(3)
                                with col_x:
                                    st.metric("Total Transactions", len(df_processed))
                                with col_y:
                                    st.metric("Fraud Cases", fraud_count)
                                with col_z:
                                    st.metric("Fraud Rate", f"{fraud_rate:.2%}")
                                
                                if fraud_rate < 0.01:
                                    st.warning("⚠️ Very low fraud rate detected. Model training might be challenging.")
                                elif fraud_rate > 0.5:
                                    st.warning("⚠️ Very high fraud rate detected. Please verify your data.")
                                else:
                                    st.success("✅ Fraud rate looks reasonable for training!")
                        
                        else:
                            # Multi-class column - need mapping
                            st.subheader("🔄 Multi-class Mapping")
                            st.write("Map each value to either Legitimate (0) or Fraud (1):")
                            
                            mapping = {}
                            for val in unique_values:
                                mapping[val] = st.selectbox(
                                    f"Value '{val}' →",
                                    [0, 1],
                                    format_func=lambda x: "Legitimate (0)" if x == 0 else "Fraud (1)",
                                    key=f"map_{val}"
                                )
                            
                            if st.button("✅ Apply Mapping"):
                                df_processed = df.copy()
                                df_processed['is_fraud'] = df_processed[fraud_col].map(mapping).fillna(0).astype(int)
                                
                                if fraud_col != 'is_fraud':
                                    df_processed = df_processed.drop(columns=[fraud_col])
                                
                                # Handle missing values
                                df_processed = df_processed.fillna(df_processed.mean())
                                
                                st.session_state['dataset'] = df_processed
                                st.session_state['data_source'] = 'uploaded'
                                
                                fraud_count = df_processed['is_fraud'].sum()
                                fraud_rate = fraud_count / len(df_processed)
                                
                                st.success(f"🎯 Dataset processed! {fraud_count} fraud cases ({fraud_rate:.2%} fraud rate)")
                    
                    # Show data preview
                    if st.checkbox("📊 Show Data Preview"):
                        st.subheader("Data Sample (First 100 rows)")
                        st.dataframe(df.head(100), use_container_width=True)
                        
                        # Statistical summary
                        if st.checkbox("📈 Show Statistical Summary"):
                            st.subheader("Statistical Summary")
                            st.dataframe(df.describe(), use_container_width=True)
                            
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    st.info("💡 Make sure your file is a valid CSV with proper formatting.")
            
            else:
                st.info("📁 Please upload a CSV file to get started")
                
                # Show expected format
                st.subheader("📋 Expected File Format")
                st.write("Your CSV file should contain:")
                st.write("• **Transaction features** (amount, time, account info, etc.)")
                st.write("• **Fraud indicator column** with values indicating fraud vs legitimate")
                
                # Example data format
                example_data = pd.DataFrame({
                    'transaction_amount': [100.50, 2500.00, 50.00, 5000.00],
                    'account_age_days': [365, 30, 1000, 15],
                    'transaction_hour': [14, 2, 10, 3],
                    'merchant_risk_score': [0.2, 0.8, 0.1, 0.9],
                    'is_fraud': [0, 1, 0, 1]
                })
                
                st.write("**Example format:**")
                st.dataframe(example_data, use_container_width=True)
    
    with col2:
        st.subheader("Model Training")
        
        if 'dataset' in st.session_state:
            df = st.session_state['dataset']
            fraud_count = df['is_fraud'].sum()
            
            # Show current dataset info
            data_source = st.session_state.get('data_source', 'unknown')
            if data_source == 'uploaded':
                st.info(f"📤 Using uploaded dataset: {len(df)} transactions, {fraud_count} fraud cases")
            else:
                st.info(f"🎲 Using generated dataset: {len(df)} transactions, {fraud_count} fraud cases")
            
            if fraud_count < 10:
                st.warning(f"⚠️ Only {fraud_count} fraud cases detected. Consider using more data for better training.")
            
            algorithm = st.selectbox(
                "Select Algorithm",
                ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"]
            )
            
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, step=0.05)
            
            if st.button("🚀 Train Model", type="primary"):
                if fraud_count < 5:
                    st.error("❌ Not enough fraud cases to train. Please use a dataset with more fraud cases.")
                else:
                    # Prepare features and target
                    X = df.drop(['is_fraud'], axis=1)
                    y = df['is_fraud']
                    
                    # Handle non-numeric columns
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) < len(X.columns):
                        st.warning("⚠️ Non-numeric columns detected. Using only numeric columns for training.")
                        X = X[numeric_cols]
                    
                    if X.empty:
                        st.error("❌ No numeric features found for training.")
                    else:
                        try:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42, stratify=y
                            )
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Train model
                            with st.spinner(f"Training {algorithm} model..."):
                                if algorithm == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                                    model.fit(X_train, y_train)
                                elif algorithm == "Gradient Boosting":
                                    model = GradientBoostingClassifier(random_state=42)
                                    model.fit(X_train, y_train)
                                elif algorithm == "Logistic Regression":
                                    model = LogisticRegression(random_state=42, class_weight='balanced')
                                    model.fit(X_train_scaled, y_train)
                                else:  # SVM
                                    model = SVC(probability=True, random_state=42, class_weight='balanced')
                                    model.fit(X_train_scaled, y_train)
                            
                            # Make predictions
                            if algorithm in ["Logistic Regression", "SVM"]:
                                y_pred = model.predict(X_test_scaled)
                                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                            else:
                                y_pred = model.predict(X_test)
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                            
                            # Calculate metrics
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            auc = roc_auc_score(y_test, y_pred_proba)
                            
                            # Store results
                            st.session_state['model'] = model
                            st.session_state['scaler'] = scaler
                            st.session_state['algorithm'] = algorithm
                            st.session_state['feature_columns'] = X.columns.tolist()
                            st.session_state['metrics'] = {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'auc': auc
                            }
                            st.session_state['test_data'] = (X_test, y_test, y_pred, y_pred_proba)
                            
                            st.success(f"✅ {algorithm} model trained successfully!")
                            
                            # Display metrics
                            col_a, col_b, col_c, col_d, col_e = st.columns(5)
                            with col_a:
                                st.metric("Accuracy", f"{accuracy:.3f}")
                            with col_b:
                                st.metric("Precision", f"{precision:.3f}")
                            with col_c:
                                st.metric("Recall", f"{recall:.3f}")
                            with col_d:
                                st.metric("F1-Score", f"{f1:.3f}")
                            with col_e:
                                st.metric("AUC", f"{auc:.3f}")
                            
                            # Show class distribution
                            st.info(f"📊 Test set: {(y_test == 0).sum()} legitimate, {(y_test == 1).sum()} fraud | Predicted: {(y_pred == 0).sum()} legitimate, {(y_pred == 1).sum()} fraud")
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {str(e)}")
                            st.info("💡 This might be due to insufficient data or data format issues.")
        else:
            st.info("Please generate or upload a dataset first!")

with tab2:
    st.header("🔍 Model Testing")
    
    if 'model' in st.session_state:
        st.subheader("Test Individual Transaction")
        
        feature_columns = st.session_state.get('feature_columns', [])
        
        if feature_columns:
            st.write(f"**Required features:** {', '.join(feature_columns)}")
            
            # Create input fields based on feature columns
            test_inputs = {}
            
            # Organize inputs in columns
            num_cols = min(3, len(feature_columns))
            cols = st.columns(num_cols)
            
            for i, feature in enumerate(feature_columns):
                col_idx = i % num_cols
                with cols[col_idx]:
                    if 'amount' in feature.lower():
                        test_inputs[feature] = st.number_input(f"{feature}", 1.0, 10000.0, 100.0, key=f"test_{feature}")
                    elif 'age' in feature.lower() or 'days' in feature.lower():
                        test_inputs[feature] = st.number_input(f"{feature}", 1, 3650, 365, key=f"test_{feature}")
                    elif 'hour' in feature.lower():
                        test_inputs[feature] = st.number_input(f"{feature}", 0, 23, 14, key=f"test_{feature}")
                    elif 'risk' in feature.lower() or 'score' in feature.lower():
                        test_inputs[feature] = st.slider(f"{feature}", 0.0, 1.0, 0.3, key=f"test_{feature}")
                    else:
                        test_inputs[feature] = st.number_input(f"{feature}", 0.0, 1000.0, 50.0, key=f"test_{feature}")
            
            if st.button("🔍 Predict Fraud Risk", type="primary"):
                # Create test sample
                test_sample = pd.DataFrame([test_inputs])
                
                # Make prediction
                algorithm = st.session_state['algorithm']
                model = st.session_state['model']
                
                try:
                    if algorithm in ["Logistic Regression", "SVM"]:
                        scaler = st.session_state['scaler']
                        test_sample_scaled = scaler.transform(test_sample)
                        fraud_probability = model.predict_proba(test_sample_scaled)[0][1]
                        prediction = model.predict(test_sample_scaled)[0]
                    else:
                        fraud_probability = model.predict_proba(test_sample)[0][1]
                        prediction = model.predict(test_sample)[0]
                    
                    # Display result
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if prediction == 1:
                            st.error(f"🚨 FRAUD DETECTED! Risk: {fraud_probability:.1%}")
                        else:
                            st.success(f"✅ Transaction appears legitimate. Risk: {fraud_probability:.1%}")
                    
                    with col_b:
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = fraud_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Fraud Risk %"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if fraud_probability > 0.5 else "darkgreen"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50}}))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.warning("⚠️ No feature information available. Please retrain the model.")
    else:
        st.info("Please train a model first in the Dataset & Training tab!")

with tab3:
    st.header("📈 Model Analytics")
    
    if 'metrics' in st.session_state:
        # Performance metrics
        st.subheader("Model Performance")
        metrics = st.session_state['metrics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <h2>{metrics['accuracy']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Precision</h3>
                <h2>{metrics['precision']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recall</h3>
                <h2>{metrics['recall']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>F1-Score</h3>
                <h2>{metrics['f1']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>AUC</h3>
                <h2>{metrics['auc']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # ROC Curve and Confusion Matrix
        if 'test_data' in st.session_state:
            X_test, y_test, y_pred, y_pred_proba = st.session_state['test_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {metrics["auc"]:.3f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                fig.update_layout(
                    title='Receiver Operating Characteristic',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Legitimate', 'Fraud'],
                              y=['Legitimate', 'Fraud'])
                fig.update_layout(title='Confusion Matrix', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed metrics
                st.subheader("Detailed Analysis")
                st.write(f"**True Negatives (Legitimate → Legitimate):** {cm[0,0]}")
                st.write(f"**False Positives (Legitimate → Fraud):** {cm[0,1]}")
                st.write(f"**False Negatives (Fraud → Legitimate):** {cm[1,0]}")
                st.write(f"**True Positives (Fraud → Fraud):** {cm[1,1]}")
        
        # Dataset info
        if 'dataset' in st.session_state:
            df = st.session_state['dataset']
            data_source = st.session_state.get('data_source', 'unknown')
            
            st.subheader("Dataset Information")
            st.write(f"**Data Source:** {'Uploaded File' if data_source == 'uploaded' else 'Generated Synthetic Data'}")
            st.write(f"**Total Transactions:** {len(df)}")
            st.write(f"**Features Used:** {len(df.columns) - 1}")  # Excluding is_fraud column
            st.write(f"**Fraud Rate:** {df['is_fraud'].mean():.2%}")
            
    else:
        st.info("Please train a model first to view analytics!")

with tab4:
    st.header("💾 Model Export")
    
    if 'model' in st.session_state:
        st.subheader("Download Trained Model")
        
        model = st.session_state['model']
        algorithm = st.session_state['algorithm']
        data_source = st.session_state.get('data_source', 'unknown')
        
        # Create model info
        model_info = {
            'algorithm': algorithm,
            'data_source': data_source,
            'feature_columns': st.session_state.get('feature_columns', []),
            'metrics': st.session_state['metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model to bytes
        model_bytes = io.BytesIO()
        joblib.dump({
            'model': model,
            'scaler': st.session_state.get('scaler'),
            'info': model_info
        }, model_bytes)
        model_bytes.seek(0)
        
        st.download_button(
            label="📥 Download Model",
            data=model_bytes.getvalue(),
            file_name=f"cybershield_ai_{algorithm.lower().replace(' ', '_')}_model.pkl",
            mime="application/octet-stream"
        )
        
        st.subheader("Model Summary")
        st.json(model_info)
        
        # Export dataset option
        if 'dataset' in st.session_state:
            st.subheader("Download Dataset")
            df = st.session_state['dataset']
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Dataset (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"fraud_detection_dataset_{data_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    else:
        st.info("Please train a model first!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛡️ CyberShield AI - Advanced Fraud Detection System</p>
    <p>Powered by Machine Learning | Built with Streamlit | Now with Custom Dataset Support</p>
</div>
""", unsafe_allow_html=True)
