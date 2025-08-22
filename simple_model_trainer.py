import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import os
from datetime import datetime
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
    page_title="🛡️ CyberShield AI - Model Trainer",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ CyberShield AI - Advanced Model Trainer</h1>
    <p>Train, Test, and Deploy Fraud Detection Models with Multiple Datasets</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}

# Data generation function
def generate_sample_data(n_samples=1000, fraud_rate=0.05):
    """Generate sample transaction data for demonstration"""
    np.random.seed(42)
    
    # Normal transactions
    normal_samples = int(n_samples * (1 - fraud_rate))
    fraud_samples = n_samples - normal_samples
    
    # Generate normal transactions
    normal_data = {
        'amount': np.random.lognormal(mean=4, sigma=1, size=normal_samples),
        'transaction_hour': np.random.choice(range(6, 22), size=normal_samples),  # Daytime bias
        'day_of_week': np.random.choice(range(7), size=normal_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], size=normal_samples),
        'account_age_days': np.random.normal(500, 200, size=normal_samples),
        'previous_transactions': np.random.poisson(50, size=normal_samples),
        'location_risk_score': np.random.beta(2, 8, size=normal_samples),  # Low risk bias
        'device_trust_score': np.random.beta(8, 2, size=normal_samples),  # High trust bias
        'is_fraud': [0] * normal_samples
    }
    
    # Generate fraud transactions
    fraud_data = {
        'amount': np.random.lognormal(mean=6, sigma=1.5, size=fraud_samples),  # Higher amounts
        'transaction_hour': np.random.choice(list(range(0, 6)) + list(range(22, 24)), size=fraud_samples),  # Night bias
        'day_of_week': np.random.choice(range(7), size=fraud_samples),
        'merchant_category': np.random.choice(['online', 'atm', 'gas', 'retail'], size=fraud_samples),
        'account_age_days': np.random.normal(100, 50, size=fraud_samples),  # Newer accounts
        'previous_transactions': np.random.poisson(10, size=fraud_samples),  # Fewer transactions
        'location_risk_score': np.random.beta(8, 2, size=fraud_samples),  # High risk bias
        'device_trust_score': np.random.beta(2, 8, size=fraud_samples),  # Low trust bias
        'is_fraud': [1] * fraud_samples
    }
    
    # Combine data
    all_data = {}
    for key in normal_data.keys():
        all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Add derived features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['transaction_hour'] < 6) | (df['transaction_hour'] > 22)).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    df['risk_composite'] = df['location_risk_score'] * (1 - df['device_trust_score'])
    
    # Encode categorical variables
    df['merchant_category_encoded'] = pd.Categorical(df['merchant_category']).codes
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Management", 
    "🔧 Model Training", 
    "📈 Performance Analysis", 
    "🎯 Model Testing", 
    "🚀 Deployment"
])

# Tab 1: Dataset Management
with tab1:
    st.header("📊 Dataset Management")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔄 Generate Sample Dataset")
        
        with st.expander("Dataset Generation Parameters"):
            n_samples = st.slider("Number of Samples", 500, 10000, 2000)
            fraud_rate = st.slider("Fraud Rate (%)", 1.0, 20.0, 5.0, 0.1) / 100
            
            dataset_name = st.text_input("Dataset Name", f"Dataset_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        if st.button("🚀 Generate Dataset", type="primary"):
            with st.spinner("Generating dataset..."):
                try:
                    df = generate_sample_data(n_samples, fraud_rate)
                    
                    # Store in session state
                    st.session_state.datasets[dataset_name] = {
                        'data': df,
                        'params': {
                            'samples': n_samples,
                            'fraud_rate': fraud_rate,
                            'generated_at': datetime.now()
                        }
                    }
                    
                    st.success(f"✅ Generated dataset: {dataset_name}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error generating dataset: {str(e)}")
    
    with col2:
        st.subheader("📁 Upload Custom Dataset")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload your own transaction dataset for training"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = f"Uploaded_{uploaded_file.name.split('.')[0]}"
                
                st.write("Dataset Preview:")
                st.dataframe(df.head())
                
                # Basic validation
                required_cols = ['amount', 'is_fraud']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("📥 Load Dataset"):
                        st.session_state.datasets[dataset_name] = {
                            'data': df,
                            'params': {'source': 'uploaded'}
                        }
                        st.success(f"✅ Loaded dataset: {dataset_name}")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Display available datasets
    if st.session_state.datasets:
        st.subheader("📋 Available Datasets")
        
        for name, data in st.session_state.datasets.items():
            with st.expander(f"📊 {name}"):
                df = data['data']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", len(df))
                with col2:
                    fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
                    st.metric("Fraud Cases", fraud_count)
                with col3:
                    fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                with col4:
                    st.metric("Features", len(df.columns))
                
                # Dataset visualization
                if st.button(f"📊 Visualize", key=f"viz_{name}"):
                    fig = px.histogram(df, x='amount', color='is_fraud', 
                                     title=f"Amount Distribution - {name}",
                                     marginal="box")
                    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Model Training
with tab2:
    st.header("🔧 Advanced Model Training")
    
    if not st.session_state.datasets:
        st.warning("⚠️ Please generate or upload a dataset first!")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Training Configuration")
            
            # Dataset selection
            selected_dataset = st.selectbox(
                "Select Dataset", 
                list(st.session_state.datasets.keys())
            )
            
            # Model selection
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"]
            )
            
            # Training parameters
            with st.expander("⚙️ Training Parameters"):
                test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 50, 500, 100)
                    max_depth = st.slider("Max Depth", 5, 50, 10)
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    max_depth = st.slider("Max Depth", 3, 15, 6)
                elif model_type == "Logistic Regression":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                    max_iter = st.slider("Max Iterations", 100, 2000, 1000)
                elif model_type == "SVM":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
                
                use_scaling = st.checkbox("Feature Scaling", True)
                cross_validation = st.checkbox("Use Cross Validation", True)
                cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
        
        with col2:
            st.subheader("📊 Training Progress")
            
            if st.button("🚀 Start Training", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Get dataset
                    df = st.session_state.datasets[selected_dataset]['data']
                    
                    status_text.text("📊 Preparing data...")
                    progress_bar.progress(10)
                    
                    # Prepare features and target
                    if 'is_fraud' not in df.columns:
                        st.error("❌ Dataset must have 'is_fraud' column")
                        st.stop()
                    
                    # Select numeric features
                    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'is_fraud' in numeric_features:
                        numeric_features.remove('is_fraud')
                    
                    X = df[numeric_features]
                    y = df['is_fraud']
                    
                    # Handle missing values
                    X = X.fillna(X.median())
                    
                    # Split data
                    status_text.text("🔄 Splitting data...")
                    progress_bar.progress(30)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Feature scaling
                    if use_scaling:
                        status_text.text("⚖️ Scaling features...")
                        progress_bar.progress(40)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train.values
                        X_test_scaled = X_test.values
                        scaler = None
                    
                    # Train model
                    status_text.text(f"🤖 Training {model_type} model...")
                    progress_bar.progress(60)
                    
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42
                        )
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(
                            C=C,
                            max_iter=max_iter,
                            random_state=42
                        )
                    elif model_type == "SVM":
                        model = SVC(
                            C=C,
                            kernel=kernel,
                            probability=True,
                            random_state=42
                        )
                    
                    # Fit model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    status_text.text("📈 Evaluating model...")
                    progress_bar.progress(80)
                    
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Metrics
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    # Cross validation
                    if cross_validation:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    
                    # Save model
                    status_text.text("💾 Saving model...")
                    progress_bar.progress(90)
                    
                    model_name = f"{model_type}_{selected_dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Store results
                    st.session_state.trained_models[model_name] = {
                        'model': model,
                        'scaler': scaler,
                        'model_type': model_type,
                        'dataset': selected_dataset,
                        'features': numeric_features,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                    
                    st.session_state.model_performance[model_name] = {
                        'train_accuracy': train_score,
                        'test_accuracy': test_score,
                        'auc_score': auc_score,
                        'cv_mean': cv_mean if cross_validation else None,
                        'cv_std': cv_std if cross_validation else None,
                        'timestamp': datetime.now()
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")
                    
                    # Display results
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>🎉 Model Trained Successfully</h3>
                        <h4>{model_name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Accuracy", f"{train_score:.4f}")
                    with col2:
                        st.metric("Test Accuracy", f"{test_score:.4f}")
                    with col3:
                        st.metric("AUC Score", f"{auc_score:.4f}")
                    
                    if cross_validation:
                        st.info(f"Cross Validation: {cv_mean:.4f} ± {cv_std:.4f}")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)

# Tab 3: Performance Analysis
with tab3:
    st.header("📈 Model Performance Analysis")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train a model first!")
    else:
        # Model selection for analysis
        selected_model = st.selectbox(
            "Select Model for Analysis",
            list(st.session_state.trained_models.keys())
        )
        
        if selected_model:
            model_data = st.session_state.trained_models[selected_model]
            performance_data = st.session_state.model_performance[selected_model]
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Training Accuracy</h3>
                    <h2>{performance_data['train_accuracy']:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Test Accuracy</h3>
                    <h2>{performance_data['test_accuracy']:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>AUC Score</h3>
                    <h2>{performance_data['auc_score']:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                if performance_data['cv_mean']:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>CV Score</h3>
                        <h2>{performance_data['cv_mean']:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                fig_cm = px.imshow(cm, 
                                 labels=dict(x="Predicted", y="Actual", color="Count"),
                                 x=['Normal', 'Fraud'],
                                 y=['Normal', 'Fraud'],
                                 title="Confusion Matrix",
                                 color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # ROC Curve
                fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'])
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                           name=f'ROC Curve (AUC = {performance_data["auc_score"]:.3f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                           mode='lines', name='Random Classifier',
                                           line=dict(dash='dash')))
                fig_roc.update_layout(title='ROC Curve',
                                    xaxis_title='False Positive Rate',
                                    yaxis_title='True Positive Rate')
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Classification Report
            st.subheader("📊 Detailed Classification Report")
            report = classification_report(model_data['y_test'], model_data['y_pred'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
            
            # Feature Importance (if available)
            if hasattr(model_data['model'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                importance_df = pd.DataFrame({
                    'feature': model_data['features'],
                    'importance': model_data['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_importance = px.bar(importance_df.head(15), 
                                      x='importance', y='feature',
                                      orientation='h',
                                      title='Top 15 Feature Importances')
                st.plotly_chart(fig_importance, use_container_width=True)

# Tab 4: Model Testing
with tab4:
    st.header("🎯 Model Testing & Validation")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available for testing!")
    else:
        selected_model = st.selectbox(
            "Select Model for Testing",
            list(st.session_state.trained_models.keys()),
            key="test_model_select"
        )
        
        if selected_model:
            model_data = st.session_state.trained_models[selected_model]
            
            st.subheader("🔍 Single Transaction Test")
            
            # Create input form for transaction testing
            with st.form("transaction_test"):
                col1, col2 = st.columns(2)
                
                with col1:
                    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
                    transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
                    day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, 2)
                    account_age_days = st.number_input("Account Age (days)", min_value=0, value=365)
                
                with col2:
                    previous_transactions = st.number_input("Previous Transactions", min_value=0, value=50)
                    location_risk_score = st.slider("Location Risk Score", 0.0, 1.0, 0.3, 0.01)
                    device_trust_score = st.slider("Device Trust Score", 0.0, 1.0, 0.8, 0.01)
                    merchant_category = st.selectbox("Merchant Category", 
                                                   ["grocery", "gas", "restaurant", "retail", "online", "atm"])
                
                submitted = st.form_submit_button("🔬 Analyze Transaction")
            
            if submitted:
                try:
                    # Create feature vector
                    test_data = {
                        'amount': amount,
                        'transaction_hour': transaction_hour,
                        'day_of_week': day_of_week,
                        'account_age_days': account_age_days,
                        'previous_transactions': previous_transactions,
                        'location_risk_score': location_risk_score,
                        'device_trust_score': device_trust_score,
                        'is_weekend': int(day_of_week >= 5),
                        'is_night': int(transaction_hour < 6 or transaction_hour > 22),
                        'amount_log': np.log1p(amount),
                        'risk_composite': location_risk_score * (1 - device_trust_score),
                        'merchant_category_encoded': ['grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'].index(merchant_category)
                    }
                    
                    # Convert to DataFrame
                    test_df = pd.DataFrame([test_data])
                    
                    # Select only the features used in training
                    available_features = [f for f in model_data['features'] if f in test_df.columns]
                    test_features = test_df[available_features]
                    
                    # Scale features if scaler is available
                    if model_data['scaler'] is not None:
                        test_features_scaled = model_data['scaler'].transform(test_features)
                    else:
                        test_features_scaled = test_features.values
                    
                    # Make prediction
                    prediction = model_data['model'].predict(test_features_scaled)[0]
                    probability = model_data['model'].predict_proba(test_features_scaled)[0]
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fraud_prob = probability[1]
                        st.metric("Fraud Probability", f"{fraud_prob:.3f}")
                    with col2:
                        prediction_text = "🚨 FRAUD" if prediction == 1 else "✅ LEGITIMATE"
                        st.metric("Prediction", prediction_text)
                    
                    # Risk level
                    if fraud_prob > 0.8:
                        risk_level = "🔴 CRITICAL"
                        risk_color = "red"
                    elif fraud_prob > 0.6:
                        risk_level = "🟠 HIGH"
                        risk_color = "orange"
                    elif fraud_prob > 0.3:
                        risk_level = "🟡 MEDIUM"
                        risk_color = "yellow"
                    else:
                        risk_level = "🟢 LOW"
                        risk_color = "green"
                    
                    st.metric("Risk Level", risk_level)
                    
                    # Probability visualization
                    fig_prob = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = fraud_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fraud Probability"},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.6], 'color': "yellow"},
                                {'range': [0.6, 0.8], 'color': "orange"},
                                {'range': [0.8, 1], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5}}))
                    
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)

# Tab 5: Deployment
with tab5:
    st.header("🚀 Model Deployment")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available for deployment!")
    else:
        selected_model = st.selectbox(
            "Select Model for Deployment",
            list(st.session_state.trained_models.keys()),
            key="deploy_model_select"
        )
        
        if selected_model:
            model_data = st.session_state.trained_models[selected_model]
            performance_data = st.session_state.model_performance[selected_model]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Model Summary")
                
                st.write(f"**Model Type:** {model_data['model_type']}")
                st.write(f"**Dataset:** {model_data['dataset']}")
                st.write(f"**Test Accuracy:** {performance_data['test_accuracy']:.4f}")
                st.write(f"**AUC Score:** {performance_data['auc_score']:.4f}")
                st.write(f"**Features:** {len(model_data['features'])}")
                
                # Model export
                st.subheader("💾 Export Model")
                
                if st.button("📦 Save Model Files"):
                    try:
                        # Create models directory if it doesn't exist
                        models_dir = "/home/sayak/coding/fraud-detection-ai/models"
                        os.makedirs(models_dir, exist_ok=True)
                        
                        # Save model
                        model_filename = f"{selected_model}.joblib"
                        model_path = os.path.join(models_dir, model_filename)
                        joblib.dump(model_data['model'], model_path)
                        
                        # Save scaler if exists
                        if model_data['scaler'] is not None:
                            scaler_filename = f"{selected_model}_scaler.joblib"
                            scaler_path = os.path.join(models_dir, scaler_filename)
                            joblib.dump(model_data['scaler'], scaler_path)
                        
                        # Save metadata
                        metadata = {
                            'model_type': model_data['model_type'],
                            'features': model_data['features'],
                            'performance': {k: v for k, v in performance_data.items() if k != 'timestamp'},
                            'has_scaler': model_data['scaler'] is not None,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        metadata_path = os.path.join(models_dir, f"{selected_model}_metadata.json")
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2, default=str)
                        
                        st.success(f"✅ Model saved: {model_filename}")
                        if model_data['scaler'] is not None:
                            st.success(f"✅ Scaler saved: {scaler_filename}")
                        st.success(f"✅ Metadata saved: {selected_model}_metadata.json")
                        
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
            
            with col2:
                st.subheader("🔄 API Integration Code")
                
                api_code = f"""
# Model Integration Code
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('models/{selected_model}.joblib')
{f"scaler = joblib.load('models/{selected_model}_scaler.joblib')" if model_data['scaler'] is not None else "# No scaler needed"}

def predict_fraud(transaction_data):
    \"\"\"
    Predict fraud probability for a transaction
    
    Expected features: {model_data['features']}
    \"\"\"
    
    # Create DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Select features
    features = {model_data['features']}
    X = df[features]
    
    # Scale features
    {f"X_scaled = scaler.transform(X)" if model_data['scaler'] is not None else "X_scaled = X.values"}
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {{
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'risk_level': 'HIGH' if probability > 0.6 else 'MEDIUM' if probability > 0.3 else 'LOW',
        'model_used': '{selected_model}'
    }}

# Example usage:
# result = predict_fraud({{
#     'amount': 1500.0,
#     'transaction_hour': 14,
#     'day_of_week': 2,
#     # ... other features
# }})
"""
                
                st.code(api_code, language='python')

# Sidebar Model Comparison
if st.session_state.trained_models:
    st.sidebar.subheader("📊 Model Comparison")
    
    comparison_data = []
    for name, perf in st.session_state.model_performance.items():
        comparison_data.append({
            'Model': name.split('_')[0],
            'Test Accuracy': perf['test_accuracy'],
            'AUC Score': perf['auc_score']
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.sidebar.dataframe(comparison_df)
        
        # Best model highlight
        best_model_idx = comparison_df['AUC Score'].idxmax()
        best_model = comparison_df.loc[best_model_idx, 'Model']
        best_auc = comparison_df.loc[best_model_idx, 'AUC Score']
        st.sidebar.success(f"🏆 Best Model: {best_model} (AUC: {best_auc:.3f})")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🛡️ CyberShield AI - Advanced Fraud Detection Model Trainer<br>
    Built with Streamlit • Machine Learning • Data Science
</div>
""", unsafe_allow_html=True)
