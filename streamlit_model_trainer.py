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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Try to import XGBoost, fallback if there's a version conflict
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost not available due to version conflict. Using Random Forest and Logistic Regression only.")

# Custom imports
from enhanced_data_generator import EnhancedTransactionDataGenerator

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
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
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

# Sidebar
st.sidebar.title("🎛️ Model Training Dashboard")

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
        st.subheader("🔄 Generate New Dataset")
        
        with st.expander("Dataset Generation Parameters"):
            n_customers = st.slider("Number of Customers", 500, 5000, 2000)
            days_period = st.slider("Time Period (days)", 30, 365, 180)
            fraud_rate = st.slider("Fraud Rate (%)", 1.0, 10.0, 3.0, 0.1) / 100
            
            dataset_size = st.selectbox("Dataset Size", ["Small", "Medium", "Large", "Extra Large"])
            size_multipliers = {"Small": 1, "Medium": 2, "Large": 4, "Extra Large": 8}
            
            complexity_level = st.selectbox("Fraud Complexity", ["Simple", "Medium", "Complex", "Advanced"])
        
        if st.button("🚀 Generate Dataset", type="primary"):
            with st.spinner("Generating comprehensive dataset..."):
                try:
                    generator = EnhancedTransactionDataGenerator()
                    
                    # Generate dataset with parameters
                    customers = n_customers * size_multipliers[dataset_size]
                    transactions, customer_profiles = generator.generate_dataset(
                        n_customers=customers,
                        days=days_period,
                        fraud_rate=fraud_rate
                    )
                    
                    # Store in session state
                    dataset_name = f"Dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.datasets[dataset_name] = {
                        'transactions': transactions,
                        'customers': customer_profiles,
                        'params': {
                            'customers': customers,
                            'days': days_period,
                            'fraud_rate': fraud_rate,
                            'size': dataset_size,
                            'complexity': complexity_level
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
                            'transactions': df,
                            'customers': None,
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
                df = data['transactions']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
                    st.metric("Fraud Cases", fraud_count)
                with col3:
                    fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                with col4:
                    st.metric("Features", len(df.columns))
                
                # Dataset actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"👁️ Preview", key=f"preview_{name}"):
                        st.dataframe(df.head(10))
                with col2:
                    if st.button(f"📊 Analyze", key=f"analyze_{name}"):
                        st.plotly_chart(
                            px.histogram(df, x='amount', color='is_fraud' if 'is_fraud' in df.columns else None,
                                       title="Transaction Amount Distribution"),
                            use_container_width=True
                        )
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{name}"):
                        del st.session_state.datasets[name]
                        st.rerun()

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
            model_options = ["Random Forest", "Logistic Regression"]
            if XGBOOST_AVAILABLE:
                model_options.insert(1, "XGBoost")
            model_options.append("Ensemble")
            
            model_type = st.selectbox(
                "Select Model Type",
                model_options
            )
            
            # Training parameters
            with st.expander("⚙️ Training Parameters"):
                test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 50, 500, 100)
                    max_depth = st.slider("Max Depth", 5, 50, 10)
                    min_samples_split = st.slider("Min Samples Split", 2, 20, 5)
                elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                    n_estimators = st.slider("Number of Rounds", 50, 500, 100)
                    learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
                    max_depth = st.slider("Max Depth", 3, 15, 6)
                elif model_type == "Logistic Regression":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                    max_iter = st.slider("Max Iterations", 100, 2000, 1000)
                
                cross_validation = st.checkbox("Use Cross Validation", True)
                cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
            
            # Feature engineering options
            with st.expander("🔬 Feature Engineering"):
                feature_scaling = st.checkbox("Feature Scaling", True)
                create_interactions = st.checkbox("Create Feature Interactions", False)
                polynomial_features = st.checkbox("Polynomial Features", False)
                
        with col2:
            st.subheader("📊 Training Progress")
            
            if st.button("🚀 Start Training", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Get dataset
                    df = st.session_state.datasets[selected_dataset]['transactions']
                    
                    status_text.text("📊 Preparing data...")
                    progress_bar.progress(10)
                    
                    # Prepare features and target
                    if 'is_fraud' not in df.columns:
                        st.error("❌ Dataset must have 'is_fraud' column")
                        st.stop()
                    
                    # Feature engineering
                    status_text.text("🔬 Engineering features...")
                    progress_bar.progress(30)
                    
                    # Select features (customize based on your data)
                    feature_columns = [col for col in df.columns if col not in ['is_fraud', 'transaction_id', 'customer_id']]
                    X = df[feature_columns].select_dtypes(include=[np.number])
                    y = df['is_fraud']
                    
                    # Handle missing values
                    X = X.fillna(X.median())
                    
                    # Split data
                    status_text.text("🔄 Splitting data...")
                    progress_bar.progress(40)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
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
                    elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                        model = xgb.XGBClassifier(
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
                    else:
                        st.error("❌ Selected model type not available")
                        st.stop()
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    status_text.text("📈 Evaluating model...")
                    progress_bar.progress(80)
                    
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Metrics
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    # Cross validation
                    if cross_validation:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    
                    # Save model
                    status_text.text("💾 Saving model...")
                    progress_bar.progress(90)
                    
                    model_name = f"{model_type}_{selected_dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Store results
                    st.session_state.trained_models[model_name] = {
                        'model': model,
                        'model_type': model_type,
                        'dataset': selected_dataset,
                        'features': feature_columns,
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
                    st.success(f"🎉 Model trained successfully: {model_name}")
                    
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
                st.markdown("""
                <div class="metric-card">
                    <h3>Training Accuracy</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(performance_data['train_accuracy']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Test Accuracy</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(performance_data['test_accuracy']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>AUC Score</h3>
                    <h2>{:.3f}</h2>
                </div>
                """.format(performance_data['auc_score']), unsafe_allow_html=True)
            
            with col4:
                if performance_data['cv_mean']:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>CV Score</h3>
                        <h2>{:.3f}</h2>
                    </div>
                    """.format(performance_data['cv_mean']), unsafe_allow_html=True)
            
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
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🔍 Single Transaction Test")
                
                # Create input form for transaction testing
                with st.form("transaction_test"):
                    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
                    merchant_category = st.selectbox("Merchant Category", 
                                                   ["grocery", "gas_station", "restaurant", "retail", "online", "atm"])
                    transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
                    day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, 2)
                    is_weekend = st.checkbox("Weekend Transaction")
                    
                    submitted = st.form_submit_button("🔬 Analyze Transaction")
                
                if submitted:
                    # Create feature vector (customize based on your model's features)
                    test_data = {
                        'amount': amount,
                        'transaction_hour': transaction_hour,
                        'day_of_week': day_of_week,
                        'is_weekend': int(is_weekend),
                        # Add more features as needed
                    }
                    
                    # Convert to DataFrame with same structure as training data
                    test_df = pd.DataFrame([test_data])
                    
                    # Ensure all required features are present
                    for feature in model_data['features']:
                        if feature not in test_df.columns:
                            test_df[feature] = 0  # Default value for missing features
                    
                    # Select only the features used in training
                    test_df = test_df[model_data['features']]
                    
                    try:
                        # Make prediction
                        prediction = model_data['model'].predict(test_df)[0]
                        probability = model_data['model'].predict_proba(test_df)[0]
                        
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
                        elif fraud_prob > 0.6:
                            risk_level = "🟠 HIGH"
                        elif fraud_prob > 0.3:
                            risk_level = "🟡 MEDIUM"
                        else:
                            risk_level = "🟢 LOW"
                        
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
            
            with col2:
                st.subheader("📈 Batch Testing")
                
                # Upload test dataset
                test_file = st.file_uploader("Upload Test Dataset", type=['csv'], key="batch_test")
                
                if test_file is not None:
                    try:
                        test_df = pd.read_csv(test_file)
                        st.write("Test Dataset Preview:")
                        st.dataframe(test_df.head())
                        
                        if st.button("🚀 Run Batch Test"):
                            with st.spinner("Running batch predictions..."):
                                # Prepare test data
                                test_features = test_df[model_data['features']].fillna(0)
                                
                                # Make predictions
                                predictions = model_data['model'].predict(test_features)
                                probabilities = model_data['model'].predict_proba(test_features)[:, 1]
                                
                                # Add results to dataframe
                                results_df = test_df.copy()
                                results_df['predicted_fraud'] = predictions
                                results_df['fraud_probability'] = probabilities
                                
                                # Display results
                                st.subheader("📊 Batch Test Results")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Transactions", len(results_df))
                                with col2:
                                    fraud_detected = predictions.sum()
                                    st.metric("Fraud Detected", fraud_detected)
                                with col3:
                                    fraud_rate = (fraud_detected / len(results_df)) * 100
                                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                                
                                # Show results table
                                st.dataframe(results_df)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results",
                                    data=csv,
                                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                    except Exception as e:
                        st.error(f"❌ Error processing test file: {str(e)}")

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
                
                if st.button("📦 Save Model File"):
                    try:
                        model_filename = f"{selected_model}.joblib"
                        model_path = f"/home/sayak/coding/fraud-detection-ai/{model_filename}"
                        
                        # Save model
                        joblib.dump(model_data['model'], model_path)
                        
                        # Save metadata
                        metadata = {
                            'model_type': model_data['model_type'],
                            'features': model_data['features'],
                            'performance': performance_data,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        metadata_path = f"/home/sayak/coding/fraud-detection-ai/{selected_model}_metadata.json"
                        import json
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2, default=str)
                        
                        st.success(f"✅ Model saved: {model_filename}")
                        st.success(f"✅ Metadata saved: {selected_model}_metadata.json")
                        
                    except Exception as e:
                        st.error(f"❌ Error saving model: {str(e)}")
            
            with col2:
                st.subheader("🔄 API Integration")
                
                st.write("**API Endpoint Configuration:**")
                
                api_code = f"""
# Update your API server to use this model
import joblib

# Load the trained model
model = joblib.load('{selected_model}.joblib')

# Example prediction function
def predict_fraud(transaction_data):
    features = {model_data['features']}
    
    # Prepare feature vector
    feature_vector = [transaction_data.get(feature, 0) for feature in features]
    
    # Make prediction
    prediction = model.predict([feature_vector])[0]
    probability = model.predict_proba([feature_vector])[0][1]
    
    return {{
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'model_used': '{selected_model}'
    }}
"""
                
                st.code(api_code, language='python')
                
                # Test API integration
                st.subheader("🧪 API Test")
                
                if st.button("🔗 Test API Integration"):
                    # Simulate API call
                    sample_transaction = {
                        'amount': 1500.0,
                        'transaction_hour': 14,
                        'day_of_week': 2,
                        'is_weekend': False
                    }
                    
                    # Prepare features
                    feature_vector = []
                    for feature in model_data['features']:
                        feature_vector.append(sample_transaction.get(feature, 0))
                    
                    # Make prediction
                    prediction = model_data['model'].predict([feature_vector])[0]
                    probability = model_data['model'].predict_proba([feature_vector])[0][1]
                    
                    api_response = {
                        'is_fraud': bool(prediction),
                        'fraud_probability': float(probability),
                        'model_used': selected_model,
                        'status': 'success'
                    }
                    
                    st.json(api_response)
                    st.success("✅ API integration test successful!")

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
        best_model = comparison_df.loc[comparison_df['AUC Score'].idxmax(), 'Model']
        st.sidebar.success(f"🏆 Best Model: {best_model}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🛡️ CyberShield AI - Advanced Fraud Detection Model Trainer<br>
    Built with Streamlit • Machine Learning • Data Science
</div>
""", unsafe_allow_html=True)
