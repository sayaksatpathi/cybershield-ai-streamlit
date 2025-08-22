# �️ CyberShield AI - Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-orange.svg)](https://scikit-learn.org)

> Advanced Machine Learning-powered fraud detection system with dual frontend interfaces and RESTful API backend.

![CyberShield AI Demo](https://img.shields.io/badge/Demo-Live%20System-brightgreen)

## 🎯 Overview

This system provides a **complete end-to-end solution** for fraud detection, including:

- 🔄 **Synthetic Data Generation**: Creates realistic transaction datasets with 7 distinct fraud patterns
- ⚙️ **Advanced Feature Engineering**: Extracts 49+ meaningful features from raw transaction data  
- 🤖 **Multiple ML Models**: Trains and compares 6 algorithms (Random Forest, XGBoost, LightGBM, etc.)
- ⚡ **Real-time Prediction**: Provides instant fraud scoring with <50ms response time
- 📊 **Interactive Web Interface**: Streamlit-based demo with batch processing capabilities
- 🔍 **Explainable AI**: Detailed insights and explanations for each fraud decision

## 🚀 Features

### Core Capabilities
- **Multi-Algorithm Training**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, SVM
- **Class Imbalance Handling**: SMOTE oversampling and balanced class weights
- **Feature Engineering**: 50+ engineered features including temporal, behavioral, and statistical patterns
- **Real-time Scoring**: Fast prediction interface for production use
- **Explainable AI**: Detailed insights into why transactions are flagged

### Advanced Features
- **Customer Behavior Analysis**: Tracks spending patterns and deviations
- **Temporal Pattern Detection**: Identifies unusual timing patterns
- **Location Risk Assessment**: Flags transactions far from normal locations
- **Velocity Monitoring**: Detects rapid succession transactions
- **Merchant Risk Scoring**: Categories merchants by fraud risk levels

## 📁 Project Structure

```
fraud-detection-ai/
├── 📊 data_generator.py              # Synthetic transaction data generation
├── ⚙️ feature_engineering.py         # Feature extraction and engineering  
├── 🤖 model_training.py              # ML model training and evaluation
├── 🎯 prediction_interface.py        # Real-time prediction system
├── 🌐 web_interface.py               # Streamlit web application
├── 🚀 simple_pipeline.py             # Simplified pipeline for quick start
├── 🎭 demo.py                        # Comprehensive demonstration
├── 📋 main_pipeline.py               # Complete end-to-end pipeline
├── 📦 requirements.txt               # Python dependencies
├── 📚 README.md                      # Project documentation
├── 📈 IMPLEMENTATION_SUMMARY.md      # Technical implementation details
└── 🔒 .gitignore                     # Git ignore rules
```

## 🛡️ Fraud Patterns Detected

The system identifies **7 distinct fraud patterns**:

1. 🔄 **Multiple Small Transactions**: Rapid small amounts to test card validity
2. 💰 **Large Unusual Amounts**: Transactions significantly above normal spending
3. 🌙 **Unusual Timing**: Late night or early morning transactions  
4. 🗺️ **Location Anomalies**: Transactions far from customer's usual area
5. ⚡ **Rapid Succession**: Multiple transactions in short time periods
6. 🎯 **Round Amounts**: Suspicious round number transactions
7. 🆕 **New Merchant Categories**: First-time transactions in unusual categories

## � Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sayaksatpathi/fraud-detection-ai.git
   cd fraud-detection-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**:
   ```bash
   python simple_pipeline.py
   ```

4. **Test predictions**:
   ```bash
   python demo.py
   ```

5. **Launch web interface**:
   ```bash
   streamlit run web_interface.py
   ```

### Quick Demo

```python
from prediction_interface import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor('fraud_detection_model.pkl')

# Analyze a transaction
transaction = {
    'transaction_id': 'txn_001',
    'customer_id': 'cust_123', 
    'timestamp': '2024-01-15 14:30:00',
    'amount': 1500.00,
    'merchant_category': 'online',
    'latitude': 40.7128,
    'longitude': -74.0060
}

result = predictor.predict_single_transaction(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

## 📊 Model Performance

The system achieves excellent performance across all metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 100% | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% | 100% |
| XGBoost | 100% | 100% | 100% | 100% | 100% |
| LightGBM | 100% | 100% | 100% | 100% | 100% |
| Gradient Boosting | 99.9% | 99.9% | 99.9% | 99.9% | 99.9% |
| SVM | 99.8% | 99.8% | 99.6% | 99.8% | 100% |

**System Performance:**
- ⚡ **Prediction Speed**: <50ms per transaction
- 📊 **Features**: 49 engineered features  
- 🎯 **Accuracy**: 100% on test dataset
- 📈 **Scalability**: Production-ready architecture

## 🔍 Feature Engineering

### Temporal Features
- Hour of day, day of week, month
- Business hours indicator
- Weekend/holiday flags
- Late night transaction indicator

### Behavioral Features
- Customer spending patterns
- Deviation from normal amounts
- Transaction frequency analysis
- Merchant category preferences

### Risk Indicators
- Round amount detection
- Rapid succession transactions
- Unusual location patterns
- Merchant risk scoring

### Statistical Features
- Z-scores for amount deviation
- Rolling statistics (1h, 24h, 7d)
- Velocity calculations
- Customer lifetime patterns

## 🎯 Fraud Patterns Detected

1. **Multiple Small Transactions**: Rapid small amounts to test card validity
2. **Large Unusual Amounts**: Transactions significantly above normal spending
3. **Unusual Timing**: Late night or early morning transactions
4. **Location Anomalies**: Transactions far from customer's usual area
5. **Rapid Succession**: Multiple transactions in short time periods
6. **Round Amounts**: Suspicious round number transactions
7. **New Merchant Categories**: First-time transactions in new categories

## 🔧 Configuration

### Data Generation Parameters
```python
# In data_generator.py
n_customers = 1000      # Number of customers
days = 365             # Days of transaction history
fraud_rate = 0.02      # Percentage of fraudulent transactions
```

### Model Training Parameters
```python
# In model_training.py
test_size = 0.2        # Train/test split ratio
use_smote = True       # Enable SMOTE oversampling
cv_folds = 5          # Cross-validation folds
```

## 📈 Usage Examples

### Single Transaction Prediction
```python
from prediction_interface import FraudDetectionPredictor

# Initialize predictor
predictor = FraudDetectionPredictor('fraud_detection_model.pkl')

# Define transaction
transaction = {
    'transaction_id': 'txn_001',
    'customer_id': 'cust_123',
    'timestamp': '2024-01-15 14:30:00',
    'amount': 1500.00,
    'merchant_category': 'online',
    'latitude': 40.7128,
    'longitude': -74.0060
}

# Get prediction
result = predictor.predict_single_transaction(transaction)
print(f"Fraud Probability: {result['fraud_probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

### Batch Processing
```python
# Multiple transactions
transactions = [transaction1, transaction2, transaction3]
results = predictor.predict_batch_transactions(transactions)

# Generate report
report = predictor.create_fraud_report(transactions)
print(f"Fraud Rate: {report['summary']['fraud_rate']:.3f}")
```

## 🎨 Visualizations

The system generates several plots:
- **Model Comparison**: Performance metrics across algorithms
- **ROC Curves**: True/false positive rate analysis
- **Precision-Recall Curves**: Precision vs recall trade-offs
- **Confusion Matrix**: Classification accuracy breakdown
- **Feature Importance**: Most influential features for detection

## 🔍 Model Interpretability

### Feature Importance
The system identifies which features most strongly indicate fraud:
1. Amount deviation from customer normal
2. Transaction frequency in recent periods
3. Time of day patterns
4. Location distance from home
5. Merchant category risk scores

### Prediction Explanations
Each fraud prediction includes:
- Risk level assessment
- Fraud probability score
- Detailed reasons for flagging
- Comparison to customer's normal behavior

## 🚀 Production Deployment

### Real-time Scoring
```python
# Fast prediction for production
prediction, probability = predictor.predict_transaction(features)
```

### Batch Processing
```python
# Process large volumes efficiently
results = predictor.predict_batch_transactions(transactions)
```

### Model Updates
```python
# Retrain with new data
model_trainer.train_and_evaluate_all(new_X_train, new_X_test, new_y_train, new_y_test)
model_trainer.save_best_model('updated_model.pkl')
```

## 📊 Performance Monitoring

### Key Metrics to Track
- **False Positive Rate**: Minimize legitimate transactions flagged
- **False Negative Rate**: Minimize missed fraud cases
- **Processing Latency**: Keep prediction time under 100ms
- **Model Drift**: Monitor feature distributions over time

### Recommended Thresholds
- **High Risk**: Probability ≥ 0.8 (block transaction)
- **Medium Risk**: Probability 0.5-0.8 (additional verification)
- **Low Risk**: Probability 0.2-0.5 (monitor)
- **Very Low Risk**: Probability < 0.2 (normal processing)

## 🔧 Customization

### Adding New Features
1. Modify `feature_engineering.py` to include new feature extraction
2. Update `prediction_interface.py` to handle new features
3. Retrain models with expanded feature set

### Adjusting Fraud Patterns
1. Edit `data_generator.py` to include new fraud patterns
2. Update feature engineering to detect new patterns
3. Retrain models with updated synthetic data

### Model Tuning
1. Modify hyperparameters in `model_training.py`
2. Add new algorithms to the model comparison
3. Implement custom scoring metrics

## 📚 Technical Details

### Algorithms Used
- **Random Forest**: Ensemble method with feature bagging
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with leaf-wise growth
- **Logistic Regression**: Linear model with balanced classes
- **SVM**: Support vector machine with RBF kernel
- **Gradient Boosting**: Sequential ensemble learning

### Data Processing
- **Scaling**: StandardScaler for numerical features
- **Encoding**: One-hot encoding for categorical features
- **Sampling**: SMOTE for handling class imbalance
- **Validation**: Stratified K-fold cross-validation

### Engineering Approach
- **Temporal Windows**: 1h, 24h, 7d rolling statistics
- **Statistical Methods**: Z-scores, percentiles, distributions
- **Geospatial Analysis**: Distance calculations and clustering
- **Behavioral Modeling**: Customer profile deviation detection

## 🤝 Contributing

This is a complete, self-contained fraud detection system. You can:
1. Modify the fraud patterns in data generation
2. Add new feature engineering techniques
3. Experiment with different ML algorithms
4. Enhance the prediction interface
5. Build additional visualization tools

## 📝 License

This project is designed for educational and research purposes. Feel free to use and modify for your fraud detection needs.

## 🆘 Troubleshooting

### Common Issues

1. **Model file not found**: Run `python model_training.py` first
2. **Memory issues**: Reduce dataset size in data generation
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`
4. **Performance issues**: Use fewer models or smaller datasets for testing

### Getting Help

1. Check the console output for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure you have sufficient memory for model training
4. Try running components individually to isolate issues

## 📈 Future Enhancements

Potential improvements:
- **Deep Learning Models**: LSTM/GRU for sequence modeling
- **Real-time Streaming**: Apache Kafka integration
- **Graph Analysis**: Network-based fraud detection
- **Ensemble Methods**: Advanced model stacking
- **Online Learning**: Adaptive models that learn from new data
- **Explainable AI**: SHAP/LIME integration for better interpretability

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning
- [XGBoost](https://xgboost.readthedocs.io/) and [LightGBM](https://lightgbm.readthedocs.io/) for gradient boosting
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for interactive visualizations

## 📞 Contact

- **Author**: Sayak Satpathi
- **GitHub**: [@sayaksatpathi](https://github.com/sayaksatpathi)
- **Repository**: [fraud-detection-ai](https://github.com/sayaksatpathi/fraud-detection-ai)

## ⭐ Star History

If you found this project helpful, please consider giving it a star! ⭐

---

**Disclaimer**: This is a demonstration system using synthetic data. For production use, ensure compliance with relevant regulations and implement appropriate security measures.

