# 📤 Custom Dataset Upload Feature

## Overview
CyberShield AI now supports uploading your own datasets for fraud detection testing! This feature allows you to test the fraud detection algorithms with real-world data.

## 🚀 How to Use

### 1. Prepare Your Dataset
Your CSV file should contain:
- **Transaction features** (numerical columns like amounts, times, scores, etc.)
- **Fraud indicator column** (values indicating fraud vs legitimate transactions)

### 2. Upload Process
1. Go to the **"Dataset & Training"** tab
2. Select **"📤 Upload Your Own Dataset"**
3. **Drag and drop** your CSV file or click to browse
4. The system will automatically analyze your data

### 3. Column Mapping
The system will:
- ✅ Auto-detect potential fraud columns (containing "fraud", "label", "target")
- 🔄 Help you map binary values (0/1, True/False, etc.)
- 🎯 Convert multi-class labels to binary fraud indicators
- 📊 Show data validation results

### 4. Train and Test
Once processed, you can:
- 🚀 Train any ML algorithm on your data
- 🔍 Test individual transactions
- 📈 View performance analytics
- 💾 Export the trained model

## 📋 Supported File Formats

### CSV Requirements:
- **File extension**: `.csv`
- **Encoding**: UTF-8 (recommended)
- **Size limit**: Up to 200MB
- **Headers**: First row should contain column names

### Data Requirements:
- **Minimum rows**: 100+ transactions (1000+ recommended)
- **Fraud column**: Binary indicator (0/1, True/False, Yes/No, etc.)
- **Features**: At least 3 numerical features
- **Missing values**: Will be automatically handled

## 📊 Example Dataset Format

```csv
transaction_amount,account_age_days,transaction_hour,merchant_risk_score,is_fraud
150.75,450,14,0.2,0
2800.00,25,3,0.7,1
89.99,1200,10,0.3,0
5500.00,15,2,0.9,1
```

## 🎯 Sample Dataset
A sample dataset (`sample_fraud_dataset.csv`) is included with the app for testing:
- 20 sample transactions
- Mix of legitimate and fraudulent cases
- Proper column formatting
- Ready to upload and test

## ⚠️ Data Validation

The system performs automatic validation:

### ✅ What's Checked:
- File format and readability
- Minimum number of rows
- Presence of fraud indicator column
- Data types and missing values
- Fraud rate (should be 1-50%)

### 🔧 Auto-corrections:
- Missing values filled with column means
- Non-numeric columns excluded from training
- Binary values converted to 0/1 format
- Column names standardized

## 🔍 Advanced Features

### Multi-class Label Support:
- Convert categorical labels (High/Medium/Low) to binary
- Custom mapping interface for complex classifications
- Automatic fraud rate calculation

### Data Preview:
- View first 100 rows of your data
- Statistical summary of all columns
- Column information (data types, null counts)
- Fraud distribution analysis

### Validation Feedback:
- Real-time data quality checks
- Suggestions for data improvements
- Warning for potential issues

## 📈 Expected Results

### With Good Data (>5% fraud rate):
- **Accuracy**: 80-95%
- **Precision**: 0.400-0.800
- **Recall**: 0.300-0.700
- **F1-Score**: 0.350-0.750

### With Challenging Data (<1% fraud rate):
- Model may struggle with class imbalance
- Consider using synthetic data generation
- Or combine with generated data for better training

## 🛠️ Troubleshooting

### Common Issues:

**File won't upload:**
- Check file size (<200MB)
- Ensure .csv extension
- Verify UTF-8 encoding

**No fraud column detected:**
- Rename column to include "fraud", "label", or "target"
- Or manually select the fraud indicator column

**Poor model performance:**
- Check fraud rate (aim for 5-20%)
- Ensure sufficient data (1000+ rows)
- Verify data quality and missing values

**Training fails:**
- Make sure you have numerical features
- Check for class imbalance
- Verify fraud column is properly formatted (0/1)

## 💡 Tips for Best Results

1. **Data Quality**: Clean data with minimal missing values
2. **Fraud Rate**: Aim for 5-20% fraud rate in your dataset
3. **Feature Engineering**: Include meaningful transaction features
4. **Sample Size**: Use 1000+ transactions for robust training
5. **Validation**: Review the data preview before training

## 🔮 Future Enhancements

- Support for additional file formats (Excel, JSON)
- Advanced feature engineering tools
- Real-time data streaming support
- Integration with external databases
- Custom model architectures

---
**Ready to test?** Use the sample dataset or upload your own fraud detection data!
