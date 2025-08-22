# 🛡️ How to Check Transaction Fraud in Main Interface
**Step-by-Step Guide to Fraud Analysis**

## 🚀 **Quick Start**

### **Step 1: Access the Main Interface**
- Open your browser and go to: `http://localhost:8080/`
- You should see the CyberShield AI main dashboard

### **Step 2: Navigate to Threat Analysis**
- Click on the **"Threat Analysis"** tab in the navigation bar
- This is where you can analyze individual transactions for fraud

---

## 📊 **Using the Fraud Analysis Feature**

### **Method 1: Individual Transaction Analysis**

**1. Fill out the Transaction Form:**
```
Transaction Details:
├── Amount: Enter transaction amount (e.g., $1500.00)
├── Merchant Category: Select from dropdown
│   ├── Grocery Store
│   ├── Gas Station  
│   ├── Restaurant
│   ├── Retail Store
│   ├── Online Purchase
│   └── ATM Withdrawal
├── Hour: Transaction time (0-23)
├── Day of Week: Monday-Sunday
├── Weekend: Check if weekend transaction
└── Customer Profile:
    ├── Account Age (days)
    ├── Previous Transactions
    ├── Average Amount
    ├── Location Risk (0-1)
    ├── Device Trust (0-1)
    └── Time Since Last Transaction
```

**2. Click "Analyze Transaction"**
- The system will process your request
- Wait for the analysis results

**3. View Results:**
```
Analysis Results:
├── 🎯 Fraud Probability: XX.X%
├── ⚠️ Risk Level: LOW/MEDIUM/HIGH/CRITICAL
├── 📊 Confidence Score: XX.X%
├── 🏷️ Risk Factors: Listed breakdown
└── 💡 Recommendation: APPROVE/REVIEW/BLOCK
```

---

## 🔍 **Example Transaction Analysis**

### **Test Case 1: Suspicious Transaction**
```
Input:
├── Amount: $8,500.00
├── Category: Online Purchase
├── Hour: 3 AM (3)
├── Day: Sunday
├── Weekend: ✅ Yes
├── Account Age: 30 days
├── Previous Transactions: 5
├── Average Amount: $50.00
├── Location Risk: 0.8 (high)
├── Device Trust: 0.2 (low)
└── Time Since Last: 0.1 hours

Expected Result:
├── 🚨 Fraud Probability: ~85%
├── ⚠️ Risk Level: CRITICAL
└── 💡 Recommendation: BLOCK TRANSACTION
```

### **Test Case 2: Normal Transaction**
```
Input:
├── Amount: $45.00
├── Category: Grocery Store
├── Hour: 2 PM (14)
├── Day: Tuesday
├── Weekend: ❌ No
├── Account Age: 1,200 days
├── Previous Transactions: 250
├── Average Amount: $65.00
├── Location Risk: 0.1 (low)
├── Device Trust: 0.9 (high)
└── Time Since Last: 24 hours

Expected Result:
├── ✅ Fraud Probability: ~5%
├── ⚠️ Risk Level: LOW
└── 💡 Recommendation: APPROVE
```

---

## 🔧 **Advanced Features**

### **Method 2: Batch Analysis**
1. Go to **"Batch Analysis"** tab
2. Upload CSV file with multiple transactions
3. Click "Process Batch"
4. View comprehensive fraud analysis report

### **Method 3: Live Monitoring**
1. Navigate to **"Live Security Feed"** tab
2. Click **"Start Real-time Monitoring"**
3. Watch live transaction stream with fraud detection
4. See real-time fraud blocking in action

---

## 📊 **Understanding the Results**

### **Fraud Probability Scale:**
- **0-25%:** ✅ LOW RISK - Safe to approve
- **26-50%:** ⚠️ MEDIUM RISK - Monitor closely
- **51-75%:** 🔶 HIGH RISK - Manual review recommended
- **76-100%:** 🚨 CRITICAL RISK - Block transaction

### **Risk Factors Analyzed:**
✅ **Amount Analysis**
- Unusual amounts compared to history
- Very high or very low transactions

✅ **Time Pattern Analysis**
- Unusual hours (late night/early morning)
- Weekend vs weekday patterns

✅ **Location & Device**
- Geographic risk assessment
- Device trust and recognition

✅ **Customer Behavior**
- Account age and history
- Transaction frequency patterns
- Spending habit analysis

---

## 🎯 **Real-Time API Integration**

The Main Interface connects to your fraud detection API for:

### **Live ML Model Predictions:**
- Uses your trained Random Forest model
- Real-time feature engineering
- 98.2% accuracy fraud detection

### **API Endpoints Used:**
```
GET  /api/status          - Check system health
POST /api/analyze         - Analyze single transaction
POST /api/batch-analyze   - Process multiple transactions
GET  /api/model-info      - Get model performance metrics
```

### **Response Format:**
```json
{
  "fraud_probability": 0.73,
  "is_fraud": true,
  "risk_level": "HIGH",
  "confidence": 0.95,
  "risk_factors": [
    "High amount for account",
    "Unusual time pattern",
    "New device detected"
  ],
  "recommendation": "MANUAL_REVIEW"
}
```

---

## 🚀 **Quick Test Instructions**

### **Test Right Now:**
1. **Open:** `http://localhost:8080/`
2. **Click:** "Threat Analysis" tab
3. **Enter:** Amount: $2000, Category: Online, Hour: 2
4. **Fill:** Other fields with realistic values
5. **Click:** "Analyze Transaction"
6. **View:** Your fraud analysis results!

### **Pro Tips:**
- 💡 Try different amount ranges to see risk changes
- 💡 Test late-night hours (0-5 AM) for higher risk
- 💡 Online/ATM categories typically score higher risk
- 💡 New accounts (low age) get more scrutiny
- 💡 Check weekend transactions for pattern differences

---

## ✅ **What You Get**

🎯 **Instant Fraud Detection**
- Real-time analysis results
- Professional risk assessment
- Clear recommendations

📊 **Detailed Analytics**
- Risk factor breakdown
- Confidence scoring
- Historical comparison

🔧 **Enterprise Features**
- Batch processing capability
- Live monitoring feed
- API integration ready

**Perfect for:** Testing transactions, client demonstrations, fraud investigation, and system validation!
