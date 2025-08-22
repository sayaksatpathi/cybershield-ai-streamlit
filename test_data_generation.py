import streamlit as st
import pandas as pd
import numpy as np

st.title("🔧 Data Generation Test")

# Copy the generate_transaction_data function
def generate_transaction_data(n_transactions=1000):
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

# Test the function
try:
    st.write("Testing data generation...")
    df = generate_transaction_data(1000)
    st.success(f"✅ Generated {len(df)} transactions successfully!")
    
    fraud_count = df['is_fraud'].sum()
    fraud_rate = df['is_fraud'].mean()
    
    st.write(f"Fraud cases: {fraud_count}")
    st.write(f"Fraud rate: {fraud_rate:.2%}")
    
    st.write("Sample data:")
    st.dataframe(df.head())
    
    st.write("Data types:")
    st.write(df.dtypes)
    
except Exception as e:
    st.error(f"❌ Error in data generation: {e}")
    import traceback
    st.code(traceback.format_exc())
