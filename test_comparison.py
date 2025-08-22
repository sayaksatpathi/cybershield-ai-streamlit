import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Import the old function
def generate_transaction_data_old(n_transactions=1000):
    """Old version with poor fraud generation"""
    np.random.seed(42)
    
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
    df['transaction_amount'] = np.clip(df['transaction_amount'], 1, 10000)
    df['account_age_days'] = np.clip(df['account_age_days'], 1, 3650)
    df['merchant_risk_score'] = np.clip(df['merchant_risk_score'], 0, 1)
    df['transaction_country_risk'] = np.clip(df['transaction_country_risk'], 0, 1)
    
    # OLD PROBLEMATIC LOGIC
    risk_score = (
        (df['transaction_amount'] > 1000) * 0.2 +
        (df['account_age_days'] < 30) * 0.3 +
        (df['transaction_hour'].isin([2, 3, 4])) * 0.2 +
        (df['days_since_last_transaction'] > 7) * 0.1 +
        df['merchant_risk_score'] * 0.2 +
        df['transaction_country_risk'] * 0.15 +
        (df['payment_method_risk'] > 0.5) * 0.1
    )
    
    fraud_probability = 1 / (1 + np.exp(-3 * (risk_score - 1)))
    df['is_fraud'] = np.random.binomial(1, fraud_probability)
    
    return df

# Import the new improved function
def generate_transaction_data_new(n_transactions=1000):
    """New improved version with proper fraud generation"""
    np.random.seed(42)
    
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
    df['transaction_amount'] = np.clip(df['transaction_amount'], 1, 10000)
    df['account_age_days'] = np.clip(df['account_age_days'], 1, 3650)
    df['merchant_risk_score'] = np.clip(df['merchant_risk_score'], 0, 1)
    df['transaction_country_risk'] = np.clip(df['transaction_country_risk'], 0, 1)
    
    # IMPROVED fraud generation logic
    high_amount = (df['transaction_amount'] > 2000).astype(int)
    new_account = (df['account_age_days'] < 90).astype(int)
    night_transaction = (df['transaction_hour'].isin([1, 2, 3, 4, 5])).astype(int)
    long_gap = (df['days_since_last_transaction'] > 30).astype(int)
    high_merchant_risk = (df['merchant_risk_score'] > 0.6).astype(int)
    high_country_risk = (df['transaction_country_risk'] > 0.5).astype(int)
    high_payment_risk = (df['payment_method_risk'] > 0.4).astype(int)
    
    fraud_score = (
        high_amount * 0.25 +
        new_account * 0.2 +
        night_transaction * 0.15 +
        long_gap * 0.1 +
        high_merchant_risk * 0.3 +
        high_country_risk * 0.2 +
        high_payment_risk * 0.15
    )
    
    # Add fraud patterns
    pattern1 = (high_amount & new_account).astype(int)
    pattern2 = (night_transaction & high_payment_risk).astype(int)
    pattern3 = (high_merchant_risk & high_country_risk).astype(int)
    
    fraud_score += pattern1 * 0.4 + pattern2 * 0.3 + pattern3 * 0.35
    fraud_score = np.clip(fraud_score, 0, 1)
    
    base_fraud_rate = 0.08
    fraud_probability = base_fraud_rate + (fraud_score * 0.6)
    df['is_fraud'] = np.random.binomial(1, fraud_probability)
    
    # Ensure minimum fraud cases
    current_fraud_rate = df['is_fraud'].mean()
    if current_fraud_rate < 0.03:
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

# Test both versions
print("🔍 Comparing Old vs New Data Generation:")
print("="*50)

# Old version
df_old = generate_transaction_data_old(10000)
fraud_count_old = df_old['is_fraud'].sum()
fraud_rate_old = df_old['is_fraud'].mean()

print(f"OLD VERSION:")
print(f"  Total transactions: {len(df_old)}")
print(f"  Fraud cases: {fraud_count_old}")
print(f"  Fraud rate: {fraud_rate_old:.4f} ({fraud_rate_old*100:.2f}%)")

# New version  
df_new = generate_transaction_data_new(10000)
fraud_count_new = df_new['is_fraud'].sum()
fraud_rate_new = df_new['is_fraud'].mean()

print(f"\nNEW VERSION:")
print(f"  Total transactions: {len(df_new)}")
print(f"  Fraud cases: {fraud_count_new}")
print(f"  Fraud rate: {fraud_rate_new:.4f} ({fraud_rate_new*100:.2f}%)")

print(f"\n✅ IMPROVEMENT:")
print(f"  Fraud cases increased by: {fraud_count_new - fraud_count_old}")
print(f"  Fraud rate improved by: {(fraud_rate_new - fraud_rate_old)*100:.2f} percentage points")

if fraud_count_new > 50:
    print(f"\n🎯 NEW VERSION HAS SUFFICIENT FRAUD CASES FOR PROPER ML TRAINING!")
else:
    print(f"\n⚠️  Still need more fraud cases for optimal training")
