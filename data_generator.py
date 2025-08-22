import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class TransactionDataGenerator:
    """
    Generates synthetic transaction data for fraud detection model training.
    Creates realistic transaction patterns with both normal and suspicious transactions.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define merchant categories and their typical transaction ranges
        self.merchant_categories = {
            'grocery': (10, 200),
            'gas_station': (20, 100),
            'restaurant': (15, 150),
            'retail': (25, 500),
            'online': (10, 300),
            'atm': (20, 500),
            'pharmacy': (5, 100),
            'entertainment': (20, 200),
            'travel': (100, 2000),
            'utilities': (50, 300)
        }
        
        # Common fraud patterns
        self.fraud_patterns = [
            'multiple_small_transactions',
            'large_unusual_amount',
            'unusual_time',
            'unusual_location',
            'rapid_succession',
            'round_amounts',
            'new_merchant_category'
        ]
    
    def generate_customer_profiles(self, n_customers: int) -> pd.DataFrame:
        """Generate customer profiles with spending patterns."""
        customers = []
        
        for customer_id in range(1, n_customers + 1):
            # Generate customer demographics and spending patterns
            age = np.random.normal(45, 15)
            age = max(18, min(80, int(age)))
            
            # Income affects spending patterns
            income = np.random.lognormal(10.5, 0.5)  # Log-normal distribution for income
            income = max(20000, min(200000, income))
            
            # Spending behavior based on income and age
            avg_transaction_amount = income / 1000 + np.random.normal(0, 20)
            avg_transaction_amount = max(10, avg_transaction_amount)
            
            # Transaction frequency (transactions per month)
            transaction_frequency = np.random.poisson(25) + 5
            
            # Preferred merchant categories
            preferred_categories = random.sample(
                list(self.merchant_categories.keys()), 
                random.randint(3, 6)
            )
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'income': income,
                'avg_transaction_amount': avg_transaction_amount,
                'transaction_frequency': transaction_frequency,
                'preferred_categories': preferred_categories
            })
        
        return pd.DataFrame(customers)
    
    def generate_normal_transactions(self, customers: pd.DataFrame, 
                                   days: int = 365) -> pd.DataFrame:
        """Generate normal transaction patterns for customers."""
        transactions = []
        transaction_id = 1
        
        for _, customer in customers.iterrows():
            customer_id = customer['customer_id']
            
            # Generate transactions over the specified period
            for day in range(days):
                date = datetime.now() - timedelta(days=days-day)
                
                # Determine number of transactions for this day
                daily_transactions = np.random.poisson(
                    customer['transaction_frequency'] / 30
                )
                
                for _ in range(daily_transactions):
                    # Choose merchant category
                    if random.random() < 0.8:  # 80% from preferred categories
                        category = random.choice(customer['preferred_categories'])
                    else:
                        category = random.choice(list(self.merchant_categories.keys()))
                    
                    # Generate transaction amount based on category and customer profile
                    min_amount, max_amount = self.merchant_categories[category]
                    base_amount = np.random.uniform(min_amount, max_amount)
                    
                    # Adjust based on customer's average spending
                    amount_factor = customer['avg_transaction_amount'] / 100
                    amount = base_amount * amount_factor
                    amount = max(5, amount)  # Minimum transaction amount
                    
                    # Add some randomness to the time
                    hour = np.random.normal(14, 4)  # Peak around 2 PM
                    hour = max(6, min(23, int(hour)))  # Business hours bias
                    
                    minute = random.randint(0, 59)
                    timestamp = date.replace(hour=hour, minute=minute, second=0)
                    
                    # Generate location (simplified as coordinates)
                    # Normal transactions cluster around customer's home location
                    base_lat = np.random.uniform(25, 45)  # US latitude range
                    base_lon = np.random.uniform(-125, -65)  # US longitude range
                    
                    lat = base_lat + np.random.normal(0, 0.1)  # Small deviation
                    lon = base_lon + np.random.normal(0, 0.1)
                    
                    transactions.append({
                        'transaction_id': transaction_id,
                        'customer_id': customer_id,
                        'timestamp': timestamp,
                        'amount': round(amount, 2),
                        'merchant_category': category,
                        'latitude': round(lat, 6),
                        'longitude': round(lon, 6),
                        'is_fraud': 0
                    })
                    
                    transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_fraudulent_transactions(self, customers: pd.DataFrame, 
                                       normal_transactions: pd.DataFrame,
                                       fraud_rate: float = 0.02) -> pd.DataFrame:
        """Generate fraudulent transactions based on common fraud patterns."""
        n_fraud_transactions = int(len(normal_transactions) * fraud_rate)
        fraud_transactions = []
        
        # Get the last transaction ID to continue numbering
        last_id = normal_transactions['transaction_id'].max()
        transaction_id = last_id + 1
        
        for _ in range(n_fraud_transactions):
            # Select a random customer
            customer = customers.sample(1).iloc[0]
            customer_id = customer['customer_id']
            
            # Select fraud pattern
            fraud_pattern = random.choice(self.fraud_patterns)
            
            # Generate fraudulent transaction based on pattern
            if fraud_pattern == 'multiple_small_transactions':
                # Multiple small transactions in quick succession
                base_time = datetime.now() - timedelta(days=random.randint(1, 365))
                
                for i in range(random.randint(3, 8)):
                    timestamp = base_time + timedelta(minutes=random.randint(1, 30))
                    amount = np.random.uniform(5, 50)  # Small amounts
                    
                    fraud_transactions.append(self._create_fraud_transaction(
                        transaction_id, customer_id, timestamp, amount, customer
                    ))
                    transaction_id += 1
            
            elif fraud_pattern == 'large_unusual_amount':
                # Single large transaction
                timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
                # Amount much larger than customer's usual spending
                amount = customer['avg_transaction_amount'] * random.uniform(5, 20)
                
                fraud_transactions.append(self._create_fraud_transaction(
                    transaction_id, customer_id, timestamp, amount, customer
                ))
                transaction_id += 1
            
            elif fraud_pattern == 'unusual_time':
                # Transaction at unusual hours
                timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
                hour = random.choice([2, 3, 4, 23, 0, 1])  # Late night/early morning
                timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
                
                amount = np.random.uniform(50, 500)
                
                fraud_transactions.append(self._create_fraud_transaction(
                    transaction_id, customer_id, timestamp, amount, customer
                ))
                transaction_id += 1
            
            elif fraud_pattern == 'unusual_location':
                # Transaction in unusual location (far from normal pattern)
                timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
                
                # Generate location far from normal patterns
                lat = np.random.uniform(25, 45)
                lon = np.random.uniform(-125, -65)
                
                amount = np.random.uniform(100, 1000)
                
                transaction = self._create_fraud_transaction(
                    transaction_id, customer_id, timestamp, amount, customer
                )
                transaction['latitude'] = round(lat, 6)
                transaction['longitude'] = round(lon, 6)
                
                fraud_transactions.append(transaction)
                transaction_id += 1
            
            elif fraud_pattern == 'rapid_succession':
                # Multiple transactions in very short time
                base_time = datetime.now() - timedelta(days=random.randint(1, 365))
                
                for i in range(random.randint(4, 10)):
                    timestamp = base_time + timedelta(seconds=random.randint(10, 300))
                    amount = np.random.uniform(20, 200)
                    
                    fraud_transactions.append(self._create_fraud_transaction(
                        transaction_id, customer_id, timestamp, amount, customer
                    ))
                    transaction_id += 1
            
            elif fraud_pattern == 'round_amounts':
                # Suspicious round amounts
                timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
                amount = random.choice([100, 200, 300, 500, 1000, 1500, 2000])
                
                fraud_transactions.append(self._create_fraud_transaction(
                    transaction_id, customer_id, timestamp, amount, customer
                ))
                transaction_id += 1
            
            elif fraud_pattern == 'new_merchant_category':
                # Transaction in category customer never uses
                timestamp = datetime.now() - timedelta(days=random.randint(1, 365))
                
                # Find categories not in customer's preferred list
                all_categories = set(self.merchant_categories.keys())
                preferred_categories = set(customer['preferred_categories'])
                unusual_categories = list(all_categories - preferred_categories)
                
                if unusual_categories:
                    category = random.choice(unusual_categories)
                    min_amount, max_amount = self.merchant_categories[category]
                    amount = np.random.uniform(max_amount * 0.5, max_amount * 2)
                    
                    transaction = self._create_fraud_transaction(
                        transaction_id, customer_id, timestamp, amount, customer
                    )
                    transaction['merchant_category'] = category
                    
                    fraud_transactions.append(transaction)
                    transaction_id += 1
        
        return pd.DataFrame(fraud_transactions)
    
    def _create_fraud_transaction(self, transaction_id: int, customer_id: int, 
                                timestamp: datetime, amount: float, 
                                customer: pd.Series) -> dict:
        """Helper method to create a fraudulent transaction."""
        # Random category
        category = random.choice(list(self.merchant_categories.keys()))
        
        # Random location
        lat = np.random.uniform(25, 45)
        lon = np.random.uniform(-125, -65)
        
        return {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'is_fraud': 1
        }
    
    def generate_dataset(self, n_customers: int = 1000, days: int = 365, 
                        fraud_rate: float = 0.02) -> pd.DataFrame:
        """Generate complete transaction dataset with normal and fraudulent transactions."""
        print(f"Generating dataset with {n_customers} customers over {days} days...")
        
        # Generate customer profiles
        print("Creating customer profiles...")
        customers = self.generate_customer_profiles(n_customers)
        
        # Generate normal transactions
        print("Generating normal transactions...")
        normal_transactions = self.generate_normal_transactions(customers, days)
        
        # Generate fraudulent transactions
        print("Generating fraudulent transactions...")
        fraud_transactions = self.generate_fraudulent_transactions(
            customers, normal_transactions, fraud_rate
        )
        
        # Combine all transactions
        all_transactions = pd.concat([normal_transactions, fraud_transactions], 
                                   ignore_index=True)
        
        # Sort by timestamp
        all_transactions = all_transactions.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Dataset generated:")
        print(f"- Total transactions: {len(all_transactions):,}")
        print(f"- Normal transactions: {len(normal_transactions):,}")
        print(f"- Fraudulent transactions: {len(fraud_transactions):,}")
        print(f"- Fraud rate: {len(fraud_transactions)/len(all_transactions)*100:.2f}%")
        
        return all_transactions, customers

if __name__ == "__main__":
    # Generate sample dataset
    generator = TransactionDataGenerator(seed=42)
    transactions, customers = generator.generate_dataset(
        n_customers=500, 
        days=180, 
        fraud_rate=0.03
    )
    
    # Save the datasets
    transactions.to_csv('transaction_data.csv', index=False)
    customers.to_csv('customer_profiles.csv', index=False)
    
    print("\nDataset saved to 'transaction_data.csv' and 'customer_profiles.csv'")
    print("\nSample transactions:")
    print(transactions.head(10))
