import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataGenerator:
    """
    Enhanced transaction data generator with multiple realistic datasets
    and sophisticated fraud patterns for better model training and testing.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the enhanced data generator."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Extended merchant categories with realistic spending patterns
        self.merchant_categories = {
            'grocery': {'range': (15, 250), 'frequency': 0.25, 'peak_hours': [17, 18, 19]},
            'gas_station': {'range': (25, 120), 'frequency': 0.15, 'peak_hours': [7, 8, 17, 18]},
            'restaurant': {'range': (20, 180), 'frequency': 0.20, 'peak_hours': [12, 13, 19, 20]},
            'retail': {'range': (30, 800), 'frequency': 0.12, 'peak_hours': [14, 15, 16, 19, 20]},
            'online': {'range': (15, 500), 'frequency': 0.18, 'peak_hours': [20, 21, 22]},
            'atm': {'range': (20, 600), 'frequency': 0.08, 'peak_hours': [12, 17, 18, 20]},
            'pharmacy': {'range': (8, 150), 'frequency': 0.06, 'peak_hours': [10, 11, 15, 16]},
            'entertainment': {'range': (25, 300), 'frequency': 0.10, 'peak_hours': [19, 20, 21, 22]},
            'travel': {'range': (150, 3000), 'frequency': 0.03, 'peak_hours': [6, 7, 8, 14, 15]},
            'utilities': {'range': (40, 400), 'frequency': 0.04, 'peak_hours': [9, 10, 11, 16]},
            'insurance': {'range': (100, 800), 'frequency': 0.02, 'peak_hours': [9, 10, 14, 15]},
            'healthcare': {'range': (50, 1200), 'frequency': 0.05, 'peak_hours': [9, 10, 11, 14, 15]},
            'education': {'range': (200, 2000), 'frequency': 0.01, 'peak_hours': [9, 10, 14, 15]},
            'automotive': {'range': (80, 1500), 'frequency': 0.03, 'peak_hours': [10, 11, 14, 15, 16]},
            'home_improvement': {'range': (50, 2000), 'frequency': 0.02, 'peak_hours': [10, 11, 14, 15, 16]}
        }
        
        # Advanced fraud patterns with realistic characteristics
        self.fraud_patterns = {
            'card_testing': {
                'description': 'Small transactions to test stolen card validity',
                'amount_range': (1, 10),
                'frequency': 'high',
                'timing': 'any',
                'merchants': ['online', 'retail']
            },
            'account_takeover': {
                'description': 'Large transactions after account compromise',
                'amount_range': (500, 5000),
                'frequency': 'medium',
                'timing': 'unusual_hours',
                'merchants': ['online', 'retail', 'travel']
            },
            'synthetic_identity': {
                'description': 'New customer with immediate high-value transactions',
                'amount_range': (1000, 10000),
                'frequency': 'low',
                'timing': 'business_hours',
                'merchants': ['travel', 'retail', 'online']
            },
            'velocity_fraud': {
                'description': 'Rapid succession of transactions',
                'amount_range': (50, 500),
                'frequency': 'very_high',
                'timing': 'any',
                'merchants': ['online', 'retail', 'atm']
            },
            'geographic_impossible': {
                'description': 'Transactions in geographically impossible locations',
                'amount_range': (100, 1000),
                'frequency': 'medium',
                'timing': 'any',
                'merchants': ['atm', 'gas_station', 'retail']
            },
            'amount_rounding': {
                'description': 'Suspicious round amount transactions',
                'amount_range': (100, 2000),
                'frequency': 'medium',
                'timing': 'any',
                'merchants': ['online', 'retail', 'atm']
            },
            'dormant_reactivation': {
                'description': 'Dormant account suddenly active with large transactions',
                'amount_range': (1000, 5000),
                'frequency': 'low',
                'timing': 'unusual_hours',
                'merchants': ['online', 'travel', 'retail']
            },
            'merchant_category_jump': {
                'description': 'Sudden change in merchant category preferences',
                'amount_range': (200, 2000),
                'frequency': 'medium',
                'timing': 'any',
                'merchants': ['travel', 'entertainment', 'retail']
            }
        }
        
        # Customer behavior profiles
        self.customer_profiles = {
            'conservative': {'spending_factor': 0.7, 'transaction_freq': 15, 'risk_tolerance': 0.2},
            'moderate': {'spending_factor': 1.0, 'transaction_freq': 25, 'risk_tolerance': 0.5},
            'active': {'spending_factor': 1.5, 'transaction_freq': 35, 'risk_tolerance': 0.8},
            'high_roller': {'spending_factor': 3.0, 'transaction_freq': 45, 'risk_tolerance': 0.9},
            'student': {'spending_factor': 0.4, 'transaction_freq': 12, 'risk_tolerance': 0.3},
            'senior': {'spending_factor': 0.8, 'transaction_freq': 18, 'risk_tolerance': 0.1}
        }
        
        # Geographic regions with different risk levels
        self.geographic_regions = {
            'low_risk': {
                'lat_range': (40.0, 45.0), 'lon_range': (-125.0, -120.0),
                'fraud_multiplier': 0.8
            },
            'medium_risk': {
                'lat_range': (35.0, 40.0), 'lon_range': (-120.0, -110.0),
                'fraud_multiplier': 1.0
            },
            'high_risk': {
                'lat_range': (30.0, 35.0), 'lon_range': (-110.0, -100.0),
                'fraud_multiplier': 1.5
            },
            'very_high_risk': {
                'lat_range': (25.0, 30.0), 'lon_range': (-100.0, -80.0),
                'fraud_multiplier': 2.0
            }
        }
    
    def generate_enhanced_customers(self, n_customers: int) -> pd.DataFrame:
        """Generate enhanced customer profiles with realistic demographics."""
        customers = []
        
        for customer_id in range(1, n_customers + 1):
            # Customer demographics
            age = max(18, min(85, int(np.random.normal(42, 16))))
            
            # Profile type affects behavior
            profile_type = np.random.choice(
                list(self.customer_profiles.keys()),
                p=[0.20, 0.35, 0.25, 0.08, 0.07, 0.05]  # Distribution of customer types
            )
            profile = self.customer_profiles[profile_type]
            
            # Income based on age and profile
            base_income = 35000 + (age - 22) * 1200
            income_factor = profile['spending_factor']
            income = max(15000, base_income * income_factor * np.random.lognormal(0, 0.3))
            
            # Credit score affects spending patterns
            credit_score = max(300, min(850, int(np.random.normal(680, 80))))
            
            # Account age
            account_age_days = max(30, int(np.random.exponential(800)))
            
            # Geographic location (affects fraud risk)
            region = np.random.choice(list(self.geographic_regions.keys()))
            region_data = self.geographic_regions[region]
            home_lat = np.random.uniform(*region_data['lat_range'])
            home_lon = np.random.uniform(*region_data['lon_range'])
            
            # Spending patterns
            avg_transaction_amount = (income / 2000) * profile['spending_factor']
            avg_transaction_amount = max(20, avg_transaction_amount)
            
            # Preferred categories based on profile
            all_categories = list(self.merchant_categories.keys())
            n_preferred = random.randint(4, 8)
            preferred_categories = random.sample(all_categories, n_preferred)
            
            # Device and security factors
            device_trust_score = np.random.beta(8, 2)  # Skewed towards high trust
            has_mobile_app = random.random() < 0.75
            uses_2fa = random.random() < 0.45
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'income': income,
                'credit_score': credit_score,
                'profile_type': profile_type,
                'account_age_days': account_age_days,
                'home_latitude': round(home_lat, 6),
                'home_longitude': round(home_lon, 6),
                'avg_transaction_amount': round(avg_transaction_amount, 2),
                'transaction_frequency': profile['transaction_freq'],
                'preferred_categories': preferred_categories,
                'device_trust_score': round(device_trust_score, 3),
                'has_mobile_app': has_mobile_app,
                'uses_2fa': uses_2fa,
                'risk_region': region
            })
        
        return pd.DataFrame(customers)
    
    def generate_realistic_transactions(self, customers: pd.DataFrame, days: int = 365) -> pd.DataFrame:
        """Generate realistic transaction patterns with seasonal and behavioral variations."""
        transactions = []
        transaction_id = 1
        
        for _, customer in customers.iterrows():
            customer_id = customer['customer_id']
            profile = self.customer_profiles[customer['profile_type']]
            
            # Generate transactions over time period
            for day in range(days):
                current_date = datetime.now() - timedelta(days=days-day)
                
                # Seasonal variations (higher spending in November/December)
                seasonal_factor = 1.0
                if current_date.month in [11, 12]:
                    seasonal_factor = 1.3
                elif current_date.month in [1, 2]:
                    seasonal_factor = 0.8
                
                # Weekend vs weekday patterns
                is_weekend = current_date.weekday() >= 5
                weekend_factor = 1.2 if is_weekend else 1.0
                
                # Daily transaction probability
                daily_prob = (customer['transaction_frequency'] / 30) * seasonal_factor * weekend_factor
                n_transactions = np.random.poisson(daily_prob)
                
                for _ in range(n_transactions):
                    # Select merchant category
                    if random.random() < 0.85:  # 85% from preferred categories
                        category = random.choice(customer['preferred_categories'])
                    else:
                        # Get frequency values and normalize them
                        categories = list(self.merchant_categories.keys())
                        frequencies = [self.merchant_categories[cat]['frequency'] for cat in categories]
                        frequencies = np.array(frequencies)
                        frequencies = frequencies / frequencies.sum()  # Normalize to sum to 1
                        category = np.random.choice(categories, p=frequencies)
                    
                    cat_data = self.merchant_categories[category]
                    
                    # Generate transaction time based on category peak hours
                    if random.random() < 0.7:  # 70% during peak hours
                        hour = random.choice(cat_data['peak_hours'])
                    else:
                        hour = random.randint(6, 23)
                    
                    minute = random.randint(0, 59)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                    
                    # Generate amount with realistic variations
                    min_amount, max_amount = cat_data['range']
                    base_amount = np.random.uniform(min_amount, max_amount)
                    
                    # Apply customer spending factor
                    amount_factor = customer['avg_transaction_amount'] / 100
                    amount = base_amount * amount_factor
                    
                    # Add random variation
                    amount *= np.random.lognormal(0, 0.2)
                    amount = max(5, round(amount, 2))
                    
                    # Generate location (near customer's home for normal transactions)
                    lat_variation = np.random.normal(0, 0.05)  # Small variation
                    lon_variation = np.random.normal(0, 0.05)
                    
                    lat = customer['home_latitude'] + lat_variation
                    lon = customer['home_longitude'] + lon_variation
                    
                    # Calculate risk factors
                    time_risk = self._calculate_time_risk(hour)
                    location_risk = self._calculate_location_risk(lat, lon, customer)
                    amount_risk = self._calculate_amount_risk(amount, customer)
                    
                    # Calculate time since last transaction (simplified)
                    if transactions:
                        last_transaction_time = transactions[-1]['timestamp']
                        time_since_last = (timestamp - last_transaction_time).total_seconds() / 3600
                    else:
                        time_since_last = 24.0  # Default for first transaction
                    
                    transactions.append({
                        'transaction_id': transaction_id,
                        'customer_id': customer_id,
                        'timestamp': timestamp,
                        'amount': amount,
                        'merchant_category': category,
                        'transaction_hour': hour,
                        'day_of_week': timestamp.weekday(),
                        'is_weekend': is_weekend,
                        'latitude': round(lat, 6),
                        'longitude': round(lon, 6),
                        'account_age_days': customer['account_age_days'] + day,
                        'time_since_last_transaction': round(time_since_last, 2),
                        'location_risk_score': round(location_risk, 3),
                        'device_trust_score': customer['device_trust_score'],
                        'time_risk_score': round(time_risk, 3),
                        'amount_risk_score': round(amount_risk, 3),
                        'is_fraud': 0
                    })
                    
                    transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def generate_sophisticated_fraud(self, customers: pd.DataFrame, 
                                   normal_transactions: pd.DataFrame,
                                   fraud_rate: float = 0.03) -> pd.DataFrame:
        """Generate sophisticated fraudulent transactions using advanced patterns."""
        n_fraud_transactions = int(len(normal_transactions) * fraud_rate)
        fraud_transactions = []
        
        last_id = normal_transactions['transaction_id'].max()
        transaction_id = last_id + 1
        
        for _ in range(n_fraud_transactions):
            # Select fraud pattern
            pattern_name = np.random.choice(list(self.fraud_patterns.keys()))
            pattern = self.fraud_patterns[pattern_name]
            
            # Select target customer
            customer = customers.sample(1).iloc[0]
            customer_id = customer['customer_id']
            
            # Generate fraud transactions based on pattern
            fraud_txns = self._generate_pattern_transactions(
                pattern_name, pattern, customer, transaction_id
            )
            
            fraud_transactions.extend(fraud_txns)
            transaction_id += len(fraud_txns)
        
        return pd.DataFrame(fraud_transactions)
    
    def _generate_pattern_transactions(self, pattern_name: str, pattern: Dict, 
                                     customer: pd.Series, start_id: int) -> List[Dict]:
        """Generate transactions for a specific fraud pattern."""
        transactions = []
        base_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        if pattern_name == 'card_testing':
            # Multiple small transactions
            for i in range(random.randint(5, 15)):
                timestamp = base_time + timedelta(minutes=random.randint(1, 60))
                amount = np.random.uniform(*pattern['amount_range'])
                category = random.choice(pattern['merchants'])
                
                transactions.append(self._create_fraud_transaction(
                    start_id + i, customer, timestamp, amount, category, pattern_name
                ))
        
        elif pattern_name == 'velocity_fraud':
            # Rapid succession
            for i in range(random.randint(8, 20)):
                timestamp = base_time + timedelta(seconds=random.randint(30, 600))
                amount = np.random.uniform(*pattern['amount_range'])
                category = random.choice(pattern['merchants'])
                
                transactions.append(self._create_fraud_transaction(
                    start_id + i, customer, timestamp, amount, category, pattern_name
                ))
        
        elif pattern_name == 'account_takeover':
            # Sudden large transactions at unusual times
            for i in range(random.randint(2, 5)):
                hour = random.choice([2, 3, 4, 23, 0, 1])  # Unusual hours
                timestamp = base_time.replace(hour=hour, minute=random.randint(0, 59))
                timestamp += timedelta(days=i)
                
                amount = np.random.uniform(*pattern['amount_range'])
                category = random.choice(pattern['merchants'])
                
                transactions.append(self._create_fraud_transaction(
                    start_id + i, customer, timestamp, amount, category, pattern_name
                ))
        
        elif pattern_name == 'geographic_impossible':
            # Transactions in impossible locations
            for i in range(random.randint(2, 4)):
                timestamp = base_time + timedelta(hours=random.randint(1, 6))
                amount = np.random.uniform(*pattern['amount_range'])
                category = random.choice(pattern['merchants'])
                
                # Generate impossible location (different continent)
                if i == 0:
                    lat, lon = customer['home_latitude'], customer['home_longitude']
                else:
                    # Jump to different continent
                    lat = np.random.uniform(45, 55)  # Europe
                    lon = np.random.uniform(0, 20)
                
                transaction = self._create_fraud_transaction(
                    start_id + i, customer, timestamp, amount, category, pattern_name
                )
                transaction['latitude'] = round(lat, 6)
                transaction['longitude'] = round(lon, 6)
                transactions.append(transaction)
        
        else:
            # Default single transaction for other patterns
            timestamp = base_time
            if pattern['timing'] == 'unusual_hours':
                hour = random.choice([2, 3, 4, 23, 0, 1])
                timestamp = timestamp.replace(hour=hour, minute=random.randint(0, 59))
            
            amount = np.random.uniform(*pattern['amount_range'])
            if pattern_name == 'amount_rounding':
                amount = random.choice([100, 200, 300, 500, 750, 1000, 1500, 2000])
            
            category = random.choice(pattern['merchants'])
            
            transactions.append(self._create_fraud_transaction(
                start_id, customer, timestamp, amount, category, pattern_name
            ))
        
        return transactions
    
    def _create_fraud_transaction(self, transaction_id: int, customer: pd.Series,
                                timestamp: datetime, amount: float, 
                                category: str, fraud_type: str) -> Dict:
        """Create a single fraudulent transaction."""
        # Default location (can be overridden)
        lat = customer['home_latitude'] + np.random.normal(0, 0.2)
        lon = customer['home_longitude'] + np.random.normal(0, 0.2)
        
        # High risk scores for fraud transactions
        location_risk = min(1.0, np.random.uniform(0.6, 1.0))
        time_risk = self._calculate_time_risk(timestamp.hour)
        amount_risk = min(1.0, amount / (customer['avg_transaction_amount'] * 2))
        
        # Device trust lower for fraud
        device_trust = customer['device_trust_score'] * np.random.uniform(0.3, 0.8)
        
        return {
            'transaction_id': transaction_id,
            'customer_id': customer['customer_id'],
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': category,
            'transaction_hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'account_age_days': customer['account_age_days'],
            'time_since_last_transaction': np.random.uniform(0.1, 2.0),
            'location_risk_score': round(location_risk, 3),
            'device_trust_score': round(device_trust, 3),
            'time_risk_score': round(time_risk, 3),
            'amount_risk_score': round(amount_risk, 3),
            'fraud_type': fraud_type,
            'is_fraud': 1
        }
    
    def _calculate_time_risk(self, hour: int) -> float:
        """Calculate risk score based on transaction time."""
        if hour in [2, 3, 4]:  # Very late night
            return 0.9
        elif hour in [0, 1, 5, 23]:  # Late night/early morning
            return 0.7
        elif hour in [6, 7, 8, 17, 18, 19]:  # Rush hours
            return 0.2
        elif hour in [9, 10, 11, 12, 13, 14, 15, 16]:  # Business hours
            return 0.1
        else:  # Evening
            return 0.3
    
    def _calculate_location_risk(self, lat: float, lon: float, customer: pd.Series) -> float:
        """Calculate risk score based on location."""
        # Distance from home
        home_lat, home_lon = customer['home_latitude'], customer['home_longitude']
        distance = ((lat - home_lat)**2 + (lon - home_lon)**2)**0.5
        
        # Normalize distance to risk score
        risk = min(1.0, distance / 0.5)  # 0.5 degrees is high risk
        
        # Add regional risk
        region_multiplier = self.geographic_regions[customer['risk_region']]['fraud_multiplier']
        risk *= region_multiplier
        
        return min(1.0, risk)
    
    def _calculate_amount_risk(self, amount: float, customer: pd.Series) -> float:
        """Calculate risk score based on transaction amount."""
        avg_amount = customer['avg_transaction_amount']
        ratio = amount / avg_amount
        
        if ratio > 5:  # 5x normal amount
            return 0.9
        elif ratio > 3:  # 3x normal amount
            return 0.7
        elif ratio > 2:  # 2x normal amount
            return 0.4
        elif ratio < 0.1:  # Very small amount
            return 0.6
        else:
            return 0.1
    
    def generate_comprehensive_dataset(self, n_customers: int = 2000, days: int = 365, 
                                     fraud_rate: float = 0.04) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate comprehensive dataset with enhanced features and realistic patterns."""
        print(f"🔄 Generating comprehensive dataset...")
        print(f"   - Customers: {n_customers:,}")
        print(f"   - Time period: {days} days")
        print(f"   - Target fraud rate: {fraud_rate:.1%}")
        
        # Generate enhanced customer profiles
        print("👥 Creating customer profiles...")
        customers = self.generate_enhanced_customers(n_customers)
        
        # Generate realistic normal transactions
        print("💳 Generating normal transactions...")
        normal_transactions = self.generate_realistic_transactions(customers, days)
        
        # Generate sophisticated fraud
        print("🚨 Generating fraudulent transactions...")
        fraud_transactions = self.generate_sophisticated_fraud(
            customers, normal_transactions, fraud_rate
        )
        
        # Combine all transactions
        all_transactions = pd.concat([normal_transactions, fraud_transactions], 
                                   ignore_index=True)
        
        # Sort by timestamp and reset index
        all_transactions = all_transactions.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features
        print("🔧 Adding derived features...")
        all_transactions = self._add_derived_features(all_transactions, customers)
        
        # Final statistics
        fraud_count = all_transactions['is_fraud'].sum()
        total_count = len(all_transactions)
        actual_fraud_rate = fraud_count / total_count
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   📊 Total transactions: {total_count:,}")
        print(f"   ✅ Normal transactions: {total_count - fraud_count:,}")
        print(f"   🚨 Fraudulent transactions: {fraud_count:,}")
        print(f"   📈 Actual fraud rate: {actual_fraud_rate:.2%}")
        print(f"   🎯 Fraud patterns: {len(self.fraud_patterns)} different types")
        
        return all_transactions, customers
    
    def _add_derived_features(self, transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better model performance."""
        # Merge customer data
        customer_features = customers[['customer_id', 'avg_transaction_amount', 'profile_type']]
        transactions = transactions.merge(
            customer_features, 
            on='customer_id', 
            how='left', 
            suffixes=('', '_customer')
        )
        
        # Calculate additional features
        transactions['amount_vs_avg_ratio'] = (
            transactions['amount'] / transactions['avg_transaction_amount']
        ).round(3)
        
        transactions['is_round_amount'] = (
            transactions['amount'] % 50 == 0
        ).astype(int)
        
        transactions['is_business_hours'] = (
            (transactions['transaction_hour'] >= 9) & 
            (transactions['transaction_hour'] <= 17)
        ).astype(int)
        
        # Drop temporary columns
        transactions = transactions.drop(['avg_transaction_amount'], axis=1)
        
        return transactions

# Example usage and testing
if __name__ == "__main__":
    # Generate enhanced dataset
    generator = EnhancedDataGenerator(seed=42)
    
    # Generate multiple dataset sizes for testing
    datasets = {
        'small': (1000, 180, 0.03),    # 1K customers, 6 months
        'medium': (3000, 365, 0.04),   # 3K customers, 1 year
        'large': (5000, 730, 0.05),    # 5K customers, 2 years
    }
    
    for name, (n_customers, days, fraud_rate) in datasets.items():
        print(f"\n🚀 Generating {name} dataset...")
        
        transactions, customers = generator.generate_comprehensive_dataset(
            n_customers=n_customers,
            days=days,
            fraud_rate=fraud_rate
        )
        
        # Save datasets
        transactions.to_csv(f'enhanced_transactions_{name}.csv', index=False)
        customers.to_csv(f'enhanced_customers_{name}.csv', index=False)
        
        print(f"💾 Saved {name} dataset files")
        
        # Show sample
        print(f"\n📋 Sample transactions from {name} dataset:")
        print(transactions[['transaction_id', 'amount', 'merchant_category', 
                          'is_fraud', 'fraud_type']].head(10))
