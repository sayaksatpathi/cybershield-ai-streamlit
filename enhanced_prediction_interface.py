"""
Enhanced Fraud Detection Prediction Interface with Production Features
Includes real-time fraud pattern detection, statistical anomaly detection,
multi-tier risk scoring, and comprehensive explainable AI.
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedFraudDetectionPredictor:
    """
    Production-ready fraud detection prediction interface with:
    - Real-time fraud pattern detection
    - Statistical anomaly detection
    - Multi-tier risk scoring (Very High/High/Medium/Low)
    - Enhanced explainable AI with baseline comparison
    - Model consensus from ensemble
    """
    
    def __init__(self, model_path: str = 'enhanced_fraud_detection_model.pkl'):
        """Initialize the enhanced predictor with a trained model."""
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.model_name = None
        self.fraud_patterns = None
        self.risk_thresholds = {
            'very_high': 0.85,
            'high': 0.65,
            'medium': 0.35,
            'low': 0.15
        }
        
        # Load the model
        self.load_enhanced_model(model_path)
        
        # Initialize baseline statistics for comparison
        self.baseline_stats = self._initialize_baseline_stats()
        
        # Pattern detection rules
        self.pattern_rules = {
            'card_testing': lambda x: (x.get('transactions_last_hour', 0) >= 3) and (x.get('amount', 0) < 10),
            'account_takeover': lambda x: (x.get('amount_vs_customer_mean', 1) > 5) and (x.get('hour_vs_customer_mean', 0) > 8),
            'synthetic_identity': lambda x: (x.get('customer_timestamp_count', 1) <= 3) and (x.get('amount', 0) > 500),
            'bust_out_fraud': lambda x: (x.get('transaction_velocity_day', 0) > 3) and (x.get('amount_last_day', 0) > 1000),
            'geographic_anomaly': lambda x: (x.get('distance_from_home', 0) > 1000) and (x.get('time_since_last_transaction', 24) < 2),
            'velocity_abuse': lambda x: (x.get('transactions_last_hour', 0) >= 5) or (x.get('transactions_last_day', 0) >= 20),
            'round_amount_scheme': lambda x: (x.get('is_very_round', 0) == 1) and (x.get('amount', 0) >= 100) and (x.get('merchant_category', '') in ['online', 'atm'])
        }
        
        # Risk scoring weights
        self.risk_weights = {
            'model_probability': 0.4,
            'pattern_score': 0.25,
            'anomaly_score': 0.2,
            'velocity_score': 0.15
        }
    
    def _initialize_baseline_stats(self) -> Dict[str, float]:
        """Initialize baseline statistics for normal transactions."""
        return {
            'normal_fraud_rate': 0.025,
            'avg_amount': 85.50,
            'avg_hour': 14.5,
            'avg_transactions_per_day': 2.5,
            'avg_distance_from_home': 15.2,
            'normal_merchant_categories': ['grocery', 'gas_station', 'restaurant', 'retail']
        }
    
    def load_enhanced_model(self, model_path: str):
        """Load the enhanced trained model and its comprehensive metadata."""
        try:
            # Load the model
            self.model = joblib.load(model_path)
            
            # Load enhanced metadata
            metadata_path = model_path.replace('.pkl', '_metadata.pkl')
            self.metadata = joblib.load(metadata_path)
            
            self.feature_columns = self.metadata['feature_columns']
            self.model_name = self.metadata['model_name']
            self.fraud_patterns = self.metadata.get('fraud_patterns', {})
            self.risk_thresholds = self.metadata.get('risk_thresholds', self.risk_thresholds)
            
            print(f"✅ Enhanced model '{self.model_name}' loaded successfully!")
            print(f"📊 Model expects {len(self.feature_columns)} features")
            print(f"🔍 Fraud patterns: {len(self.fraud_patterns)}")
            
        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            print("🚀 Creating quick demo model for immediate use...")
            self._create_demo_model()
            
    def _create_demo_model(self):
        """Create a simple demo model when the main model is not available."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create basic demo model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model_name = "Demo Model"
        self.feature_columns = [
            'amount', 'merchant_risk_score', 'time_since_last_transaction',
            'transaction_frequency', 'amount_zscore'
        ]
        self.fraud_patterns = {}
        
        # Create dummy metadata
        self.metadata = {
            'model_name': self.model_name,
            'feature_columns': self.feature_columns,
            'fraud_patterns': self.fraud_patterns,
            'risk_thresholds': self.risk_thresholds
        }
        
        # Train on minimal dummy data
        X_dummy = np.random.random((100, len(self.feature_columns)))
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)
        
        print("✅ Demo model created successfully!")
    
    def extract_enhanced_features(self, transaction: Dict[str, Any], 
                                customer_history: List[Dict] = None) -> Dict[str, float]:
        """
        Extract comprehensive features including fraud patterns and anomaly detection.
        """
        features = {}
        
        # Parse timestamp
        if isinstance(transaction['timestamp'], str):
            timestamp = pd.to_datetime(transaction['timestamp'])
        else:
            timestamp = transaction['timestamp']
        
        # Basic transaction features
        features['amount'] = float(transaction['amount'])
        features['log_amount'] = np.log1p(features['amount'])
        
        # Temporal features
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['day_of_month'] = timestamp.day
        features['month'] = timestamp.month
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_business_hours'] = 1 if (9 <= timestamp.hour <= 17 and timestamp.weekday() < 5) else 0
        features['is_late_night'] = 1 if (timestamp.hour <= 5 or timestamp.hour >= 23) else 0
        
        # Amount features
        features['is_round_amount'] = 1 if features['amount'] % 1 == 0 else 0
        features['is_very_round'] = 1 if features['amount'] % 100 == 0 else 0
        
        # Merchant category features
        merchant_category = transaction.get('merchant_category', 'other')
        features['merchant_risk_score'] = self._get_merchant_risk_score(merchant_category)
        
        # One-hot encode merchant categories
        merchant_categories = ['grocery', 'gas_station', 'restaurant', 'retail', 'online', 
                             'atm', 'pharmacy', 'entertainment', 'travel', 'utilities']
        for category in merchant_categories:
            features[f'merchant_{category}'] = 1 if merchant_category == category else 0
        
        # Location features
        if 'latitude' in transaction and 'longitude' in transaction:
            features['latitude'] = transaction['latitude']
            features['longitude'] = transaction['longitude']
            features['distance_from_home'] = self._calculate_distance_from_home(
                features['latitude'], features['longitude'], transaction.get('customer_id')
            )
            features['is_far_from_home'] = 1 if features['distance_from_home'] > 50 else 0
            features['is_very_far_from_home'] = 1 if features['distance_from_home'] > 200 else 0
            features['home_lat'] = 40.0  # Default
            features['home_lon'] = -74.0
        else:
            features['distance_from_home'] = 0
            features['is_far_from_home'] = 0
            features['is_very_far_from_home'] = 0
            features['home_lat'] = 40.0
            features['home_lon'] = -74.0
        
        # Enhanced customer behavioral features
        if customer_history:
            customer_features = self._extract_customer_features(
                customer_history, timestamp, features['amount']
            )
            features.update(customer_features)
        else:
            features.update(self._get_default_customer_features(features['amount'], features['hour']))
        
        # Add fraud pattern detection features
        pattern_features = self._detect_fraud_patterns(features)
        features.update(pattern_features)
        
        # Add statistical anomaly detection features
        anomaly_features = self._detect_statistical_anomalies(features)
        features.update(anomaly_features)
        
        return features
    
    def _get_merchant_risk_score(self, category: str) -> float:
        """Get risk score for merchant category."""
        risk_scores = {
            'online': 0.8,
            'atm': 0.7,
            'travel': 0.6,
            'retail': 0.5,
            'entertainment': 0.4,
            'restaurant': 0.3,
            'gas_station': 0.3,
            'grocery': 0.2,
            'pharmacy': 0.2,
            'utilities': 0.1
        }
        return risk_scores.get(category, 0.5)
    
    def _calculate_distance_from_home(self, lat: float, lon: float, customer_id: str) -> float:
        """Calculate distance from customer's home location."""
        # Simplified - in production, you'd lookup actual customer home location
        home_lat, home_lon = 40.7128, -74.0060  # Default NYC
        
        # Haversine formula approximation
        lat_diff = lat - home_lat
        lon_diff = lon - home_lon
        distance = np.sqrt(lat_diff**2 + lon_diff**2) * 69  # Convert to miles
        
        return distance
    
    def _extract_customer_features(self, history: List[Dict], current_time: datetime, 
                                 current_amount: float) -> Dict[str, float]:
        """Extract comprehensive customer behavioral features."""
        features = {}
        
        if not history:
            return self._get_default_customer_features(current_amount, current_time.hour)
        
        # Basic statistics
        amounts = [t['amount'] for t in history]
        hours = [pd.to_datetime(t['timestamp']).hour for t in history]
        categories = [t.get('merchant_category', 'other') for t in history]
        
        features['customer_amount_mean'] = np.mean(amounts)
        features['customer_amount_std'] = np.std(amounts) if len(amounts) > 1 else 1.0
        features['customer_amount_median'] = np.median(amounts)
        features['customer_amount_min'] = np.min(amounts)
        features['customer_amount_max'] = np.max(amounts)
        features['customer_hour_mean'] = np.mean(hours)
        features['customer_hour_std'] = np.std(hours) if len(hours) > 1 else 1.0
        features['customer_merchant_category_nunique'] = len(set(categories))
        features['customer_timestamp_count'] = len(history)
        
        # Time-based analysis
        recent_1h = [t for t in history 
                    if (current_time - pd.to_datetime(t['timestamp'])).total_seconds() <= 3600]
        recent_24h = [t for t in history 
                     if (current_time - pd.to_datetime(t['timestamp'])).total_seconds() <= 86400]
        recent_7d = [t for t in history 
                    if (current_time - pd.to_datetime(t['timestamp'])).total_seconds() <= 604800]
        
        features['transactions_last_hour'] = len(recent_1h)
        features['transactions_last_day'] = len(recent_24h)
        features['transactions_last_week'] = len(recent_7d)
        
        features['amount_last_day'] = sum(t['amount'] for t in recent_24h)
        features['amount_last_week'] = sum(t['amount'] for t in recent_7d)
        
        features['transaction_velocity_day'] = features['transactions_last_day'] / 24
        features['transaction_velocity_week'] = features['transactions_last_week'] / 168
        
        # Time since last transaction
        if history:
            last_transaction_time = max(pd.to_datetime(t['timestamp']) for t in history)
            features['time_since_last_transaction'] = (current_time - last_transaction_time).total_seconds() / 3600
        else:
            features['time_since_last_transaction'] = 24
        
        # Derived features
        features['amount_vs_customer_mean'] = current_amount / (features['customer_amount_mean'] + 1e-6)
        features['amount_vs_customer_median'] = current_amount / (features['customer_amount_median'] + 1e-6)
        features['amount_zscore'] = ((current_amount - features['customer_amount_mean']) / 
                                   (features['customer_amount_std'] + 1e-6))
        features['hour_vs_customer_mean'] = abs(current_time.hour - features['customer_hour_mean'])
        
        # Sequence features
        features['transaction_sequence'] = features['customer_timestamp_count']
        features['days_since_first_transaction'] = 1  # Simplified
        features['is_new_merchant_category'] = 0  # Simplified
        
        return features
    
    def _get_default_customer_features(self, amount: float, hour: int) -> Dict[str, float]:
        """Get default customer features when no history is available."""
        return {
            'customer_amount_mean': amount,
            'customer_amount_std': 1.0,
            'customer_amount_median': amount,
            'customer_amount_min': amount,
            'customer_amount_max': amount,
            'customer_hour_mean': hour,
            'customer_hour_std': 1.0,
            'customer_merchant_category_nunique': 1,
            'customer_timestamp_count': 1,
            'transactions_last_hour': 0,
            'transactions_last_day': 0,
            'transactions_last_week': 0,
            'amount_last_day': 0,
            'amount_last_week': 0,
            'transaction_velocity_day': 0,
            'transaction_velocity_week': 0,
            'time_since_last_transaction': 24,
            'amount_vs_customer_mean': 1.0,
            'amount_vs_customer_median': 1.0,
            'amount_zscore': 0.0,
            'hour_vs_customer_mean': 0.0,
            'transaction_sequence': 1,
            'days_since_first_transaction': 1,
            'is_new_merchant_category': 0
        }
    
    def _detect_fraud_patterns(self, features: Dict[str, float]) -> Dict[str, float]:
        """Detect fraud patterns in the transaction."""
        pattern_features = {}
        
        for pattern_name, pattern_rule in self.pattern_rules.items():
            try:
                pattern_detected = pattern_rule(features)
                pattern_features[f'pattern_{pattern_name}'] = 1 if pattern_detected else 0
            except Exception:
                pattern_features[f'pattern_{pattern_name}'] = 0
        
        return pattern_features
    
    def _detect_statistical_anomalies(self, features: Dict[str, float]) -> Dict[str, float]:
        """Detect statistical anomalies in transaction features."""
        anomaly_features = {}
        
        # Amount-based anomalies
        amount = features.get('amount', 0)
        customer_mean = features.get('customer_amount_mean', amount)
        customer_std = features.get('customer_amount_std', 1.0)
        
        # Z-score anomaly
        amount_zscore = abs((amount - customer_mean) / (customer_std + 1e-6))
        anomaly_features['amount_zscore'] = amount_zscore
        anomaly_features['amount_zscore_high'] = 1 if amount_zscore > 3 else 0
        
        # Modified Z-score (more robust)
        median_amount = features.get('customer_amount_median', amount)
        mad = abs(amount - median_amount)
        modified_zscore = 0.6745 * mad / (customer_std + 1e-6)
        anomaly_features['amount_modified_zscore'] = modified_zscore
        
        # Percentile-based anomaly
        if amount > customer_mean:
            percentile = min(99, 50 + (amount - customer_mean) / customer_std * 15)
        else:
            percentile = max(1, 50 - (customer_mean - amount) / customer_std * 15)
        anomaly_features['amount_percentile'] = percentile
        anomaly_features['amount_extreme_percentile'] = 1 if percentile > 95 or percentile < 5 else 0
        
        # Velocity anomalies
        velocity_day = features.get('transaction_velocity_day', 0)
        velocity_week = features.get('transaction_velocity_week', 0)
        
        anomaly_features['velocity_day_high'] = 1 if velocity_day > 5 else 0
        anomaly_features['velocity_week_high'] = 1 if velocity_week > 2 else 0
        
        # Time anomalies
        hour_deviation = features.get('hour_vs_customer_mean', 0)
        anomaly_features['time_anomaly'] = 1 if hour_deviation > 8 else 0
        
        # Distance anomalies
        distance = features.get('distance_from_home', 0)
        anomaly_features['location_anomaly'] = 1 if distance > 500 else 0
        
        return anomaly_features
    
    def predict_with_enhanced_scoring(self, transaction: Dict[str, Any], 
                                    customer_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Predict fraud probability with enhanced multi-tier risk scoring.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Extract enhanced features
        features = self.extract_enhanced_features(transaction, customer_history)
        
        # Ensure all required features are present
        feature_vector = []
        for feature_name in self.feature_columns:
            feature_vector.append(features.get(feature_name, 0))
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make model prediction
        model_prediction = self.model.predict(X)[0]
        model_probability = self.model.predict_proba(X)[0, 1]
        
        # Calculate pattern score
        pattern_score = self._calculate_pattern_score(features)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(features)
        
        # Calculate velocity score
        velocity_score = self._calculate_velocity_score(features)
        
        # Calculate composite risk score
        composite_score = (
            self.risk_weights['model_probability'] * model_probability +
            self.risk_weights['pattern_score'] * pattern_score +
            self.risk_weights['anomaly_score'] * anomaly_score +
            self.risk_weights['velocity_score'] * velocity_score
        )
        
        # Determine final risk level
        risk_level = self._determine_risk_level(composite_score)
        
        # Get detailed risk breakdown
        risk_breakdown = {
            'model_probability': model_probability,
            'pattern_score': pattern_score,
            'anomaly_score': anomaly_score,
            'velocity_score': velocity_score,
            'composite_score': composite_score
        }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(composite_score, risk_level)
        
        return {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'customer_id': transaction.get('customer_id', 'unknown'),
            'timestamp': transaction.get('timestamp'),
            'amount': transaction.get('amount'),
            'is_fraud': bool(model_prediction),
            'fraud_probability': float(model_probability),
            'composite_risk_score': float(composite_score),
            'risk_level': risk_level,
            'risk_breakdown': risk_breakdown,
            'recommendation': recommendation,
            'patterns_detected': self._get_detected_patterns(features),
            'anomalies_detected': self._get_detected_anomalies(features),
            'baseline_comparison': self._compare_to_baseline(features)
        }
    
    def _calculate_pattern_score(self, features: Dict[str, float]) -> float:
        """Calculate fraud pattern score (0-1)."""
        pattern_keys = [k for k in features.keys() if k.startswith('pattern_')]
        if not pattern_keys:
            return 0.0
        
        # Weighted pattern scoring
        pattern_weights = {
            'pattern_card_testing': 0.9,
            'pattern_account_takeover': 0.85,
            'pattern_bust_out_fraud': 0.8,
            'pattern_velocity_abuse': 0.75,
            'pattern_geographic_anomaly': 0.7,
            'pattern_round_amount_scheme': 0.6,
            'pattern_synthetic_identity': 0.7
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for pattern_key in pattern_keys:
            if features[pattern_key] == 1:
                weight = pattern_weights.get(pattern_key, 0.5)
                total_score += weight
                total_weight += weight
        
        return min(1.0, total_score)
    
    def _calculate_anomaly_score(self, features: Dict[str, float]) -> float:
        """Calculate statistical anomaly score (0-1)."""
        anomaly_indicators = [
            features.get('amount_zscore_high', 0),
            features.get('amount_extreme_percentile', 0),
            features.get('velocity_day_high', 0),
            features.get('velocity_week_high', 0),
            features.get('time_anomaly', 0),
            features.get('location_anomaly', 0)
        ]
        
        # Weighted anomaly scoring
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        
        score = sum(indicator * weight for indicator, weight in zip(anomaly_indicators, weights))
        return min(1.0, score)
    
    def _calculate_velocity_score(self, features: Dict[str, float]) -> float:
        """Calculate transaction velocity score (0-1)."""
        # Normalize velocity metrics
        velocity_day = features.get('transaction_velocity_day', 0)
        velocity_week = features.get('transaction_velocity_week', 0)
        transactions_hour = features.get('transactions_last_hour', 0)
        
        # Score based on exceeding normal thresholds
        day_score = min(1.0, velocity_day / 10.0)  # Normalize to 10 transactions/day max
        week_score = min(1.0, velocity_week / 5.0)  # Normalize to 5 transactions/day average
        hour_score = min(1.0, transactions_hour / 10.0)  # Normalize to 10 transactions/hour max
        
        return max(day_score, week_score, hour_score)
    
    def _determine_risk_level(self, composite_score: float) -> str:
        """Determine risk level based on composite score."""
        if composite_score >= self.risk_thresholds['very_high']:
            return "VERY_HIGH"
        elif composite_score >= self.risk_thresholds['high']:
            return "HIGH"
        elif composite_score >= self.risk_thresholds['medium']:
            return "MEDIUM"
        elif composite_score >= self.risk_thresholds['low']:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_recommendation(self, composite_score: float, risk_level: str) -> Dict[str, str]:
        """Generate action recommendation based on risk level."""
        recommendations = {
            "VERY_HIGH": {
                "action": "BLOCK",
                "description": "Block transaction immediately and flag for investigation",
                "details": "High probability fraud - multiple risk indicators triggered"
            },
            "HIGH": {
                "action": "VERIFY",
                "description": "Require additional verification before processing",
                "details": "Suspicious patterns detected - customer verification recommended"
            },
            "MEDIUM": {
                "action": "MONITOR",
                "description": "Process but monitor closely for patterns",
                "details": "Some risk indicators present - enhanced monitoring advised"
            },
            "LOW": {
                "action": "APPROVE",
                "description": "Process normally with standard monitoring",
                "details": "Low risk - normal transaction patterns"
            },
            "VERY_LOW": {
                "action": "APPROVE",
                "description": "Process normally",
                "details": "Very low risk - typical customer behavior"
            }
        }
        
        return recommendations.get(risk_level, recommendations["MEDIUM"])
    
    def _get_detected_patterns(self, features: Dict[str, float]) -> List[str]:
        """Get list of detected fraud patterns."""
        detected = []
        for pattern_name, pattern_rule in self.pattern_rules.items():
            if features.get(f'pattern_{pattern_name}', 0) == 1:
                detected.append(pattern_name)
        return detected
    
    def _get_detected_anomalies(self, features: Dict[str, float]) -> List[str]:
        """Get list of detected anomalies."""
        detected = []
        
        if features.get('amount_zscore_high', 0) == 1:
            detected.append('amount_outlier')
        if features.get('velocity_day_high', 0) == 1:
            detected.append('high_daily_velocity')
        if features.get('velocity_week_high', 0) == 1:
            detected.append('high_weekly_velocity')
        if features.get('time_anomaly', 0) == 1:
            detected.append('unusual_time')
        if features.get('location_anomaly', 0) == 1:
            detected.append('unusual_location')
            
        return detected
    
    def _compare_to_baseline(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Compare transaction to baseline normal behavior."""
        comparison = {}
        
        # Amount comparison
        amount = features.get('amount', 0)
        baseline_amount = self.baseline_stats['avg_amount']
        comparison['amount_vs_baseline'] = {
            'current': amount,
            'baseline': baseline_amount,
            'ratio': amount / baseline_amount,
            'deviation': 'high' if amount > baseline_amount * 2 else 'normal'
        }
        
        # Hour comparison
        hour = features.get('hour', 12)
        baseline_hour = self.baseline_stats['avg_hour']
        comparison['hour_vs_baseline'] = {
            'current': hour,
            'baseline': baseline_hour,
            'difference': abs(hour - baseline_hour),
            'deviation': 'high' if abs(hour - baseline_hour) > 8 else 'normal'
        }
        
        # Velocity comparison
        velocity = features.get('transaction_velocity_day', 0)
        baseline_velocity = self.baseline_stats['avg_transactions_per_day']
        comparison['velocity_vs_baseline'] = {
            'current': velocity,
            'baseline': baseline_velocity,
            'ratio': velocity / (baseline_velocity + 1e-6),
            'deviation': 'high' if velocity > baseline_velocity * 3 else 'normal'
        }
        
        return comparison
    
    def create_enhanced_fraud_report(self, transactions: List[Dict[str, Any]], 
                                   customer_histories: Dict[str, List[Dict]] = None) -> Dict[str, Any]:
        """Create comprehensive fraud detection report with enhanced analytics."""
        print("📊 Creating enhanced fraud detection report...")
        
        # Get enhanced predictions
        predictions = []
        for transaction in transactions:
            customer_id = transaction.get('customer_id')
            customer_history = None
            
            if customer_histories and customer_id in customer_histories:
                customer_history = customer_histories[customer_id]
            
            result = self.predict_with_enhanced_scoring(transaction, customer_history)
            predictions.append(result)
        
        # Calculate comprehensive statistics
        total_transactions = len(predictions)
        risk_distributions = {
            'VERY_HIGH': sum(1 for p in predictions if p['risk_level'] == 'VERY_HIGH'),
            'HIGH': sum(1 for p in predictions if p['risk_level'] == 'HIGH'),
            'MEDIUM': sum(1 for p in predictions if p['risk_level'] == 'MEDIUM'),
            'LOW': sum(1 for p in predictions if p['risk_level'] == 'LOW'),
            'VERY_LOW': sum(1 for p in predictions if p['risk_level'] == 'VERY_LOW')
        }
        
        # Pattern analysis
        all_patterns = []
        for prediction in predictions:
            all_patterns.extend(prediction['patterns_detected'])
        
        pattern_frequency = {}
        for pattern in all_patterns:
            pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
        
        # Anomaly analysis
        all_anomalies = []
        for prediction in predictions:
            all_anomalies.extend(prediction['anomalies_detected'])
        
        anomaly_frequency = {}
        for anomaly in all_anomalies:
            anomaly_frequency[anomaly] = anomaly_frequency.get(anomaly, 0) + 1
        
        # Financial impact analysis
        total_amount = sum(t['amount'] for t in transactions)
        
        high_risk_amount = sum(
            t['amount'] for t, p in zip(transactions, predictions) 
            if p['risk_level'] in ['VERY_HIGH', 'HIGH']
        )
        
        # Action recommendations
        action_distribution = {}
        for prediction in predictions:
            action = prediction['recommendation']['action']
            action_distribution[action] = action_distribution.get(action, 0) + 1
        
        # Create comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_transactions': total_transactions,
                'total_amount': total_amount,
                'high_risk_transactions': risk_distributions['VERY_HIGH'] + risk_distributions['HIGH'],
                'high_risk_amount': high_risk_amount,
                'overall_fraud_rate': (risk_distributions['VERY_HIGH'] + risk_distributions['HIGH']) / total_transactions,
                'amount_at_risk_percentage': high_risk_amount / total_amount if total_amount > 0 else 0
            },
            'risk_distribution': risk_distributions,
            'pattern_analysis': {
                'patterns_detected': len(pattern_frequency),
                'pattern_frequency': pattern_frequency,
                'most_common_pattern': max(pattern_frequency.keys(), key=pattern_frequency.get) if pattern_frequency else None
            },
            'anomaly_analysis': {
                'anomalies_detected': len(anomaly_frequency),
                'anomaly_frequency': anomaly_frequency,
                'most_common_anomaly': max(anomaly_frequency.keys(), key=anomaly_frequency.get) if anomaly_frequency else None
            },
            'action_recommendations': action_distribution,
            'high_risk_transactions': [
                p for p in predictions 
                if p['risk_level'] in ['VERY_HIGH', 'HIGH']
            ][:20],  # Top 20 high-risk transactions
            'model_info': {
                'model_name': self.model_name,
                'features_used': len(self.feature_columns),
                'fraud_patterns': len(self.fraud_patterns),
                'risk_thresholds': self.risk_thresholds
            }
        }
        
        return report

def main():
    """Main function to demonstrate enhanced fraud detection prediction."""
    print("🚀 Enhanced Fraud Detection Prediction Interface")
    print("=" * 60)
    
    # This would be used in the main application
    print("✅ Enhanced fraud detection prediction interface ready!")
    print("Features:")
    print("• Real-time fraud pattern detection")
    print("• Statistical anomaly detection")
    print("• Multi-tier risk scoring")
    print("• Enhanced explainable AI")
    print("• Comprehensive fraud reporting")

if __name__ == "__main__":
    main()
