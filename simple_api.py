#!/usr/bin/env python3
"""
Simple CyberShield API Server for Testing
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'operational',
        'version': '2.1.0',
        'model_loaded': True,
        'features_count': 7,
        'timestamp': datetime.now().isoformat(),
        'server': 'CyberShield Simple API'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json() or {}
        amount = data.get('amount', 100)
        hour = data.get('transaction_hour', 12)
        
        # Simple fraud probability calculation
        amount_risk = min(amount / 1000, 1.0)
        time_risk = 1.0 if hour < 6 or hour > 22 else 0.3
        fraud_prob = (amount_risk * 0.6 + time_risk * 0.4) * random.uniform(0.7, 1.3)
        fraud_prob = max(0.0, min(1.0, fraud_prob))
        
        risk_level = 'CRITICAL' if fraud_prob >= 0.85 else 'HIGH' if fraud_prob >= 0.65 else 'MEDIUM' if fraud_prob >= 0.35 else 'LOW'
        
        return jsonify({
            'fraud_probability': round(fraud_prob, 3),
            'is_fraud': fraud_prob > 0.5,
            'risk_level': risk_level,
            'confidence': 0.9,
            'transaction_id': f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'analysis_timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json() or {}
        batch_size = data.get('batch_size', 100)
        
        fraud_count = random.randint(2, batch_size // 10)
        
        return jsonify({
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_transactions': batch_size,
            'fraud_detected': fraud_count,
            'fraud_rate_percent': round((fraud_count / batch_size) * 100, 2),
            'processing_time_seconds': round(random.uniform(1.0, 3.0), 2),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Simple CyberShield API...")
    print("🌐 API running on http://localhost:5001/api")
    app.run(debug=True, host='0.0.0.0', port=5001)
