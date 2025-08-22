import json
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler
import pickle
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "CyberShield AI Fraud Detection API",
            "status": "active",
            "version": "2.0",
            "endpoints": ["/api/predict"]
        }
        
        self.wfile.write(json.dumps(response).encode())
        return

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Simple fraud detection logic
            transaction_data = data.get('transaction', {})
            
            # Extract features
            amount = float(transaction_data.get('amount', 0))
            merchant_category = transaction_data.get('merchant_category', 'unknown')
            hour = int(transaction_data.get('hour', 12))
            day_of_week = int(transaction_data.get('day_of_week', 1))
            
            # Simple rule-based fraud detection
            fraud_score = 0.0
            
            # High amount transactions
            if amount > 1000:
                fraud_score += 0.3
            elif amount > 5000:
                fraud_score += 0.5
            
            # Unusual hours
            if hour < 6 or hour > 22:
                fraud_score += 0.2
            
            # Weekend transactions
            if day_of_week in [6, 7]:
                fraud_score += 0.1
            
            # High-risk categories
            high_risk_categories = ['gambling', 'adult_entertainment', 'cash_advance']
            if merchant_category.lower() in high_risk_categories:
                fraud_score += 0.4
            
            # Calculate final prediction
            is_fraud = fraud_score > 0.5
            confidence = min(fraud_score, 1.0)
            
            response = {
                "prediction": "fraud" if is_fraud else "legitimate",
                "confidence": round(confidence * 100, 2),
                "fraud_score": round(fraud_score, 3),
                "risk_level": "high" if fraud_score > 0.7 else "medium" if fraud_score > 0.3 else "low",
                "transaction_id": data.get('transaction_id', 'unknown'),
                "analysis": {
                    "amount_risk": amount > 1000,
                    "time_risk": hour < 6 or hour > 22,
                    "category_risk": merchant_category.lower() in high_risk_categories,
                    "weekend_risk": day_of_week in [6, 7]
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                "error": "Prediction failed",
                "message": str(e),
                "status": "error"
            }
            
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
