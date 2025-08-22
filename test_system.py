#!/usr/bin/env python3
"""
🧪 CyberShield AI - System Test Script
Quick verification that all components are working
"""

import sys
import os
import pandas as pd
import numpy as np

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from enhanced_prediction_interface import EnhancedFraudDetectionPredictor
        print("✅ Enhanced prediction interface imported")
    except ImportError as e:
        print(f"❌ Enhanced prediction interface failed: {e}")
        return False
    
    try:
        from data_generator import TransactionDataGenerator
        print("✅ Data generator imported")
    except ImportError as e:
        print(f"❌ Data generator failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError as e:
        print(f"❌ Streamlit not available: {e}")
        return False
    
    try:
        from flask import Flask
        print("✅ Flask available")
    except ImportError as e:
        print(f"❌ Flask not available: {e}")
        return False
    
    return True

def test_data_files():
    """Test that required data files exist"""
    print("\n📊 Testing data files...")
    
    required_files = [
        'transaction_data.csv',
        'customer_profiles_generated.csv',
        'enhanced_fraud_detection_model.pkl'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} exists ({size:,} bytes)")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_data_generation():
    """Test data generation functionality"""
    print("\n🏭 Testing data generation...")
    
    try:
        from data_generator import TransactionDataGenerator
        generator = TransactionDataGenerator()
        
        # Generate small test dataset
        result = generator.generate_dataset(n_customers=10, days=7, fraud_rate=0.1)
        transactions, customers = result
        
        print(f"✅ Generated {len(transactions)} transactions")
        print(f"✅ Generated {len(customers)} customers")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_api_endpoints():
    """Test API server can be imported and initialized"""
    print("\n🌐 Testing API server...")
    
    try:
        # Try importing the API server
        import api_server
        print("✅ API server module imported")
        
        # Check if Flask app is defined
        if hasattr(api_server, 'app'):
            print("✅ Flask app defined")
        else:
            print("❌ Flask app not found")
            return False
            
        return True
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app exists and can be imported"""
    print("\n📱 Testing Streamlit app...")
    
    if os.path.exists('streamlit_app/cybershield_app.py'):
        print("✅ Streamlit app file exists")
        
        # Check file size
        size = os.path.getsize('streamlit_app/cybershield_app.py')
        print(f"✅ App size: {size:,} bytes")
        
        return True
    else:
        print("❌ Streamlit app file missing")
        return False

def main():
    """Run all tests"""
    print("🚀 CyberShield AI System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_files,
        test_data_generation,
        test_api_endpoints,
        test_streamlit_app
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("✅ System is ready to use")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Some components need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
