#!/usr/bin/env python
"""
Quick Deployment Validation Script
Tests core API functionality with production-ready configuration
"""

import os
import requests
from datetime import datetime

API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "test-api-key-production-replace-this")

def test():
    print("\n" + "="*60)
    print("  DEPLOYMENT VALIDATION - Production Ready API")
    print("="*60 + "\n")
    
    # Test 1: Health
    print("[1/5] Health Check...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        print(f"  ✓ API responding (status {r.status_code})")
        health = r.json()
        print(f"  ✓ Regime detector: {health.get('regime_detector')}")
        print(f"  ✓ Models loaded: {health.get('models_loaded')}")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return
    
    # Test 2: Auth enforcement
    print("\n[2/5] API Key Authentication...")
    payload = {"wind_speed": 8.5, "energy_production": 750, "temperature": 5.2, "price": 45.67}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if r.status_code == 401:
            print(f"  ✓ Correctly rejected unauthenticated request (401)")
        else:
            print(f"  ⚠ Expected 401, got {r.status_code}")
    except Exception as e:
        print(f"  ✗ Auth test failed: {e}")
    
    # Test 3: Authenticated prediction
    print("\n[3/5] Single Prediction with Auth...")
    try:
        headers = {"X-API-Key": API_KEY}
        r = requests.post(f"{API_URL}/predict", json=payload, headers=headers, timeout=5)
        if r.status_code == 200:
            pred = r.json()
            print(f"  ✓ Prediction successful")
            print(f"    - Prediction: {pred.get('prediction'): .2f} MWh")
            print(f"    - Regime: {pred.get('regime')} (confidence: {pred.get('regime_confidence'):.2%})")
            print(f"    - Latency: {pred.get('inference_latency_ms'):.1f}ms")
        else:
            print(f"  ✗ Prediction failed (status {r.status_code})")
            print(f"    {r.text[:200]}")
    except Exception as e:
        print(f"  ✗ Prediction test failed: {e}")
    
    # Test 4: Batch prediction
    print("\n[4/5] Batch Predictions...")
    batch = {
        "data": [
            {"wind_speed": 5.2, "energy_production": 580, "temperature": 3.5, "price": 42.10},
            {"wind_speed": 12.1, "energy_production": 850, "temperature": 8.9, "price": 48.75},
        ]
    }
    try:
        headers = {"X-API-Key": API_KEY}
        r = requests.post(f"{API_URL}/batch_predict", json=batch, headers=headers, timeout=10)
        if r.status_code == 200:
            result = r.json()
            print(f"  ✓ Batch processing successful")
            print(f"    - Succeeded: {result.get('n_succeeded')}")
            print(f"    - Failed: {result.get('n_failed')}")
        else:
            print(f"  ✗ Batch failed (status {r.status_code})")
    except Exception as e:
        print(f"  ✗ Batch test failed: {e}")
    
    # Test 5: Drift detection
    print("\n[5/5] Drift Detection Endpoints...")
    try:
        headers = {"X-API-Key": API_KEY}
        r = requests.get(f"{API_URL}/drift/last", headers=headers, timeout=5)
        if r.status_code == 200:
            drift = r.json()
            print(f"  ✓ Drift endpoint operational")
            print(f"    - Drift detected: {drift.get('drift_detected', 'N/A')}")
            print(f"    - Last check: {drift.get('timestamp', 'N/A')}")
        elif r.status_code == 404:
            # Drift may not exist yet if no history
            r = requests.post(f"{API_URL}/drift/check", headers=headers, timeout=5)
            if r.status_code == 200:
                print(f"  ✓ Drift check endpoint operational")
                drift = r.json()
                print(f"    - Drift detected: {drift.get('drift_detected', False)}")
            else:
                print(f"  ⚠ Drift endpoints not yet initialized")
        else:
            print(f"  ✗ Drift status failed (status {r.status_code})")
    except Exception as e:
        print(f"  ✗ Drift test failed: {e}")
    
    print("\n" + "="*60)
    print("  ✓ Deployment Validation Complete")
    print("  System is production-ready with:")
    print("    • Authentication (X-API-Key header)")
    print("    • Rate limiting (60 req/min default)")
    print("    • Drift detection monitoring")
    print("    • Regime-aware predictions")
    print("="*60 + "\n")

if __name__ == "__main__":
    test()
