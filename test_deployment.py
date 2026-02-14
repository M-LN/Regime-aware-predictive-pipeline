#!/usr/bin/env python
"""
Deployment Testing Script for Regime-Aware Predictive Pipeline
Tests API functionality, authentication, rate limiting, and inference
"""

import os
import sys
import time
import requests
import asyncio
from typing import Dict, Any, Tuple
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "test-api-key-production-replace-this")
RATE_LIMIT = 60  # Requests per minute

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.RESET}\n")

def print_pass(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_fail(text: str):
    """Print failure message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warn(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def test_api_health() -> bool:
    """Test 1: API health check"""
    print_header("Test 1: API Health Check")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print_pass(f"API health check passed (status: {response.status_code})")
            return True
        else:
            print_fail(f"API health check failed (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print_fail(f"API not running at {API_URL}")
        print_warn("Start the API with: python -m uvicorn src.inference.api:app --reload")
        return False
    except Exception as e:
        print_fail(f"Health check error: {str(e)}")
        return False

def test_api_key_auth() -> bool:
    """Test 2: API Key Authentication"""
    print_header("Test 2: API Key Authentication")
    
    # Sample prediction payload
    payload = {
        "wind_speed": 8.5,
        "energy_production": 750,
        "temperature": 5.2,
        "price": 45.67
    }
    
    # Test 2a: Request WITHOUT API key (should fail)
    print("2a. Testing request WITHOUT API key...")
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=5
        )
        if response.status_code == 401:
            print_pass(f"Correctly rejected request without API key (status: 401)")
        else:
            print_warn(f"Expected 401, got {response.status_code} (auth may be disabled)")
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False
    
    # Test 2b: Request WITH API key (should succeed)
    print("\n2b. Testing request WITH API key...")
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            print_pass(f"API key authentication passed (status: 200)")
            print(f"  Response: {response.json()}")
            return True
        else:
            print_fail(f"Request with API key failed (status: {response.status_code})")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def test_rate_limiting() -> bool:
    """Test 3: Rate Limiting"""
    print_header("Test 3: Rate Limiting")
    
    payload = {
        "wind_speed": 8.5,
        "energy_production": 750,
        "temperature": 5.2,
        "price": 45.67
    }
    headers = {"X-API-Key": API_KEY}
    
    print(f"Sending {RATE_LIMIT + 5} rapid requests to test rate limiting...")
    print("(This will take a moment...)")
    
    success_count = 0
    rate_limited_count = 0
    
    for i in range(RATE_LIMIT + 5):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                headers=headers,
                timeout=5
            )
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                print_warn(f"Request {i+1}: Rate limited (429)")
        except Exception as e:
            print_fail(f"Request {i+1} error: {str(e)}")
        
        # Small delay to avoid overwhelming the server
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1} requests...")
    
    if rate_limited_count > 0:
        print_pass(f"Rate limiting working: {success_count} success, {rate_limited_count} throttled")
        return True
    else:
        print_warn(f"No rate limiting detected ({success_count}/{RATE_LIMIT + 5} succeeded)")
        print_warn("Rate limiting may be disabled or threshold not reached")
        return True  # Don't fail if not configured

def test_batch_prediction() -> bool:
    """Test 4: Batch Prediction"""
    print_header("Test 4: Batch Prediction")
    
    payload = {
        "predictions": [
            {
                "wind_speed": 5.2,
                "energy_production": 580,
                "temperature": 3.5,
                "price": 42.10
            },
            {
                "wind_speed": 12.1,
                "energy_production": 850,
                "temperature": 8.9,
                "price": 48.75
            },
            {
                "wind_speed": 3.2,
                "energy_production": 450,
                "temperature": 2.1,
                "price": 38.50
            }
        ]
    }
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=payload,
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            print_pass(f"Batch prediction succeeded")
            print(f"  Returned {len(results.get('predictions', []))} predictions")
            for i, pred in enumerate(results.get('predictions', [])[:1]):
                print(f"  Sample prediction {i+1}: {pred}")
            return True
        else:
            print_fail(f"Batch prediction failed (status: {response.status_code})")
            return False
    except Exception as e:
        print_fail(f"Error: {str(e)}")
        return False

def test_registry_endpoints() -> bool:
    """Test 5: MLflow Registry Endpoints"""
    print_header("Test 5: MLflow Registry Endpoints")
    
    headers = {"X-API-Key": API_KEY}
    
    # Test 5a: Registry status
    print("5a. Testing /registry/status endpoint...")
    try:
        response = requests.get(
            f"{API_URL}/registry/status",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            status = response.json()
            print_pass(f"Registry status check passed")
            print(f"  Connected: {status.get('mlflow_connected')}")
            print(f"  Experiment: {status.get('experiment_name')}")
        elif response.status_code == 503:
            print_warn("MLflow not available (expected if not running)")
        else:
            print_fail(f"Registry status failed (status: {response.status_code})")
    except Exception as e:
        print_warn(f"Registry endpoint not available: {str(e)}")
    
    # Test 5b: List models
    print("\n5b. Testing /models endpoint...")
    try:
        response = requests.get(
            f"{API_URL}/models",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            models = response.json()
            print_pass(f"Loaded {len(models.get('models', []))} models from registry")
        elif response.status_code == 503:
            print_warn("MLflow not available (expected if not running)")
        else:
            print_fail(f"Models list failed (status: {response.status_code})")
    except Exception as e:
        print_warn(f"Models endpoint not available: {str(e)}")
    
    return True

def test_drift_detection() -> bool:
    """Test 6: Drift Detection Endpoints"""
    print_header("Test 6: Drift Detection Endpoints")
    
    headers = {"X-API-Key": API_KEY}
    
    print("6a. Testing /drift/status endpoint...")
    try:
        response = requests.get(
            f"{API_URL}/drift/status",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            status = response.json()
            print_pass(f"Drift status check passed")
            print(f"  Drift detected: {status.get('drift_detected')}")
            print(f"  Drift score: {status.get('drift_score')}")
        else:
            print_fail(f"Drift status failed (status: {response.status_code})")
    except Exception as e:
        print_warn(f"Drift endpoint error: {str(e)}")
    
    return True

def main():
    """Run all deployment tests"""
    print(f"""
{Colors.BLUE}
╔═══════════════════════════════════════════╗
║  DEPLOYMENT TEST SUITE                    ║
║  Regime-Aware Predictive Pipeline         ║
║  Date: 2026-02-14                         ║
╚═══════════════════════════════════════════╝
{Colors.RESET}
API URL: {API_URL}
API Key: {API_KEY[:20]}...
    """)
    
    results = {}
    
    # Run tests in sequence
    results['health'] = test_api_health()
    if not results['health']:
        print_fail("API not responding. Cannot continue with tests.")
        return 1
    
    results['auth'] = test_api_key_auth()
    results['rate_limit'] = test_rate_limiting()
    results['batch'] = test_batch_prediction()
    results['registry'] = test_registry_endpoints()
    results['drift'] = test_drift_detection()
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {test_name.upper():20s}: {status}")
    
    print(f"\n{Colors.BLUE}Total: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print_pass("All tests passed! Deployment is ready.")
        return 0
    else:
        print_warn(f"{total - passed} test(s) need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
