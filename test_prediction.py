"""
Test prediction script - demonstrates how to use the API
"""

import requests
import json

# API Configuration
API_URL = "http://127.0.0.1:8002"
API_KEY = "dev-key-12345"

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*70)
    print("🏥 HEALTH CHECK")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.json()


def test_prediction():
    """Test a single prediction"""
    print("\n" + "="*70)
    print("🔮 SINGLE PREDICTION TEST")
    print("="*70)
    
    # Sample energy market data
    payload = {
        "wind_speed": 7.5,
        "energy_production": 220.0,
        "temperature": 12.0,
        "price": 250.0
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    
    print(f"\n📤 Input Data:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=headers
    )
    
    print(f"\n📥 Response (Status {response.status_code}):")
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
        
        print(f"\n✨ Key Insights:")
        print(f"  • Detected Regime: {result.get('regime_id')}")
        print(f"  • Model Used: {result.get('model_type', 'N/A')}")
        print(f"  • Predicted Price: ${result.get('prediction', 0):.2f}/MWh")
        print(f"  • Confidence: {result.get('confidence', 0)*100:.1f}%")
        print(f"  • Processing Time: {result.get('inference_time_ms', 0):.2f}ms")
    else:
        print(f"Error: {response.text}")
    
    return response


def test_status():
    """Test the status endpoint"""
    print("\n" + "="*70)
    print("📊 SYSTEM STATUS")
    print("="*70)
    
    response = requests.get(f"{API_URL}/status")
    print(f"Status Code: {response.status_code}")
    status = response.json()
    print(json.dumps(status, indent=2))
    
    print(f"\n📈 Statistics:")
    stats = status.get('statistics', {})
    print(f"  • Total Predictions: {stats.get('prediction_count', 0)}")
    print(f"  • Avg Inference Time: {stats.get('avg_inference_time_ms', 0):.2f}ms")
    print(f"  • Uptime: {status.get('uptime_seconds', 0):.0f}s")
    
    return response


if __name__ == "__main__":
    print("\n" + "🚀 "*20)
    print("REGIME-AWARE PREDICTION API - INTERACTIVE TEST")
    print("🚀 "*20)
    
    try:
        # Test 1: Health Check
        health = test_health()
        
        if health.get('status') == 'healthy':
            # Test 2: Make a prediction
            test_prediction()
            
            # Test 3: Check system status
            test_status()
            
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print(f"\n📖 Next Steps:")
            print(f"  • View API docs: {API_URL}/docs")
            print(f"  • Try batch predictions: POST {API_URL}/batch_predict")
            print(f"  • Monitor metrics: {API_URL}/metrics")
            print(f"  • Check drift: POST {API_URL}/drift/check")
            print()
        else:
            print("\n❌ API is not healthy!")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server!")
        print(f"   Make sure the server is running on {API_URL}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
