# API Reference

## Base URL
```
http://localhost:8000
```

## Authentication
API key authentication is optional. If `API_KEY` is set, include the header:

```
X-API-Key: <your_api_key>
```

Requests without a valid key will return 401.

Rate limiting may return 429 when `RATE_LIMIT` is configured.

---

## Endpoints

### 1. GET /
**Get API Information**

Returns endpoint directory.

**Response:**
```json
{
  "name": "Regime-Aware Energy Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict (POST)",
    "batch_predict": "/batch_predict (POST)",
    "health": "/health (GET)",
    "registry_status": "/registry/status (GET)",
    "registry_models": "/registry/models (GET)",
    "registry_health": "/registry/health (GET)",
    "registry_reload": "/registry/reload (POST)",
    "metrics": "/metrics (GET)"
  }
}
```

---

### 2. GET /health
**Health Check**

Check if the API is healthy and ready.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "regime_detector": "loaded",
  "models_loaded": 3,
  "mlflow_connected": true,
  "timestamp": "2026-02-13T14:05:00.123Z"
}
```

**Response (503 Service Unavailable):**
```json
{
  "detail": "Service not ready: ..."
}
```

---

### 3. POST /predict
**Generate Single Prediction**
**Auth:** Requires `X-API-Key` if configured.

Get a single energy production prediction with regime detection.

**Request:**
```json
{
  "wind_speed": 8.5,
  "energy_production": 400,
  "temperature": 12,
  "price": 150,
  "timestamp": "2026-02-13T14:00:00Z"
}
```

**Parameters:**
- `wind_speed` (float, 0-30): Wind speed in m/s **[Required]**
- `energy_production` (float, 0-1000): Current energy production in MWh **[Required]**
- `temperature` (float, -40 to 50): Temperature in Celsius **[Required]**
- `price` (float, 0-500): Electricity price in DKK/MWh **[Required]**
- `timestamp` (string, ISO 8601): Optional timestamp. Defaults to current time.

**Response (200 OK):**
```json
{
  "prediction": 420.5,
  "unit": "MWh",
  "regime": "neutral",
  "regime_id": 1,
  "regime_confidence": 0.82,
  "model_version": "1.0.0",
  "inference_latency_ms": 45.2,
  "timestamp": "2026-02-13T14:05:32.123456"
}
```

**Response (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "loc": ["body", "wind_speed"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}

**Response (401 Unauthorized):**
```json
{
  "error": "Invalid API key",
  "timestamp": "2026-02-13T14:05:32.123456"
}
```

**Response (429 Too Many Requests):**
```json
{
  "error": "Rate limit exceeded",
  "timestamp": "2026-02-13T14:05:32.123456"
}
```
```

---

### 4. POST /batch_predict
**Generate Batch Predictions**
**Auth:** Requires `X-API-Key` if configured.
**Rate limit:** Returns 429 when the limit is exceeded.

Get predictions for multiple data points at once.

**Request:**
```json
{
  "data": [
    {
      "wind_speed": 8.5,
      "energy_production": 400,
      "temperature": 12,
      "price": 150
    },
    {
      "wind_speed": 12.0,
      "energy_production": 600,
      "temperature": 10,
      "price": 120
    },
    {
      "wind_speed": 3.2,
      "energy_production": 200,
      "temperature": 15,
      "price": 200
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "prediction": 420.5,
      "unit": "MWh",
      "regime": "neutral",
      "regime_id": 1,
      "regime_confidence": 0.82,
      "model_version": "1.0.0",
      "inference_latency_ms": 45.2,
      "timestamp": "2026-02-13T14:05:32.123456"
    },
    {
      "prediction": 630.1,
      "unit": "MWh",
      "regime": "volatile",
      "regime_id": 0,
      "regime_confidence": 0.87,
      "model_version": "1.0.0",
      "inference_latency_ms": 48.5,
      "timestamp": "2026-02-13T14:05:32.234567"
    },
    {
      "prediction": 210.3,
      "unit": "MWh",
      "regime": "stable",
      "regime_id": 2,
      "regime_confidence": 0.79,
      "model_version": "1.0.0",
      "inference_latency_ms": 42.1,
      "timestamp": "2026-02-13T14:05:32.345678"
    }
  ],
  "n_succeeded": 3,
  "n_failed": 0
}
```

---

### 5. GET /registry/health
**Registry Connectivity**

Checks MLflow registry connectivity and access.

**Auth:** Requires `X-API-Key` if configured.

**Response (200 OK):**
```json
{
  "connected": true,
  "access_ok": true,
  "model_count": 3,
  "timestamp": "2026-02-13T14:05:32.123456"
}
```

---

### 6. GET /registry/models
**List Registry Models**

Returns registered models and latest versions.

**Auth:** Requires `X-API-Key` if configured.

---

### 5. GET /status
**Get System Status**

Detailed status of the prediction system.

**Response:**
```json
{
  "ready": true,
  "config": {
    "n_regimes": 3,
    "feature_windows": [1, 6, 24]
  },
  "models": {
    "regime_detector": "hmm",
    "regime_models": ["xgboost", "lstm", "random_forest"]
  },
  "timestamp": "2026-02-13T14:05:00.123Z"
}
```

---

### 6. GET /metrics
**Prometheus Metrics**

Export metrics in Prometheus text format.

**Response:**
```
# HELP regime_predictions_total Total predictions made
# TYPE regime_predictions_total counter
regime_predictions_total{regime="0"} 1234
regime_predictions_total{regime="1"} 5678
regime_predictions_total{regime="2"} 3456

# HELP regime_inference_latency_seconds Inference latency in seconds
# TYPE regime_inference_latency_seconds histogram
regime_inference_latency_seconds_bucket{le="0.05"} 1200
regime_inference_latency_seconds_bucket{le="0.1"} 1234
```

---

## Regime Types

| ID | Name | Characteristics |
|----|------|---|
| 0 | `volatile` | High wind, variable market, rapid changes |
| 1 | `neutral` | Moderate conditions, stable patterns |
| 2 | `stable` | Low wind, predictable, calm market |

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid input parameters",
  "timestamp": "2026-02-13T14:05:00.123Z"
}
```

### 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "loc": ["body", "wind_speed"],
      "msg": "value must be between 0 and 30",
      "type": "value_error"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error: ...",
  "timestamp": "2026-02-13T14:05:00.123Z"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Service not ready: Models not loaded"
}
```

---

## Rate Limiting

Currently unlimited. Add rate limiting in production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app = limiter(app)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request): ...
```

---

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Examples

### Using curl
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "wind_speed": 8.5,
    "energy_production": 400,
    "temperature": 12,
    "price": 150
  }'

# Health check
curl http://localhost:8000/health
```

### Using Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "wind_speed": 8.5,
    "energy_production": 400,
    "temperature": 12,
    "price": 150
}

response = requests.post(url, json=data)
print(response.json())
```

### Using JavaScript/Node
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    wind_speed: 8.5,
    energy_production: 400,
    temperature: 12,
    price: 150
  })
});

const data = await response.json();
console.log(data);
```

---

## Versioning

Current API version: **1.0.0**

Breaking changes will increment major version (e.g., 2.0.0).

---

**Last Updated:** 2026-02-13
