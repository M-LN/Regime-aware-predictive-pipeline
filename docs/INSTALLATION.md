# Getting Started Guide

## Quick Setup (5 minutes)

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Installation

1. **Clone and navigate to project**
```bash
cd regime-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run quickstart**
```bash
python quickstart.py
```

This will:
- Generate 30 days of synthetic energy data
- Engineer features (rolling stats, volatility, trends)
- Train HMM regime detector (3 regimes)
- Save trained model

**Output:** Regime detector saved to `data/models/regime_detector.pkl`

---

## Starting the API Server

```bash
uvicorn src.inference.api:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Testing the Predictions

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "wind_speed": 8.5,
    "energy_production": 400,
    "temperature": 12,
    "price": 150
  }'
```

### Response Example
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

### Health Check
```bash
curl http://localhost:8000/health
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H 'Content-Type: application/json' \
  -d '{
    "data": [
      {"wind_speed": 8.5, "energy_production": 400, "temperature": 12, "price": 150},
      {"wind_speed": 12.0, "energy_production": 600, "temperature": 10, "price": 120}
    ]
  }'
```

---

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_ingestion.py -v

# Run specific test
pytest tests/test_api.py::TestAPIEndpoints::test_prediction_endpoint -v
```

---

## Docker Usage

### Build Docker Image
```bash
docker build -t regime-ai:latest .
```

### Run Docker Container
```bash
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  regime-ai:latest
```

### Docker Compose (with MLflow)
```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### Prerequisites
- kubectl configured
- Kubernetes cluster ready

### Deploy
```bash
# Create namespace
kubectl create namespace ml-pipeline

# Apply configurations
kubectl apply -f k8s/configmap.yaml -n ml-pipeline
kubectl apply -f k8s/deployment.yaml -n ml-pipeline
kubectl apply -f k8s/service.yaml -n ml-pipeline

# Check status
kubectl get pods -n ml-pipeline
```

### Port Forward
```bash
kubectl port-forward svc/regime-predictor 8000:8000 -n ml-pipeline
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and modify:

```bash
cp .env.example .env
```

Key variables:
- `ENVIRONMENT`: development/production
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `N_REGIMES`: Number of regimes (default 3)
- `DEBUG`: Enable debug logging

### Real Data Ingestion (Energi Data Service)

Set the following in `.env` to enable real ingestion:

```bash
EDS_BASE_URL=https://your-eds-api
EDS_API_KEY=your_api_key
EDS_ENERGY_ENDPOINT=/energy
EDS_WEATHER_ENDPOINT=/weather
EDS_PRICE_ENDPOINT=/prices
```

Run ingestion locally:

```bash
python -m src.ingestion.run_ingestion --lookback-hours 24
```

For scheduled runs in GitHub Actions, add repository secrets:

- `EDS_BASE_URL`
- `EDS_API_KEY`

### Config File

Edit `config.yaml` for detailed configuration:

```yaml
regime:
  n_regimes: 3
  hmm_n_iter: 100

features:
  rolling_windows: [1, 6, 24]
  volatility_windows: [6, 24]
```

---

## Training Models

Once regime detector is trained, train per-regime models:

```python
# Example: Train XGBoost for Regime A
from src.training.train_regime_a import train_regime_a_xgboost

train_regime_a_xgboost(
    data_path="data/features",
    output_path="data/models",
    test_size=0.2
)
```

---

## Monitoring

### Check API Metrics
```bash
curl http://localhost:8000/metrics
```

### MLflow Dashboard
```bash
mlflow ui
# Navigate to http://localhost:5000
```

### Prometheus Metrics
The API exports Prometheus metrics at `/metrics`

---

## Next Steps

1. **Connect to real data source**: Modify `src/ingestion/data_fetcher.py` to connect to your API
2. **Train production models**: Use `src/training/` modules to train per-regime models
3. **Set up monitoring**: Configure drift detection alerts in `src/monitoring/`
4. **Deploy to cloud**: Use Docker images with CI/CD pipelines

---

## Troubleshooting

### ModuleNotFoundError
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Port Already in Use
```bash
# Change port
uvicorn src.inference.api:app --port 8001
```

### MLflow Connection Error
```bash
# Start MLflow locally
mlflow server --backend-store-uri sqlite:///mlflow.db

# Or set to local backend
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## Resources

- [API Documentation](./API.md)
- [Architecture Guide](../ARCHITECTURE.md)
- [IBM Alignment Document](../IBM_ALIGNMENT.md)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [HMMLearn Docs](https://hmmlearn.readthedocs.io/)
- [MLflow Docs](https://mlflow.org/docs/)

---

**Last Updated:** 2026-02-13
