# 📘 Regime AI - Complete User Guide

> **Your Complete Guide to Using the Regime-Aware Energy Prediction System**

---

## 🚀 Quick Start (What You Just Did!)

You've already set up and tested the system! Here's what's running:

### ✅ **System Status**
- **API Server**: Running on http://127.0.0.1:8002
- **API Docs**: http://127.0.0.1:8002/docs (Interactive Swagger UI)
- **API Key**: `dev-key-12345` (for protected endpoints)
- **Models Loaded**: 2 (XGBoost for Regime 0, Random Forest for Regime 2)
- **Regime Detector**: HMM trained and loaded

---

## 📁 Where to Find Your Data

### 1. **Raw Data** (Ingested Energy Market Data)
**Location**: `data/raw/`

```
data/raw/
├── year=2025/
│   ├── month=9/
│   │   └── day=30/
│   └── month=10/
│       └── day=1/
└── failed/                    # Failed ingestion attempts
    ├── failed_20260214T184041.json
    └── ...
```

**View your data:**
```bash
# List all raw data files
ls data/raw/year=*/month=*/day=*

# View a specific file (if any)
cat data/raw/year=2025/month=10/day=1/data.parquet
```

### 2. **Engineered Features** (Processed Data)
**Location**: `data/features/`

This folder contains the 112+ engineered features created from raw data:
- Rolling statistics (mean, std, min, max)
- Volatility metrics
- Temporal features (hour, day, month)
- Domain-specific energy features

### 3. **Trained Models** (Your ML Models)
**Location**: `data/models/`

```
data/models/
├── regime_detector.pkl              # HMM regime detector
├── regime_0_xgboost.pkl            # XGBoost for high wind regime
├── regime_1_lstm.keras             # LSTM for neutral regime
└── regime_2_random_forest.pkl      # Random Forest for volatile regime
```

### 4. **MLflow Tracking** (Experiment History)
**Location**: `mlruns/`

All your experiments, metrics, and model versions are tracked here.

**View experiments:**
```bash
# Start MLflow UI
mlflow ui

# Open in browser
http://localhost:5000
```

---

## 🔮 How to View Predictions

### **Method 1: Interactive API Documentation (Easiest!)**

**Open in your browser:**
```
http://127.0.0.1:8002/docs
```

**Steps:**
1. Click on **`/predict`** endpoint
2. Click **"Try it out"**
3. Click the **🔒 Authorize** button (top right)
4. Enter API Key: `dev-key-12345`
5. Click **"Authorize"** → **"Close"**
6. Fill in the request body:
```json
{
  "wind_speed": 7.5,
  "energy_production": 220.0,
  "temperature": 12.0,
  "price": 250.0
}
```
7. Click **"Execute"**
8. View the prediction in the **Response body**!

### **Method 2: Python Script**

Run the test script:
```bash
python test_prediction.py
```

**Output shows:**
- ✨ Detected regime
- 📊 Predicted price
- 🎯 Model used
- ⏱️ Processing time

### **Method 3: Using curl (Command Line)**

```bash
curl -X POST http://127.0.0.1:8002/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{
    "wind_speed": 7.5,
    "energy_production": 220.0,
    "temperature": 12.0,
    "price": 250.0
  }'
```

### **Method 4: Using PowerShell**

```powershell
$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = "dev-key-12345"
}

$body = @{
    wind_speed = 7.5
    energy_production = 220.0
    temperature = 12.0
    price = 250.0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8002/predict -Method Post -Headers $headers -Body $body
```

### **Method 5: Python Code**

```python
import requests

# Make a prediction
response = requests.post(
    "http://127.0.0.1:8002/predict",
    headers={"X-API-Key": "dev-key-12345"},
    json={
        "wind_speed": 7.5,
        "energy_production": 220.0,
        "temperature": 12.0,
        "price": 250.0
    }
)

result = response.json()
print(f"Predicted Price: ${result['prediction']:.2f}/MWh")
print(f"Regime Detected: {result['regime']}")
print(f"Confidence: {result['regime_confidence']*100:.1f}%")
```

---

## 📊 How to View System Status & Metrics

### **1. Health Check**
```bash
# Browser
http://127.0.0.1:8002/health

# curl
curl http://127.0.0.1:8002/health
```

**Response:**
```json
{
  "status": "healthy",
  "regime_detector": "loaded",
  "models_loaded": 2,
  "mlflow_connected": true,
  "timestamp": "2026-02-15T15:00:16.863797"
}
```

### **2. System Status (Detailed)**
```bash
# Browser
http://127.0.0.1:8002/status

# curl
curl http://127.0.0.1:8002/status
```

Shows:
- Configuration details
- Loaded models
- Monitoring status
- Statistics

### **3. Prometheus Metrics**
```bash
# Browser
http://127.0.0.1:8002/metrics

# curl
curl http://127.0.0.1:8002/metrics
```

Track:
- Total predictions
- Average inference time
- Regime distribution
- Error rates

---

## 🎮 Complete Setup Instructions

### **Initial Setup (Already Done!)**

1. ✅ **Installed Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. ✅ **Ran Demo**
   ```bash
   python demo_minimal.py    # Basic demo
   python demo_api.py        # Full pipeline demo
   ```

3. ✅ **Started API Server**
   ```bash
   .\start_api.ps1           # With authentication
   ```

### **Train Your Own Models** (Optional)

```bash
# Train all regime models
python -m src.training.regime_trainer

# Or use the quickstart
python quickstart.py
```

This will:
- Fetch data
- Engineer features  
- Detect regimes
- Train XGBoost, LSTM, and Random Forest models
- Save models to `data/models/`
- Track experiments in MLflow

### **Ingest Real Data** (When You Have a Real API)

1. Configure data source in `config.yaml`:
```yaml
data:
  api_base_url: "https://your-energy-api.com"
  api_key: "your-key-here"
```

2. Run ingestion:
```bash
python -m src.ingestion.run_ingestion
```

---

## 📂 Project Structure

```
Regime_AI/
│
├── api_dashboard.html          # 🌐 Quick access dashboard (OPEN THIS!)
├── test_prediction.py          # 🧪 Test script for predictions
├── start_api.ps1               # ▶️ Start API server script
│
├── data/                       # 📁 All your data
│   ├── raw/                   # Raw ingested data
│   ├── features/              # Engineered features
│   └── models/                # Trained ML models
│
├── src/                        # 💻 Source code
│   ├── ingestion/             # Data fetching
│   ├── features/              # Feature engineering
│   ├── regime/                # Regime detection
│   ├── training/              # Model training
│   ├── inference/             # API server
│   └── monitoring/            # MLflow, drift detection
│
├── mlruns/                     # 📊 MLflow experiments
├── monitoring/                 # 📈 Grafana dashboards
└── tests/                      # ✅ Unit tests
```

---

## 🎯 Common Tasks

### **Task 1: Make a Prediction**
```bash
# Open browser
http://127.0.0.1:8002/docs

# Or run test script
python test_prediction.py
```

### **Task 2: View Experiment History**
```bash
# Start MLflow UI
mlflow ui

# Open browser
http://localhost:5000
```

### **Task 3: Check System Health**
```bash
# Browser
http://127.0.0.1:8002/health
```

### **Task 4: View Prometheus Metrics**
```bash
# Browser
http://127.0.0.1:8002/metrics
```

### **Task 5: Stop the API Server**
```bash
# Find the process
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Stop it
Stop-Process -Name python -Force

# Or just close the PowerShell window running the server
```

### **Task 6: Restart the API Server**
```bash
.\start_api.ps1
```

---

## 🔍 Understanding the System

### **What is a Regime?**
A regime is a distinct market condition:
- **Regime 0**: High wind → Low prices, stable
- **Regime 1**: Neutral → Medium prices, moderate volatility
- **Regime 2**: Low wind → High prices, volatile

### **How Does Prediction Work?**

1. **You send data** (wind speed, temperature, price, production)
   ↓
2. **Feature Engineering** (112 features calculated)
   ↓
3. **Regime Detection** (HMM determines which regime)
   ↓
4. **Model Selection** (Routes to specialized model)
   ↓
5. **Prediction** (Returns forecast + confidence)

### **Why Multiple Models?**

Instead of one model trying to predict ALL market conditions (and doing poorly), you have:
- **XGBoost** → Excels at stable, high-wind patterns
- **LSTM** → Captures temporal sequences in neutral conditions
- **Random Forest** → Handles non-linear, volatile markets

**Result**: Much better accuracy than a single model!

---

## 📞 Need Help?

### **Check System Status**
```bash
python -c "import requests; print(requests.get('http://127.0.0.1:8002/status').json())"
```

### **View Logs**
The API server terminal shows all requests and any errors.

### **Run Tests**
```bash
pytest tests/
```

---

## 🎉 Quick Reference

| What You Want | Where to Go |
|--------------|-------------|
| **Make predictions** | http://127.0.0.1:8002/docs |
| **View API docs** | http://127.0.0.1:8002/docs |
| **Test predictions** | `python test_prediction.py` |
| **Check health** | http://127.0.0.1:8002/health |
| **View metrics** | http://127.0.0.1:8002/metrics |
| **See experiments** | `mlflow ui` → http://localhost:5000 |
| **View data** | `data/raw/`, `data/features/` |
| **View models** | `data/models/` |
| **Start server** | `.\start_api.ps1` |
| **Quick dashboard** | `api_dashboard.html` (open in browser) |

---

## 💡 Pro Tips

1. **Always use Swagger UI** (http://127.0.0.1:8002/docs) for testing - it's the easiest way!

2. **Check health before predictions** to ensure models are loaded

3. **Use the test script** (`python test_prediction.py`) to verify everything works

4. **Monitor MLflow** for experiment tracking and model performance

5. **API Key is required** for `/predict` endpoint - use `dev-key-12345`

---

**You're all set! 🚀 Start making predictions now!**

Open http://127.0.0.1:8002/docs and click "Try it out" on the `/predict` endpoint!
