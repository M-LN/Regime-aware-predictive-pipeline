# Regime-Aware Predictive Pipeline: Architecture Documentation

## 🏗️ System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    REGIME-AWARE ENERGY PREDICTION                     │
│                            PLATFORM                                   │
└──────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────────┐
                         │      Data Sources (API)       │
                         │  • Energy production (MWh)    │
                         │  • Wind speed (m/s)           │
                         │  • Temperature (°C)           │
                         │  • Electricity prices (DKK)   │
                         │  • Refresh: Hourly/Real-time  │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │   🔄 Data Ingestion Layer     │
                         │  • Scheduler (cron/GH Actions)│
                         │  • Raw data storage (Parquet) │
                         │  • Schema validation          │
                         │  • Timestamp logging          │
                         │  • Data quality checks        │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 📊 Feature Engineering Layer  │
                         │  • Rolling windows (1h-24h)   │
                         │  • Volatility metrics         │
                         │  • Trend/seasonality features │
                         │  • Normalization pipeline     │
                         │  • Feature versioning         │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │🎯 Regime Detection Layer (ML)│
                         │  • Hidden Markov Models       │
                         │  • Bayesian Change Pt. (plan) │
                         │  • Regime labeling & tagging  │
                         │  • Regime probability scores  │
                         │  • Metadata registry          │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  🤖 Model Training Layer      │
                         │  • Train model PER regime     │
                         │  • XGBoost / LSTM / RandomF.  │
                         │  • Hyperparameter tuning      │
                         │  • Cross-validation (regime)  │
                         │  • MLflow tracking (registry) │
                         │  • Model versioning & tags    │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ 🔀 Routing Engine / Inference │
                         │  • Real-time regime detection │
                         │  • Model selection logic      │
                         │  • Prediction aggregation     │
                         │  • Confidence scoring         │
                         │  • Inference logging          │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │   ⚙️ Deployment Layer         │
                         │  • FastAPI microservice       │
                         │  • Docker containerization    │
                         │  • CI/CD pipeline (GH Actions)│
                         │  • Kubernetes-ready manifest  │
                         │  • Secrets management         │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │📈 Monitoring & Governance     │
                         │  • Model performance tracking │
                         │  • Data drift detection       │
                         │  • Regime drift detection     │
                         │  • Prometheus metrics         │
                         │  • MLflow metrics store       │
                         │  • Alert thresholds           │
                         └──────────────────────────────┘
```

---

## 📋 Layer-by-Layer Architecture Details

### 1️⃣ **Data Sources (Energy Domain)**

**What:**
- Real-time and historical energy data from Danish/Nordic grid operators
- Meteorological data (wind, temperature)
- Market data (electricity prices)
- Data freshness: Hourly updates, pushed via API

**Technologies:**
- HTTP REST APIs or message queues (Kafka)
- Data providers: Energi Data Service, ENTSOE-E, or custom IoT sensors

**Why it matters (IBM perspective):**
- Shows deep domain knowledge in energy systems
- Demonstrates ability to work with complex, real-world messy data
- Enterprise-scale data volume handling

---

### 2️⃣ **Data Ingestion Layer**

**Responsibilities:**
- **Scheduler**: Trigger ingestion on fixed intervals (hourly) or event-driven
- **Raw Data Storage**: Store all incoming data in Parquet format (columnar, compressible, versioned)
- **Schema Validation**: Enforce consistent column names, types, null handling
- **Timestamp Logging**: Capture ingestion timestamp, data latency, API response time
- **Quality Checks**: Detect missing values, outliers, schema drift

**Architecture:**
```
API → Scheduler → Validation → Parquet Lake → Metadata Registry
                      ↓
                  [Failed Records] → Error Log
```

**Implementation (current MVP):**
- Python ingestion runner (`src/ingestion/run_ingestion.py`)
- GitHub Actions scheduled workflow (`.github/workflows/ingest.yml`)
- Parquet partitioning by date under `data/raw/`
- Failed payload capture under `data/raw/failed/`

**Why IBM cares:**
- Shows understanding of data governance and reproducibility
- Demonstrates data lineage tracking capability
- Enables audit trails (compliance, regulatory)

---

### 3️⃣ **Feature Engineering Layer**

**Core Features Generated:**

| Feature Category | Examples | Window Size |
|---|---|---|
| **Temporal** | Hour of day, day of week, season | Static |
| **Statistical** | Rolling mean, std, skewness, kurtosis | 1h, 6h, 24h |
| **Volatility** | Rolling coefficient of variation (CV) | 6h, 24h |
| **Trend** | Linear regression slope, rate of change | 1h, 12h |
| **Seasonality** | Fourier features, STL decomposition | Daily, weekly |
| **Lagged** | Previous 1h, 3h, 12h, 24h values | Variable |
| **Domain** | Energy balance, wind power density | Custom |

**Pipeline Flow:**
```
Raw Data → Normalization → Windowing → Feature Calc → Feature Store
                 ↓                              ↓
           [Scaling fitted]           [Version: 2.1.0]
```

**Why it's important:**
- **Non-stationary energy data** requires careful feature engineering
- Volatility features are critical for regime detection
- Rolling windows capture temporal dependencies without leakage

---

### 4️⃣ **Regime Detection Layer** ⭐ (The Differentiator)

**The Core Insight:**
Energy markets don't follow one pattern—they shift between **distinct operating regimes**:
- **Regime A**: High wind, low prices, stable
- **Regime B**: Calm, high demand, volatile prices
- **Regime C**: Transition periods, extreme variability

> **This is where IBM Watsonx Context Engine logic applies.** You're building context-aware AI.

**Algorithms:**

#### Option A: Hidden Markov Models (HMM)
```python
# Pseudocode
from hmmlearn import GaussianHMM

hmm = GaussianHMM(n_components=3)  # 3 regimes
hmm.fit(features)
regime_labels = hmm.predict(features)
regime_probs = hmm.predict_proba(features)  # Confidence scores
```

**Advantages:**
- Interpretable state transitions
- Handles uncertainty natively (probability scores)
- Fast inference

#### Option B: Bayesian Change Point Detection (Planned)
```python
# Detects when regime boundaries shift
from bayesian_blocks import bayesian_blocks

changepoints = bayesian_blocks(data)  # Identifies regime shifts
```

**Advantages:**
- No need to pre-specify number of regimes
- Statistically principled
- Handles non-uniform sampling

**Current MVP:** HMM-based regime detection is implemented; Bayesian CPD is a planned extension.

**Output Structure:**
```json
{
  "timestamp": "2026-02-13T14:00:00Z",
  "regime_id": "high_wind",
  "regime_probability": 0.87,
  "transition_risk": 0.12,
  "regime_duration_hours": 6,
  "metadata": {
    "wind_speed_avg": 8.5,
    "volatility_score": 0.34
  }
}
```

---

### 5️⃣ **Model Training Layer**

**Multi-Model Strategy:**

```
Regime A (High Wind) → XGBoost Regressor
Regime B (Calm)      → LSTM Sequence Model
Regime C (Volatile)  → Random Forest + Gradient Boosting
```

**Training Pipeline:**
```
Feature Store (regime-filtered) 
    ↓
    ├→ Train/test split (temporal, no leakage)
  ├→ Hyperparameter tuning (planned for MVP)
    ├→ Cross-validation (k-fold per regime)
    ├→ Evaluation metrics:
    │   • MAPE (Mean Absolute Percent Error)
    │   • RMSE
    │   • MAE
    │   • R² per regime
    ↓
Local Model Artifacts + MLflow Tracking
  ├→ model_path: data/models/regime_<id>_<model>.pkl|.keras
  ├→ tags/params/metrics logged to MLflow (tracking)
  ├→ registry promotion (planned)
```

**Key Details:**
- **No data leakage**: Use temporal split (train before test date)
- **Regime-specific metrics**: Track accuracy per regime, not global
- **Retraining schedule**: Monthly or on drift detection signal
- **Current MVP**: Fixed hyperparameters; MLflow registry promotion planned

---

### 6️⃣ **Routing Engine / Inference API**

**Real-Time Prediction Flow:**

```
New Energy Data Point
    ↓
┌─────────────────────────┐
│ Regime Classifier       │  ← Lightweight HMM/Classifier
│ (Detect current regime) │
└─────────────┬───────────┘
              ↓
        ┌─────────────┐
        │ Regime = B? │
        └─────────────┘
              ↓
    ┌─────────────────────┐
    │ Load Model B        │  ← From MLflow Registry
    │ (LSTM)              │
    └──────────┬──────────┘
               ↓
     ┌──────────────────┐
     │ Generate Pred    │
     │ Confidence: 0.89 │
     └────────┬─────────┘
              ↓
    ┌──────────────────────┐
    │ Return JSON:         │
    │ {                    │
    │   prediction: 2450   │ MWh
    │   confidence: 0.89   │
    │   regime: "B"        │
    │   model_version: 1.0 │
    │   timestamp: ...     │
    │ }                    │
    └──────────────────────┘
```

**API Endpoint (FastAPI, MVP):**
```python
@app.post("/predict")
async def predict(input_data: EnergyDataPoint):
    # Detect regime
    regime = regime_detector.predict(input_data.features)
    
  # Load regime-specific model (local artifacts)
  model = local_model_registry.load(regime)
    
    # Generate prediction
    pred = model.predict([input_data.features])
    
    # Log for monitoring
    logger.info({
        "regime": regime,
        "prediction": pred[0],
        "timestamp": datetime.now()
    })
    
    return {
        "prediction": float(pred[0]),
        "regime": regime,
        "confidence": 0.92,
        "model_version": "1.0.0"
    }
```

  **Current MVP:** Models are loaded from `data/models/` at startup; MLflow registry loading is planned.

**Advantages:**
- **Low latency**: Simple model selection logic
- **Interpretability**: Know which regime the system is in
- **Robustness**: Multiple models reduce single-point failures

---

### 7️⃣ **Deployment Layer**

**Docker Containerization:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**CI/CD Pipeline (GitHub Actions):**

```yaml
# .github/workflows/deploy.yml
name: Build & Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src
      - run: flake8 src/ tests/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

**Kubernetes Deployment (Optional):**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: regime-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: regime-predictor
  template:
    metadata:
      labels:
        app: regime-predictor
    spec:
      containers:
      - name: api
        image: ghcr.io/your-org/regime-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        - name: LOG_LEVEL
          value: "INFO"
```

---

### 8️⃣ **Monitoring & Governance Layer**

**Key Metrics Tracked:**

#### Model Performance
```
• Prediction Error (MAPE, RMSE) per regime
• Inference Latency (p50, p95, p99)
• Model Confidence Distribution
• Prediction Bias (under/over-prediction)
```

#### Data Drift Detection
```
• Feature distribution shift (KL divergence, Wasserstein)
• Input data statistics vs. training baseline
• Null / missing value rates
```

#### Regime Drift Detection
```
• Regime transition frequency (should be stable)
• Time spent in each regime (should match historical patterns)
• Regime probability confidence (sudden drops = suspicious)
```

**MVP Metrics Export:** Prometheus-compatible `/metrics` endpoint from the FastAPI service.

**Monitoring Dashboard (Planned):**

```
┌─────────────────────────────────────────────────┐
│ Regime-Aware Energy Prediction - Live Dashboard │
├─────────────────────────────────────────────────┤
│                                                  │
│  Current Regime: B (Calm)  | Confidence: 0.87  │
│  ───────────────────────────────────────────   │
│                                                  │
│  Model Accuracy (Today)                          │
│  ├─ Regime A: MAPE 2.3% ✓                       │
│  ├─ Regime B: MAPE 4.1% ⚠️                      │
│  └─ Regime C: MAPE 6.2% ⚠️                      │
│                                                  │
│  Data Quality                                    │
│  ├─ Missing Values: 0.1%                        │
│  ├─ Drift Score: 0.12 (normal: < 0.3)          │
│  └─ Ingestion Latency: 2.3s avg                 │
│                                                  │
│  Last Prediction: 2450 MWh @ 14:00              │
│  Next Retraining: 2026-03-13                    │
│                                                  │
└─────────────────────────────────────────────────┘
```

**MLflow Integration (Tracking):**
```python
# Log metrics to MLflow
mlflow.log_metric("mape_regime_a", 0.023)
mlflow.log_metric("inference_latency_ms", 45)
mlflow.log_param("regime", "A")

# Tag model
mlflow.set_tag("production_ready", True)
mlflow.set_tag("last_retrained", "2026-02-13")
mlflow.set_tag("regime_count", "3")
```

**Alert Conditions (Planned):**
```
• MAPE > 8% for any regime       → Investigate data
• Inference latency > 200ms      → Check system load
• Drift score > 0.5              → Trigger retraining
• Regime transition freq unusual → Check input data quality

**Current MVP:** Drift detection and MLflow tracking are implemented; Grafana dashboards and alerting are planned.
```

---

## 🎯 Why This Architecture Excels (IBM Perspective)

| Aspect | Why IBM Values It |
|---|---|
| **Multi-Model Approach** | Shows sophisticated ML engineering, not one-off modeling |
| **Regime Awareness** | Context-aware AI is IBM Watsonx's core differentiator |
| **End-to-End Governance** | MLflow + monitoring = enterprise-ready AI ops |
| **Energy Domain** | Physical systems (IoT, grid) are IBM's sweet spot |
| **Reproducibility** | Data versioning + feature versioning = auditability |
| **Real-Time Inference** | FastAPI + Kubernetes = production-grade deployment |
| **Monitoring-First** | Drift detection & drift handling = responsible AI |

---

## 🔄 Data Flow Example: Hour-by-Hour

**Scenario:** 14:00 CET on 2026-02-13

```
14:00 ─→ Data Ingestion
         • Fetch wind speed, energy production, price
         • Validate schema
         • Store to Parquet: data/2026-02-13/14/raw.parquet

14:05 ─→ Feature Engineering
         • Calculate rolling stats (1h, 6h, 24h windows)
         • Volatility metrics
         • Store to Feature Store

14:07 ─→ Regime Detection
         • Load HMM model
         • Predict regime probability
         • Output: Regime B (high confidence 0.87)

14:08 ─→ Model Routing
         • Select LSTM model (trained for Regime B)
         • Load from MLflow

14:09 ─→ Inference
         • Generate prediction: 2450 MWh (±3%)
         • Log to Prometheus/Grafana

14:10 ─→ Response to Client
         • HTTP 200 + JSON with prediction + confidence
         • Store inference log for audit trail

Next → Continuous monitoring for drift signals
```

---

## 📦 Repository Structure

```
regime-ai/
├── README.md                      # Project overview
├── ARCHITECTURE.md                # This file
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container image
├── .github/
│   └── workflows/
│       ├── tests.yml             # Unit & integration tests
│       ├── deploy.yml            # CI/CD pipeline
│       └── monitoring.yml        # Data/model drift checks
├── src/
│   ├── ingestion/
│   │   ├── data_fetcher.py      # API client
│   │   ├── validator.py         # Schema validation
│   │   └── storage.py           # Parquet/Delta writing
│   ├── features/
│   │   ├── engineer.py          # Feature computation
│   │   ├── registry.py          # Feature versioning
│   │   └── normalization.py     # Scaling/transformation
│   ├── regime/
│   │   ├── hmm_detector.py      # HMM-based regime detection
│   │   ├── cpd_detector.py      # Bayesian change point
│   │   └── models.py            # Regime artifact storage
│   ├── training/
│   │   ├── train_regime_a.py   # Per-regime trainers
│   │   ├── train_regime_b.py
│   │   ├── train_regime_c.py
│   │   ├── evaluator.py        # Cross-regime metrics
│   │   └── hyperopt.py         # Tuning interface
│   ├── inference/
│   │   ├── router.py           # Routing engine logic
│   │   ├── api.py              # FastAPI app
│   │   └── cache.py            # Model caching
│   └── monitoring/
│       ├── drift.py            # Drift detector
│       ├── metrics.py          # MAPE, RMSE, latency
│       └── logger.py           # Structured logging
├── data/
│   ├── raw/                     # Parquet lake (partitioned by date)
│   ├── features/                # Feature store
│   └── models/                  # Model artifacts (MLflow)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis
│   ├── 02_regime_analysis.ipynb # Regime characteristics
│   └── 03_model_comparison.ipynb
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_regime_detection.py
│   ├── test_inference.py
│   └── test_e2e.py
├── k8s/
│   ├── deployment.yaml         # Kubernetes manifests
│   ├── service.yaml
│   └── configmap.yaml
└── docs/
    ├── API.md                  # OpenAPI docs
    ├── INSTALLATION.md
    └── CONTRIBUTING.md
```

---

## 🚀 Next Steps

1. **Project Setup**: Create Python virtual environment, install dependencies
2. **Data Pipeline**: Build ingestion scripts for your energy data source
3. **HMM Training**: Train regime detector on 6+ months of historical data
4. **Per-Regime Models**: XGBoost/LSTM/RF training with regime-specific validation
5. **API Development**: FastAPI routing engine with inference logging
6. **CI/CD Setup**: GitHub Actions for automated testing & deployment
7. **Monitoring**: Grafana + MLflow dashboards for drift detection
8. **Documentation**: API docs, model cards, deployment runbooks

---

**Version:** 1.0 | **Last Updated:** 2026-02-13 | **Author:** Morten Lund

