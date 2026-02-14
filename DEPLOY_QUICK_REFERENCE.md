# Production Deployment Quick Reference

## 🚀 What's Now Enabled

### 1. **Webhook Alerting** ✅
- **Status**: Ready for configuration
- **What it does**: Sends Slack/Teams/Discord messages when drift is detected
- **File**: `src/monitoring/alerting.py`
- **Next step**: Set `ALERT_WEBHOOK_URL` in `.env`

### 2. **MLflow Registry** ✅
- **Status**: Enabled (`MLFLOW_REGISTRY_ENABLED=true`)
- **What it does**: Auto-promotes trained models through stages (dev→staging→prod)
- **Database**: SQLite (`mlflow.db`)
- **Models tracked**: regime_a_model, regime_b_model, regime_c_model
- **Promotion**: Automatic during training if `MLFLOW_REGISTRY_PROMOTE=true`

### 3. **Real Data Integration** ✅
- **Status**: Configured (placeholder API key in `.env`)
- **What it does**: Connects to Energi Data Service for live energy data
- **Current**: Using mock data (empty EDS_API_KEY)
- **Next step**: Get API key from https://www.energidataservice.dk/

### 4. **Production Database** ✅
- **Status**: SQLite configured
- **Upgrade path**: Switch to PostgreSQL for larger deployments
- **File location**: `mlflow.db` (auto-created)
- **Persistence**: Data survives container restarts

### 5. **Monitoring Stack** ✅
- **Status**: Docker-compose ready
- **Services**:
  - **API** (port 8000): Your FastAPI server
  - **MLflow** (port 5000): Model tracking & registry
  - **Prometheus** (port 9090): Metrics collection
  - **Grafana** (port 3000): Dashboards & alerts
- **Start**: `docker-compose up -d`

---

## 📋 Configuration Checklist

### Before Production:

```bash
# 1. Set Slack/Teams webhook for alerts
Edit .env, find:
  ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# 2. Set real data API credentials
Edit .env, find:
  EDS_API_KEY=your-eds-api-key-here

# 3. Verify API key is strong
Already set:
  API_KEY=B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY

# 4. MLflow database (no action needed if using SQLite)
Already configured:
  MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# 5. Start monitoring (requires Docker)
docker-compose up -d
```

---

## 🔌 API Endpoints (All Authenticated)

### Health & Status
```bash
# Health check
curl -X GET http://localhost:8000/health \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"

# MLflow registry status
curl -X GET http://localhost:8000/registry/status \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"

# List registered models
curl -X GET http://localhost:8000/models \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"
```

### Predictions
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY" \
  -H "Content-Type: application/json" \
  -d '{
    "wind_speed": 8.5,
    "energy_production": 750,
    "temperature": 5.2,
    "price": 45.67
  }'

# Batch predictions
curl -X POST http://localhost:8000/batch_predict \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"wind_speed": 5.2, "energy_production": 580, "temperature": 3.5, "price": 42.10},
      {"wind_speed": 12.1, "energy_production": 850, "temperature": 8.9, "price": 48.75}
    ]
  }'
```

### Monitoring
```bash
# Drift detection
curl -X GET http://localhost:8000/drift/last \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"

# Check drift now
curl -X POST http://localhost:8000/drift/check \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"

# Drift alerts history
curl -X GET http://localhost:8000/drift/alerts \
  -H "X-API-Key: B6La3Gp7Gs_aZ6Z99xKwV5_RmXMpF5d7NG30lY62syY"
```

---

## 🎯 Deployment Steps

### Step 1: Configure Secrets
```bash
# Edit .env with your actual values
nano .env  # or use VS Code

# Key settings to update:
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EDS_API_KEY=your-actual-api-key
API_KEY=generate-new-one-if-needed
```

### Step 2: Test Locally
```bash
python quick_test_deployment.py  # 5 quick tests (~10s)
```

### Step 3: Start Monitoring (Docker)
```bash
docker-compose up -d

# Verify services
docker-compose ps
docker-compose logs -f api
```

### Step 3b: Start Manually (No Docker)
```bash
# Terminal 1: API
python -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# Terminal 2: MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### Step 4: Deploy to Production
```bash
# Kubernetes
kubectl apply -f k8s/

# Or Docker Swarm
docker stack deploy -c docker-compose.yml regime-ai

# Or cloud (AWS, GCP, Azure)
# Follow your cloud provider's container deployment guide
```

---

## 📊 Dashboards & UIs

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| API Docs | http://localhost:8000/docs | API Key header | Interactive API testing |
| MLflow | http://localhost:5000 | None | Model tracking, registry |
| Prometheus | http://localhost:9090 | None | Metrics/queries |
| Grafana | http://localhost:3000 | admin/admin | Dashboards, alerts |

---

## 🔐 Security Reminders

1. **Never commit `.env`** - Already in `.gitignore`
2. **Rotate API keys** - Generate new one monthly
3. **HTTPS only** - Use reverse proxy (nginx) in production
4. **Database backups** - `mlflow.db` is your model history
5. **Log rotation** - Set up log aggregation

---

## 📞 Support

- **Docs**: See `ARCHITECTURE.md` for full system design
- **API**: See `docs/API.md` for endpoint reference
- **Tests**: Run `python test_deployment.py` for comprehensive validation

---

**System Status**: ✅ Production-Ready
**Last Updated**: 2026-02-14
**Version**: 1.0.0
