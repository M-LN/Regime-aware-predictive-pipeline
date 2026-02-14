"""
FastAPI inference server for regime-aware predictions
"""

import logging
import os
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

from src.config import get_config
from src.ingestion.data_fetcher import MockEnergyDataFetcher
from src.features.engineer import FeatureEngineer
from src.regime.detector import RegimeDetectionPipeline
from src.monitoring import (
    MetricsTracker,
    get_prometheus_metrics,
    set_api_info,
    set_models_loaded,
    record_batch_request,
    initialize_mlflow,
    get_mlflow_tracker,
    DriftDetector,
    RequestLogger,
    PredictionLogger,
    StructuredLogger,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id
)

try:
    from tensorflow import keras
    HAS_TF = True
except Exception:
    keras = None
    HAS_TF = False

logger = logging.getLogger(__name__)
drift_logger = StructuredLogger("drift")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class EnergyDataPoint(BaseModel):
    """Input schema for energy data"""
    wind_speed: float = Field(..., ge=0, le=30, description="Wind speed in m/s")
    energy_production: float = Field(..., ge=0, le=1000, description="Energy production in MWh")
    temperature: float = Field(..., ge=-40, le=50, description="Temperature in Celsius")
    price: float = Field(..., ge=0, le=500, description="Electricity price in DKK/MWh")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")


class PredictionResponse(BaseModel):
    """Output schema for predictions"""
    prediction: float
    unit: str = "MWh"
    regime: str
    regime_id: int
    regime_confidence: float
    model_name: str
    model_version: str = "1.0.0"
    inference_latency_ms: float
    timestamp: str


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    regime_detector: str
    models_loaded: int
    mlflow_connected: bool
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    data: List[EnergyDataPoint]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    n_succeeded: int
    n_failed: int


# ============================================================================
# Global State & Initialization
# ============================================================================

class AppState:
    """Application state container"""
    
    def __init__(self):
        self.config = get_config()
        self.regime_pipeline = None
        self.feature_engineer = None
        self.regime_models: Dict[int, Dict[str, object]] = {}
        self.mlflow_tracker = None
        self.drift_detector = None
        self.drift_auto_check_enabled = os.getenv("DRIFT_AUTO_CHECK_ENABLED", "true").lower() == "true"
        self.drift_check_interval_seconds = int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "300"))
        self.last_drift_check = None
        self.last_drift_result = None
        self.ready = False
        self.error_message = None


app_state = AppState()


def _load_regime_models(config) -> Dict[int, Dict[str, object]]:
    models: Dict[int, Dict[str, object]] = {}
    model_dir = Path(config.data.model_registry_path)
    model_map = {
        0: config.model.regime_a_model,
        1: config.model.regime_b_model,
        2: config.model.regime_c_model,
    }

    for regime_id, model_name in model_map.items():
        base_name = f"regime_{regime_id}_{model_name}"
        keras_path = model_dir / f"{base_name}.keras"
        pkl_path = model_dir / f"{base_name}.pkl"

        if keras_path.exists() and HAS_TF:
            try:
                model = keras.models.load_model(keras_path)
                models[regime_id] = {"model": model, "type": "keras", "name": model_name}
                logger.info("Loaded regime %s model from %s", regime_id, keras_path)
                continue
            except Exception as e:
                logger.warning("Failed to load Keras model %s: %s", keras_path, e)

        if pkl_path.exists():
            try:
                model = joblib.load(pkl_path)
                models[regime_id] = {"model": model, "type": "sklearn", "name": model_name}
                logger.info("Loaded regime %s model from %s", regime_id, pkl_path)
                continue
            except Exception as e:
                logger.warning("Failed to load model %s: %s", pkl_path, e)

        logger.warning("No model file found for regime %s (%s)", regime_id, model_name)

    return models


def _load_registry_models(tracker, stage: str) -> Dict[int, Dict[str, object]]:
    models: Dict[int, Dict[str, object]] = {}
    model_env_map = {
        0: os.getenv("MLFLOW_REGISTRY_MODEL_A"),
        1: os.getenv("MLFLOW_REGISTRY_MODEL_B"),
        2: os.getenv("MLFLOW_REGISTRY_MODEL_C"),
    }

    for regime_id, registry_name in model_env_map.items():
        if not registry_name:
            continue

        model = tracker.load_model_from_registry(registry_name, stage=stage)
        if model is None:
            continue

        models[regime_id] = {
            "model": model,
            "type": "mlflow_pyfunc",
            "name": registry_name,
        }
        logger.info("Loaded regime %s model from MLflow registry: %s/%s", regime_id, registry_name, stage)

    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Regime-Aware Prediction Pipeline API...")
    
    try:
        # Initialize feature engineer
        app_state.feature_engineer = FeatureEngineer(
            rolling_windows=app_state.config.features.rolling_windows,
            volatility_windows=app_state.config.features.volatility_windows
        )
        
        # Initialize regime detection pipeline
        app_state.regime_pipeline = RegimeDetectionPipeline(
            n_regimes=app_state.config.regime.n_regimes
        )
        
        # Try to load pre-trained regime detector
        regime_model_path = f"{app_state.config.data.model_registry_path}/regime_detector.pkl"
        
        try:
            app_state.regime_pipeline.load_model(regime_model_path)
            logger.info("Regime detector loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load regime detector: {e}")
            logger.info("Will train regime detector on first data batch")

        # Load per-regime models if available
        app_state.regime_models = _load_regime_models(app_state.config)
        
        # Initialize MLflow tracking
        try:
            app_state.mlflow_tracker = initialize_mlflow(
                tracking_uri=None,  # Uses ./mlruns by default
                experiment_name="regime-aware-energy-prediction"
            )
            if app_state.mlflow_tracker.is_connected():
                logger.info("MLflow tracking initialized")
            else:
                logger.warning("MLflow tracking disabled")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
        
        # Initialize drift detector
        try:
            app_state.drift_detector = DriftDetector(
                reference_window=1000,
                detection_window=100,
                kl_threshold=0.1,
                wasserstein_threshold=0.3
            )
            logger.info("Drift detector initialized")
        except Exception as e:
            logger.warning(f"Drift detector initialization failed: {e}")
        
        # Set Prometheus metrics
        model_info = {rid: entry["name"] for rid, entry in app_state.regime_models.items()}
        set_api_info("1.0.0", app_state.config.regime.n_regimes, model_info)
        set_models_loaded(len(app_state.regime_models))
        
        app_state.ready = True
        logger.info("API initialization complete with monitoring enabled")
    
    except Exception as e:
        app_state.error_message = str(e)
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Regime-Aware Energy Prediction API",
    description="Context-aware forecasting using multi-regime modeling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (optional, for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Add correlation ID to all requests"""
    # Get or generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or get_correlation_id()
    set_correlation_id(correlation_id)
    
    # Process request
    response = await call_next(request)
    
    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    
    # Clear correlation ID after request
    clear_correlation_id()
    
    return response


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """Root endpoint"""
    return {
        "name": "Regime-Aware Energy Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    
    if not app_state.ready:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {app_state.error_message}"
        )
    
    mlflow_connected = app_state.mlflow_tracker and app_state.mlflow_tracker.is_connected() if app_state.mlflow_tracker else False
    
    return HealthCheckResponse(
        status="healthy",
        regime_detector="loaded" if app_state.regime_pipeline else "not_loaded",
        models_loaded=len(app_state.regime_models),
        mlflow_connected=mlflow_connected,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(data: EnergyDataPoint) -> PredictionResponse:
    """
    Generate a single prediction with regime detection.
    
    Args:
        data: Energy data point
    
    Returns:
        Prediction with regime information
    """
    start_time = datetime.utcnow()
    
    # Initialize metrics tracker and loggers
    metrics_tracker = MetricsTracker()
    metrics_tracker.start_request()
    pred_logger = PredictionLogger()
    
    try:
        # Normalize timestamp; fall back to current time on missing/invalid values
        if data.timestamp:
            parsed_timestamp = pd.to_datetime(data.timestamp, errors="coerce")
            if pd.isna(parsed_timestamp):
                parsed_timestamp = pd.Timestamp.utcnow()
        else:
            parsed_timestamp = pd.Timestamp.utcnow()

        # Convert to DataFrame for processing
        df = pd.DataFrame([{
            'timestamp': parsed_timestamp,
            'wind_speed': data.wind_speed,
            'energy_production': data.energy_production,
            'temperature': data.temperature,
            'price': data.price
        }])
        
        # Engineer features
        # For single sample, keep NaNs to avoid empty frames
        metrics_tracker.start_feature_engineering()
        df_with_features = app_state.feature_engineer.engineer_features(df, dropna=False)
        metrics_tracker.end_feature_engineering()

        excluded_cols = ['timestamp', 'ingestion_timestamp', 'data_source', 'energy_production', 'regime']
        feature_cols = [col for col in df_with_features.columns if col not in excluded_cols]
        X = df_with_features[feature_cols].fillna(0).values

        # Predict regime using trained detector if available
        metrics_tracker.start_regime_detection()
        regime_id = None
        confidence = 0.0
        if app_state.regime_pipeline and app_state.regime_pipeline.detector.model is not None:
            try:
                regime_proba = app_state.regime_pipeline.detector.predict_proba(X)
                regime_id = int(np.argmax(regime_proba[0]))
                confidence = float(np.max(regime_proba[0]))
            except Exception as e:
                logger.warning(f"Regime prediction failed; falling back to heuristic: {e}")
        metrics_tracker.end_regime_detection()

        if regime_id is None:
            if data.wind_speed > 10:
                regime_id = 0
                confidence = 0.87
            elif data.wind_speed > 5:
                regime_id = 1
                confidence = 0.82
            else:
                regime_id = 2
                confidence = 0.78

        regime_name = {0: "volatile", 1: "neutral", 2: "stable"}.get(regime_id, "unknown")

        # Generate prediction using trained model if available
        model_entry = app_state.regime_models.get(regime_id)
        if model_entry:
            model = model_entry["model"]
            model_type = model_entry["type"]
            model_name = model_entry.get("name", "unknown")
            if model_type == "keras":
                X_seq = X.reshape((X.shape[0], 1, X.shape[1]))
                base_pred = float(model.predict(X_seq, verbose=0).reshape(-1)[0])
            elif model_type == "mlflow_pyfunc":
                X_df = pd.DataFrame(X, columns=feature_cols)
                base_pred = float(model.predict(X_df)[0])
            else:
                base_pred = float(model.predict(X)[0])
        else:
            model_name = "mock"
            base_pred = data.energy_production * 1.05 + np.random.normal(0, 10)
        
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Record metrics
        model_type = model_entry.get("type", "mock") if model_entry else "mock"
        metrics_tracker.record_prediction(
            regime_id=regime_id,
            regime_name=regime_name,
            model_name=model_name,
            model_type=model_type,
            confidence=confidence,
            pred_value=base_pred,
            status="success"
        )
        
        # Log prediction with structured logger
        pred_logger.log_prediction(
            regime_id=regime_id,
            regime_name=regime_name,
            model_name=model_name,
            prediction=base_pred,
            confidence=confidence,
            latency_ms=latency_ms
        )
        
        # Update drift detector
        if app_state.drift_detector:
            try:
                # Extract a few key features for drift monitoring
                key_features = {
                    "wind_speed": data.wind_speed,
                    "temperature": data.temperature,
                    "price": data.price
                }
                app_state.drift_detector.update_features(key_features)
                app_state.drift_detector.update_prediction(base_pred)
                app_state.drift_detector.update_regime(regime_id)
            except Exception as e:
                logger.debug(f"Drift detector update failed: {e}")

        # Auto-check drift at configured intervals
        if app_state.drift_detector and app_state.drift_auto_check_enabled:
            try:
                now = datetime.utcnow()
                if app_state.last_drift_check is None or (
                    now - app_state.last_drift_check
                ).total_seconds() >= app_state.drift_check_interval_seconds:
                    status = app_state.drift_detector.get_status()
                    if status.get("reference_distributions_set"):
                        results = app_state.drift_detector.check_drift(auto_update_reference=False)
                        app_state.last_drift_result = results
                        app_state.last_drift_check = now

                        if results.get("drift_detected"):
                            drift_logger.warning("Drift detected", alerts=results.get("alerts", []))
                        else:
                            drift_logger.info("Drift check complete", drift_detected=False)
            except Exception as e:
                drift_logger.error("Drift auto-check failed", error=e)
        
        # MLflow tracking (sampled)
        if app_state.mlflow_tracker and app_state.mlflow_tracker.is_connected():
            try:
                app_state.mlflow_tracker.log_inference_sample(
                    regime_id=regime_id,
                    model_name=model_name,
                    prediction=base_pred,
                    confidence=confidence,
                    latency_ms=latency_ms
                )
            except Exception as e:
                logger.debug(f"MLflow tracking failed: {e}")
        
        logger.info(
            "Prediction complete | regime=%s model=%s confidence=%.3f",
            regime_name,
            model_name,
            confidence,
        )

        return PredictionResponse(
            prediction=float(base_pred),
            regime=regime_name,
            regime_id=regime_id,
            regime_confidence=float(confidence),
            model_name=model_name,
            inference_latency_ms=float(latency_ms),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        # Record error metrics
        metrics_tracker.record_error(type(e).__name__)
        pred_logger = PredictionLogger()
        pred_logger.log_prediction_error(e)
        
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["prediction"])
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Generate batch predictions.
    
    Args:
        request: Batch prediction request with array of data points
    
    Returns:
        Array of predictions with success/failure counts
    """
    predictions = []
    n_failed = 0
    
    for data_point in request.data:
        try:
            pred = await predict(data_point)
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Batch prediction failed for data point: {e}")
            n_failed += 1
    
    # Record batch metrics
    record_batch_request(n_succeeded=len(predictions), n_failed=n_failed)
    
    return BatchPredictionResponse(
        predictions=predictions,
        n_succeeded=len(predictions),
        n_failed=n_failed
    )


@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus text format.
    """
    metrics_data, content_type = get_prometheus_metrics()
    return Response(content=metrics_data, media_type=content_type)


@app.get("/status", tags=["info"])
async def status():
    """
    Detailed status endpoint with monitoring info.
    """
    mlflow_status = "disconnected"
    if app_state.mlflow_tracker:
        mlflow_status = "connected" if app_state.mlflow_tracker.is_connected() else "disabled"
    
    drift_status = None
    if app_state.drift_detector:
        drift_status = app_state.drift_detector.get_status()
    
    return {
        "ready": app_state.ready,
        "config": {
            "n_regimes": app_state.config.regime.n_regimes,
            "feature_windows": app_state.config.features.rolling_windows,
        },
        "models": {
            "regime_detector": "hmm",
            "regime_models": [entry["name"] for entry in app_state.regime_models.values()]
        },
        "monitoring": {
            "mlflow": mlflow_status,
            "drift_detector": drift_status,
            "prometheus_metrics": "enabled"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/drift/check", tags=["monitoring"])
async def check_drift():
    """
    Manually trigger drift detection check.
    
    Returns drift analysis results.
    """
    if not app_state.drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        results = app_state.drift_detector.check_drift(auto_update_reference=False)
        return results
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


@app.post("/drift/set_reference", tags=["monitoring"])
async def set_drift_reference():
    """
    Set current distributions as reference baseline for drift detection.
    """
    if not app_state.drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    try:
        app_state.drift_detector.set_reference_distributions()
        status = app_state.drift_detector.get_status()
        return {
            "message": "Reference distributions updated",
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to set reference: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set reference: {str(e)}")


@app.get("/drift/alerts", tags=["monitoring"])
async def get_drift_alerts(hours: int = 24):
    """
    Get recent drift alerts.
    
    Args:
        hours: Number of hours to look back (default: 24)
    """
    if not app_state.drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    
    alerts = app_state.drift_detector.get_recent_alerts(hours=hours)
    return {
        "alerts": alerts,
        "count": len(alerts),
        "hours": hours,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/drift/last", tags=["monitoring"])
async def get_last_drift_check():
    """
    Get the most recent drift check results.
    """
    if not app_state.drift_detector:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")

    return {
        "last_check": app_state.last_drift_check.isoformat() if app_state.last_drift_check else None,
        "last_result": app_state.last_drift_result,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on app startup"""
    logger.info("API startup complete")


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "src.inference.api:app",
        host=config.inference.host,
        port=config.inference.port,
        reload=config.inference.reload,
        log_level=config.inference.log_level
    )
