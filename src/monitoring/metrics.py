"""
Prometheus metrics for API monitoring

Tracks:
- Prediction request counts
- Inference latency
- Regime distribution
- Model usage
- Error rates
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from typing import Dict
import time


# ============================================================================
# Metrics Definitions
# ============================================================================

# Request counters
prediction_requests_total = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['regime', 'model', 'status']
)

batch_prediction_requests_total = Counter(
    'batch_prediction_requests_total',
    'Total number of batch prediction requests'
)

# Latency histograms
prediction_latency_seconds = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    ['regime', 'model'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

feature_engineering_latency_seconds = Histogram(
    'feature_engineering_latency_seconds',
    'Feature engineering latency in seconds',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

regime_detection_latency_seconds = Histogram(
    'regime_detection_latency_seconds',
    'Regime detection latency in seconds',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# Distribution metrics
regime_distribution = Counter(
    'regime_distribution_total',
    'Distribution of predictions by regime',
    ['regime_id', 'regime_name']
)

regime_confidence = Histogram(
    'regime_confidence',
    'Regime detection confidence scores',
    ['regime'],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)

# Model metrics
model_usage_total = Counter(
    'model_usage_total',
    'Total model usage by regime and model type',
    ['regime_id', 'model_name', 'model_type']
)

# Prediction value distribution
prediction_value = Histogram(
    'prediction_value_mwh',
    'Distribution of predicted energy values in MWh',
    buckets=(0, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000)
)

# Error tracking
prediction_errors_total = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# System metrics
models_loaded = Gauge(
    'models_loaded',
    'Number of models currently loaded'
)

api_info = Info(
    'api_info',
    'API version and configuration information'
)


# ============================================================================
# Helper Functions
# ============================================================================

class MetricsTracker:
    """Helper class for tracking metrics during request processing"""
    
    def __init__(self):
        self.start_time = None
        self.feature_time = None
        self.regime_time = None
        self.model_time = None
        
    def start_request(self) -> None:
        """Start timing a request"""
        self.start_time = time.time()
    
    def start_feature_engineering(self) -> None:
        """Start timing feature engineering"""
        self.feature_time = time.time()
    
    def end_feature_engineering(self) -> None:
        """End timing feature engineering and record metric"""
        if self.feature_time:
            latency = time.time() - self.feature_time
            feature_engineering_latency_seconds.observe(latency)
    
    def start_regime_detection(self) -> None:
        """Start timing regime detection"""
        self.regime_time = time.time()
    
    def end_regime_detection(self) -> None:
        """End timing regime detection and record metric"""
        if self.regime_time:
            latency = time.time() - self.regime_time
            regime_detection_latency_seconds.observe(latency)
    
    def record_prediction(
        self,
        regime_id: int,
        regime_name: str,
        model_name: str,
        model_type: str,
        confidence: float,
        pred_value: float,
        status: str = "success"
    ) -> None:
        """
        Record a completed prediction with all metrics
        
        Args:
            regime_id: Regime identifier (0, 1, 2)
            regime_name: Regime name (volatile, neutral, stable)
            model_name: Model name (xgboost, lstm, random_forest)
            model_type: Model type (sklearn, keras)
            confidence: Regime detection confidence (0-1)
            pred_value: Predicted energy value in MWh
            status: Request status (success, error)
        """
        # Request counter
        prediction_requests_total.labels(
            regime=regime_name,
            model=model_name,
            status=status
        ).inc()
        
        # Latency (total request time)
        if self.start_time and status == "success":
            latency = time.time() - self.start_time
            prediction_latency_seconds.labels(
                regime=regime_name,
                model=model_name
            ).observe(latency)
        
        # Regime distribution
        regime_distribution.labels(
            regime_id=str(regime_id),
            regime_name=regime_name
        ).inc()
        
        # Regime confidence
        regime_confidence.labels(regime=regime_name).observe(confidence)
        
        # Model usage
        model_usage_total.labels(
            regime_id=str(regime_id),
            model_name=model_name,
            model_type=model_type
        ).inc()
        
        # Prediction value distribution
        if status == "success":
            prediction_value.observe(pred_value)
    
    def record_error(self, error_type: str) -> None:
        """Record an error"""
        prediction_errors_total.labels(error_type=error_type).inc()


def set_api_info(version: str, n_regimes: int, models_info: Dict[int, str]) -> None:
    """
    Set API information metric
    
    Args:
        version: API version
        n_regimes: Number of regimes
        models_info: Dictionary mapping regime_id to model_name
    """
    api_info.info({
        'version': version,
        'n_regimes': str(n_regimes),
        'models': ','.join([f"regime_{k}:{v}" for k, v in models_info.items()])
    })


def set_models_loaded(count: int) -> None:
    """Update the count of loaded models"""
    models_loaded.set(count)


def get_prometheus_metrics() -> tuple:
    """
    Generate Prometheus metrics in text format
    
    Returns:
        Tuple of (metrics_text, content_type)
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ============================================================================
# Batch Metrics
# ============================================================================

def record_batch_request(n_succeeded: int, n_failed: int) -> None:
    """
    Record metrics for a batch prediction request
    
    Args:
        n_succeeded: Number of successful predictions
        n_failed: Number of failed predictions
    """
    batch_prediction_requests_total.inc()
    
    if n_failed > 0:
        prediction_errors_total.labels(error_type='batch_item_failed').inc(n_failed)
