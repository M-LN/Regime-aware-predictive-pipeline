"""Monitoring module for drift detection and observability"""

from .metrics import (
    MetricsTracker,
    get_prometheus_metrics,
    set_api_info,
    set_models_loaded,
    record_batch_request,
    prediction_requests_total,
    prediction_latency_seconds
)

from .mlflow_tracker import (
    MLflowTracker,
    initialize_mlflow,
    get_mlflow_tracker
)

from .drift_detector import DriftDetector

from .alerting import (
    send_webhook,
    should_send_alert
)

from .logging import (
    StructuredLogger,
    RequestLogger,
    PredictionLogger,
    PerformanceLogger,
    get_structured_logger,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    with_correlation_id
)

__all__ = [
    # Metrics
    "MetricsTracker",
    "get_prometheus_metrics",
    "set_api_info",
    "set_models_loaded",
    "record_batch_request",
    "prediction_requests_total",
    "prediction_latency_seconds",
    
    # MLflow
    "MLflowTracker",
    "initialize_mlflow",
    "get_mlflow_tracker",
    
    # Drift detection
    "DriftDetector",
    
    # Logging
    "StructuredLogger",
    "RequestLogger",
    "PredictionLogger",
    "PerformanceLogger",
    "get_structured_logger",
    "get_correlation_id",
    "set_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    "send_webhook",
    "should_send_alert"
]
