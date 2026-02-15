"""
Structured logging with correlation IDs

Provides:
- Request correlation tracking
- Structured JSON logging
- Performance tracking
- Error context capture
"""

import logging
import json
import uuid
from datetime import datetime, UTC
from typing import Optional, Dict, Any
from contextvars import ContextVar
from functools import wraps

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredLogger:
    """
    Structured logger with correlation ID support
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _build_log_dict(
        self,
        message: str,
        level: str,
        extra: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """
        Build structured log dictionary
        
        Args:
            message: Log message
            level: Log level
            extra: Extra fields
            error: Exception object
        
        Returns:
            Structured log dictionary
        """
        log_dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            "correlation_id": correlation_id_var.get()
        }
        
        if extra:
            log_dict.update(extra)
        
        if error:
            log_dict["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": None  # Could add full traceback if needed
            }
        
        return log_dict
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        log_dict = self._build_log_dict(message, "INFO", kwargs)
        self.logger.info(json.dumps(log_dict))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        log_dict = self._build_log_dict(message, "WARNING", kwargs)
        self.logger.warning(json.dumps(log_dict))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        log_dict = self._build_log_dict(message, "ERROR", kwargs, error)
        self.logger.error(json.dumps(log_dict))
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        log_dict = self._build_log_dict(message, "DEBUG", kwargs)
        self.logger.debug(json.dumps(log_dict))


def get_correlation_id() -> str:
    """
    Get or create correlation ID for current request
    
    Returns:
        Correlation ID string
    """
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for current request
    
    Args:
        correlation_id: Correlation ID string
    """
    correlation_id_var.set(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID"""
    correlation_id_var.set(None)


def with_correlation_id(func):
    """
    Decorator to ensure function has a correlation ID
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Generate new correlation ID if not present
        if correlation_id_var.get() is None:
            set_correlation_id(str(uuid.uuid4()))
        
        try:
            return await func(*args, **kwargs)
        finally:
            # Don't clear here - let middleware handle it
            pass
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Generate new correlation ID if not present
        if correlation_id_var.get() is None:
            set_correlation_id(str(uuid.uuid4()))
        
        try:
            return func(*args, **kwargs)
        finally:
            # Don't clear here - let middleware handle it
            pass
    
    # Check if function is async
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class RequestLogger:
    """
    Request-level logging helper
    """
    
    def __init__(self):
        self.logger = StructuredLogger("request")
        self.start_time = None
    
    def start_request(self, method: str, path: str, correlation_id: Optional[str] = None):
        """
        Log request start
        
        Args:
            method: HTTP method
            path: Request path
            correlation_id: Optional correlation ID
        """
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            get_correlation_id()  # Generate if needed
        
        self.start_time = datetime.now(UTC)
        
        self.logger.info(
            "Request started",
            method=method,
            path=path,
            correlation_id=get_correlation_id()
        )
    
    def end_request(
        self,
        status_code: int,
        error: Optional[Exception] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """
        Log request completion
        
        Args:
            status_code: HTTP status code
            error: Optional exception
            extra: Optional extra fields
        """
        if self.start_time:
            duration_ms = (datetime.now(UTC) - self.start_time).total_seconds() * 1000
        else:
            duration_ms = None
        
        log_data = {
            "status_code": status_code,
            "duration_ms": duration_ms,
            "correlation_id": get_correlation_id()
        }
        
        if extra:
            log_data.update(extra)
        
        if error:
            self.logger.error("Request failed", error=error, **log_data)
        elif status_code >= 400:
            self.logger.warning("Request completed with error", **log_data)
        else:
            self.logger.info("Request completed", **log_data)
        
        # Clear correlation ID after request
        clear_correlation_id()


class PredictionLogger:
    """
    Prediction-specific logging helper
    """
    
    def __init__(self):
        self.logger = StructuredLogger("prediction")
    
    def log_prediction(
        self,
        regime_id: int,
        regime_name: str,
        model_name: str,
        prediction: float,
        confidence: float,
        latency_ms: float,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log prediction details
        
        Args:
            regime_id: Regime identifier
            regime_name: Regime name
            model_name: Model name
            prediction: Predicted value
            confidence: Regime confidence
            latency_ms: Inference latency
            input_data: Optional input features (sampled)
        """
        log_data = {
            "regime_id": regime_id,
            "regime_name": regime_name,
            "model_name": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "correlation_id": get_correlation_id()
        }
        
        # Only log input data for sampled requests to avoid verbosity
        import random
        if input_data and random.random() < 0.05:  # 5% sampling
            log_data["input_sample"] = input_data
        
        self.logger.info("Prediction completed", **log_data)
    
    def log_prediction_error(
        self,
        error: Exception,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """
        Log prediction error
        
        Args:
            error: Exception object
            input_data: Optional input data for debugging
        """
        log_data = {
            "correlation_id": get_correlation_id()
        }
        
        if input_data:
            log_data["input_data"] = input_data
        
        self.logger.error("Prediction failed", error=error, **log_data)


class PerformanceLogger:
    """
    Performance tracking logger
    """
    
    def __init__(self):
        self.logger = StructuredLogger("performance")
        self.timers: Dict[str, datetime] = {}
    
    def start_timer(self, name: str):
        """Start a named timer"""
        self.timers[name] = datetime.now(UTC)
    
    def end_timer(self, name: str, extra: Optional[Dict[str, Any]] = None):
        """
        End a named timer and log duration
        
        Args:
            name: Timer name
            extra: Optional extra fields
        """
        if name not in self.timers:
            return
        
        duration_ms = (datetime.now(UTC) - self.timers[name]).total_seconds() * 1000
        
        log_data = {
            "operation": name,
            "duration_ms": duration_ms,
            "correlation_id": get_correlation_id()
        }
        
        if extra:
            log_data.update(extra)
        
        self.logger.debug(f"Timer: {name}", **log_data)
        
        del self.timers[name]


# ============================================================================
# Module-level convenience functions
# ============================================================================

def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def log_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """
    Log a custom metric
    
    Args:
        metric_name: Metric name
        value: Metric value
        tags: Optional tags
    """
    logger = StructuredLogger("metrics")
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "correlation_id": get_correlation_id()
    }
    
    if tags:
        log_data["tags"] = tags
    
    logger.info("Metric recorded", **log_data)
