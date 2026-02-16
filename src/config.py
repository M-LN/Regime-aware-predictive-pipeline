"""
Configuration management for Regime-Aware Predictive Pipeline
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class DataConfig:
    """Data source and storage configuration"""

    raw_data_path: str = "data/raw"
    feature_store_path: str = "data/features"
    model_registry_path: str = "data/models"

    # API Configuration
    api_base_url: Optional[str] = None
    api_timeout: int = 30

    # Data validation
    schema_validation_enabled: bool = True
    null_value_threshold: float = 0.1  # 10% nulls = fail


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    rolling_windows: list = None  # [1, 6, 24] hours
    volatility_windows: list = None  # [6, 24] hours
    normalization_method: str = "minmax"  # or "zscore"
    feature_versioning_enabled: bool = True

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [1, 6, 24]
        if self.volatility_windows is None:
            self.volatility_windows = [6, 24]


@dataclass
class RegimeConfig:
    """Regime detection configuration"""

    algorithm: str = "hmm"  # "hmm" or "bayesian_cpd"
    n_regimes: int = 3
    hmm_n_iter: int = 100
    hmm_random_state: int = 42

    # For Bayesian CPD
    bayesian_min_segment_length: int = 10
    bayesian_penalty: float = 10.0
    bayesian_signal_col: str = "price"


@dataclass
class ModelConfig:
    """ML model configuration"""

    regime_a_model: str = "xgboost"
    regime_b_model: str = "lstm"
    regime_c_model: str = "random_forest"

    # Training
    test_size: float = 0.2
    validation_split: float = 0.1
    random_state: int = 42

    # Hyperparameters (regime A - XGBoost)
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100

    # Hyperparameters (regime B - LSTM)
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32

    # Hyperparameters (regime C - Random Forest)
    rf_n_estimators: int = 100
    rf_max_depth: int = 15


@dataclass
class InferenceConfig:
    """Inference and API configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"

    # Inference settings
    batch_prediction_enabled: bool = True
    cache_models: bool = True
    inference_timeout: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring and drift detection configuration"""

    enable_drift_detection: bool = True
    drift_detection_method: str = "kl_divergence"  # or "wasserstein"
    drift_threshold: float = 0.3

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "regime-predictor"
    mlflow_backend_store_uri: str = "file:///data/mlflow"

    # Metrics
    track_inference_latency: bool = True
    track_model_confidence: bool = True
    track_regime_transitions: bool = True


@dataclass
class Config:
    """Master configuration class"""

    data: DataConfig = None
    features: FeatureConfig = None
    regime: RegimeConfig = None
    model: ModelConfig = None
    inference: InferenceConfig = None
    monitoring: MonitoringConfig = None

    # Environment
    environment: str = "development"
    debug: bool = False

    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.regime is None:
            self.regime = RegimeConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file or environment variables.

    Args:
        config_path: Path to YAML config file.
                    If None, tries to load from CONFIG_PATH env var.

    Returns:
        Config object with all settings
    """
    load_dotenv()

    config = Config()

    # Load from YAML if provided
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)

        # Update from YAML (if present)
        if yaml_config:
            # Skip nested config objects - they're already initialized with defaults
            # Only update simple top-level values
            simple_keys = {"app_name", "version", "environment", "debug"}
            for key, value in yaml_config.items():
                if key in simple_keys and hasattr(config, key):
                    setattr(config, key, value)

    # Override with environment variables
    config.environment = os.getenv("ENVIRONMENT", config.environment)
    config.debug = os.getenv("DEBUG", "false").lower() == "true"

    # Optional: Override specific nested values if needed via env vars
    if os.getenv("MLFLOW_TRACKING_URI"):
        config.monitoring.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or initialize global config instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
