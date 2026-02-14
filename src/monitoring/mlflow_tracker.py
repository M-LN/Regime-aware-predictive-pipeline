"""
MLflow integration for experiment tracking and model versioning

Tracks:
- Model training experiments
- Inference predictions (sampled)
- Model performance metrics
- Hyperparameters
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    MlflowClient = None
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow tracking wrapper for experiment management
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "regime-aware-energy-prediction",
        enable_autolog: bool = False
    ):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking URI (default: ./mlruns)
            experiment_name: Experiment name
            enable_autolog: Enable automatic logging for sklearn/keras
        """
        self.enabled = HAS_MLFLOW
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        
        if not self.enabled:
            logger.warning("MLflow not available - tracking disabled")
            return
        
        try:
            # Set tracking URI
            if tracking_uri is None:
                tracking_uri = f"file:///{Path('./mlruns').absolute().as_posix()}"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
            
            # Create or get experiment
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=str(Path("./mlartifacts").absolute())
            ) if mlflow.get_experiment_by_name(experiment_name) is None else mlflow.get_experiment_by_name(experiment_name).experiment_id
            
            mlflow.set_experiment(experiment_name)
            
            # Initialize client
            self.client = MlflowClient()
            
            # Enable autologging if requested
            if enable_autolog:
                try:
                    mlflow.sklearn.autolog()
                    mlflow.tensorflow.autolog()
                    logger.info("MLflow autologging enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable autologging: {e}")
            
            logger.info(f"MLflow tracker initialized - experiment: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False
    
    def is_connected(self) -> bool:
        """Check if MLflow is connected and available"""
        return self.enabled and self.client is not None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Optional run name
            tags: Optional tags dictionary
            
        Returns:
            MLflow run context manager
        """
        if not self.enabled:
            return _DummyContext()
        
        return mlflow.start_run(run_name=run_name, tags=tags or {})
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to current run
        
        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to current run
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = "sklearn",
        signature: Optional[Any] = None
    ) -> None:
        """
        Log a model to MLflow
        
        Args:
            model: Model object
            artifact_path: Path within run's artifact directory
            model_type: Type of model ('sklearn', 'keras', 'pytorch')
            signature: Optional model signature
        """
        if not self.enabled:
            return
        
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path, signature=signature)
            elif model_type == "keras":
                mlflow.keras.log_model(model, artifact_path, signature=signature)
            else:
                logger.warning(f"Unsupported model type for MLflow: {model_type}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact file
        
        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact directory
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Set tags on current run
        
        Args:
            tags: Dictionary of tags
        """
        if not self.enabled:
            return
        
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.warning(f"Failed to set tags: {e}")
    
    def log_training_run(
        self,
        regime_id: int,
        model_name: str,
        model: Any,
        model_type: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        n_samples: int
    ) -> None:
        """
        Log a complete training run
        
        Args:
            regime_id: Regime identifier
            model_name: Model name (e.g., 'xgboost', 'lstm')
            model: Trained model object
            model_type: Model type ('sklearn' or 'keras')
            metrics: Training metrics (mae, rmse, etc.)
            params: Model hyperparameters
            n_samples: Number of training samples
        """
        if not self.enabled:
            return
        
        run_name = f"regime_{regime_id}_{model_name}"
        
        with self.start_run(run_name=run_name):
            # Set tags
            self.set_tags({
                "regime_id": str(regime_id),
                "model_name": model_name,
                "model_type": model_type,
                "training_stage": "regime_model"
            })
            
            # Log parameters
            self.log_params({
                **params,
                "n_samples": n_samples,
                "regime_id": regime_id
            })
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            self.log_model(
                model,
                artifact_path=f"regime_{regime_id}_model",
                model_type=model_type
            )
            
            logger.info(f"Logged training run: {run_name} - MAE: {metrics.get('mae', 'N/A')}")
    
    def log_inference_sample(
        self,
        regime_id: int,
        model_name: str,
        prediction: float,
        confidence: float,
        latency_ms: float,
        input_features: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log a sampled inference prediction (use sparingly for monitoring)
        
        Args:
            regime_id: Regime identifier
            model_name: Model used for prediction
            prediction: Predicted value
            confidence: Regime detection confidence
            latency_ms: Inference latency in milliseconds
            input_features: Optional input feature values
        """
        if not self.enabled:
            return
        
        # Only log samples (e.g., 1% of predictions) to avoid overhead
        import random
        if random.random() > 0.01:
            return
        
        with self.start_run(run_name="inference"):
            self.set_tags({
                "regime_id": str(regime_id),
                "model_name": model_name,
                "inference_stage": "production"
            })
            
            metrics_dict = {
                "prediction": prediction,
                "regime_confidence": confidence,
                "latency_ms": latency_ms
            }
            
            self.log_metrics(metrics_dict)
            
            if input_features:
                self.log_params(input_features)

    def load_model_from_registry(self, model_name: str, stage: str = "Production") -> Optional[Any]:
        """
        Load a model from the MLflow Model Registry.

        Args:
            model_name: Registered model name
            stage: Model stage (e.g., "Production", "Staging")

        Returns:
            Loaded MLflow model or None on failure
        """
        if not self.enabled:
            return None

        try:
            model_uri = f"models:/{model_name}/{stage}"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning("Failed to load model from registry %s (%s): %s", model_name, stage, e)
            return None

    def list_registry_models(self) -> Optional[list]:
        """
        List registered models and their latest versions.

        Returns:
            List of model metadata dicts or None if unavailable
        """
        if not self.enabled or self.client is None:
            return None

        try:
            models = []
            for model in self.client.search_registered_models():
                latest_versions = []
                for version in model.latest_versions or []:
                    latest_versions.append({
                        "version": version.version,
                        "stage": version.current_stage,
                        "status": version.status,
                        "run_id": version.run_id,
                    })

                models.append({
                    "name": model.name,
                    "latest_versions": latest_versions,
                })

            return models
        except Exception as e:
            logger.warning("Failed to list registry models: %s", e)
            return None


class _DummyContext:
    """Dummy context manager for when MLflow is disabled"""
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# ============================================================================
# Global Tracker Instance
# ============================================================================

_global_tracker: Optional[MLflowTracker] = None


def initialize_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "regime-aware-energy-prediction"
) -> MLflowTracker:
    """
    Initialize global MLflow tracker
    
    Args:
        tracking_uri: MLflow tracking URI
        experiment_name: Experiment name
        
    Returns:
        MLflowTracker instance
    """
    global _global_tracker
    _global_tracker = MLflowTracker(tracking_uri, experiment_name)
    return _global_tracker


def get_mlflow_tracker() -> Optional[MLflowTracker]:
    """Get the global MLflow tracker instance"""
    return _global_tracker
