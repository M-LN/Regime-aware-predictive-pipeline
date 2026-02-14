"""
Per-regime model training utilities with MLflow tracking.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.monitoring import get_mlflow_tracker

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    xgb = None
    HAS_XGBOOST = False

try:
    from tensorflow import keras
    HAS_TF = True
except Exception:
    keras = None
    HAS_TF = False


@dataclass
class TrainResult:
    model_name: str
    n_samples: int
    mae: float
    rmse: float
    model_path: str


class RegimeModelTrainer:
    """Train one model per regime and persist artifacts."""

    def __init__(self, config, model_dir: str = "data/models"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def train_all_regimes(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "energy_production",
        min_samples: int = 50,
    ) -> Dict[int, TrainResult]:
        results: Dict[int, TrainResult] = {}

        if "regime" not in df.columns:
            raise ValueError("Input DataFrame must include a 'regime' column")

        for regime_id in sorted(df["regime"].unique()):
            result = self._train_regime(
                df=df,
                regime_id=int(regime_id),
                feature_cols=feature_cols,
                target_col=target_col,
                min_samples=min_samples,
            )
            if result is not None:
                results[int(regime_id)] = result

        return results

    def _train_regime(
        self,
        df: pd.DataFrame,
        regime_id: int,
        feature_cols: list,
        target_col: str,
        min_samples: int,
    ) -> Optional[TrainResult]:
        df_regime = df[df["regime"] == regime_id].dropna()
        n_samples = len(df_regime)

        if n_samples < min_samples:
            self.logger.warning(
                "Skipping regime %s: only %s samples (min %s)",
                regime_id,
                n_samples,
                min_samples,
            )
            return None

        X = df_regime[feature_cols].values
        y = df_regime[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
        )

        model_name = self._model_name_for_regime(regime_id)
        model, is_keras = self._build_model(model_name, X_train.shape[1])

        if is_keras:
            X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            model.fit(
                X_train_seq,
                y_train,
                epochs=self.config.model.lstm_epochs,
                batch_size=self.config.model.lstm_batch_size,
                verbose=0,
            )
            preds = model.predict(X_test_seq, verbose=0).reshape(-1)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        model_path = self._save_model(model, model_name, regime_id, is_keras)

        # Log to MLflow if available
        mlflow_tracker = get_mlflow_tracker()
        if mlflow_tracker and mlflow_tracker.is_connected():
            try:
                params = {
                    "model_name": model_name,
                    "regime_id": regime_id,
                    "test_size": self.config.model.test_size,
                    "random_state": self.config.model.random_state
                }
                
                # Add model-specific params
                if model_name == "xgboost":
                    params.update({
                        "max_depth": self.config.model.xgb_max_depth,
                        "learning_rate": self.config.model.xgb_learning_rate,
                        "n_estimators": self.config.model.xgb_n_estimators
                    })
                elif model_name == "lstm":
                    params.update({
                        "lstm_units": self.config.model.lstm_units,
                        "lstm_dropout": self.config.model.lstm_dropout,
                        "lstm_epochs": self.config.model.lstm_epochs,
                        "lstm_batch_size": self.config.model.lstm_batch_size
                    })
                elif model_name == "random_forest":
                    params.update({
                        "n_estimators": self.config.model.rf_n_estimators,
                        "max_depth": self.config.model.rf_max_depth
                    })
                
                mlflow_tracker.log_training_run(
                    regime_id=regime_id,
                    model_name=model_name,
                    model=model,
                    model_type="keras" if is_keras else "sklearn",
                    metrics={"mae": mae, "rmse": rmse},
                    params=params,
                    n_samples=n_samples
                )
                self.logger.info("Logged training run to MLflow for regime %s", regime_id)
            except Exception as e:
                self.logger.warning("Failed to log to MLflow: %s", e)

        self.logger.info(
            "Regime %s model trained (%s). MAE=%.3f RMSE=%.3f",
            regime_id,
            model_name,
            mae,
            rmse,
        )

        return TrainResult(
            model_name=model_name,
            n_samples=n_samples,
            mae=mae,
            rmse=rmse,
            model_path=model_path,
        )

    def _model_name_for_regime(self, regime_id: int) -> str:
        if regime_id == 0:
            return self.config.model.regime_a_model
        if regime_id == 1:
            return self.config.model.regime_b_model
        return self.config.model.regime_c_model

    def _build_model(self, model_name: str, input_dim: int):
        model_name = model_name.lower()

        if model_name == "xgboost" and HAS_XGBOOST:
            model = xgb.XGBRegressor(
                max_depth=self.config.model.xgb_max_depth,
                learning_rate=self.config.model.xgb_learning_rate,
                n_estimators=self.config.model.xgb_n_estimators,
                objective="reg:squarederror",
                random_state=self.config.model.random_state,
            )
            return model, False

        if model_name == "lstm" and HAS_TF:
            model = keras.Sequential(
                [
                    keras.layers.Input(shape=(1, input_dim)),
                    keras.layers.LSTM(self.config.model.lstm_units, dropout=self.config.model.lstm_dropout),
                    keras.layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            return model, True

        if model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=self.config.model.rf_n_estimators,
                max_depth=self.config.model.rf_max_depth,
                random_state=self.config.model.random_state,
            )
            return model, False

        if model_name == "xgboost":
            self.logger.warning("XGBoost not available; using HistGradientBoostingRegressor")
            return HistGradientBoostingRegressor(random_state=self.config.model.random_state), False

        if model_name == "lstm":
            self.logger.warning("TensorFlow not available; using MLPRegressor")
            return MLPRegressor(hidden_layer_sizes=(64, 32), random_state=self.config.model.random_state), False

        self.logger.warning("Unknown model '%s'; using RandomForestRegressor", model_name)
        return RandomForestRegressor(random_state=self.config.model.random_state), False

    def _save_model(self, model, model_name: str, regime_id: int, is_keras: bool) -> str:
        filename = f"regime_{regime_id}_{model_name}"
        if is_keras:
            path = self.model_dir / f"{filename}.keras"
            model.save(path)
        else:
            path = self.model_dir / f"{filename}.pkl"
            joblib.dump(model, path)
        return str(path)
