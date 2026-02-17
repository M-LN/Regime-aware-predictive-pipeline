"""
Feature engineering module for computing and managing features
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes engineered features from raw energy data"""

    def __init__(
        self, rolling_windows: List[int] = None, volatility_windows: List[int] = None
    ):
        """
        Initialize feature engineer.

        Args:
            rolling_windows: Window sizes (hours) for rolling statistics [1, 6, 24]
            volatility_windows: Window sizes for volatility computation [6, 24]
        """
        self.rolling_windows = rolling_windows or [1, 6, 24]
        self.volatility_windows = volatility_windows or [6, 24]
        self.logger = logging.getLogger(__name__)

    def engineer_features(self, df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            df: Raw DataFrame with columns [timestamp, wind_speed, energy_production,
                                            temperature, price]

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = df.sort_values("timestamp")

        # Temporal features
        df = self._add_temporal_features(df)

        # Rolling statistics
        df = self._add_rolling_statistics(df)

        # Volatility features
        df = self._add_volatility_features(df)

        # Trend features
        df = self._add_trend_features(df)

        # Lag features
        df = self._add_lag_features(df)

        # Drop rows with NaN (from rolling/lag operations)
        if dropna:
            df = df.dropna()

        self.logger.info(f"Generated {len(df.columns)} features for {len(df)} records")
        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        features = {}
        features["hour"] = df["timestamp"].dt.hour
        features["day_of_week"] = df["timestamp"].dt.dayofweek
        features["day_of_year"] = df["timestamp"].dt.dayofyear
        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for hour and day_of_week
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics"""
        numeric_cols = ["wind_speed", "energy_production", "temperature", "price"]
        features = {}

        for col in numeric_cols:
            for window in self.rolling_windows:
                # Mean and std
                features[f"{col}_ma_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                features[f"{col}_std_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).std(ddof=0)
                )

                # Min and max
                features[f"{col}_min_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                features[f"{col}_max_{window}h"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility metrics"""
        numeric_cols = ["wind_speed", "energy_production", "temperature", "price"]
        features = {}

        for col in numeric_cols:
            for window in self.volatility_windows:
                # Coefficient of variation (volatility)
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std(ddof=0)
                features[f"{col}_cv_{window}h"] = rolling_std / (
                    rolling_mean + 1e-6
                )  # Add epsilon for safety

                # Range (max - min) as volatility proxy
                rolling_range = (
                    df[col].rolling(window=window, min_periods=1).max()
                    - df[col].rolling(window=window, min_periods=1).min()
                )
                features[f"{col}_range_{window}h"] = rolling_range

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend and rate-of-change features"""
        numeric_cols = ["wind_speed", "energy_production", "temperature", "price"]
        features = {}

        for col in numeric_cols:
            # Rate of change (1h, 6h, 24h)
            features[f"{col}_roc_1h"] = df[col].pct_change(periods=1)
            features[f"{col}_roc_6h"] = df[col].pct_change(periods=6)
            features[f"{col}_roc_24h"] = df[col].pct_change(periods=24)

            # Difference
            features[f"{col}_diff_1h"] = df[col].diff(1)
            features[f"{col}_diff_6h"] = df[col].diff(6)

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for temporal dependencies"""
        numeric_cols = ["wind_speed", "energy_production", "temperature", "price"]
        lag_periods = [1, 6, 12, 24]
        features = {}

        for col in numeric_cols:
            for lag in lag_periods:
                features[f"{col}_lag_{lag}h"] = df[col].shift(lag)

        return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


class FeatureScaler:
    """Scales features for normalization"""

    def __init__(self, method: str = "minmax"):
        """
        Initialize scaler.

        Args:
            method: "minmax" or "zscore"
        """
        self.method = method
        self.scalers: Dict[str, object] = {}
        self.logger = logging.getLogger(__name__)

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """
        Fit scaler on data.

        Args:
            df: Training DataFrame
            feature_cols: Columns to scale
        """
        for col in feature_cols:
            if self.method == "minmax":
                scaler = MinMaxScaler()
            else:  # zscore
                scaler = StandardScaler()

            scaler.fit(df[[col]])
            self.scalers[col] = scaler

        self.logger.info(
            f"Fitted {self.method} scaler for {len(feature_cols)} features"
        )

    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Transform features.

        Args:
            df: DataFrame to transform
            feature_cols: Columns to scale

        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()

        for col in feature_cols:
            if col in self.scalers:
                df_scaled[[col]] = self.scalers[col].transform(df[[col]])

        return df_scaled

    def fit_transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)


class FeatureStore:
    """Manages feature versioning and storage"""

    def __init__(self, storage_path: str = "data/features"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_features(
        self,
        df: pd.DataFrame,
        version: str = "1.0.0",
        feature_set_name: str = "energy_features",
    ) -> bool:
        """
        Save features with versioning metadata.

        Args:
            df: Features DataFrame
            version: Feature set version
            feature_set_name: Name of feature set

        Returns:
            True if successful
        """
        try:
            file_path = (
                self.storage_path
                / f"{feature_set_name}_v{version}_{datetime.utcnow().isoformat()}.parquet"
            )

            df.to_parquet(file_path, engine="pyarrow", index=False)

            # Save metadata
            metadata = {
                "version": version,
                "feature_set_name": feature_set_name,
                "n_samples": len(df),
                "n_features": len(df.columns),
                "columns": df.columns.tolist(),
                "created_at": datetime.utcnow().isoformat(),
            }

            self.logger.info(f"Saved features v{version}: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save features: {e}")
            return False

    def load_features(self, version: str = "latest") -> Optional[pd.DataFrame]:
        """Load features by version"""
        try:
            # For now, load the latest file
            parquet_files = list(self.storage_path.glob("*.parquet"))
            if not parquet_files:
                self.logger.warning("No feature files found")
                return None

            latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(latest_file)

            self.logger.info(f"Loaded features from {latest_file}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load features: {e}")
            return None
