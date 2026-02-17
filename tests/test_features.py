"""
Unit tests for feature engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.engineer import FeatureEngineer, FeatureScaler


@pytest.fixture
def sample_data():
    """Generate sample energy data"""
    dates = pd.date_range("2026-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "wind_speed": np.random.rand(100) * 20,
            "energy_production": np.random.rand(100) * 1000,
            "temperature": np.random.rand(100) * 50,
            "price": np.random.rand(100) * 500,
        }
    )


@pytest.fixture
def feature_engineer():
    return FeatureEngineer(rolling_windows=[1, 6], volatility_windows=[6])


class TestFeatureEngineer:
    def test_engineer_features_returns_dataframe(self, feature_engineer, sample_data):
        """Test that engineering returns DataFrame"""
        result = feature_engineer.engineer_features(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_engineer_features_generates_columns(self, feature_engineer, sample_data):
        """Test that new features are generated"""
        result = feature_engineer.engineer_features(sample_data)

        # Should have more columns than input
        assert len(result.columns) > len(sample_data.columns)

    def test_temporal_features_created(self, feature_engineer, sample_data):
        """Test temporal features are created"""
        result = feature_engineer.engineer_features(sample_data)

        temporal_features = ["hour", "day_of_week", "is_weekend"]
        for feat in temporal_features:
            assert feat in result.columns

    def test_rolling_features_created(self, feature_engineer, sample_data):
        """Test rolling window features are created"""
        result = feature_engineer.engineer_features(sample_data)

        # Should have rolling mean/std for each window and numeric column
        assert any("_ma_" in col for col in result.columns)
        assert any("_std_" in col for col in result.columns)


class TestFeatureScaler:
    def test_minmax_scaler_fit(self, sample_data):
        """Test MinMax scaler fitting"""
        scaler = FeatureScaler(method="minmax")
        cols = ["wind_speed", "price"]
        scaler.fit(sample_data, cols)

        assert all(col in scaler.scalers for col in cols)

    def test_minmax_scaler_transform(self, sample_data):
        """Test MinMax scaler transforms values to [0, 1]"""
        scaler = FeatureScaler(method="minmax")
        cols = ["wind_speed", "price"]
        scaler.fit(sample_data, cols)

        result = scaler.transform(sample_data, cols)

        # scaled values should be between 0 and 1
        assert (result[cols] >= -0.01).all().all()  # Allow small numerical errors
        assert (result[cols] <= 1.01).all().all()

    def test_zscore_scaler(self, sample_data):
        """Test Standard (Z-score) scaler"""
        scaler = FeatureScaler(method="zscore")
        cols = ["wind_speed", "price"]

        result = scaler.fit_transform(sample_data, cols)

        # Standard scaled features should have mean~0 and std~1
        for col in cols:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std() - 1.0) < 0.1
