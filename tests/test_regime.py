"""
Unit tests for regime detection module
"""

import pytest
import numpy as np
import pandas as pd

from src.regime.detector import HMMRegimeDetector, RegimeDetectionPipeline


@pytest.fixture
def sample_features():
    """Generate sample feature data"""
    np.random.seed(42)
    return np.random.randn(500, 10)


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame with regimes"""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=500, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "regime": np.repeat([0, 1, 2], 167)[:500],
            "feature_1": np.random.randn(500),
            "feature_2": np.random.randn(500),
        }
    )


class TestHMMRegimeDetector:
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = HMMRegimeDetector(n_regimes=3)

        assert detector.n_regimes == 3
        assert detector.model is None

    def test_detector_training(self, sample_features):
        """Test detector can be trained"""
        detector = HMMRegimeDetector(n_regimes=3)
        labels, ll = detector.train(sample_features, n_iter=10)

        assert len(labels) == len(sample_features)
        assert detector.model is not None
        assert isinstance(ll, (int, float))

    def test_detector_prediction(self, sample_features):
        """Test trained detector can make predictions"""
        detector = HMMRegimeDetector(n_regimes=3)
        detector.train(sample_features, n_iter=10)

        predictions = detector.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert len(np.unique(predictions)) <= 3

    def test_detector_proba(self, sample_features):
        """Test probability predictions"""
        detector = HMMRegimeDetector(n_regimes=3)
        detector.train(sample_features, n_iter=10)

        proba = detector.predict_proba(sample_features)

        assert proba.shape == (len(sample_features), 3)
        assert np.allclose(proba.sum(axis=1), 1)  # Probabilities sum to 1


class TestRegimeDetectionPipeline:
    def test_pipeline_fit_predict(self, sample_dataframe):
        """Test pipeline fit and predict"""
        pipeline = RegimeDetectionPipeline(n_regimes=3)
        feature_cols = ["feature_1", "feature_2"]

        result = pipeline.fit_and_predict(sample_dataframe, feature_cols)

        assert "regime" in result.columns
        assert len(result) == len(sample_dataframe)

    def test_transition_detection(self, sample_dataframe):
        """Test regime transition detection"""
        sample_dataframe["regime"] = [0, 0, 1, 1, 2, 2, 0, 0] + [0] * (
            len(sample_dataframe) - 8
        )

        pipeline = RegimeDetectionPipeline(n_regimes=3)
        transitions = pipeline.detect_transitions(sample_dataframe)

        # Should detect transitions
        assert len(transitions) > 0
