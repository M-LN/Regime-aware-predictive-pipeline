"""
Regime detection module using Hidden Markov Models
"""

import logging
from typing import Tuple, Optional, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm

try:
    import ruptures as rpt

    HAS_RUPTURES = True
except Exception:
    rpt = None
    HAS_RUPTURES = False

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.

    Assumes 3 regimes:
    - Regime 0: High volatility/unstable market
    - Regime 1: Medium volatility/normal market
    - Regime 2: Low volatility/stable market
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden regimes (default 3)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self.regime_names = {0: "volatile", 1: "neutral", 2: "stable"}
        self.logger = logging.getLogger(__name__)

    def train(
        self, features: np.ndarray, n_iter: int = 200
    ) -> Tuple[np.ndarray, float]:
        """
        Train HMM model on feature data.

        Args:
            features: Feature array (n_samples, n_features)
            n_iter: Number of iterations for EM algorithm

        Returns:
            Tuple of (regime labels, log likelihood)
        """
        try:
            # Initialize and train HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                n_iter=n_iter,
                tol=1e-2,
                covariance_type="diag",
                random_state=self.random_state,
                verbose=1,
            )

            self.model.fit(features)

            # Get regime labels
            regime_labels = self.model.predict(features)
            log_likelihood = self.model.score(features)

            self.logger.info(
                f"HMM trained successfully. "
                f"Log likelihood: {log_likelihood:.2f}. "
                f"Regime distribution: {np.bincount(regime_labels)}"
            )

            return regime_labels, log_likelihood

        except Exception as e:
            self.logger.error(f"Failed to train HMM: {e}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new features.

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Array of regime labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities (soft labels).

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_regimes) with probability per regime
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Compute posterior probability of states
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)

        _, posteriors = self.model.score_samples(features)
        return posteriors

    def get_regime_metadata(self, df: pd.DataFrame, regime_label: int) -> Dict:
        """
        Get statistics for a specific regime.

        Args:
            df: DataFrame with regime column
            regime_label: Regime number

        Returns:
            Dictionary with regime statistics
        """
        regime_df = df[df["regime"] == regime_label]

        if len(regime_df) == 0:
            return {}

        numeric_cols = regime_df.select_dtypes(include=[np.number]).columns

        metadata = {
            "regime": regime_label,
            "regime_name": self.regime_names.get(
                regime_label, f"regime_{regime_label}"
            ),
            "n_samples": len(regime_df),
            "date_range": {
                "start": (
                    regime_df["timestamp"].min().isoformat()
                    if "timestamp" in regime_df.columns
                    else None
                ),
                "end": (
                    regime_df["timestamp"].max().isoformat()
                    if "timestamp" in regime_df.columns
                    else None
                ),
            },
            "statistics": regime_df[numeric_cols].describe().to_dict(),
        }

        return metadata


class BayesianCPDRegimeDetector:
    """
    Detects regimes using change point detection (CPD).

    Uses ruptures PELT to detect change points, then assigns regimes
    by segment volatility.
    """

    def __init__(self, n_regimes: int = 3, penalty: float = 10.0, min_size: int = 24):
        self.n_regimes = n_regimes
        self.penalty = penalty
        self.min_size = min_size
        self.breakpoints: List[int] = []
        self.regime_names = {
            0: "volatile",
            1: "neutral",
            2: "stable",
        }
        self.logger = logging.getLogger(__name__)

    def train(self, signal: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if not HAS_RUPTURES:
            raise RuntimeError("ruptures is not installed")

        if signal.ndim > 1:
            signal = signal.reshape(-1)

        algo = rpt.Pelt(model="rbf", min_size=self.min_size).fit(signal)
        breakpoints = algo.predict(pen=self.penalty)
        self.breakpoints = breakpoints

        labels = self._labels_from_breakpoints(signal, breakpoints)
        return labels, breakpoints

    def predict(self, signal: np.ndarray) -> np.ndarray:
        if not self.breakpoints:
            labels, _ = self.train(signal)
            return labels
        return self._labels_from_breakpoints(signal.reshape(-1), self.breakpoints)

    def predict_proba(self, signal: np.ndarray) -> np.ndarray:
        labels = self.predict(signal)
        proba = np.zeros((len(labels), self.n_regimes))
        for idx, label in enumerate(labels):
            proba[idx, int(label)] = 1.0
        return proba

    def _labels_from_breakpoints(
        self, signal: np.ndarray, breakpoints: List[int]
    ) -> np.ndarray:
        segments = []
        start = 0
        for end in breakpoints:
            segment = signal[start:end]
            if len(segment) == 0:
                continue
            segments.append((start, end, float(np.std(segment))))
            start = end

        if not segments:
            return np.zeros(len(signal), dtype=int)

        # Rank segments by volatility (std) and map to regime ids
        volatilities = np.array([seg[2] for seg in segments])
        order = np.argsort(volatilities)

        # Map lowest volatility to stable, highest to volatile
        regime_map = {}
        for rank, seg_idx in enumerate(order):
            if self.n_regimes == 1:
                regime_id = 1
            else:
                bucket = int(
                    round((rank / max(len(segments) - 1, 1)) * (self.n_regimes - 1))
                )
                regime_id = (self.n_regimes - 1) - bucket
            regime_map[seg_idx] = regime_id

        labels = np.zeros(len(signal), dtype=int)
        for seg_idx, (start, end, _) in enumerate(segments):
            labels[start:end] = regime_map.get(seg_idx, 1)

        return labels


class RegimeDetectionPipeline:
    """Orchestrates regime detection workflow"""

    def __init__(
        self,
        n_regimes: int = 3,
        algorithm: str = "hmm",
        cpd_penalty: float = 10.0,
        cpd_min_size: int = 24,
    ):
        self.algorithm = algorithm
        if algorithm == "bayesian_cpd":
            self.detector = BayesianCPDRegimeDetector(
                n_regimes=n_regimes, penalty=cpd_penalty, min_size=cpd_min_size
            )
        else:
            self.detector = HMMRegimeDetector(n_regimes=n_regimes)
        self.logger = logging.getLogger(__name__)

    def fit_and_predict(
        self, df: pd.DataFrame, feature_cols: list, signal_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fit HMM and predict regimes for DataFrame.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names

        Returns:
            DataFrame with 'regime' and 'regime_proba_*' columns added
        """
        # Extract features
        X = df[feature_cols].values

        if self.algorithm == "bayesian_cpd":
            if signal_col and signal_col in df.columns:
                signal = df[signal_col].values
            else:
                signal = X[:, 0]

            regime_labels, _ = self.detector.train(signal)
            regime_proba = self.detector.predict_proba(signal)
        else:
            # Train HMM
            regime_labels, _ = self.detector.train(X)

            # Get probabilities
            regime_proba = self.detector.predict_proba(X)

        # Add to DataFrame
        df_with_regime = df.copy()
        df_with_regime["regime"] = regime_labels

        for i in range(self.detector.n_regimes):
            df_with_regime[f"regime_proba_{i}"] = regime_proba[:, i]

        # Add regime name
        df_with_regime["regime_name"] = df_with_regime["regime"].map(
            self.detector.regime_names
        )

        self.logger.info(
            "Regime detection complete (%s). Unique regimes: %s",
            self.algorithm,
            df_with_regime["regime"].nunique(),
        )

        return df_with_regime

    def detect_transitions(
        self, df: pd.DataFrame, min_duration: int = 1
    ) -> pd.DataFrame:
        """
        Detect regime transitions (when regime changes).

        Args:
            df: DataFrame with 'regime' column
            min_duration: Minimum hours in a regime to count as transition

        Returns:
            DataFrame of transitions with timestamps and durations
        """
        transitions = []

        regime_shift = df["regime"].diff() != 0
        transition_indices = np.where(regime_shift)[0]

        for idx in transition_indices:
            if idx > 0:
                prev_regime = df.iloc[idx - 1]["regime"]
                curr_regime = df.iloc[idx]["regime"]
                timestamp = df.iloc[idx]["timestamp"]

                transitions.append(
                    {
                        "timestamp": timestamp,
                        "from_regime": prev_regime,
                        "to_regime": curr_regime,
                        "from_regime_name": self.detector.regime_names.get(prev_regime),
                        "to_regime_name": self.detector.regime_names.get(curr_regime),
                    }
                )

        transitions_df = pd.DataFrame(transitions)

        if len(transitions_df) > 0:
            self.logger.info(f"Detected {len(transitions_df)} regime transitions")

        return transitions_df

    def save_model(self, path: str = "data/models/regime_detector.pkl") -> bool:
        """Save trained HMM model"""
        try:
            import pickle

            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(self.detector, f)

            self.logger.info(f"Model saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str = "data/models/regime_detector.pkl") -> bool:
        """Load pre-trained HMM model"""
        try:
            import pickle

            with open(path, "rb") as f:
                self.detector = pickle.load(f)

            self.logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
