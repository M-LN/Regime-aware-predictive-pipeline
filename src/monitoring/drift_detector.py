"""
Drift detection for monitoring distribution shifts

Detects:
- Feature distribution drift (KL divergence, Wasserstein distance)
- Prediction distribution drift
- Regime distribution shifts
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta, UTC
import numpy as np
from scipy import stats
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Monitor distribution drift using statistical tests
    """
    
    def __init__(
        self,
        reference_window: int = 1000,
        detection_window: int = 100,
        kl_threshold: float = 0.1,
        wasserstein_threshold: float = 0.3,
        ks_p_value: float = 0.05
    ):
        """
        Initialize drift detector
        
        Args:
            reference_window: Number of samples for reference distribution
            detection_window: Number of samples for current distribution
            kl_threshold: KL divergence threshold for drift alert
            wasserstein_threshold: Wasserstein distance threshold
            ks_p_value: Kolmogorov-Smirnov test p-value threshold
        """
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.kl_threshold = kl_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.ks_p_value = ks_p_value
        
        # Buffers for feature values
        self.feature_buffers: Dict[str, deque] = {}
        self.prediction_buffer = deque(maxlen=reference_window)
        self.regime_buffer = deque(maxlen=reference_window)
        
        # Reference distributions
        self.reference_features: Dict[str, np.ndarray] = {}
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_regime_counts: Optional[Dict[int, int]] = None
        
        # Drift alerts
        self.drift_alerts: List[Dict] = []
        self.last_check_time = datetime.now(UTC)
    
    def update_features(self, features: Dict[str, float]) -> None:
        """
        Update feature buffers with new observation
        
        Args:
            features: Dictionary of feature values
        """
        for name, value in features.items():
            if name not in self.feature_buffers:
                self.feature_buffers[name] = deque(maxlen=self.reference_window)
            self.feature_buffers[name].append(value)
    
    def update_prediction(self, prediction: float) -> None:
        """
        Update prediction buffer
        
        Args:
            prediction: Predicted value
        """
        self.prediction_buffer.append(prediction)
    
    def update_regime(self, regime_id: int) -> None:
        """
        Update regime buffer
        
        Args:
            regime_id: Regime identifier
        """
        self.regime_buffer.append(regime_id)
    
    def set_reference_distributions(self) -> None:
        """
        Set current distributions as reference baseline
        """
        # Feature reference distributions
        for name, buffer in self.feature_buffers.items():
            if len(buffer) >= self.detection_window:
                self.reference_features[name] = np.array(list(buffer))
        
        # Prediction reference distribution
        if len(self.prediction_buffer) >= self.detection_window:
            self.reference_predictions = np.array(list(self.prediction_buffer))
        
        # Regime reference distribution
        if len(self.regime_buffer) >= self.detection_window:
            regime_arr = np.array(list(self.regime_buffer))
            unique, counts = np.unique(regime_arr, return_counts=True)
            self.reference_regime_counts = dict(zip(unique, counts))
        
        logger.info("Reference distributions set with %d features", len(self.reference_features))
    
    def check_drift(
        self,
        auto_update_reference: bool = False
    ) -> Dict[str, any]:
        """
        Check for distribution drift
        
        Args:
            auto_update_reference: Automatically update reference if no drift detected
        
        Returns:
            Dictionary with drift detection results
        """
        results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "drift_detected": False,
            "feature_drift": {},
            "prediction_drift": None,
            "regime_drift": None,
            "alerts": []
        }
        
        # Check feature drift
        for name, buffer in self.feature_buffers.items():
            if name not in self.reference_features:
                continue
            
            if len(buffer) < self.detection_window:
                continue
            
            current = np.array(list(buffer)[-self.detection_window:])
            reference = self.reference_features[name]
            
            drift_metrics = self._detect_feature_drift(current, reference, name)
            results["feature_drift"][name] = drift_metrics
            
            if drift_metrics["drift_detected"]:
                results["drift_detected"] = True
                results["alerts"].append({
                    "type": "feature_drift",
                    "timestamp": results["timestamp"],
                    "feature": name,
                    "severity": drift_metrics["severity"],
                    "metrics": drift_metrics
                })
        
        # Check prediction drift
        if self.reference_predictions is not None and len(self.prediction_buffer) >= self.detection_window:
            current_preds = np.array(list(self.prediction_buffer)[-self.detection_window:])
            pred_drift = self._detect_feature_drift(
                current_preds,
                self.reference_predictions,
                "predictions"
            )
            results["prediction_drift"] = pred_drift
            
            if pred_drift["drift_detected"]:
                results["drift_detected"] = True
                results["alerts"].append({
                    "type": "prediction_drift",
                    "timestamp": results["timestamp"],
                    "severity": pred_drift["severity"],
                    "metrics": pred_drift
                })
        
        # Check regime distribution drift
        if self.reference_regime_counts is not None and len(self.regime_buffer) >= self.detection_window:
            current_regimes = np.array(list(self.regime_buffer)[-self.detection_window:])
            regime_drift = self._detect_regime_drift(current_regimes)
            results["regime_drift"] = regime_drift
            
            if regime_drift["drift_detected"]:
                results["drift_detected"] = True
                results["alerts"].append({
                    "type": "regime_drift",
                    "timestamp": results["timestamp"],
                    "severity": regime_drift["severity"],
                    "current_distribution": regime_drift["current_distribution"],
                    "reference_distribution": regime_drift["reference_distribution"]
                })
        
        # Store alerts
        if results["drift_detected"]:
            self.drift_alerts.extend(results["alerts"])
            logger.warning("Drift detected: %d alerts", len(results["alerts"]))
        
        # Auto-update reference if requested and no drift
        if auto_update_reference and not results["drift_detected"]:
            self.set_reference_distributions()
        
        self.last_check_time = datetime.now(UTC)
        return results
    
    def _detect_feature_drift(
        self,
        current: np.ndarray,
        reference: np.ndarray,
        name: str
    ) -> Dict[str, any]:
        """
        Detect drift in a single feature using multiple metrics
        
        Args:
            current: Current distribution samples
            reference: Reference distribution samples
            name: Feature name
        
        Returns:
            Dictionary with drift metrics
        """
        # Remove NaNs
        current = current[~np.isnan(current)]
        reference = reference[~np.isnan(reference)]
        
        if len(current) < 10 or len(reference) < 10:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        ks_drift = ks_pvalue < self.ks_p_value
        
        # Wasserstein distance (earth mover's distance)
        wasserstein = stats.wasserstein_distance(reference, current)
        wasserstein_drift = wasserstein > self.wasserstein_threshold
        
        # KL divergence (requires binning)
        try:
            kl_div = self._compute_kl_divergence(reference, current)
            kl_drift = kl_div > self.kl_threshold
        except Exception as e:
            logger.debug(f"KL divergence computation failed for {name}: {e}")
            kl_div = None
            kl_drift = False
        
        # Mean shift
        mean_shift = abs(np.mean(current) - np.mean(reference))
        mean_shift_pct = mean_shift / (abs(np.mean(reference)) + 1e-8) * 100
        
        # Std shift
        std_shift = abs(np.std(current) - np.std(reference))
        std_shift_pct = std_shift / (abs(np.std(reference)) + 1e-8) * 100
        
        # Overall drift decision
        drift_detected = ks_drift or wasserstein_drift or kl_drift
        
        # Severity: high if multiple tests fail, medium if one fails
        severity = "high" if sum([ks_drift, wasserstein_drift, kl_drift]) >= 2 else \
                   "medium" if drift_detected else "low"
        
        return {
            "drift_detected": drift_detected,
            "severity": severity,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "ks_drift": ks_drift,
            "wasserstein_distance": float(wasserstein),
            "wasserstein_drift": wasserstein_drift,
            "kl_divergence": float(kl_div) if kl_div is not None else None,
            "kl_drift": kl_drift,
            "mean_shift_pct": float(mean_shift_pct),
            "std_shift_pct": float(std_shift_pct),
            "current_mean": float(np.mean(current)),
            "reference_mean": float(np.mean(reference)),
            "current_std": float(np.std(current)),
            "reference_std": float(np.std(reference))
        }
    
    def _compute_kl_divergence(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """
        Compute KL divergence between two distributions
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for histograms
        
        Returns:
            KL divergence value
        """
        # Create common bins
        all_data = np.concatenate([reference, current])
        bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)
        
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        
        # Normalize to probability distributions
        ref_prob = (ref_hist + 1e-10) / (ref_hist.sum() + 1e-10 * n_bins)
        cur_prob = (cur_hist + 1e-10) / (cur_hist.sum() + 1e-10 * n_bins)
        
        # Compute KL divergence
        kl_div = np.sum(cur_prob * np.log(cur_prob / ref_prob))
        
        return kl_div
    
    def _detect_regime_drift(self, current_regimes: np.ndarray) -> Dict[str, any]:
        """
        Detect drift in regime distribution
        
        Args:
            current_regimes: Current regime assignments
        
        Returns:
            Dictionary with regime drift metrics
        """
        # Current distribution
        unique, counts = np.unique(current_regimes, return_counts=True)
        current_counts = dict(zip(unique, counts))
        
        # Normalize to probabilities
        total_current = sum(current_counts.values())
        current_probs = {k: v/total_current for k, v in current_counts.items()}
        
        total_ref = sum(self.reference_regime_counts.values())
        ref_probs = {k: v/total_ref for k, v in self.reference_regime_counts.items()}
        
        # Chi-square test
        all_regimes = set(list(current_counts.keys()) + list(self.reference_regime_counts.keys()))
        observed = [current_counts.get(r, 0) for r in sorted(all_regimes)]
        expected = [self.reference_regime_counts.get(r, 0) * (total_current / total_ref) 
                   for r in sorted(all_regimes)]
        
        # Avoid division by zero
        expected = [max(e, 1e-10) for e in expected]
        
        chi2_stat = sum([(obs - exp)**2 / exp for obs, exp in zip(observed, expected)])
        
        # Simple threshold-based drift detection
        drift_detected = chi2_stat > 10.0  # Threshold for chi-square with 2 df at p<0.01
        
        return {
            "drift_detected": drift_detected,
            "severity": "high" if chi2_stat > 20 else "medium" if drift_detected else "low",
            "chi2_statistic": float(chi2_stat),
            "current_distribution": current_probs,
            "reference_distribution": ref_probs
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Get recent drift alerts
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of recent alerts
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        return [alert for alert in self.drift_alerts 
                if datetime.fromisoformat(alert.get("timestamp", "2000-01-01")) > cutoff]
    
    def get_status(self) -> Dict[str, any]:
        """
        Get drift detector status
        
        Returns:
            Status dictionary
        """
        return {
            "reference_distributions_set": len(self.reference_features) > 0,
            "n_features_tracked": len(self.feature_buffers),
            "n_reference_features": len(self.reference_features),
            "prediction_samples": len(self.prediction_buffer),
            "regime_samples": len(self.regime_buffer),
            "recent_alerts_24h": len(self.get_recent_alerts(24)),
            "last_check": self.last_check_time.isoformat()
        }
