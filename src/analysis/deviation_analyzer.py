"""Deviation analysis for clustering."""
import numpy as np
from typing import Dict, Any
from src.utils.logger import LoggerMixin


class DeviationAnalyzer(LoggerMixin):
    """Analyze deviation between clusters and ground truth."""

    def __init__(self):
        """Initialize deviation analyzer."""
        self.results = {}

    def analyze_deviation(self, X: np.ndarray, cluster_labels: np.ndarray,
                         true_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering deviation.

        Args:
            X: Feature matrix
            cluster_labels: Predicted cluster labels
            true_labels: True category labels

        Returns:
            Deviation analysis results
        """
        self.logger.info("Analyzing clustering deviation")

        mismatch_rate = float(np.sum(cluster_labels != true_labels) / len(true_labels))

        self.results = {
            'mismatch_rate': mismatch_rate,
        }

        self.logger.info(f"Mismatch rate: {mismatch_rate:.4f}")
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get deviation analysis summary.

        Returns:
            Summary dictionary
        """
        return self.results
