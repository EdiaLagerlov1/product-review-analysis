"""Cluster analysis and evaluation."""
import numpy as np
from typing import Dict, Any
from src.utils.metrics import calculate_clustering_metrics, get_confusion_matrix
from src.utils.logger import LoggerMixin


class ClusterAnalyzer(LoggerMixin):
    """Analyze K-means clustering performance."""

    def __init__(self):
        """Initialize cluster analyzer."""
        self.results = {}

    def map_clusters_to_labels(self, y_true: np.ndarray,
                               y_pred: np.ndarray) -> np.ndarray:
        """Map cluster IDs to semantic labels based on majority voting.

        Args:
            y_true: True labels (0=Negative, 1=Neutral, 2=Positive)
            y_pred: Predicted cluster IDs

        Returns:
            Relabeled predictions with semantic labels
        """
        mapping = {}
        for cluster in np.unique(y_pred):
            cluster_mask = y_pred == cluster
            cluster_true_labels = y_true[cluster_mask]
            most_common_label = np.bincount(cluster_true_labels).argmax()
            mapping[cluster] = most_common_label

        self.logger.info(f"Cluster to label mapping: {mapping}")
        relabeled = np.array([mapping[cluster] for cluster in y_pred])
        return relabeled

    def analyze(self, X: np.ndarray, y_true: np.ndarray,
                y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering results.

        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted cluster labels

        Returns:
            Analysis results dictionary
        """
        self.logger.info("Analyzing clustering performance")

        # Relabel clusters to semantic labels
        y_pred_relabeled = self.map_clusters_to_labels(y_true, y_pred)

        metrics = calculate_clustering_metrics(X, y_true, y_pred_relabeled)
        conf_matrix = get_confusion_matrix(y_true, y_pred_relabeled)

        cluster_distribution = {
            int(i): int(np.sum(y_pred_relabeled == i))
            for i in np.unique(y_pred_relabeled)
        }

        true_distribution = {
            int(i): int(np.sum(y_true == i))
            for i in np.unique(y_true)
        }

        self.results = {
            'metrics': metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'cluster_distribution': cluster_distribution,
            'true_distribution': true_distribution
        }

        # Store relabeled predictions separately (not in JSON results)
        self._relabeled_predictions = y_pred_relabeled

        self.logger.info(f"Clustering metrics: {metrics}")
        # Return results with relabeled predictions for pipeline use
        return {**self.results, 'relabeled_predictions': y_pred_relabeled}

    def get_misclassified_samples(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """Get indices of misclassified samples.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Indices of misclassified samples
        """
        misclassified = np.where(y_true != y_pred)[0]
        self.logger.info(f"Found {len(misclassified)} misclassified samples")
        return misclassified

    def get_cluster_purity(self, y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[int, float]:
        """Calculate purity for each cluster.

        Args:
            y_true: True labels
            y_pred: Predicted cluster labels

        Returns:
            Dictionary mapping cluster to purity
        """
        purity = {}

        for cluster in np.unique(y_pred):
            cluster_mask = y_pred == cluster
            cluster_true_labels = y_true[cluster_mask]

            if len(cluster_true_labels) > 0:
                most_common = np.bincount(cluster_true_labels).argmax()
                purity[int(cluster)] = float(
                    np.sum(cluster_true_labels == most_common) / len(cluster_true_labels)
                )

        self.logger.info(f"Cluster purity: {purity}")
        return purity

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary.

        Returns:
            Summary dictionary
        """
        return self.results
