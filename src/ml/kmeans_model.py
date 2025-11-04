"""K-means clustering model."""
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any
from src.utils.logger import LoggerMixin


class KMeansModel(LoggerMixin):
    """K-means clustering for product reviews."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, max_iter: int = 300):
        """Initialize K-means model.

        Args:
            n_clusters: Number of clusters
            random_state: Random seed
            max_iter: Maximum iterations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10
        )
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        """Fit K-means model.

        Args:
            X: Feature matrix
        """
        self.logger.info(f"Fitting K-means with {self.n_clusters} clusters")
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        self.logger.info(f"K-means fitted. Inertia: {self.inertia_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels.

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        self.logger.info(f"Predicting clusters for {len(X)} samples")
        predictions = self.model.predict(X)
        return predictions

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict cluster labels.

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        self.logger.info(f"Fit-predicting K-means on {len(X)} samples")
        labels = self.model.fit_predict(X)
        self.labels_ = labels
        self.cluster_centers_ = self.model.cluster_centers_
        self.inertia_ = self.model.inertia_
        self.logger.info(f"K-means fitted. Inertia: {self.inertia_:.4f}")
        return labels

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information.

        Returns:
            Dictionary with cluster information
        """
        return {
            'n_clusters': self.n_clusters,
            'inertia': float(self.inertia_) if self.inertia_ else None,
            'cluster_sizes': {
                int(i): int(np.sum(self.labels_ == i))
                for i in range(self.n_clusters)
            } if self.labels_ is not None else {}
        }
