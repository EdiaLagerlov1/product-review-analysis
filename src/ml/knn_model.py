"""K-Nearest Neighbors classification model."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any
from src.utils.logger import LoggerMixin


class KNNModel(LoggerMixin):
    """KNN classifier for product reviews."""

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        """Initialize KNN model.

        Args:
            n_neighbors: Number of neighbors
            metric: Distance metric
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit KNN model.

        Args:
            X: Feature matrix
            y: Target labels
        """
        self.logger.info(f"Fitting KNN with {self.n_neighbors} neighbors")
        self.model.fit(X, y)
        self.logger.info("KNN fitted successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        self.logger.info(f"Predicting labels for {len(X)} samples")
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        self.logger.info(f"Predicting probabilities for {len(X)} samples")
        probabilities = self.model.predict_proba(X)
        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Accuracy score
        """
        accuracy = self.model.score(X, y)
        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        return accuracy

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'n_samples_fit': self.model.n_samples_fit_ if hasattr(self.model, 'n_samples_fit_') else None
        }
