"""KNN classification analysis."""
import numpy as np
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
from src.utils.metrics import calculate_classification_metrics, calculate_per_class_metrics, get_confusion_matrix
from src.utils.logger import LoggerMixin


class KNNAnalyzer(LoggerMixin):
    """Analyze KNN classification performance."""

    def __init__(self):
        """Initialize KNN analyzer."""
        self.results = {}

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze KNN classification results.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Analysis results dictionary
        """
        self.logger.info("Analyzing KNN classification performance")

        overall_metrics = calculate_classification_metrics(y_true, y_pred)
        per_class_metrics = calculate_per_class_metrics(y_true, y_pred)
        conf_matrix = get_confusion_matrix(y_true, y_pred)

        self.results = {
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': conf_matrix.tolist()
        }

        self.logger.info(f"KNN accuracy: {overall_metrics['accuracy']:.4f}")
        return self.results

    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation.

        Args:
            model: KNN model
            X: Feature matrix
            y: Target labels
            cv: Number of folds

        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {cv}-fold cross-validation")

        scores = cross_val_score(model.model, X, y, cv=cv, scoring='accuracy')

        cv_results = {
            'scores': scores.tolist(),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }

        self.results['cross_validation'] = cv_results
        self.logger.info(f"CV mean accuracy: {cv_results['mean_score']:.4f}")

        return cv_results

    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary.

        Returns:
            Summary dictionary
        """
        return self.results
