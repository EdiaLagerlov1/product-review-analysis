"""Common metrics calculation utilities."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
)
from typing import Dict, Any


def calculate_classification_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     average: str = 'weighted') -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def calculate_clustering_metrics(X: np.ndarray,
                                 labels_true: np.ndarray,
                                 labels_pred: np.ndarray) -> Dict[str, float]:
    """Calculate clustering evaluation metrics.

    Args:
        X: Feature matrix
        labels_true: True labels
        labels_pred: Predicted cluster labels

    Returns:
        Dictionary of clustering metrics
    """
    metrics = {
        'adjusted_rand_index': adjusted_rand_score(labels_true, labels_pred),
        'normalized_mutual_info': normalized_mutual_info_score(labels_true, labels_pred)
    }

    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def calculate_per_class_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Calculate per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary mapping class to its metrics
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))

    return {
        int(cls): {
            'precision': float(precision[i]) if i < len(precision) else 0.0,
            'recall': float(recall[i]) if i < len(recall) else 0.0,
            'f1_score': float(f1[i]) if i < len(f1) else 0.0
        }
        for i, cls in enumerate(unique_classes)
    }
