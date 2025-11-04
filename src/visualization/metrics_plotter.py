"""Metrics visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
from src.utils.logger import LoggerMixin


class MetricsPlotter(LoggerMixin):
    """Visualize performance metrics."""

    def __init__(self, output_dir: str = "outputs/plots", figsize: tuple = (12, 8)):
        """Initialize metrics plotter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, labels: list = None,
                             title: str = 'Confusion Matrix',
                             filename: str = 'confusion_matrix.png'):
        """Plot confusion matrix."""
        self.logger.info("Plotting confusion matrix")

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved confusion matrix to {filepath}")

    def plot_metrics_bar(self, metrics: Dict[str, float], title: str = 'Performance Metrics',
                        filename: str = 'metrics_bar.png'):
        """Plot metrics as bar chart."""
        self.logger.info("Plotting metrics bar chart")

        plt.figure(figsize=(10, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}', ha='center', va='bottom')

        plt.title(title)
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved metrics bar chart to {filepath}")

    def plot_comparison_metrics(self, kmeans_metrics: Dict[str, float],
                               knn_metrics: Dict[str, float],
                               filename: str = 'model_comparison.png'):
        """Plot comparison between models."""
        self.logger.info("Plotting model comparison")

        common_metrics = set(kmeans_metrics.keys()) & set(knn_metrics.keys())
        metric_names = list(common_metrics)
        kmeans_vals = [kmeans_metrics[m] for m in metric_names]
        knn_vals = [knn_metrics[m] for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, kmeans_vals, width, label='K-means', alpha=0.7)
        ax.bar(x + width / 2, knn_vals, width, label='KNN', alpha=0.7)

        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved comparison plot to {filepath}")
