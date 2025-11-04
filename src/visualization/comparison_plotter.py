"""Comparison visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from src.utils.logger import LoggerMixin


class ComparisonPlotter(LoggerMixin):
    """Compare predictions vs ground truth."""

    def __init__(self, output_dir: str = "outputs/plots", figsize: tuple = (12, 8)):
        """Initialize comparison plotter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

    def _reduce_dimensions(self, X: np.ndarray, method: str):
        """Reduce dimensions for visualization."""
        if method == 'tsne':
            return TSNE(n_components=2, random_state=42).fit_transform(X)
        return PCA(n_components=2, random_state=42).fit_transform(X)

    def plot_side_by_side(self, X: np.ndarray, y_true: np.ndarray,
                         y_pred: np.ndarray, method: str = 'pca',
                         filename: str = 'comparison.png'):
        """Plot true vs predicted labels side by side."""
        self.logger.info("Plotting side-by-side comparison")
        X_reduced = self._reduce_dimensions(X, method)

        colors = ['red', 'green', 'blue']
        labels_text = ['Negative', 'Neutral', 'Positive']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] // 1.5))

        for i in np.unique(y_true):
            mask = y_true == i
            ax1.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=colors[int(i)], label=labels_text[int(i)], alpha=0.8, s=15, edgecolors='black', linewidths=0.3)
        ax1.set_title('Ground Truth Labels')
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for i in np.unique(y_pred):
            mask = y_pred == i
            ax2.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=colors[int(i)], label=f'Cluster {int(i)}', alpha=0.8, s=15, edgecolors='black', linewidths=0.3)
        ax2.set_title('Predicted Cluster Labels')
        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved comparison plot to {filepath}")

    def plot_agreement_map(self, X: np.ndarray, y_true: np.ndarray,
                          y_pred: np.ndarray, method: str = 'tsne',
                          filename: str = 'agreement_map.png'):
        """Plot agreement/disagreement map."""
        self.logger.info("Plotting agreement map")
        X_reduced = self._reduce_dimensions(X, method)
        agreement = (y_true == y_pred).astype(int)

        plt.figure(figsize=self.figsize)
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                            c=agreement, cmap='RdYlGn', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Agreement (0=Mismatch, 1=Match)', ticks=[0, 1])
        plt.title('Prediction Agreement Map')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved agreement map to {filepath}")
