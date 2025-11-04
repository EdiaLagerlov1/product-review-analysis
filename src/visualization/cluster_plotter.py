"""Cluster visualization utilities."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from src.utils.logger import LoggerMixin


class ClusterPlotter(LoggerMixin):
    """Visualize clustering results."""

    def __init__(self, output_dir: str = "outputs/plots", figsize: tuple = (12, 8)):
        """Initialize cluster plotter.

        Args:
            output_dir: Output directory for plots
            figsize: Figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

    def plot_clusters_2d(self, X: np.ndarray, labels: np.ndarray,
                        method: str = 'tsne', title: str = 'Clusters',
                        filename: str = 'clusters_2d.png'):
        """Plot clusters in 2D.

        Args:
            X: Feature matrix
            labels: Cluster labels
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            filename: Output filename
        """
        self.logger.info(f"Plotting 2D clusters using {method}")

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")

        X_reduced = reducer.fit_transform(X)

        colors = ['red', 'green', 'blue']
        plt.figure(figsize=self.figsize)
        for i in np.unique(labels):
            mask = labels == i
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                       c=colors[int(i)], label=f'Cluster {int(i)}', alpha=0.8, s=15, edgecolors='black', linewidths=0.3)
        plt.legend()
        plt.title(title)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot to {filepath}")

    def plot_cluster_centers(self, centers: np.ndarray, method: str = 'pca',
                            filename: str = 'cluster_centers.png'):
        """Plot cluster centers.

        Args:
            centers: Cluster centers
            method: Dimensionality reduction method
            filename: Output filename
        """
        self.logger.info("Plotting cluster centers")

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42)

        centers_reduced = reducer.fit_transform(centers)

        plt.figure(figsize=self.figsize)
        plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidths=2)

        for i, (x, y) in enumerate(centers_reduced):
            plt.annotate(f'C{i}', (x, y), fontsize=12, ha='center', va='center')

        plt.title('Cluster Centers')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Saved plot to {filepath}")
