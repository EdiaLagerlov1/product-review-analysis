"""Vector normalization utilities."""
import numpy as np
from sklearn.preprocessing import normalize, MinMaxScaler
from src.utils.logger import LoggerMixin


class VectorNormalizer(LoggerMixin):
    """Normalize vector representations."""

    def __init__(self, method: str = 'l2'):
        """Initialize normalizer.

        Args:
            method: Normalization method ('l2' or 'minmax')
        """
        self.method = method
        self.scaler = None

        if method == 'minmax':
            self.scaler = MinMaxScaler()

    def fit(self, vectors: np.ndarray):
        """Fit normalizer on vectors.

        Args:
            vectors: Vector matrix
        """
        if self.method == 'minmax':
            self.logger.info(f"Fitting MinMax scaler on vectors with shape {vectors.shape}")
            self.scaler.fit(vectors)
        else:
            self.logger.info("L2 normalization does not require fitting")

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors.

        Args:
            vectors: Vector matrix

        Returns:
            Normalized vectors
        """
        self.logger.info(f"Normalizing vectors using {self.method} method")

        if self.method == 'l2':
            normalized = normalize(vectors, norm='l2')
        elif self.method == 'minmax':
            normalized = self.scaler.transform(vectors)
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        self.logger.info(f"Normalized vectors to shape {normalized.shape}")
        return normalized

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit and normalize vectors.

        Args:
            vectors: Vector matrix

        Returns:
            Normalized vectors
        """
        self.logger.info(f"Fit-transforming vectors using {self.method} method")

        if self.method == 'l2':
            normalized = normalize(vectors, norm='l2')
        elif self.method == 'minmax':
            normalized = self.scaler.fit_transform(vectors)
        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        self.logger.info(f"Normalized vectors to shape {normalized.shape}")
        return normalized
