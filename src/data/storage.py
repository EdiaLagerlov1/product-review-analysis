"""Data storage utilities for saving and loading data."""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import LoggerMixin


class DataStorage(LoggerMixin):
    """Handle data persistence for sentences and vectors."""

    def __init__(self, base_dir: str = "outputs/data"):
        """Initialize data storage.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_sentences(self, df: pd.DataFrame, filename: str = "sentences.csv"):
        """Save sentences to CSV file.

        Args:
            df: DataFrame with sentences
            filename: Output filename
        """
        filepath = self.base_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved sentences to {filepath}")

    def load_sentences(self, filename: str = "sentences.csv") -> pd.DataFrame:
        """Load sentences from CSV file.

        Args:
            filename: Input filename

        Returns:
            DataFrame with sentences
        """
        filepath = self.base_dir / filename
        df = pd.read_csv(filepath)
        self.logger.info(f"Loaded sentences from {filepath}")
        return df

    def save_vectors(self, vectors: np.ndarray, filename: str = "vectors.npy"):
        """Save vectors to numpy file.

        Args:
            vectors: Vector array
            filename: Output filename
        """
        filepath = self.base_dir / filename
        np.save(filepath, vectors)
        self.logger.info(f"Saved vectors to {filepath}")

    def load_vectors(self, filename: str = "vectors.npy") -> np.ndarray:
        """Load vectors from numpy file.

        Args:
            filename: Input filename

        Returns:
            Vector array
        """
        filepath = self.base_dir / filename
        vectors = np.load(filepath)
        self.logger.info(f"Loaded vectors from {filepath}")
        return vectors

    def save_json(self, data: Dict[Any, Any], filename: str):
        """Save dictionary to JSON file.

        Args:
            data: Dictionary to save
            filename: Output filename
        """
        filepath = self.base_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Saved JSON to {filepath}")

    def load_json(self, filename: str) -> Dict[Any, Any]:
        """Load dictionary from JSON file."""
        filepath = self.base_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.logger.info(f"Loaded JSON from {filepath}")
        return data
