"""Configuration management utility."""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager for loading and accessing config values."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Dot-separated key path (e.g., 'data.num_sentences_per_category')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self._config.get('data', {})

    def get_vectorization_config(self) -> Dict[str, Any]:
        """Get vectorization configuration section."""
        return self._config.get('vectorization', {})

    def get_normalization_config(self) -> Dict[str, Any]:
        """Get normalization configuration section."""
        return self._config.get('normalization', {})

    def get_kmeans_config(self) -> Dict[str, Any]:
        """Get K-means configuration section."""
        return self._config.get('kmeans', {})

    def get_knn_config(self) -> Dict[str, Any]:
        """Get KNN configuration section."""
        return self._config.get('knn', {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self._config.get('visualization', {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration section."""
        return self._config.get('paths', {})

    @property
    def num_sentences_per_category(self) -> int:
        """Get number of sentences per category."""
        return self.get('data.num_sentences_per_category', 100)

    @property
    def word_range(self) -> tuple:
        """Get word range for sentence generation."""
        wr = self.get('data.word_range', [10, 15])
        return tuple(wr)

    @property
    def categories(self) -> list:
        """Get categories configuration."""
        return self.get('data.categories', [])
