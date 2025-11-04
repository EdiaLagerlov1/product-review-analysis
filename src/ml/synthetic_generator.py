"""Synthetic sentence generator based on cluster analysis."""
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.utils.logger import LoggerMixin


class SyntheticGenerator(LoggerMixin):
    """Generate synthetic sentences based on cluster deviation."""

    def __init__(self):
        """Initialize synthetic generator."""
        self.cluster_characteristics = {}

    def analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                        true_labels: np.ndarray):
        """Analyze cluster characteristics.

        Args:
            df: Original dataframe with sentences
            cluster_labels: Predicted cluster labels
            true_labels: True category labels
        """
        self.logger.info("Analyzing cluster characteristics")

        for cluster in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster
            cluster_sentences = df[cluster_mask]

            true_categories = true_labels[cluster_mask]
            dominant_category = pd.Series(true_categories).mode()[0]

            self.cluster_characteristics[int(cluster)] = {
                'dominant_category': int(dominant_category),
                'size': int(np.sum(cluster_mask)),
                'purity': float(np.sum(true_categories == dominant_category) / len(true_categories))
            }

        self.logger.info(f"Analyzed {len(self.cluster_characteristics)} clusters")

    def generate_synthetic_sentences(self, num_per_category: int,
                                    original_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic sentences.

        Args:
            num_per_category: Number of sentences per category
            original_df: Original sentences dataframe

        Returns:
            DataFrame with synthetic sentences
        """
        self.logger.info(f"Generating {num_per_category} synthetic sentences per category")

        synthetic_data = []
        categories = {0: ('Negative', 1), 1: ('Neutral', 3), 2: ('Positive', 5)}

        for cat_label, (cat_name, rating) in categories.items():
            category_sentences = original_df[original_df['rating'] == rating]['sentence'].tolist()

            for _ in range(num_per_category):
                base_sentence = random.choice(category_sentences)
                words = base_sentence.split()

                if len(words) > 5:
                    idx1, idx2 = random.sample(range(len(words)), 2)
                    words[idx1], words[idx2] = words[idx2], words[idx1]

                synthetic_data.append({
                    'sentence': ' '.join(words),
                    'category': cat_name,
                    'rating': rating
                })

        df = pd.DataFrame(synthetic_data)
        self.logger.info(f"Generated {len(df)} synthetic sentences")
        return df

    def get_cluster_summary(self) -> Dict[int, Dict[str, Any]]:
        """Get cluster characteristics summary.

        Returns:
            Cluster characteristics
        """
        return self.cluster_characteristics
