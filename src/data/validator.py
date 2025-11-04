"""Data validator for sentences."""
import pandas as pd
from typing import Tuple, List
from src.utils.logger import LoggerMixin


class DataValidator(LoggerMixin):
    """Validate generated sentence data."""

    def __init__(self, word_range: Tuple[int, int] = (10, 15)):
        """Initialize data validator.

        Args:
            word_range: Tuple of (min_words, max_words)
        """
        self.word_range = word_range

    def validate_word_length(self, sentence: str) -> bool:
        """Validate sentence word count.

        Args:
            sentence: Sentence to validate

        Returns:
            True if valid, False otherwise
        """
        word_count = len(sentence.split())
        min_words, max_words = self.word_range
        return min_words <= word_count <= max_words

    def validate_distribution(self, df: pd.DataFrame) -> bool:
        """Validate equal distribution across categories.

        Args:
            df: DataFrame with sentences

        Returns:
            True if valid, False otherwise
        """
        counts = df['category'].value_counts()
        return len(counts.unique()) == 1

    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required = ['sentence', 'category', 'rating']
        return all(col in df.columns for col in required)

    def validate_no_empty_sentences(self, df: pd.DataFrame) -> bool:
        """Validate no empty sentences.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        return not df['sentence'].isna().any() and (df['sentence'].str.len() > 0).all()

    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Run all validations.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.validate_required_columns(df):
            errors.append("Missing required columns")

        if not self.validate_no_empty_sentences(df):
            errors.append("Contains empty sentences")

        if not self.validate_distribution(df):
            errors.append("Unequal distribution across categories")

        for idx, row in df.iterrows():
            if not self.validate_word_length(row['sentence']):
                errors.append(f"Row {idx}: Invalid word count")

        is_valid = len(errors) == 0

        if is_valid:
            self.logger.info("All validations passed")
        else:
            self.logger.warning(f"Validation failed: {errors}")

        return is_valid, errors
