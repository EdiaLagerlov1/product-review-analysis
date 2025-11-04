"""Text vectorization utilities."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from src.utils.logger import LoggerMixin


class TextVectorizer(LoggerMixin):
    """Convert text sentences to numerical vectors."""

    def __init__(self, method: str = 'tfidf', max_features: int = 1000, vector_size: int = 100):
        """Initialize vectorizer.

        Args:
            method: Vectorization method ('tfidf', 'word2vec')
            max_features: Maximum number of features (for tfidf)
            vector_size: Vector size for word2vec
        """
        self.method = method
        self.max_features = max_features
        self.vector_size = vector_size
        self.vectorizer = None
        self.w2v_model = None
        self._initialize_vectorizer()

    def _initialize_vectorizer(self):
        """Initialize the appropriate vectorizer."""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=1,
                stop_words='english'
            )
        elif self.method == 'word2vec':
            from gensim.models import Word2Vec
            self.w2v_model = None  # Will be trained in fit()
        else:
            raise ValueError(f"Unsupported vectorization method: {self.method}")

    def fit(self, sentences: List[str]):
        """Fit vectorizer on sentences."""
        self.logger.info(f"Fitting {self.method} vectorizer on {len(sentences)} sentences")
        if self.method == 'tfidf':
            self.vectorizer.fit(sentences)
            self.logger.info(f"Vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
        elif self.method == 'word2vec':
            from gensim.models import Word2Vec
            tokenized = [sent.lower().split() for sent in sentences]
            self.w2v_model = Word2Vec(tokenized, vector_size=self.vector_size, window=5, min_count=1, workers=4)
            self.logger.info(f"Word2Vec trained with {len(self.w2v_model.wv)} words, {self.vector_size}D vectors")

    def transform(self, sentences: List[str]) -> np.ndarray:
        """Transform sentences to vectors."""
        self.logger.info(f"Transforming {len(sentences)} sentences to vectors")
        if self.method == 'tfidf':
            vectors = self.vectorizer.transform(sentences).toarray()
        elif self.method == 'word2vec':
            vectors = np.array([self._sentence_to_vec(sent) for sent in sentences])
        self.logger.info(f"Generated vectors with shape {vectors.shape}")
        return vectors

    def fit_transform(self, sentences: List[str]) -> np.ndarray:
        """Fit and transform sentences."""
        self.logger.info(f"Fit-transforming {len(sentences)} sentences")
        if self.method == 'tfidf':
            vectors = self.vectorizer.fit_transform(sentences).toarray()
        elif self.method == 'word2vec':
            self.fit(sentences)
            vectors = self.transform(sentences)
        self.logger.info(f"Generated vectors with shape {vectors.shape}")
        return vectors

    def _sentence_to_vec(self, sentence: str) -> np.ndarray:
        """Convert sentence to Word2Vec vector by averaging word vectors."""
        words = sentence.lower().split()
        word_vecs = [self.w2v_model.wv[word] for word in words if word in self.w2v_model.wv]
        if len(word_vecs) == 0:
            return np.zeros(self.vector_size)
        return np.mean(word_vecs, axis=0)

    def get_feature_names(self) -> List[str]:
        """Get feature names.

        Returns:
            List of feature names
        """
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out().tolist()
        return []
