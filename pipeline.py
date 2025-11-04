"""Main pipeline orchestrator for product review analysis."""
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data.generator import SentenceGenerator
from src.data.validator import DataValidator
from src.data.storage import DataStorage
from src.preprocessing.vectorizer import TextVectorizer
from src.preprocessing.normalizer import VectorNormalizer
from src.ml.kmeans_model import KMeansModel
from src.ml.knn_model import KNNModel
from src.ml.synthetic_generator import SyntheticGenerator
from src.analysis.cluster_analyzer import ClusterAnalyzer
from src.analysis.deviation_analyzer import DeviationAnalyzer
from src.analysis.knn_analyzer import KNNAnalyzer
from src.visualization.cluster_plotter import ClusterPlotter
from src.visualization.comparison_plotter import ComparisonPlotter
from src.visualization.knn_plotter import KNNPlotter
from src.visualization.metrics_plotter import MetricsPlotter


class Pipeline:
    """Main pipeline for product review analysis."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize pipeline."""
        self.config = Config(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f"outputs/run_{timestamp}")
        self.data_dir = self.run_dir / "data"
        self.plots_dir = self.run_dir / "plots"
        self.reports_dir = self.run_dir / "reports"
        self.logs_dir = self.run_dir / "logs"
        for dir_path in [self.data_dir, self.plots_dir, self.reports_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(log_file=str(self.logs_dir / "pipeline.log"))
        self.storage = DataStorage(str(self.data_dir))
        self._setup_components()

    def _setup_components(self):
        """Setup all pipeline components."""
        self.generator, self.validator = SentenceGenerator(self.config.word_range), DataValidator(self.config.word_range)
        vec_cfg = self.config.get('vectorization')
        self.vectorizer = TextVectorizer(vec_cfg.get('method'), vec_cfg.get('max_features', 1000), vec_cfg.get('vector_size', 100))
        self.normalizer = VectorNormalizer(self.config.get('normalization.method'))
        self.kmeans, self.knn = KMeansModel(**self.config.get_kmeans_config()), KNNModel(**self.config.get_knn_config())
        self.synthetic_gen = SyntheticGenerator()
        self.cluster_analyzer, self.deviation_analyzer, self.knn_analyzer = ClusterAnalyzer(), DeviationAnalyzer(), KNNAnalyzer()
        viz_cfg = self.config.get_visualization_config()
        viz_args = (str(self.plots_dir), tuple(viz_cfg['figure_size']))
        self.cluster_plotter, self.comparison_plotter = ClusterPlotter(*viz_args), ComparisonPlotter(*viz_args)
        self.knn_plotter, self.metrics_plotter = KNNPlotter(*viz_args), MetricsPlotter(*viz_args)

    def run(self):
        """Execute full pipeline."""
        self.logger.info("=" * 60 + "\nStarting Product Review Analysis Pipeline\n" + "=" * 60)
        df_original = self.generator.generate_sentences(self.config.num_sentences_per_category)
        is_valid, errors = self.validator.validate_all(df_original)
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")
        self.storage.save_sentences(df_original, "original_sentences.csv")
        vectors = self.vectorizer.fit_transform(df_original['sentence'].tolist())
        vectors_normalized = self.normalizer.fit_transform(vectors)
        self.storage.save_vectors(vectors_normalized, "original_vectors.npy")
        labels_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        true_labels = df_original['category'].map(labels_map).values
        cluster_labels = self.kmeans.fit_predict(vectors_normalized)
        cluster_results = self.cluster_analyzer.analyze(vectors_normalized, true_labels, cluster_labels)
        cluster_labels_relabeled = cluster_results['relabeled_predictions']
        deviation_results = self.deviation_analyzer.analyze_deviation(
            vectors_normalized, cluster_labels_relabeled, true_labels)
        dim_method = self.config.get('visualization.dimensionality_reduction')
        self.cluster_plotter.plot_clusters_2d(vectors_normalized, cluster_labels_relabeled, method=dim_method,
                                             title='K-means Clusters', filename='01_kmeans_clusters.png')
        self.comparison_plotter.plot_side_by_side(vectors_normalized, true_labels, cluster_labels_relabeled, filename='02_original_vs_predicted.png')
        self.metrics_plotter.plot_confusion_matrix(np.array(cluster_results['confusion_matrix']), labels=['Neg', 'Neu', 'Pos'],
                                                   title='K-means Confusion Matrix', filename='03_confusion_matrix_kmeans.png')
        self.synthetic_gen.analyze_clusters(df_original, cluster_labels, true_labels)
        df_synthetic = self.synthetic_gen.generate_synthetic_sentences(
            self.config.num_sentences_per_category, df_original)
        self.storage.save_sentences(df_synthetic, "synthetic_sentences.csv")
        self.knn.fit(vectors_normalized, true_labels)
        knn_predictions = self.knn.predict(vectors_normalized)
        knn_results = self.knn_analyzer.analyze(true_labels, knn_predictions)
        self.knn_analyzer.cross_validate(self.knn, vectors_normalized, true_labels)
        self.knn_plotter.plot_knn_results(vectors_normalized, true_labels, knn_predictions,
                                         filename='05_knn_classification.png')
        self.metrics_plotter.plot_confusion_matrix(np.array(knn_results['confusion_matrix']),
                                                   labels=['Neg', 'Neu', 'Pos'], title='KNN Confusion Matrix',
                                                   filename='06_confusion_matrix_knn.png')
        self._save_reports(cluster_results, deviation_results, knn_results)
        self.logger.info("Pipeline completed successfully!")
        return cluster_results, knn_results

    def _save_reports(self, cluster_results, deviation_results, knn_results):
        """Save analysis reports."""
        # Remove non-serializable items before saving
        cluster_results_json = {k: v for k, v in cluster_results.items() if k != 'relabeled_predictions'}
        with open(self.reports_dir / 'kmeans_analysis.json', 'w') as f:
            json.dump(cluster_results_json, f, indent=2)
        with open(self.reports_dir / 'deviation_analysis.json', 'w') as f:
            json.dump(deviation_results, f, indent=2)
        with open(self.reports_dir / 'knn_analysis.json', 'w') as f:
            json.dump(knn_results, f, indent=2)
        self.logger.info("Reports saved successfully")
