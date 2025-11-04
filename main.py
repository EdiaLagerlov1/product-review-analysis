"""Main entry point for product review analysis."""
import argparse
import sys
from pathlib import Path
from pipeline import Pipeline
from src.utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Product Review Analysis using K-means and KNN'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--num-sentences',
        type=int,
        help='Number of sentences per category (overrides config)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    logger = setup_logger()

    try:
        logger.info("Initializing Product Review Analysis")

        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)

        pipeline = Pipeline(config_path=args.config)

        if args.num_sentences:
            pipeline.config._config['data']['num_sentences_per_category'] = args.num_sentences
            logger.info(f"Overriding config: {args.num_sentences} sentences per category")

        logger.info(f"Output directory: {pipeline.run_dir}")
        cluster_results, knn_results = pipeline.run()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nK-means Clustering Metrics:")
        print(f"  Adjusted Rand Index: {cluster_results['metrics']['adjusted_rand_index']:.4f}")
        print(f"  Normalized Mutual Info: {cluster_results['metrics']['normalized_mutual_info']:.4f}")

        print("\nKNN Classification Metrics:")
        print(f"  Accuracy: {knn_results['overall_metrics']['accuracy']:.4f}")
        print(f"  Precision: {knn_results['overall_metrics']['precision']:.4f}")
        print(f"  Recall: {knn_results['overall_metrics']['recall']:.4f}")
        print(f"  F1 Score: {knn_results['overall_metrics']['f1_score']:.4f}")

        if 'cross_validation' in knn_results:
            cv = knn_results['cross_validation']
            print(f"\nCross-Validation (5-fold):")
            print(f"  Mean Accuracy: {cv['mean_score']:.4f} (+/- {cv['std_score']:.4f})")

        print(f"\nOutputs saved to: {pipeline.run_dir}/")
        print(f"  Data: {pipeline.data_dir}/")
        print(f"  Plots: {pipeline.plots_dir}/")
        print(f"  Reports: {pipeline.reports_dir}/")
        print(f"  Logs: {pipeline.logs_dir}/")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
