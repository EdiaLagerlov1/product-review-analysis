# Product Review Analysis - PRD

## Project Overview
A machine learning system demonstrating **unsupervised clustering (K-means)** vs **supervised classification (KNN)** for sentiment analysis using **Word2Vec embeddings**, **PCA visualization**, and comprehensive performance comparison.

## Technical Requirements
- **Maximum file length**: 100 lines per file (strict limit)
- **Single Responsibility**: Each file/class/function does ONE thing
- **No duplicate code**: Extract common logic into reusable functions
- **Separation of Concerns**: Separate data, business logic, and presentation
- **Timestamped Outputs**: Each run saved in unique directory

## System Architecture

### 1. Core Components (src/)

#### 1.1 Data Layer (`src/data/`)
- **`generator.py`** (95 lines)
  - Generate X sentences (word-combination based, no templates)
  - Uses 5 word groups per sentiment (25 total)
  - Random combinations: 3-5 groups per sentence
  - Random connectors: "with", "and", "but", "while"
  - Random intensifiers: "very", "extremely", "quite"
  - Categories: Positive (5★), Neutral (3★), Negative (1★)
  - Equal distribution across categories
  - **100% unique sentences** (no template-based clustering artifacts)

- **`validator.py`** (97 lines)
  - Validate sentence length (10-15 words)
  - Validate equal distribution
  - Validate data quality

- **`storage.py`** (93 lines)
  - Save/load sentences to CSV
  - Save vectors to NumPy (.npy files)
  - Manage data persistence

#### 1.2 Preprocessing Layer (`src/preprocessing/`)
- **`vectorizer.py`** (90 lines)
  - Convert sentences to vectors using **Word2Vec** or TF-IDF
  - Default: Word2Vec (100-dimensional semantic vectors)
  - Trains Word2Vec model on generated sentences
  - Sentence embedding: Average of word vectors
  - Single vectorization strategy per instance

- **`normalizer.py`** (74 lines)
  - Normalize vectors using **L2 normalization** (unit length)
  - Preserves negative values (not 0-1 scaling)
  - Single normalization method per instance

#### 1.3 Machine Learning Layer (`src/ml/`)
- **`kmeans_model.py`** (88 lines)
  - Execute K-means clustering (k=3)
  - Fit and predict methods
  - Return cluster assignments
  - Works on 100D Word2Vec vectors

- **`knn_model.py`** (86 lines)
  - KNN classification on original sentences
  - Fit and predict methods
  - Default k=5 neighbors
  - Euclidean distance in 100D space

- **`synthetic_generator.py`** (85 lines)
  - Generate new sentences based on cluster characteristics
  - Equal distribution across 3 categories
  - Based on cluster center analysis

#### 1.4 Analysis Layer (`src/analysis/`)
- **`cluster_analyzer.py`** (99 lines)
  - Compare K-means results to ground truth labels
  - Calculate metrics: ARI, NMI
  - Generate confusion matrix
  - Cluster distribution analysis

- **`deviation_analyzer.py`** (43 lines)
  - Analyze cluster deviation from ground truth
  - Calculate mismatch rate
  - Track clustering errors
  - **No silhouette analysis** (removed)

- **`knn_analyzer.py`** (77 lines)
  - Analyze KNN performance
  - Calculate classification metrics
  - 5-fold cross-validation results
  - Per-class precision/recall/F1

#### 1.5 Visualization Layer (`src/visualization/`)
- **`cluster_plotter.py`** (98 lines)
  - Visualize K-means clusters using **PCA** (2D projection)
  - Discrete colors: Red (Cluster 0), Green (Cluster 1), Blue (Cluster 2)
  - Small dots (s=15) with black edges
  - Single plot type per function

- **`comparison_plotter.py`** (77 lines)
  - Compare ground truth labels vs K-means predictions
  - Side-by-side visualization
  - Same PCA projection for both panels

- **`knn_plotter.py`** (82 lines)
  - Visualize KNN classification results
  - Plot true labels vs predictions
  - Same PCA projection

- **`metrics_plotter.py`** (95 lines)
  - Plot confusion matrices (heatmaps)
  - Annotated with counts
  - Blue colormap

#### 1.6 Utilities Layer (`src/utils/`)
- **`metrics.py`** (89 lines)
  - Common metric calculations
  - ARI, NMI calculation
  - Confusion matrix generation
  - **No silhouette score** (removed)

- **`logger.py`** (70 lines)
  - Logging configuration
  - Structured logging to file and console

- **`config.py`** (87 lines)
  - YAML configuration management
  - Constants and parameters

### 2. Orchestration Layer

#### **`pipeline.py`** (97 lines)
Main pipeline orchestrator that:
1. Generate initial sentences (word-combination based)
2. Vectorize using Word2Vec (100D)
3. Normalize using L2
4. Run K-means clustering
5. Analyze K-means performance (ARI, NMI, confusion matrix)
6. Generate synthetic sentences based on clusters
7. Run KNN on original sentences
8. Analyze KNN performance (accuracy, precision, recall, CV)
9. Generate all visualizations (5 plots)
10. Export results to timestamped directory
11. **No silhouette analysis** (removed)

### 3. Entry Points

#### **`main.py`** (89 lines)
- CLI interface with argparse
- Parameter configuration (num-sentences)
- Execute pipeline
- Display summary results
- Print timestamped output directory path
- **No silhouette score print** (removed)

### 4. Configuration Files

#### **`config.yaml`**
```yaml
data:
  num_sentences_per_category: 200  # X parameter (default: 200)
  word_range: [10, 15]
  categories:
    - name: "Positive"
      rating: 5
    - name: "Neutral"
      rating: 3
    - name: "Negative"
      rating: 1

vectorization:
  method: "word2vec"  # word2vec (recommended), tfidf (not recommended)
  max_features: 1000  # For TF-IDF if used
  vector_size: 100    # Word2Vec embedding dimensions (50, 100, 200)

normalization:
  method: "l2"  # L2 unit-length normalization

kmeans:
  n_clusters: 3
  random_state: 42
  max_iter: 300

knn:
  n_neighbors: 5
  metric: "euclidean"

visualization:
  dimensionality_reduction: "pca"  # PCA (recommended), tsne (not recommended)
  figure_size: [12, 8]
  save_plots: true
```

## Workflow

### Phase 1: Data Generation & Vectorization
1. Generate 3X sentences (X per category)
2. Use word-combination method (5 groups × 3-5 selections)
3. Vectorize using Word2Vec (train on 3X sentences)
4. Normalize to unit length (L2)

### Phase 2: K-means Clustering (Unsupervised)
1. Apply K-means (k=3) on 100D normalized vectors
2. Analyze deviation from ground truth
3. Calculate ARI, NMI, confusion matrix
4. Visualize clusters using PCA

### Phase 3: Synthetic Data Generation
1. Based on K-means cluster centers
2. Generate 3X new sentences (X per category)
3. Vectorize and normalize

### Phase 4: KNN Classification (Supervised)
1. Use original sentences as training data
2. Classify using KNN (k=5)
3. Analyze performance (accuracy, precision, recall)
4. 5-fold cross-validation
5. Visualize results

### Phase 5: Comprehensive Analysis
1. Compare K-means vs KNN performance
2. Generate all visualizations
3. Export analysis reports to timestamped directory

## Output Structure

Each run creates a timestamped directory:

```
outputs/
└── run_YYYYMMDD_HHMMSS/          # Example: run_20251104_000509
    ├── data/
    │   ├── original_sentences.csv      # 3X unique sentences
    │   ├── synthetic_sentences.csv     # 3X synthetic sentences
    │   └── original_vectors.npy        # 3X × 100 Word2Vec matrix
    ├── plots/                          # 300 DPI PNG visualizations
    │   ├── 01_kmeans_clusters.png           # K-means PCA projection
    │   ├── 02_original_vs_predicted.png     # True vs predicted
    │   ├── 03_confusion_matrix_kmeans.png   # K-means errors
    │   ├── 05_knn_classification.png        # KNN results
    │   └── 06_confusion_matrix_knn.png      # KNN errors
    ├── reports/                        # JSON analysis results
    │   ├── kmeans_analysis.json       # ARI, NMI, confusion matrix
    │   ├── deviation_analysis.json    # Mismatch rate
    │   └── knn_analysis.json          # Accuracy, precision, recall, CV
    └── logs/
        └── pipeline.log               # Timestamped log messages
```

## Key Metrics to Track

### K-means Analysis (Unsupervised)
- **Adjusted Rand Index (ARI)**: 0.5465 (moderate clustering quality)
- **Normalized Mutual Information (NMI)**: 0.5820 (moderate information gain)
- **Mismatch Rate**: 89.5% (misleading - doesn't account for label permutations)
- **Accuracy after relabeling**: 82.8% (497/600 correct)
- **Inertia**: 0.0704 (very tight clusters)
- **Confusion matrix**: Shows cluster-to-label mapping

### KNN Analysis (Supervised)
- **Training Accuracy**: 98.67% (592/600 correct)
- **Precision**: 98.69% (very few false positives)
- **Recall**: 98.67% (finds almost all instances)
- **F1 Score**: 98.66% (balanced performance)
- **Cross-validation**: 95.67% ± 1.62% (5-fold, stable)
- **Confusion matrix**: 8 total errors

### Performance Comparison
- **K-means**: 82.8% accuracy (no labels needed)
- **KNN**: 98.67% accuracy (requires labeled training data)
- **Improvement**: +15.9% accuracy with supervised learning

## Key Performance Results

### With Word2Vec (Current)
- **K-means ARI**: 0.5465 (moderate-to-good clustering)
- **KNN Accuracy**: 98.67% (near-perfect classification)
- **Vector Dimensions**: 100 (dense, semantic)
- **Sentence Diversity**: 100% unique (word-combination based)

### With TF-IDF (Not Recommended)
- **K-means ARI**: 0.02 (almost random - 27× worse)
- **KNN Accuracy**: 100% (unrealistic - overfitting to templates)
- **Vector Dimensions**: 804 (sparse, high-dimensional)

**Conclusion**: Word2Vec dramatically improves K-means clustering while maintaining realistic KNN performance.

## Dependencies

```
numpy>=1.24.0          # Numerical computations
pandas>=2.0.0          # Data manipulation
scikit-learn>=1.3.0    # ML algorithms (K-means, KNN, PCA)
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical visualizations
PyYAML>=6.0            # Configuration parsing
nltk>=3.8              # Text processing
gensim>=4.3.0          # Word2Vec embeddings
```

## Success Criteria

1. ✅ All files < 100 lines
2. ✅ Clear separation of concerns (6 layers)
3. ✅ No code duplication
4. ✅ Comprehensive visualization (5 plots per run)
5. ✅ Detailed performance comparison (K-means vs KNN)
6. ✅ Reproducible results (random_state=42)
7. ✅ Timestamped outputs (no overwriting)
8. ✅ Word2Vec embeddings (semantic understanding)
9. ✅ Diverse sentence generation (100% unique)
10. ✅ Realistic performance metrics (98.67% KNN, 82.8% K-means)

## Design Decisions

### Why Word2Vec over TF-IDF?
- **Semantic understanding**: "good" and "great" have similar vectors
- **Fixed dimensions**: 100D regardless of vocabulary size
- **Dense vectors**: All values non-zero (better for K-means)
- **27× better K-means ARI**: 0.55 vs 0.02

### Why PCA over t-SNE?
- **Linear**: Preserves global structure
- **Fast**: ~1 second vs ~5 seconds for t-SNE
- **Deterministic**: Same results every run
- **No artificial clusters**: t-SNE creates misleading tight clusters

### Why word-combination generation?
- **Diversity**: 100% unique sentences (no template artifacts)
- **Realistic**: Better KNN performance (98.67% vs 100%)
- **No clustering artifacts**: Eliminated "groups of 5" pattern
- **Smooth distributions**: Natural-looking PCA projections

### Why no silhouette analysis?
- **Removed**: Not needed for comparative analysis
- **Simplified codebase**: Fewer dependencies and calculations
- **ARI/NMI sufficient**: Better metrics for cluster quality

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings (200 sentences per category)
python main.py

# Run with custom size (100 sentences per category)
python main.py --num-sentences 100

# Enable verbose logging
python main.py --verbose
```

## Expected Output

```
Product Review Analysis System
==============================

Generating sentences...
  Generated 600 sentences (200 per category)

Vectorizing sentences using word2vec...
  Created 600 vectors with 100 features

Running K-means clustering...
  Adjusted Rand Index: 0.5465
  Normalized Mutual Info: 0.5820

Running KNN classification...
  KNN Accuracy: 98.67%
  Cross-validation Score: 95.67% (±1.62%)

Results saved to: outputs/run_20251104_000509/
```

## Future Enhancements (Out of Scope)

- BERT embeddings (more semantic depth)
- DBSCAN clustering (non-spherical clusters)
- Real product review dataset (Amazon, Yelp)
- Web interface for visualization
- Interactive plots (Plotly)
- Hyperparameter tuning (GridSearchCV)
