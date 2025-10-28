# cancer_pattern_mining.py

A comprehensive command-line tool for sequential pattern mining analysis of cancer diagnosis data using the Wisconsin Breast Cancer dataset.

## Overview

This script performs end-to-end sequential pattern mining to discover discriminative patterns between malignant and benign breast cancer cases. It transforms continuous biomarker features into meaningful sequential representations and applies advanced pattern mining algorithms to uncover diagnostic insights.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Configuration Options](#configuration-options)
- [Output Files](#output-files)
- [Usage Examples](#usage-examples)
- [Algorithm Overview](#algorithm-overview)
- [Interpretation Guide](#interpretation-guide)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd SC4020-Group-Project-2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Dependencies

The script requires the following packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.2.0
- scipy >= 1.9.0

## Quick Start

Run the analysis with default parameters:

```bash
python scripts/cancer_pattern_mining.py
```

This will:
1. Load the Wisconsin Breast Cancer dataset
2. Generate feature sequences using quantile discretization
3. Discover sequential patterns using GSP algorithm
4. Perform sensitivity analysis across different configurations
5. Save results to the `outputs/` directory

## Command Line Interface

```bash
python scripts/cancer_pattern_mining.py [OPTIONS]
```

### Options

#### Data and Output
- `--data-path PATH`: Path to cancer dataset CSV file (default: `datasets/raw/wisconsin_breast_cancer.csv`)
- `--output-dir DIR`: Directory to save analysis results (default: `outputs`)

#### Sequence Generation
- `--max-sequence-length INT`: Maximum length of generated sequences (default: 5)
- `--max-gap INT`: Maximum gap allowed in sequential patterns (default: 1)
- `--top-k INT`: Number of top features to consider for sequence generation (default: 10)

#### Feature Discretization
- `--discretization-strategy {uniform,quantile,kmeans}`: Strategy for discretizing continuous features (default: quantile)
- `--n-bins INT`: Number of bins for feature discretization (default: 3)
- `--ranking-method {mutual_info,zscore,correlation}`: Method for ranking feature importance (default: mutual_info)

#### Pattern Mining
- `--min-support FLOAT`: Minimum support threshold for pattern mining (default: 0.1)
- `--max-pattern-length INT`: Maximum length of discovered patterns (default: 4)

#### Analysis Options
- `--skip-sensitivity`: Skip sensitivity analysis to speed up execution
- `--verbose`: Enable verbose output
- `--help`: Show help message and exit

## Configuration Options

### Discretization Strategies

#### Quantile (Recommended)
- **Best for**: Datasets with skewed distributions
- **Creates**: Equal-frequency bins based on data distribution
- **Example**: Each bin contains approximately 33% of samples (for 3 bins)

```bash
python scripts/cancer_pattern_mining.py --discretization-strategy quantile --n-bins 3
```

#### Uniform
- **Best for**: Uniformly distributed features
- **Creates**: Equal-width bins across feature range
- **Example**: For range 0-100, creates bins [0-33], [33-67], [67-100]

```bash
python scripts/cancer_pattern_mining.py --discretization-strategy uniform --n-bins 5
```

#### K-means
- **Best for**: Multi-modal distributions
- **Creates**: Bins based on k-means clustering
- **Example**: Natural clusters in feature space

```bash
python scripts/cancer_pattern_mining.py --discretization-strategy kmeans --n-bins 4
```

### Feature Ranking Methods

#### Mutual Information (Default)
- **Global ranking**: Same feature order for all samples
- **Based on**: Information gain relative to diagnosis
- **Best for**: Finding universally important biomarkers

```bash
python scripts/cancer_pattern_mining.py --ranking-method mutual_info --top-k 8
```

#### Z-Score
- **Individual ranking**: Different feature order per sample
- **Based on**: Standardized feature values
- **Best for**: Personalized diagnostic patterns

```bash
python scripts/cancer_pattern_mining.py --ranking-method zscore
```

#### Correlation
- **Global ranking**: Linear relationship with diagnosis
- **Based on**: Pearson correlation coefficient
- **Best for**: Simple linear relationships

```bash
python scripts/cancer_pattern_mining.py --ranking-method correlation --top-k 10
```

### Support Thresholds

- **High Support (0.3-0.5)**: Conservative, few high-confidence patterns
- **Medium Support (0.1-0.3)**: Balanced approach (recommended)
- **Low Support (0.05-0.1)**: Liberal, many patterns including rare ones

## Output Files

The script generates the following output files in the specified output directory:

### `analysis_summary.txt`
Contains high-level analysis results:
```
Cancer Sequential Pattern Mining Analysis Summary
==================================================

Total sequences: 569
Malignant cases: 212
Benign cases: 357
Unique malignant patterns: 15
Unique benign patterns: 12
Discriminative patterns: 8

Top Discriminative Patterns:
------------------------------
1. {radius_worst_high}
   Discriminative for: malignant
   Malignant support: 0.847
   Benign support: 0.156
```

### `feature_importance.txt`
Provides feature-level insights:
```
Feature Importance Analysis
==============================

Top Features in Malignant Patterns:
  radius_worst: 0.847
  perimeter_worst: 0.840
  concave points_worst: 0.840

Top Features in Benign Patterns:
  fractal_dimension_mean: 0.527
  smoothness_mean: 0.510
  symmetry_se: 0.487
```

## Usage Examples

### Basic Analysis
```bash
# Run with default parameters
python scripts/cancer_pattern_mining.py
```

### Custom Configuration
```bash
# Detailed analysis with 5 bins and uniform discretization
python scripts/cancer_pattern_mining.py \
    --discretization-strategy uniform \
    --n-bins 5 \
    --min-support 0.15 \
    --max-pattern-length 3 \
    --verbose
```

### Quick Analysis (Skip Sensitivity)
```bash
# Faster execution without sensitivity analysis
python scripts/cancer_pattern_mining.py \
    --skip-sensitivity \
    --min-support 0.2 \
    --top-k 5
```

### Custom Data Path
```bash
# Use custom dataset and output location
python scripts/cancer_pattern_mining.py \
    --data-path /path/to/custom_cancer_data.csv \
    --output-dir /path/to/results
```

### Personalized Analysis
```bash
# Patient-specific patterns using z-score ranking
python scripts/cancer_pattern_mining.py \
    --ranking-method zscore \
    --max-sequence-length 7 \
    --discretization-strategy quantile
```

## Algorithm Overview

### 1. Data Preprocessing
- **Input**: Wisconsin Breast Cancer dataset with 30 continuous features
- **Cleaning**: Remove NaN values and irrelevant columns
- **Validation**: Ensure data quality and consistency

### 2. Feature Discretization
- **Purpose**: Convert continuous biomarker values to categorical representations
- **Methods**: Uniform, quantile, or k-means binning
- **Output**: Features labeled as 'low', 'medium', 'high' (or 'bin_0', 'bin_1', etc.)

### 3. Feature Ranking
- **Purpose**: Determine importance order for sequence generation
- **Methods**: Mutual information, z-score, or correlation
- **Output**: Ordered list of features by importance

### 4. Sequence Generation
- **Purpose**: Create ordered sequences representing patient biomarker profiles
- **Process**: Combine top-k most important features in ranked order
- **Format**: `[['feature1_high'], ['feature2_medium'], ['feature3_low'], ...]`

### 5. Pattern Mining (GSP Algorithm)
- **Purpose**: Discover frequent sequential patterns
- **Process**: 
  1. Generate candidate patterns
  2. Calculate support in sequence database
  3. Prune patterns below minimum support
  4. Iterate until no new patterns found
- **Output**: Frequent sequential patterns with support values

### 6. Discriminative Analysis
- **Purpose**: Identify patterns that distinguish malignant from benign cases
- **Metrics**: Support difference, lift, statistical significance
- **Output**: Ranked list of discriminative patterns

### 7. Sensitivity Analysis (Optional)
- **Purpose**: Evaluate robustness across different parameter settings
- **Components**: Binning strategy comparison, support threshold analysis, ranking method evaluation
- **Output**: Recommended optimal configuration

## Interpretation Guide

### Pattern Types

#### 1. Unique Patterns
Patterns that appear only in one class (malignant or benign):
```
Pattern: {radius_worst_high}
Type: unique_malignant
Interpretation: High worst-case radius measurements are exclusive to malignant tumors
```

#### 2. Discriminative Patterns
Patterns with significantly different frequencies between classes:
```
Pattern: {texture_mean_medium}
Type: discriminative
Malignant support: 0.156
Benign support: 0.445
Interpretation: Medium texture values are 2.85x more common in benign cases
```

#### 3. Sequential Patterns
Multi-step patterns showing feature relationships:
```
Pattern: {radius_worst_high} → {perimeter_worst_high}
Interpretation: Malignant cases often show high radius followed by high perimeter
```

### Clinical Insights

#### Malignant Indicators
Common patterns in malignant cases typically involve:
- **Size-related features**: `radius_worst_high`, `area_worst_high`, `perimeter_worst_high`
- **Texture irregularities**: `texture_worst_high`, `texture_mean_high`
- **Shape complexity**: `concave_points_worst_high`, `concavity_worst_high`

#### Benign Indicators
Common patterns in benign cases typically involve:
- **Regular measurements**: Features with 'low' or 'medium' values
- **Symmetry preservation**: `symmetry_mean_low`, `fractal_dimension_mean_low`
- **Smooth boundaries**: `smoothness_mean_medium`, `compactness_mean_low`

#### Sequential Relationships
- **Malignant progression**: Size → Texture → Shape complexity
- **Benign characteristics**: Consistent low/medium values across features
- **Diagnostic hierarchy**: Worst-case measurements are most predictive

## Performance Considerations

### Computational Complexity
- **Time complexity**: O(n × m × k) where n=sequences, m=pattern length, k=candidates
- **Space complexity**: O(p) where p=number of patterns found
- **Bottlenecks**: Sensitivity analysis (multiple configurations tested)

### Optimization Strategies

#### For Large Datasets (>10,000 samples)
```bash
# Reduce computational load
python scripts/cancer_pattern_mining.py \
    --min-support 0.2 \
    --max-pattern-length 3 \
    --top-k 5 \
    --skip-sensitivity
```

#### For Detailed Analysis
```bash
# Maximum detail (slower execution)
python scripts/cancer_pattern_mining.py \
    --min-support 0.05 \
    --max-pattern-length 5 \
    --top-k 15 \
    --max-sequence-length 7
```

#### Memory Optimization
- Use higher support thresholds to reduce pattern count
- Limit maximum pattern length
- Process data in smaller batches if needed

### Expected Runtimes
- **Basic analysis**: 30-60 seconds
- **With sensitivity analysis**: 2-5 minutes
- **Detailed configuration**: 5-10 minutes

## Troubleshooting

### Common Issues

#### 1. Memory Errors
**Symptoms**: "MemoryError" or system slowdown
**Solutions**:
```bash
# Increase minimum support
python scripts/cancer_pattern_mining.py --min-support 0.3

# Reduce pattern complexity
python scripts/cancer_pattern_mining.py --max-pattern-length 2 --top-k 5
```

#### 2. No Patterns Found
**Symptoms**: "0 discriminative patterns found"
**Solutions**:
```bash
# Lower support threshold
python scripts/cancer_pattern_mining.py --min-support 0.05

# Increase number of bins
python scripts/cancer_pattern_mining.py --n-bins 5
```

#### 3. Low Pattern Diversity
**Symptoms**: Many similar patterns
**Solutions**:
```bash
# Try different discretization
python scripts/cancer_pattern_mining.py --discretization-strategy uniform

# Use z-score ranking for more variety
python scripts/cancer_pattern_mining.py --ranking-method zscore
```

#### 4. File Not Found Errors
**Symptoms**: "FileNotFoundError: cancer dataset not found"
**Solutions**:
```bash
# Verify data path
ls datasets/raw/wisconsin_breast_cancer.csv

# Use absolute path
python scripts/cancer_pattern_mining.py --data-path /full/path/to/data.csv
```

### Debugging Options

#### Verbose Output
```bash
# Enable detailed logging
python scripts/cancer_pattern_mining.py --verbose
```

#### Validation Mode
```bash
# Quick validation with minimal processing
python scripts/cancer_pattern_mining.py \
    --min-support 0.5 \
    --max-pattern-length 1 \
    --skip-sensitivity \
    --verbose
```

### Getting Help

#### Command Line Help
```bash
python scripts/cancer_pattern_mining.py --help
```

#### Error Reporting
When reporting issues, include:
1. Command line arguments used
2. Error message (full traceback)
3. Dataset characteristics (size, format)
4. System specifications (Python version, memory)

## Advanced Usage

### Integration with Other Tools
```python
# Use as Python module
from scripts.cancer_pattern_mining import load_data, perform_basic_analysis

# Load custom data
features, target = load_data('custom_cancer_data.csv')

# Run analysis programmatically
config = {'min_support': 0.15, 'n_bins': 5}
results = perform_basic_analysis(features, target, config)
```

### Batch Processing
```bash
# Process multiple configurations
for support in 0.1 0.15 0.2; do
    python scripts/cancer_pattern_mining.py \
        --min-support $support \
        --output-dir "outputs_support_$support" \
        --skip-sensitivity
done
```

### Custom Output Processing
```python
# Parse output files
import pandas as pd

# Read analysis summary
with open('outputs/analysis_summary.txt', 'r') as f:
    summary = f.read()

# Extract discriminative patterns for further analysis
# [Custom processing code here]
```

This comprehensive tool provides a complete solution for cancer pattern analysis, from data preprocessing through interpretable results generation. The flexible CLI interface allows for extensive customization while maintaining ease of use for standard analyses.