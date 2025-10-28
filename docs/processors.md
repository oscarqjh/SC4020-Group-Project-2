# Processors Documentation

This document provides comprehensive documentation for the data processing components in the sequential pattern mining system.

## Overview

The processors module contains classes responsible for transforming raw cancer dataset features into sequential representations suitable for pattern mining. The module follows an object-oriented design with a base class and specialized implementations.

## Base Classes

### BasePreprocessor

The foundation class for all preprocessing components.

```python
from src.processors.base import BasePreprocessor
```

#### Description
`BasePreprocessor` provides the core interface and common functionality for all data preprocessing operations. It implements a standardized workflow for fitting, transforming, and validating preprocessing steps.

#### Constructor Parameters
- `config` (Optional[Dict[str, Any]]): Configuration dictionary containing preprocessing parameters

#### Key Methods

##### `fit(data, target=None)`
Fits the preprocessor to the training data.

**Parameters:**
- `data` (pd.DataFrame): Input features for training
- `target` (Optional[pd.Series]): Target variable (if supervised)

**Returns:**
- `Self`: Returns self for method chaining

##### `transform(data)`
Transforms input data using the fitted preprocessor.

**Parameters:**
- `data` (pd.DataFrame): Data to transform

**Returns:**
- `pd.DataFrame`: Transformed data

##### `fit_transform(data, target=None)`
Convenience method that fits and transforms in one step.

**Parameters:**
- `data` (pd.DataFrame): Input data
- `target` (Optional[pd.Series]): Target variable

**Returns:**
- `pd.DataFrame`: Transformed data

##### `validate_fitted()`
Ensures the preprocessor has been fitted before transformation.

**Raises:**
- `ValueError`: If the preprocessor hasn't been fitted

#### Usage Example
```python
# Basic usage pattern for any preprocessor
preprocessor = SomePreprocessor(config={'param': 'value'})
preprocessor.fit(training_data, target)
transformed_data = preprocessor.transform(new_data)

# Or use fit_transform for one-step operation
transformed_data = preprocessor.fit_transform(training_data, target)
```

---

## Sequence Generation Classes

### FeatureDiscretizer

Converts continuous features into categorical bins for sequence generation.

```python
from src.processors.sequence_generator import FeatureDiscretizer
```

#### Description
`FeatureDiscretizer` transforms continuous numerical features into discrete categorical representations using various binning strategies. This is essential for creating meaningful sequential patterns from continuous cancer biomarker data.

#### Constructor Parameters
- `config` (Optional[Dict[str, Any]]): Configuration dictionary with keys:
  - `n_bins` (int, default=3): Number of bins for discretization
  - `strategy` (str, default='quantile'): Binning strategy ('uniform', 'quantile', 'kmeans')
  - `encode` (str, default='ordinal'): Encoding method ('ordinal', 'onehot')

#### Key Methods

##### `fit(data, target=None)`
Learns discretization boundaries from training data.

**Parameters:**
- `data` (pd.DataFrame): Continuous features to discretize
- `target` (Optional[pd.Series]): Target variable (unused)

**Returns:**
- `FeatureDiscretizer`: Self for method chaining

##### `transform(data)`
Transforms continuous features to discrete categorical labels.

**Parameters:**
- `data` (pd.DataFrame): Input features to discretize

**Returns:**
- `pd.DataFrame`: Discretized features with meaningful labels ('low', 'medium', 'high')

##### `transform_numeric(data)`
Transforms continuous features to discrete numeric values (for ranking).

**Parameters:**
- `data` (pd.DataFrame): Input features to discretize

**Returns:**
- `pd.DataFrame`: Discretized features with numeric values (0, 1, 2, ...)

#### Usage Example
```python
# Create discretizer with 5 uniform bins
discretizer = FeatureDiscretizer({
    'n_bins': 5,
    'strategy': 'uniform'
})

# Fit and transform cancer features
discretizer.fit(cancer_features)
discrete_features = discretizer.transform(cancer_features)

# For ranking purposes, use numeric output
numeric_features = discretizer.transform_numeric(cancer_features)
```

#### Binning Strategies

1. **Uniform**: Equal-width bins across feature range
   - Best for: Features with uniform distribution
   - Example: Dividing 0-100 range into 5 bins of width 20

2. **Quantile**: Equal-frequency bins based on data distribution
   - Best for: Skewed distributions
   - Example: Each bin contains same number of samples

3. **K-means**: Bins based on k-means clustering
   - Best for: Multi-modal distributions
   - Example: Natural clusters in feature space

---

### FeatureRanker

Ranks features by importance for sequence ordering.

```python
from src.processors.sequence_generator import FeatureRanker
```

#### Description
`FeatureRanker` determines the importance ordering of features for each sample or globally. This ranking is crucial for creating meaningful sequences where more important features appear earlier in the sequence.

#### Constructor Parameters
- `config` (Optional[Dict[str, Any]]): Configuration dictionary with keys:
  - `method` (str, default='mutual_info'): Ranking method
  - `top_k` (int, default=10): Number of top features to select

#### Ranking Methods

##### 1. Mutual Information (`mutual_info`)
Uses mutual information between features and target to rank features globally.

**Best for:**
- Supervised learning scenarios
- Finding features most predictive of cancer diagnosis
- Global feature importance ranking

**How it works:**
- Calculates mutual information score for each feature
- Higher scores indicate stronger relationship with target
- All samples use the same feature ranking

##### 2. Z-Score (`zscore`)
Ranks features by their standardized values for each individual sample.

**Best for:**
- Patient-specific feature importance
- Capturing individual variation patterns
- Personalized sequence generation

**How it works:**
- Standardizes each feature across all samples
- For each sample, ranks features by absolute z-score
- Each sample gets a unique feature ranking

##### 3. Correlation (`correlation`)
Ranks features by absolute correlation with target variable.

**Best for:**
- Linear relationships with target
- Simple interpretable rankings
- Quick feature selection

#### Key Methods

##### `fit(data, target)`
Learns feature importance from training data.

**Parameters:**
- `data` (pd.DataFrame): Input features
- `target` (pd.Series): Target variable for supervised ranking

**Returns:**
- `FeatureRanker`: Self for method chaining

##### `transform(data)`
Applies feature ranking and selection.

**Parameters:**
- `data` (pd.DataFrame): Data to rank and select features from

**Returns:**
- `Tuple[pd.DataFrame, List[str]]`: 
  - Ranked/selected features
  - List of top feature names (for mutual_info/correlation)
- `Tuple[pd.DataFrame, List[List[str]]]`:
  - Original data
  - Per-sample feature rankings (for zscore)

#### Usage Examples

```python
# Mutual information ranking (global)
ranker = FeatureRanker({
    'method': 'mutual_info',
    'top_k': 5
})
ranker.fit(features, diagnosis)
ranked_data, top_features = ranker.transform(features)
print(f"Top features: {top_features}")

# Z-score ranking (per-sample)
ranker = FeatureRanker({'method': 'zscore'})
ranker.fit(features, diagnosis)
data, sample_rankings = ranker.transform(features)
print(f"Patient 1 top features: {sample_rankings[0][:3]}")

# Correlation ranking
ranker = FeatureRanker({
    'method': 'correlation',
    'top_k': 8
})
ranker.fit(features, diagnosis)
ranked_data, top_features = ranker.transform(features)
```

---

### CancerSequenceGenerator

Main class that orchestrates the complete sequence generation pipeline.

```python
from src.processors.sequence_generator import CancerSequenceGenerator
```

#### Description
`CancerSequenceGenerator` is the high-level interface that combines feature discretization and ranking to generate sequential representations of cancer patient data. It transforms continuous biomarker measurements into ordered sequences suitable for pattern mining.

#### Constructor Parameters
- `config` (Dict[str, Any]): Configuration dictionary with keys:
  - `max_sequence_length` (int, default=5): Maximum items per sequence
  - `max_gap` (int, default=1): Maximum gap allowed in patterns
  - `ranking_method` (str, default='mutual_info'): Feature ranking method
  - `discretization_strategy` (str, default='quantile'): Binning strategy
  - `n_bins` (int, default=3): Number of discretization bins
  - `top_k` (int, default=10): Number of top features for mutual_info

#### Sequence Generation Process

1. **Feature Discretization**: Continuous values → Categorical bins
2. **Feature Ranking**: Determine importance ordering
3. **Sequence Construction**: Create ordered itemsets based on ranking
4. **Output**: List of sequences where each sequence is a list of itemsets

#### Key Methods

##### `fit(data, target)`
Fits both discretizer and ranker components.

**Parameters:**
- `data` (pd.DataFrame): Cancer biomarker features
- `target` (pd.Series): Diagnosis labels ('M' for malignant, 'B' for benign)

**Returns:**
- `CancerSequenceGenerator`: Self for method chaining

##### `generate_sequences(data)`
Generates sequences from fitted components.

**Parameters:**
- `data` (pd.DataFrame): Cancer features to convert to sequences

**Returns:**
- `List[List[List[str]]]`: List of sequences, where each sequence contains itemsets

##### `fit_generate(data, target)`
Convenience method combining fit and generate_sequences.

**Parameters:**
- `data` (pd.DataFrame): Cancer features
- `target` (pd.Series): Diagnosis labels

**Returns:**
- `List[List[List[str]]]`: Generated sequences

#### Sequence Format

Each sequence follows this structure:
```python
[
    ['feature1_high'],           # Itemset 1: Most important feature
    ['feature2_medium'],         # Itemset 2: Second most important
    ['feature3_low'],            # Itemset 3: Third most important
    # ... up to max_sequence_length
]
```

#### Usage Example

```python
# Configure sequence generator
config = {
    'max_sequence_length': 5,
    'ranking_method': 'mutual_info',
    'discretization_strategy': 'quantile',
    'n_bins': 3,
    'top_k': 10
}

# Generate sequences
generator = CancerSequenceGenerator(config)
sequences = generator.fit_generate(cancer_features, diagnosis_labels)

# Example output
print(f"Generated {len(sequences)} sequences")
print(f"Example sequence: {sequences[0]}")
# Output: [['radius_worst_high'], ['perimeter_worst_high'], ['area_worst_medium'], ...]
```

#### Configuration Guidelines

**For Exploratory Analysis:**
```python
{
    'max_sequence_length': 3,
    'discretization_strategy': 'quantile',
    'n_bins': 3,
    'ranking_method': 'mutual_info',
    'top_k': 15
}
```

**For Detailed Analysis:**
```python
{
    'max_sequence_length': 7,
    'discretization_strategy': 'uniform',
    'n_bins': 5,
    'ranking_method': 'zscore',
    'top_k': 20
}
```

**For Clinical Interpretation:**
```python
{
    'max_sequence_length': 4,
    'discretization_strategy': 'quantile',
    'n_bins': 3,  # Simple low/medium/high
    'ranking_method': 'mutual_info',
    'top_k': 8
}
```

---

## Integration Example

Here's how the processor components work together:

```python
import pandas as pd
from src.processors.sequence_generator import CancerSequenceGenerator

# Load cancer dataset
cancer_data = pd.read_csv('wisconsin_breast_cancer.csv')
features = cancer_data.drop(['id', 'diagnosis'], axis=1)
target = cancer_data['diagnosis']

# Configure and generate sequences
config = {
    'max_sequence_length': 5,
    'ranking_method': 'mutual_info',
    'discretization_strategy': 'quantile',
    'n_bins': 3,
    'top_k': 10
}

generator = CancerSequenceGenerator(config)
sequences = generator.fit_generate(features, target)

# Sequences are now ready for pattern mining
print(f"Generated {len(sequences)} sequences for pattern mining")
```

This preprocessing pipeline transforms complex cancer biomarker data into interpretable sequential patterns that can reveal diagnostic insights through pattern mining algorithms.