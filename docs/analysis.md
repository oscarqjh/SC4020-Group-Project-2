# Analysis Documentation

This document provides comprehensive documentation for the analysis components in the sequential pattern mining system for cancer diagnosis.

## Overview

The analysis module contains classes responsible for discovering patterns in sequential cancer data, evaluating their significance, and performing sensitivity analysis. The module implements sequential pattern mining algorithms and provides interpretable results for medical insights.

## Base Classes

### BaseAnalyzer

The foundation class for all analysis components.

```python
from src.analysis.base import BaseAnalyzer
```

#### Description
`BaseAnalyzer` provides the core interface and common functionality for all analysis operations. It establishes a standardized workflow for pattern discovery, evaluation, and result interpretation.

#### Constructor Parameters
- `config` (Optional[Dict[str, Any]]): Configuration dictionary containing analysis parameters

#### Key Methods

##### `analyze(sequences, target)`
Performs the main analysis on sequential data.

**Parameters:**
- `sequences` (List): Input sequences for analysis
- `target` (List): Target labels corresponding to sequences

**Returns:**
- `Dict`: Analysis results dictionary

##### `get_results()`
Retrieves the latest analysis results.

**Returns:**
- `Dict`: Stored analysis results

##### `validate_input(sequences, target)`
Validates input data format and consistency.

**Parameters:**
- `sequences` (List): Input sequences
- `target` (List): Target labels

**Raises:**
- `ValueError`: If input validation fails

#### Usage Example
```python
# Basic usage pattern for any analyzer
analyzer = SomeAnalyzer(config={'param': 'value'})
results = analyzer.analyze(sequences, target_labels)
```

---

## Pattern Mining Classes

### GSPAlgorithm

Implementation of the Generalized Sequential Pattern (GSP) algorithm.

```python
from src.analysis.pattern_mining import GSPAlgorithm
```

#### Description
`GSPAlgorithm` discovers frequent sequential patterns in cancer diagnosis data. It identifies common subsequences that occur frequently within malignant or benign cases, enabling the discovery of diagnostic patterns.

#### Constructor Parameters
- `min_support` (float, default=0.1): Minimum support threshold for patterns
- `max_pattern_length` (int, default=4): Maximum length of discovered patterns
- `max_gap` (int, default=1): Maximum gap allowed between consecutive elements

#### Algorithm Overview

The GSP algorithm operates in multiple phases:

1. **Candidate Generation**: Create potential patterns of length k+1 from frequent patterns of length k
2. **Support Calculation**: Count occurrences of candidates in sequence database
3. **Pruning**: Remove candidates below minimum support threshold
4. **Iteration**: Repeat until no new frequent patterns found

#### Key Methods

##### `find_frequent_patterns(sequences, target_labels=None)`
Discovers frequent sequential patterns in the input sequences.

**Parameters:**
- `sequences` (List[List[List[str]]]): Input sequences where each sequence contains itemsets
- `target_labels` (Optional[List[str]]): Class labels for sequences

**Returns:**
- `Dict`: Dictionary containing:
  - `'patterns'`: List of frequent patterns with support values
  - `'class_patterns'`: Patterns grouped by class (if target_labels provided)

**Example:**
```python
gsp = GSPAlgorithm(min_support=0.2, max_pattern_length=3)
results = gsp.find_frequent_patterns(cancer_sequences, diagnosis_labels)

# Access frequent patterns
for pattern, support in results['patterns']:
    print(f"Pattern: {pattern}, Support: {support:.3f}")
```

##### `calculate_support(pattern, sequences)`
Calculates support value for a specific pattern.

**Parameters:**
- `pattern` (List[List[str]]): Sequential pattern to evaluate
- `sequences` (List[List[List[str]]]): Sequence database

**Returns:**
- `float`: Support value (proportion of sequences containing the pattern)

##### `is_subsequence(pattern, sequence)`
Checks if a pattern is a subsequence of a given sequence.

**Parameters:**
- `pattern` (List[List[str]]): Pattern to search for
- `sequence` (List[List[str]]): Sequence to search in

**Returns:**
- `bool`: True if pattern is found as subsequence

#### Pattern Format

Patterns are represented as lists of itemsets:
```python
# Single-item pattern
pattern = [['radius_worst_high']]

# Two-item sequential pattern
pattern = [['radius_worst_high'], ['texture_mean_medium']]

# Multi-item pattern with gap
pattern = [['radius_worst_high'], ['texture_mean_medium'], ['smoothness_worst_low']]
```

#### Usage Example

```python
# Configure GSP algorithm
gsp = GSPAlgorithm(
    min_support=0.15,
    max_pattern_length=4,
    max_gap=1
)

# Find patterns in cancer sequences
results = gsp.find_frequent_patterns(sequences, diagnosis_labels)

# Analyze results
print(f"Found {len(results['patterns'])} frequent patterns")

# Class-specific patterns
malignant_patterns = results['class_patterns']['M']
benign_patterns = results['class_patterns']['B']

print(f"Malignant-specific patterns: {len(malignant_patterns)}")
print(f"Benign-specific patterns: {len(benign_patterns)}")
```

---

### SequentialPatternAnalyzer

High-level analyzer that orchestrates pattern discovery and evaluation.

```python
from src.analysis.pattern_mining import SequentialPatternAnalyzer
```

#### Description
`SequentialPatternAnalyzer` provides a comprehensive analysis framework that combines pattern discovery, statistical evaluation, and interpretable results generation. It serves as the main interface for cancer pattern analysis.

#### Constructor Parameters
- `config` (Dict[str, Any]): Configuration dictionary with keys:
  - `min_support` (float, default=0.1): Minimum support threshold
  - `max_pattern_length` (int, default=4): Maximum pattern length
  - `max_gap` (int, default=1): Maximum gap in patterns
  - `confidence_threshold` (float, default=0.7): Minimum confidence for rules

#### Key Methods

##### `analyze(sequences, target_labels)`
Performs comprehensive sequential pattern analysis.

**Parameters:**
- `sequences` (List[List[List[str]]]): Input sequences
- `target_labels` (List[str]): Diagnosis labels ('M' or 'B')

**Returns:**
- `Dict`: Comprehensive analysis results containing:
  - `'summary'`: Basic statistics
  - `'frequent_patterns'`: All frequent patterns
  - `'class_patterns'`: Patterns by diagnosis class
  - `'class_comparison'`: Discriminative pattern analysis
  - `'feature_importance'`: Feature importance metrics
  - `'pattern_rules'`: Association rules between patterns

**Example:**
```python
analyzer = SequentialPatternAnalyzer({
    'min_support': 0.1,
    'max_pattern_length': 3,
    'confidence_threshold': 0.8
})

results = analyzer.analyze(cancer_sequences, diagnosis_labels)

# Access different result components
summary = results['summary']
discriminative = results['class_comparison']['discriminative_patterns']
feature_importance = results['feature_importance']
```

##### `get_interpretable_results()`
Generates human-readable interpretations of discovered patterns.

**Returns:**
- `Dict`: Interpretable results containing:
  - `'interpretations'`: List of pattern interpretations
  - `'clinical_insights'`: Medical insights from patterns
  - `'recommendations'`: Suggested clinical actions

**Example:**
```python
# After running analysis
interpretable = analyzer.get_interpretable_results()

for interpretation in interpretable['interpretations']:
    print(f"Pattern: {interpretation['pattern']}")
    print(f"Meaning: {interpretation['interpretation']}")
    print(f"Clinical Relevance: {interpretation['clinical_relevance']}")
```

#### Analysis Components

##### 1. Pattern Discovery
- Uses GSP algorithm to find frequent patterns
- Identifies patterns specific to malignant/benign cases
- Calculates support, confidence, and lift metrics

##### 2. Discriminative Analysis
- Compares pattern frequencies between classes
- Identifies patterns that strongly discriminate between malignant/benign
- Calculates statistical significance of differences

##### 3. Feature Importance
- Analyzes which features appear most frequently in patterns
- Ranks features by their discriminative power
- Provides feature-level insights for clinical interpretation

##### 4. Association Rules
- Discovers rules between patterns (e.g., pattern A → pattern B)
- Calculates confidence and lift for rules
- Identifies sequential dependencies in cancer progression

#### Result Structure

```python
{
    'summary': {
        'total_sequences': int,
        'malignant_count': int,
        'benign_count': int,
        'total_patterns': int,
        'unique_malignant_patterns': int,
        'unique_benign_patterns': int,
        'discriminative_patterns': int
    },
    
    'frequent_patterns': [
        {
            'pattern': [['feature_value']],
            'support': float,
            'class_distribution': {'M': float, 'B': float}
        }
    ],
    
    'class_comparison': {
        'discriminative_patterns': [
            {
                'pattern': [['feature_value']],
                'malignant_support': float,
                'benign_support': float,
                'lift': float,
                'discriminative_for': str
            }
        ]
    },
    
    'feature_importance': {
        'top_malignant_features': [(str, float)],
        'top_benign_features': [(str, float)],
        'discriminative_features': [(str, float)]
    }
}
```

---

## Evaluation Classes

### SensitivityAnalyzer

Performs comprehensive sensitivity analysis across different parameter configurations.

```python
from src.analysis.evaluation import SensitivityAnalyzer
```

#### Description
`SensitivityAnalyzer` evaluates how different parameter choices affect pattern discovery results. It helps identify optimal configurations and assesses the robustness of findings across different experimental settings.

#### Constructor Parameters
- `base_config` (Dict[str, Any]): Base configuration for experiments

#### Analysis Dimensions

##### 1. Binning Strategy Analysis
Compares different discretization approaches:
- **Uniform**: Equal-width bins
- **Quantile**: Equal-frequency bins  
- **K-means**: Clustering-based bins

##### 2. Support Threshold Analysis
Evaluates impact of different minimum support values:
- High support: More conservative, fewer patterns
- Low support: More liberal, more patterns
- Optimal range: Balance between coverage and specificity

##### 3. Ranking Method Analysis
Compares feature ranking approaches:
- **Mutual Information**: Global feature importance
- **Z-score**: Sample-specific ranking
- **Correlation**: Linear relationship-based

#### Key Methods

##### `analyze_binning_strategies(features, target)`
Compares different discretization strategies.

**Parameters:**
- `features` (pd.DataFrame): Cancer biomarker features
- `target` (pd.Series): Diagnosis labels

**Returns:**
- `Dict`: Results comparing binning strategies with metrics:
  - Pattern count per strategy
  - Discriminative power metrics
  - Stability across runs

##### `analyze_support_thresholds(features, target)`
Evaluates impact of different support thresholds.

**Parameters:**
- `features` (pd.DataFrame): Cancer features
- `target` (pd.Series): Diagnosis labels

**Returns:**
- `Dict`: Support threshold analysis results

##### `analyze_ranking_methods(features, target)`
Compares different feature ranking methods.

**Parameters:**
- `features` (pd.DataFrame): Cancer features
- `target` (pd.Series): Diagnosis labels

**Returns:**
- `Dict`: Ranking method comparison results

##### `get_best_configuration()`
Determines optimal parameter configuration based on analysis.

**Returns:**
- `Dict`: Best configuration with rationale:
  - `'best_config'`: Optimal parameter settings
  - `'rationale'`: Explanation of choices
  - `'performance_metrics'`: Supporting metrics

#### Usage Example

```python
# Configure sensitivity analyzer
base_config = {
    'max_sequence_length': 5,
    'max_gap': 1,
    'min_support': 0.1,
    'max_pattern_length': 4
}

analyzer = SensitivityAnalyzer(base_config)

# Run comprehensive analysis
binning_results = analyzer.analyze_binning_strategies(features, target)
support_results = analyzer.analyze_support_thresholds(features, target)
ranking_results = analyzer.analyze_ranking_methods(features, target)

# Get optimal configuration
best_config_info = analyzer.get_best_configuration()
print("Recommended configuration:")
for param, value in best_config_info['best_config'].items():
    print(f"  {param}: {value}")

print("\nRationale:")
for reason in best_config_info['rationale']:
    print(f"  - {reason}")
```

#### Evaluation Metrics

The analyzer uses multiple metrics to assess configuration quality:

1. **Pattern Quality Metrics:**
   - Number of discriminative patterns
   - Average pattern support
   - Pattern diversity (unique patterns found)

2. **Stability Metrics:**
   - Consistency across different random seeds
   - Robustness to parameter variations
   - Reproducibility of key findings

3. **Interpretability Metrics:**
   - Clinical relevance of discovered patterns
   - Simplicity of pattern structure
   - Ease of medical interpretation

---

## Integration Example

Here's a complete example showing how to use all analysis components together:

```python
import pandas as pd
from src.processors.sequence_generator import CancerSequenceGenerator
from src.analysis.pattern_mining import SequentialPatternAnalyzer
from src.analysis.evaluation import SensitivityAnalyzer

# Load and prepare data
cancer_data = pd.read_csv('wisconsin_breast_cancer.csv')
features = cancer_data.drop(['id', 'diagnosis'], axis=1)
target = cancer_data['diagnosis']

# Generate sequences
config = {
    'max_sequence_length': 5,
    'ranking_method': 'mutual_info',
    'discretization_strategy': 'quantile',
    'n_bins': 3,
    'top_k': 10,
    'min_support': 0.1,
    'max_pattern_length': 4
}

generator = CancerSequenceGenerator(config)
sequences = generator.fit_generate(features, target)

# Analyze patterns
analyzer = SequentialPatternAnalyzer(config)
results = analyzer.analyze(sequences, target.tolist())

# Get interpretable results
interpretable = analyzer.get_interpretable_results()
print("Key Discoveries:")
for interpretation in interpretable['interpretations'][:5]:
    print(f"- {interpretation['interpretation']}")

# Perform sensitivity analysis
sensitivity_analyzer = SensitivityAnalyzer(config)
sensitivity_analyzer.analyze_binning_strategies(features, target)
sensitivity_analyzer.analyze_support_thresholds(features, target)
sensitivity_analyzer.analyze_ranking_methods(features, target)

# Get optimal configuration
best_config = sensitivity_analyzer.get_best_configuration()
print(f"\nRecommended configuration: {best_config['best_config']}")
```

This analysis pipeline provides comprehensive insights into cancer diagnosis patterns, helping clinicians understand the sequential relationships between biomarker measurements and diagnostic outcomes.