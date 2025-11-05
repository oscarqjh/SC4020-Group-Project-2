# Random Forest Classifier Training Guide

## Overview

This guide provides comprehensive instructions for training a Random Forest binary classifier on the Wisconsin Breast Cancer dataset. The training pipeline integrates intelligent feature selection based on pattern mining results with hyperparameter tuning using GridSearchCV.

The training script performs the following operations:
1. **Feature Selection**: Uses Task 2 pattern mining results to select discriminative features
2. **Hyperparameter Tuning**: GridSearchCV with customizable parameter grid
3. **Model Training**: Random Forest classifier with class imbalance handling
4. **Model Evaluation**: Comprehensive metrics (F1, Recall, Precision, ROC-AUC)
5. **Model Persistence**: Timestamped pickle files for reproducibility

## Prerequisites

### 1. Completed Task 2

Pattern mining must be run first to generate `outputs/feature_importance.txt`:

```bash
python scripts/cancer_pattern_mining.py
```

This generates the feature importance file that the feature selector uses.

### 2. Dataset

Wisconsin Breast Cancer dataset at `data/raw/wisconsin_breast_cancer.csv`:
- 569 samples
- 30 features
- Binary classification: B (Benign) vs M (Malignant)
- Class distribution: 357 benign (62.7%), 212 malignant (37.3%)

### 3. Python Environment

- Python 3.8+ with dependencies from requirements.txt
- Virtual environment (recommended)

## Virtual Environment Setup

### Step 1: Create Virtual Environment

Navigate to project root and create virtual environment:

```bash
# Navigate to project root
cd /Users/bytedance/GitHub/SC4020-Group-Project-2

# Create virtual environment
python3 -m venv .venv
```

This creates an isolated Python environment in the `.venv` directory.

### Step 2: Activate Virtual Environment

Platform-specific activation commands:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

After activation, your terminal prompt should show `(.venv)` prefix.

### Step 3: Install Dependencies

```bash
# Install project in development mode
python3 -m pip install -e .

# Verify installation
python scripts/train_random_forest.py --help
```

The `-e` flag installs in editable mode, allowing code changes without reinstallation.

### Step 4: Verify Environment

Verification commands:

```bash
# Check Python version
python --version

# Check scikit-learn installation
python -c "import sklearn; print(sklearn.__version__)"

# List installed packages
pip list
```

## Training Pipeline Usage

### Basic Usage

Simplest command to run training:

```bash
python scripts/train_random_forest.py
```

This runs with default parameters:
- Uses all 30 features (no feature selection)
- GridSearchCV with default hyperparameter grid
- 5-fold cross-validation
- Saves model to `scripts/random_forest_model_YYYYMMDD_HHMMSS.pkl`

### With Feature Selection

Recommended approach using intelligent feature selection:

```bash
python scripts/train_random_forest.py \
    --use-feature-selection \
    --top-n-features 10 \
    --correlation-threshold 0.9 \
    --verbose
```

This enables intelligent feature selection based on Task 2 pattern mining results, selecting top 10 non-correlated features.

### Custom Hyperparameter Grid

For advanced hyperparameter tuning:

```bash
python scripts/train_random_forest.py \
    --use-feature-selection \
    --n-estimators 50 100 200 \
    --max-depth 5 10 20 None \
    --min-samples-split 2 5 10 \
    --cv-folds 10 \
    --verbose
```

This tests 4 × 4 × 3 = 48 parameter combinations with 10-fold CV.

### Custom Data Paths

For different file locations:

```bash
python scripts/train_random_forest.py \
    --data-path /path/to/data.csv \
    --feature-importance-path /path/to/importance.txt \
    --output-dir /path/to/output
```

## Configuration Parameters

### Feature Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-feature-selection` | flag | False | Enable feature selection |
| `--top-n-features` | int | 10 | Number of features to select |
| `--correlation-threshold` | float | 0.9 | Multicollinearity threshold |
| `--aggregation-method` | str | 'sum' | How to combine malignant/benign scores |

**Aggregation Methods:**
- `sum`: Sum of malignant and benign support values (default, recommended)
- `mean`: Average of support values
- `max`: Maximum of support values

**Correlation Threshold:**
- Higher values (0.95): More permissive, keeps more features
- Lower values (0.85): More restrictive, removes more redundant features
- Default (0.9): Balanced approach

### Random Forest Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--n-estimators` | list[int] | [100, 200, 300] | Number of trees to test |
| `--max-depth` | list | [10, 20, None] | Maximum tree depth |
| `--min-samples-split` | list[int] | [2, 5, 10] | Minimum samples to split |
| `--class-weight` | str | 'balanced' | Class imbalance handling |

**Impact on Performance:**
- `n_estimators`: More trees = better performance but slower training
- `max_depth`: Deeper trees = more complex models, risk of overfitting
- `min_samples_split`: Higher values = simpler trees, less overfitting
- `class_weight='balanced'`: Automatically adjusts for class imbalance

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--cv-folds` | int | 5 | Cross-validation folds |
| `--test-size` | float | 0.2 | Test set proportion |
| `--random-state` | int | 42 | Random seed |
| `--n-jobs` | int | -1 | Parallel jobs (-1 = all cores) |

**Guidelines:**
- `cv_folds`: 5-10 folds for stable estimates, more folds = slower but more accurate
- `test_size`: 0.2 (80/20 split) is standard, 0.3 for smaller datasets
- `random_state`: Set to 42 for reproducibility
- `n_jobs`: -1 uses all CPU cores, set to 2-4 for shared systems

## Understanding the Output

### Console Output Sections

The training script produces several output sections:

#### 1. Data Loading
- Dataset shape and number of samples
- Class distribution with counts and percentages
- First 10 feature names

#### 2. Feature Selection (if enabled)
- Number of original vs selected features
- List of selected features
- Top 10 feature scores
- Features removed due to multicollinearity
- Correlation matrix (if verbose)

#### 3. Training Progress
- Configuration summary
- Number of parameter combinations
- GridSearchCV progress output

#### 4. Evaluation Results
- Best hyperparameters
- Training metrics (F1, Recall, Precision, ROC-AUC)
- Test metrics
- Confusion matrix
- Classification report
- Top 10 feature importances

#### 5. Model Persistence
- Saved file locations
- Timestamp information

### Interpreting Metrics

#### F1 Score (Northstar Metric)

- **Range**: 0 to 1 (higher is better)
- **Definition**: Harmonic mean of precision and recall
- **Formula**: F1 = 2 × (precision × recall) / (precision + recall)
- **Target**: > 0.90 for good performance
- **Why Primary**: Balanced measure for imbalanced datasets

#### Recall (Sensitivity)

- **Range**: 0 to 1 (higher is better)
- **Definition**: Proportion of actual malignant cases correctly identified
- **Formula**: Recall = TP / (TP + FN)
- **Target**: > 0.90 to avoid missing malignant cases
- **Clinical Importance**: Critical for cancer diagnosis (minimize false negatives)

#### Precision

- **Range**: 0 to 1 (higher is better)
- **Definition**: Proportion of predicted malignant cases that are actually malignant
- **Formula**: Precision = TP / (TP + FP)
- **Target**: > 0.85 for clinical utility
- **Clinical Importance**: Reduces unnecessary biopsies (minimize false positives)

#### ROC-AUC

- **Range**: 0 to 1 (higher is better)
- **Definition**: Area under ROC curve
- **Interpretation**: Overall discriminative ability
- **Target**: > 0.95 for excellent performance
- **Use Case**: Overall model performance assessment

### Confusion Matrix Interpretation

Example confusion matrix:

```
[[TN  FP]
 [FN  TP]]
```

**For Wisconsin Breast Cancer dataset:**
- **TN (True Negative)**: Correctly identified benign cases (B → B)
- **FP (False Positive)**: Benign cases incorrectly classified as malignant (B → M)
- **FN (False Negative)**: Malignant cases incorrectly classified as benign (M → B) - **most critical error**
- **TP (True Positive)**: Correctly identified malignant cases (M → M)

**Clinical Implications:**
- **False Negatives (FN)**: Most dangerous - missed cancer diagnosis
- **False Positives (FP)**: Less critical - unnecessary follow-up tests
- **Goal**: Minimize FN while keeping FP reasonable

## Saved Files

### Model File

**Location**: `scripts/random_forest_model_YYYYMMDD_HHMMSS.pkl`

**Content**: Entire RandomForestBinaryClassifier object including:
- Trained Random Forest model
- Best hyperparameters
- Feature names
- Training/test metrics
- Configuration

**Usage:**
```python
from src.classifiers import RandomForestBinaryClassifier

# Load model
classifier = RandomForestBinaryClassifier.load_model('scripts/random_forest_model_20240101_120000.pkl')

# Make predictions
predictions = classifier.predict(X_new)
```

### Feature Selector File

**Location**: `scripts/feature_selector_YYYYMMDD_HHMMSS.pkl`

**Content**: Fitted FeatureSelector object

**Usage:**
```python
import pickle

# Load selector
with open('scripts/feature_selector_20240101_120000.pkl', 'rb') as f:
    selector = pickle.load(f)

# Transform new data
X_selected = selector.transform(X_new)
```

### Results Summary File

**Location**: `scripts/training_results_YYYYMMDD_HHMMSS.txt`

**Content**: Text summary with:
- Best hyperparameters
- Test set metrics
- Selected features (if feature selection used)
- Top 10 feature importances

**Usage**: Human-readable summary for reporting and documentation.

## Troubleshooting

### Common Issues

#### Issue 1: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution:**
1. Ensure virtual environment is activated
2. Install project: `python3 -m pip install -e .`
3. Verify installation: `pip list | grep SC4020`

#### Issue 2: Feature Importance File Not Found

**Error**: `FileNotFoundError: Feature importance file not found`

**Solution:**
1. Run Task 2 pattern mining first: `python scripts/cancer_pattern_mining.py`
2. Verify file exists: `ls outputs/feature_importance.txt`
3. Or disable feature selection: `--no-feature-selection`

#### Issue 3: Insufficient Memory

**Error**: `MemoryError` during GridSearchCV

**Solution:**
1. Reduce hyperparameter grid size
2. Reduce cv_folds: `--cv-folds 3`
3. Reduce n_jobs: `--n-jobs 2`
4. Use smaller n_estimators: `--n-estimators 50 100`

#### Issue 4: Long Training Time

**Solution:**
1. Reduce hyperparameter combinations
2. Use fewer CV folds: `--cv-folds 3`
3. Reduce n_estimators: `--n-estimators 100 200`
4. Enable verbose to monitor progress: `--verbose`

#### Issue 5: Poor Model Performance

**Symptoms**: Low F1 score (< 0.85)

**Solutions:**
1. Enable feature selection: `--use-feature-selection`
2. Increase hyperparameter grid: `--n-estimators 100 200 300 400`
3. Adjust max_depth: `--max-depth 15 20 25 None`
4. Check data quality and preprocessing

## Best Practices

### Recommended Workflow

#### 1. First Run: Baseline with Defaults

```bash
python scripts/train_random_forest.py --verbose
```

Establishes baseline performance with all features.

#### 2. Enable Feature Selection: Reduce Dimensionality

```bash
python scripts/train_random_forest.py \
    --use-feature-selection \
    --top-n-features 10 \
    --verbose
```

Uses intelligent feature selection to reduce dimensionality.

#### 3. Fine-tune Hyperparameters: Based on Initial Results

```bash
python scripts/train_random_forest.py \
    --use-feature-selection \
    --n-estimators 150 200 250 \
    --max-depth 15 20 25 \
    --verbose
```

Fine-tunes based on initial results.

#### 4. Final Model: Train with Best Configuration

```bash
python scripts/train_random_forest.py \
    --use-feature-selection \
    --n-estimators 200 \
    --max-depth 20 \
    --min-samples-split 5
```

Trains final model with best configuration.

### Performance Optimization

- Use `--n-jobs -1` to utilize all CPU cores
- Start with smaller grids for quick iteration
- Use `--verbose` to monitor progress
- Save intermediate results for comparison

### Model Selection Criteria

Prioritize metrics in order:

1. **F1 Score** (primary): Balance between precision and recall
2. **Recall**: Minimize false negatives (critical for cancer diagnosis)
3. **Precision**: Minimize false positives (reduce unnecessary procedures)
4. **ROC-AUC**: Overall discriminative ability

## Integration with Task 2

### Workflow

#### 1. Task 2 (Pattern Mining): Discovers Discriminative Features

**Command:**
```bash
python scripts/cancer_pattern_mining.py
```

**Output**: `outputs/feature_importance.txt`
- Contains support values for malignant and benign patterns
- Format: Feature name → support value

#### 2. Feature Selection: Uses Task 2 Results

**Process:**
1. Combines support values from both classes (malignant + benign)
2. Detects multicollinearity (radius, perimeter, area are correlated)
3. Selects top-N non-redundant features

**Example:**
- `concave points_worst`: Score 15.5 (high)
- `radius_worst`: Score 12.3 (high)
- `perimeter_worst`: Score 12.2 (high, but correlated with radius)
- Result: Selects `concave points_worst` and `radius_worst`, removes `perimeter_worst`

#### 3. Model Training: Uses Selected Features

**Process:**
1. Trains Random Forest with GridSearchCV
2. Handles class imbalance with `class_weight='balanced'`
3. Evaluates with F1 as northstar metric

**Output**: Trained model with best hyperparameters

#### 4. Model Persistence: Saves for Deployment

**Process:**
1. Saves timestamped pickle files
2. Includes all metadata for reproducibility
3. Generates human-readable summary

**Files:**
- `random_forest_model_YYYYMMDD_HHMMSS.pkl`
- `feature_selector_YYYYMMDD_HHMMSS.pkl`
- `training_results_YYYYMMDD_HHMMSS.txt`

## Example Session

Complete example with expected output:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training with feature selection
python scripts/train_random_forest.py --use-feature-selection --top-n-features 10 --verbose
```

**Expected Output:**

```
============================================================
Random Forest Classifier Training
============================================================
Timestamp: 2024-01-01 12:00:00

Configuration:
  Data path: /Users/bytedance/GitHub/SC4020-Group-Project-2/data/raw/wisconsin_breast_cancer.csv
  Feature importance path: /Users/bytedance/GitHub/SC4020-Group-Project-2/outputs/feature_importance.txt
  Output directory: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts
  Use feature selection: True
  Top N features: 10
  Correlation threshold: 0.9
  Aggregation method: sum

Loading Wisconsin Breast Cancer dataset...
Data path: /Users/bytedance/GitHub/SC4020-Group-Project-2/data/raw/wisconsin_breast_cancer.csv

Dataset shape: (569, 30)
Number of samples: 569

Class distribution:
  B: 357 (62.7%)
  M: 212 (37.3%)

Feature names (first 10): ['radius_mean', 'texture_mean', 'perimeter_mean', ...]

Data validation passed!

============================================================
FEATURE SELECTION
============================================================

Selected 10 features from 30 original features

Selected features: ['concave points_worst', 'radius_worst', 'texture_worst', ...]

Top 10 feature scores:
  concave points_worst: 15.5234
  radius_worst: 12.3456
  ...

============================================================
TRAINING RANDOM FOREST CLASSIFIER
============================================================

Configuration:
  Hyperparameter grid:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, None]
    min_samples_split: [2, 5, 10]
  Cross-validation folds: 5
  Test size: 0.2
  Class weight: balanced

Fitting GridSearchCV with 27 parameter combinations...
[GridSearchCV progress output]

Training completed!

============================================================
EVALUATION RESULTS
============================================================
==================================================
Random Forest Classifier Summary
==================================================
Best Hyperparameters:
{'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5}

Training Metrics:
  F1 Score:    0.9824
  Recall:      0.9789
  Precision:   0.9859
  ROC-AUC:     0.9934

Test Metrics:
  F1 Score:    0.9524
  Recall:      0.9500
  Precision:   0.9548
  ROC-AUC:     0.9876

Confusion Matrix:
[[70  2]
 [ 3 39]]

Classification Report:
              precision    recall  f1-score   support
           B       0.96      0.97      0.96        72
           M       0.95      0.93      0.94        42
    accuracy                           0.96       114
   macro avg       0.95      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

Top 10 Feature Importances:
                     feature  importance
0         concave points_worst    0.123456
1              radius_worst    0.098765
...

============================================================
SAVING MODEL AND RESULTS
============================================================

Model saved to: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts/random_forest_model_20240101_120000.pkl
Feature selector saved to: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts/feature_selector_20240101_120000.pkl
Results summary saved to: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts/training_results_20240101_120000.txt

============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================

Key Findings:
  Best F1 Score: 0.9524
  Number of features used: 10
  Model saved to: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts/random_forest_model_20240101_120000.pkl
  Results saved to: /Users/bytedance/GitHub/SC4020-Group-Project-2/scripts/training_results_20240101_120000.txt
```

## References

- **Task 2 Pattern Mining Guide**: [`docs/cancer_pattern_mining.md`](cancer_pattern_mining.md)
- **Feature Selector API**: `src/classifiers/feature_selector.py`
- **Random Forest Classifier API**: `src/classifiers/random_forest_classifier.py`
- **Project README**: [`../README.md`](../README.md)

