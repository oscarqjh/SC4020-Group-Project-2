# SC4020-Group-Project-2

NTU SC4020 Data Analytics & Mining AY25 Group Project 2

## Project Overview

This project implements comprehensive healthcare data analytics solutions using biomedical datasets to analyze disease patterns and cancer diagnosis characteristics. The project consists of three main tasks focusing on different aspects of medical data mining and pattern discovery.

## Environment Setup

### Create and Activate Environment

```bash
# Create the virtual environment using Python's built-in venv module
python3 -m venv .venv

# Activate the environment

# On macOS / Linux (bash/zsh)
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\\Scripts\\activate.bat

# On Windows (PowerShell)
.venv\\Scripts\\Activate.ps1
```

### Install Dependencies

```bash
# Install the project in development mode using pip
python3 -m pip install -e .

# Verify installation (Task 2)
python scripts/cancer_pattern_mining.py --help
```

## Project Structure

```
SC4020-Group-Project-2/
├── README.md                    # This overview file
├── docs/                        # Comprehensive documentation
│   ├── README.md               # Documentation navigation
│   ├── symptom_analysis.md     # Task 1 complete guide
│   ├── cancer_pattern_mining.md # Task 2 complete guide
│   ├── random_forest_training.md # Task 3 complete guide
│   ├── processors.md           # Data processing components
│   └── analysis.md             # Analysis and mining algorithms
├── src/                         # Source code (Task 1 & 2 implemented)
│   ├── processors/             # Data preprocessing pipeline
│   ├── analysis/               # Pattern mining and evaluation
│   └── classifiers/            # Feature selection and classification
├── scripts/                     # Executable scripts
│   ├── symptom_analysis.py     # Task 1 CLI tool
│   ├── cancer_pattern_mining.py # Task 2 CLI tool
│   └── train_random_forest.py   # Task 3 training pipeline
├── data/                        # Data storage
│   ├── dataset.csv             # Symptom data (Task 1)
│   └── raw/
│       └── wisconsin_breast_cancer.csv # Cancer data (Task 2)
├── outputs/                     # Analysis results
├── tests/                       # Unit tests
└── [configuration files]
```

## Getting Started by Task

### Task 1: Symptom Analysis

```bash
# Basic analysis
python scripts/symptom_analysis.py

# Advanced analysis with custom parameters
python scripts/symptom_analysis.py \
    --data-path data/dataset.csv \
    --output-dir outputs \
    --min-support 0.01 \
    --verbose
```

### Task 2: Cancer Pattern Mining

```bash
# Basic analysis
python scripts/cancer_pattern_mining.py

# Advanced analysis with custom parameters
python scripts/cancer_pattern_mining.py \
    --discretization-strategy quantile \
    --n-bins 5 \
    --min-support 0.1 \
    --ranking-method mutual_info \
    --verbose
```

### Task 3: Random Forest Classifier Training

```bash
# Basic training with default parameters
python scripts/train_random_forest.py

# Training with feature selection (recommended)
python scripts/train_random_forest.py \
    --use-feature-selection \
    --top-n-features 10 \
    --verbose

# Advanced training with custom hyperparameters
python scripts/train_random_forest.py \
    --use-feature-selection \
    --n-estimators 100 200 300 \
    --max-depth 10 20 None \
    --cv-folds 10 \
    --verbose
```

## Feature Selection Based on Pattern Mining

The `FeatureSelector` class performs intelligent feature selection based on pattern mining results, automatically detecting and removing multicollinear features.

### Key Capabilities

- Parses feature importance scores from pattern mining analysis
- Combines support values from both malignant and benign patterns
- Detects multicollinearity using correlation analysis
- Selects top-N most important features while avoiding redundancy

### Usage Example

```python
from src.classifiers import FeatureSelector, load_cancer_data

# Load data
X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')

# Create and fit selector
selector = FeatureSelector({
    'feature_importance_path': 'outputs/feature_importance.txt',
    'top_n': 5,
    'correlation_threshold': 0.9,
    'aggregation_method': 'sum'
})

# Select features
X_selected = selector.fit_transform(X, y)

# View selected features
print(selector.get_selected_features())
# or use select_features() method
print(selector.select_features())
```

### Configuration Parameters

- `feature_importance_path`: Path to feature importance file (required)
- `top_n`: Number of features to select (default: 10)
- `correlation_threshold`: Threshold for multicollinearity detection (default: 0.9)
- `aggregation_method`: How to combine scores - 'sum', 'mean', or 'max' (default: 'sum')

### Note on Multicollinearity

The selector automatically handles highly correlated features. For example, radius, perimeter, and area are mathematically related (area = π × radius², perimeter = 2π × radius). The selector keeps only the most important feature from each correlated group, preventing redundancy and improving model interpretability.

## Random Forest Binary Classifier

The `RandomForestBinaryClassifier` implements a Random Forest classifier with automatic hyperparameter tuning via GridSearchCV, specifically designed for binary classification of breast cancer diagnosis (Benign vs Malignant).

### Key Features

- Automatic hyperparameter tuning using GridSearchCV
- Handles class imbalance with `class_weight='balanced'`
- Comprehensive evaluation using F1 (primary), Recall, Precision, and ROC-AUC
- Automatic train/test split with stratification
- Feature importance analysis
- Timestamped model persistence
- Confusion matrix and classification report generation

### Usage Example

```python
from src.classifiers import RandomForestBinaryClassifier, load_cancer_data, FeatureSelector

# Load data
X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')

# Optional: Select important features
selector = FeatureSelector({
    'feature_importance_path': 'outputs/feature_importance.txt',
    'top_n': 10
})
X_selected = selector.fit_transform(X, y)

# Create and train classifier
classifier = RandomForestBinaryClassifier({
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'cv_folds': 5,
    'random_state': 42
})

# Fit with automatic GridSearchCV tuning
classifier.fit(X_selected, y)

# View results
classifier.print_summary()

# Get best hyperparameters
print(classifier.get_best_params())

# Get feature importance
importance = classifier.get_feature_importance()
print(importance.head(10))

# Save model with timestamped filename
saved_path = classifier.save_model()
print(f"Model saved to: {saved_path}")

# Load model later
loaded_classifier = RandomForestBinaryClassifier.load_model(saved_path)
```

### Configuration Parameters

- `n_estimators`: List of values for number of trees (default: [100, 200, 300])
- `max_depth`: List of values for maximum tree depth (default: [10, 20, None])
- `min_samples_split`: List of values for minimum samples to split (default: [2, 5, 10])
- `class_weight`: Strategy for handling imbalance (default: 'balanced')
- `cv_folds`: Number of cross-validation folds (default: 5)
- `test_size`: Proportion of data for testing (default: 0.2)
- `scoring`: Metric for GridSearchCV (default: 'f1' - the northstar metric)
- `random_state`: Random seed for reproducibility (default: 42)
- `n_jobs`: Number of parallel jobs (default: -1 for all cores)

### Evaluation Metrics

- **F1 Score (Primary)**: Harmonic mean of precision and recall, used as the northstar metric for model selection
- **Recall**: Ability to find all malignant cases (critical for cancer diagnosis)
- **Precision**: Accuracy of malignant predictions
- **ROC-AUC**: Overall discriminative ability

### Model Persistence

Models are saved as pickle files with timestamped filenames. By default, both the model and results are saved under the directory specified by `--output-dir` (default: `scripts/`). The model file is saved as `random_forest_model_YYYYMMDD_HHMMSS.pkl` in the output directory. Custom paths can be specified via `--output-dir` or `--model-filename`.

### Class Imbalance Handling

The Wisconsin breast cancer dataset has 357 benign and 212 malignant cases. `class_weight='balanced'` automatically adjusts weights inversely proportional to class frequencies. This ensures the model doesn't bias toward the majority class. Stratified train/test split maintains class distribution.

## Training Pipeline

The training pipeline integrates feature selection and Random Forest classification into a single CLI workflow.

### Prerequisites

1. **Complete Task 2 first**: Run pattern mining to generate feature importance scores

   ```bash
   python scripts/cancer_pattern_mining.py
   ```

2. **Activate virtual environment**: Ensure dependencies are installed

   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate  # Windows
   ```

### Quick Start

```bash
# Train with feature selection (recommended)
python scripts/train_random_forest.py --use-feature-selection --verbose
```

This will:

1. Load the Wisconsin breast cancer dataset
2. Select top 10 features based on pattern mining results
3. Train Random Forest with GridSearchCV (5-fold CV)
4. Evaluate on test set (F1, Recall, Precision, ROC-AUC)
5. Save model and results to the directory specified by `--output-dir` (default: `scripts/`)

### Key Features

- **Intelligent Feature Selection**: Uses Task 2 pattern mining results to select discriminative features
- **Multicollinearity Detection**: Automatically removes redundant features (e.g., keeps only one of radius/perimeter/area)
- **Hyperparameter Tuning**: GridSearchCV with customizable parameter grid
- **Class Imbalance Handling**: Uses `class_weight='balanced'` for fair evaluation
- **Comprehensive Evaluation**: F1 (northstar), Recall, Precision, ROC-AUC, confusion matrix
- **Model Persistence**: Timestamped pickle files for reproducibility

### Virtual Environment Setup

**Important**: Always activate the virtual environment before running the training script.

```bash
# Create virtual environment (first time only)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies (first time only)
python3 -m pip install -e .

# Verify installation
python scripts/train_random_forest.py --help
```

### Configuration Options

See detailed documentation in `docs/random_forest_training.md` for:

- Complete parameter reference
- Usage examples
- Troubleshooting guide
- Best practices
- Output interpretation

### Example Workflow

```bash
# Step 1: Activate virtual environment
source .venv/bin/activate

# Step 2: Run pattern mining (if not done already)
python scripts/cancer_pattern_mining.py

# Step 3: Train classifier with feature selection
python scripts/train_random_forest.py \
    --use-feature-selection \
    --top-n-features 10 \
    --correlation-threshold 0.9 \
    --verbose

# Step 4: Review results
# - Console output shows metrics and feature importance
# - Model saved to scripts/random_forest_model_*.pkl
# - Results summary saved to scripts/training_results_*.txt
```

## Academic Context

This project is developed for **NTU SC4020 Data Analytics & Mining** coursework, emphasizing:

- Algorithm implementation from research literature
- Medical data analysis and clinical interpretation
- Parameter sensitivity and optimization techniques
- Software engineering best practices for research code

---

For detailed implementation guides, API references, and troubleshooting, see the comprehensive documentation in [`docs/`](docs/), including:

- Task 1: Symptom Analysis (`docs/symptom_analysis.md`)
- Task 2: Cancer Pattern Mining (`docs/cancer_pattern_mining.md`)
- Task 3: Random Forest Training (`docs/random_forest_training.md`)
- Processors API (`docs/processors.md`)
- Analysis API (`docs/analysis.md`)
