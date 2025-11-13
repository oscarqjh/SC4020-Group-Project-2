# SC4020-Group-Project-2

NTU SC4020 Data Analytics & Mining AY25 Group Project 2

## Project Overview

This project implements comprehensive healthcare data analytics solutions using biomedical datasets to analyze disease patterns and cancer diagnosis characteristics. The project consists of three main tasks focusing on different aspects of medical data mining and pattern discovery:

- **Task 1**: Symptom Analysis and Pattern Mining
- **Task 2**: Breast Cancer Pattern Mining and Feature Analysis
- **Task 3**: AI-Powered Medical Analysis System with Natural Language Processing

The system combines traditional data mining techniques with modern AI capabilities to provide intelligent medical analysis tools.

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

# For AI-powered medical analysis system (Task 3), install additional dependencies
pip install crewai

# Verify installation (Task 2)
python scripts/cancer_pattern_mining.py --help

# Verify AI system (Task 3)
python app.py --help
```

## Project Structure

```
SC4020-Group-Project-2/
├── README.md                    # This overview file
├── app.py                       # Main AI medical analysis application (Task 3)
├── CLI_USAGE_GUIDE.md          # CLI usage documentation
├── AI_MEDICAL_SYSTEM_SUMMARY.md # AI system implementation summary
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
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── task1/                  # Symptom analysis notebooks
│   └── task2/                  # Cancer pattern mining notebooks
├── data/                        # Data storage
│   ├── dataset.csv             # Symptom data (Task 1)
│   ├── symptom_Description.csv # Symptom descriptions
│   ├── symptom_precaution.csv  # Symptom precautions
│   ├── Symptom-severity.csv    # Symptom severity data
│   └── raw/
│       └── wisconsin_breast_cancer.csv # Cancer data (Task 2)
├── outputs/                     # Analysis results
├── tests/                       # Unit tests
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── [configuration files]      # Environment and setup files
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

### Task 3: AI-Powered Medical Analysis System

#### Interactive Mode

```bash
# Run in interactive mode with full CLI interface
python app.py
```

#### Direct Prompt Mode

```bash
# Symptom analysis
python app.py --prompt "I have a fever, cough, and headache"

# Breast cancer analysis
python app.py --prompt "Patient has tumor radius 12.5, area 490.1, malignant characteristics"

# Get help
python app.py --help
```

#### Key Features

- **AI-Powered Symptom Extraction**: Natural language processing for symptom identification
- **Breast Cancer Feature Analysis**: Intelligent tumor measurement and characteristic extraction using Random Forest models
- **Dynamic Medical Analysis**: Adapts to medical terminology variations
- **Professional Medical Disclaimers**: Comprehensive safety recommendations
- **Symptom Follow-up Suggestions**: When fewer than five symptoms are detected, the system now proposes additional high-support symptom combinations (mined from `outputs/task3_tool1/disease_frequent_itemsets.pkl`) to guide the user until the classifier reaches its optimal context window.

> **Model coverage note**: The disease classifier was trained on records with 3–17 symptoms and reaches stable confidence once at least **5** symptoms are confirmed (≈75% of the dataset). Providing fewer symptoms will trigger follow-up questions so the agent can refine the prediction safely.

> **Note**: The breast cancer analysis tool utilizes Random Forest machine learning models. For detailed information about model training, parameters, and performance metrics, see [`docs/random_forest_training.md`](docs/random_forest_training.md).

## Random Forest Binary Classifier

The `RandomForestBinaryClassifier` provides a fully-tuned ensemble model for breast cancer diagnosis with automated GridSearchCV optimization.

### Key Features

- **Robust Ensemble**: Random Forest with configurable tree depth, estimators, and split criteria
- **Hyperparameter Tuning**: Automated GridSearchCV across `n_estimators`, `max_depth`, and `min_samples_split`
- **Class Imbalance Handling**: Defaults to `class_weight='balanced'`
- **Comprehensive Evaluation**: F1 (primary), Recall, Precision, ROC-AUC, confusion matrix, and classification report
- **Feature Importance**: Ranks diagnostic features using mean decrease impurity
- **Model Persistence**: Timestamped pickle files with full metadata

### Usage Example

```python
from src.classifiers import RandomForestBinaryClassifier, load_cancer_data, FeatureSelector

X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')

selector = FeatureSelector({
    'feature_importance_path': 'outputs/feature_importance.txt',
    'top_n': 12
})
X_selected = selector.fit_transform(X, y)

classifier = RandomForestBinaryClassifier({
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'cv_folds': 5,
    'random_state': 42
})

classifier.fit(X_selected, y)
classifier.print_summary()
print(classifier.get_best_params())

importance = classifier.get_feature_importance()
print(importance.head())

model_path = classifier.save_model()
loaded_classifier = RandomForestBinaryClassifier.load_model(model_path)
predictions = loaded_classifier.predict(X_selected)
```

### Configuration Parameters

- `n_estimators`: List of tree counts (default `[100, 200, 300]`)
- `max_depth`: List of tree depths (default `[10, 20, None]`)
- `min_samples_split`: List for minimum split counts (default `[2, 5, 10]`)
- `class_weight`: Imbalance strategy (default `'balanced'`)
- `cv_folds`: Cross-validation folds (default `5`)
- `test_size`: Held-out proportion (default `0.2`)
- `scoring`: GridSearchCV metric (default `'f1'`)
- `random_state`: Seed for reproducibility (default `42`)
- `n_jobs`: Parallel workers (default `-1`)

### Feature Importance

- Extract via `get_feature_importance()` returning a sorted DataFrame
- Higher scores indicate greater influence in malignant/benign discrimination
- Combine with Task 2 results for richer interpretability

### Model Persistence

- Models saved under `scripts/random_forest_model_YYYYMMDD_HHMMSS.pkl`
- Compatible with `RandomForestBinaryClassifier.load_model(path)`
- Include call to `.save_model('custom/path.pkl')` for custom locations

### SVM Binary Classifier

The `SVMBinaryClassifier` implements a Support Vector Machine with automatic feature scaling and hyperparameter tuning for breast cancer diagnosis.

**Key Features:**

- **Automatic Feature Scaling**: StandardScaler normalization ensures equitable feature contribution
- **Hyperparameter Tuning**: GridSearchCV over `C`, `kernel`, and `gamma`
- **Class Imbalance Handling**: Uses `class_weight='balanced'` by default
- **Comprehensive Evaluation**: F1 (primary), Recall, Precision, ROC-AUC with train/test metrics
- **Baseline Comparison**: Optional side-by-side metrics against Random Forest via `--compare-rf`
- **Support Vector Analysis**: Provides total support vectors and per-class counts
- **Feature Importance**: Available when the optimal kernel is linear
- **Model Persistence**: Timestamped pickle files containing both SVC model and scaler

**Usage Example:**

```python
from src.classifiers import SVMBinaryClassifier, load_cancer_data, FeatureSelector

# Load dataset
X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')

# Optional feature selection
selector = FeatureSelector({
    'feature_importance_path': 'outputs/feature_importance.txt',
    'top_n': 10
})
X_selected = selector.fit_transform(X, y)

classifier = SVMBinaryClassifier({
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'],
    'cv_folds': 5,
    'random_state': 42
})

classifier.fit(X_selected, y)
classifier.print_summary()

print(classifier.get_best_params())
print(classifier.get_support_vectors())

importance = classifier.get_feature_importance()
if importance is not None:
    print(importance.head(10))

saved_path = classifier.save_model()
loaded_classifier = SVMBinaryClassifier.load_model(saved_path)

predictions = loaded_classifier.predict(X_selected)
```

**Configuration Parameters:**

- `C`: Regularization values (default `[0.1, 1, 10]`)
- `kernel`: Kernel options (default `['rbf', 'linear']`)
- `gamma`: Kernel coefficients (default `['scale', 'auto']`)
- `class_weight`: Imbalance strategy (default `'balanced'`)
- `probability`: Enable probability estimates (default `True`)
- `cv_folds`: Cross-validation folds (default `5`)
- `test_size`: Held-out proportion (default `0.2`)
- `scoring`: GridSearchCV metric (default `'f1'`)
- `random_state`: Seed for reproducibility (default `42`)
- `n_jobs`: Parallel workers for GridSearchCV (default `-1`)

**Feature Scaling:**

- SVM uses StandardScaler internally; features are centered to mean 0 and variance 1
- The fitted scaler is persisted with the model for consistent preprocessing
- Predictions automatically apply the stored scaler, preventing data leakage

**Support Vectors:**

- Critical samples lying on or inside the margin boundaries
- Typically 10–30% of the training data depending on kernel choice
- Investigate via `get_support_vectors()` to understand boundary complexity

**Feature Importance:**

- Available only for linear kernel via coefficient magnitudes
- Returns `None` for non-linear kernels; combine with Task 2 insights if needed

**Model Persistence:**

- Saved under `scripts/svm_model_YYYYMMDD_HHMMSS.pkl` by default
- Includes both the SVC estimator and fitted StandardScaler
- Use `save_model('custom_filename.pkl')` to specify alternate paths

## Training Pipeline

```bash
# Random Forest training with default parameters
python scripts/train_random_forest.py

# Random Forest with feature selection and verbose logging
python scripts/train_random_forest.py \
    --use-feature-selection \
    --top-n-features 10 \
    --verbose

# Random Forest advanced tuning
python scripts/train_random_forest.py \
    --use-feature-selection \
    --n-estimators 100 200 300 \
    --max-depth 10 20 None \
    --min-samples-split 2 5 \
    --cv-folds 10 \
    --verbose
```

### SVM Classifier Training

```bash
# Basic training with default parameters
python scripts/train_svm.py

# Training with feature selection (recommended)
python scripts/train_svm.py \
    --use-feature-selection \
    --top-n-features 10 \
    --verbose

# Advanced training with custom hyperparameters
python scripts/train_svm.py \
    --use-feature-selection \
    --C 0.1 1 10 100 \
    --kernel rbf linear \
    --gamma scale auto 0.01 \
    --cv-folds 10 \
    --verbose

# Training with Random Forest comparison
python scripts/train_svm.py \
    --use-feature-selection \
    --top-n-features 10 \
    --compare-rf
```

`--compare-rf` fits a lightweight Random Forest on the identical train/test split and prints a compact table comparing F1, Recall, Precision, and ROC-AUC for both models.

**Key Differences from Random Forest:**

- **Feature Scaling**: SVM automatically applies StandardScaler; Random Forest is scale-invariant
- **Hyperparameters**: SVM tunes `C`, `kernel`, `gamma`; Random Forest tunes tree-related parameters
- **Training Time**: SVM can be slower on larger datasets due to quadratic complexity
- **Feature Importance**: Linear SVM offers coefficient-based importance; Random Forest provides it for all runs
- **Support Vectors**: Unique to SVM, offering insight into margin-critical samples

**When to Use SVM:**

- High-dimensional data with potential non-linear boundaries
- Need for memory-efficient models focused on critical samples
- Requirement for flexible kernel-based decision boundaries

**When to Use Random Forest:**

- Large datasets or when training speed matters
- Need for always-available feature importance
- Preference for models less sensitive to hyperparameter selection

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
- Task 3: SVM Training (`docs/svm_training.md`)
- Processors API (`docs/processors.md`)
- Analysis API (`docs/analysis.md`)
