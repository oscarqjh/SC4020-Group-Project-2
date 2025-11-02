# SC4020 Group Project 2 - Repository Structure

## Project Overview
This is a **Data Analytics & Mining** course project focusing on healthcare data analytics. The project implements pattern mining algorithms (Apriori, PrefixSpan) for analyzing symptom co-occurrence patterns and cancer diagnosis patterns from biomedical datasets.

---

## Repository Tree Structure

```
SC4020-Group-Project-2/
│
├── 📄 Project 2 V3.pdf
│   └── Project requirements and specifications document
│
├── 📄 README.md
│   └── Project overview, setup instructions, and usage guide
│
├── 📄 REPOSITORY_STRUCTURE.md (this file)
│   └── Complete repository structure documentation
│
├── 📄 requirements.txt
│   └── Python package dependencies:
│       • prefixspan==0.5.2 (sequential pattern mining)
│       • pandas==2.3.3 (data manipulation)
│       • numpy==2.3.4 (numerical computing)
│       • scikit-learn==1.7.2 (machine learning)
│       • jupyter==1.1.1 (notebook environment)
│       • mlxtend==0.23.0 (Apriori algorithm)
│       • nltk==3.8.1 (natural language processing)
│       • tqdm==4.66.1 (progress bars)
│       • seaborn==0.13.2 (visualization)
│
├── 📄 pyproject.toml
│   └── Python project configuration (setuptools)
│       • Project metadata
│       • Dependencies
│       • Package build settings
│
├── 📁 data/
│   ├── 📄 dataset.csv
│   │   └── Symptom dataset for Task 1 (symptom co-occurrence analysis)
│   │
│   ├── 📄 symptom_Description.csv
│   │   └── Descriptions for different symptoms
│   │
│   ├── 📄 symptom_precaution.csv
│   │   └── Precautionary measures for symptoms
│   │
│   ├── 📄 Symptom-severity.csv
│   │   └── Severity ratings for symptoms
│   │
│   └── 📁 raw/
│       └── 📄 wisconsin_breast_cancer.csv
│           └── Wisconsin Breast Cancer dataset for Task 2 (cancer pattern mining)
│
├── 📁 src/
│   └── Main source code (Python package)
│   │
│   ├── 📁 processors/
│   │   ├── 📄 __init__.py
│   │   │   └── Package initialization
│   │   │
│   │   ├── 📄 base.py
│   │   │   └── Base processor class (abstract base class)
│   │   │
│   │   ├── 📄 symptom_data_processor.py
│   │   │   └── Task 1: Processes symptom dataset
│   │   │       • Data loading and cleaning
│   │   │       • Transaction generation for Apriori
│   │   │
│   │   └── 📄 sequence_generator.py
│   │       └── Task 2: Generates sequences from cancer data
│   │           • Data discretization (quantile/uniform)
│   │           • Feature ranking and selection
│   │           • Sequence generation for PrefixSpan
│   │
│   └── 📁 analysis/
│       ├── 📄 __init__.py
│       │   └── Package initialization
│       │
│       ├── 📄 base.py
│       │   └── Base analyzer class (abstract base class)
│       │
│       ├── 📄 symptom_pattern_miner.py
│       │   └── Task 1: Apriori algorithm implementation
│       │       • Frequent itemset mining
│       │       • Association rule generation
│       │
│       ├── 📄 pattern_mining.py
│       │   └── Task 2: Sequential pattern mining
│       │       • PrefixSpan algorithm wrapper
│       │       • Pattern ranking and evaluation
│       │       • Discriminative pattern discovery
│       │
│       └── 📄 evaluation.py
│           └── Task 2: Pattern evaluation and sensitivity analysis
│               • Support calculation
│               • Sensitivity to discretization parameters
│               • Pattern stability analysis
│
├── 📁 scripts/
│   └── Command-line executable scripts
│   │
│   ├── 📄 symptom_analysis.py
│   │   └── Task 1 CLI tool
│   │       • Analyzes symptom co-occurrence patterns
│   │       • Uses Apriori algorithm
│   │       • Command-line arguments: --data-path, --output-dir, --min-support, --verbose
│   │
│   └── 📄 cancer_pattern_mining.py
│       └── Task 2 CLI tool
│           • Performs sequential pattern mining on cancer data
│           • Uses PrefixSpan algorithm
│           • Command-line arguments:
│             - --data-path, --output-dir
│             - --discretization-strategy, --n-bins
│             - --min-support, --max-pattern-length, --top-k
│             - --ranking-method, --skip-sensitivity, --verbose
│
├── 📁 notebooks/
│   └── Jupyter notebooks for exploratory analysis
│   │
│   ├── 📁 task1/
│   │   ├── 📄 symptom_analysis.ipynb
│   │   │   └── Task 1 notebook (template/development)
│   │   │
│   │   └── 📄 symptom_analysis_executed.ipynb
│   │       └── Task 1 executed notebook with results
│   │
│   ├── 📁 task2/
│   │   ├── 📄 cancer_pattern_mining.ipynb
│   │   │   └── Task 2 notebook (template/development)
│   │   │
│   │   └── 📄 cancer_pattern_mining_executed.ipynb
│   │       └── Task 2 executed notebook with results
│   │
│   └── 📁 task3/
│       └── (Empty - for future Task 3 implementation)
│
├── 📁 docs/
│   └── Comprehensive documentation
│   │
│   ├── 📄 symptom_analysis.md
│   │   └── Complete guide for Task 1 (symptom analysis)
│   │
│   ├── 📄 cancer_pattern_mining.md
│   │   └── Complete guide for Task 2 (cancer pattern mining)
│   │
│   ├── 📄 processors.md
│   │   └── Documentation for data processing components
│   │
│   └── 📄 analysis.md
│       └── Documentation for analysis and mining algorithms
│
├── 📁 outputs/
│   └── Generated analysis results
│   │
│   ├── 📄 analysis_summary.txt
│   │   └── Summary statistics and results
│   │
│   └── 📄 feature_importance.txt
│       └── Feature importance rankings
│
├── 📁 tests/
│   └── Unit tests for the project
│   │
│   ├── 📄 test_pattern_mining.py
│   │   └── Tests for pattern mining algorithms
│   │
│   └── 📄 test_sequence_generation.py
│       └── Tests for sequence generation logic
│
├── 📁 sc4020_project_2.egg-info/
│   └── Python package metadata (generated by setuptools)
│       • dependency_links.txt
│       • PKG-INFO
│       • requires.txt
│       • SOURCES.txt
│       • top_level.txt
│
└── 📁 venv/
    └── Python virtual environment (local development)
        • bin/ - Executable scripts
        • lib/ - Installed packages
        • include/ - Header files
        • share/ - Shared resources (Jupyter configs, etc.)
        • pyvenv.cfg - Virtual environment configuration
```

---

## Key Components Breakdown

### 1. **Data Processing (`src/processors/`)**
   - **SymptomDataProcessor**: Handles Task 1 symptom data preprocessing
   - **CancerSequenceGenerator**: Handles Task 2 cancer data preprocessing, discretization, and sequence generation

### 2. **Analysis Algorithms (`src/analysis/`)**
   - **SymptomPatternMiner**: Implements Apriori algorithm for Task 1
   - **SequentialPatternAnalyzer**: Implements PrefixSpan for Task 2
   - **SensitivityAnalyzer**: Evaluates pattern stability for Task 2

### 3. **Executable Scripts (`scripts/`)**
   - **symptom_analysis.py**: Task 1 entry point with CLI
   - **cancer_pattern_mining.py**: Task 2 entry point with CLI

### 4. **Notebooks (`notebooks/`)**
   - Interactive Jupyter notebooks for each task
   - Both template and executed versions for reference

### 5. **Documentation (`docs/`)**
   - Detailed guides for each task
   - API documentation for processors and analyzers

### 6. **Data Files (`data/`)**
   - Symptom-related datasets for Task 1
   - Wisconsin Breast Cancer dataset for Task 2

---

## Task Organization

### Task 1: Symptom Analysis
- **Purpose**: Analyze symptom co-occurrence patterns using Apriori algorithm
- **Data**: `data/dataset.csv`, `data/symptom_Description.csv`, `data/symptom_precaution.csv`, `data/Symptom-severity.csv`
- **Script**: `scripts/symptom_analysis.py`
- **Notebook**: `notebooks/task1/symptom_analysis*.ipynb`
- **Processor**: `src/processors/symptom_data_processor.py`
- **Analyzer**: `src/analysis/symptom_pattern_miner.py`

### Task 2: Cancer Pattern Mining
- **Purpose**: Discover discriminative sequential patterns in cancer data using PrefixSpan
- **Data**: `data/raw/wisconsin_breast_cancer.csv`
- **Script**: `scripts/cancer_pattern_mining.py`
- **Notebook**: `notebooks/task2/cancer_pattern_mining*.ipynb`
- **Processor**: `src/processors/sequence_generator.py`
- **Analyzer**: `src/analysis/pattern_mining.py`, `src/analysis/evaluation.py`

### Task 3: Advanced Analytics
- **Status**: Not yet implemented
- **Location**: `notebooks/task3/` (empty)

---

## Usage Workflow

1. **Setup**: Activate virtual environment and install dependencies
2. **Task 1**: Run `python scripts/symptom_analysis.py` or use notebooks
3. **Task 2**: Run `python scripts/cancer_pattern_mining.py` or use notebooks
4. **Results**: Check `outputs/` directory for generated analysis files

---

## Development Notes

- The project uses a modular architecture with clear separation between data processing and analysis
- Abstract base classes (`base.py`) ensure consistent interfaces
- Both CLI scripts and Jupyter notebooks are available for different use cases
- Comprehensive documentation is provided in the `docs/` directory
- Unit tests are available in the `tests/` directory

---

*Last updated: 2025-11-02*

