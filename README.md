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

**Important:** For this project, `requirements.txt` is the **authoritative dependency specification**. Use it for all environment setup.

```bash
# Install the project in development mode using pip
python3 -m pip install -e .

# Install all dependencies from requirements.txt
python3 -m pip install -r requirements.txt

# Note: pip install -e . is NOT recommended as pyproject.toml dependencies 
# may be outdated compared to requirements.txt specifications

# Verify installation (Task 2)
python scripts/cancer_pattern_mining.py --help
```

**Rationale:** `requirements.txt` contains the complete, verified, and secure dependency specifications for all phases of the project. `pyproject.toml` serves only as a minimal package configuration for development workflows and should not be used as the source of truth for dependency management.

## Project Structure

```
SC4020-Group-Project-2/
├── README.md                    # This overview file
├── docs/                        # Comprehensive documentation
│   ├── README.md               # Documentation navigation
│   ├── symptom_analysis.md     # Task 1 complete guide
│   ├── cancer_pattern_mining.md # Task 2 complete guide
│   ├── processors.md           # Data processing components
│   └── analysis.md             # Analysis and mining algorithms
├── src/                         # Source code (Task 1 & 2 implemented)
│   ├── processors/             # Data preprocessing pipeline
│   └── analysis/               # Pattern mining and evaluation
├── scripts/                     # Executable scripts
│   ├── symptom_analysis.py     # Task 1 CLI tool
│   └── cancer_pattern_mining.py # Task 2 CLI tool
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

### Task 3: Advanced Analytics

```bash
# Will be available upon implementation

```

## Academic Context

This project is developed for **NTU SC4020 Data Analytics & Mining** coursework, emphasizing:

- Algorithm implementation from research literature
- Medical data analysis and clinical interpretation
- Parameter sensitivity and optimization techniques
- Software engineering best practices for research code

---

For detailed implementation guides, API references, and troubleshooting, see the comprehensive documentation in [`docs/`](docs/).
