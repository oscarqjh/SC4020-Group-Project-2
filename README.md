# SC4020-Group-Project-2
NTU SC4020 Data Analytics & Mining AY25 Group Project 2

## Project Overview

This project implements comprehensive healthcare data analytics solutions using biomedical datasets to analyze disease patterns and cancer diagnosis characteristics. The project consists of three main tasks focusing on different aspects of medical data mining and pattern discovery.

## Environment Setup

### Create and Activate Environment

```bash
# Create the virtual environment 
uv venv -p 3.11

# Activate the environment

# On macOS / Linux (bash/zsh)
source .venv/bin/activate

# On Windows (Command Prompt)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
# Install the project in development mode
uv pip install -e .

# Verify installation (Task 2)
python scripts/cancer_pattern_mining.py --help
```

## Project Structure

```
SC4020-Group-Project-2/
├── README.md                    # This overview file
├── docs/                        # Comprehensive documentation
│   ├── README.md               # Documentation navigation
│   ├── cancer_pattern_mining.md # Task 2 complete guide
│   ├── processors.md           # Data processing components
│   └── analysis.md             # Analysis and mining algorithms
├── src/                         # Source code (Task 2 implemented)
│   ├── processors/             # Data preprocessing pipeline
│   └── analysis/               # Pattern mining and evaluation
├── scripts/                     # Executable scripts
│   └── cancer_pattern_mining.py # Task 2 CLI tool
├── datasets/                    # Data storage
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
# Will be available upon implementation
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
