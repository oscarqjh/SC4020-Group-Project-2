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
│   ├── processors.md           # Data processing components
│   ├── analysis.md             # Analysis and mining algorithms
│   └── task3_implementation_plan.md # Task 3 implementation guide
├── src/                         # Source code (All tasks implemented)
│   ├── processors/             # Data preprocessing pipeline (Task 1 & 2)
│   ├── analysis/               # Pattern mining and evaluation (Task 1 & 2)
│   └── crew/                   # AI medical analysis system (Task 3)
│       ├── __init__.py         # Package initialization
│       ├── base.py             # Base classes and interfaces
│       ├── agent.py            # Medical analysis AI agents
│       ├── tools.py            # Medical analysis tools
│       ├── cli.py              # Interactive command-line interface
│       ├── crew_manager.py     # Crew coordination system
│       ├── symptom_extractor.py # AI-powered symptom extraction
│       └── breast_cancer_extractor.py # AI-powered cancer feature extraction
├── scripts/                     # Executable scripts
│   ├── symptom_analysis.py     # Task 1 CLI tool
│   └── cancer_pattern_mining.py # Task 2 CLI tool
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

## Academic Context

This project is developed for **NTU SC4020 Data Analytics & Mining** coursework, emphasizing:

- Algorithm implementation from research literature
- Medical data analysis and clinical interpretation
- Parameter sensitivity and optimization techniques
- Software engineering best practices for research code
- **AI-Powered Healthcare Analytics**: Modern machine learning approaches for medical data
- **Natural Language Processing**: Advanced text analysis for medical terminology
- **Intelligent Agent Systems**: Multi-agent coordination for complex analysis tasks

## Quick Reference

### Common Commands

```bash
# Traditional data mining (Tasks 1 & 2)
python scripts/symptom_analysis.py
python scripts/cancer_pattern_mining.py

# AI-powered analysis (Task 3)
python app.py --prompt "your medical query"
python app.py  # Interactive mode

# Documentation
python app.py --help
```

### Key Files

- `app.py` - Main AI medical analysis application
- `CLI_USAGE_GUIDE.md` - Detailed CLI usage instructions
- `AI_MEDICAL_SYSTEM_SUMMARY.md` - Complete AI system documentation
- `src/crew/` - AI system source code
- `docs/` - Comprehensive project documentation

---

For detailed implementation guides, API references, and troubleshooting, see the comprehensive documentation in [`docs/`](docs/).
