# Symptom Co-occurrence Analysis (`symptom_analysis.py`)

## Overview

This script performs symptom co-occurrence analysis on the Disease Symptom dataset using the Apriori algorithm. It identifies frequent combinations of symptoms that appear together in disease profiles, providing insights into symptom patterns.

## Features

- **Data Cleaning**: Automatically cleans and normalizes symptom names.
- **Synonym Handling**: Uses WordNet to group synonymous symptoms for more accurate analysis.
- **Apriori Implementation**: Employs the `mlxtend` library for an efficient Apriori implementation.
- **Customizable Parameters**: Allows configuration of minimum support and other parameters.
- **Output Generation**: Saves frequent itemsets and association rules to CSV files.

## Command-Line Interface

```bash
python scripts/symptom_analysis.py [OPTIONS]
```

### Options

- `--data-path PATH`: Path to the symptom dataset CSV file (default: `data/dataset.csv`).
- `--output-dir DIR`: Directory to save analysis results (default: `outputs`).
- `--min-support FLOAT`: Minimum support threshold for the Apriori algorithm (default: 0.01).
- `--verbose`: Enable verbose output during execution.
- `--help`: Show the help message and exit.

## Quick Start

To run the analysis with default settings:

```bash
python scripts/symptom_analysis.py --verbose
```

This command will:

1. Load the symptom dataset from `data/dataset.csv`.
2. Clean and normalize the symptom data.
3. Apply the Apriori algorithm to find frequent itemsets.
4. Generate association rules.
5. Save the results to the `outputs/` directory.

## Usage Examples

### Basic Analysis

```bash
# Run with default parameters and verbose output
python scripts/symptom_analysis.py --verbose
```

### Custom Parameters

```bash
# Use a higher minimum support threshold and a custom data path
python scripts/symptom_analysis.py \
    --data-path /path/to/your/symptom_data.csv \
    --min-support 0.05 \
    --verbose
```

## Output Files

The script generates two main output files in the specified output directory:

1.  **`frequent_itemsets.csv`**: Contains the sets of symptoms that frequently co-occur, along with their support values.
2.  **`association_rules.csv`**: Contains association rules derived from the frequent itemsets, showing relationships between symptoms (e.g., if symptom A is present, symptom B is also likely present).

## Algorithm Overview

1.  **Data Loading**: The script loads the raw symptom data from the provided CSV file.
2.  **Data Cleaning**: Each symptom is cleaned to remove leading/trailing whitespace and inconsistent formatting (e.g., underscores).
3.  **Synonym Normalization**: Symptoms are normalized to a canonical form using `nltk` and WordNet. For example, 'pyrexia' might be normalized to 'fever'. This ensures that different terms for the same symptom are treated as one.
4.  **Transaction Encoding**: The data is transformed into a transactional format, where each row (representing a disease case) is a "basket" of symptoms.
5.  **Apriori Algorithm**: The Apriori algorithm is applied to the transactional data to find itemsets (combinations of symptoms) that meet the `min-support` threshold.
6.  **Association Rule Generation**: From the frequent itemsets, association rules are generated to identify significant co-occurrence patterns.

This process provides a clear and quantitative way to understand which symptoms tend to appear together, which can be valuable for clinical insights and further medical research.

## Interpreting the Results

The analysis of the association rules provides valuable insights into how different symptoms co-occur. The key metrics to focus on are **confidence** and **lift**.

- **Confidence** measures the reliability of a rule. A high confidence in the rule "IF {Symptom A} THEN {Symptom B}" means that patients with Symptom A are very likely to also have Symptom B.
- **Lift** measures the importance of a rule. A lift value greater than 1 indicates that the symptoms appear together more frequently than would be expected by chance. The higher the lift, the stronger the association.

### Key Findings from the Analysis

The results from the run with `min_support=0.02` reveal several strong patterns, particularly centered around 'abdominal pain'.

1.  **Strong Predictors for Abdominal Pain:** The analysis identified several symptoms that are highly predictive of 'abdominal pain'. The rules below show extremely high confidence (95-100%) and a strong lift (>4.5), indicating a significant, non-random relationship.

    - `acute liver failure -> abdominal pain` (Confidence: 100%)
    - `coma -> abdominal pain` (Confidence: 100%)
    - `dark urine -> abdominal pain` (Confidence: 96%)
    - `distention of abdomen -> abdominal pain` (Confidence: 95%)

    **Conclusion:** In this dataset, when a patient presents with a severe and specific symptom like 'acute liver failure' or 'coma', the presence of the more general symptom 'abdominal pain' is almost certain. A lift of over 4.5 signifies that this co-occurrence is over 4.5 times more likely than random chance, highlighting a powerful clinical pattern.

2.  **Common Symptom Associations:** The analysis also uncovered relationships between more common, less severe symptoms.

    - `abdominal pain -> fatigue` (Confidence: 53%, Lift: 1.36)

    **Conclusion:** While this rule's confidence and lift are lower, it is still significant. It suggests that a patient with 'abdominal pain' is 36% more likely to also experience 'fatigue' compared to a random patient. This represents a weaker but noteworthy association between two of the most frequently occurring symptoms.

These findings demonstrate the utility of association rule mining in uncovering both strong, specific symptom relationships and weaker, more general patterns within a clinical dataset.
