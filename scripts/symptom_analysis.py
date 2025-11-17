import argparse
import warnings
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processors.symptom_data_processor import SymptomDataProcessor
from src.analysis.symptom_pattern_miner import SymptomPatternMiner

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parses command-line arguments for the symptom analysis script."""
    parser = argparse.ArgumentParser(
        description="Symptom Co-occurrence Pattern Analysis using Apriori Algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/symptom_analysis.py

  # Use custom support threshold
  python scripts/symptom_analysis.py --min-support 0.05

  # Save visualizations
  python scripts/symptom_analysis.py --save-plots

  # Skip visualizations (faster for batch processing)
  python scripts/symptom_analysis.py --no-plots
        """
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/dataset.csv',
        help='Path to the symptom dataset CSV file (default: data/dataset.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save the analysis results (default: outputs)'
    )
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.03,
        help='Minimum support threshold for the Apriori algorithm (default: 0.03)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for association rules (default: 0.5)'
    )
    parser.add_argument(
        '--min-transactions',
        type=int,
        default=10,
        help='Minimum number of transactions required (default: 10)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save visualization plots to output directory'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (faster execution)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()


def find_diseases_with_itemset(itemset: Set, disease_to_transaction: Dict[str, List[str]]) -> List[str]:
    """
    Find all diseases that contain all symptoms in the given itemset.
    
    Args:
        itemset: A set or list of symptoms
        disease_to_transaction: Dictionary mapping disease names to their symptom lists
        
    Returns:
        List of disease names that contain all symptoms in the itemset
    """
    itemset_set = set(itemset) if isinstance(itemset, (list, frozenset)) else itemset
    matching_diseases = []
    for disease, symptoms in disease_to_transaction.items():
        symptom_set = set(symptoms)
        if itemset_set.issubset(symptom_set):
            matching_diseases.append(disease)
    return matching_diseases


def create_disease_mapping(df_raw: pd.DataFrame, processor: SymptomDataProcessor) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Create a mapping of disease names to their symptom lists.
    
    Args:
        df_raw: Raw DataFrame from CSV
        processor: SymptomDataProcessor instance
        
    Returns:
        Tuple of (disease_to_transaction dict, disease_names list)
    """
    disease_symptoms = {}
    for index, row in df_raw.iterrows():
        disease = row['Disease'] if pd.notna(row['Disease']) else None
        if disease:
            symptom_cols = [col for col in df_raw.columns if 'Symptom' in col]
            if disease not in disease_symptoms:
                disease_symptoms[disease] = set()
            for col in symptom_cols:
                symptom = processor._clean_symptom(row[col])
                if symptom:
                    normalized_symptom = processor._normalize_symptom(symptom)
                    disease_symptoms[disease].add(normalized_symptom)
    
    disease_to_transaction = {disease: list(symptoms) for disease, symptoms in disease_symptoms.items()}
    disease_names = list(disease_to_transaction.keys())
    
    return disease_to_transaction, disease_names


def analyze_frequent_itemsets(frequent_itemsets: pd.DataFrame, output_dir: Path, save_plots: bool = False, no_plots: bool = False) -> None:
    """
    Analyze and visualize frequent itemsets.
    
    Args:
        frequent_itemsets: DataFrame with frequent itemsets
        output_dir: Directory to save outputs
        save_plots: Whether to save plots
        no_plots: Whether to skip plots
    """
    print("\n" + "="*60)
    print("FREQUENT ITEMSETS ANALYSIS")
    print("="*60)
    
    # Add itemset size column
    frequent_itemsets['itemset_size'] = frequent_itemsets['itemsets'].apply(len)
    size_counts = frequent_itemsets.groupby('itemset_size').size()
    
    print(f"\nItemset size distribution:")
    for size, count in size_counts.items():
        print(f"  Size {size}: {count} itemsets")
    
    # Plot itemset size distribution
    if not no_plots:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(size_counts.index, size_counts.values, color='skyblue')
        plt.xticks(size_counts.index)
        plt.bar_label(bars, padding=3)
        plt.xlabel('Itemset Size')
        plt.ylabel('Count')
        plt.title('Distribution of Itemset Sizes')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_dir / 'itemset_size_distribution.png', dpi=300, bbox_inches='tight')
            print(f"  Saved plot to: {output_dir / 'itemset_size_distribution.png'}")
        else:
            plt.show()
        plt.close()
    
    # Top 10 most frequent single symptoms
    print(f"\nTop 10 Most Frequent Single Symptoms:")
    single_symptoms = frequent_itemsets[frequent_itemsets['itemset_size'] == 1].nlargest(10, 'support')
    for idx, row in single_symptoms.iterrows():
        symptom = list(row['itemsets'])[0]
        print(f"  {symptom}: {row['support']:.4f} ({row['support']*100:.2f}% of diseases)")
    
    # Top 10 most frequent symptom combinations (size == 3)
    # Focusing on size 3 as it represents meaningful symptom triads
    print(f"\nTop 10 Most Frequent Symptom Combinations (size == 3):")
    multi_symptoms = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(10, 'support')
    for idx, row in multi_symptoms.iterrows():
        symptoms = ', '.join(list(row['itemsets']))
        print(f"  {{{symptoms}}}: {row['support']:.4f} ({row['support']*100:.2f}% of diseases)")
    
    # Plot top 10 combinations (size == 3)
    if not no_plots:
        top_itemsets = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(10, 'support').copy()
        top_itemsets['itemsets_str'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='support', y='itemsets_str', data=top_itemsets, palette='viridis')
        plt.title('Top 10 Most Frequent Symptom Combinations (== 3 symptoms)')
        plt.xlabel('Support')
        plt.ylabel('Symptom Sets')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_dir / 'top_symptom_combinations.png', dpi=300, bbox_inches='tight')
            print(f"  Saved plot to: {output_dir / 'top_symptom_combinations.png'}")
        else:
            plt.show()
        plt.close()
        
        print("\nTop 10 Most Frequent Symptom Combinations ({Symptoms Combination}:support):")
        for i, (idx, row) in enumerate(top_itemsets.iterrows(), 1):
            symptoms = row['itemsets_str']
            print(f"{i}. {{{symptoms}}}: {row['support']:.4f} ({row['support']*100:.2f}% of diseases)")


def analyze_association_rules(rules: pd.DataFrame) -> None:
    """
    Analyze and display association rules.
    
    Association Rule Metrics Explanation:
    - antecedents: The "IF" part of the rule (symptom set A)
    - consequents: The "THEN" part of the rule (symptom set B)
    - support: How often A and B appear together in the dataset
    - confidence: The rule's reliability - probability of B given A
    - lift: The rule's importance - how much more likely B is when A is present (>1 is good)
    - leverage: Difference between observed co-occurrence and expected if independent
    - conviction: Measure of rule's implication strength
    - zhangs_metric: Association measure from -1 (negative) to +1 (positive correlation)
    - kulczynski: Symmetric measure of how strongly two symptoms are related
    
    Args:
        rules: DataFrame with association rules
    """
    print(f"\nTotal rules generated: {len(rules)}")
    
    print(f"\nTop 10 Rules by Confidence:")
    top_conf = rules.nlargest(10, 'confidence')
    for idx, row in top_conf.iterrows():
        ante = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        print(f"  IF {{{ante}}} THEN {{{cons}}}")
        print(f"     Confidence: {row['confidence']:.4f}, Lift: {row['lift']:.4f}, Support: {row['support']:.4f}")
    
    print(f"\nTop 10 Rules by Lift (strongest associations):")
    top_lift = rules.nlargest(10, 'lift')
    for idx, row in top_lift.iterrows():
        ante = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        print(f"  IF {{{ante}}} THEN {{{cons}}}")
        print(f"     Lift: {row['lift']:.4f}, Confidence: {row['confidence']:.4f}, Support: {row['support']:.4f}")


def analyze_disease_level_patterns(
    frequent_itemsets: pd.DataFrame,
    disease_to_transaction: Dict[str, List[str]],
    disease_names: List[str],
    output_dir: Path,
    save_plots: bool = False,
    no_plots: bool = False
) -> None:
    """
    Analyze frequent itemsets at the disease level.
    
    Args:
        frequent_itemsets: DataFrame with frequent itemsets
        disease_to_transaction: Dictionary mapping diseases to symptoms
        disease_names: List of disease names
        output_dir: Directory to save outputs
        save_plots: Whether to save plots
    """
    print("\n" + "="*60)
    print("DISEASE-LEVEL ANALYSIS: Mapping Frequent Itemsets to Diseases")
    print("="*60)
    
    # Analyze top 10 most frequent symptom combinations (size == 3)
    # Focusing on size 3 as it represents meaningful symptom triads
    top_itemsets = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(10, 'support').copy()
    
    print("\nTop 10 Most Frequent Symptom Combinations and Their Associated Diseases:")
    print("-" * 60)
    
    for idx, row in top_itemsets.iterrows():
        itemset = row['itemsets']
        support = row['support']
        diseases = find_diseases_with_itemset(itemset, disease_to_transaction)
        
        symptoms_str = ', '.join(list(itemset))
        num_diseases = len(diseases)
        support_pct = support * 100
        
        print(f"\nItemset: {{{symptoms_str}}}")
        print(f"  Support: {support:.4f} ({support_pct:.2f}% of diseases)")
        print(f"  Appears in {num_diseases} disease(s): {', '.join(diseases)}")
    
    # Count how many top frequent itemsets each disease contains
    disease_itemset_counts = {disease: 0 for disease in disease_names}
    disease_itemset_details = {disease: [] for disease in disease_names}
    
    # Focus on size == 3 itemsets for disease-level analysis
    top_itemsets_list = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(20, 'support')
    
    for idx, row in top_itemsets_list.iterrows():
        itemset = row['itemsets']
        support = row['support']
        diseases = find_diseases_with_itemset(itemset, disease_to_transaction)
        
        for disease in diseases:
            disease_itemset_counts[disease] += 1
            disease_itemset_details[disease].append({
                'itemset': list(itemset),
                'support': support
            })
    
    # Sort diseases by number of frequent itemsets
    sorted_diseases = sorted(disease_itemset_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("DISEASES WITH MOST FREQUENT SYMPTOM PATTERNS")
    print("="*60)
    print("\nTop 10 diseases containing the most frequent symptom combinations:")
    print("-" * 60)
    
    for i, (disease, count) in enumerate(sorted_diseases[:10], 1):
        print(f"\n{i}. {disease}")
        print(f"   Contains {count} of the top 20 frequent symptom combinations")
        print(f"   Number of symptoms: {len(disease_to_transaction[disease])}")
        
        # Show top 3 itemsets for this disease
        top_for_disease = sorted(disease_itemset_details[disease], 
                                 key=lambda x: x['support'], reverse=True)[:3]
        print(f"   Top symptom combinations:")
        for j, itemset_info in enumerate(top_for_disease, 1):
            itemset_str = ', '.join(itemset_info['itemset'])
            support_pct = itemset_info['support'] * 100
            print(f"      {j}. {{{itemset_str}}} (support: {itemset_info['support']:.4f}, {support_pct:.2f}% of diseases)")
    
    # Create heatmap
    if not no_plots:
        top_10_itemsets = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(10, 'support')
        top_10_diseases = [disease for disease, _ in sorted_diseases[:10]]
        
        # Create a matrix: diseases (rows) x itemsets (columns)
        heatmap_data = []
        
        for idx, row in top_10_itemsets.iterrows():
            itemset = row['itemsets']
            matching_diseases = find_diseases_with_itemset(itemset, disease_to_transaction)
            
            row_data = []
            for disease in top_10_diseases:
                row_data.append(1 if disease in matching_diseases else 0)
            heatmap_data.append(row_data)
        
        # Transpose for better visualization
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[', '.join(list(row['itemsets']))[:30] + '...' 
                   if len(', '.join(list(row['itemsets']))) > 30 
                   else ', '.join(list(row['itemsets']))
                   for _, row in top_10_itemsets.iterrows()],
            columns=top_10_diseases
        )
        
        plt.figure(figsize=(14, 8))
        binary_cmap = ListedColormap(['#FFFFCC', '#800026'])
        sns.heatmap(heatmap_df.T, annot=False, cmap=binary_cmap, 
                   cbar_kws={'ticks': [0, 1], 'label': 'Itemset Present'},
                   linewidths=0.5, linecolor='gray')
        plt.title('Disease-Symptom Pattern Co-occurrence Heatmap\n(Top 10 Diseases vs Top 10 Frequent Symptom Combinations)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Frequent Symptom Combinations', fontsize=12)
        plt.ylabel('Diseases', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_dir / 'disease_symptom_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"\nSaved heatmap to: {output_dir / 'disease_symptom_heatmap.png'}")
        else:
            plt.show()
        plt.close()
        
        print("\nHeatmap Interpretation:")
        print("- Yellow/Red cells indicate that the disease contains all symptoms in that combination")
        print("- This visualization shows which diseases share common symptom patterns")


def export_disease_itemsets(
    frequent_itemsets: pd.DataFrame,
    disease_to_transaction: Dict[str, List[str]],
    output_dir: Path,
    save_plots: bool = False,
    no_plots: bool = False
) -> pd.DataFrame:
    """
    Export disease-specific frequent itemsets (top 3 per disease).
    
    This function filters itemsets to size == 3 (symptom triads) with support > 0.05,
    then maps them to diseases and keeps the top 3 highest support itemsets per disease.
    This filtered dataset is used for generating features in machine learning models.
    
    Args:
        frequent_itemsets: DataFrame with frequent itemsets
        disease_to_transaction: Dictionary mapping diseases to symptoms
        output_dir: Directory to save outputs
        save_plots: Whether to save plots
        no_plots: Whether to skip plots
        
    Returns:
        DataFrame with disease itemsets
    """
    # Filter frequent itemsets: size == 3 and support > 0.05
    # Focusing on size 3 as it represents meaningful symptom triads
    filtered_itemsets = frequent_itemsets[
        (frequent_itemsets['itemset_size'] == 3) & 
        (frequent_itemsets['support'] > 0.05)
    ].copy()
    
    print(f"\nFiltered to {len(filtered_itemsets)} itemsets (size == 3, support > 0.05)")
    print(f"Support range: {filtered_itemsets['support'].min():.4f} to {filtered_itemsets['support'].max():.4f}")
    
    # Build a list of records: {itemset, support, disease}
    disease_itemsets_records = []
    
    for disease, symptoms in disease_to_transaction.items():
        disease_symptom_set = set(symptoms)
        
        # Find all filtered itemsets that are subsets of this disease's symptoms
        for idx, row in filtered_itemsets.iterrows():
            itemset = row['itemsets']
            support = row['support']
            
            # Check if itemset is a subset of disease symptoms
            if itemset.issubset(disease_symptom_set):
                disease_itemsets_records.append({
                    'itemset': itemset,
                    'support': support,
                    'disease': disease
                })
    
    # Create dataframe
    disease_itemsets_df = pd.DataFrame(disease_itemsets_records)
    
    print(f"\nTotal records before filtering: {len(disease_itemsets_df)}")
    print(f"Unique diseases: {disease_itemsets_df['disease'].nunique()}")
    print(f"Unique itemsets: {disease_itemsets_df['itemset'].nunique()}")
    
    # Keep only top 3 highest support itemsets per disease
    disease_itemsets_df = disease_itemsets_df.groupby('disease', group_keys=False).apply(
        lambda x: x.nlargest(3, 'support')
    ).reset_index(drop=True)
    
    print(f"\nTotal records after keeping top 3 per disease: {len(disease_itemsets_df)}")
    print(f"Unique diseases with itemsets: {disease_itemsets_df['disease'].nunique()}")
    
    # Save to pickle file
    output_path = output_dir / 'disease_frequent_itemsets.pkl'
    disease_itemsets_df.to_pickle(output_path)
    print(f"\nExported to: {output_path}")
    
    # Visualize distribution
    if not no_plots:
        itemsets_per_disease_full = pd.DataFrame(disease_itemsets_records).groupby('disease').size().sort_values(ascending=False)
        itemsets_per_disease_top15 = itemsets_per_disease_full.head(15)
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(range(len(itemsets_per_disease_top15)), itemsets_per_disease_top15.values, color='steelblue')
        plt.yticks(range(len(itemsets_per_disease_top15)), itemsets_per_disease_top15.index)
        plt.xlabel('Number of Frequent Itemsets (Before Filtering)', fontsize=12)
        plt.ylabel('Disease', fontsize=12)
        plt.title('Top 15 Diseases by Number of Frequent Symptom Itemsets (Size == 3, Support > 0.05)', 
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                     f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(output_dir / 'disease_itemsets_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {output_dir / 'disease_itemsets_distribution.png'}")
        else:
            plt.show()
        plt.close()
    
    return disease_itemsets_df


def print_summary(
    transactions: List[List[str]],
    frequent_itemsets: pd.DataFrame,
    rules: pd.DataFrame,
    disease_to_transaction: Dict[str, List[str]],
    disease_itemset_counts: Dict[str, int],
    min_confidence: float
) -> None:
    """
    Print summary statistics.
    
    Args:
        transactions: List of transactions
        frequent_itemsets: DataFrame with frequent itemsets
        rules: DataFrame with association rules
        disease_to_transaction: Dictionary mapping diseases to symptoms
        disease_itemset_counts: Dictionary with disease itemset counts
    """
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_diseases = len(transactions)
    # Focus on size == 3 itemsets for summary (most meaningful symptom triads)
    top_combination = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(1, 'support').iloc[0]
    top_symptoms = ', '.join(list(top_combination['itemsets']))
    top_support = top_combination['support']
    top_diseases = find_diseases_with_itemset(top_combination['itemsets'], disease_to_transaction)
    num_diseases_with_top = len(top_diseases)
    
    print(f"Key Findings:")
    print(f"1. Most frequent symptom combination: {{{top_symptoms}}}")
    print(f"   - Appears in {top_support*100:.2f}% of all diseases ({num_diseases_with_top} out of {total_diseases} diseases)")
    print(f"   - Specifically found in: {', '.join(top_diseases)}")
    
    # Calculate average itemset support for size == 3
    avg_itemset_support = frequent_itemsets[frequent_itemsets['itemset_size'] == 3]['support'].mean()
    print(f"\n2. Average support for symptom combinations (size == 3): {avg_itemset_support:.4f} ({avg_itemset_support*100:.2f}% of diseases)")
    print(f"   - This means symptom combinations typically appear together in {avg_itemset_support*100:.2f}% of disease profiles")
    
    # Calculate how many diseases share common patterns
    diseases_with_multiple_patterns = sum(1 for count in disease_itemset_counts.values() if count >= 5)
    print(f"\n3. Pattern sharing:")
    print(f"   - {diseases_with_multiple_patterns} diseases contain 5+ of the top 20 frequent symptom combinations")
    print(f"   - This indicates significant symptom pattern overlap across different diseases")
    
    print(f"\n4. Association rules:")
    print(f"   - Generated {len(rules)} association rules with confidence >= {min_confidence}")
    print(f"   - Average confidence: {rules['confidence'].mean():.4f} ({rules['confidence'].mean()*100:.2f}%)")
    print(f"   - Average lift: {rules['lift'].mean():.4f}")
    print(f"   - {((rules['lift'] > 1.5).sum())} rules show strong positive associations (lift > 1.5)")
    
    print(f"\n5. Overall statistics:")
    print(f"   - Total transactions (disease baskets): {total_diseases}")
    print(f"   - Total frequent itemsets: {len(frequent_itemsets)}")
    print(f"   - Total association rules: {len(rules)}")


def main():
    """Main execution function for the symptom analysis."""
    args = parse_arguments()
    
    # Setup paths
    data_path = project_root / args.data_path
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print("Starting Symptom Co-occurrence Analysis...")
        print(f"Data path: {data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum Support: {args.min_support}")
        print(f"Minimum Confidence: {args.min_confidence}")
        print("-" * 50)
    
    # 1. Load and process data
    print("\n[1/6] Processing symptom data...")
    print("Note: Each disease represents a 'basket' and symptoms are the 'items' in this basket.")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    processor = SymptomDataProcessor(data_path=data_path)
    transactions = processor.process_data(group_by_disease=True, min_transactions=args.min_transactions)
    
    if not transactions:
        print("No transactions were generated. Please check the dataset.")
        return
    
    # Create disease mapping
    df_raw = pd.read_csv(data_path)
    disease_to_transaction, disease_names = create_disease_mapping(df_raw, processor)
    
    # Verify mapping matches transactions
    assert len(transactions) == len(disease_names), "Mismatch between transactions and disease names"
    
    print(f"Successfully processed {len(transactions)} transactions (disease baskets).")
    total_symptoms = sum(len(t) for t in transactions)
    avg_symptoms_per_disease = total_symptoms / len(transactions)
    print(f"Data Statistics:")
    print(f"  Total diseases: {len(transactions)}")
    print(f"  Total symptoms across all diseases: {total_symptoms}")
    print(f"  Average symptoms per disease: {avg_symptoms_per_disease:.2f}")
    
    # 2. Mine for frequent itemsets
    print("\n[2/6] Mining for frequent symptom patterns...")
    print(f"Using minimum support threshold: {args.min_support}")
    miner = SymptomPatternMiner(transactions, min_support=args.min_support)
    frequent_itemsets = miner.mine_frequent_itemsets()
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets.")
    print(f"Itemset sizes range from 1 to {frequent_itemsets['itemsets'].apply(len).max()}")
    print("Note: Analysis focuses on size == 3 itemsets (symptom triads) for meaningful combinations.")
    
    # 3. Generate association rules
    print("\n[3/6] Generating association rules...")
    rules = miner.generate_association_rules(metric="confidence", min_threshold=args.min_confidence)
    print(f"Generated {len(rules)} association rules with confidence >= {args.min_confidence}")
    
    # 4. Save results to pickle files
    print("\n[4/6] Saving results...")
    itemsets_path = output_dir / 'frequent_itemsets.pkl'
    rules_path = output_dir / 'association_rules.pkl'
    
    frequent_itemsets.to_pickle(itemsets_path)
    rules.to_pickle(rules_path)
    
    print(f"Frequent itemsets saved to: {itemsets_path}")
    print(f"Association rules saved to: {rules_path}")
    
    # 5. Perform analysis
    print("\n[5/6] Performing analysis...")
    analyze_frequent_itemsets(frequent_itemsets, output_dir, args.save_plots, args.no_plots)
    analyze_association_rules(rules)
    
    # Calculate disease itemset counts for summary (using size == 3)
    disease_itemset_counts = {disease: 0 for disease in disease_names}
    top_itemsets_list = frequent_itemsets[frequent_itemsets['itemset_size'] == 3].nlargest(20, 'support')
    for idx, row in top_itemsets_list.iterrows():
        itemset = row['itemsets']
        diseases = find_diseases_with_itemset(itemset, disease_to_transaction)
        for disease in diseases:
            disease_itemset_counts[disease] += 1
    
    analyze_disease_level_patterns(frequent_itemsets, disease_to_transaction, disease_names, output_dir, args.save_plots, args.no_plots)
    
    # 6. Export disease-specific itemsets
    print("\n[6/6] Exporting disease-specific itemsets...")
    disease_itemsets_df = export_disease_itemsets(frequent_itemsets, disease_to_transaction, output_dir, args.save_plots, args.no_plots)
    
    # Print summary
    print_summary(transactions, frequent_itemsets, rules, disease_to_transaction, disease_itemset_counts, args.min_confidence)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
