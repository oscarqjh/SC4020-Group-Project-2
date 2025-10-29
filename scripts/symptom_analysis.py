import argparse
import os
from pathlib import Path
import sys

# Add src to Python path to be able to import modules from src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.processors.symptom_data_processor import SymptomDataProcessor
from src.analysis.symptom_pattern_miner import SymptomPatternMiner

def parse_arguments():
    """Parses command-line arguments for the symptom analysis script."""
    parser = argparse.ArgumentParser(description="Symptom Co-occurrence Pattern Analysis using Apriori.")
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/dataset.csv',
        help='Path to the symptom dataset CSV file.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save the analysis results.'
    )
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.02, # Change this if you want to change the minimum support threshold
        help='Minimum support threshold for the Apriori algorithm.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output to the console.'
    )
    return parser.parse_args()

def main():
    """Main execution function for the symptom analysis."""
    args = parse_arguments()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("Starting Symptom Co-occurrence Analysis...")
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Minimum Support: {args.min_support}")
        print("-" * 50)

    # 1. Process the data
    print("Processing symptom data...")
    processor = SymptomDataProcessor(data_path=args.data_path)
    transactions = processor.process_data()
    
    if not transactions:
        print("No transactions were generated. Please check the dataset.")
        return

    print(f"Data processing complete. Found {len(transactions)} transactions.")
    
    # 2. Mine for patterns
    print("Mining for frequent symptom patterns...")
    print("This may take a few minutes depending on the dataset size and min_support...")
    miner = SymptomPatternMiner(transactions, min_support=args.min_support)
    frequent_itemsets = miner.mine_frequent_itemsets()
    
    print("Frequent itemset mining complete. Generating association rules...")
    rules = miner.generate_association_rules()

    if args.verbose:
        print("Pattern mining complete.")
        print("-" * 50)
        print("Top 10 Frequent Itemsets:")
        print(frequent_itemsets.nlargest(10, 'support'))
        print("\nTop 10 Association Rules (by confidence):")
        print(rules.nlargest(10, 'confidence'))
        print("-" * 50)

    # 3. Save the results
    itemsets_path = output_dir / 'frequent_itemsets.csv'
    rules_path = output_dir / 'association_rules.csv'
    
    frequent_itemsets.to_csv(itemsets_path, index=False)
    rules.to_csv(rules_path, index=False)

    print("Analysis complete.")
    print(f"Frequent itemsets saved to: {itemsets_path}")
    print(f"Association rules saved to: {rules_path}")

if __name__ == '__main__':
    main()

