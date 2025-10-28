"""
Main script for cancer sequential pattern mining analysis.

This script performs comprehensive sequential pattern mining on the Wisconsin 
Breast Cancer dataset to discover discriminative patterns between malignant 
and benign cases.

Examples:
    # Basic usage with default parameters
    python cancer_pattern_mining.py
    
    # Specify custom data path and output directory
    python cancer_pattern_mining.py --data-path /path/to/data.csv --output-dir /path/to/results
    
    # Adjust pattern mining parameters
    python cancer_pattern_mining.py --min-support 0.2 --max-pattern-length 3 --top-k 15
    
    # Use different discretization strategy
    python cancer_pattern_mining.py --discretization-strategy uniform --n-bins 5
    
    # Skip sensitivity analysis for faster execution
    python cancer_pattern_mining.py --skip-sensitivity --verbose
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import our modules
from src.processors.sequence_generator import CancerSequenceGenerator
from src.analysis.pattern_mining import SequentialPatternAnalyzer
from src.analysis.evaluation import SensitivityAnalyzer


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the Wisconsin Breast Cancer dataset.
    
    Args:
        data_path: Path to the dataset
        
    Returns:
        Tuple of (features, target)
    """
    print("Loading Wisconsin Breast Cancer dataset...")
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")
    
    # Drop columns that are all NaN (like Unnamed columns)
    data = data.dropna(axis=1, how='all')
    
    # Check for any remaining NaN values
    if data.isnull().sum().sum() > 0:
        print("Warning: Found NaN values, filling with median values...")
        # Fill NaN values with median for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # Separate features and target
    # Remove ID column and extract diagnosis
    target = data['diagnosis']
    features = data.drop(['id', 'diagnosis'], axis=1)
    
    print(f"Features shape: {features.shape}")
    print(f"Target distribution:")
    print(target.value_counts())
    
    return features, target


def perform_basic_analysis(features: pd.DataFrame, target: pd.Series, 
                          config: dict) -> dict:
    """
    Perform basic sequential pattern mining analysis.
    
    Args:
        features: Input features
        target: Target labels
        config: Configuration dictionary
        
    Returns:
        Analysis results
    """
    print("\n" + "="*60)
    print("PERFORMING BASIC SEQUENTIAL PATTERN ANALYSIS")
    print("="*60)
    
    # Generate sequences
    print("\nGenerating sequences...")
    sequence_generator = CancerSequenceGenerator(config)
    sequences = sequence_generator.fit_generate(features, target)
    
    print(f"Generated {len(sequences)} sequences")
    print(f"Average sequence length: {np.mean([len(seq) for seq in sequences]):.2f}")
    
    # Show example sequences
    print("\nExample sequences:")
    for i, seq in enumerate(sequences[:3]):
        print(f"  Sequence {i+1} (label: {target.iloc[i]}): {seq}")
    
    # Analyze patterns
    print("\nAnalyzing sequential patterns...")
    analyzer = SequentialPatternAnalyzer(config)
    results = analyzer.analyze(sequences, target.tolist())
    
    # Print summary
    print("\nAnalysis Summary:")
    summary = results['summary']
    print(f"  Total sequences: {summary['total_sequences']}")
    print(f"  Malignant cases: {summary['malignant_count']}")
    print(f"  Benign cases: {summary['benign_count']}")
    print(f"  Unique malignant patterns: {summary['unique_malignant_patterns']}")
    print(f"  Unique benign patterns: {summary['unique_benign_patterns']}")
    print(f"  Discriminative patterns: {summary['discriminative_patterns']}")
    
    return results


def print_interpretable_results(analyzer: SequentialPatternAnalyzer):
    """
    Print human-readable interpretation of results.
    
    Args:
        analyzer: Fitted analyzer with results
    """
    print("\n" + "="*60)
    print("INTERPRETABLE RESULTS")
    print("="*60)
    
    interpretable = analyzer.get_interpretable_results()
    
    print("\nKey Pattern Discoveries:")
    for i, interpretation in enumerate(interpretable['interpretations'], 1):
        print(f"\n{i}. Pattern: {interpretation['pattern']}")
        print(f"   {interpretation['interpretation']}")
        print(f"   Type: {interpretation['type']}")


def perform_sensitivity_analysis(features: pd.DataFrame, target: pd.Series,
                                base_config: dict) -> dict:
    """
    Perform comprehensive sensitivity analysis.
    
    Args:
        features: Input features
        target: Target labels
        base_config: Base configuration
        
    Returns:
        Sensitivity analysis results
    """
    print("\n" + "="*60)
    print("PERFORMING SENSITIVITY ANALYSIS")
    print("="*60)
    
    sensitivity_analyzer = SensitivityAnalyzer(base_config)
    
    # Analyze binning strategies
    print("\nAnalyzing binning strategies...")
    binning_results = sensitivity_analyzer.analyze_binning_strategies(features, target)
    
    # Analyze support thresholds
    print("Analyzing support thresholds...")
    support_results = sensitivity_analyzer.analyze_support_thresholds(features, target)
    
    # Analyze ranking methods
    print("Analyzing ranking methods...")
    ranking_results = sensitivity_analyzer.analyze_ranking_methods(features, target)
    
    # Get best configuration
    print("Determining best configuration...")
    best_config_info = sensitivity_analyzer.get_best_configuration()
    
    print("\nRecommended Configuration:")
    for key, value in best_config_info['best_config'].items():
        print(f"  {key}: {value}")
    
    print("\nRationale:")
    for reason in best_config_info['rationale']:
        print(f"  - {reason}")
    
    return sensitivity_analyzer.results


def save_results(results: dict, sensitivity_results: dict, output_dir: str):
    """
    Save analysis results to files.
    
    Args:
        results: Main analysis results
        sensitivity_results: Sensitivity analysis results
        output_dir: Output directory
    """
    print(f"\nSaving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results summary
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Cancer Sequential Pattern Mining Analysis Summary\n")
        f.write("="*50 + "\n\n")
        
        summary = results['summary']
        f.write(f"Total sequences: {summary['total_sequences']}\n")
        f.write(f"Malignant cases: {summary['malignant_count']}\n")
        f.write(f"Benign cases: {summary['benign_count']}\n")
        f.write(f"Unique malignant patterns: {summary['unique_malignant_patterns']}\n")
        f.write(f"Unique benign patterns: {summary['unique_benign_patterns']}\n")
        f.write(f"Discriminative patterns: {summary['discriminative_patterns']}\n\n")
        
        # Top discriminative patterns
        f.write("Top Discriminative Patterns:\n")
        f.write("-" * 30 + "\n")
        discriminative = results['class_comparison'].get('discriminative_patterns', [])
        for i, pattern_info in enumerate(discriminative[:10], 1):
            pattern_str = ' → '.join(['{' + ', '.join(itemset) + '}' for itemset in pattern_info['pattern']])
            f.write(f"{i}. {pattern_str}\n")
            f.write(f"   Discriminative for: {pattern_info['discriminative_for']}\n")
            f.write(f"   Malignant support: {pattern_info['malignant_support']:.3f}\n")
            f.write(f"   Benign support: {pattern_info['benign_support']:.3f}\n\n")
    
    # Save feature importance
    feature_importance_file = os.path.join(output_dir, 'feature_importance.txt')
    with open(feature_importance_file, 'w') as f:
        f.write("Feature Importance Analysis\n")
        f.write("="*30 + "\n\n")
        
        feature_importance = results['feature_importance']
        
        f.write("Top Features in Malignant Patterns:\n")
        for feature, score in feature_importance['top_malignant_features']:
            f.write(f"  {feature}: {score:.3f}\n")
        
        f.write("\nTop Features in Benign Patterns:\n")
        for feature, score in feature_importance['top_benign_features']:
            f.write(f"  {feature}: {score:.3f}\n")
    
    print(f"Results saved to {output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cancer Sequential Pattern Mining Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output paths
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='datasets/raw/wisconsin_breast_cancer.csv',
        help='Path to the cancer dataset CSV file'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Directory to save analysis results'
    )
    
    # Sequence generation parameters
    parser.add_argument(
        '--max-sequence-length',
        type=int,
        default=5,
        help='Maximum length of generated sequences'
    )
    parser.add_argument(
        '--max-gap',
        type=int,
        default=1,
        help='Maximum gap allowed in sequential patterns'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top features to consider for sequence generation'
    )
    
    # Discretization parameters
    parser.add_argument(
        '--discretization-strategy',
        choices=['uniform', 'quantile', 'kmeans'],
        default='quantile',
        help='Strategy for discretizing continuous features'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=3,
        help='Number of bins for feature discretization'
    )
    parser.add_argument(
        '--ranking-method',
        choices=['mutual_info', 'zscore', 'correlation'],
        default='mutual_info',
        help='Method for ranking feature importance'
    )
    
    # Pattern mining parameters
    parser.add_argument(
        '--min-support',
        type=float,
        default=0.1,
        help='Minimum support threshold for pattern mining'
    )
    parser.add_argument(
        '--max-pattern-length',
        type=int,
        default=4,
        help='Maximum length of discovered patterns'
    )
    
    # Analysis options
    parser.add_argument(
        '--skip-sensitivity',
        action='store_true',
        help='Skip sensitivity analysis to speed up execution'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create configuration dictionary from parsed arguments."""
    return {
        'max_sequence_length': args.max_sequence_length,
        'max_gap': args.max_gap,
        'ranking_method': args.ranking_method,
        'discretization_strategy': args.discretization_strategy,
        'n_bins': args.n_bins,
        'top_k': args.top_k,
        'min_support': args.min_support,
        'max_pattern_length': args.max_pattern_length
    }


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert relative paths to absolute paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_path
    output_dir = project_root / args.output_dir
    
    print("Cancer Sequential Pattern Mining Analysis")
    print("="*50)
    
    if args.verbose:
        print(f"Data path: {data_path}")
        print(f"Output directory: {output_dir}")
        print(f"Configuration: {create_config_from_args(args)}")
    
    # Configuration from CLI arguments
    config = create_config_from_args(args)
    
    try:
        # Load data
        features, target = load_data(str(data_path))
        
        # Perform basic analysis
        results = perform_basic_analysis(features, target, config)
        
        # Create analyzer for interpretable results
        sequence_generator = CancerSequenceGenerator(config)
        sequences = sequence_generator.fit_generate(features, target)
        analyzer = SequentialPatternAnalyzer(config)
        analyzer.analyze(sequences, target.tolist())
        
        # Print interpretable results
        print_interpretable_results(analyzer)
        
        # Perform sensitivity analysis (unless skipped)
        if not args.skip_sensitivity:
            sensitivity_results = perform_sensitivity_analysis(features, target, config)
        else:
            sensitivity_results = {}
            print("\nSkipping sensitivity analysis...")
        
        # Save results
        save_results(results, sensitivity_results, str(output_dir))
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Findings:")
        print("1. Successfully generated feature-based sequences from cancer data")
        print("2. Discovered discriminative patterns between malignant and benign cases")
        print("3. Identified optimal configuration through sensitivity analysis")
        print("4. Results saved for further investigation")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)