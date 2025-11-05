"""
Main script for training Random Forest binary classifier on Wisconsin Breast Cancer dataset.

This script performs feature selection based on pattern mining results, trains a Random Forest 
classifier with GridSearchCV hyperparameter tuning, and saves the trained model with 
comprehensive evaluation metrics.

Examples:
    # Basic usage with defaults
    python scripts/train_random_forest.py
    
    # Custom feature selection parameters
    python scripts/train_random_forest.py --use-feature-selection --top-n-features 10 --correlation-threshold 0.9
    
    # Custom hyperparameter grid
    python scripts/train_random_forest.py --n-estimators 100 200 300 --max-depth 10 20 None --cv-folds 10
    
    # Verbose output mode
    python scripts/train_random_forest.py --use-feature-selection --verbose
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import argparse
import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import make_scorer, f1_score

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import our modules
from src.classifiers import RandomForestBinaryClassifier, FeatureSelector, load_cancer_data

# Suppress warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest binary classifier on Wisconsin Breast Cancer dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output paths
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/wisconsin_breast_cancer.csv',
        help='Path to cancer dataset CSV file'
    )
    parser.add_argument(
        '--feature-importance-path',
        type=str,
        default='outputs/feature_importance.txt',
        help='Path to feature importance file from pattern mining'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts',
        help='Directory to save results and model'
    )
    
    # Feature selection parameters
    parser.add_argument(
        '--use-feature-selection',
        action='store_true',
        default=False,
        help='Enable feature selection based on pattern mining results'
    )
    parser.add_argument(
        '--no-feature-selection',
        dest='use_feature_selection',
        action='store_false',
        help='Disable feature selection (use all features)'
    )
    parser.add_argument(
        '--top-n-features',
        type=int,
        default=10,
        help='Number of top features to select'
    )
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.9,
        help='Threshold for multicollinearity detection'
    )
    parser.add_argument(
        '--aggregation-method',
        choices=['sum', 'mean', 'max'],
        default='sum',
        help='Method to combine malignant and benign support values'
    )
    
    # Random Forest hyperparameters
    parser.add_argument(
        '--n-estimators',
        nargs='+',
        type=int,
        default=[100, 200, 300],
        help='List of values for number of trees in grid search'
    )
    parser.add_argument(
        '--max-depth',
        nargs='+',
        type=str,
        default=['10', '20', 'None'],
        help='List of values for maximum tree depth (use "None" for no limit)'
    )
    parser.add_argument(
        '--min-samples-split',
        nargs='+',
        type=int,
        default=[2, 5, 10],
        help='List of values for minimum samples to split in grid search'
    )
    parser.add_argument(
        '--class-weight',
        type=str,
        default='balanced',
        help='Strategy for handling class imbalance'
    )
    
    # Training configuration
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        default=True,
        help='Save trained model'
    )
    parser.add_argument(
        '--no-save-model',
        dest='save_model',
        action='store_false',
        help='Do not save trained model'
    )
    parser.add_argument(
        '--model-filename',
        type=str,
        default=None,
        help='Custom model filename (if not provided, uses timestamped default)'
    )
    parser.add_argument(
        '--strict-venv',
        action='store_true',
        help='Exit with error if virtual environment is not activated'
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert parsed arguments into configuration dictionaries.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (feature_selector_config, classifier_config)
    """
    # Handle 'None' string conversion for max_depth
    max_depth_values = []
    for val in args.max_depth:
        if val.lower() == 'none':
            max_depth_values.append(None)
        else:
            try:
                max_depth_values.append(int(val))
            except ValueError:
                raise ValueError(f"Invalid max_depth value: {val}. Must be an integer or 'None'")
    
    # Validate parameter ranges
    if not (0 < args.test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1, got {args.test_size}")
    if args.cv_folds < 2:
        raise ValueError(f"cv_folds must be at least 2, got {args.cv_folds}")
    
    # Feature selector configuration (only if feature selection is enabled)
    if args.use_feature_selection:
        feature_selector_config = {
            'feature_importance_path': args.feature_importance_path,
            'top_n': args.top_n_features,
            'correlation_threshold': args.correlation_threshold,
            'aggregation_method': args.aggregation_method
        }
    else:
        feature_selector_config = {}
    
    # Classifier configuration
    classifier_config = {
        'n_estimators': args.n_estimators,
        'max_depth': max_depth_values,
        'min_samples_split': args.min_samples_split,
        'class_weight': args.class_weight,
        'cv_folds': args.cv_folds,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs,
        'verbose': 1 if args.verbose else 0
    }
    
    return feature_selector_config, classifier_config


def load_and_validate_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series, Optional[str]]:
    """
    Load dataset and perform validation.
    
    Args:
        data_path: Path to the dataset CSV file
        
    Returns:
        Tuple of (X, y, detected_positive) where:
            - X: Features DataFrame
            - y: Target Series
            - detected_positive: Positive class label if 'M' not found (for scorer config), None otherwise
        
    Raises:
        FileNotFoundError: If dataset file is not found
        ValueError: If data validation fails
    """
    print("Loading Wisconsin Breast Cancer dataset...")
    print(f"Data path: {data_path}")
    
    try:
        # Load data
        X, y = load_cancer_data(data_path)
        
        # Print dataset statistics
        print(f"\nDataset shape: {X.shape}")
        print(f"Number of samples: {len(X)}")
        
        # Class distribution
        class_counts = y.value_counts()
        class_percentages = y.value_counts(normalize=True) * 100
        print(f"\nClass distribution:")
        for class_label in class_counts.index:
            count = class_counts[class_label]
            percentage = class_percentages[class_label]
            print(f"  {class_label}: {count} ({percentage:.1f}%)")
        
        # Feature names (first 10)
        print(f"\nFeature names (first 10): {list(X.columns[:10])}")
        
        # Validate data
        # Check for missing values
        missing_values = X.isnull().sum().sum()
        if missing_values > 0:
            print(f"\nWarning: Found {missing_values} missing values in features")
        
        # Ensure binary classification
        unique_classes = y.unique()
        if len(unique_classes) != 2:
            raise ValueError(
                f"Expected binary classification (2 classes), got {len(unique_classes)} classes: {unique_classes}"
            )
        
        # Check that 'M' is among the labels for F1 scorer pos_label
        if 'M' not in unique_classes:
            detected_positive = unique_classes[0]  # Use first class as positive
            print(f"\nWarning: 'M' (malignant) not found in labels. Detected classes: {unique_classes}")
            print(f"Will use '{detected_positive}' as positive class for F1 scoring.")
            print("Consider mapping labels to ['B', 'M'] before training for consistency.")
            # Return detected positive class for scorer configuration
            return X, y, detected_positive
        else:
            return X, y, None
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    except Exception as e:
        raise ValueError(f"Error loading or validating data: {str(e)}")


def perform_feature_selection(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], 
                               verbose: bool = False) -> Tuple[pd.DataFrame, Optional[FeatureSelector]]:
    """
    Perform feature selection if enabled.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        config: Feature selector configuration
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (X_selected, selector) where selector is None if feature selection is disabled
    """
    # Check if feature selection is enabled
    if not config.get('feature_importance_path'):
        print("\nFeature selection disabled: feature_importance_path not provided")
        return X, None
    
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    try:
        # Create FeatureSelector
        selector = FeatureSelector(config)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Print results
        print(f"\nSelected {len(selector.get_selected_features())} features from {X.shape[1]} original features")
        
        selected_features = selector.get_selected_features()
        print(f"\nSelected features: {selected_features}")
        
        # Feature scores (top 10)
        feature_scores = selector.get_feature_scores()
        sorted_scores = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 feature scores:")
        for feature, score in sorted_scores[:10]:
            print(f"  {feature}: {score:.4f}")
        
        # Removed features due to multicollinearity
        removed_features = selector.get_removed_features()
        if removed_features:
            print(f"\nRemoved {len(removed_features)} features due to multicollinearity:")
            for removed in removed_features[:5]:  # Show first 5
                print(f"  {removed['feature']} (correlated with {removed['reason']}, r={removed['correlation']:.3f})")
        
        # If verbose, print correlation matrix for selected features
        if verbose and len(selected_features) > 1:
            correlation_matrix = selector.get_correlation_matrix()
            selected_corr = correlation_matrix.loc[selected_features, selected_features]
            print(f"\nCorrelation matrix for selected features:")
            print(selected_corr.round(3))
        
        return X_selected, selector
        
    except FileNotFoundError as e:
        print(f"\nWarning: Feature importance file not found: {config['feature_importance_path']}")
        print("Continuing without feature selection...")
        return X, None
    except Exception as e:
        print(f"\nError during feature selection: {str(e)}")
        print("Continuing without feature selection...")
        return X, None


def train_classifier(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any], 
                     verbose: bool = False) -> RandomForestBinaryClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        config: Classifier configuration
        verbose: Whether to print verbose output
        
    Returns:
        Fitted RandomForestBinaryClassifier
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Hyperparameter grid:")
    print(f"    n_estimators: {config['n_estimators']}")
    print(f"    max_depth: {config['max_depth']}")
    print(f"    min_samples_split: {config['min_samples_split']}")
    print(f"  Cross-validation folds: {config['cv_folds']}")
    print(f"  Test size: {config['test_size']}")
    print(f"  Class weight: {config['class_weight']}")
    
    # Calculate number of parameter combinations
    n_combinations = len(config['n_estimators']) * len(config['max_depth']) * len(config['min_samples_split'])
    print(f"\nFitting GridSearchCV with {n_combinations} parameter combinations...")
    
    # Create classifier
    classifier = RandomForestBinaryClassifier(config)
    
    # Fit classifier
    classifier.fit(X, y)
    
    print("\nTraining completed!")
    
    return classifier


def evaluate_and_display_results(classifier: RandomForestBinaryClassifier, verbose: bool = False):
    """
    Display comprehensive evaluation results.
    
    Args:
        classifier: Fitted RandomForestBinaryClassifier
        verbose: Whether to print verbose output
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Call classifier's print_summary method
    classifier.print_summary()
    
    # If verbose, print additional details
    if verbose:
        print("\n" + "-"*60)
        print("ADDITIONAL DETAILS")
        print("-"*60)
        
        # Cross-validation results
        cv_results = classifier.get_cv_results()
        print("\nCross-validation results (mean ± std):")
        
        # Get top 5 parameter combinations by mean score
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']
        params_list = cv_results['params']
        
        # Sort by mean score
        sorted_indices = sorted(range(len(mean_scores)), key=lambda i: mean_scores[i], reverse=True)
        
        print("\nTop 5 parameter combinations:")
        for i, idx in enumerate(sorted_indices[:5], 1):
            params = params_list[idx]
            mean = mean_scores[idx]
            std = std_scores[idx]
            print(f"  {i}. {params}")
            print(f"     Score: {mean:.4f} ± {std:.4f}")
        
        # Feature importance for all features
        print("\nFeature importance for all features:")
        importance_df = classifier.get_feature_importance()
        print(importance_df.to_string(index=False))


def save_model_and_results(classifier: RandomForestBinaryClassifier, selector: Optional[FeatureSelector],
                          output_dir: str, model_filename: Optional[str] = None) -> Dict[str, str]:
    """
    Save trained model and results.
    
    Args:
        classifier: Fitted RandomForestBinaryClassifier
        selector: Optional fitted FeatureSelector
        output_dir: Directory to save results
        model_filename: Optional custom model filename
        
    Returns:
        Dictionary with saved file paths
    """
    print("\n" + "="*60)
    print("SAVING MODEL AND RESULTS")
    print("="*60)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save classifier
    if model_filename:
        model_path = output_path / model_filename
        saved_path = classifier.save_model(str(model_path))
    else:
        # Generate timestamped filename in output_dir
        model_filename_timestamped = f'random_forest_model_{timestamp}.pkl'
        model_path = output_path / model_filename_timestamped
        saved_path = classifier.save_model(str(model_path))
    
    saved_files['model'] = str(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature selector (if used)
    if selector is not None:
        selector_filename = f'feature_selector_{timestamp}.pkl'
        selector_path = output_path / selector_filename
        with open(selector_path, 'wb') as f:
            pickle.dump(selector, f)
        saved_files['selector'] = str(selector_path)
        print(f"Feature selector saved to: {selector_path}")
    
    # Save results summary to text file
    results_filename = f'training_results_{timestamp}.txt'
    results_path = output_path / results_filename
    
    with open(results_path, 'w') as f:
        f.write("Random Forest Classifier Training Results\n")
        f.write("="*50 + "\n\n")
        
        # Best hyperparameters
        best_params = classifier.get_best_params()
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Test metrics
        test_metrics = classifier.get_test_metrics()
        f.write("Test Set Metrics:\n")
        f.write(f"  F1 Score:    {test_metrics['f1_score']:.4f}\n")
        f.write(f"  Recall:      {test_metrics['recall']:.4f}\n")
        f.write(f"  Precision:   {test_metrics['precision']:.4f}\n")
        if test_metrics['roc_auc'] is not None:
            f.write(f"  ROC-AUC:     {test_metrics['roc_auc']:.4f}\n")
        f.write("\n")
        
        # Selected features (if feature selection was used)
        if selector is not None:
            selected_features = selector.get_selected_features()
            f.write(f"Selected Features ({len(selected_features)}):\n")
            for feature in selected_features:
                f.write(f"  {feature}\n")
            f.write("\n")
        
        # Feature importance (top 10)
        importance_df = classifier.get_feature_importance()
        f.write("Top 10 Feature Importances:\n")
        for _, row in importance_df.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    saved_files['results'] = str(results_path)
    print(f"Results summary saved to: {results_path}")
    
    return saved_files


def print_venv_instructions():
    """Print virtual environment setup instructions."""
    print("\n" + "="*60)
    print("VIRTUAL ENVIRONMENT SETUP")
    print("="*60)
    print("\nTo set up the virtual environment, run the following commands:\n")
    print("1. Create virtual environment:")
    print("   python3 -m venv .venv\n")
    print("2. Activate virtual environment:")
    print("   # On macOS/Linux:")
    print("   source .venv/bin/activate")
    print("   # On Windows (Command Prompt):")
    print("   .venv\\Scripts\\activate.bat")
    print("   # On Windows (PowerShell):")
    print("   .venv\\Scripts\\Activate.ps1\n")
    print("3. Install dependencies:")
    print("   python3 -m pip install -e .\n")
    print("4. Verify installation:")
    print("   python scripts/train_random_forest.py --help\n")
    print("Note: Ensure the virtual environment is activated before running the script.\n")


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if virtual environment is activated
    in_venv = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if not in_venv:
        print("\n" + "="*60)
        print("WARNING: Virtual environment not detected")
        print("="*60)
        print("\nIt is recommended to run this script within a virtual environment.")
        print("\nTo activate the virtual environment:")
        print("  # On macOS/Linux:")
        print("  source .venv/bin/activate")
        print("  # On Windows (Command Prompt):")
        print("  .venv\\Scripts\\activate.bat")
        print("  # On Windows (PowerShell):")
        print("  .venv\\Scripts\\Activate.ps1")
        print("\nIf you continue, dependencies may conflict with system Python.")
        
        if args.strict_venv:
            print("\nExiting due to --strict-venv flag.")
            return 1
        
        print("\nContinuing without virtual environment...")
        print("-"*60)
    
    # Convert relative paths to absolute paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data_path
    feature_importance_path = project_root / args.feature_importance_path
    output_dir = project_root / args.output_dir
    
    # Print script header
    print("="*60)
    print("Random Forest Classifier Training")
    print("="*60)
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.verbose:
        print("\nConfiguration:")
        print(f"  Data path: {data_path}")
        print(f"  Feature importance path: {feature_importance_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Use feature selection: {args.use_feature_selection}")
        if args.use_feature_selection:
            print(f"  Top N features: {args.top_n_features}")
            print(f"  Correlation threshold: {args.correlation_threshold}")
            print(f"  Aggregation method: {args.aggregation_method}")
        print(f"  N estimators: {args.n_estimators}")
        print(f"  Max depth: {args.max_depth}")
        print(f"  Min samples split: {args.min_samples_split}")
        print(f"  CV folds: {args.cv_folds}")
        print(f"  Test size: {args.test_size}")
        print(f"  Random state: {args.random_state}")
        print(f"  N jobs: {args.n_jobs}")
    
    try:
        # Create configurations from arguments
        feature_selector_config, classifier_config = create_config_from_args(args)
        
        # Update feature_importance_path to absolute path (if feature selection is enabled)
        if args.use_feature_selection and feature_selector_config.get('feature_importance_path'):
            feature_selector_config['feature_importance_path'] = str(feature_importance_path)
        
        # Load and validate data
        X, y, detected_positive = load_and_validate_data(str(data_path))
        
        # Configure scorer if 'M' label not found
        if detected_positive is not None:
            f1_m_scorer = make_scorer(f1_score, pos_label=detected_positive)
            classifier_config['scoring'] = f1_m_scorer
        
        # Perform feature selection (if enabled)
        if args.use_feature_selection:
            X, selector = perform_feature_selection(X, y, feature_selector_config, verbose=args.verbose)
        else:
            X, selector = X, None
        
        # Train classifier
        classifier = train_classifier(X, y, classifier_config, verbose=args.verbose)
        
        # Evaluate and display results
        evaluate_and_display_results(classifier, verbose=args.verbose)
        
        # Save model and results (if enabled)
        saved_files = {}
        if args.save_model:
            saved_files = save_model_and_results(
                classifier, selector, str(output_dir), args.model_filename
            )
        
        # Print completion message
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        test_metrics = classifier.get_test_metrics()
        print(f"\nKey Findings:")
        print(f"  Best F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"  Number of features used: {X.shape[1]}")
        if saved_files:
            print(f"  Model saved to: {saved_files.get('model', 'N/A')}")
            if 'results' in saved_files:
                print(f"  Results saved to: {saved_files['results']}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: File not found: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    except ValueError as e:
        print(f"\nError: Invalid parameter: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

