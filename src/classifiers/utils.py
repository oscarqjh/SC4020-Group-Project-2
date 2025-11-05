"""
Utility functions for data loading, feature extraction, and metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    confusion_matrix, classification_report
)


def load_cancer_data(filepath: str, target_column: str = 'diagnosis') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Wisconsin breast cancer dataset and separate features from target.
    
    Args:
        filepath: Path to the CSV file containing the dataset
        target_column: Name of the target column (default: 'diagnosis')
        
    Returns:
        Tuple of (X, y) where:
            - X: DataFrame containing feature columns
            - y: Series containing target labels
            
    Example:
        >>> X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')
        >>> print(X.shape, y.shape)
    """
    try:
        data = pd.read_csv(filepath)
        
        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
        # Extract target
        y = data[target_column]
        
        # Extract features (drop target and id columns if present)
        columns_to_drop = [target_column]
        if 'id' in data.columns:
            columns_to_drop.append('id')
        
        X = data.drop(columns=columns_to_drop)
        
        # Filter out non-feature columns (unnamed index columns)
        X = X.loc[:, ~X.columns.str.startswith('Unnamed:')]
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        return X, y
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def extract_feature_importance(filepath: str) -> Dict[str, Dict[str, float]]:
    """
    Extract feature importance scores from pattern mining analysis results.
    
    Args:
        filepath: Path to the feature_importance.txt file
        
    Returns:
        Dictionary with structure:
            {
                'malignant': {'feature_name': support_value, ...},
                'benign': {'feature_name': support_value, ...}
            }
            
    Example:
        >>> importance = extract_feature_importance('outputs/feature_importance.txt')
        >>> print(importance['malignant']['concave points'])
        7.759
    """
    try:
        filepath = Path(filepath)
        content = filepath.read_text()
        
        result = {'malignant': {}, 'benign': {}}
        current_section = None
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if 'Top Features in Malignant Patterns:' in line:
                current_section = 'malignant'
                continue
            elif 'Top Features in Benign Patterns:' in line:
                current_section = 'benign'
                continue
            
            # Parse feature lines (format: "feature_name: support_value")
            if current_section and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    feature_name = parts[0].strip()
                    try:
                        support_value = float(parts[1].strip())
                        result[current_section][feature_name] = support_value
                    except ValueError:
                        # Skip malformed lines
                        continue
        
        return result
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature importance file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error parsing feature importance file: {str(e)}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None,
                     average: str = 'binary',
                     pos_label: str = 'M') -> Dict[str, float]:
    """
    Calculate classification metrics with F1 as northstar metric, followed by Recall and Precision.
    
    Args:
        y_true: True labels array
        y_pred: Predicted labels array
        y_proba: Predicted probabilities for positive class (needed for ROC-AUC)
        average: Averaging strategy for metrics (default: 'binary')
        pos_label: Label of the positive class (default: 'M' for malignant)
        
    Returns:
        Dictionary with keys: 'f1_score', 'recall', 'precision', 'roc_auc'
        Note: 'roc_auc' will be None if y_proba is not provided
        
    Note:
        Metric priority: F1 (primary) > Recall > Precision
    """
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}")
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average=average, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    precision = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    
    # Calculate ROC-AUC if probabilities are provided
    roc_auc = None
    if y_proba is not None:
        try:
            # Convert y_true to binary array based on pos_label
            y_true_bin = (y_true == pos_label).astype(int)
            
            # Handle 2D probability arrays (select positive class column)
            y_proba_1d = y_proba
            if y_proba.ndim == 2:
                # Standard binary proba output: use second column for positive class
                y_proba_1d = y_proba[:, 1]
            
            roc_auc = roc_auc_score(y_true_bin, y_proba_1d)
        except ValueError:
            # Handle cases where ROC-AUC cannot be calculated (e.g., single-class y_true)
            roc_auc = None
    
    return {
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'roc_auc': roc_auc
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                        labels: Optional[List[str]] = None) -> np.ndarray:
    """
    Generate confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of class labels in order
        
    Returns:
        Numpy array representing the confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                             target_names: Optional[List[str]] = None,
                             labels: Optional[List[str]] = None) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names for display
        labels: Optional list of labels to include in the report
        
    Returns:
        String containing the formatted classification report
    """
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names)

