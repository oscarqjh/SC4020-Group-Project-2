"""
Feature selection module based on pattern mining results.

This module provides intelligent feature selection by combining pattern mining
analysis results with multicollinearity detection to identify the most important
and non-redundant features for classification.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..processors.base import BasePreprocessor
from .utils import extract_feature_importance


class FeatureSelector(BasePreprocessor):
    """
    Feature selector based on pattern mining results.
    
    This class performs intelligent feature selection by:
    1. Parsing feature importance scores from pattern mining analysis
    2. Detecting multicollinearity among correlated features
    3. Selecting top-N features based on combined support values
    
    The selector automatically handles highly correlated features (e.g., radius,
    perimeter, and area are mathematically related: area = π × radius²,
    perimeter = 2π × radius) by keeping only the most important feature from
    each correlated group.
    
    Example:
        >>> from src.classifiers import FeatureSelector, load_cancer_data
        >>> 
        >>> # Load data
        >>> X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')
        >>> 
        >>> # Create and fit selector
        >>> selector = FeatureSelector({
        ...     'feature_importance_path': 'outputs/feature_importance.txt',
        ...     'top_n': 5,
        ...     'correlation_threshold': 0.9,
        ...     'aggregation_method': 'sum'
        ... })
        >>> 
        >>> # Select features
        >>> X_selected = selector.fit_transform(X, y)
        >>> 
        >>> # View selected features
        >>> print(selector.get_selected_features())
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature selector.
        
        Args:
            config: Configuration dictionary containing:
                - feature_importance_path: Path to feature_importance.txt file (required)
                - top_n: Number of top features to select (default: 10)
                - correlation_threshold: Threshold for detecting multicollinearity (default: 0.9)
                - aggregation_method: How to combine malignant and benign support values
                  ('sum', 'mean', 'max') (default: 'sum')
        """
        super().__init__(config)
        
        # Extract config parameters
        self.feature_importance_path = self.config.get('feature_importance_path')
        self.top_n = self.config.get('top_n', 10)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.9)
        self.aggregation_method = self.config.get('aggregation_method', 'sum')
        
        # Initialize instance variables
        self.selected_features_ = None
        self.feature_scores_ = None
        self.correlation_matrix_ = None
        self.removed_features_ = None
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fit the feature selector to identify top features.
        
        Args:
            data: Input DataFrame with features
            target: Target variable (optional, not used but kept for interface compatibility)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If feature_importance_path is not provided or if file cannot be parsed
        """
        if self.feature_importance_path is None:
            raise ValueError("feature_importance_path must be provided in config")
        
        # 1. Parse feature importance
        importance_dict = extract_feature_importance(self.feature_importance_path)
        malignant_scores = importance_dict.get('malignant', {})
        benign_scores = importance_dict.get('benign', {})
        
        # 2. Combine scores from both classes
        unified_scores = {}
        all_features = set(malignant_scores.keys()) | set(benign_scores.keys())
        
        for feature in all_features:
            malignant_score = malignant_scores.get(feature, 0.0)
            benign_score = benign_scores.get(feature, 0.0)
            
            if self.aggregation_method == 'sum':
                combined_score = malignant_score + benign_score
            elif self.aggregation_method == 'mean':
                combined_score = (malignant_score + benign_score) / 2
            elif self.aggregation_method == 'max':
                combined_score = max(malignant_score, benign_score)
            else:
                raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}. "
                               f"Must be 'sum', 'mean', or 'max'")
            
            unified_scores[feature] = combined_score
        
        # 3. Match feature names to dataset columns
        expanded_scores = {}
        for base_feature, score in unified_scores.items():
            matching_columns = self._match_feature_to_columns(base_feature, data.columns.tolist())
            for column in matching_columns:
                expanded_scores[column] = score
        
        # Store feature scores
        self.feature_scores_ = expanded_scores.copy()
        
        # 4. Detect multicollinearity
        numeric_data = data.select_dtypes(include=[np.number])
        self.correlation_matrix_ = numeric_data.corr()
        
        # 5. Select features while avoiding multicollinearity
        # Sort features by score (descending)
        sorted_features = sorted(
            expanded_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_features = []
        removed_features = []
        
        for feature, score in sorted_features:
            if feature not in numeric_data.columns:
                continue  # Skip non-numeric features
            
            if len(selected_features) >= self.top_n:
                break
            
            # Check if feature is correlated with any already-selected feature
            is_correlated = False
            correlated_with = None
            
            for selected_feature in selected_features:
                if (feature in self.correlation_matrix_.index and 
                    selected_feature in self.correlation_matrix_.columns):
                    correlation = abs(self.correlation_matrix_.loc[feature, selected_feature])
                    if correlation > self.correlation_threshold:
                        is_correlated = True
                        correlated_with = selected_feature
                        break
            
            if not is_correlated:
                selected_features.append(feature)
            else:
                removed_features.append({
                    'feature': feature,
                    'score': score,
                    'reason': f'correlated with {correlated_with}',
                    'correlation': self.correlation_matrix_.loc[feature, correlated_with]
                })
        
        self.selected_features_ = selected_features
        self.removed_features_ = removed_features
        
        # Set fitted flag
        self.is_fitted = True
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to only include selected features.
        
        Args:
            data: Input DataFrame to transform
            
        Returns:
            DataFrame with only selected features
            
        Raises:
            ValueError: If selector has not been fitted or if required features are missing
        """
        self.validate_fitted()
        
        # Validate that all selected features exist in data
        missing_features = [f for f in self.selected_features_ if f not in data.columns]
        if missing_features:
            raise ValueError(
                f"Selected features not found in data: {missing_features}. "
                f"Available columns: {list(data.columns)}"
            )
        
        # Return subset with selected features
        return data[self.selected_features_].copy()
    
    def get_feature_scores(self) -> Dict[str, float]:
        """
        Return the computed feature scores.
        
        Returns:
            Dictionary mapping feature names to their combined importance scores
            
        Raises:
            ValueError: If selector has not been fitted
        """
        self.validate_fitted()
        return self.feature_scores_.copy()
    
    def get_selected_features(self) -> List[str]:
        """
        Return the list of selected feature names.
        
        Returns:
            List of selected feature names
            
        Raises:
            ValueError: If selector has not been fitted
        """
        self.validate_fitted()
        return self.selected_features_.copy()
    
    def select_features(self) -> List[str]:
        """
        Return the list of selected top-N features.
        
        This method is an alias for get_selected_features() for API consistency.
        It returns the list of features that were selected during fitting.
        
        Returns:
            List of selected feature names
            
        Raises:
            ValueError: If selector has not been fitted (fit() must be called first)
        """
        self.validate_fitted()
        return self.selected_features_.copy()
    
    def get_removed_features(self) -> List[Dict[str, Any]]:
        """
        Return the list of features removed due to multicollinearity.
        
        Returns:
            List of dictionaries containing information about removed features:
            - feature: Feature name
            - score: Feature importance score
            - reason: Reason for removal
            - correlation: Correlation value with the kept feature
            
        Raises:
            ValueError: If selector has not been fitted
        """
        self.validate_fitted()
        return self.removed_features_.copy()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Return the correlation matrix computed during fitting.
        
        Returns:
            DataFrame containing correlation matrix for all numeric features
            
        Raises:
            ValueError: If selector has not been fitted
        """
        self.validate_fitted()
        return self.correlation_matrix_.copy()
    
    def _match_feature_to_columns(self, base_feature: str, columns: List[str]) -> List[str]:
        """
        Match base feature names to actual column names.
        
        This helper method handles the mapping between base feature names
        (e.g., 'concave points', 'radius') and actual dataset column names
        (e.g., 'concave points_mean', 'radius_mean', 'radius_se', 'radius_worst').
        
        Args:
            base_feature: Base feature name from importance file (e.g., 'concave points')
            columns: List of actual column names in the dataset
            
        Returns:
            List of column names that match the base feature
        """
        base_feature_clean = base_feature.strip().lower()
        matching_columns = []
        
        for column in columns:
            column_clean = column.strip().lower()
            # Check if base_feature is contained in column name
            if base_feature_clean in column_clean:
                matching_columns.append(column)
        
        return matching_columns
    
    def _find_correlated_groups(self, correlation_matrix: pd.DataFrame, threshold: float) -> List[List[str]]:
        """
        Identify groups of correlated features.
        
        This method uses a simple grouping approach to find all features that
        are correlated above the threshold. Features in the same group are
        highly correlated with each other.
        
        Note: This method is currently unused but kept for future enhancements
        where group-based feature selection might be implemented.
        
        Args:
            correlation_matrix: DataFrame containing correlation matrix
            threshold: Correlation threshold for grouping
            
        Returns:
            List of groups, where each group is a list of correlated feature names
        """
        groups = []
        processed_features = set()
        
        for feature1 in correlation_matrix.index:
            if feature1 in processed_features:
                continue
            
            # Find all features correlated with feature1
            correlated_with_feature1 = [feature1]
            
            for feature2 in correlation_matrix.index:
                if feature1 == feature2 or feature2 in processed_features:
                    continue
                
                correlation = abs(correlation_matrix.loc[feature1, feature2])
                if correlation > threshold:
                    correlated_with_feature1.append(feature2)
                    processed_features.add(feature2)
            
            if len(correlated_with_feature1) > 1:
                groups.append(correlated_with_feature1)
            
            processed_features.add(feature1)
        
        return groups

