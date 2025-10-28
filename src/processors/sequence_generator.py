"""
Feature discretization and sequence generation components.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional, Tuple
from .base import BasePreprocessor, BaseSequenceGenerator


class FeatureDiscretizer(BasePreprocessor):
    """Discretizes continuous features into categorical bins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature discretizer.
        
        Args:
            config: Configuration dictionary with keys:
                - n_bins: Number of bins (default: 3)
                - strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
                - encode: Encoding method ('onehot', 'onehot-dense', 'ordinal')
        """
        super().__init__(config)
        self.n_bins = self.config.get('n_bins', 3)
        self.strategy = self.config.get('strategy', 'quantile')
        self.encode = self.config.get('encode', 'ordinal')
        self.discretizer = None
        self.feature_names = None
        self.bin_labels = ['low', 'medium', 'high'] if self.n_bins == 3 else [f'bin_{i}' for i in range(self.n_bins)]
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureDiscretizer':
        """
        Fit the discretizer to the data.
        
        Args:
            data: Input features
            target: Target variable (unused)
            
        Returns:
            Self for method chaining
        """
        self.feature_names = data.columns.tolist()
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode=self.encode,
            strategy=self.strategy,
            dtype=np.float64
        )
        self.discretizer.fit(data)
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform continuous features to discrete bins.
        
        Args:
            data: Input features
            
        Returns:
            Discretized features with meaningful labels
        """
        self.validate_fitted()
        
        # Transform to ordinal encoding
        discretized = self.discretizer.transform(data)
        
        # Convert to DataFrame with meaningful labels
        result_data = {}
        for i, col in enumerate(self.feature_names):
            # Map ordinal values to meaningful labels
            discretized_col = discretized[:, i]
            labeled_col = [self.bin_labels[int(val)] for val in discretized_col]
            result_data[col] = labeled_col
            
        return pd.DataFrame(result_data, index=data.index)
    
    def transform_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform continuous features to discrete bins with numeric values.
        
        Args:
            data: Input features
            
        Returns:
            Discretized features with numeric values (for ranking)
        """
        self.validate_fitted()
        
        # Transform to ordinal encoding (keep numeric)
        discretized = self.discretizer.transform(data)
        
        # Convert to DataFrame with numeric values
        return pd.DataFrame(discretized, columns=self.feature_names, index=data.index)


class FeatureRanker(BasePreprocessor):
    """Ranks features by importance using mutual information or z-score."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature ranker.
        
        Args:
            config: Configuration dictionary with keys:
                - method: Ranking method ('mutual_info', 'zscore')
                - top_k: Number of top features to select
        """
        super().__init__(config)
        self.method = self.config.get('method', 'mutual_info')
        self.top_k = self.config.get('top_k', 10)
        self.feature_importance_ = None
        self.scaler = StandardScaler() if self.method == 'zscore' else None
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureRanker':
        """
        Fit the ranker to compute feature importance.
        
        Args:
            data: Input features
            target: Target variable (required for mutual_info method)
            
        Returns:
            Self for method chaining
        """
        if self.method == 'mutual_info':
            if target is None:
                raise ValueError("Target is required for mutual information ranking")
            # Convert target to numeric if it's categorical
            target_numeric = pd.Categorical(target).codes
            self.feature_importance_ = mutual_info_classif(data, target_numeric, random_state=42)
        elif self.method == 'zscore':
            self.scaler.fit(data)
            # For z-score method, we'll compute importance during transform
            self.feature_importance_ = np.ones(len(data.columns))
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Rank features for each sample and return top-k features.
        
        Args:
            data: Input features
            
        Returns:
            Tuple of (ranked_features, feature_order) where ranked_features
            contains the top-k most important features for each sample
        """
        self.validate_fitted()
        
        if self.method == 'mutual_info':
            # Get top-k features based on mutual information
            feature_indices = np.argsort(self.feature_importance_)[::-1][:self.top_k]
            top_features = data.columns[feature_indices].tolist()
            return data[top_features], top_features
        
        elif self.method == 'zscore':
            # Standardize features and rank by absolute z-score for each sample
            scaled_data = self.scaler.transform(data)
            abs_scaled = np.abs(scaled_data)
            
            # For each sample, get top-k features by z-score
            ranked_data = {}
            feature_orders = []
            
            for i in range(len(data)):
                sample_scores = abs_scaled[i]
                top_indices = np.argsort(sample_scores)[::-1][:self.top_k]
                top_features = data.columns[top_indices].tolist()
                feature_orders.append(top_features)
                
                # Store the top features for this sample
                for j, feature in enumerate(top_features):
                    if feature not in ranked_data:
                        ranked_data[feature] = [None] * len(data)
                    ranked_data[feature][i] = data.iloc[i][feature]
            
            result_df = pd.DataFrame(ranked_data, index=data.index)
            return result_df, feature_orders


class CancerSequenceGenerator(BaseSequenceGenerator):
    """Generates sequences from cancer diagnosis data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sequence generator.
        
        Args:
            config: Configuration dictionary with keys:
                - max_sequence_length: Maximum length of sequences
                - max_gap: Maximum gap between items in sequence
                - ranking_method: Method for ranking features ('mutual_info', 'zscore')
                - discretization_strategy: Strategy for binning ('uniform', 'quantile', 'kmeans')
                - n_bins: Number of bins for discretization
                - top_k: Number of top features to consider
        """
        super().__init__(config)
        self.max_sequence_length = self.config.get('max_sequence_length', 5)
        self.max_gap = self.config.get('max_gap', 1)
        self.ranking_method = self.config.get('ranking_method', 'mutual_info')
        self.discretization_strategy = self.config.get('discretization_strategy', 'quantile')
        self.n_bins = self.config.get('n_bins', 3)
        self.top_k = self.config.get('top_k', 10)
        
        # Initialize components
        self.discretizer = FeatureDiscretizer({
            'n_bins': self.n_bins,
            'strategy': self.discretization_strategy,
            'encode': 'ordinal'
        })
        
        self.ranker = FeatureRanker({
            'method': self.ranking_method,
            'top_k': self.top_k
        })
        
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CancerSequenceGenerator':
        """
        Fit the sequence generator.
        
        Args:
            data: Input features
            target: Target variable
            
        Returns:
            Self for method chaining
        """
        # First discretize the features
        self.discretizer.fit(data, target)
        discretized_data = self.discretizer.transform_numeric(data)  # Use numeric for ranking
        
        # Then fit the ranker
        self.ranker.fit(discretized_data, target)
        
        self.is_fitted = True
        return self
    
    def generate_sequences(self, data: pd.DataFrame) -> List[List[List[str]]]:
        """
        Generate sequences from the data.
        
        Args:
            data: Input features
            
        Returns:
            List of sequences for each sample
        """
        self.validate_fitted()
        
        # Discretize features
        discretized_data = self.discretizer.transform(data)  # String labels for final sequences
        discretized_data_numeric = self.discretizer.transform_numeric(data)  # Numeric for ranking
        
        sequences = []
        
        if self.ranking_method == 'mutual_info':
            # Get top-k features based on mutual information
            ranked_data, top_features = self.ranker.transform(discretized_data_numeric)
            
            for i in range(len(discretized_data)):
                sequence = []
                for feature in top_features[:self.max_sequence_length]:
                    if feature in discretized_data.columns:
                        value = discretized_data.iloc[i][feature]
                        if pd.notna(value):
                            # Create itemset with feature and its value
                            itemset = [f"{feature}_{value}"]
                            sequence.append(itemset)
                
                sequences.append(sequence)
                
        elif self.ranking_method == 'zscore':
            # Get feature ranking for each sample
            _, feature_orders = self.ranker.transform(discretized_data_numeric)
            
            for i, sample_features in enumerate(feature_orders):
                sequence = []
                for j, feature in enumerate(sample_features[:self.max_sequence_length]):
                    value = discretized_data.iloc[i][feature]  # Use string labels for final sequence
                    if pd.notna(value):
                        # Create itemset with feature and its value
                        itemset = [f"{feature}_{value}"]
                        sequence.append(itemset)
                
                sequences.append(sequence)
        
        return sequences