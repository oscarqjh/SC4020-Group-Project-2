"""
Base classes for data preprocessing components.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


class BasePreprocessor(ABC):
    """Abstract base class for data preprocessing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'BasePreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Input data
            target: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> Any:
        """
        Transform the data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> Any:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: Input data
            target: Target variable (optional)
            
        Returns:
            Transformed data
        """
        return self.fit(data, target).transform(data)
    
    def validate_fitted(self):
        """Check if the preprocessor has been fitted."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation.")


class BaseSequenceGenerator(ABC):
    """Abstract base class for sequence generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sequence generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'BaseSequenceGenerator':
        """
        Fit the sequence generator to the data.
        
        Args:
            data: Input data
            target: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def generate_sequences(self, data: pd.DataFrame) -> List[List[List[str]]]:
        """
        Generate sequences from the data.
        
        Args:
            data: Input data
            
        Returns:
            List of sequences, where each sequence is a list of itemsets,
            and each itemset is a list of items
        """
        pass
    
    def fit_generate(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> List[List[List[str]]]:
        """
        Fit the generator and generate sequences.
        
        Args:
            data: Input data
            target: Target variable (optional)
            
        Returns:
            Generated sequences
        """
        return self.fit(data, target).generate_sequences(data)
    
    def validate_fitted(self):
        """Check if the generator has been fitted."""
        if not self.is_fitted:
            raise ValueError("Sequence generator must be fitted before generating sequences.")
