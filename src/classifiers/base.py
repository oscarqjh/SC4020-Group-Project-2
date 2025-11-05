"""
Base classes for binary classification components.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path


class BaseClassifier(ABC):
    """Abstract base class for binary classification."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration dictionary for classifier parameters
        """
        self.config = config or {}
        self.is_fitted = False
        self.model = None
        self.feature_names = None
        self.classes_ = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseClassifier':
        """
        Fit the classifier to the training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series with class labels
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability estimates for each class.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of class probabilities (shape: n_samples x n_classes)
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using F1, Recall, Precision, and ROC-AUC metrics.
        
        Args:
            X: Feature DataFrame
            y: True labels Series
            
        Returns:
            Dictionary with metric names as keys and scores as values
        """
        pass
    
    def save_model(self, filepath: str) -> str:
        """
        Save the trained classifier to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            String path where the model was saved (resolved absolute path)
        """
        self.validate_fitted()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        return str(filepath.resolve())
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseClassifier':
        """
        Load a trained classifier from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded classifier instance
        """
        try:
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
            
            # Validate that loaded object is an instance of cls
            if not isinstance(classifier, cls):
                raise TypeError(
                    f"Loaded object is not an instance of {cls.__name__}. "
                    f"Got {type(classifier).__name__} instead."
                )
            
            return classifier
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
    
    def validate_fitted(self) -> None:
        """Validate that the classifier has been trained."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions or saving.")
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retrieve classifier configuration parameters.
        
        Returns:
            Copy of the configuration dictionary
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseClassifier':
        """
        Update classifier configuration parameters.
        
        Args:
            **params: Keyword arguments to update in config
            
        Returns:
            Self for method chaining
        """
        self.config.update(params)
        return self

