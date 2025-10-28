"""
Base classes for data analysis components.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Union


class BaseAnalyzer(ABC):
    """Abstract base class for data analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.results = None
        
    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Perform analysis on the data.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        pass
    
    def save_results(self, filepath: str):
        """
        Save analysis results to file.
        
        Args:
            filepath: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Run analyze() first.")
        
        # Implementation depends on the specific analyzer
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.
        
        Returns:
            Summary dictionary
        """
        if self.results is None:
            raise ValueError("No results available. Run analyze() first.")
        
        return {"status": "completed", "results_available": True}


class BasePatternMiner(ABC):
    """Abstract base class for pattern mining algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pattern miner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.patterns = None
        self.is_fitted = False
        
    @abstractmethod
    def mine_patterns(self, sequences: List[List[List[str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Mine patterns from sequences.
        
        Args:
            sequences: List of sequences to mine patterns from
            **kwargs: Additional parameters
            
        Returns:
            List of discovered patterns
        """
        pass
    
    def get_frequent_patterns(self, min_support: float = 0.1) -> List[Dict[str, Any]]:
        """
        Get patterns that meet minimum support threshold.
        
        Args:
            min_support: Minimum support threshold
            
        Returns:
            List of frequent patterns
        """
        if self.patterns is None:
            raise ValueError("No patterns available. Run mine_patterns() first.")
        
        return [p for p in self.patterns if p.get('support', 0) >= min_support]
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of discovered patterns.
        
        Returns:
            Summary dictionary
        """
        if self.patterns is None:
            raise ValueError("No patterns available. Run mine_patterns() first.")
        
        return {
            "total_patterns": len(self.patterns),
            "avg_support": sum(p.get('support', 0) for p in self.patterns) / len(self.patterns),
            "max_support": max(p.get('support', 0) for p in self.patterns),
            "min_support": min(p.get('support', 0) for p in self.patterns)
        }


class BaseClassificationAnalyzer(ABC):
    """Abstract base class for classification analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the classification analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.class_patterns = {}
        self.performance_metrics = {}
        
    @abstractmethod
    def analyze_by_class(self, sequences: List[List[List[str]]], 
                        labels: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns by class.
        
        Args:
            sequences: List of sequences
            labels: Class labels for each sequence
            
        Returns:
            Analysis results by class
        """
        pass
    
    def compare_classes(self) -> Dict[str, Any]:
        """
        Compare patterns between different classes.
        
        Returns:
            Comparison results
        """
        if not self.class_patterns:
            raise ValueError("No class patterns available. Run analyze_by_class() first.")
        
        return {"comparison_available": True}
    
    def get_discriminative_patterns(self, min_confidence: float = 0.7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get patterns that are discriminative for each class.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            Discriminative patterns by class
        """
        if not self.class_patterns:
            raise ValueError("No class patterns available. Run analyze_by_class() first.")
        
        return {cls: [] for cls in self.class_patterns.keys()}
