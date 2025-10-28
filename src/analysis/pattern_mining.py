"""
Sequential pattern mining implementation for cancer diagnosis analysis.
"""
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from .base import BaseAnalyzer, BasePatternMiner, BaseClassificationAnalyzer


class GSPMiner(BasePatternMiner):
    """Generalized Sequential Pattern (GSP) algorithm implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GSP miner.
        
        Args:
            config: Configuration dictionary with keys:
                - min_support: Minimum support threshold (default: 0.1)
                - max_pattern_length: Maximum pattern length (default: 5)
                - max_gap: Maximum gap between elements (default: 1)
        """
        super().__init__(config)
        self.min_support = self.config.get('min_support', 0.1)
        self.max_pattern_length = self.config.get('max_pattern_length', 5)
        self.max_gap = self.config.get('max_gap', 1)
        
    def _get_support(self, pattern: List[List[str]], sequences: List[List[List[str]]]) -> float:
        """
        Calculate support for a pattern.
        
        Args:
            pattern: Pattern to check
            sequences: List of sequences
            
        Returns:
            Support value (between 0 and 1)
        """
        count = 0
        for sequence in sequences:
            if self._pattern_matches_sequence(pattern, sequence):
                count += 1
        return count / len(sequences) if sequences else 0
    
    def _pattern_matches_sequence(self, pattern: List[List[str]], sequence: List[List[str]]) -> bool:
        """
        Check if a pattern matches a sequence.
        
        Args:
            pattern: Pattern to match
            sequence: Sequence to check against
            
        Returns:
            True if pattern matches sequence
        """
        if not pattern or not sequence:
            return len(pattern) == 0
        
        pattern_idx = 0
        sequence_idx = 0
        
        while pattern_idx < len(pattern) and sequence_idx < len(sequence):
            # Check if current pattern itemset matches current sequence itemset
            if self._itemset_contained_in(pattern[pattern_idx], sequence[sequence_idx]):
                pattern_idx += 1
                if pattern_idx < len(pattern):
                    sequence_idx += 1  # Move to next position for next pattern element
            else:
                sequence_idx += 1
                
        return pattern_idx == len(pattern)
    
    def _itemset_contained_in(self, pattern_itemset: List[str], sequence_itemset: List[str]) -> bool:
        """
        Check if pattern itemset is contained in sequence itemset.
        
        Args:
            pattern_itemset: Pattern itemset
            sequence_itemset: Sequence itemset
            
        Returns:
            True if pattern itemset is contained in sequence itemset
        """
        return all(item in sequence_itemset for item in pattern_itemset)
    
    def _generate_candidates(self, frequent_patterns: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Generate candidate patterns of length k+1 from frequent patterns of length k.
        
        Args:
            frequent_patterns: List of frequent patterns
            
        Returns:
            List of candidate patterns
        """
        candidates = []
        
        for i, pattern1 in enumerate(frequent_patterns):
            for j, pattern2 in enumerate(frequent_patterns):
                if i != j:
                    # Try to join patterns
                    candidate = self._join_patterns(pattern1, pattern2)
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
        
        return candidates
    
    def _join_patterns(self, pattern1: List[List[str]], pattern2: List[List[str]]) -> Optional[List[List[str]]]:
        """
        Join two patterns to create a candidate pattern.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Joined pattern or None if cannot join
        """
        if not pattern1 or not pattern2:
            return None
        
        # Try sequence extension (add new itemset)
        if pattern1[:-1] == pattern2[1:]:
            return pattern1 + [pattern2[-1]]
        
        # Try itemset extension (add item to last itemset)
        if pattern1[:-1] == pattern2[:-1]:
            last_itemset1 = set(pattern1[-1])
            last_itemset2 = set(pattern2[-1])
            if len(last_itemset1.symmetric_difference(last_itemset2)) == 2:
                new_itemset = sorted(list(last_itemset1.union(last_itemset2)))
                return pattern1[:-1] + [new_itemset]
        
        return None
    
    def mine_patterns(self, sequences: List[List[List[str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Mine sequential patterns using GSP algorithm.
        
        Args:
            sequences: List of sequences to mine
            **kwargs: Additional parameters
            
        Returns:
            List of discovered patterns with support values
        """
        if not sequences:
            return []
        
        all_patterns = []
        
        # Generate 1-itemset candidates
        items = set()
        for sequence in sequences:
            for itemset in sequence:
                items.update(itemset)
        
        # Find frequent 1-itemsets
        frequent_1 = []
        for item in items:
            pattern = [[item]]
            support = self._get_support(pattern, sequences)
            if support >= self.min_support:
                frequent_1.append(pattern)
                all_patterns.append({
                    'pattern': pattern,
                    'support': support,
                    'length': 1
                })
        
        current_frequent = frequent_1
        length = 1
        
        # Generate longer patterns
        while current_frequent and length < self.max_pattern_length:
            candidates = self._generate_candidates(current_frequent)
            next_frequent = []
            
            for candidate in candidates:
                support = self._get_support(candidate, sequences)
                if support >= self.min_support:
                    next_frequent.append(candidate)
                    all_patterns.append({
                        'pattern': candidate,
                        'support': support,
                        'length': length + 1
                    })
            
            current_frequent = next_frequent
            length += 1
        
        # Sort patterns by support (descending)
        all_patterns.sort(key=lambda x: x['support'], reverse=True)
        self.patterns = all_patterns
        return all_patterns


class CancerPatternAnalyzer(BaseClassificationAnalyzer):
    """Analyzer for cancer diagnosis patterns."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cancer pattern analyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.gsp_miner = GSPMiner(config)
        
    def analyze_by_class(self, sequences: List[List[List[str]]], 
                        labels: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns by diagnosis class (malignant vs benign).
        
        Args:
            sequences: List of sequences
            labels: Diagnosis labels ('M' for malignant, 'B' for benign)
            
        Returns:
            Analysis results by class
        """
        # Separate sequences by class
        malignant_sequences = [seq for seq, label in zip(sequences, labels) if label == 'M']
        benign_sequences = [seq for seq, label in zip(sequences, labels) if label == 'B']
        
        # Mine patterns for each class
        malignant_patterns = self.gsp_miner.mine_patterns(malignant_sequences)
        benign_patterns = self.gsp_miner.mine_patterns(benign_sequences)
        
        # Store class patterns
        self.class_patterns = {
            'malignant': malignant_patterns,
            'benign': benign_patterns
        }
        
        # Calculate class-specific metrics
        results = {
            'malignant': {
                'total_sequences': len(malignant_sequences),
                'total_patterns': len(malignant_patterns),
                'top_patterns': malignant_patterns[:10] if malignant_patterns else []
            },
            'benign': {
                'total_sequences': len(benign_sequences),
                'total_patterns': len(benign_patterns),
                'top_patterns': benign_patterns[:10] if benign_patterns else []
            }
        }
        
        return results
    
    def compare_classes(self) -> Dict[str, Any]:
        """
        Compare patterns between malignant and benign cases.
        
        Returns:
            Comparison results including discriminative patterns
        """
        if not self.class_patterns:
            raise ValueError("No class patterns available. Run analyze_by_class() first.")
        
        malignant_patterns = self.class_patterns.get('malignant', [])
        benign_patterns = self.class_patterns.get('benign', [])
        
        # Find patterns unique to each class
        malignant_pattern_strings = {str(p['pattern']): p for p in malignant_patterns}
        benign_pattern_strings = {str(p['pattern']): p for p in benign_patterns}
        
        unique_to_malignant = []
        unique_to_benign = []
        common_patterns = []
        
        for pattern_str, pattern_info in malignant_pattern_strings.items():
            if pattern_str in benign_pattern_strings:
                common_patterns.append({
                    'pattern': pattern_info['pattern'],
                    'malignant_support': pattern_info['support'],
                    'benign_support': benign_pattern_strings[pattern_str]['support']
                })
            else:
                unique_to_malignant.append(pattern_info)
        
        for pattern_str, pattern_info in benign_pattern_strings.items():
            if pattern_str not in malignant_pattern_strings:
                unique_to_benign.append(pattern_info)
        
        # Calculate discriminative power
        discriminative_patterns = []
        for pattern in common_patterns:
            mal_support = pattern['malignant_support']
            ben_support = pattern['benign_support']
            
            # Calculate lift (how much more likely in one class vs other)
            if ben_support > 0:
                malignant_lift = mal_support / ben_support
                benign_lift = ben_support / mal_support
                
                discriminative_patterns.append({
                    'pattern': pattern['pattern'],
                    'malignant_support': mal_support,
                    'benign_support': ben_support,
                    'malignant_lift': malignant_lift,
                    'benign_lift': benign_lift,
                    'discriminative_for': 'malignant' if malignant_lift > 1.5 else 'benign' if benign_lift > 1.5 else 'neutral'
                })
        
        return {
            'unique_to_malignant': unique_to_malignant[:10],
            'unique_to_benign': unique_to_benign[:10],
            'discriminative_patterns': sorted(discriminative_patterns, 
                                            key=lambda x: max(x['malignant_lift'], x['benign_lift']), 
                                            reverse=True)[:10],
            'common_patterns_count': len(common_patterns),
            'total_malignant_patterns': len(malignant_patterns),
            'total_benign_patterns': len(benign_patterns)
        }
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze feature importance based on pattern frequency.
        
        Returns:
            Feature importance analysis
        """
        if not self.class_patterns:
            raise ValueError("No class patterns available. Run analyze_by_class() first.")
        
        # Count feature occurrences in patterns
        malignant_features = Counter()
        benign_features = Counter()
        
        for pattern_info in self.class_patterns.get('malignant', []):
            pattern = pattern_info['pattern']
            support = pattern_info['support']
            for itemset in pattern:
                for item in itemset:
                    feature = item.split('_')[0]  # Extract feature name
                    malignant_features[feature] += support
        
        for pattern_info in self.class_patterns.get('benign', []):
            pattern = pattern_info['pattern']
            support = pattern_info['support']
            for itemset in pattern:
                for item in itemset:
                    feature = item.split('_')[0]  # Extract feature name
                    benign_features[feature] += support
        
        # Get top features for each class
        top_malignant_features = malignant_features.most_common(10)
        top_benign_features = benign_features.most_common(10)
        
        return {
            'top_malignant_features': top_malignant_features,
            'top_benign_features': top_benign_features,
            'all_malignant_features': dict(malignant_features),
            'all_benign_features': dict(benign_features)
        }


class SequentialPatternAnalyzer(BaseAnalyzer):
    """Main analyzer for sequential pattern mining in cancer diagnosis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the sequential pattern analyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.cancer_analyzer = CancerPatternAnalyzer(config)
        
    def analyze(self, sequences: List[List[List[str]]], labels: List[str], **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive sequential pattern analysis.
        
        Args:
            sequences: List of sequences
            labels: Class labels
            **kwargs: Additional parameters
            
        Returns:
            Complete analysis results
        """
        # Analyze patterns by class
        class_analysis = self.cancer_analyzer.analyze_by_class(sequences, labels)
        
        # Compare classes
        comparison = self.cancer_analyzer.compare_classes()
        
        # Get feature importance
        feature_importance = self.cancer_analyzer.get_feature_importance()
        
        # Compile results
        self.results = {
            'class_analysis': class_analysis,
            'class_comparison': comparison,
            'feature_importance': feature_importance,
            'summary': {
                'total_sequences': len(sequences),
                'malignant_count': sum(1 for label in labels if label == 'M'),
                'benign_count': sum(1 for label in labels if label == 'B'),
                'unique_malignant_patterns': len(comparison.get('unique_to_malignant', [])),
                'unique_benign_patterns': len(comparison.get('unique_to_benign', [])),
                'discriminative_patterns': len(comparison.get('discriminative_patterns', []))
            }
        }
        
        return self.results
    
    def get_interpretable_results(self) -> Dict[str, Any]:
        """
        Get human-readable interpretation of results.
        
        Returns:
            Interpretable results
        """
        if self.results is None:
            raise ValueError("No results available. Run analyze() first.")
        
        interpretations = []
        
        # Interpret discriminative patterns
        discriminative_patterns = self.results['class_comparison'].get('discriminative_patterns', [])
        for pattern_info in discriminative_patterns[:5]:  # Top 5 most discriminative
            pattern = pattern_info['pattern']
            discriminative_for = pattern_info['discriminative_for']
            
            if discriminative_for != 'neutral':
                pattern_str = ' → '.join(['{' + ', '.join(itemset) + '}' for itemset in pattern])
                lift = pattern_info.get(f'{discriminative_for}_lift', 1)
                
                interpretations.append({
                    'pattern': pattern_str,
                    'interpretation': f"Pattern strongly associated with {discriminative_for} cases (lift: {lift:.2f})",
                    'type': 'discriminative'
                })
        
        # Interpret unique patterns
        unique_malignant = self.results['class_comparison'].get('unique_to_malignant', [])
        for pattern_info in unique_malignant[:3]:  # Top 3 unique to malignant
            pattern = pattern_info['pattern']
            pattern_str = ' → '.join(['{' + ', '.join(itemset) + '}' for itemset in pattern])
            support = pattern_info['support']
            
            interpretations.append({
                'pattern': pattern_str,
                'interpretation': f"Pattern unique to malignant cases (support: {support:.3f})",
                'type': 'unique_malignant'
            })
        
        unique_benign = self.results['class_comparison'].get('unique_to_benign', [])
        for pattern_info in unique_benign[:3]:  # Top 3 unique to benign
            pattern = pattern_info['pattern']
            pattern_str = ' → '.join(['{' + ', '.join(itemset) + '}' for itemset in pattern])
            support = pattern_info['support']
            
            interpretations.append({
                'pattern': pattern_str,
                'interpretation': f"Pattern unique to benign cases (support: {support:.3f})",
                'type': 'unique_benign'
            })
        
        return {
            'interpretations': interpretations,
            'summary': self.results['summary']
        }