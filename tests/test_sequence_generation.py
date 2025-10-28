"""
Test suite for sequence generation components.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processors.sequence_generator import FeatureDiscretizer, FeatureRanker, CancerSequenceGenerator


class TestFeatureDiscretizer(unittest.TestCase):
    """Test cases for FeatureDiscretizer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.uniform(0, 10, 100)
        })
        
    def test_quantile_discretization(self):
        """Test quantile-based discretization."""
        discretizer = FeatureDiscretizer({'strategy': 'quantile', 'n_bins': 3})
        discretizer.fit(self.data)
        
        result = discretizer.transform(self.data)
        
        # Check output format
        self.assertEqual(result.shape, self.data.shape)
        self.assertTrue(all(col in result.columns for col in self.data.columns))
        
        # Check label format
        unique_labels = set()
        for col in result.columns:
            unique_labels.update(result[col].unique())
        
        expected_labels = {'low', 'medium', 'high'}
        self.assertTrue(unique_labels.issubset(expected_labels))
        
    def test_uniform_discretization(self):
        """Test uniform discretization."""
        discretizer = FeatureDiscretizer({'strategy': 'uniform', 'n_bins': 5})
        discretizer.fit(self.data)
        
        result = discretizer.transform(self.data)
        
        # Check that we get 5 bins
        unique_labels = set()
        for col in result.columns:
            unique_labels.update(result[col].unique())
        
        # Should have bin_0, bin_1, ..., bin_4
        expected_labels = {f'bin_{i}' for i in range(5)}
        self.assertTrue(unique_labels.issubset(expected_labels))


class TestFeatureRanker(unittest.TestCase):
    """Test cases for FeatureRanker."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.uniform(0, 10, 100),
            'feature4': np.random.exponential(2, 100)
        })
        # Create target correlated with feature1
        self.target = pd.Series(['M' if x > 0 else 'B' for x in self.data['feature1']])
        
    def test_mutual_info_ranking(self):
        """Test mutual information ranking."""
        ranker = FeatureRanker({'method': 'mutual_info', 'top_k': 3})
        ranker.fit(self.data, self.target)
        
        ranked_data, top_features = ranker.transform(self.data)
        
        # Check output
        self.assertEqual(len(top_features), 3)
        self.assertEqual(ranked_data.shape[1], 3)
        self.assertTrue(all(feat in self.data.columns for feat in top_features))
        
    def test_zscore_ranking(self):
        """Test z-score ranking."""
        ranker = FeatureRanker({'method': 'zscore', 'top_k': 2})
        ranker.fit(self.data)
        
        ranked_data, feature_orders = ranker.transform(self.data)
        
        # Check output
        self.assertEqual(len(feature_orders), len(self.data))
        self.assertTrue(all(len(order) == 2 for order in feature_orders))


class TestCancerSequenceGenerator(unittest.TestCase):
    """Test cases for CancerSequenceGenerator."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Create synthetic cancer-like data
        self.data = pd.DataFrame({
            'radius_mean': np.random.normal(14, 3, 50),
            'texture_mean': np.random.normal(19, 4, 50),
            'smoothness_mean': np.random.normal(0.1, 0.02, 50),
            'compactness_mean': np.random.normal(0.1, 0.05, 50),
            'concavity_mean': np.random.normal(0.08, 0.04, 50)
        })
        
        # Create target with some correlation
        self.target = pd.Series(['M' if (row['radius_mean'] > 14 and row['texture_mean'] > 19) 
                               else 'B' for _, row in self.data.iterrows()])
        
    def test_sequence_generation_mutual_info(self):
        """Test sequence generation with mutual info ranking."""
        config = {
            'ranking_method': 'mutual_info',
            'discretization_strategy': 'quantile',
            'n_bins': 3,
            'top_k': 3,
            'max_sequence_length': 3
        }
        
        generator = CancerSequenceGenerator(config)
        sequences = generator.fit_generate(self.data, self.target)
        
        # Check output format
        self.assertEqual(len(sequences), len(self.data))
        self.assertTrue(all(isinstance(seq, list) for seq in sequences))
        self.assertTrue(all(isinstance(itemset, list) for seq in sequences for itemset in seq))
        
        # Check sequence content
        for seq in sequences:
            self.assertLessEqual(len(seq), 3)  # max_sequence_length
            for itemset in seq:
                for item in itemset:
                    self.assertTrue('_' in item)  # Should be feature_value format
                    
    def test_sequence_generation_zscore(self):
        """Test sequence generation with z-score ranking."""
        config = {
            'ranking_method': 'zscore',
            'discretization_strategy': 'uniform',
            'n_bins': 3,
            'top_k': 4,
            'max_sequence_length': 4
        }
        
        generator = CancerSequenceGenerator(config)
        sequences = generator.fit_generate(self.data, self.target)
        
        # Check output format
        self.assertEqual(len(sequences), len(self.data))
        
        # Check that sequences can vary per sample (z-score is sample-specific)
        sequence_lengths = [len(seq) for seq in sequences]
        self.assertTrue(any(length > 0 for length in sequence_lengths))


if __name__ == '__main__':
    unittest.main()