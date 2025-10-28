"""
Test suite for pattern mining components.
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.pattern_mining import GSPMiner, CancerPatternAnalyzer, SequentialPatternAnalyzer


class TestGSPMiner(unittest.TestCase):
    """Test cases for GSP algorithm implementation."""
    
    def setUp(self):
        """Set up test sequences."""
        # Simple test sequences
        self.sequences = [
            [['a'], ['b'], ['c']],  # Sequence 1: a → b → c
            [['a'], ['b']],         # Sequence 2: a → b
            [['a'], ['c']],         # Sequence 3: a → c
            [['b'], ['c']],         # Sequence 4: b → c
            [['a'], ['b'], ['c']],  # Sequence 5: a → b → c
        ]
        
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        miner = GSPMiner({'min_support': 0.2})
        
        # Test basic pattern matching
        pattern = [['a'], ['b']]
        sequence = [['a'], ['b'], ['c']]
        
        self.assertTrue(miner._pattern_matches_sequence(pattern, sequence))
        
        # Test non-matching pattern
        pattern = [['a'], ['d']]
        self.assertFalse(miner._pattern_matches_sequence(pattern, sequence))
        
    def test_support_calculation(self):
        """Test support calculation."""
        miner = GSPMiner({'min_support': 0.2})
        
        # Pattern [a] should appear in 3/5 sequences
        pattern = [['a']]
        support = miner._get_support(pattern, self.sequences)
        self.assertAlmostEqual(support, 0.6, places=2)
        
        # Pattern [a, b] should appear in 3/5 sequences
        pattern = [['a'], ['b']]
        support = miner._get_support(pattern, self.sequences)
        self.assertAlmostEqual(support, 0.6, places=2)
        
    def test_pattern_mining(self):
        """Test complete pattern mining."""
        miner = GSPMiner({'min_support': 0.4, 'max_pattern_length': 3})
        
        patterns = miner.mine_patterns(self.sequences)
        
        # Check that we get some patterns
        self.assertGreater(len(patterns), 0)
        
        # Check that all patterns meet minimum support
        for pattern_info in patterns:
            self.assertGreaterEqual(pattern_info['support'], 0.4)
            
        # Check pattern format
        for pattern_info in patterns:
            self.assertIn('pattern', pattern_info)
            self.assertIn('support', pattern_info)
            self.assertIn('length', pattern_info)


class TestCancerPatternAnalyzer(unittest.TestCase):
    """Test cases for CancerPatternAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic sequences with labels
        self.sequences = [
            [['radius_high'], ['texture_high']],     # Malignant pattern
            [['radius_high'], ['compactness_high']], # Malignant pattern
            [['radius_low'], ['smoothness_low']],    # Benign pattern
            [['radius_low'], ['texture_low']],       # Benign pattern
            [['radius_high'], ['texture_high']],     # Malignant pattern
        ]
        
        self.labels = ['M', 'M', 'B', 'B', 'M']
        
    def test_analyze_by_class(self):
        """Test class-based analysis."""
        analyzer = CancerPatternAnalyzer({'min_support': 0.3})
        
        results = analyzer.analyze_by_class(self.sequences, self.labels)
        
        # Check structure
        self.assertIn('malignant', results)
        self.assertIn('benign', results)
        
        # Check that we have patterns for each class
        self.assertGreater(results['malignant']['total_sequences'], 0)
        self.assertGreater(results['benign']['total_sequences'], 0)
        
    def test_compare_classes(self):
        """Test class comparison."""
        analyzer = CancerPatternAnalyzer({'min_support': 0.3})
        
        # First analyze by class
        analyzer.analyze_by_class(self.sequences, self.labels)
        
        # Then compare
        comparison = analyzer.compare_classes()
        
        # Check structure
        self.assertIn('unique_to_malignant', comparison)
        self.assertIn('unique_to_benign', comparison)
        self.assertIn('discriminative_patterns', comparison)
        
    def test_feature_importance(self):
        """Test feature importance analysis."""
        analyzer = CancerPatternAnalyzer({'min_support': 0.3})
        
        # First analyze by class
        analyzer.analyze_by_class(self.sequences, self.labels)
        
        # Get feature importance
        importance = analyzer.get_feature_importance()
        
        # Check structure
        self.assertIn('top_malignant_features', importance)
        self.assertIn('top_benign_features', importance)


class TestSequentialPatternAnalyzer(unittest.TestCase):
    """Test cases for SequentialPatternAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        self.sequences = [
            [['radius_high'], ['texture_high']],
            [['radius_low'], ['smoothness_low']],
            [['radius_high'], ['compactness_high']],
        ]
        
        self.labels = ['M', 'B', 'M']
        
    def test_analyze(self):
        """Test complete analysis."""
        analyzer = SequentialPatternAnalyzer({'min_support': 0.3})
        
        results = analyzer.analyze(self.sequences, self.labels)
        
        # Check structure
        self.assertIn('class_analysis', results)
        self.assertIn('class_comparison', results)
        self.assertIn('feature_importance', results)
        self.assertIn('summary', results)
        
        # Check summary
        summary = results['summary']
        self.assertEqual(summary['total_sequences'], 3)
        self.assertEqual(summary['malignant_count'], 2)
        self.assertEqual(summary['benign_count'], 1)
        
    def test_interpretable_results(self):
        """Test interpretable results generation."""
        analyzer = SequentialPatternAnalyzer({'min_support': 0.3})
        
        # First run analysis
        analyzer.analyze(self.sequences, self.labels)
        
        # Get interpretable results
        interpretable = analyzer.get_interpretable_results()
        
        # Check structure
        self.assertIn('interpretations', interpretable)
        self.assertIn('summary', interpretable)


if __name__ == '__main__':
    unittest.main()