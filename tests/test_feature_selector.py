"""
Test suite for feature selection components.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import tempfile
from pathlib import Path

# Add parent directory to path (so we can import from src package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers.feature_selector import FeatureSelector


class TestFeatureSelector(unittest.TestCase):
    """Test cases for FeatureSelector."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic dataset with correlated features
        n_samples = 100
        
        # Create radius values
        radius = np.random.normal(14, 3, n_samples)
        
        # Create highly correlated features
        # Perimeter ≈ 2π × radius + noise
        perimeter = 2 * np.pi * radius + np.random.normal(0, 0.1, n_samples)
        
        # Area ≈ π × radius² + noise
        area = np.pi * radius ** 2 + np.random.normal(0, 1, n_samples)
        
        # Independent features
        concave_points = np.random.normal(0.1, 0.05, n_samples)
        texture = np.random.normal(19, 4, n_samples)
        smoothness = np.random.normal(0.1, 0.02, n_samples)
        
        self.data = pd.DataFrame({
            'radius_mean': radius,
            'perimeter_mean': perimeter,
            'area_mean': area,
            'concave points_mean': concave_points,
            'texture_mean': texture,
            'smoothness_mean': smoothness
        })
        
        # Create temporary feature importance file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        self.feature_importance_path = self.temp_file.name
        
        feature_importance_content = """Feature Importance Analysis
==============================

Top Features in Malignant Patterns:
  concave points: 7.5
  radius: 4.0
  perimeter: 3.8
  area: 3.7
  texture: 2.5

Top Features in Benign Patterns:
  concave points: 9.0
  radius: 4.5
  area: 4.3
  perimeter: 4.2
  smoothness: 3.0
"""
        self.temp_file.write(feature_importance_content)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        import os
        if os.path.exists(self.feature_importance_path):
            os.unlink(self.feature_importance_path)
    
    def test_initialization(self):
        """Test that FeatureSelector initializes correctly."""
        # Test default initialization
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path
        })
        
        self.assertEqual(selector.top_n, 10)
        self.assertEqual(selector.correlation_threshold, 0.9)
        self.assertEqual(selector.aggregation_method, 'sum')
        
        # Test custom configuration
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 5,
            'correlation_threshold': 0.85,
            'aggregation_method': 'mean'
        })
        
        self.assertEqual(selector.top_n, 5)
        self.assertEqual(selector.correlation_threshold, 0.85)
        self.assertEqual(selector.aggregation_method, 'mean')
        
        # Test missing required parameter (should raise error during fit)
        selector = FeatureSelector({})
        with self.assertRaises(ValueError):
            selector.fit(self.data)
    
    def test_feature_score_aggregation(self):
        """Test that feature scores are correctly combined."""
        # Test 'sum' aggregation
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'aggregation_method': 'sum'
        })
        selector.fit(self.data)
        
        scores = selector.get_feature_scores()
        # 'concave points' should have score 7.5 + 9.0 = 16.5
        self.assertAlmostEqual(scores.get('concave points_mean', 0), 16.5, places=1)
        # 'radius' should have score 4.0 + 4.5 = 8.5
        self.assertAlmostEqual(scores.get('radius_mean', 0), 8.5, places=1)
        
        # Test 'mean' aggregation
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'aggregation_method': 'mean'
        })
        selector.fit(self.data)
        
        scores = selector.get_feature_scores()
        # 'concave points' should have score (7.5 + 9.0) / 2 = 8.25
        self.assertAlmostEqual(scores.get('concave points_mean', 0), 8.25, places=1)
        
        # Test 'max' aggregation
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'aggregation_method': 'max'
        })
        selector.fit(self.data)
        
        scores = selector.get_feature_scores()
        # 'concave points' should have score max(7.5, 9.0) = 9.0
        self.assertAlmostEqual(scores.get('concave points_mean', 0), 9.0, places=1)
    
    def test_multicollinearity_detection(self):
        """Test that correlated features are correctly identified."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'correlation_threshold': 0.9,
            'top_n': 10
        })
        selector.fit(self.data)
        
        selected_features = selector.get_selected_features()
        removed_features = selector.get_removed_features()
        
        # Check that only ONE of {radius_mean, perimeter_mean, area_mean} is selected
        correlated_group = ['radius_mean', 'perimeter_mean', 'area_mean']
        selected_from_group = [f for f in selected_features if f in correlated_group]
        self.assertLessEqual(len(selected_from_group), 1, 
                           "Should select at most one feature from correlated group")
        
        # Check that removed features are recorded
        if len(selected_from_group) == 1:
            # The other two should be in removed_features
            removed_from_group = [f['feature'] for f in removed_features 
                                if f['feature'] in correlated_group]
            # At least one other should be removed
            self.assertGreaterEqual(len(removed_from_group), 1)
        
        # Verify that the highest-scored feature is kept within correlated groups
        if len(selected_from_group) > 0:
            # Compute expected highest-scored column among correlated group
            feature_scores = selector.get_feature_scores()
            group_scores = {
                feature: feature_scores.get(feature, 0.0)
                for feature in correlated_group
                if feature in feature_scores
            }
            
            if group_scores:
                expected_highest = max(group_scores.items(), key=lambda x: x[1])[0]
                # Assert that the expected highest-scored feature is in selected_features
                self.assertIn(expected_highest, selected_features,
                            f"Highest-scored feature {expected_highest} from correlated group "
                            f"should be selected")
    
    def test_feature_name_matching(self):
        """Test that base feature names correctly match dataset columns."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path
        })
        selector.fit(self.data)
        
        scores = selector.get_feature_scores()
        
        # Verify that base features match columns
        self.assertIn('concave points_mean', scores)
        self.assertIn('radius_mean', scores)
        self.assertIn('texture_mean', scores)
    
    def test_transform(self):
        """Test that transform correctly filters the dataset."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 3
        })
        selector.fit(self.data)
        
        transformed = selector.transform(self.data)
        
        # Check output shape
        self.assertEqual(transformed.shape[1], 3)
        self.assertEqual(transformed.shape[0], self.data.shape[0])
        
        # Check that all columns are in selected_features
        selected_features = selector.get_selected_features()
        self.assertEqual(set(transformed.columns), set(selected_features))
        
        # Check that original index is preserved
        self.assertTrue(transformed.index.equals(self.data.index))
    
    def test_fit_transform(self):
        """Test the convenience method fit_transform."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 3
        })
        
        # Test fit_transform
        result1 = selector.fit_transform(self.data)
        
        # Verify result is same as fit() then transform()
        selector2 = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 3
        })
        result2 = selector2.fit(self.data).transform(self.data)
        
        pd.testing.assert_frame_equal(result1, result2)
        
        # Verify is_fitted flag is True
        self.assertTrue(selector.is_fitted)
    
    def test_validate_fitted(self):
        """Test that methods raise errors when called before fitting."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path
        })
        
        # Attempt to call transform() without fitting
        with self.assertRaises(ValueError):
            selector.transform(self.data)
        
        # Attempt to call get_selected_features() without fitting
        with self.assertRaises(ValueError):
            selector.get_selected_features()
        
        # Attempt to call select_features() without fitting
        with self.assertRaises(ValueError):
            selector.select_features()
        
        # Fit the selector
        selector.fit(self.data)
        
        # Assert methods now work without errors
        self.assertIsNotNone(selector.get_selected_features())
        self.assertIsNotNone(selector.select_features())
        self.assertIsNotNone(selector.transform(self.data))
        
        # Verify select_features() matches get_selected_features()
        self.assertEqual(selector.select_features(), selector.get_selected_features())
    
    def test_top_n_selection(self):
        """Test that exactly top_n features are selected."""
        # Test with top_n=2
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 2
        })
        selector.fit(self.data)
        
        selected_features = selector.get_selected_features()
        self.assertEqual(len(selected_features), 2)
        
        # Test with top_n larger than available features
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 100
        })
        selector.fit(self.data)
        
        selected_features = selector.get_selected_features()
        # Should have at most 6 features (the number of features in test data)
        self.assertLessEqual(len(selected_features), 6)
    
    def test_correlation_matrix(self):
        """Test that correlation matrix is computed and accessible."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path
        })
        selector.fit(self.data)
        
        corr_matrix = selector.get_correlation_matrix()
        
        # Check matrix properties
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape[0], corr_matrix.shape[1])
        
        # Check diagonal values are 1.0
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values),
            np.ones(corr_matrix.shape[0])
        )
        
        # Check symmetry
        np.testing.assert_array_almost_equal(
            corr_matrix.values,
            corr_matrix.values.T
        )
        
        # Verify high correlation between radius_mean and perimeter_mean
        if 'radius_mean' in corr_matrix.index and 'perimeter_mean' in corr_matrix.columns:
            correlation = abs(corr_matrix.loc['radius_mean', 'perimeter_mean'])
            self.assertGreater(correlation, 0.9)
    
    def test_missing_features_in_transform(self):
        """Test error handling when transform is called with missing features."""
        selector = FeatureSelector({
            'feature_importance_path': self.feature_importance_path,
            'top_n': 3
        })
        selector.fit(self.data)
        
        # Create new DataFrame missing some selected features
        selected_features = selector.get_selected_features()
        if len(selected_features) > 0:
            # Remove one of the selected features
            missing_feature = selected_features[0]
            new_data = self.data.drop(columns=[missing_feature])
            
            # Attempt to transform
            with self.assertRaises(ValueError):
                selector.transform(new_data)
    
    def test_empty_feature_importance(self):
        """Test handling of edge case with empty feature importance file."""
        # Create temporary file with no feature data
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        empty_file_path = temp_file.name
        temp_file.write("Feature Importance Analysis\n==============================\n")
        temp_file.close()
        
        selector = FeatureSelector({
            'feature_importance_path': empty_file_path
        })
        
        # Should not raise error during fit, but may have no features selected
        selector.fit(self.data)
        
        # Clean up
        import os
        if os.path.exists(empty_file_path):
            os.unlink(empty_file_path)


if __name__ == '__main__':
    unittest.main()

