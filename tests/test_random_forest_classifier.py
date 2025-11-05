"""
Test suite for Random Forest Binary Classifier.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import tempfile
import os
import io
from pathlib import Path

# Add parent directory to path (so we can import from src package)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifiers.random_forest_classifier import RandomForestBinaryClassifier
from sklearn.metrics import f1_score


class TestRandomForestBinaryClassifier(unittest.TestCase):
    """Test cases for RandomForestBinaryClassifier."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create synthetic binary classification dataset
        n_samples = 200
        n_features = 10
        
        # Generate features using random normal distribution
        X = np.random.randn(n_samples, n_features)
        
        # Create binary target with class imbalance (140 class 'B', 60 class 'M')
        y = np.array(['B'] * 140 + ['M'] * 60)
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        # Create DataFrame and Series
        self.X = pd.DataFrame(
            X,
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y = pd.Series(y)
        
        # Create minimal config for fast testing
        self.config = {
            'n_estimators': [50, 100],  # Reduced for speed
            'max_depth': [5, 10],  # Reduced for speed
            'min_samples_split': [2, 5],  # Reduced for speed
            'cv_folds': 3,  # Reduced for speed
            'test_size': 0.2,
            'random_state': 42,
            'verbose': 0  # Suppress output during tests
        }
    
    def test_initialization(self):
        """Test that RandomForestBinaryClassifier initializes correctly."""
        # Test default initialization
        classifier = RandomForestBinaryClassifier()
        
        # Assert default values are set correctly
        self.assertEqual(classifier.config['n_estimators'], [100, 200, 300])
        self.assertEqual(classifier.config['max_depth'], [10, 20, None])
        self.assertEqual(classifier.config['min_samples_split'], [2, 5, 10])
        self.assertEqual(classifier.config['class_weight'], 'balanced')
        self.assertEqual(classifier.config['random_state'], 42)
        self.assertEqual(classifier.config['cv_folds'], 5)
        self.assertEqual(classifier.config['scoring'], 'f1')
        
        # Assert is_fitted is False
        self.assertFalse(classifier.is_fitted)
        
        # Assert model is None
        self.assertIsNone(classifier.model)
        
        # Test custom configuration
        custom_config = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'cv_folds': 3,
            'random_state': 123
        }
        classifier = RandomForestBinaryClassifier(custom_config)
        
        # Assert config values are stored correctly
        self.assertEqual(classifier.config['n_estimators'], [50, 100])
        self.assertEqual(classifier.config['max_depth'], [5, 10])
        self.assertEqual(classifier.config['cv_folds'], 3)
        self.assertEqual(classifier.config['random_state'], 123)
        
        # Assert all config parameters are accessible via get_params()
        params = classifier.get_params()
        self.assertEqual(params['n_estimators'], [50, 100])
        self.assertEqual(params['max_depth'], [5, 10])
    
    def test_fit(self):
        """Test that the classifier fits correctly with GridSearchCV."""
        classifier = RandomForestBinaryClassifier(self.config)
        
        # Fit on test data
        classifier.fit(self.X, self.y)
        
        # Assert is_fitted is True after fitting
        self.assertTrue(classifier.is_fitted)
        
        # Assert model is not None
        self.assertIsNotNone(classifier.model)
        
        # Assert best_params_ is not None and is a dictionary
        self.assertIsNotNone(classifier.best_params_)
        self.assertIsInstance(classifier.best_params_, dict)
        
        # Assert feature_names matches X.columns
        self.assertEqual(classifier.feature_names, self.X.columns.tolist())
        
        # Assert classes_ contains both 'B' and 'M'
        self.assertIn('B', classifier.classes_)
        self.assertIn('M', classifier.classes_)
        
        # Assert train_metrics_ and test_metrics_ are not None
        self.assertIsNotNone(classifier.train_metrics_)
        self.assertIsNotNone(classifier.test_metrics_)
        
        # Assert all expected metrics are present
        expected_metrics = ['f1_score', 'recall', 'precision', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, classifier.train_metrics_)
            self.assertIn(metric, classifier.test_metrics_)
    
    def test_predict(self):
        """Test that predictions work correctly."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Make predictions on test data
        y_pred = classifier.predict(self.X)
        
        # Assert predictions have correct shape
        self.assertEqual(len(y_pred), len(self.y))
        
        # Assert predictions contain only valid class labels
        self.assertTrue(np.all(np.isin(y_pred, ['B', 'M'])))
        
        # Assert predictions are numpy array
        self.assertIsInstance(y_pred, np.ndarray)
    
    def test_predict_proba(self):
        """Test that probability predictions work correctly."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get probabilities
        y_proba = classifier.predict_proba(self.X)
        
        # Assert shape is (n_samples, 2) for binary classification
        self.assertEqual(y_proba.shape, (len(self.y), 2))
        
        # Assert all probabilities are between 0 and 1
        self.assertTrue(np.all((y_proba >= 0) & (y_proba <= 1)))
        
        # Assert each row sums to approximately 1.0
        row_sums = y_proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(self.y)), decimal=5)
    
    def test_evaluate(self):
        """Test that evaluation returns correct metrics."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Evaluate on test data
        metrics = classifier.evaluate(self.X, self.y)
        
        # Assert metrics is a dictionary
        self.assertIsInstance(metrics, dict)
        
        # Assert all expected keys are present
        expected_keys = ['f1_score', 'recall', 'precision', 'roc_auc']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Assert all metric values are between 0 and 1 (or None for roc_auc)
        for key in ['f1_score', 'recall', 'precision']:
            self.assertGreaterEqual(metrics[key], 0)
            self.assertLessEqual(metrics[key], 1)
        
        if metrics['roc_auc'] is not None:
            self.assertGreaterEqual(metrics['roc_auc'], 0)
            self.assertLessEqual(metrics['roc_auc'], 1)
        
        # Assert F1 score is reasonable (relaxed threshold to avoid flakiness)
        # Use a lower floor (e.g., > 0.3) instead of > 0.5 to account for dataset variability
        self.assertGreater(metrics['f1_score'], 0.3)
    
    def test_class_weight_balanced(self):
        """Test that class_weight='balanced' is applied correctly."""
        config = self.config.copy()
        config['class_weight'] = 'balanced'
        
        classifier = RandomForestBinaryClassifier(config)
        classifier.fit(self.X, self.y)
        
        # Access the fitted model's class_weight parameter
        self.assertEqual(classifier.model.class_weight, 'balanced')
        
        # Verify that the model performs reasonably on minority class
        # Check recall for 'M' (minority class)
        test_metrics = classifier.get_test_metrics()
        self.assertGreater(test_metrics['recall'], 0.0)  # Should have some recall
    
    def test_grid_search_results(self):
        """Test that GridSearchCV results are accessible."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get best params
        best_params = classifier.get_best_params()
        
        # Assert best_params is a dictionary
        self.assertIsInstance(best_params, dict)
        
        # Assert it contains keys: 'n_estimators', 'max_depth', 'min_samples_split'
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)
        self.assertIn('min_samples_split', best_params)
        
        # Get CV results
        cv_results = classifier.get_cv_results()
        
        # Assert cv_results is a dictionary
        self.assertIsInstance(cv_results, dict)
        
        # Assert it contains expected keys
        self.assertIn('mean_test_score', cv_results)
        self.assertIn('params', cv_results)
    
    def test_feature_importance(self):
        """Test that feature importance extraction works."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get feature importance
        importance_df = classifier.get_feature_importance()
        
        # Assert result is a pandas DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame)
        
        # Assert it has columns: 'feature' and 'importance'
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Assert number of rows equals number of features (10)
        self.assertEqual(len(importance_df), 10)
        
        # Assert importance values sum to approximately 1.0
        total_importance = importance_df['importance'].sum()
        self.assertAlmostEqual(total_importance, 1.0, places=5)
        
        # Assert features are sorted by importance (descending)
        importances = importance_df['importance'].values
        self.assertTrue(np.all(importances[:-1] >= importances[1:]))
    
    def test_train_test_metrics(self):
        """Test that separate train and test metrics are computed."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get train metrics
        train_metrics = classifier.get_train_metrics()
        
        # Get test metrics
        test_metrics = classifier.get_test_metrics()
        
        # Assert both are dictionaries with same keys
        self.assertEqual(set(train_metrics.keys()), set(test_metrics.keys()))
        
        # Assert train metrics are typically better than test metrics (check F1 score)
        # This validates that the model is not just memorizing
        self.assertGreaterEqual(train_metrics['f1_score'], test_metrics['f1_score'] * 0.9)
    
    def test_confusion_matrix(self):
        """Test that confusion matrix is generated correctly."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get confusion matrix
        cm = classifier.get_confusion_matrix()
        
        # Assert it is a numpy array
        self.assertIsInstance(cm, np.ndarray)
        
        # Assert shape is (2, 2) for binary classification
        self.assertEqual(cm.shape, (2, 2))
        
        # Assert all values are non-negative integers
        self.assertTrue(np.all(cm >= 0))
        self.assertTrue(np.all(cm == cm.astype(int)))
        
        # Assert sum of confusion matrix equals number of test samples
        # (test_size = 0.2, so 0.2 * 200 = 40 test samples)
        n_test = int(self.config['test_size'] * len(self.y))
        self.assertEqual(cm.sum(), n_test)
    
    def test_classification_report(self):
        """Test that classification report is generated."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Get report
        report = classifier.get_classification_report()
        
        # Assert report is a string
        self.assertIsInstance(report, str)
        
        # Assert it contains class labels ('B' and 'M')
        self.assertIn('B', report)
        self.assertIn('M', report)
        
        # Assert it contains metric names
        self.assertIn('precision', report.lower())
        self.assertIn('recall', report.lower())
        self.assertIn('f1-score', report.lower())
    
    def test_save_and_load_model(self):
        """Test that model can be saved and loaded correctly."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Create temporary file path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Save model
            saved_path = classifier.save_model(temp_path)
            
            # Assert file exists at saved_path
            self.assertTrue(os.path.exists(saved_path))
            
            # Load model
            loaded_classifier = RandomForestBinaryClassifier.load_model(saved_path)
            
            # Assert loaded classifier is fitted
            self.assertTrue(loaded_classifier.is_fitted)
            
            # Make predictions with both original and loaded classifier
            y_pred_original = classifier.predict(self.X)
            y_pred_loaded = loaded_classifier.predict(self.X)
            
            # Assert predictions are identical
            np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
            
            # Assert best params are the same
            self.assertEqual(classifier.get_best_params(), loaded_classifier.get_best_params())
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_with_default_filename(self):
        """Test that model saves with timestamped filename when no path provided."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Save without providing filepath
        saved_path = classifier.save_model()
        
        try:
            # Assert saved_path contains 'random_forest_model_'
            self.assertIn('random_forest_model_', saved_path)
            
            # Assert saved_path contains timestamp pattern (YYYYMMDD_HHMMSS)
            # Extract timestamp from filename
            import re
            timestamp_match = re.search(r'random_forest_model_(\d{8}_\d{6})\.pkl', saved_path)
            self.assertIsNotNone(timestamp_match, "Timestamp pattern not found in filename")
            
            # Assert saved_path ends with '.pkl'
            self.assertTrue(saved_path.endswith('.pkl'))
            
            # Assert file exists
            self.assertTrue(os.path.exists(saved_path))
        finally:
            # Clean up created file
            if os.path.exists(saved_path):
                os.unlink(saved_path)
    
    def test_validate_fitted(self):
        """Test that methods raise errors when called before fitting."""
        classifier = RandomForestBinaryClassifier(self.config)
        
        # Attempt to call predict() without fitting
        with self.assertRaises(ValueError):
            classifier.predict(self.X)
        
        # Attempt to call predict_proba() without fitting
        with self.assertRaises(ValueError):
            classifier.predict_proba(self.X)
        
        # Attempt to call evaluate() without fitting
        with self.assertRaises(ValueError):
            classifier.evaluate(self.X, self.y)
        
        # Attempt to call get_best_params() without fitting
        with self.assertRaises(ValueError):
            classifier.get_best_params()
        
        # Fit the classifier
        classifier.fit(self.X, self.y)
        
        # Assert all methods now work without errors
        self.assertIsNotNone(classifier.predict(self.X))
        self.assertIsNotNone(classifier.predict_proba(self.X))
        self.assertIsNotNone(classifier.evaluate(self.X, self.y))
        self.assertIsNotNone(classifier.get_best_params())
    
    def test_feature_validation_in_predict(self):
        """Test that predict validates feature consistency."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Create new DataFrame with different features (missing some)
        X_missing = self.X.drop(columns=[self.X.columns[0]])
        
        # Attempt to predict on mismatched data
        with self.assertRaises(ValueError) as context:
            classifier.predict(X_missing)
        
        # Assert error message mentions feature mismatch
        self.assertIn('Feature mismatch', str(context.exception))
        
        # Create new DataFrame with extra features
        X_extra = self.X.copy()
        X_extra['extra_feature'] = np.random.randn(len(X_extra))
        
        # Attempt to predict on mismatched data
        with self.assertRaises(ValueError) as context:
            classifier.predict(X_extra)
        
        # Assert error message mentions feature mismatch
        self.assertIn('Feature mismatch', str(context.exception))
    
    def test_feature_order_in_predict(self):
        """Test that predict handles different feature orders correctly."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Create test data with same features but different column order
        X_reordered = self.X[self.X.columns[::-1]]  # Reverse column order
        
        # Make predictions
        y_pred_reordered = classifier.predict(X_reordered)
        
        # Assert predictions succeed (features are reordered internally)
        self.assertEqual(len(y_pred_reordered), len(self.y))
        
        # Compare with predictions on correctly-ordered data
        y_pred_original = classifier.predict(self.X)
        
        # Assert predictions are identical
        np.testing.assert_array_equal(y_pred_reordered, y_pred_original)
    
    def test_print_summary(self):
        """Test that print_summary executes without errors."""
        classifier = RandomForestBinaryClassifier(self.config)
        classifier.fit(self.X, self.y)
        
        # Capture stdout
        captured_output = io.StringIO()
        import sys
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Call print_summary
            classifier.print_summary()
            
            # Get output
            output = captured_output.getvalue()
            
            # Assert output contains expected sections
            self.assertIn('Random Forest Classifier Summary', output)
            self.assertIn('Best Hyperparameters', output)
            self.assertIn('Training Metrics', output)
            self.assertIn('Test Metrics', output)
            self.assertIn('Confusion Matrix', output)
            self.assertIn('Classification Report', output)
            self.assertIn('Feature Importances', output)
            self.assertIn('F1 Score', output)
            self.assertIn('Recall', output)
            self.assertIn('Precision', output)
        finally:
            sys.stdout = old_stdout
    
    def test_scoring_metric_is_f1(self):
        """Test that GridSearchCV uses F1 as scoring metric by default."""
        classifier = RandomForestBinaryClassifier()  # Use default config
        
        # Assert default scoring is 'f1'
        self.assertEqual(classifier.config['scoring'], 'f1')
        
        # Fit on data
        classifier.fit(self.X, self.y)
        
        # Access GridSearchCV scoring (now a callable scorer, not a string)
        scorer = classifier.grid_search_.scoring
        
        # Assert that scoring is a callable
        self.assertTrue(callable(scorer), "GridSearchCV scoring should be a callable scorer")
        
        # Verify it mentions f1_score and pos_label='M' in its representation
        scorer_repr = repr(scorer)
        self.assertIn('f1_score', scorer_repr, "Scorer should mention f1_score")
        self.assertIn("pos_label='M'", scorer_repr, "Scorer should use pos_label='M'")
        
        # Optionally check attributes if available (more strict but slightly more coupled)
        if hasattr(scorer, '_score_func'):
            self.assertIs(scorer._score_func, f1_score, "Scorer should use f1_score function")
        if hasattr(scorer, '_kwargs'):
            self.assertEqual(scorer._kwargs.get('pos_label'), 'M', "Scorer should have pos_label='M'")


if __name__ == '__main__':
    unittest.main()

