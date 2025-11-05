"""
Random Forest Binary Classifier with automatic hyperparameter tuning.

This module implements a Random Forest classifier with GridSearchCV for automatic
hyperparameter optimization, specifically designed for binary classification of
breast cancer diagnosis (Benign vs Malignant).

The classifier handles class imbalance using class_weight='balanced' and uses F1
score as the primary metric for model selection (northstar metric).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import datetime
from pathlib import Path
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, f1_score

from .base import BaseClassifier
from .utils import (
    calculate_metrics,
    get_confusion_matrix,
    get_classification_report
)


class RandomForestBinaryClassifier(BaseClassifier):
    """
    Random Forest Binary Classifier with automatic hyperparameter tuning.
    
    This classifier implements a Random Forest model with GridSearchCV for
    hyperparameter optimization. It is specifically designed for binary
    classification tasks, particularly breast cancer diagnosis (Benign vs Malignant).
    
    Key features:
    - Automatic hyperparameter tuning using GridSearchCV
    - Handles class imbalance with class_weight='balanced'
    - Comprehensive evaluation using F1 (primary), Recall, Precision, and ROC-AUC
    - Automatic train/test split with stratification
    - Feature importance analysis
    - Timestamped model persistence
    
    Example:
        >>> from src.classifiers import RandomForestBinaryClassifier, load_cancer_data
        >>> 
        >>> # Load data
        >>> X, y = load_cancer_data('data/raw/wisconsin_breast_cancer.csv')
        >>> 
        >>> # Create and train classifier
        >>> classifier = RandomForestBinaryClassifier({
        ...     'n_estimators': [100, 200, 300],
        ...     'max_depth': [10, 20, None],
        ...     'min_samples_split': [2, 5, 10],
        ...     'cv_folds': 5,
        ...     'random_state': 42
        ... })
        >>> 
        >>> # Fit with automatic GridSearchCV tuning
        >>> classifier.fit(X, y)
        >>> 
        >>> # View results
        >>> classifier.print_summary()
        >>> 
        >>> # Get best hyperparameters
        >>> print(classifier.get_best_params())
        >>> 
        >>> # Get feature importance
        >>> importance = classifier.get_feature_importance()
        >>> print(importance.head(10))
        >>> 
        >>> # Save model with timestamped filename
        >>> saved_path = classifier.save_model()
        >>> print(f"Model saved to: {saved_path}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Random Forest Binary Classifier.
        
        Args:
            config: Configuration dictionary for classifier parameters.
                   Supported keys:
                   - n_estimators: List of values for grid search (default: [100, 200, 300])
                   - max_depth: List of values for grid search (default: [10, 20, None])
                   - min_samples_split: List of values for grid search (default: [2, 5, 10])
                   - class_weight: Strategy for handling imbalance (default: 'balanced')
                   - random_state: Random seed for reproducibility (default: 42)
                   - cv_folds: Number of cross-validation folds (default: 5)
                   - n_jobs: Number of parallel jobs (default: -1 for all cores)
                   - verbose: Verbosity level for GridSearchCV (default: 1)
                   - test_size: Proportion of data for testing (default: 0.2)
                   - scoring: Scoring metric for GridSearchCV (default: 'f1')
        
        Returns:
            None
        """
        super().__init__(config)
        
        # Extract config parameters with defaults
        self.config.setdefault('n_estimators', [100, 200, 300])
        self.config.setdefault('max_depth', [10, 20, None])
        self.config.setdefault('min_samples_split', [2, 5, 10])
        self.config.setdefault('class_weight', 'balanced')
        self.config.setdefault('random_state', 42)
        self.config.setdefault('cv_folds', 5)
        self.config.setdefault('n_jobs', -1)
        self.config.setdefault('verbose', 1)
        self.config.setdefault('test_size', 0.2)
        self.config.setdefault('scoring', 'f1')  # F1 is northstar metric
        
        # Initialize instance variables
        self.grid_search_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self.train_metrics_ = None
        self.test_metrics_ = None
        self.confusion_matrix_ = None
        self.classification_report_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestBinaryClassifier':
        """
        Fit the Random Forest classifier with GridSearchCV hyperparameter tuning.
        
        This method performs the following steps:
        1. Stores feature names and classes
        2. Splits data into train/test sets with stratification
        3. Defines parameter grid for GridSearchCV
        4. Creates base Random Forest classifier
        5. Performs GridSearchCV to find best hyperparameters
        6. Evaluates on both training and test sets
        7. Generates confusion matrix and classification report
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
            y: Target Series with binary class labels
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data is empty or invalid
        
        Example:
            >>> classifier = RandomForestBinaryClassifier()
            >>> classifier.fit(X, y)
            RandomForestBinaryClassifier(...)
        """
        # 1. Store feature names and classes
        self.feature_names = X.columns.tolist()
        self.classes_ = np.unique(y)
        
        # 2. Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # 3. Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': self.config['n_estimators'],
            'max_depth': self.config['max_depth'],
            'min_samples_split': self.config['min_samples_split']
        }
        
        # 4. Create base Random Forest classifier
        base_classifier = RandomForestClassifier(
            class_weight=self.config['class_weight'],
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        # 5. Determine scoring for GridSearchCV
        # If scoring is 'f1' or not provided, use custom scorer with pos_label='M'
        # Otherwise, allow overriding by accepting a callable in config['scoring']
        scoring_config = self.config['scoring']
        if scoring_config == 'f1' or (scoring_config is None):
            f1_m_scorer = make_scorer(f1_score, pos_label='M')
            scoring = f1_m_scorer
        elif callable(scoring_config):
            # User provided a callable scorer
            scoring = scoring_config
        else:
            # Use as-is (string like 'roc_auc', 'precision', etc.)
            scoring = scoring_config
        
        # 6. Perform GridSearchCV
        self.grid_search_ = GridSearchCV(
            estimator=base_classifier,
            param_grid=param_grid,
            cv=self.config['cv_folds'],
            scoring=scoring,
            n_jobs=self.config['n_jobs'],
            verbose=self.config['verbose'],
            refit=True
        )
        
        # Fit GridSearchCV on training data
        self.grid_search_.fit(X_train, y_train)
        
        # 7. Extract best model and parameters
        self.model = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        self.cv_results_ = self.grid_search_.cv_results_
        
        # 8. Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)
        self.train_metrics_ = calculate_metrics(
            y_train, y_train_pred, y_train_proba[:, 1]
        )
        
        # 9. Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)
        self.test_metrics_ = calculate_metrics(
            y_test, y_test_pred, y_test_proba[:, 1]
        )
        
        # 10. Generate confusion matrix and classification report
        self.confusion_matrix_ = get_confusion_matrix(
            y_test, y_test_pred, labels=self.classes_
        )
        # Align labels and target_names to the same list to avoid mismatch
        labels_to_use = list(self.classes_)
        self.classification_report_ = get_classification_report(
            y_test, y_test_pred, labels=labels_to_use, target_names=labels_to_use
        )
        
        # 11. Set fitted flag
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
        
        Returns:
            Numpy array of predicted class labels with shape (n_samples,)
        
        Raises:
            ValueError: If classifier is not fitted or if features don't match
        
        Example:
            >>> y_pred = classifier.predict(X_new)
            >>> print(y_pred)
            array(['B', 'M', 'B', ...])
        """
        self.validate_fitted()
        
        # Validate that X has the same features as training data
        if set(X.columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            error_msg = "Feature mismatch detected. "
            if missing:
                error_msg += f"Missing features: {sorted(missing)}. "
            if extra:
                error_msg += f"Extra features: {sorted(extra)}. "
            error_msg += f"Expected features: {sorted(self.feature_names)}."
            raise ValueError(error_msg)
        
        # Ensure column order matches training data
        X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability estimates for each class.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
        
        Returns:
            Numpy array of class probabilities with shape (n_samples, n_classes)
            For binary classification, shape is (n_samples, 2)
        
        Raises:
            ValueError: If classifier is not fitted or if features don't match
        
        Example:
            >>> y_proba = classifier.predict_proba(X_new)
            >>> print(y_proba.shape)
            (100, 2)
            >>> print(y_proba[0])
            array([0.85, 0.15])  # [P(B), P(M)]
        """
        self.validate_fitted()
        
        # Validate and reorder features same as in predict()
        if set(X.columns) != set(self.feature_names):
            missing = set(self.feature_names) - set(X.columns)
            extra = set(X.columns) - set(self.feature_names)
            error_msg = "Feature mismatch detected. "
            if missing:
                error_msg += f"Missing features: {sorted(missing)}. "
            if extra:
                error_msg += f"Extra features: {sorted(extra)}. "
            error_msg += f"Expected features: {sorted(self.feature_names)}."
            raise ValueError(error_msg)
        
        # Ensure column order matches training data
        X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance using F1, Recall, Precision, and ROC-AUC metrics.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
            y: True labels Series with shape (n_samples,)
        
        Returns:
            Dictionary with keys: 'f1_score', 'recall', 'precision', 'roc_auc'
            Note: 'roc_auc' may be None if it cannot be calculated
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> metrics = classifier.evaluate(X_test, y_test)
            >>> print(metrics)
            {'f1_score': 0.95, 'recall': 0.94, 'precision': 0.96, 'roc_auc': 0.98}
        """
        self.validate_fitted()
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return calculate_metrics(y, y_pred, y_proba[:, 1])
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Return the best hyperparameters found by GridSearchCV.
        
        Returns:
            Dictionary with best hyperparameters
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> best_params = classifier.get_best_params()
            >>> print(best_params)
            {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5}
        """
        self.validate_fitted()
        return self.best_params_.copy()
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Return cross-validation results from GridSearchCV.
        
        Returns:
            Dictionary with cross-validation results including:
            - 'mean_test_score': Mean test scores for each parameter combination
            - 'std_test_score': Standard deviation of test scores
            - 'params': List of parameter dictionaries tested
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> cv_results = classifier.get_cv_results()
            >>> print(cv_results['mean_test_score'][:5])
            [0.92, 0.93, 0.94, ...]
        """
        self.validate_fitted()
        return self.cv_results_.copy()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance scores from the trained Random Forest.
        
        Returns:
            DataFrame with columns ['feature', 'importance'], sorted by
            importance in descending order
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> importance = classifier.get_feature_importance()
            >>> print(importance.head(10))
                     feature  importance
            0  concave points    0.123456
            1      radius_mean    0.098765
            ...
        """
        self.validate_fitted()
        
        # Extract feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.reset_index(drop=True)
    
    def get_train_metrics(self) -> Dict[str, float]:
        """
        Return training set evaluation metrics.
        
        Returns:
            Dictionary with keys: 'f1_score', 'recall', 'precision', 'roc_auc'
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> train_metrics = classifier.get_train_metrics()
            >>> print(train_metrics)
            {'f1_score': 0.98, 'recall': 0.97, 'precision': 0.99, 'roc_auc': 0.99}
        """
        self.validate_fitted()
        return self.train_metrics_.copy()
    
    def get_test_metrics(self) -> Dict[str, float]:
        """
        Return test set evaluation metrics.
        
        Returns:
            Dictionary with keys: 'f1_score', 'recall', 'precision', 'roc_auc'
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> test_metrics = classifier.get_test_metrics()
            >>> print(test_metrics)
            {'f1_score': 0.95, 'recall': 0.94, 'precision': 0.96, 'roc_auc': 0.98}
        """
        self.validate_fitted()
        return self.test_metrics_.copy()
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Return confusion matrix from test set evaluation.
        
        Returns:
            2D numpy array with shape (n_classes, n_classes)
            For binary classification, shape is (2, 2)
            Format: [[TN, FP], [FN, TP]]
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> cm = classifier.get_confusion_matrix()
            >>> print(cm)
            [[70,  2],
             [ 3, 39]]
        """
        self.validate_fitted()
        return self.confusion_matrix_.copy()
    
    def get_classification_report(self) -> str:
        """
        Return classification report from test set evaluation.
        
        Returns:
            String containing formatted classification report with precision,
            recall, F1-score, and support for each class
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> report = classifier.get_classification_report()
            >>> print(report)
            precision    recall  f1-score   support
            B       0.95      0.97      0.96        72
            M       0.96      0.93      0.94        42
            ...
        """
        self.validate_fitted()
        return self.classification_report_
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model with timestamped filename.
        
        If no filepath is provided, generates a default filename with timestamp:
        `random_forest_model_YYYYMMDD_HHMMSS.pkl` in the `/scripts/` directory.
        
        Args:
            filepath: Optional path to save the model. If None, generates
                     timestamped filename in `/scripts/` directory
        
        Returns:
            String path where the model was saved
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> saved_path = classifier.save_model()
            >>> print(saved_path)
            /path/to/scripts/random_forest_model_20240101_120000.pkl
            >>> 
            >>> # Or with custom path
            >>> saved_path = classifier.save_model('models/my_model.pkl')
            >>> print(saved_path)
            models/my_model.pkl
        """
        self.validate_fitted()
        
        if filepath is None:
            # Generate default filename with timestamp
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'random_forest_model_{timestamp}.pkl'
            # Compute default directory relative to repository root
            # From src/classifiers/random_forest_classifier.py, go up 2 levels to repo root
            project_root = Path(__file__).resolve().parents[2]
            default_dir = project_root / 'scripts'
            filepath = str(default_dir / filename)
        
        # Call parent's save_model method and return the resolved path
        return super().save_model(filepath)
    
    def print_summary(self) -> None:
        """
        Print a comprehensive summary of model performance.
        
        Displays:
        - Best hyperparameters from GridSearchCV
        - Training metrics (F1, Recall, Precision, ROC-AUC)
        - Test metrics
        - Confusion matrix
        - Classification report
        - Top 10 feature importances
        
        Raises:
            ValueError: If classifier is not fitted
        
        Example:
            >>> classifier.print_summary()
            ========================================
            Random Forest Classifier Summary
            ========================================
            Best Hyperparameters:
            {'n_estimators': 200, 'max_depth': 20, ...}
            ...
        """
        self.validate_fitted()
        
        print("=" * 50)
        print("Random Forest Classifier Summary")
        print("=" * 50)
        
        print("\nBest Hyperparameters:")
        print(self.best_params_)
        
        print("\nTraining Metrics:")
        print(f"  F1 Score:    {self.train_metrics_['f1_score']:.4f}")
        print(f"  Recall:      {self.train_metrics_['recall']:.4f}")
        print(f"  Precision:   {self.train_metrics_['precision']:.4f}")
        if self.train_metrics_['roc_auc'] is not None:
            print(f"  ROC-AUC:     {self.train_metrics_['roc_auc']:.4f}")
        
        print("\nTest Metrics:")
        print(f"  F1 Score:    {self.test_metrics_['f1_score']:.4f}")
        print(f"  Recall:      {self.test_metrics_['recall']:.4f}")
        print(f"  Precision:   {self.test_metrics_['precision']:.4f}")
        if self.test_metrics_['roc_auc'] is not None:
            print(f"  ROC-AUC:     {self.test_metrics_['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix_)
        
        print("\nClassification Report:")
        print(self.classification_report_)
        
        print("\nTop 10 Feature Importances:")
        importance_df = self.get_feature_importance()
        print(importance_df.head(10).to_string(index=False))
        
        print("=" * 50)

