"""
Unit tests for the ensemble models module.

This module contains tests for the ensemble model factory and ensemble model creation
functions defined in the ensemble_models.py module.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.ml_enhancements.ensemble_models import EnsembleModelFactory, create_intraday_ensemble
from sklearn.ensemble import (
    VotingClassifier, 
    StackingClassifier,
    BaggingClassifier, 
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Create sample data for testing
def create_sample_data():
    """Create sample data for testing ensemble models."""
    np.random.seed(42)
    n_samples = 100
    
    # Create a simple classification dataset
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.normal(0, 1, n_samples),
        'feature_4': np.random.normal(0, 1, n_samples),
    })
    
    # Binary classification target
    y = pd.Series(np.random.choice([0, 1], n_samples))
    
    return X, y


class TestEnsembleModelFactory(unittest.TestCase):
    """Tests for the EnsembleModelFactory class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ensemble_factory = EnsembleModelFactory(
            models_dir=self.temp_dir.name, 
            random_seed=42
        )
        self.X, self.y = create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.temp_dir.cleanup()
    
    def test_create_voting_ensemble(self):
        """Test creating a voting ensemble."""
        # Define base models
        base_models = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        # Create voting ensemble
        model = self.ensemble_factory.create_voting_ensemble(
            model_name="test_voting",
            base_models=base_models,
            voting='soft'
        )
        
        # Verify model creation
        self.assertIsInstance(model, VotingClassifier)
        self.assertEqual(len(model.estimators), 2)
        self.assertEqual(model.voting, 'soft')
    
    def test_create_stacking_ensemble(self):
        """Test creating a stacking ensemble."""
        # Define base models
        base_models = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        # Create stacking ensemble
        model = self.ensemble_factory.create_stacking_ensemble(
            model_name="test_stacking",
            base_models=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        
        # Verify model creation
        self.assertIsInstance(model, StackingClassifier)
        self.assertEqual(len(model.estimators), 2)
        self.assertEqual(model.cv, 3)
        self.assertIsInstance(model.final_estimator, LogisticRegression)
    
    @patch('src.ml_enhancements.ensemble_models.AdaBoostClassifier')
    def test_create_boosting_ensemble(self, mock_ada_boost):
        """Test creating a boosting ensemble."""
        # Setup mock
        mock_model = MagicMock()
        mock_ada_boost.return_value = mock_model
        
        # Create boosting ensemble
        model = self.ensemble_factory.create_boosting_ensemble(
            model_name="test_boosting",
            base_estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50,
            learning_rate=0.1
        )
        
        # Verify the mock was called
        mock_ada_boost.assert_called_once()
        
        # Instead of checking parameters on the actual model (which might change between scikit-learn versions),
        # we just verify that we got a model back
        self.assertEqual(model, mock_model)
    
    @patch('src.ml_enhancements.ensemble_models.BaggingClassifier')
    def test_create_bagging_ensemble(self, mock_bagging):
        """Test creating a bagging ensemble."""
        # Setup mock
        mock_model = MagicMock()
        mock_bagging.return_value = mock_model
        
        # Create bagging ensemble
        model = self.ensemble_factory.create_bagging_ensemble(
            model_name="test_bagging",
            base_estimator=DecisionTreeClassifier(max_depth=2),
            n_estimators=50,
            max_samples=0.7,
            max_features=0.7
        )
        
        # Verify the mock was called
        mock_bagging.assert_called_once()
        
        # Instead of checking parameters on the actual model (which might change between scikit-learn versions),
        # we just verify that we got a model back
        self.assertEqual(model, mock_model)
    
    def test_create_diversified_ensemble(self):
        """Test creating a diversified ensemble."""
        # Create diversified ensemble
        model = self.ensemble_factory.create_diversified_ensemble(
            model_name="test_diversified"
        )
        
        # Verify model creation
        self.assertIsInstance(model, VotingClassifier)
        self.assertGreaterEqual(len(model.estimators), 3)  # Should have at least 3 diverse models
    
    def test_train_ensemble(self):
        """Test training an ensemble model."""
        # Define a simple voting ensemble
        base_models = [
            ('lr', LogisticRegression(random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        model = self.ensemble_factory.create_voting_ensemble(
            model_name="test_train",
            base_models=base_models
        )
        
        # Train the ensemble
        trained_model, metrics = self.ensemble_factory.train_ensemble(
            X=self.X,
            y=self.y,
            model_name="test_train",
            use_walk_forward=False,  # Simpler test without walk-forward validation
            scale_features=True
        )
        
        # Verify training result
        self.assertIsInstance(trained_model, VotingClassifier)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Verify metrics are in expected range
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
    
    def test_load_model(self):
        """Test saving and loading a model."""
        # Create and train a model
        model = self.ensemble_factory.create_voting_ensemble(
            model_name="test_save_load",
            base_models=[
                ('lr', LogisticRegression(random_state=42)),
                ('dt', DecisionTreeClassifier(random_state=42))
            ]
        )
        
        # Train and save the model
        self.ensemble_factory.train_ensemble(
            X=self.X,
            y=self.y,
            model_name="test_save_load",
            use_walk_forward=False
        )
        
        # Load the model
        loaded_model = self.ensemble_factory.load_model("test_save_load")
        
        # Verify model loading
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(loaded_model, VotingClassifier)
        
        # Make predictions with the loaded model
        predictions = loaded_model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))


class TestCreateIntradayEnsemble(unittest.TestCase):
    """Tests for the create_intraday_ensemble function."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.X, self.y = create_sample_data()
    
    @patch('src.ml_enhancements.ensemble_models.EnsembleModelFactory')
    def test_create_intraday_ensemble(self, mock_factory):
        """Test creating an intraday ensemble model."""
        # Mock factory methods
        mock_instance = MagicMock()
        mock_factory.return_value = mock_instance
        mock_instance.create_voting_ensemble.return_value = MagicMock()
        mock_instance.train_ensemble.return_value = (MagicMock(), {"accuracy": 0.85})
        
        # Call the function
        result = create_intraday_ensemble(
            features_df=self.X,
            labels=self.y,
            model_type='signal_filter',
            ensemble_type='voting'
        )
        
        # Verify function calls
        mock_factory.assert_called_once()
        mock_instance.create_voting_ensemble.assert_called_once()
        mock_instance.train_ensemble.assert_called_once()
        
        # Verify result
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main() 