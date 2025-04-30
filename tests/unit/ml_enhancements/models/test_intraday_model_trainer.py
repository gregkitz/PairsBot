"""
Unit tests for the intraday model trainer module.

This module contains tests for the IntradayModelTrainer class defined in
the intraday_model_trainer.py module.
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

from src.ml_enhancements.intraday_model_trainer import IntradayModelTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Create sample data for testing
def create_sample_data():
    """Create sample data for testing intraday models."""
    np.random.seed(42)
    n_samples = 100
    
    # Create a simple classification dataset
    X = pd.DataFrame({
        'price_momentum': np.random.normal(0, 1, n_samples),
        'volume_ratio': np.random.normal(0, 1, n_samples),
        'zscore': np.random.normal(0, 1, n_samples),
        'volatility': np.random.normal(0, 1, n_samples),
    })
    
    # Binary classification target representing buy/sell signals
    y = pd.Series(np.random.choice([0, 1], n_samples))
    
    return X, y


class TestIntradayModelTrainer(unittest.TestCase):
    """Tests for the IntradayModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_trainer = IntradayModelTrainer(
            models_dir=self.temp_dir.name, 
            random_seed=42
        )
        self.X, self.y = create_sample_data()
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.temp_dir.cleanup()
    
    def test_train_model_with_default_model(self):
        """Test training a model with default settings."""
        # Train model with default settings
        model, metrics = self.model_trainer.train_model(
            X=self.X,
            y=self.y,
            model_name="test_default_model",
            model_type="signal_filter",
            use_walk_forward=False  # Simpler test without walk-forward validation
        )
        
        # Verify model and metrics
        self.assertIsNotNone(model)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Verify metrics are in expected range
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
    
    def test_train_model_with_custom_model(self):
        """Test training a custom model."""
        # Define a custom model and parameters
        custom_model = LogisticRegression
        custom_params = {
            "C": 0.1,
            "class_weight": "balanced",
            "random_state": 42
        }
        
        # Train model with custom settings
        model, metrics = self.model_trainer.train_model(
            X=self.X,
            y=self.y,
            model_name="test_custom_model",
            model_type="signal_filter",
            custom_model=custom_model,
            custom_params=custom_params,
            use_walk_forward=False
        )
        
        # Verify model is of the correct type
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.C, 0.1)
        self.assertEqual(model.class_weight, "balanced")
        
        # Verify metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
    
    def test_train_model_with_walk_forward(self):
        """Test training a model with walk-forward validation."""
        # Mock WalkForwardValidator to avoid actual cross-validation computation
        with patch('src.ml_enhancements.intraday_model_trainer.WalkForwardValidator') as mock_validator:
            # Configure mock validator
            mock_validator_instance = MagicMock()
            mock_validator.return_value = mock_validator_instance
            mock_validator_instance.split.return_value = [(
                np.arange(80),  # Train indices
                np.arange(80, 100)  # Test indices
            )]
            
            # Train model with walk-forward validation
            model, metrics = self.model_trainer.train_model(
                X=self.X,
                y=self.y,
                model_name="test_walk_forward",
                use_walk_forward=True,
                n_splits=3
            )
            
            # Verify mock was called with correct parameters
            mock_validator.assert_called_once()
            mock_validator_instance.split.assert_called_once()
            
            # Verify model and metrics
            self.assertIsNotNone(model)
            self.assertIsInstance(metrics, dict)
    
    def test_load_and_predict_model(self):
        """Test saving, loading, and predicting with a model."""
        # Train and save a model
        self.model_trainer.train_model(
            X=self.X,
            y=self.y,
            model_name="test_prediction",
            use_walk_forward=False
        )
        
        # Load the model
        loaded_model = self.model_trainer.load_model("test_prediction")
        
        # Verify model loading
        self.assertIsNotNone(loaded_model)
        
        # Test prediction with probabilities
        predictions, probabilities = self.model_trainer.predict(
            X=self.X,
            model_name="test_prediction",
            return_proba=True
        )
        
        # Verify predictions
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(probabilities.shape[0], len(self.X))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        
        # Test prediction without probabilities
        predictions_only = self.model_trainer.predict(
            X=self.X,
            model_name="test_prediction",
            return_proba=False
        )
        
        # Verify predictions
        self.assertEqual(len(predictions_only), len(self.X))
    
    def test_get_model(self):
        """Test the get_model method."""
        # Train and save a model
        original_model, _ = self.model_trainer.train_model(
            X=self.X,
            y=self.y,
            model_name="test_get_model",
            use_walk_forward=False
        )
        
        # Use get_model to retrieve the model
        retrieved_model = self.model_trainer.get_model("test_get_model")
        
        # Verify model retrieval
        self.assertIsNotNone(retrieved_model)
        
        # Make predictions with both models and compare
        original_preds = original_model.predict(self.X)
        retrieved_preds = retrieved_model.predict(self.X)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_preds, retrieved_preds)
    
    def test_calculate_metrics(self):
        """Test the metrics calculation function."""
        # Create some test prediction data
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        
        # Calculate metrics using the private method
        metrics = self.model_trainer._calculate_metrics(y_true, y_pred)
        
        # Verify all expected metrics are present
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        
        # Verify metrics are correctly calculated
        self.assertEqual(metrics["accuracy"], 0.75)  # 6/8 correct predictions
    
    def test_missing_model_handling(self):
        """Test handling of missing model files."""
        # The implementation logs errors but apparently doesn't raise exceptions
        # So we test for returning None instead
        
        # Attempt to load a non-existent model
        result = self.model_trainer.load_model("non_existent_model")
        self.assertIsNone(result)
        
        # Attempt to get a non-existent model
        result = self.model_trainer.get_model("non_existent_model")
        self.assertIsNone(result)
        
        # Attempt to predict with a non-existent model - should return None or empty prediction
        result = self.model_trainer.predict(
            X=pd.DataFrame({'feature1': [1, 2, 3]}),
            model_name="non_existent_model"
        )
        # Either None or empty numpy array would be acceptable
        self.assertTrue(result is None or len(result) == 0)


if __name__ == "__main__":
    unittest.main() 