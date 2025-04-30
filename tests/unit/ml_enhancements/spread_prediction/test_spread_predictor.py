"""
Unit tests for the SpreadPredictor class.

This module contains tests for the SpreadPredictor class, which is used
to make predictions on future spread movements using ML models.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.ml_enhancements.spread_prediction.spread_predictor import SpreadPredictor
from src.utils.error_handling import DataValidationError, ModelPredictionError


@pytest.fixture
def sample_spread_data():
    """Create a sample spread time series for testing."""
    # Create a random walk with some mean reversion for spread data
    np.random.seed(42)
    n = 300
    spread = np.zeros(n)
    spread[0] = 100
    
    for i in range(1, n):
        # Mean reversion component
        mean_reversion = 0.1 * (100 - spread[i-1])  
        # Random walk component
        random_walk = np.random.normal(0, 1)
        spread[i] = spread[i-1] + mean_reversion + random_walk
    
    # Create a Series with datetime index
    dates = pd.date_range(start='2023-01-01', periods=n, freq='5min')
    spread_series = pd.Series(spread, index=dates)
    
    return spread_series


class TestSpreadPredictor(unittest.TestCase):
    """Test cases for the SpreadPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a spread predictor with default parameters
        self.predictor = SpreadPredictor(
            model_type='gradient_boosting',
            forecast_horizon=5,
            feature_lookback=20,
            test_size=0.2,
            random_state=42
        )
    
    def test_initialization(self):
        """Test initializing the SpreadPredictor."""
        # Check default parameters
        self.assertEqual(self.predictor.model_type, 'gradient_boosting')
        self.assertEqual(self.predictor.forecast_horizon, 5)
        self.assertEqual(self.predictor.feature_lookback, 20)
        self.assertEqual(self.predictor.test_size, 0.2)
        self.assertEqual(self.predictor.random_state, 42)
        
        # Check that model pipeline is created
        self.assertIsNotNone(self.predictor.model)
        
        # Verify model parameters are passed correctly
        custom_predictor = SpreadPredictor(
            model_type='random_forest', 
            model_params={'n_estimators': 100, 'max_depth': 5},
            random_state=100
        )
        self.assertEqual(custom_predictor.model_type, 'random_forest')
        self.assertEqual(custom_predictor.model_params['n_estimators'], 100)
        self.assertEqual(custom_predictor.model_params['max_depth'], 5)
    
    def test_feature_creation(self, sample_spread_data):
        """Test feature creation functionality."""
        # Get features from sample data
        X, y = self.predictor._create_features(sample_spread_data)
        
        # Check shapes
        self.assertEqual(len(y), len(sample_spread_data) - self.predictor.feature_lookback - self.predictor.forecast_horizon)
        self.assertEqual(len(X), len(y))
        
        # Check for expected feature columns
        expected_feature_types = [
            'lag_', 'rolling_mean_', 'rolling_std_', 'zscore_', 
            'return_', 'roc_', 'bb_', 'rsi_', 'macd'
        ]
        
        for feature_type in expected_feature_types:
            self.assertTrue(any(feature_type in col for col in X.columns), 
                           f"Expected feature type {feature_type} not found in columns")
        
        # Check target creation
        self.assertEqual(y.name, 'target')
        
        # Check data types
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
    
    def test_train_without_time_series_cv(self, sample_spread_data):
        """Test training without time series cross-validation."""
        # Train the model
        metrics = self.predictor.train(sample_spread_data, use_time_series_cv=False)
        
        # Check that metrics are computed
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('test_mse', metrics)
        self.assertIn('test_rmse', metrics)
        
        # Check prediction history is created
        self.assertIsNotNone(self.predictor.prediction_history)
        self.assertIn('actual', self.predictor.prediction_history.columns)
        self.assertIn('predicted', self.predictor.prediction_history.columns)
        self.assertIn('is_train', self.predictor.prediction_history.columns)
        
        # Check feature importance is computed for GB model
        self.assertIsNotNone(self.predictor.feature_importance)
    
    def test_train_with_time_series_cv(self, sample_spread_data):
        """Test training with time series cross-validation."""
        # Train with time series cross-validation
        metrics = self.predictor.train(sample_spread_data, use_time_series_cv=True, n_splits=3)
        
        # Check that metrics are computed
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
    
    def test_predict(self, sample_spread_data):
        """Test making predictions."""
        # Train the model first
        self.predictor.train(sample_spread_data, use_time_series_cv=False)
        
        # Make a prediction
        prediction = self.predictor.predict(sample_spread_data)
        
        # Check prediction type and range
        self.assertIsInstance(prediction, float)
        # Predictions should be in a reasonable range (within 3 std devs of the mean)
        mean, std = sample_spread_data.mean(), sample_spread_data.std()
        self.assertTrue(mean - 3*std <= prediction <= mean + 3*std)
    
    def test_predict_validation_error(self):
        """Test prediction with invalid data."""
        # Empty data
        with self.assertRaises(DataValidationError):
            self.predictor.predict(pd.Series([]))
        
        # Not enough data
        short_data = pd.Series(range(10))
        with self.assertRaises(DataValidationError):
            self.predictor.predict(short_data)
    
    def test_generate_signal(self, sample_spread_data):
        """Test signal generation."""
        # Train the model first
        self.predictor.train(sample_spread_data, use_time_series_cv=False)
        
        # Make a prediction
        current_spread = 100.0
        predicted_spread = 102.0
        z_score = 1.5
        
        # Generate signal
        signal = self.predictor.generate_signal(
            current_spread=current_spread,
            predicted_spread=predicted_spread,
            z_score=z_score,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Check signal dictionary
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('predicted_change', signal)
        self.assertIn('predicted_change_pct', signal)
        
        # Verify signal logic: no position when z-score < entry threshold
        self.assertEqual(signal['signal'], 0)
        
        # Test with z-score above threshold
        high_zscore_signal = self.predictor.generate_signal(
            current_spread=100.0,
            predicted_spread=95.0,  # Predicted to decrease
            z_score=2.5,            # High enough for entry
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Should generate a short signal (sell spread)
        self.assertEqual(high_zscore_signal['signal'], -1)
        
        # Test with low z-score and prediction to increase
        low_zscore_signal = self.predictor.generate_signal(
            current_spread=100.0,
            predicted_spread=105.0,  # Predicted to increase
            z_score=-2.5,            # Low enough for entry
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Should generate a long signal (buy spread)
        self.assertEqual(low_zscore_signal['signal'], 1)
    
    def test_save_and_load_model(self, sample_spread_data):
        """Test saving and loading a model."""
        # Train the model
        self.predictor.train(sample_spread_data, use_time_series_cv=False)
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.predictor.save_model(temp_path)
            
            # Load model
            loaded_predictor = SpreadPredictor.load_model(temp_path)
            
            # Check that loaded model has the same attributes
            self.assertEqual(loaded_predictor.model_type, self.predictor.model_type)
            self.assertEqual(loaded_predictor.forecast_horizon, self.predictor.forecast_horizon)
            self.assertEqual(loaded_predictor.feature_lookback, self.predictor.feature_lookback)
            
            # Check that both models make the same prediction
            original_pred = self.predictor.predict(sample_spread_data)
            loaded_pred = loaded_predictor.predict(sample_spread_data)
            
            # Predictions should be exactly the same (same random state, same data)
            self.assertEqual(original_pred, loaded_pred)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main() 