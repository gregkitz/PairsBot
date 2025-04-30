"""
Integration tests for the ML enhancement workflow.

This module contains tests that verify the interaction between different ML components
in the workflow, from feature engineering to signal generation.
"""

import unittest
import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.ml_enhancements.feature_engineering.advanced_features import AdvancedFeatureEngineering
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from src.ml_enhancements.spread_prediction.spread_predictor import SpreadPredictor
from src.ml_enhancements.intraday_signals import IntradaySignalProcessor
from src.optimization.adaptive_parameter_manager import AdaptiveParameterManager


@pytest.fixture
def sample_data():
    """Create sample price and spread data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Create dates - hourly for a month
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    
    # Create prices for two assets with some correlation
    asset1_prices = np.zeros(n_samples)
    asset2_prices = np.zeros(n_samples)
    
    # Initial values
    asset1_prices[0] = 100
    asset2_prices[0] = 50
    
    # Generate correlated price movements with mean reversion
    for i in range(1, n_samples):
        # Asset 1 movement
        asset1_mean_reversion = 0.05 * (100 - asset1_prices[i-1])
        asset1_random = np.random.normal(0, 1)
        asset1_prices[i] = asset1_prices[i-1] + asset1_mean_reversion + asset1_random
        
        # Asset 2 movement (correlated with Asset 1 + own movement)
        asset2_mean_reversion = 0.05 * (50 - asset2_prices[i-1])
        asset2_correlated = 0.7 * asset1_random  # Correlation with Asset 1
        asset2_random = np.random.normal(0, 0.8)  # Own random component
        asset2_prices[i] = asset2_prices[i-1] + asset2_mean_reversion + asset2_correlated + asset2_random
    
    # Create DataFrames
    prices_df = pd.DataFrame({
        'ASSET1': asset1_prices,
        'ASSET2': asset2_prices
    }, index=dates)
    
    # Generate volumes
    volumes_df = pd.DataFrame({
        'ASSET1': np.random.randint(100, 1000, size=n_samples),
        'ASSET2': np.random.randint(100, 1000, size=n_samples)
    }, index=dates)
    
    # Calculate spread
    spread = asset1_prices - 2 * asset2_prices  # Example spread calculation
    
    # Calculate z-score (standardized spread)
    rolling_mean = pd.Series(spread).rolling(window=20).mean()
    rolling_std = pd.Series(spread).rolling(window=20).std()
    zscore = (spread - rolling_mean) / rolling_std
    
    # Create spread DataFrame
    spread_df = pd.DataFrame({
        'spread': spread,
        'zscore': zscore
    }, index=dates)
    
    return {
        'prices': prices_df,
        'volumes': volumes_df,
        'spread': spread_df
    }


class TestMLWorkflow(unittest.TestCase):
    """Integration tests for the ML enhancement workflow."""
    
    def test_complete_workflow(self, sample_data):
        """Test the complete ML workflow from feature engineering to signal generation."""
        prices_df = sample_data['prices']
        volumes_df = sample_data['volumes']
        spread_df = sample_data['spread']
        
        # Step 1: Feature Engineering
        feature_eng = AdvancedFeatureEngineering()
        features = feature_eng.generate_features(
            prices_df=prices_df,
            volumes_df=volumes_df,
            spread_data=spread_df['spread']
        )
        
        # Verify feature generation
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 10)
        self.assertEqual(len(features), len(prices_df) - feature_eng.lookback_window)
        
        # Step 2: Regime Detection
        regime_detector = MarketRegimeClassifier(n_regimes=3)
        regime_features = regime_detector.extract_features(prices_df)
        regime_detector.fit(regime_features)
        regimes = regime_detector.predict(regime_features)
        
        # Verify regime detection
        self.assertIsInstance(regimes, pd.Series)
        self.assertTrue(regimes.nunique() >= 1)
        self.assertEqual(len(regimes), len(regime_features))
        
        # Step 3: Spread Prediction
        # Use a shorter horizon for testing
        predictor = SpreadPredictor(
            forecast_horizon=3,
            feature_lookback=10,
            random_state=42
        )
        
        # Train on the first 80% of data
        split_idx = int(len(spread_df) * 0.8)
        train_spread = spread_df['spread'].iloc[:split_idx]
        
        training_metrics = predictor.train(train_spread, use_time_series_cv=False)
        
        # Verify training
        self.assertIsNotNone(training_metrics)
        self.assertIn('mse', training_metrics)
        self.assertIn('test_rmse', training_metrics)
        
        # Make a prediction on newer data
        test_spread = spread_df['spread'].iloc[split_idx-10:]
        prediction = predictor.predict(test_spread)
        
        # Verify prediction
        self.assertIsInstance(prediction, float)
        
        # Step 4: Signal Generation with Adaptive Parameters
        # Create adaptive parameter manager
        param_manager = AdaptiveParameterManager(
            config_dict={
                "regime_responses": {
                    "0": {  # Default regime
                        "entry_zscore": 2.0,
                        "exit_zscore": 0.5
                    }
                }
            }
        )
        
        # Detect current regime and adapt parameters
        current_prices = prices_df.iloc[-20:]
        adapted_params = param_manager.adapt_parameters(current_prices)
        
        # Verify parameter adaptation
        self.assertIsNotNone(adapted_params)
        self.assertIn("entry_zscore", adapted_params)
        self.assertIn("exit_zscore", adapted_params)
        
        # Generate signal based on prediction and parameters
        current_spread = spread_df['spread'].iloc[-1]
        current_zscore = spread_df['zscore'].iloc[-1]
        
        signal = predictor.generate_signal(
            current_spread=current_spread,
            predicted_spread=prediction,
            z_score=current_zscore,
            entry_threshold=adapted_params["entry_zscore"],
            exit_threshold=adapted_params["exit_zscore"]
        )
        
        # Verify signal generation
        self.assertIsInstance(signal, dict)
        self.assertIn('signal', signal)
        self.assertIn('confidence', signal)
        self.assertIn('predicted_change', signal)
        
        # Step 5: Intraday Signal Processor
        signal_processor = IntradaySignalProcessor(
            symbol_pair='ASSET1_ASSET2',
            config={
                'use_ml_enhancement': True,
                'ml_signal_weight': 0.7,
                'trend_filter_window': 5,
                'signal_smoothing_window': 3
            }
        )
        
        # Prepare input for signal processor
        pair_data = {
            'ASSET1_ASSET2': {
                'prices': {
                    'ASSET1': prices_df['ASSET1'],
                    'ASSET2': prices_df['ASSET2']
                },
                'spread': spread_df,
                'ml_signals': pd.Series(
                    [signal['signal']] * 10,  # Replicate the signal for testing
                    index=prices_df.index[-10:]
                ),
                'ml_confidences': pd.Series(
                    [signal['confidence']] * 10,  # Replicate the confidence for testing
                    index=prices_df.index[-10:]
                )
            }
        }
        
        # Generate enhanced signals
        enhanced_signals = signal_processor.generate_signals(pair_data, current_timestamp=prices_df.index[-1])
        
        # Verify enhanced signal generation
        self.assertIsNotNone(enhanced_signals)
        self.assertIn('ASSET1_ASSET2', enhanced_signals)
        self.assertIn('signal', enhanced_signals['ASSET1_ASSET2'])
        
        # The workflow has successfully executed all components in sequence
        print("ML workflow integration test completed successfully")


if __name__ == "__main__":
    pytest.main() 