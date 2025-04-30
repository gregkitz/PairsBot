"""
Unit tests for the SignalEnhancer component.

This module contains tests to verify the functionality of the SignalEnhancer component.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.paper_trading.components.signal_enhancer import SignalEnhancer
from src.ml_enhancements.intraday_integration import IntradayMLSystem
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Constants for testing
TEST_PAIR = 'GC_SI'
TEST_TIMEFRAME = '5min'


@pytest.fixture
def mock_ml_system():
    """Fixture providing a mock ML system."""
    ml_system = Mock(spec=IntradayMLSystem)
    ml_system.process_data.return_value = pd.DataFrame({
        'zscore': np.random.randn(100),
        'signal': np.random.choice([-1, 0, 1], size=100),
        'quality': np.random.random(100)
    })
    ml_system.enhance_signals.return_value = (
        pd.Series(np.random.choice([-1, 0, 1], size=100)),
        pd.DataFrame({'signal_quality': np.random.random(100)})
    )
    return ml_system


@pytest.fixture
def mock_regime_classifier():
    """Fixture providing a mock regime classifier."""
    classifier = Mock(spec=MarketRegimeClassifier)
    classifier.predict_regime.return_value = "normal"
    classifier.get_regime_parameters.return_value = {
        "z_entry": 2.0,
        "z_exit": 0.5,
        "max_holding_period": 120
    }
    return classifier


@pytest.fixture
def signal_enhancer(mock_ml_system, mock_regime_classifier):
    """Fixture providing a SignalEnhancer instance."""
    config = {
        "feature_lookback": 20,
        "prediction_threshold": 0.6,
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "use_volatility_filter": True,
        "enable_ml_filtering": True,
        "enable_ml_timing": True,
        "enable_ml_adaptation": True
    }
    return SignalEnhancer(
        ml_system=mock_ml_system,
        regime_classifier=mock_regime_classifier,
        config=config
    )


class TestSignalEnhancer:
    """Test class for SignalEnhancer."""
    
    def test_initialization(self, signal_enhancer, mock_ml_system, mock_regime_classifier):
        """Test that the signal enhancer is initialized correctly."""
        assert signal_enhancer.ml_system == mock_ml_system
        assert signal_enhancer.regime_classifier == mock_regime_classifier
        assert signal_enhancer.regime == "unknown"
        assert signal_enhancer.regime_parameters == {}
    
    def test_process_market_data(self, signal_enhancer, mock_ml_system):
        """Test processing market data."""
        # Create sample price data
        prices_df = pd.DataFrame({
            'GC': np.random.rand(100) * 1000 + 1800,
            'SI': np.random.rand(100) * 10 + 20
        })
        
        # Process market data
        result = signal_enhancer.process_market_data(TEST_PAIR, prices_df, TEST_TIMEFRAME)
        
        # Check that ML system was called
        mock_ml_system.process_data.assert_called_once_with(
            pair_id=TEST_PAIR,
            prices_df=prices_df,
            timeframe=TEST_TIMEFRAME
        )
        
        # Check that we got data back
        assert not result.empty
    
    def test_detect_regime(self, signal_enhancer, mock_regime_classifier):
        """Test market regime detection."""
        # Create sample price data
        prices_df = pd.DataFrame({
            'GC': np.random.rand(100) * 1000 + 1800,
            'SI': np.random.rand(100) * 10 + 20
        })
        
        # Detect regime
        result = signal_enhancer.detect_regime(prices_df)
        
        # Check that regime classifier was called
        mock_regime_classifier.predict_regime.assert_called_once_with(prices_df)
        
        # Check that we got a regime back
        assert result == "normal"
        assert signal_enhancer.regime == "normal"
        assert signal_enhancer.regime_parameters == {
            "z_entry": 2.0,
            "z_exit": 0.5,
            "max_holding_period": 120
        }
    
    def test_generate_signals(self, signal_enhancer, mock_ml_system):
        """Test generating trading signals."""
        # Create sample processed data
        processed_data = pd.DataFrame({
            'zscore': np.random.randn(100),
            'signal': np.random.choice([-1, 0, 1], size=100),
            'spread': np.random.randn(100)
        })
        
        # Generate signals
        result = signal_enhancer.generate_signals(TEST_PAIR, processed_data)
        
        # Check that ML system was called
        mock_ml_system.enhance_signals.assert_called_once()
        
        # Check that we got signals back
        assert isinstance(result, pd.Series)
        assert not result.empty
    
    def test_adapt_signals_to_regime_volatile(self, signal_enhancer):
        """Test signal adaptation for volatile regime."""
        # Set regime to volatile
        signal_enhancer.regime = "volatile"
        
        # Create sample signals
        signals = pd.Series(np.random.choice([-1, 0, 1], size=100))
        pair_config = {"z_entry": 2.0, "z_exit": 0.5}
        
        # Adapt signals
        result = signal_enhancer.adapt_signals_to_regime(signals, pair_config)
        
        # Check that signals were adapted
        assert isinstance(result, pd.Series)
        assert len(result) == len(signals)
        # In volatile regime, every other signal should be 0
        assert sum(result[::2] != 0) == 0
    
    def test_adapt_signals_to_regime_trending(self, signal_enhancer):
        """Test signal adaptation for trending regime."""
        # Set regime to trending
        signal_enhancer.regime = "trending"
        
        # Create sample signals
        signals = pd.Series(np.random.choice([-1, 0, 1], size=100))
        pair_config = {"z_entry": 2.0, "z_exit": 0.5}
        
        # Adapt signals
        result = signal_enhancer.adapt_signals_to_regime(signals, pair_config)
        
        # Check that signals were adapted (in this case, no changes)
        assert isinstance(result, pd.Series)
        assert len(result) == len(signals)
        assert (result == signals).all()
    
    def test_adapt_signals_to_regime_low_vol(self, signal_enhancer):
        """Test signal adaptation for low volatility regime."""
        # Set regime to low_vol
        signal_enhancer.regime = "low_vol"
        
        # Create sample signals
        signals = pd.Series(np.random.choice([-1, 0, 1], size=100))
        pair_config = {"z_entry": 2.0, "z_exit": 0.5}
        
        # Adapt signals
        result = signal_enhancer.adapt_signals_to_regime(signals, pair_config)
        
        # Check that signals were adapted (in this case, no changes)
        assert isinstance(result, pd.Series)
        assert len(result) == len(signals)
        assert (result == signals).all()
    
    def test_get_current_regime(self, signal_enhancer):
        """Test getting the current regime and parameters."""
        # Set regime and parameters
        signal_enhancer.regime = "trending"
        signal_enhancer.regime_parameters = {"z_entry": 1.5, "z_exit": 0.3}
        
        # Get current regime
        regime, params = signal_enhancer.get_current_regime()
        
        # Check results
        assert regime == "trending"
        assert params == {"z_entry": 1.5, "z_exit": 0.3} 