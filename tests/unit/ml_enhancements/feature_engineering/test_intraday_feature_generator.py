"""
Unit tests for the IntradayFeatureGenerator class.

This module contains tests for the IntradayFeatureGenerator class, which is responsible
for creating features for ML signal enhancement.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.ml_enhancements.feature_engineering.intraday_feature_generator import IntradayFeatureGenerator


@pytest.fixture
def test_config():
    """Fixture providing a test configuration for the IntradayFeatureGenerator."""
    return {
        "feature_lookback": 20,
    }


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    # Create sample dates
    dates = pd.date_range(start=datetime.now() - timedelta(days=5), periods=100, freq='1h')
    
    # Create sample prices data
    prices = pd.DataFrame({
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(105, 115, 100),
        'low': np.linspace(95, 105, 100),
        'close': np.linspace(102, 112, 100),
        'volume': np.linspace(1000, 2000, 100)
    }, index=dates)
    
    # Create sample spreads data
    spreads = pd.DataFrame({
        'spread': np.linspace(1, 2, 100) + np.sin(np.linspace(0, 10, 100)) * 0.1,
        'zscore': np.linspace(-2, 2, 100) + np.sin(np.linspace(0, 10, 100)) * 0.3,
        'mean': np.linspace(1.5, 1.6, 100),
        'std': np.full(100, 0.5)
    }, index=dates)
    
    # Create multi-symbol prices data structure
    prices_data = {
        'symbol1': prices['close'],
        'symbol2': prices['close'] * 1.5  # Slightly different prices
    }
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    # Create multi-symbol volumes data structure
    volumes_data = {
        'symbol1': prices['volume'],
        'symbol2': prices['volume'] * 1.2  # Slightly different volumes
    }
    volumes_df = pd.DataFrame(volumes_data, index=dates)
    
    return {
        'dates': dates,
        'prices_df': prices_df,
        'spreads_df': spreads,
        'volumes_df': volumes_df
    }


class TestIntradayFeatureGenerator:
    """Tests for the IntradayFeatureGenerator class."""
    
    def test_initialization(self, test_config):
        """Test that the IntradayFeatureGenerator initializes correctly."""
        generator = IntradayFeatureGenerator(test_config)
        
        # Check that configuration is loaded
        assert generator.config['feature_lookback'] == test_config['feature_lookback']
    
    def test_calculate_features(self, test_config, sample_data):
        """Test feature calculation."""
        generator = IntradayFeatureGenerator(test_config)
        
        # Get test data
        prices_df = sample_data['prices_df']
        spreads_df = sample_data['spreads_df']
        volumes_df = sample_data['volumes_df']
        
        # Calculate features
        features = generator.calculate_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=volumes_df
        )
        
        # Check that features are calculated correctly
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # Check that various feature categories are included
        # 1. Spread features
        assert 'zscore' in features.columns
        assert 'zscore_change' in features.columns
        
        # 2. Time features
        assert 'hour' in features.columns
        assert 'session_progress' in features.columns
        
        # 3. Instrument-specific features
        assert 'symbol1_rsi' in features.columns
        assert 'symbol2_vol_5' in features.columns
        
        # 4. Correlation features
        assert 'corr_symbol1_symbol2' in features.columns
        
        # 5. Mean reversion features
        assert any('mr_speed' in col for col in features.columns)
    
    def test_calculate_features_without_volumes(self, test_config, sample_data):
        """Test feature calculation without volume data."""
        generator = IntradayFeatureGenerator(test_config)
        
        # Get test data
        prices_df = sample_data['prices_df']
        spreads_df = sample_data['spreads_df']
        
        # Calculate features without volume data
        features = generator.calculate_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=None
        )
        
        # Check that features are calculated correctly
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        # Volume-specific features should not be present
        assert not any('vol_ratio' in col for col in features.columns)
    
    def test_feature_consistency(self, test_config, sample_data):
        """Test that feature calculation is consistent."""
        generator = IntradayFeatureGenerator(test_config)
        
        # Get test data
        prices_df = sample_data['prices_df']
        spreads_df = sample_data['spreads_df']
        volumes_df = sample_data['volumes_df']
        
        # Calculate features twice
        features1 = generator.calculate_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=volumes_df
        )
        
        features2 = generator.calculate_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=volumes_df
        )
        
        # Features should be identical
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_feature_count(self, test_config, sample_data):
        """Test that the expected number of features are generated."""
        generator = IntradayFeatureGenerator(test_config)
        
        # Get test data
        prices_df = sample_data['prices_df']
        spreads_df = sample_data['spreads_df']
        volumes_df = sample_data['volumes_df']
        
        # Calculate features
        features = generator.calculate_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=volumes_df
        )
        
        # Check number of features
        # The exact number will depend on the implementation details,
        # but we should have a significant number of features
        assert len(features.columns) > 30 