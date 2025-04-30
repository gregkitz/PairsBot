"""
Unit tests for the SpreadAnalyzer class.

This module contains tests for the spread calculation and analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from src.spread_analytics.spread_analyzer import SpreadAnalyzer
from pykalman import KalmanFilter

@pytest.fixture
def sample_data():
    """Fixture providing sample price data for spread analysis."""
    # Create synthetic price series
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate cointegrated price series
    # Series 1: random walk
    random_changes = np.random.normal(0, 1, 200)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 (with noise)
    noise = np.random.normal(0, 0.5, 200)
    series2 = 0.5 * series1 + 50 + noise
    
    # Create DataFrames
    df1 = pd.DataFrame({
        'Open': series1 * 0.99,
        'High': series1 * 1.01,
        'Low': series1 * 0.98,
        'Close': series1,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    df2 = pd.DataFrame({
        'Open': series2 * 0.99,
        'High': series2 * 1.01,
        'Low': series2 * 0.98,
        'Close': series2,
        'Volume': np.random.randint(500, 5000, len(dates))
    }, index=dates)
    
    # Return as dict of data
    return {'S1': df1, 'S2': df2}

@pytest.fixture
def test_config_file():
    """Fixture providing the path to the test configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(current_dir, '..', '..', 'fixtures')
    return os.path.join(fixtures_dir, 'test_config.json')

@pytest.fixture
def spread_analyzer(test_config_file):
    """Fixture providing a SpreadAnalyzer instance."""
    # Load config
    with open(test_config_file, 'r') as f:
        config = json.load(f)
    
    # Create analyzer with config
    analyzer = SpreadAnalyzer(config.get('spread_analyzer', {}))
    return analyzer

class TestSpreadAnalyzer:
    """Test class for SpreadAnalyzer."""
    
    def test_initialization(self, test_config_file):
        """Test that the analyzer is initialized correctly."""
        # Load config
        with open(test_config_file, 'r') as f:
            config = json.load(f)
            
        # Create analyzer
        analyzer = SpreadAnalyzer(config.get('spread_analyzer', {}))
        
        # Check config loaded correctly
        assert analyzer.default_hedge_ratio_method == config['spread_analyzer']['default_hedge_ratio_method']
        assert analyzer.z_score_window == config['spread_analyzer']['z_score_window']
    
    def test_calculate_hedge_ratio_ols(self, spread_analyzer, sample_data):
        """Test calculating hedge ratio using OLS."""
        # Get price series
        s1 = sample_data['S1']['Close']
        s2 = sample_data['S2']['Close']
        
        # Calculate hedge ratio
        hedge_ratio = spread_analyzer.calculate_hedge_ratio_ols(s1, s2)
        
        # Check results
        assert isinstance(hedge_ratio, float)
        assert 0.4 < hedge_ratio < 0.6  # Should be close to 0.5 (as defined in sample data)
    
    def test_calculate_hedge_ratio_kalman(self, spread_analyzer, sample_data):
        """Test calculating hedge ratio using Kalman filter."""
        # Get price series
        s1 = sample_data['S1']['Close']
        s2 = sample_data['S2']['Close']
        
        # Calculate hedge ratio
        hedge_ratio_series = spread_analyzer.calculate_hedge_ratio_kalman(s1, s2)
        
        # Check results
        assert isinstance(hedge_ratio_series, pd.Series)
        assert len(hedge_ratio_series) == len(s1)
        
        # Should converge to a value close to 0.5
        assert 0.4 < hedge_ratio_series.iloc[-1] < 0.6
    
    def test_calculate_spread(self, spread_analyzer, sample_data):
        """Test calculating spread between two series."""
        # Get price series
        s1 = sample_data['S1']['Close']
        s2 = sample_data['S2']['Close']
        
        # Calculate spread with fixed hedge ratio
        hedge_ratio = 0.5
        spread = spread_analyzer.calculate_spread(s1, s2, hedge_ratio)
        
        # Check results
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(s1)
        
        # Calculate expected spread manually and compare
        expected_spread = s2 - hedge_ratio * s1
        pd.testing.assert_series_equal(spread, expected_spread)
    
    def test_calculate_z_score(self, spread_analyzer):
        """Test calculating z-score from a spread series."""
        # Create a spread series with known properties
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        spread = pd.Series(np.random.normal(0, 1, 100), index=dates)
        
        # Calculate z-score with default window
        z_score = spread_analyzer.calculate_z_score(spread)
        
        # Check results
        assert isinstance(z_score, pd.Series)
        assert len(z_score) == len(spread)
        
        # Z-score of normal distribution should have mean close to 0 and std close to 1
        # But due to the rolling window, we'll check only the later values
        later_z = z_score.iloc[50:]
        assert -0.5 < later_z.mean() < 0.5
        assert 0.7 < later_z.std() < 1.3
    
    def test_detect_regime_change(self, spread_analyzer):
        """Test detection of regime changes in a spread."""
        # Create a spread series with a regime change
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # First regime: mean 0, std 1
        regime1 = np.random.normal(0, 1, 100)
        # Second regime: mean 2, std 1.5
        regime2 = np.random.normal(2, 1.5, 100)
        
        # Combine regimes
        values = np.concatenate([regime1, regime2])
        spread = pd.Series(values, index=dates)
        
        # Detect regime changes
        regimes = spread_analyzer.detect_regime_change(spread)
        
        # Check results
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(spread)
        
        # Should detect at least one regime change (though not exactly at index 100
        # due to the rolling window detection method)
        changes = regimes.diff().fillna(0).abs().sum()
        assert changes >= 1
    
    def test_analyze_pair(self, spread_analyzer, sample_data):
        """Test the main analyze_pair method."""
        # Get pair data
        data = sample_data
        
        # Analyze pair
        result = spread_analyzer.analyze_pair(
            pair_id='S1_S2',
            data=data,
            leg1='S1',
            leg2='S2',
            hedge_ratio_method='ols'
        )
        
        # Check results
        assert isinstance(result, dict)
        assert 'spread_series' in result
        assert 'z_score_series' in result
        assert 'hedge_ratio' in result
        
        # Check types and lengths
        assert isinstance(result['spread_series'], pd.Series)
        assert isinstance(result['z_score_series'], pd.Series)
        assert isinstance(result['hedge_ratio'], float)
        assert len(result['spread_series']) == len(data['S1'])
        assert len(result['z_score_series']) == len(data['S1'])
        
        # Test with Kalman filter
        result_kalman = spread_analyzer.analyze_pair(
            pair_id='S1_S2',
            data=data,
            leg1='S1',
            leg2='S2',
            hedge_ratio_method='kalman'
        )
        
        # Check results
        assert 'spread_series' in result_kalman
        assert 'z_score_series' in result_kalman
        assert 'hedge_ratio' in result_kalman
        assert 'hedge_ratio_series' in result_kalman
        
        # With Kalman, should have a hedge ratio series
        assert isinstance(result_kalman['hedge_ratio_series'], pd.Series)
        assert len(result_kalman['hedge_ratio_series']) == len(data['S1'])
    
    def test_identify_entry_exit_points(self, spread_analyzer, sample_data):
        """Test identification of entry and exit points based on z-score."""
        # Get pair data
        data = sample_data
        
        # Analyze pair
        result = spread_analyzer.analyze_pair(
            pair_id='S1_S2',
            data=data,
            leg1='S1',
            leg2='S2'
        )
        
        # Identify entry/exit points
        entries_exits = spread_analyzer.identify_entry_exit_points(
            z_score_series=result['z_score_series'],
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Check results
        assert isinstance(entries_exits, dict)
        assert 'long_entries' in entries_exits
        assert 'long_exits' in entries_exits
        assert 'short_entries' in entries_exits
        assert 'short_exits' in entries_exits
        
        # Each should be a series of booleans
        assert isinstance(entries_exits['long_entries'], pd.Series)
        assert entries_exits['long_entries'].dtype == bool
        
        # Entry points should correspond to z-score crossing thresholds
        assert all(result['z_score_series'][entries_exits['long_entries']] <= -2.0)
        assert all(result['z_score_series'][entries_exits['short_entries']] >= 2.0)
    
    def test_calculate_spread_statistics(self, spread_analyzer, sample_data):
        """Test calculation of spread statistics."""
        # Get pair data and calculate spread
        s1 = sample_data['S1']['Close']
        s2 = sample_data['S2']['Close']
        hedge_ratio = 0.5
        spread = spread_analyzer.calculate_spread(s1, s2, hedge_ratio)
        
        # Calculate statistics
        stats = spread_analyzer.calculate_spread_statistics(spread)
        
        # Check results
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'half_life' in stats
        assert 'adf_p_value' in stats
        
        # Check types
        assert isinstance(stats['mean'], float)
        assert isinstance(stats['std'], float)
        assert isinstance(stats['half_life'], float)
        
        # Half-life should be positive for a mean-reverting series
        assert stats['half_life'] > 0 