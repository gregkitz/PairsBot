"""
Unit tests for the SignalGenerator class.

This module contains tests for the signal generation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from src.signal_generation.signal_generator import SignalGenerator

@pytest.fixture
def sample_spread_data():
    """Fixture providing sample spread data for signal generation."""
    # Create synthetic z-score and spread series
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 days of data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate z-score series with some patterns
    # Start around 0
    z_score = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 2.1, 2.4]  # crosses 2.0 threshold
    # Then drifts back to 0
    z_score += [2.2, 2.0, 1.8, 1.5, 1.2, 0.8, 0.5, 0.3, 0.1, -0.1]  # crosses 0.5 threshold down
    # Then goes negative
    z_score += [-0.3, -0.5, -0.8, -1.2, -1.5, -1.8, -2.1, -2.4, -2.6, -2.8]  # crosses -2.0 threshold
    # Then returns to 0
    z_score += [-2.5, -2.2, -1.8, -1.5, -1.2, -0.8, -0.5, -0.3, -0.1, 0.1]  # crosses -0.5 threshold up
    # Then repeat the pattern
    z_score = z_score * 2 + list(np.random.normal(0, 0.2, 20))  # Add some more data with noise
    
    # Make sure length is 100
    z_score = z_score[:100]
    
    # Create z-score Series
    z_score_series = pd.Series(z_score, index=dates, name='z_score')
    
    # Create dummy spread Series
    spread_series = pd.Series(np.random.normal(0, 1, 100), index=dates, name='spread')
    
    # Return as dict
    return {
        'z_score_series': z_score_series,
        'spread_series': spread_series
    }

@pytest.fixture
def sample_spread_analysis(sample_spread_data):
    """Fixture providing a sample spread analysis result."""
    return {
        'pair_id': 'S1_S2',
        'leg1': 'S1',
        'leg2': 'S2',
        'hedge_ratio': 0.5,
        'z_score_series': sample_spread_data['z_score_series'],
        'spread_series': sample_spread_data['spread_series']
    }

@pytest.fixture
def test_config_file():
    """Fixture providing the path to the test configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(current_dir, '..', '..', 'fixtures')
    return os.path.join(fixtures_dir, 'test_config.json')

@pytest.fixture
def signal_generator(test_config_file):
    """Fixture providing a SignalGenerator instance."""
    # Load config
    with open(test_config_file, 'r') as f:
        config = json.load(f)
    
    # Create generator with config
    generator = SignalGenerator(config.get('signal_generator', {}))
    return generator

class TestSignalGenerator:
    """Test class for SignalGenerator."""
    
    def test_initialization(self, test_config_file):
        """Test that the generator is initialized correctly."""
        # Load config
        with open(test_config_file, 'r') as f:
            config = json.load(f)
            
        # Create generator
        generator = SignalGenerator(config.get('signal_generator', {}))
        
        # Check config loaded correctly
        assert generator.entry_threshold == config['signal_generator']['entry_threshold']
        assert generator.exit_threshold == config['signal_generator']['exit_threshold']
        assert generator.stop_loss_threshold == config['signal_generator']['stop_loss_threshold']
    
    def test_generate_basic_signals(self, signal_generator, sample_spread_data):
        """Test generating basic trading signals from z-score."""
        # Get z-score series
        z_score = sample_spread_data['z_score_series']
        
        # Generate signals
        signals = signal_generator.generate_basic_signals(
            z_score,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Check results
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(z_score)
        
        # Check signal values
        # 1 = long position, -1 = short position, 0 = no position
        assert signals.isin([1, 0, -1]).all()
        
        # Check specific signals at threshold crossings
        # Find where z-score crosses below -2.0 from above (long entry)
        long_entries = ((z_score.shift(1) > -2.0) & (z_score <= -2.0)).values
        # Find where z-score crosses above 2.0 from below (short entry)
        short_entries = ((z_score.shift(1) < 2.0) & (z_score >= 2.0)).values
        
        # Check if signals match these entries
        assert any(signals.iloc[np.where(long_entries)[0]] == 1)
        assert any(signals.iloc[np.where(short_entries)[0]] == -1)
    
    def test_apply_stop_loss(self, signal_generator, sample_spread_data):
        """Test applying stop-loss to trading signals."""
        # Get z-score series
        z_score = sample_spread_data['z_score_series']
        
        # Generate basic signals first
        basic_signals = signal_generator.generate_basic_signals(
            z_score,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Apply stop-loss
        signals_with_stop = signal_generator.apply_stop_loss(
            basic_signals,
            z_score,
            stop_loss_threshold=3.0
        )
        
        # Check results
        assert isinstance(signals_with_stop, pd.Series)
        assert len(signals_with_stop) == len(basic_signals)
        
        # Find where stop-loss should trigger for long positions
        long_stops = (basic_signals.shift(1) == 1) & (z_score <= -3.0)
        # Find where stop-loss should trigger for short positions
        short_stops = (basic_signals.shift(1) == -1) & (z_score >= 3.0)
        
        # Verify that positions are closed at stop-loss points
        assert all(signals_with_stop[long_stops | short_stops] == 0)
    
    def test_apply_time_stop(self, signal_generator, sample_spread_data):
        """Test applying time-based stops to trading signals."""
        # Get z-score series
        z_score = sample_spread_data['z_score_series']
        
        # Generate basic signals first
        basic_signals = signal_generator.generate_basic_signals(
            z_score,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Apply time stop (exit after 5 periods)
        signals_with_time_stop = signal_generator.apply_time_stop(
            basic_signals,
            time_stop_bars=5
        )
        
        # Check results
        assert isinstance(signals_with_time_stop, pd.Series)
        assert len(signals_with_time_stop) == len(basic_signals)
        
        # Find where time stop should trigger
        # This is complex to test directly since it depends on the pattern of signals
        # But we can check that no position is held for more than time_stop_bars
        position_durations = []
        current_pos = 0
        duration = 0
        
        for signal in signals_with_time_stop:
            if signal == current_pos and signal != 0:
                duration += 1
            elif signal != current_pos:
                if duration > 0:
                    position_durations.append(duration)
                current_pos = signal
                duration = 1 if signal != 0 else 0
        
        if duration > 0:
            position_durations.append(duration)
        
        # No position should be held longer than the time stop
        assert all(d <= 5 for d in position_durations)
    
    def test_apply_confirmation_filter(self, signal_generator, sample_spread_data):
        """Test applying confirmation filter to trading signals."""
        # Get z-score series
        z_score = sample_spread_data['z_score_series']
        
        # Create dummy volume data
        dates = z_score.index
        vol1 = pd.Series(np.random.randint(1000, 10000, len(dates)), index=dates)
        vol2 = pd.Series(np.random.randint(500, 5000, len(dates)), index=dates)
        
        # Generate basic signals first
        basic_signals = signal_generator.generate_basic_signals(
            z_score,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Apply confirmation filter
        filtered_signals = signal_generator.apply_confirmation_filter(
            basic_signals,
            z_score,
            volume1=vol1,
            volume2=vol2,
            min_volume_percentile=20
        )
        
        # Check results
        assert isinstance(filtered_signals, pd.Series)
        assert len(filtered_signals) == len(basic_signals)
        
        # Filtered signals should have fewer or equal entries
        assert filtered_signals.diff().abs().sum() <= basic_signals.diff().abs().sum()
    
    def test_generate_signals(self, signal_generator, sample_spread_analysis):
        """Test the main generate_signals method."""
        # Generate signals from spread analysis
        results = signal_generator.generate_signals(
            spread_analysis=sample_spread_analysis,
            entry_threshold=2.0,
            exit_threshold=0.5,
            stop_loss_threshold=3.0
        )
        
        # Check results
        assert isinstance(results, dict)
        assert 'signal' in results
        assert 'entries' in results
        assert 'exits' in results
        
        # Check signal series
        signal = results['signal']
        assert isinstance(signal, pd.Series)
        assert len(signal) == len(sample_spread_analysis['z_score_series'])
        assert signal.isin([1, 0, -1]).all()
        
        # Test with customized parameters
        results_custom = signal_generator.generate_signals(
            spread_analysis=sample_spread_analysis,
            entry_threshold=1.5,  # Lower threshold
            exit_threshold=0.3,   # Lower exit threshold
            stop_loss_threshold=4.0,  # Higher stop-loss
            time_stop_bars=10     # Longer time stop
        )
        
        # Should have different signals
        assert not results['signal'].equals(results_custom['signal'])
        
        # Lower entry threshold should result in more entries
        assert results_custom['entries'].sum() >= results['entries'].sum()
    
    def test_calculate_returns(self, signal_generator, sample_spread_analysis):
        """Test calculating returns from signals."""
        # Generate signals
        signal_results = signal_generator.generate_signals(
            spread_analysis=sample_spread_analysis
        )
        
        # Calculate returns
        returns = signal_generator.calculate_returns(
            signal_results['signal'],
            sample_spread_analysis['spread_series']
        )
        
        # Check results
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(signal_results['signal'])
        
        # Manually calculate some returns to verify
        signal = signal_results['signal']
        spread = sample_spread_analysis['spread_series']
        
        # For signals at t-1, we get returns from t-1 to t
        manual_returns = signal.shift(1) * spread.pct_change()
        
        # Compare returns (ignoring NaNs)
        valid_idx = ~(returns.isna() | manual_returns.isna())
        pd.testing.assert_series_equal(returns[valid_idx], manual_returns[valid_idx])
    
    def test_generate_summary(self, signal_generator, sample_spread_analysis):
        """Test generating a trading summary."""
        # Generate signals
        signal_results = signal_generator.generate_signals(
            spread_analysis=sample_spread_analysis
        )
        
        # Calculate returns
        returns = signal_generator.calculate_returns(
            signal_results['signal'],
            sample_spread_analysis['spread_series']
        )
        
        # Generate summary
        summary = signal_generator.generate_summary(
            signal_results,
            returns
        )
        
        # Check results
        assert isinstance(summary, dict)
        assert 'total_trades' in summary
        assert 'winning_trades' in summary
        assert 'losing_trades' in summary
        assert 'win_rate' in summary
        assert 'avg_return' in summary
        assert 'max_drawdown' in summary
        
        # Check types and values
        assert isinstance(summary['total_trades'], int)
        assert isinstance(summary['win_rate'], float)
        assert 0 <= summary['win_rate'] <= 1
        assert summary['winning_trades'] + summary['losing_trades'] == summary['total_trades'] 