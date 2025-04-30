"""
Tests for edge cases and extreme scenarios for the Z-Score Strategy Backtest.

This module contains tests for edge cases and extreme scenarios that the
ZScoreStrategyBacktest class should handle properly, such as:
- Handling data with missing values
- Dealing with extreme volatility
- Handling correlation breakdowns
- Testing with invalid inputs
- Stress testing with extreme parameter values
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.backtest.zscore_strategy_backtest import ZScoreStrategyBacktest
from tests.fixtures.backtest_data.zscore_test_pairs import (
    generate_mean_reverting_pair,
    generate_trending_pair_with_correlation_breakdown,
    generate_extreme_volatility_pair,
    generate_dataset_with_gaps
)

@pytest.fixture
def correlation_breakdown_data():
    """Return data with correlation breakdown for testing."""
    return generate_trending_pair_with_correlation_breakdown()

@pytest.fixture
def extreme_volatility_data():
    """Return data with extreme volatility spikes for testing."""
    return generate_extreme_volatility_pair()

@pytest.fixture
def gappy_data():
    """Return data with missing values for testing."""
    return generate_dataset_with_gaps()

@pytest.fixture
def base_strategy():
    """Return a basic strategy instance for testing."""
    return ZScoreStrategyBacktest(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        window_size=20
    )

class TestZScoreStrategyEdgeCases:
    """Test class for Z-Score Strategy Backtest edge cases."""
    
    def test_handling_missing_data(self, base_strategy, gappy_data):
        """Test that the strategy properly handles data with missing values."""
        price1 = gappy_data['price1']
        price2 = gappy_data['price2']
        
        # Should run without errors
        result = base_strategy.backtest(price1, price2)
        
        # Basic checks
        assert isinstance(result, dict)
        assert 'equity_curve' in result
        assert 'metrics' in result
        
        # Check that the equity curve has no NaN values
        assert not result['equity_curve'].isna().any()
        
        # Verify strategy skips dates with missing data in signals
        signals = result['signals']
        for date in price1.index[price1.isna()]:
            if date in signals.index:
                assert np.isnan(signals.loc[date, 'zscore']) or signals.loc[date, 'position'] == 0
    
    def test_correlation_breakdown(self, base_strategy, correlation_breakdown_data):
        """Test strategy performance during correlation breakdown periods."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        breakdown_date = correlation_breakdown_data['breakdown_date']
        
        result = base_strategy.backtest(price1, price2)
        
        # Compare performance before and after breakdown
        signals = result['signals']
        equity = result['equity_curve']
        
        # Get indices for before and after breakdown
        before_breakdown = equity.index < breakdown_date
        after_breakdown = equity.index >= breakdown_date
        
        # Calculate returns for each period
        if sum(before_breakdown) > 0 and sum(after_breakdown) > 0:
            before_return = (equity[before_breakdown][-1] / equity[before_breakdown][0]) - 1
            after_return = (equity[after_breakdown][-1] / equity[after_breakdown][0]) - 1
            
            # Verify the strategy either reduces exposure or performs worse after breakdown
            # (Performance deterioration is expected after correlation breakdown)
            signals_before = signals[before_breakdown]
            signals_after = signals[after_breakdown]
            
            # Average absolute position size (exposure)
            avg_exposure_before = abs(signals_before['position']).mean() if len(signals_before) > 0 else 0
            avg_exposure_after = abs(signals_after['position']).mean() if len(signals_after) > 0 else 0
            
            # Either the strategy should reduce exposure or performance should deteriorate
            assert avg_exposure_after <= avg_exposure_before * 1.2 or after_return <= before_return
    
    def test_extreme_volatility(self, base_strategy, extreme_volatility_data):
        """Test strategy behavior during periods of extreme volatility."""
        price1 = extreme_volatility_data['price1']
        price2 = extreme_volatility_data['price2']
        spike_dates = extreme_volatility_data['volatility_spike_dates']
        
        # Run backtest
        result = base_strategy.backtest(price1, price2)
        
        # Check if stop-losses are triggered more during high volatility periods
        signals = result['signals']
        
        # Define windows around volatility spikes (5 days before and after)
        high_vol_periods = []
        for spike_date in spike_dates:
            start_date = spike_date - pd.Timedelta(days=5)
            end_date = spike_date + pd.Timedelta(days=15)  # Include the 10-day spike period
            high_vol_periods.append((start_date, end_date))
        
        # Identify stop-loss events (position changes from non-zero to zero without crossing exit threshold)
        stop_loss_events = []
        prev_position = 0
        for i, row in signals.iterrows():
            # If position went from non-zero to zero and z-score is beyond exit threshold
            if prev_position != 0 and row['position'] == 0 and abs(row['zscore']) > base_strategy.exit_threshold:
                stop_loss_events.append(i)
            prev_position = row['position']
        
        # Count stop-loss events during high volatility periods
        high_vol_stop_losses = 0
        for event_date in stop_loss_events:
            for start_date, end_date in high_vol_periods:
                if start_date <= event_date <= end_date:
                    high_vol_stop_losses += 1
                    break
        
        # If there are stop-loss events, verify a higher proportion occurs during volatility spikes
        if len(stop_loss_events) > 0:
            # Calculate the proportion of time that's considered "high volatility"
            high_vol_days = sum((min(period[1], signals.index[-1]) - max(period[0], signals.index[0])).days + 1 
                              for period in high_vol_periods)
            total_days = (signals.index[-1] - signals.index[0]).days + 1
            high_vol_proportion = high_vol_days / total_days
            
            # Calculate the proportion of stop-loss events during high volatility
            stop_loss_proportion = high_vol_stop_losses / len(stop_loss_events)
            
            # The proportion of stop-losses during high volatility should be higher than the time proportion
            # This may not always hold due to randomness, so we use a relaxed assertion
            assert stop_loss_proportion >= high_vol_proportion * 0.5
    
    def test_invalid_inputs(self):
        """Test that the strategy handles invalid inputs appropriately."""
        strategy = ZScoreStrategyBacktest()
        
        # Test with empty series
        empty_series = pd.Series([])
        with pytest.raises(ValueError):
            strategy.backtest(empty_series, empty_series)
        
        # Test with series of different lengths
        series1 = pd.Series(np.random.random(100), index=pd.date_range('2023-01-01', periods=100))
        series2 = pd.Series(np.random.random(50), index=pd.date_range('2023-01-01', periods=50))
        
        # Should not raise error, but should trim to common dates
        result = strategy.backtest(series1, series2)
        assert len(result['signals']) <= 50
        
        # Test with non-DatetimeIndex series
        bad_series1 = pd.Series(np.random.random(100))
        bad_series2 = pd.Series(np.random.random(100))
        
        # Should raise ValueError or convert to DatetimeIndex
        try:
            strategy.backtest(bad_series1, bad_series2)
            # If it doesn't raise error, it should have converted the index
            assert isinstance(strategy.backtest(bad_series1, bad_series2)['signals'].index, pd.DatetimeIndex)
        except ValueError:
            # This is also an acceptable behavior
            pass
    
    def test_extreme_parameters(self, correlation_breakdown_data):
        """Test strategy with extreme parameter values."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        
        # Test with very small window size
        small_window_strategy = ZScoreStrategyBacktest(window_size=3)
        small_result = small_window_strategy.backtest(price1, price2)
        
        # Test with very large window size
        large_window_strategy = ZScoreStrategyBacktest(window_size=100)
        large_result = large_window_strategy.backtest(price1, price2)
        
        # Test with very tight thresholds
        tight_strategy = ZScoreStrategyBacktest(entry_threshold=0.5, exit_threshold=0.1)
        tight_result = tight_strategy.backtest(price1, price2)
        
        # Test with very wide thresholds
        wide_strategy = ZScoreStrategyBacktest(entry_threshold=5.0, exit_threshold=2.0)
        wide_result = wide_strategy.backtest(price1, price2)
        
        # Test with very high commission
        high_cost_strategy = ZScoreStrategyBacktest(commission_per_trade=100.0)
        high_cost_result = high_cost_strategy.backtest(price1, price2)
        
        # Basic checks that all backtests ran without errors
        strategies = [small_result, large_result, tight_result, wide_result, high_cost_result]
        for result in strategies:
            assert isinstance(result, dict)
            assert 'equity_curve' in result
            assert 'metrics' in result
        
        # Verify extreme parameters lead to expected behaviors:
        
        # 1. Small window size should generate more trades than large window size
        assert small_result['metrics']['total_trades'] >= large_result['metrics']['total_trades']
        
        # 2. Tight thresholds should generate more trades than wide thresholds
        assert tight_result['metrics']['total_trades'] >= wide_result['metrics']['total_trades']
        
        # 3. High costs should result in worse returns
        standard_strategy = ZScoreStrategyBacktest()
        standard_result = standard_strategy.backtest(price1, price2)
        assert high_cost_result['metrics']['total_return'] <= standard_result['metrics']['total_return']

    def test_long_only_and_short_only_modes(self, base_strategy, correlation_breakdown_data):
        """Test strategy in long-only and short-only modes."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        
        # Create long-only and short-only strategies
        long_only = ZScoreStrategyBacktest(allow_long=True, allow_short=False)
        short_only = ZScoreStrategyBacktest(allow_long=False, allow_short=True)
        
        # Run backtests
        long_result = long_only.backtest(price1, price2)
        short_result = short_only.backtest(price1, price2)
        
        # Verify position constraints are respected
        assert (long_result['signals']['position'] >= 0).all()
        assert (short_result['signals']['position'] <= 0).all()
        
        # Check that both strategies have some trades
        assert long_result['metrics']['total_trades'] > 0
        assert short_result['metrics']['total_trades'] > 0
    
    def test_max_holding_period(self, correlation_breakdown_data):
        """Test the max holding period constraint."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        
        # Create strategy with short max holding period
        max_hold_strategy = ZScoreStrategyBacktest(
            max_holding_period=5,
            entry_threshold=1.5  # Lower threshold to ensure more trades
        )
        
        # Run backtest
        result = max_hold_strategy.backtest(price1, price2)
        
        # Check position durations
        signals = result['signals']
        
        # Find trade entry and exit points
        position_changes = signals['position'].diff().fillna(0) != 0
        trade_points = signals[position_changes]
        
        # Group into trades
        in_trade = False
        trade_start = None
        for date, row in trade_points.iterrows():
            if not in_trade and row['position'] != 0:
                # Trade entry
                trade_start = date
                in_trade = True
            elif in_trade and row['position'] == 0:
                # Trade exit
                trade_duration = (date - trade_start).days
                # Allow +1 for potential execution on the next day
                assert trade_duration <= max_hold_strategy.max_holding_period + 1
                in_trade = False

    def test_trailing_stop(self, correlation_breakdown_data):
        """Test the trailing stop functionality."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        
        # Create strategy with trailing stop
        trailing_stop_strategy = ZScoreStrategyBacktest(
            use_trailing_stop=True,
            entry_threshold=1.5  # Lower threshold to ensure more trades
        )
        
        # Run backtest
        result = trailing_stop_strategy.backtest(price1, price2)
        
        # Check that trailing stops are working
        # This is hard to test deterministically, but we can check that the strategy
        # has trades and doesn't encounter errors
        assert result['metrics']['total_trades'] > 0
        
        # For a more detailed test, we'd need to track the maximum profit for each
        # trade and verify the exit occurs after a specific retracement from that max
        
    def test_z_score_calculation_methods(self, base_strategy, correlation_breakdown_data):
        """Test different z-score calculation methods."""
        price1 = correlation_breakdown_data['price1']
        price2 = correlation_breakdown_data['price2']
        
        # Create strategies with different calculation methods
        rolling_strategy = ZScoreStrategyBacktest(calculate_method='rolling')
        ewm_strategy = ZScoreStrategyBacktest(calculate_method='ewm')
        
        # Run backtests
        rolling_result = rolling_strategy.backtest(price1, price2)
        ewm_result = ewm_strategy.backtest(price1, price2)
        
        # Verify both methods ran without errors
        assert isinstance(rolling_result, dict)
        assert isinstance(ewm_result, dict)
        
        # Check that z-scores are different between methods
        # Get z-scores for a comparison period
        mid_point = len(rolling_result['signals']) // 2
        start_idx = max(0, mid_point - 10)
        end_idx = min(len(rolling_result['signals']), mid_point + 10)
        
        rolling_zscores = rolling_result['signals']['zscore'][start_idx:end_idx]
        ewm_zscores = ewm_result['signals']['zscore'][start_idx:end_idx]
        
        # There should be at least some differences in the z-scores
        assert not rolling_zscores.equals(ewm_zscores) 