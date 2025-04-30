"""
Unit tests for the Z-Score Strategy.

This module contains tests for the Z-Score strategy implementation,
including spread calculation, z-score computation, signal generation,
position management, and return calculation.
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

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing the Z-Score strategy."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create series with cointegrated relationship
    np.random.seed(42)  # For reproducibility
    
    # Series 1: random walk
    random_changes = np.random.normal(0, 1, 100)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 (with noise)
    hedge_ratio = 0.7
    noise = np.random.normal(0, 0.5, 100)
    series2 = (series1 / hedge_ratio) + noise
    
    # Convert to pandas Series
    price1 = pd.Series(series1, index=dates, name='price1')
    price2 = pd.Series(series2, index=dates, name='price2')
    
    # Create mean-reverting spread
    spread = price1 - hedge_ratio * price2
    
    return {
        'price1': price1,
        'price2': price2,
        'spread': spread,
        'hedge_ratio': hedge_ratio,
        'dates': dates
    }

@pytest.fixture
def strategy():
    """Create a Z-Score Strategy instance for testing."""
    return ZScoreStrategyBacktest(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        window_size=20,
        max_holding_period=10,
        use_trailing_stop=False,
        use_time_filter=False,
        commission_per_trade=1.0,
        slippage_per_trade=0.5,
        account_size=100000,
        max_position_size=0.25,
        calculate_method='rolling',
        use_log_prices=False
    )

class TestZScoreStrategy:
    """Test class for the Z-Score Strategy implementation."""
    
    def test_calculate_hedge_ratio(self, strategy, sample_price_data):
        """Test hedge ratio calculation."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        expected_ratio = sample_price_data['hedge_ratio']
        
        # Calculate hedge ratio
        hedge_ratio = strategy.calculate_hedge_ratio(price1, price2)
        
        # Assertions
        assert isinstance(hedge_ratio, float)
        # Allow for some difference due to noise in the data
        assert abs(hedge_ratio - expected_ratio) < 0.1
    
    def test_calculate_spread(self, strategy, sample_price_data):
        """Test spread calculation."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        hedge_ratio = sample_price_data['hedge_ratio']
        expected_spread = sample_price_data['spread']
        
        # Calculate spread
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        
        # Assertions
        assert isinstance(spread, pd.Series)
        assert len(spread) == len(price1)
        assert spread.equals(expected_spread)
        
        # Test without providing hedge ratio
        spread_auto = strategy.calculate_spread(price1, price2)
        assert isinstance(spread_auto, pd.Series)
        assert len(spread_auto) == len(price1)
    
    def test_calculate_zscore(self, strategy, sample_price_data):
        """Test z-score calculation with different methods."""
        spread = sample_price_data['spread']
        
        # Test rolling method (default)
        zscore_rolling = strategy.calculate_zscore(spread)
        
        # Test other methods
        strategy.calculate_method = 'ewm'
        zscore_ewm = strategy.calculate_zscore(spread)
        
        strategy.calculate_method = 'full'
        zscore_full = strategy.calculate_zscore(spread)
        
        # Reset to default
        strategy.calculate_method = 'rolling'
        
        # Assertions
        assert isinstance(zscore_rolling, pd.Series)
        assert isinstance(zscore_ewm, pd.Series)
        assert isinstance(zscore_full, pd.Series)
        assert len(zscore_rolling) == len(spread)
        assert len(zscore_ewm) == len(spread)
        assert len(zscore_full) == len(spread)
        
        # First few values should be NaN for rolling (due to window)
        assert zscore_rolling.iloc[:strategy.window_size-1].isna().all()
        
        # All values should be populated for full
        assert not zscore_full.isna().any()
        
        # Verify statistical properties
        non_nan_zscore = zscore_rolling.dropna()
        assert abs(non_nan_zscore.mean()) < 0.5  # Mean should be close to 0
        assert 0.5 < non_nan_zscore.std() < 1.5  # Std should be close to 1
    
    def test_generate_signals(self, strategy, sample_price_data):
        """Test signal generation based on z-scores."""
        spread = sample_price_data['spread']
        zscore = strategy.calculate_zscore(spread)
        
        # Generate signals
        signals = strategy.generate_signals(zscore)
        
        # Assertions
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(zscore)
        
        # Check signal columns exist
        required_columns = ['zscore', 'entry_long', 'entry_short', 
                           'exit_long', 'exit_short', 'stop_long', 'stop_short']
        for col in required_columns:
            assert col in signals.columns
        
        # Verify signal logic
        assert (signals['entry_long'] == (zscore <= -strategy.entry_threshold).astype(int)).all()
        assert (signals['entry_short'] == (zscore >= strategy.entry_threshold).astype(int)).all()
        assert (signals['exit_long'] == (zscore >= -strategy.exit_threshold).astype(int)).all()
        assert (signals['exit_short'] == (zscore <= strategy.exit_threshold).astype(int)).all()
        assert (signals['stop_long'] == (zscore <= -strategy.stop_loss_threshold).astype(int)).all()
        assert (signals['stop_short'] == (zscore >= strategy.stop_loss_threshold).astype(int)).all()
    
    def test_apply_position_logic(self, strategy, sample_price_data):
        """Test position logic application."""
        spread = sample_price_data['spread']
        zscore = strategy.calculate_zscore(spread)
        signals = strategy.generate_signals(zscore)
        
        # Apply position logic
        positions = strategy.apply_position_logic(signals)
        
        # Assertions
        assert isinstance(positions, pd.DataFrame)
        assert 'position' in positions.columns
        assert 'holding_period' in positions.columns
        
        # Position should be -1, 0, or 1
        assert positions['position'].isin([-1, 0, 1]).all()
        
        # First position should be 0
        assert positions['position'].iloc[0] == 0
        
        # Holding period should be >= 0
        assert (positions['holding_period'] >= 0).all()
        
        # Check position changes logic
        for i in range(1, len(positions)):
            prev_pos = positions['position'].iloc[i-1]
            curr_pos = positions['position'].iloc[i]
            
            # If position changes from 0 to non-zero, it should be due to entry signal
            if prev_pos == 0 and curr_pos == 1:
                assert positions['entry_long'].iloc[i] == 1
            elif prev_pos == 0 and curr_pos == -1:
                assert positions['entry_short'].iloc[i] == 1
            
            # If position changes from non-zero to 0, it should be due to exit or stop signal
            if prev_pos == 1 and curr_pos == 0:
                assert (positions['exit_long'].iloc[i] == 1 or 
                        positions['stop_long'].iloc[i] == 1 or 
                        positions['holding_period'].iloc[i-1] >= strategy.max_holding_period - 1)
            elif prev_pos == -1 and curr_pos == 0:
                assert (positions['exit_short'].iloc[i] == 1 or 
                        positions['stop_short'].iloc[i] == 1 or 
                        positions['holding_period'].iloc[i-1] >= strategy.max_holding_period - 1)
    
    def test_calculate_returns(self, strategy, sample_price_data):
        """Test returns calculation."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        hedge_ratio = sample_price_data['hedge_ratio']
        
        # Run signal generation and position logic
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        zscore = strategy.calculate_zscore(spread)
        signals = strategy.generate_signals(zscore)
        positions = strategy.apply_position_logic(signals)
        
        # Calculate returns
        returns = strategy.calculate_returns(positions, price1, price2, hedge_ratio)
        
        # Assertions
        assert isinstance(returns, pd.DataFrame)
        assert 'returns' in returns.columns
        
        # First return should be 0
        assert returns['returns'].iloc[0] == 0
        
        # Returns should be present when in a position
        for i in range(1, len(returns)):
            if positions['position'].iloc[i-1] != 0:
                # Either return is non-zero or transaction costs were applied
                assert returns['returns'].iloc[i] != 0 or positions['position'].iloc[i] == 0
    
    def test_backtest(self, strategy, sample_price_data):
        """Test the full backtest process."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        hedge_ratio = sample_price_data['hedge_ratio']
        
        # Run backtest
        results = strategy.backtest(price1, price2, hedge_ratio)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'signals' in results
        assert 'equity_curve' in results
        assert 'trade_history' in results
        assert 'metrics' in results
        
        # Check equity curve
        equity_curve = results['equity_curve']
        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == len(price1)
        assert equity_curve.iloc[0] == strategy.initial_account_size
        
        # Check metrics
        metrics = results['metrics']
        assert isinstance(metrics, dict)
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        
        # Check instance variables
        assert strategy.results is not None
        assert strategy.equity_curve is not None
        assert len(strategy.trade_history) >= 0
        assert strategy.metrics is not None
    
    def test_backtest_with_auto_hedge_ratio(self, strategy, sample_price_data):
        """Test backtest with automatic hedge ratio calculation."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        
        # Run backtest without providing hedge ratio
        results = strategy.backtest(price1, price2)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'signals' in results
        assert 'equity_curve' in results
        assert 'trade_history' in results
        assert 'metrics' in results
    
    def test_strategies_with_different_parameters(self, sample_price_data):
        """Test strategies with different parameter sets."""
        price1 = sample_price_data['price1']
        price2 = sample_price_data['price2']
        
        # Define a set of strategies with different parameters
        strategies = [
            # Standard strategy
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=20),
            
            # More aggressive entry
            ZScoreStrategyBacktest(entry_threshold=1.5, window_size=20),
            
            # Slower strategy
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=30),
            
            # With max holding period
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=20, max_holding_period=5),
            
            # With transaction costs
            ZScoreStrategyBacktest(
                entry_threshold=2.0, window_size=20,
                commission_per_trade=2.0, slippage_per_trade=1.0
            ),
            
            # Different calculation method
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=20, calculate_method='ewm')
        ]
        
        # Run backtests for all strategies
        results = []
        for strat in strategies:
            res = strat.backtest(price1, price2)
            results.append(res)
        
        # Assertions
        for res in results:
            assert isinstance(res, dict)
            assert 'equity_curve' in res
            assert 'metrics' in res
        
        # Verify different parameters lead to different results
        equity_values = [res['equity_curve'].iloc[-1] for res in results]
        assert len(set(equity_values)) > 1  # Should have different final equity values 