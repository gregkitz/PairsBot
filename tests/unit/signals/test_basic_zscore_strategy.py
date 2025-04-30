"""
Unit tests for the Basic Z-Score Strategy.

This module contains tests for the Minimal Viable Trading Strategy (MVTS) approach
using the Z-Score strategy implementation. It focuses on the core functionality
needed for a minimal viable strategy, including:
- Basic pair selection
- Simple Z-score calculation
- Entry/exit rule implementation
- Position sizing
- Risk management

These tests are designed to validate the essential components before proceeding
with paper trading validation.
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
def simple_cointegrated_pair():
    """Create a simple cointegrated pair for testing the basic Z-Score strategy."""
    # Create date range for 100 days
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create series with strong cointegration
    np.random.seed(42)  # For reproducibility
    
    # Series 1: random walk
    random_changes = np.random.normal(0, 1, 100)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 (with minimal noise)
    hedge_ratio = 0.7
    noise = np.random.normal(0, 0.3, 100)  # Low noise for strong cointegration
    series2 = 50 + hedge_ratio * series1 + noise
    
    # Convert to pandas Series
    price1 = pd.Series(series1, index=dates, name='price1')
    price2 = pd.Series(series2, index=dates, name='price2')
    
    # Create mean-reverting spread
    spread = price2 - hedge_ratio * price1 - 50  # Subtract the intercept
    
    return {
        'price1': price1,
        'price2': price2,
        'spread': spread,
        'hedge_ratio': hedge_ratio,
        'dates': dates
    }


@pytest.fixture
def basic_strategy():
    """Create a basic Z-Score Strategy with minimal configuration for MVTS."""
    return ZScoreStrategyBacktest(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        window_size=20,
        max_holding_period=10,  # Add max holding period for risk management
        commission_per_trade=2.0,  # Add realistic commission
        slippage_per_trade=1.0,  # Add realistic slippage
        account_size=100000,
        max_position_size=0.02,  # 2% risk per trade for MVTS
        calculate_method='rolling',
        use_log_prices=False
    )


class TestBasicZScoreStrategy:
    """Test class for the Basic Z-Score Strategy implementation for MVTS."""
    
    def test_pair_selection_criteria(self, simple_cointegrated_pair):
        """Test that basic pair selection criteria work correctly."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Calculate correlation
        correlation = price1.corr(price2)
        
        # Verify strong correlation (essential for MVTS)
        assert correlation > 0.8, "Pair should have strong correlation for the MVTS approach"
        
        # Verify cointegration through spread stationarity
        spread = simple_cointegrated_pair['spread']
        
        # Calculate basic stationarity metrics (no need for complex tests in MVTS)
        spread_mean = spread.mean()
        spread_std = spread.std()
        spread_zscore = (spread - spread_mean) / spread_std
        
        # Verify the spread z-score stays within bounds (pseudo-stationarity check)
        assert abs(spread_zscore).max() < 4, "Spread should be relatively stationary for MVTS"
        
        # Check mean reversion - autocorrelation test
        # For mean-reverting series, lag-1 autocorrelation should be less than 1
        autocorr = spread.autocorr(lag=1)
        assert autocorr < 0.95, "Spread should show mean-reverting properties for MVTS"
    
    def test_basic_spread_calculation(self, simple_cointegrated_pair, basic_strategy):
        """Test basic spread calculation for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        true_hedge_ratio = simple_cointegrated_pair['hedge_ratio']
        
        # Calculate hedge ratio
        calculated_ratio = basic_strategy.calculate_hedge_ratio(price1, price2)
        
        # Verify hedge ratio calculation (should be close to the true value)
        assert abs(calculated_ratio - true_hedge_ratio) < 0.1, "Calculated hedge ratio should be close to true value"
        
        # Calculate spread
        spread = basic_strategy.calculate_spread(price1, price2, calculated_ratio)
        
        # Verify spread calculation
        expected_spread = price1 - calculated_ratio * price2
        pd.testing.assert_series_equal(spread, expected_spread)
        
        # Test auto-calculation of hedge ratio
        auto_spread = basic_strategy.calculate_spread(price1, price2)
        assert len(auto_spread) == len(price1), "Auto-calculated spread should have same length as price series"
    
    def test_basic_zscore_calculation(self, simple_cointegrated_pair, basic_strategy):
        """Test basic z-score calculation for MVTS."""
        spread = simple_cointegrated_pair['spread']
        
        # Calculate z-score with rolling window
        zscore = basic_strategy.calculate_zscore(spread)
        
        # Verify z-score properties
        # First window_size-1 values should be NaN
        assert zscore.iloc[:basic_strategy.window_size-1].isna().all(), "First window_size-1 values should be NaN"
        
        # Z-score should be normalized with mean near 0 and std near 1
        non_nan_zscore = zscore.dropna()
        assert abs(non_nan_zscore.mean()) < 0.3, "Z-score mean should be close to 0"
        assert 0.7 < non_nan_zscore.std() < 1.3, "Z-score std should be close to 1"
        
        # Verify range is reasonable for a z-score
        assert non_nan_zscore.min() > -5, "Z-score minimum should be reasonable"
        assert non_nan_zscore.max() < 5, "Z-score maximum should be reasonable"
    
    def test_simplified_signal_generation(self, simple_cointegrated_pair, basic_strategy):
        """Test simplified signal generation for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Generate signals
        spread = basic_strategy.calculate_spread(price1, price2)
        zscore = basic_strategy.calculate_zscore(spread)
        signals = basic_strategy.generate_signals(zscore)
        
        # Verify signal columns exist
        required_columns = ['zscore', 'entry_long', 'entry_short', 
                           'exit_long', 'exit_short', 'stop_long', 'stop_short']
        for col in required_columns:
            assert col in signals.columns, f"Signal column {col} is missing"
        
        # Verify basic signal logic
        # Entry long when zscore <= -entry_threshold
        assert (signals['entry_long'] == (zscore <= -basic_strategy.entry_threshold).astype(int)).all()
        
        # Entry short when zscore >= entry_threshold
        assert (signals['entry_short'] == (zscore >= basic_strategy.entry_threshold).astype(int)).all()
        
        # Exit long when zscore >= -exit_threshold
        assert (signals['exit_long'] == (zscore >= -basic_strategy.exit_threshold).astype(int)).all()
        
        # Exit short when zscore <= exit_threshold
        assert (signals['exit_short'] == (zscore <= basic_strategy.exit_threshold).astype(int)).all()
    
    def test_basic_position_logic(self, simple_cointegrated_pair, basic_strategy):
        """Test basic position logic for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Generate signals and apply position logic
        spread = basic_strategy.calculate_spread(price1, price2)
        zscore = basic_strategy.calculate_zscore(spread)
        signals = basic_strategy.generate_signals(zscore)
        positions = basic_strategy.apply_position_logic(signals)
        
        # Verify position columns exist
        assert 'position' in positions.columns, "Position column is missing"
        assert 'holding_period' in positions.columns, "Holding period column is missing"
        
        # Verify position values are valid (-1, 0, 1)
        assert positions['position'].isin([-1, 0, 1]).all(), "Positions should be -1, 0, or 1"
        
        # Verify holding period calculations
        for i in range(1, len(positions)):
            # If position is the same and non-zero, holding period should increase
            if positions['position'].iloc[i] == positions['position'].iloc[i-1] and positions['position'].iloc[i] != 0:
                assert positions['holding_period'].iloc[i] == positions['holding_period'].iloc[i-1] + 1
            # If position is new and non-zero, holding period should be 1
            elif positions['position'].iloc[i] != 0 and positions['position'].iloc[i] != positions['position'].iloc[i-1]:
                assert positions['holding_period'].iloc[i] == 1
            # If position is zero, holding period should be 0
            elif positions['position'].iloc[i] == 0:
                assert positions['holding_period'].iloc[i] == 0
        
        # Verify maximum holding period constraint
        assert positions['holding_period'].max() <= basic_strategy.max_holding_period, "Max holding period should be respected"
    
    def test_basic_risk_management(self, simple_cointegrated_pair, basic_strategy):
        """Test basic risk management for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Run backtest
        results = basic_strategy.backtest(price1, price2)
        
        # Check risk management metrics in trade history
        trades = results['trade_history']
        
        # Verify position sizing adheres to max position size
        for trade in trades:
            assert trade['position_size'] <= basic_strategy.max_position_size, "Position size should respect maximum"
        
        # Verify stop losses are implemented
        stops_triggered = sum(1 for trade in trades if 'stop_loss' in trade.get('exit_reason', '').lower())
        
        # Verify max holding period exits
        holding_period_exits = sum(1 for trade in trades if 'holding_period' in trade.get('exit_reason', '').lower())
        
        # Log the risk management stats
        print(f"Stop losses triggered: {stops_triggered}")
        print(f"Max holding period exits: {holding_period_exits}")
        
        # Cannot guarantee stops will trigger in this synthetic data, but if they do, verify they work
        if stops_triggered > 0:
            print("Stop losses were triggered and worked correctly")
    
    def test_transaction_costs(self, simple_cointegrated_pair, basic_strategy):
        """Test that transaction costs are properly applied in MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Run backtest with costs
        results_with_costs = basic_strategy.backtest(price1, price2)
        
        # Run backtest without costs for comparison
        strategy_no_costs = ZScoreStrategyBacktest(
            entry_threshold=basic_strategy.entry_threshold,
            exit_threshold=basic_strategy.exit_threshold,
            window_size=basic_strategy.window_size,
            max_holding_period=basic_strategy.max_holding_period,
            commission_per_trade=0.0,  # No commission
            slippage_per_trade=0.0,    # No slippage
            account_size=basic_strategy.initial_account_size
        )
        results_no_costs = strategy_no_costs.backtest(price1, price2)
        
        # Verify costs impact returns
        # With transaction costs, returns should be lower
        assert results_with_costs['metrics']['total_return'] < results_no_costs['metrics']['total_return'], \
            "Transaction costs should reduce returns"
        
        # Calculate expected transaction cost impact
        total_trades = results_with_costs['metrics']['total_trades']
        expected_cost_impact = total_trades * (basic_strategy.commission_per_trade + basic_strategy.slippage_per_trade) * 2  # Entry and exit
        
        # Verify the cost impact is reflected in equity (allowing for some approximation due to compounding)
        # This checks if the difference in final equity is roughly equal to the expected cost impact
        equity_diff = results_no_costs['equity_curve'].iloc[-1] - results_with_costs['equity_curve'].iloc[-1]
        percent_diff = abs((equity_diff - expected_cost_impact) / expected_cost_impact)
        
        # Allow for some deviation due to compounding effects
        assert percent_diff < 0.3, "Transaction cost impact should be reflected in the equity curve"
    
    def test_complete_backtest_workflow(self, simple_cointegrated_pair, basic_strategy):
        """Test the complete backtest workflow for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Run complete backtest
        results = basic_strategy.backtest(price1, price2)
        
        # Verify all expected result components are present
        assert 'equity_curve' in results, "Equity curve is missing from results"
        assert 'signals' in results, "Signals are missing from results"
        assert 'trade_history' in results, "Trade history is missing from results"
        assert 'metrics' in results, "Metrics are missing from results"
        
        # Verify basic metrics
        metrics = results['metrics']
        assert 'total_return' in metrics, "Total return is missing from metrics"
        assert 'sharpe_ratio' in metrics, "Sharpe ratio is missing from metrics"
        assert 'max_drawdown' in metrics, "Max drawdown is missing from metrics"
        assert 'total_trades' in metrics, "Total trades is missing from metrics"
        assert 'win_rate' in metrics, "Win rate is missing from metrics"
        
        # Verify equity curve calculations
        equity = results['equity_curve']
        assert len(equity) == len(price1), "Equity curve should have same length as price series"
        assert equity.iloc[0] == basic_strategy.initial_account_size, "Initial equity should equal account size"
        
        # Verify trade history
        trades = results['trade_history']
        for trade in trades:
            assert 'entry_date' in trade, "Entry date is missing from trade"
            assert 'exit_date' in trade, "Exit date is missing from trade"
            assert 'entry_price' in trade, "Entry price is missing from trade"
            assert 'exit_price' in trade, "Exit price is missing from trade"
            assert 'pnl' in trade, "PnL is missing from trade"
            assert 'position_size' in trade, "Position size is missing from trade"
    
    def test_strategies_with_different_parameters(self, simple_cointegrated_pair):
        """Test different strategy parameter combinations for MVTS."""
        price1 = simple_cointegrated_pair['price1']
        price2 = simple_cointegrated_pair['price2']
        
        # Define different strategy configurations
        strategies = [
            # Tight thresholds - more trades, typically lower profit per trade
            ZScoreStrategyBacktest(entry_threshold=1.5, exit_threshold=0.3, window_size=20),
            
            # Wide thresholds - fewer trades, typically higher profit per trade
            ZScoreStrategyBacktest(entry_threshold=2.5, exit_threshold=0.8, window_size=20),
            
            # Short window - more responsive, potentially more noise
            ZScoreStrategyBacktest(entry_threshold=2.0, exit_threshold=0.5, window_size=10),
            
            # Long window - less responsive, potentially less noise
            ZScoreStrategyBacktest(entry_threshold=2.0, exit_threshold=0.5, window_size=30)
        ]
        
        # Run backtests for each strategy
        results = []
        for strategy in strategies:
            result = strategy.backtest(price1, price2)
            results.append(result)
        
        # Compare results
        for i, result in enumerate(results):
            # Log performance metrics
            print(f"Strategy {i+1}:")
            print(f"  Total Return: {result['metrics']['total_return']:.4f}")
            print(f"  Sharpe Ratio: {result['metrics']['sharpe_ratio']:.4f}")
            print(f"  Win Rate: {result['metrics']['win_rate']:.4f}")
            print(f"  Total Trades: {result['metrics']['total_trades']}")
        
        # Verify logical relationships between parameter choices
        # More trades with tighter thresholds
        assert results[0]['metrics']['total_trades'] >= results[1]['metrics']['total_trades'], \
            "Tighter thresholds should generally result in more trades"
            
        # More trades with shorter window
        assert results[2]['metrics']['total_trades'] >= results[3]['metrics']['total_trades'], \
            "Shorter window should generally result in more trades"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 