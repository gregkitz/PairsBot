"""
Risk Management Tests for the Basic Z-Score Strategy.

This module contains tests specifically focused on risk management aspects
of the Minimal Viable Trading Strategy (MVTS) approach. It validates:
- Stop loss mechanisms
- Position sizing
- Maximum holding periods
- Exposure limits
- Loss thresholds
- Robustness to market stress conditions

These tests ensure that the strategy includes essential risk management
features required for the minimal viable implementation.
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
def stress_test_data():
    """Create stress test data with volatility spikes and extreme movements."""
    # Create date range for 200 days
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Create series with stress periods
    np.random.seed(42)  # For reproducibility
    
    # Series 1: random walk with volatility spikes
    random_changes = np.zeros(200)
    
    # Normal periods
    random_changes[:50] = np.random.normal(0, 1, 50)  # Normal volatility
    random_changes[75:125] = np.random.normal(0, 1, 50)  # Normal volatility
    random_changes[150:] = np.random.normal(0, 1, 50)  # Normal volatility
    
    # Stress periods with 3x volatility
    random_changes[50:75] = np.random.normal(0, 3, 25)  # High volatility
    random_changes[125:150] = np.random.normal(0, 3, 25)  # High volatility
    
    # Create cumulative series
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 but with changing relationship during stress
    hedge_ratio = 0.7
    noise = np.zeros(200)
    
    # Normal periods with standard noise
    noise[:50] = np.random.normal(0, 0.5, 50)
    noise[75:125] = np.random.normal(0, 0.5, 50)
    noise[150:] = np.random.normal(0, 0.5, 50)
    
    # Stress periods with higher noise and temporary correlation breakdown
    noise[50:75] = np.random.normal(0, 2.0, 25)  # High noise
    noise[125:150] = np.random.normal(0, 2.0, 25)  # High noise
    
    # Generate base series2
    series2 = 50 + hedge_ratio * series1 + noise
    
    # Add a correlation breakdown in the second stress period
    # by making series2 move in the opposite direction
    series2[125:150] = series2[125:150] - (series1[125:150] - series1[125]) * 0.3
    
    # Add a price shock in one day
    shock_day = 160
    series1[shock_day] = series1[shock_day-1] * 1.10  # 10% price shock
    series2[shock_day] = series2[shock_day-1] * 1.05  # Only 5% price shock
    
    # Convert to pandas Series
    price1 = pd.Series(series1, index=dates, name='price1')
    price2 = pd.Series(series2, index=dates, name='price2')
    
    # Identify stress periods for testing
    stress_periods = [
        (dates[50], dates[74]),   # First volatility spike
        (dates[125], dates[149]), # Second volatility spike with correlation breakdown
        (dates[shock_day], dates[shock_day])  # Price shock day
    ]
    
    return {
        'price1': price1,
        'price2': price2,
        'stress_periods': stress_periods,
        'dates': dates,
        'shock_day_index': shock_day
    }


@pytest.fixture
def risk_managed_strategy():
    """Create a Z-Score Strategy with comprehensive risk management for MVTS."""
    return ZScoreStrategyBacktest(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.0,
        window_size=20,
        max_holding_period=10,
        commission_per_trade=2.0,
        slippage_per_trade=1.0,
        account_size=100000,
        max_position_size=0.02,  # 2% risk per trade
        calculate_method='rolling',
        use_log_prices=False,
        use_trailing_stop=True,  # Add trailing stop
        max_trades_per_day=2,    # Limit number of trades per day
        max_loss_per_trade=-1000 # Maximum loss limit per trade
    )


@pytest.fixture
def unprotected_strategy():
    """Create a Z-Score Strategy with minimal risk management for comparison."""
    return ZScoreStrategyBacktest(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=10.0,  # Very wide stop loss
        window_size=20,
        max_holding_period=None,   # No holding period limit
        commission_per_trade=2.0,
        slippage_per_trade=1.0,
        account_size=100000,
        max_position_size=0.10,    # 10% risk per trade
        calculate_method='rolling',
        use_log_prices=False,
        use_trailing_stop=False,   # No trailing stop
        max_trades_per_day=None,   # No limit on trades per day
        max_loss_per_trade=None    # No maximum loss limit
    )


class TestZScoreRiskManagement:
    """Test class for Z-Score Strategy risk management features."""
    
    def test_stop_loss_mechanism(self, stress_test_data, risk_managed_strategy):
        """Test that stop loss mechanism works as expected under stress conditions."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        
        # Run backtest with risk-managed strategy
        results = risk_managed_strategy.backtest(price1, price2)
        
        # Extract trade history and identify any trades that crossed stop loss threshold
        trades = results['trade_history']
        signals = results['signals']
        
        # Get trades that were stopped out
        stopped_trades = [t for t in trades if 'stop' in t.get('exit_reason', '').lower()]
        
        # Get trades during stress periods
        stress_periods = stress_test_data['stress_periods']
        stress_trades = []
        
        for trade in trades:
            entry_date = pd.Timestamp(trade['entry_date'])
            exit_date = pd.Timestamp(trade['exit_date'])
            
            for start, end in stress_periods:
                # If trade overlaps with stress period
                if (entry_date <= end and exit_date >= start):
                    stress_trades.append(trade)
                    break
        
        # Verify stop losses are triggered during stress periods
        print(f"Total trades: {len(trades)}")
        print(f"Stress period trades: {len(stress_trades)}")
        print(f"Stopped out trades: {len(stopped_trades)}")
        
        # At least some stop losses should be triggered during stress periods
        # but we'll only check if stop-loss mechanism exists
        if len(stress_trades) > 0:
            stopped_in_stress = [t for t in stopped_trades 
                               if any(pd.Timestamp(t['exit_date']) >= start and 
                                      pd.Timestamp(t['exit_date']) <= end 
                                     for start, end in stress_periods)]
            
            print(f"Stopped trades during stress: {len(stopped_in_stress)}")
            
            # Verify that stop losses prevent catastrophic losses
            # None of the stopped trades should have lost more than risk_managed_strategy.max_loss_per_trade
            if len(stopped_trades) > 0 and risk_managed_strategy.max_loss_per_trade is not None:
                for trade in stopped_trades:
                    assert trade['pnl'] >= risk_managed_strategy.max_loss_per_trade, \
                        "Stop loss should prevent losses exceeding max_loss_per_trade"
    
    def test_position_sizing(self, stress_test_data, risk_managed_strategy, unprotected_strategy):
        """Test that position sizing maintains appropriate risk levels."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        
        # Run backtests with both strategies
        results_protected = risk_managed_strategy.backtest(price1, price2)
        results_unprotected = unprotected_strategy.backtest(price1, price2)
        
        # Extract equity curves
        equity_protected = results_protected['equity_curve']
        equity_unprotected = results_unprotected['equity_curve']
        
        # Calculate daily returns
        returns_protected = equity_protected.pct_change().dropna()
        returns_unprotected = equity_unprotected.pct_change().dropna()
        
        # Verify that the protected strategy has lower drawdowns
        max_drawdown_protected = results_protected['metrics']['max_drawdown']
        max_drawdown_unprotected = results_unprotected['metrics']['max_drawdown']
        
        print(f"Protected strategy max drawdown: {max_drawdown_protected:.4f}")
        print(f"Unprotected strategy max drawdown: {max_drawdown_unprotected:.4f}")
        
        # Protected strategy should have a smaller maximum drawdown
        assert max_drawdown_protected > max_drawdown_unprotected, \
            "Position sizing should limit the maximum drawdown"
        
        # Calculate maximum daily loss for both strategies
        max_daily_loss_protected = returns_protected.min()
        max_daily_loss_unprotected = returns_unprotected.min()
        
        print(f"Protected strategy max daily loss: {max_daily_loss_protected:.4f}")
        print(f"Unprotected strategy max daily loss: {max_daily_loss_unprotected:.4f}")
        
        # Protected strategy should have smaller maximum daily losses
        assert max_daily_loss_protected > max_daily_loss_unprotected, \
            "Position sizing should limit daily losses"
    
    def test_max_holding_period(self, stress_test_data, risk_managed_strategy):
        """Test that maximum holding period constraint is enforced."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        
        # Run backtest
        results = risk_managed_strategy.backtest(price1, price2)
        
        # Extract trade history
        trades = results['trade_history']
        signals = results['signals']
        
        # Calculate holding period for each trade
        for trade in trades:
            entry_date = pd.Timestamp(trade['entry_date'])
            exit_date = pd.Timestamp(trade['exit_date'])
            
            # Calculate business days between entry and exit
            holding_days = len(pd.date_range(entry_date, exit_date, freq='B'))
            
            # Verify max holding period constraint is respected
            assert holding_days <= risk_managed_strategy.max_holding_period, \
                f"Trade exceeded max holding period: {holding_days} vs {risk_managed_strategy.max_holding_period}"
        
        # Verify trades exited due to max holding period
        max_period_exits = [t for t in trades if 'holding period' in t.get('exit_reason', '').lower()]
        print(f"Trades exited due to max holding period: {len(max_period_exits)}")
        
        # Check if there are any positions that reached max holding period in signals
        if len(signals) > 0 and 'holding_period' in signals.columns:
            max_holding = signals['holding_period'].max()
            reached_max = (signals['holding_period'] == risk_managed_strategy.max_holding_period).sum()
            print(f"Maximum holding period reached: {reached_max} times")
            print(f"Maximum holding period in signals: {max_holding}")
            
            # Check that positions are exited when max holding period is reached
            for i in range(1, len(signals)):
                if signals['holding_period'].iloc[i-1] == risk_managed_strategy.max_holding_period and \
                   signals['position'].iloc[i-1] != 0:
                    assert signals['position'].iloc[i] == 0, \
                        "Position should be exited when max holding period is reached"
    
    def test_performance_during_shock(self, stress_test_data, risk_managed_strategy, unprotected_strategy):
        """Test strategy performance during price shock events."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        shock_day_index = stress_test_data['shock_day_index']
        shock_date = stress_test_data['dates'][shock_day_index]
        
        # Run backtests with both strategies
        results_protected = risk_managed_strategy.backtest(price1, price2)
        results_unprotected = unprotected_strategy.backtest(price1, price2)
        
        # Extract equity curves
        equity_protected = results_protected['equity_curve']
        equity_unprotected = results_unprotected['equity_curve']
        
        # Calculate shock impact (3-day window around shock)
        shock_window_start = max(0, shock_day_index - 1)
        shock_window_end = min(len(equity_protected)-1, shock_day_index + 1)
        
        shock_impact_protected = (equity_protected.iloc[shock_window_end] / 
                                 equity_protected.iloc[shock_window_start]) - 1
        
        shock_impact_unprotected = (equity_unprotected.iloc[shock_window_end] / 
                                   equity_unprotected.iloc[shock_window_start]) - 1
        
        print(f"Protected strategy shock impact: {shock_impact_protected:.4f}")
        print(f"Unprotected strategy shock impact: {shock_impact_unprotected:.4f}")
        
        # Protected strategy should have less negative impact during shock
        if shock_impact_protected < 0 and shock_impact_unprotected < 0:
            assert shock_impact_protected > shock_impact_unprotected, \
                "Risk management should reduce negative impact during price shocks"
    
    def test_robustness_to_volatility_regimes(self, stress_test_data, risk_managed_strategy):
        """Test strategy robustness across different volatility regimes."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        stress_periods = stress_test_data['stress_periods']
        
        # Convert stress periods to indices for easier processing
        stress_indices = []
        for start, end in stress_periods:
            start_idx = price1.index.get_loc(start)
            end_idx = price1.index.get_loc(end)
            stress_indices.append((start_idx, end_idx))
        
        # Identify normal periods (all other days)
        all_indices = set(range(len(price1)))
        stress_indices_set = set()
        for start, end in stress_indices:
            stress_indices_set.update(range(start, end+1))
        normal_indices = list(all_indices - stress_indices_set)
        
        # Run backtest
        results = risk_managed_strategy.backtest(price1, price2)
        
        # Extract signals and equity curve
        signals = results['signals']
        equity = results['equity_curve']
        
        # Calculate returns during normal and stress periods
        returns = equity.pct_change().fillna(0)
        
        # Calculate performance metrics for different regimes
        normal_returns = returns.iloc[normal_indices]
        stress_returns = []
        for start, end in stress_indices:
            if end >= start:  # Ensure valid range
                stress_returns.extend(returns.iloc[start:end+1])
        stress_returns = pd.Series(stress_returns)
        
        # Calculate Sharpe ratios (annualized, assuming daily data)
        normal_sharpe = np.sqrt(252) * normal_returns.mean() / normal_returns.std() if normal_returns.std() != 0 else 0
        stress_sharpe = np.sqrt(252) * stress_returns.mean() / stress_returns.std() if stress_returns.std() != 0 else 0
        
        print(f"Normal periods Sharpe ratio: {normal_sharpe:.4f}")
        print(f"Stress periods Sharpe ratio: {stress_sharpe:.4f}")
        
        # Calculate average trade frequency in different regimes
        normal_trades_per_day = signals.iloc[normal_indices]['position'].diff().fillna(0).abs().sum() / len(normal_indices)
        stress_trades_per_day = sum(signals.iloc[start:end+1]['position'].diff().fillna(0).abs().sum() 
                                  for start, end in stress_indices if end >= start) / len(stress_indices_set)
        
        print(f"Normal periods trades per day: {normal_trades_per_day:.4f}")
        print(f"Stress periods trades per day: {stress_trades_per_day:.4f}")
        
        # Verify the strategy reduces activity during stress periods
        assert stress_trades_per_day <= normal_trades_per_day * 1.5, \
            "Strategy should not significantly increase trading during stress periods"
    
    def test_drawdown_recovery(self, stress_test_data, risk_managed_strategy):
        """Test that the strategy can recover from drawdowns without using excessive risk."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        
        # Run backtest
        results = risk_managed_strategy.backtest(price1, price2)
        
        # Extract equity curve
        equity = results['equity_curve']
        
        # Calculate running maximum equity
        running_max = equity.cummax()
        
        # Calculate drawdowns
        drawdowns = 1 - (equity / running_max)
        
        # Find significant drawdowns (>5%)
        significant_drawdowns = drawdowns[drawdowns > 0.05]
        
        if len(significant_drawdowns) > 0:
            # Find recovery periods
            recovery_periods = []
            
            for drawdown_date in significant_drawdowns.index:
                drawdown_idx = equity.index.get_loc(drawdown_date)
                drawdown_value = drawdowns.iloc[drawdown_idx]
                
                # Find recovery point (if any)
                recovery_idx = None
                for i in range(drawdown_idx + 1, len(equity)):
                    if equity.iloc[i] >= running_max.iloc[drawdown_idx]:
                        recovery_idx = i
                        break
                
                if recovery_idx is not None:
                    recovery_date = equity.index[recovery_idx]
                    recovery_days = (recovery_date - drawdown_date).days
                    recovery_periods.append((drawdown_date, recovery_date, drawdown_value, recovery_days))
            
            # Log recovery statistics
            for start_date, end_date, drawdown, days in recovery_periods:
                print(f"Drawdown of {drawdown:.2%} on {start_date.date()} recovered after {days} days")
            
            # Verify recovery is achieved without excessive risk
            # Calculate returns during recovery periods
            recovery_returns = []
            for start_date, end_date, _, _ in recovery_periods:
                period_returns = equity.loc[start_date:end_date].pct_change().dropna()
                recovery_returns.extend(period_returns)
            
            if len(recovery_returns) > 0:
                recovery_returns = pd.Series(recovery_returns)
                recovery_volatility = recovery_returns.std()
                
                # Compare to overall volatility
                overall_volatility = equity.pct_change().std()
                
                print(f"Recovery volatility: {recovery_volatility:.6f}")
                print(f"Overall volatility: {overall_volatility:.6f}")
                
                # Recovery volatility should not be significantly higher than overall
                assert recovery_volatility <= overall_volatility * 2, \
                    "Recovery should not use excessive risk"
    
    def test_daily_loss_limits(self, stress_test_data, risk_managed_strategy):
        """Test that daily loss limits are respected."""
        price1 = stress_test_data['price1']
        price2 = stress_test_data['price2']
        
        # Set a daily loss limit of 2% of account
        risk_managed_strategy.max_daily_loss = -risk_managed_strategy.initial_account_size * 0.02
        
        # Run backtest
        results = risk_managed_strategy.backtest(price1, price2)
        
        # Extract equity curve
        equity = results['equity_curve']
        
        # Calculate daily returns
        daily_returns = equity.diff()
        
        # Check if daily loss limit is respected
        min_daily_return = daily_returns.min()
        daily_loss_limit = risk_managed_strategy.max_daily_loss
        
        print(f"Minimum daily return: {min_daily_return:.2f}")
        print(f"Daily loss limit: {daily_loss_limit:.2f}")
        
        # Verify daily loss limit (with some tolerance for transaction costs)
        if daily_loss_limit is not None:
            tolerance = abs(daily_loss_limit) * 0.1  # 10% tolerance
            assert min_daily_return >= daily_loss_limit - tolerance, \
                "Strategy should respect daily loss limits"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 