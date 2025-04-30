"""
Integration tests for Kalman filter with other system components.

This module contains integration tests that verify the Kalman filter implementation
works correctly with other system components like the z-score strategy and backtesting engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.cointegration.kalman_filter import estimate_timevarying_hedge_ratio
from src.spread_analytics.spread_analyzer import SpreadAnalyzer
from src.signal_generation.signal_generator import SignalGenerator
from src.backtest.backtest_engine import BacktestEngine


@pytest.fixture
def synthetic_trading_data():
    """Fixture providing synthetic price data suitable for backtesting."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 500 days of price data with mean-reverting spread
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    
    # Generate price1 as a random walk
    random_changes = np.random.normal(0, 1, 500)
    price1_values = 100 + np.cumsum(random_changes)
    
    # Create a time-varying hedge ratio that switches between two values
    time_steps = np.arange(500)
    hedge_ratio = np.zeros(500)
    hedge_ratio[time_steps < 250] = 0.7  # First regime
    hedge_ratio[time_steps >= 250] = 0.9  # Second regime
    
    # Generate mean-reverting spread
    spread_mean = 0
    spread_ar_coef = 0.8  # Strong mean reversion
    spread_vol = 2.0
    
    # AR(1) process for spread
    spread = np.zeros(500)
    for t in range(1, 500):
        spread[t] = spread_mean + spread_ar_coef * (spread[t-1] - spread_mean) + np.random.normal(0, spread_vol)
    
    # Generate price2 based on price1, hedge ratio, and spread
    intercept = 50
    price2_values = intercept + hedge_ratio * price1_values + spread
    
    # Create DataFrame with OHLCV format for backtesting
    df1 = pd.DataFrame({
        'open': price1_values,
        'high': price1_values * 1.005,
        'low': price1_values * 0.995,
        'close': price1_values,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    df2 = pd.DataFrame({
        'open': price2_values,
        'high': price2_values * 1.005,
        'low': price2_values * 0.995,
        'close': price2_values,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    return {
        'price1': df1,
        'price2': df2,
        'true_hedge_ratio': hedge_ratio,
        'regime_change_idx': 250
    }


class TestKalmanFilterZScoreIntegration:
    """Integration tests for Kalman filter with z-score strategy."""
    
    def test_dynamic_hedge_ratio_zscore_strategy(self, synthetic_trading_data):
        """Test that a z-score strategy works properly with Kalman filter for dynamic hedge ratios."""
        price1_df = synthetic_trading_data['price1']
        price2_df = synthetic_trading_data['price2']
        
        price1 = price1_df['close']
        price2 = price2_df['close']
        
        # Create spread analyzer
        spread_analyzer = SpreadAnalyzer()
        
        # Strategy 1: Fixed hedge ratio (OLS regression on the entire dataset)
        X = pd.DataFrame({'const': 1, 'price1': price1})
        y = price2
        from statsmodels.regression.linear_model import OLS
        model = OLS(y, X).fit()
        fixed_intercept, fixed_hedge_ratio = model.params
        
        fixed_spread = price2 - (fixed_intercept + fixed_hedge_ratio * price1)
        fixed_zscore = spread_analyzer.calculate_zscore(fixed_spread, window=20, method='rolling')
        fixed_signals = generate_zscore_signals(fixed_zscore, entry_threshold=2.0, exit_threshold=0.5)
        
        # Run backtest with fixed hedge ratio
        fixed_returns, fixed_positions = run_simple_backtest(
            price1_df, price2_df, fixed_signals, fixed_hedge_ratio, 
            include_costs=False
        )
        fixed_metrics = calculate_performance_metrics(fixed_returns)
        
        # Strategy 2: Dynamic hedge ratio with Kalman filter
        kalman_results = estimate_timevarying_hedge_ratio(price1, price2)
        dynamic_spread = kalman_results['spread']
        dynamic_zscore = spread_analyzer.calculate_zscore(dynamic_spread, window=20, method='rolling')
        dynamic_signals = generate_zscore_signals(dynamic_zscore, entry_threshold=2.0, exit_threshold=0.5)
        
        # Create a Series of time-varying hedge ratios for the backtest
        dynamic_hedge_ratios = kalman_results['hedge_ratio']
        
        # Run backtest with dynamic hedge ratio
        dynamic_returns, dynamic_positions = run_simple_backtest(
            price1_df, price2_df, dynamic_signals, dynamic_hedge_ratios, 
            include_costs=False
        )
        dynamic_metrics = calculate_performance_metrics(dynamic_returns)
        
        # The dynamic hedge ratio should adapt to the regime change, so it should outperform
        # the fixed hedge ratio strategy when there's a significant regime change
        
        # Check that dynamic strategy recognized regime change
        regime_change_idx = synthetic_trading_data['regime_change_idx']
        
        # Calculate average hedge ratio before and after regime change
        hr_before = dynamic_hedge_ratios.iloc[:regime_change_idx].mean()
        hr_after = dynamic_hedge_ratios.iloc[regime_change_idx:].mean()
        
        # Verify the dynamic hedge ratio adapted to the regime change
        assert abs(hr_before - 0.7) < 0.15
        assert abs(hr_after - 0.9) < 0.15
        
        # Compare performance (Sharpe ratio should be better for dynamic strategy)
        # Allow some tolerance since randomness can affect short-term results
        assert dynamic_metrics['sharpe_ratio'] >= fixed_metrics['sharpe_ratio'] * 0.9
        
        # Check for specific improvements after regime change
        # Extract returns after the regime change
        fixed_returns_after = fixed_returns.iloc[regime_change_idx:]
        dynamic_returns_after = dynamic_returns.iloc[regime_change_idx:]
        
        # Calculate metrics for post-regime change period
        fixed_metrics_after = calculate_performance_metrics(fixed_returns_after)
        dynamic_metrics_after = calculate_performance_metrics(dynamic_returns_after)
        
        # Dynamic should definitely outperform in the period after regime change
        assert dynamic_metrics_after['sharpe_ratio'] > fixed_metrics_after['sharpe_ratio']

    def test_kalman_enhanced_spread_integration(self):
        """Test the integration of Kalman filter with enhanced spread calculation methods."""
        np.random.seed(42)
        
        # Create synthetic data with regime changes
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        
        # Generate price1 as a random walk
        random_changes = np.random.normal(0, 1, 500)
        price1_values = 100 + np.cumsum(random_changes)
        
        # Create time-varying hedge ratio with a structural break
        hedge_ratio = np.zeros(500)
        hedge_ratio[:250] = 0.7  # First regime
        hedge_ratio[250:] = 0.9  # Second regime
        
        # Create intercept with a structural break
        intercept = np.zeros(500)
        intercept[:250] = 50
        intercept[250:] = 40
        
        # Create residuals with time-varying volatility
        residual_vol = np.ones(500)
        residual_vol[100:200] = 3.0  # High volatility period
        residual_vol[300:400] = 2.0  # Medium volatility period
        residuals = np.array([np.random.normal(0, vol) for vol in residual_vol])
        
        # Generate price2 based on price1, hedge ratio, and residuals
        price2_values = intercept + hedge_ratio * price1_values + residuals
        
        # Create Series objects
        price1 = pd.Series(price1_values, index=dates)
        price2 = pd.Series(price2_values, index=dates)
        
        # Import SpreadAnalyzer directly for this test
        from src.spread_analytics.spread_analyzer import SpreadAnalyzer
        analyzer = SpreadAnalyzer()
        
        # Method 1: Standard z-score calculation with fixed hedge ratio
        # Estimate fixed hedge ratio using OLS
        from statsmodels.regression.linear_model import OLS
        X = pd.DataFrame({'const': 1, 'price1': price1})
        y = price2
        model = OLS(y, X).fit()
        fixed_intercept, fixed_hedge_ratio = model.params
        
        # Calculate fixed spread and z-score
        fixed_spread = price2 - (fixed_intercept + fixed_hedge_ratio * price1)
        fixed_zscore = analyzer.calculate_zscore(fixed_spread, window=20, method='rolling')
        
        # Method 2: Kalman filter with dynamic hedge ratio
        kalman_results = analyzer.kalman_filter_hedge_ratio(price2, price1)
        kalman_spread = kalman_results['spread']
        kalman_zscore = analyzer.calculate_zscore(kalman_spread, window=20, method='rolling')
        
        # Method 3: Kalman filter with volatility adjustment
        vol_adjusted_spread = analyzer.calculate_volatility_adjusted_spread(
            kalman_spread, None, vol_window=20, vol_method='garch'
        )
        vol_adjusted_zscore = analyzer.calculate_zscore(vol_adjusted_spread, window=20, method='rolling')
        
        # Method 4: Multi-timeframe approach
        multi_tf_results = analyzer.calculate_multitimeframe_spread(
            price2, price1, timeframes=[10, 20, 50], 
            hedge_ratio_method='kalman'
        )
        
        # Check consistency across different methods
        # 1. All methods should detect the regime change
        # For fixed hedge ratio, the z-score should show a structural break at regime change
        regime_change_idx = 250
        pre_change = fixed_zscore.iloc[regime_change_idx-20:regime_change_idx-1]
        post_change = fixed_zscore.iloc[regime_change_idx:regime_change_idx+20]
        
        # The absolute mean of post-change z-scores should be higher (regime change creates inefficiency)
        assert abs(post_change.mean()) > abs(pre_change.mean()) * 1.5
        
        # 2. For Kalman filter with dynamic hedge ratio, the transition should be smoother
        pre_change_kalman = kalman_zscore.iloc[regime_change_idx-20:regime_change_idx-1]
        post_change_kalman = kalman_zscore.iloc[regime_change_idx:regime_change_idx+20]
        
        # The difference should be less pronounced with Kalman (adapts to regime change)
        fixed_diff = abs(post_change.mean()) - abs(pre_change.mean())
        kalman_diff = abs(post_change_kalman.mean()) - abs(pre_change_kalman.mean())
        assert kalman_diff < fixed_diff
        
        # 3. Volatility-adjusted spread should normalize the high volatility period
        high_vol_period = slice(100, 200)
        normal_period = slice(200, 300)
        
        # Regular z-score should show higher std in high vol period vs normal period
        fixed_high_vol_std = fixed_zscore.iloc[high_vol_period].std()
        fixed_normal_std = fixed_zscore.iloc[normal_period].std()
        assert fixed_high_vol_std > fixed_normal_std * 1.3
        
        # Vol-adjusted z-score should have more consistent std across periods
        vol_adj_high_vol_std = vol_adjusted_zscore.iloc[high_vol_period].std()
        vol_adj_normal_std = vol_adjusted_zscore.iloc[normal_period].std()
        
        # The ratio of high/normal should be closer to 1 for vol-adjusted
        fixed_ratio = fixed_high_vol_std / fixed_normal_std
        vol_adj_ratio = vol_adj_high_vol_std / vol_adj_normal_std
        assert abs(vol_adj_ratio - 1.0) < abs(fixed_ratio - 1.0)
        
        # 4. Multi-timeframe approach should provide consistent signals
        assert 'zscore_10' in multi_tf_results
        assert 'zscore_20' in multi_tf_results
        assert 'zscore_50' in multi_tf_results
        
        # Longer timeframes should be less volatile
        assert multi_tf_results['zscore_10'].std() > multi_tf_results['zscore_50'].std()
        
        # Signal consistency calculation should work
        consistency = analyzer.calculate_signal_consistency(multi_tf_results)
        assert 0 <= consistency <= 1


class TestKalmanFilterNonlinearIntegration:
    """Integration tests for nonlinear Kalman filter models."""
    
    def test_threshold_model_integration(self, synthetic_trading_data):
        """Test that the threshold model correctly integrates with the trading strategy framework."""
        from src.cointegration.kalman_filter import estimate_nonlinear_timevarying_hedge_ratio
        
        price1_df = synthetic_trading_data['price1']
        price2_df = synthetic_trading_data['price2']
        
        price1 = price1_df['close']
        price2 = price2_df['close']
        
        # Create spread analyzer
        spread_analyzer = SpreadAnalyzer()
        
        # Use threshold model to capture regime change
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2,
            model_type='threshold'
        )
        
        # Check results have the expected structure
        assert 'effective_hedge_ratio' in results.columns
        assert 'beta_low' in results.columns
        assert 'beta_high' in results.columns
        assert 'spread' in results.columns
        
        # Verify threshold model captures the regime change
        # The last values should be closer to the second regime
        beta_low_final = results['beta_low'].iloc[-20:].mean()
        beta_high_final = results['beta_high'].iloc[-20:].mean()
        
        # One of these should be close to the first regime value (0.7) and one to the second (0.9)
        assert (abs(beta_low_final - 0.7) < 0.15 and abs(beta_high_final - 0.9) < 0.15) or \
               (abs(beta_low_final - 0.9) < 0.15 and abs(beta_high_final - 0.7) < 0.15)
        
        # Calculate z-scores and signals
        threshold_spread = results['spread']
        threshold_zscore = spread_analyzer.calculate_zscore(threshold_spread, window=20, method='rolling')
        threshold_signals = generate_zscore_signals(threshold_zscore, entry_threshold=2.0, exit_threshold=0.5)
        
        # Create a Series of time-varying hedge ratios for the backtest
        threshold_hedge_ratios = results['effective_hedge_ratio']
        
        # Run backtest with dynamic hedge ratio
        threshold_returns, threshold_positions = run_simple_backtest(
            price1_df, price2_df, threshold_signals, threshold_hedge_ratios, 
            include_costs=False
        )
        threshold_metrics = calculate_performance_metrics(threshold_returns)
        
        # Basic check that strategy produces reasonable results
        assert threshold_metrics['sharpe_ratio'] > 0
        assert threshold_metrics['win_rate'] > 0.4


# Add our own implementation of the required functions
def generate_zscore_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate trading signals based on z-score values.
    
    Parameters:
    -----------
    zscore : pd.Series
        Z-score time series
    entry_threshold : float
        Threshold for entry signals
    exit_threshold : float
        Threshold for exit signals
        
    Returns:
    --------
    pd.Series
        Series with signals: 1 (long), -1 (short), 0 (no position)
    """
    signals = pd.Series(0, index=zscore.index)
    
    # Long signals (when z-score is below negative threshold)
    signals[zscore < -entry_threshold] = 1
    
    # Short signals (when z-score is above positive threshold)
    signals[zscore > entry_threshold] = -1
    
    # Exit long positions when z-score crosses back above negative exit threshold
    exit_long = (zscore > -exit_threshold) & (zscore.shift(1) <= -exit_threshold)
    signals[exit_long] = 0
    
    # Exit short positions when z-score crosses back below positive exit threshold
    exit_short = (zscore < exit_threshold) & (zscore.shift(1) >= exit_threshold)
    signals[exit_short] = 0
    
    return signals

def run_simple_backtest(price1_df, price2_df, signals, hedge_ratio, include_costs=True):
    """
    Run a simplified backtest for pairs trading strategy.
    
    Parameters:
    -----------
    price1_df : pd.DataFrame
        Price data for the first asset with columns: open, high, low, close, volume
    price2_df : pd.DataFrame
        Price data for the second asset with columns: open, high, low, close, volume
    signals : pd.Series
        Trading signals: 1 (long), -1 (short), 0 (no position)
    hedge_ratio : float or pd.Series
        Hedge ratio between the two assets (can be time-varying)
    include_costs : bool
        Whether to include transaction costs
        
    Returns:
    --------
    pd.Series, pd.DataFrame
        Returns series and positions dataframe
    """
    # Extract close prices
    price1 = price1_df['close']
    price2 = price2_df['close']
    
    # Ensure all series have the same index
    common_idx = signals.index.intersection(price1.index).intersection(price2.index)
    signals = signals.loc[common_idx]
    price1 = price1.loc[common_idx]
    price2 = price2.loc[common_idx]
    
    # Handle time-varying hedge ratio
    if isinstance(hedge_ratio, pd.Series):
        hedge_ratio = hedge_ratio.loc[common_idx]
    
    # Create positions dataframe
    positions = pd.DataFrame(index=common_idx)
    positions['signal'] = signals
    
    # Calculate number of shares for each asset
    positions['asset1_pos'] = 0.0
    positions['asset2_pos'] = 0.0
    
    # Calculate positions based on signals
    for i in range(1, len(positions)):
        if positions['signal'].iloc[i] == 1:  # Long spread
            positions['asset1_pos'].iloc[i] = 1.0
            positions['asset2_pos'].iloc[i] = -hedge_ratio.iloc[i] if isinstance(hedge_ratio, pd.Series) else -hedge_ratio
        elif positions['signal'].iloc[i] == -1:  # Short spread
            positions['asset1_pos'].iloc[i] = -1.0
            positions['asset2_pos'].iloc[i] = hedge_ratio.iloc[i] if isinstance(hedge_ratio, pd.Series) else hedge_ratio
        else:  # No position
            positions['asset1_pos'].iloc[i] = 0.0
            positions['asset2_pos'].iloc[i] = 0.0
    
    # Calculate daily returns
    positions['asset1_return'] = price1.pct_change()
    positions['asset2_return'] = price2.pct_change()
    
    # Apply position returns
    positions['asset1_strat_return'] = positions['asset1_pos'].shift(1) * positions['asset1_return']
    positions['asset2_strat_return'] = positions['asset2_pos'].shift(1) * positions['asset2_return']
    
    # Calculate strategy returns
    positions['strategy_return'] = positions['asset1_strat_return'] + positions['asset2_strat_return']
    
    # Include transaction costs if requested
    if include_costs:
        # Calculate position changes
        positions['asset1_pos_change'] = positions['asset1_pos'].diff().abs()
        positions['asset2_pos_change'] = positions['asset2_pos'].diff().abs()
        
        # Assume 5 basis points commission cost per transaction
        commission_rate = 0.0005
        positions['commission_cost'] = (
            positions['asset1_pos_change'] * commission_rate + 
            positions['asset2_pos_change'] * commission_rate
        )
        
        # Apply commission costs to returns
        positions['strategy_return'] = positions['strategy_return'] - positions['commission_cost']
    
    return positions['strategy_return'], positions

def calculate_performance_metrics(returns):
    """
    Calculate performance metrics from a returns series.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns series
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Drop NaN values
    returns = returns.dropna()
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    daily_std = returns.std()
    annual_std = daily_std * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std if annual_std != 0 else 0
    
    # Drawdown calculation
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
    
    # Collect metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_std,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    return metrics


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 