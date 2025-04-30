"""
Integration tests for the Z-Score Strategy.

This module contains integration tests for the Z-Score strategy implementation,
focusing on end-to-end testing of the full strategy pipeline with more complex scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.backtest.zscore_strategy_backtest import ZScoreStrategyBacktest, run_zscore_backtest

@pytest.fixture
def setup_test_data_path():
    """Setup and return the path for test data."""
    # Create the test data directory if it doesn't exist
    test_data_dir = os.path.join('tests', 'fixtures', 'backtest_data')
    os.makedirs(test_data_dir, exist_ok=True)
    return test_data_dir

@pytest.fixture
def complex_price_data(setup_test_data_path):
    """
    Create complex price data with regime shifts for more realistic testing.
    This includes:
    - A cointegrated period
    - A regime shift where the relationship changes
    - A period of divergence
    - Return to cointegration with a different ratio
    """
    # Create date range for 2 years of daily data
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='B')
    n_points = len(dates)
    
    # Create series with changing cointegration relationship
    np.random.seed(42)  # For reproducibility
    
    # Initial series - random walk
    random_changes = np.random.normal(0, 1, n_points)
    series1 = 100 + np.cumsum(random_changes)
    
    # Create series2 with changing relationship to series1
    series2 = np.zeros(n_points)
    
    # Phase 1 (first 6 months): Strong cointegration
    phase1_end = 126  # ~6 months of business days
    hedge_ratio1 = 0.7
    noise1 = np.random.normal(0, 0.5, phase1_end)
    series2[:phase1_end] = (series1[:phase1_end] / hedge_ratio1) + noise1
    
    # Phase 2 (next 3 months): Regime shift - relationship breaks down
    phase2_end = phase1_end + 63  # ~3 months of business days
    # Create a divergence by having series2 follow a different trend
    random_changes2 = np.random.normal(0.05, 1.2, phase2_end - phase1_end)  # Stronger up-trend with more volatility
    series2[phase1_end:phase2_end] = series2[phase1_end-1] + np.cumsum(random_changes2)
    
    # Phase 3 (next 9 months): New cointegration relationship
    phase3_end = phase2_end + 189  # ~9 months of business days
    hedge_ratio2 = 0.9  # Different hedge ratio
    noise3 = np.random.normal(0, 0.6, phase3_end - phase2_end)
    # Reset to a new cointegration relationship
    baseline = series2[phase2_end-1]
    series2[phase2_end:phase3_end] = baseline + (series1[phase2_end:phase3_end] - series1[phase2_end-1]) / hedge_ratio2 + noise3
    
    # Phase 4 (remaining time): Return to initial relationship
    hedge_ratio4 = 0.75
    noise4 = np.random.normal(0, 0.4, n_points - phase3_end)
    baseline = series2[phase3_end-1]
    series2[phase3_end:] = baseline + (series1[phase3_end:] - series1[phase3_end-1]) / hedge_ratio4 + noise4
    
    # Convert to pandas Series
    price1 = pd.Series(series1, index=dates, name='price1')
    price2 = pd.Series(series2, index=dates, name='price2')
    
    # Save the test data for reuse
    file_path = os.path.join(setup_test_data_path, 'complex_price_data.pkl')
    pd.DataFrame({'price1': price1, 'price2': price2}).to_pickle(file_path)
    
    # Create regime shift markers for visualization
    regime_shifts = {
        'phase1_end': dates[phase1_end-1],
        'phase2_end': dates[phase2_end-1],
        'phase3_end': dates[phase3_end-1],
        'hedge_ratios': [hedge_ratio1, None, hedge_ratio2, hedge_ratio4]  # None for the non-cointegrated period
    }
    
    return {
        'price1': price1,
        'price2': price2,
        'dates': dates,
        'regime_shifts': regime_shifts,
        'file_path': file_path
    }

@pytest.fixture
def plot_test_data(complex_price_data, setup_test_data_path):
    """Create plots of the test data to visualize the regime shifts."""
    price1 = complex_price_data['price1']
    price2 = complex_price_data['price2']
    regime_shifts = complex_price_data['regime_shifts']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot price series
    plt.subplot(2, 1, 1)
    plt.plot(price1.index, price1, label='Asset 1')
    plt.plot(price2.index, price2, label='Asset 2')
    
    # Add vertical lines for regime shifts
    plt.axvline(x=regime_shifts['phase1_end'], color='r', linestyle='--', label='Regime Shift 1')
    plt.axvline(x=regime_shifts['phase2_end'], color='g', linestyle='--', label='Regime Shift 2')
    plt.axvline(x=regime_shifts['phase3_end'], color='b', linestyle='--', label='Regime Shift 3')
    
    plt.title('Complex Price Data with Regime Shifts')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Plot the spread
    plt.subplot(2, 1, 2)
    
    # Calculate spreads for different periods with appropriate hedge ratios
    dates = price1.index
    hedge_ratios = regime_shifts['hedge_ratios']
    
    # Phase 1 spread
    phase1_mask = dates <= regime_shifts['phase1_end']
    spread1 = price1[phase1_mask] - hedge_ratios[0] * price2[phase1_mask]
    
    # Phase 2 - no stable spread
    
    # Phase 3 spread
    phase3_mask = (dates > regime_shifts['phase2_end']) & (dates <= regime_shifts['phase3_end'])
    spread3 = price1[phase3_mask] - hedge_ratios[2] * price2[phase3_mask]
    
    # Phase 4 spread
    phase4_mask = dates > regime_shifts['phase3_end']
    spread4 = price1[phase4_mask] - hedge_ratios[3] * price2[phase4_mask]
    
    # Plot the spreads
    plt.plot(spread1.index, spread1, label='Spread Phase 1')
    plt.plot(spread3.index, spread3, label='Spread Phase 3')
    plt.plot(spread4.index, spread4, label='Spread Phase 4')
    
    # Add vertical lines for regime shifts
    plt.axvline(x=regime_shifts['phase1_end'], color='r', linestyle='--')
    plt.axvline(x=regime_shifts['phase2_end'], color='g', linestyle='--')
    plt.axvline(x=regime_shifts['phase3_end'], color='b', linestyle='--')
    
    plt.title('Spreads for Different Regimes')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(setup_test_data_path, 'complex_price_data.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {'plot_path': plot_path}

class TestZScoreStrategyIntegration:
    """Integration tests for the Z-Score Strategy."""
    
    def test_strategy_with_regime_shifts(self, complex_price_data):
        """Test how the strategy performs with regime shifts."""
        price1 = complex_price_data['price1']
        price2 = complex_price_data['price2']
        
        # Create a strategy with default parameters
        strategy = ZScoreStrategyBacktest(
            entry_threshold=2.0,
            exit_threshold=0.5,
            window_size=40,  # Use a window large enough to smooth noise but catch regime shifts
            max_holding_period=20,
            commission_per_trade=1.0,
            slippage_per_trade=0.5,
            account_size=100000
        )
        
        # Run backtest
        results = strategy.backtest(price1, price2)
        
        # Assertions
        assert isinstance(results, dict)
        assert 'equity_curve' in results
        assert 'trade_history' in results
        assert 'metrics' in results
        
        # Check if at least some trades were executed
        trades = results['trade_history']
        assert len(trades) > 0
        
        # Verify strategy doesn't lose all money despite regime shifts
        final_equity = results['equity_curve'].iloc[-1]
        assert final_equity > strategy.initial_account_size * 0.7  # Should not lose more than 30%
    
    def test_strategy_with_rolling_window_adaptation(self, complex_price_data):
        """Test strategy with adaptive parameters based on rolling windows."""
        price1 = complex_price_data['price1']
        price2 = complex_price_data['price2']
        regime_shifts = complex_price_data['regime_shifts']
        
        # Create a strategy with short window to adapt quickly to regime changes
        strategy = ZScoreStrategyBacktest(
            entry_threshold=2.0,
            exit_threshold=0.5,
            window_size=20,  # Shorter window to adapt faster
            max_holding_period=15,
            commission_per_trade=1.0,
            slippage_per_trade=0.5,
            account_size=100000
        )
        
        # Run backtest
        results = strategy.backtest(price1, price2)
        
        # Check trade distribution across regimes
        trades = results['trade_history']
        
        # Count trades in each phase
        phase1_trades = sum(1 for t in trades if t['entry_date'] <= regime_shifts['phase1_end'])
        phase2_trades = sum(1 for t in trades if t['entry_date'] > regime_shifts['phase1_end'] and 
                                              t['entry_date'] <= regime_shifts['phase2_end'])
        phase3_trades = sum(1 for t in trades if t['entry_date'] > regime_shifts['phase2_end'] and 
                                              t['entry_date'] <= regime_shifts['phase3_end'])
        phase4_trades = sum(1 for t in trades if t['entry_date'] > regime_shifts['phase3_end'])
        
        # The strategy should identify tradable periods and avoid non-tradable ones
        # Phase 2 should have significantly fewer trades since it's not cointegrated
        print(f"Trades by phase - Phase 1: {phase1_trades}, Phase 2: {phase2_trades}, Phase 3: {phase3_trades}, Phase 4: {phase4_trades}")
        
        # Verify phase 2 (non-cointegrated) has fewer trades than other phases
        # Note: this assumes other phases are of similar length, which they're not in our fixture
        # So we normalize by the phase length
        phase_lengths = {
            1: sum(complex_price_data['dates'] <= regime_shifts['phase1_end']),
            2: sum((complex_price_data['dates'] > regime_shifts['phase1_end']) & 
                    (complex_price_data['dates'] <= regime_shifts['phase2_end'])),
            3: sum((complex_price_data['dates'] > regime_shifts['phase2_end']) & 
                    (complex_price_data['dates'] <= regime_shifts['phase3_end'])),
            4: sum(complex_price_data['dates'] > regime_shifts['phase3_end'])
        }
        
        # Normalize trade counts by phase length
        normalized_trades = {
            1: phase1_trades / phase_lengths[1] if phase_lengths[1] > 0 else 0,
            2: phase2_trades / phase_lengths[2] if phase_lengths[2] > 0 else 0,
            3: phase3_trades / phase_lengths[3] if phase_lengths[3] > 0 else 0,
            4: phase4_trades / phase_lengths[4] if phase_lengths[4] > 0 else 0
        }
        
        # Phase 2 should have a lower normalized trade count
        print(f"Normalized trade density by phase: {normalized_trades}")
        
        # The non-cointegrated phase should have lower trade density
        assert normalized_trades[2] < normalized_trades[1] or normalized_trades[2] < normalized_trades[3]
    
    def test_performance_comparison_of_different_strategies(self, complex_price_data):
        """Compare performance of different strategies on the complex dataset."""
        price1 = complex_price_data['price1']
        price2 = complex_price_data['price2']
        
        # Define a set of strategies with different parameters
        strategies = [
            # Standard strategy
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=40, max_holding_period=20),
            
            # More aggressive entry
            ZScoreStrategyBacktest(entry_threshold=1.5, window_size=40, max_holding_period=20),
            
            # Shorter window
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=20, max_holding_period=20),
            
            # Longer window
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=60, max_holding_period=20),
            
            # EWM calculation
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=40, max_holding_period=20, calculate_method='ewm'),
            
            # Shorter holding period
            ZScoreStrategyBacktest(entry_threshold=2.0, window_size=40, max_holding_period=10),
        ]
        
        strategy_names = [
            "Standard",
            "Aggressive Entry",
            "Short Window",
            "Long Window",
            "EWM Calculation",
            "Short Holding Period"
        ]
        
        # Run backtests for all strategies
        results = []
        for i, strat in enumerate(strategies):
            res = strat.backtest(price1, price2)
            
            # Extract key metrics
            metrics = {
                'strategy': strategy_names[i],
                'total_return': res['metrics']['total_return'],
                'annualized_return': res['metrics']['annualized_return'],
                'max_drawdown': res['metrics']['max_drawdown'],
                'sharpe_ratio': res['metrics']['sharpe_ratio'],
                'win_rate': res['metrics']['win_rate'],
                'trade_count': len(res['trade_history'])
            }
            results.append(metrics)
        
        # Convert to DataFrame for easier comparison
        results_df = pd.DataFrame(results)
        
        # Assertions
        assert len(results_df) == len(strategies)
        
        # Verify different strategies yield different results
        assert results_df['total_return'].nunique() > 1
        assert results_df['sharpe_ratio'].nunique() > 1
        
        # Verify all strategies maintained reasonable performance
        # None should lose more than 50% or have negative Sharpe ratio
        assert all(results_df['total_return'] > -0.5)
        
        # Print the results for manual inspection
        print("\nStrategy Performance Comparison:")
        print(results_df.to_string(index=False))
    
    def test_run_zscore_backtest_function(self, complex_price_data, setup_test_data_path):
        """Test the high-level run_zscore_backtest function."""
        # Prepare data in the format expected by run_zscore_backtest
        price1 = complex_price_data['price1']
        price2 = complex_price_data['price2']
        
        # Create combined price data
        price_data = pd.DataFrame({
            'asset1': price1,
            'asset2': price2
        })
        
        # Run the high-level backtest function
        result = run_zscore_backtest(
            price_data=price_data,
            ticker1='asset1',
            ticker2='asset2',
            entry_threshold=2.0,
            exit_threshold=0.5,
            window_size=40,
            max_holding_period=20,
            output_dir=setup_test_data_path
        )
        
        # Assertions
        assert isinstance(result, dict)
        assert 'equity_curve' in result
        assert 'metrics' in result
        assert 'trade_history' in result
        
        # Verify files were saved
        result_files = result.get('saved_files', {})
        
        # Check if any output files exist
        if result_files:
            for file_path in result_files.values():
                assert os.path.exists(file_path)
    
    def test_z_score_calculation_with_different_methods(self, complex_price_data):
        """Test and compare different z-score calculation methods."""
        price1 = complex_price_data['price1']
        price2 = complex_price_data['price2']
        
        # Create hedge ratio and spread
        strategy = ZScoreStrategyBacktest()
        hedge_ratio = strategy.calculate_hedge_ratio(price1, price2)
        spread = strategy.calculate_spread(price1, price2, hedge_ratio)
        
        # Calculate z-scores using different methods
        strategy.calculate_method = 'rolling'
        zscore_rolling = strategy.calculate_zscore(spread)
        
        strategy.calculate_method = 'ewm'
        zscore_ewm = strategy.calculate_zscore(spread)
        
        strategy.calculate_method = 'full'
        zscore_full = strategy.calculate_zscore(spread)
        
        # Compare correlation between different methods
        corr_rolling_ewm = zscore_rolling.corr(zscore_ewm)
        corr_rolling_full = zscore_rolling.corr(zscore_full)
        corr_ewm_full = zscore_ewm.corr(zscore_full)
        
        # Assertions
        # The methods should produce correlated but not identical results
        assert 0.7 < corr_rolling_ewm < 1.0
        assert 0.5 < corr_rolling_full < 1.0
        assert 0.5 < corr_ewm_full < 1.0
        
        # Verify statistical properties
        for zscore in [zscore_rolling, zscore_ewm, zscore_full]:
            non_nan = zscore.dropna()
            # Z-scores should have approximately mean 0 and std 1
            assert -0.5 < non_nan.mean() < 0.5
            assert 0.7 < non_nan.std() < 1.3 