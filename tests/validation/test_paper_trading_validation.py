"""
Paper Trading Validation Framework for MVTS (Minimal Viable Trading Strategy).

This module provides test metrics, scenarios, and fixtures for validating
the performance of a minimal viable trading strategy in paper trading.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import required modules
from src.paper_trading.paper_trader import PaperTrader
from src.risk_management.basic_position_sizer import BasicPositionSizer
from src.risk_management.basic_risk_controls import BasicRiskControls
from src.signals.basic_zscore_strategy import BasicZScoreStrategy

# Constants for testing
TEST_PAIR = "GC_SI"  # Gold-Silver pair
TEST_TIMEFRAME = "1h"
INITIAL_CAPITAL = 100000.0


class PaperTradingMetrics:
    """
    Metrics calculator for paper trading validation.
    
    This class calculates various performance metrics to evaluate
    the effectiveness of a trading strategy during paper trading.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """Calculate Sharpe ratio from a series of returns."""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.0, target_return=0.0):
        """Calculate Sortino ratio focusing on downside deviation."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < target_return]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_deviation == 0:
            return np.nan
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown from equity curve."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve / peak) - 1.0
        return drawdown.min()
    
    @staticmethod
    def calculate_win_rate(trades):
        """Calculate win rate from a list of trades."""
        if not trades:
            return 0.0
            
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades):
        """Calculate profit factor (gross profit / gross loss)."""
        if not trades:
            return 0.0
            
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    @staticmethod
    def calculate_average_trade(trades):
        """Calculate average trade P&L."""
        if not trades:
            return 0.0
            
        return sum(trade['pnl'] for trade in trades) / len(trades)
    
    @staticmethod
    def calculate_average_win(trades):
        """Calculate average winning trade."""
        winning_trades = [trade for trade in trades if trade['pnl'] > 0]
        
        if not winning_trades:
            return 0.0
            
        return sum(trade['pnl'] for trade in winning_trades) / len(winning_trades)
    
    @staticmethod
    def calculate_average_loss(trades):
        """Calculate average losing trade."""
        losing_trades = [trade for trade in trades if trade['pnl'] < 0]
        
        if not losing_trades:
            return 0.0
            
        return sum(trade['pnl'] for trade in losing_trades) / len(losing_trades)
    
    @staticmethod
    def calculate_expectancy(trades):
        """Calculate trade expectancy (win rate * avg win - loss rate * avg loss)."""
        if not trades:
            return 0.0
            
        win_rate = PaperTradingMetrics.calculate_win_rate(trades)
        loss_rate = 1 - win_rate
        
        avg_win = PaperTradingMetrics.calculate_average_win(trades)
        avg_loss = abs(PaperTradingMetrics.calculate_average_loss(trades))
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    @staticmethod
    def calculate_recovery_factor(equity_curve, trades):
        """Calculate recovery factor (total return / max drawdown)."""
        if not trades:
            return 0.0
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        max_drawdown = abs(PaperTradingMetrics.calculate_max_drawdown(equity_curve))
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
            
        return total_return / max_drawdown
    
    @staticmethod
    def calculate_calmar_ratio(equity_curve, period=252):
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(equity_curve) < 2:
            return 0.0
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        trading_days = len(equity_curve)
        annualized_return = (1 + total_return) ** (period / trading_days) - 1
        
        max_drawdown = abs(PaperTradingMetrics.calculate_max_drawdown(equity_curve))
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
            
        return annualized_return / max_drawdown
    
    @staticmethod
    def calculate_daily_stats(equity_curve):
        """Calculate daily return statistics."""
        daily_returns = equity_curve.pct_change().dropna()
        
        return {
            'mean': daily_returns.mean(),
            'std': daily_returns.std(),
            'min': daily_returns.min(),
            'max': daily_returns.max(),
            'positive_days': len(daily_returns[daily_returns > 0]),
            'negative_days': len(daily_returns[daily_returns < 0]),
        }
    
    @staticmethod
    def evaluate_strategy(equity_curve, trades):
        """Evaluate a trading strategy using multiple metrics."""
        if len(equity_curve) < 2 or not trades:
            return {
                'error': 'Insufficient data for evaluation'
            }
            
        # Convert equity curve to pandas Series if it's not already
        if not isinstance(equity_curve, pd.Series):
            equity_curve = pd.Series(equity_curve)
        
        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()
        
        return {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            'annualized_return': (1 + ((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1)) ** (252 / len(equity_curve)) - 1,
            'sharpe_ratio': PaperTradingMetrics.calculate_sharpe_ratio(daily_returns),
            'sortino_ratio': PaperTradingMetrics.calculate_sortino_ratio(daily_returns),
            'max_drawdown': PaperTradingMetrics.calculate_max_drawdown(equity_curve),
            'win_rate': PaperTradingMetrics.calculate_win_rate(trades),
            'profit_factor': PaperTradingMetrics.calculate_profit_factor(trades),
            'average_trade': PaperTradingMetrics.calculate_average_trade(trades),
            'expectancy': PaperTradingMetrics.calculate_expectancy(trades),
            'recovery_factor': PaperTradingMetrics.calculate_recovery_factor(equity_curve, trades),
            'calmar_ratio': PaperTradingMetrics.calculate_calmar_ratio(equity_curve),
            'trade_count': len(trades),
            'daily_stats': PaperTradingMetrics.calculate_daily_stats(equity_curve)
        }


@pytest.fixture
def paper_trader_fixtures(tmpdir):
    """Create a set of fixtures for paper trading validation."""
    # Create test directories
    data_dir = os.path.join(tmpdir, "data")
    output_dir = os.path.join(tmpdir, "output")
    logs_dir = os.path.join(output_dir, "logs")
    results_dir = os.path.join(output_dir, "results")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create test data
    start_date = datetime.now() - timedelta(days=30)
    
    # Generate sample price data for testing
    dates = pd.date_range(start=start_date, periods=30*8, freq='1H')
    
    # Create test price data for leg1 (Gold)
    gc_prices = pd.Series(index=dates, data=np.linspace(1800, 1900, len(dates)) + np.random.normal(0, 20, len(dates)))
    
    # Create test price data for leg2 (Silver) with cointegration relationship
    si_prices = gc_prices / 80 + np.random.normal(0, 0.5, len(dates))
    
    # Save test data
    gc_df = pd.DataFrame({'timestamp': dates, 'close': gc_prices})
    si_df = pd.DataFrame({'timestamp': dates, 'close': si_prices})
    
    gc_df.to_csv(os.path.join(data_dir, 'GC_1h.csv'), index=False)
    si_df.to_csv(os.path.join(data_dir, 'SI_1h.csv'), index=False)
    
    # Create test configuration
    config = {
        'initial_capital': INITIAL_CAPITAL,
        'commission_model': 'flat',
        'commission_amount': 2.0,
        'slippage_model': 'fixed',
        'slippage_amount': 0.0001,
        'data_directory': data_dir,
        'output_directory': output_dir,
        'risk_free_rate': 0.02 / 252,  # Daily risk-free rate (2% annual)
        'pairs': [
            {
                'pair_id': TEST_PAIR,
                'leg1': 'GC',
                'leg2': 'SI',
                'timeframe': TEST_TIMEFRAME,
                'z_entry': 2.0,
                'z_exit': 0.5,
                'stop_loss_pct': 0.02,
                'max_holding_period': 5,  # days
                'position_size_pct': 0.02  # 2% of capital per trade
            }
        ]
    }
    
    # Save config
    with open(os.path.join(data_dir, 'test_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    return {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'config': config,
        'gc_prices': gc_df,
        'si_prices': si_df,
        'dates': dates
    }


class TestPaperTradingValidation:
    """Test class for paper trading validation framework."""
    
    def test_framework_initialization(self, paper_trader_fixtures):
        """Test that the validation framework is properly initialized."""
        # Check that fixtures are properly created
        assert os.path.exists(paper_trader_fixtures['data_dir'])
        assert os.path.exists(paper_trader_fixtures['output_dir'])
        assert os.path.exists(os.path.join(paper_trader_fixtures['output_dir'], 'logs'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['output_dir'], 'results'))
        
        # Check that test data exists
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'GC_1h.csv'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'SI_1h.csv'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'test_config.json'))
        
        # Check data shapes
        assert len(paper_trader_fixtures['gc_prices']) == len(paper_trader_fixtures['dates'])
        assert len(paper_trader_fixtures['si_prices']) == len(paper_trader_fixtures['dates'])
    
    def test_metrics_calculations(self):
        """Test that the performance metrics are calculated correctly."""
        # Create sample equity curve and trades
        equity_curve = pd.Series([
            100000, 101000, 102000, 101500, 102500, 103500, 
            103000, 102000, 103000, 104000, 103500, 104500
        ])
        
        trades = [
            {'entry_price': 100, 'exit_price': 110, 'quantity': 10, 'pnl': 100},
            {'entry_price': 105, 'exit_price': 95, 'quantity': 10, 'pnl': -100},
            {'entry_price': 98, 'exit_price': 105, 'quantity': 20, 'pnl': 140},
            {'entry_price': 110, 'exit_price': 105, 'quantity': 10, 'pnl': -50}
        ]
        
        # Calculate metrics
        metrics = PaperTradingMetrics.evaluate_strategy(equity_curve, trades)
        
        # Verify basic metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        
        # Check specific metrics
        assert metrics['win_rate'] == 0.5  # 2 winning trades out of 4
        assert metrics['trade_count'] == 4
        assert metrics['total_return'] > 0  # Positive total return
        
        # Check that max drawdown calculation is working
        assert metrics['max_drawdown'] < 0  # Should be negative
    
    def test_validation_scenario_generation(self, paper_trader_fixtures):
        """Test generation of validation scenarios for paper trading."""
        # We'll test different market conditions by modifying our test data
        
        # Original data - trending market
        gc_trending = paper_trader_fixtures['gc_prices'].copy()
        si_trending = paper_trader_fixtures['si_prices'].copy()
        
        # Create mean-reverting market
        dates = paper_trader_fixtures['dates']
        gc_mean_reverting = pd.DataFrame({
            'timestamp': dates,
            'close': 1850 + np.sin(np.linspace(0, 6*np.pi, len(dates))) * 50 + np.random.normal(0, 10, len(dates))
        })
        
        si_mean_reverting = pd.DataFrame({
            'timestamp': dates,
            'close': gc_mean_reverting['close'] / 80 + np.random.normal(0, 0.3, len(dates))
        })
        
        # Create volatile market
        gc_volatile = pd.DataFrame({
            'timestamp': dates,
            'close': gc_trending['close'].values + np.random.normal(0, 40, len(dates))
        })
        
        si_volatile = pd.DataFrame({
            'timestamp': dates,
            'close': si_trending['close'].values + np.random.normal(0, 1, len(dates))
        })
        
        # Save test scenario data
        gc_mean_reverting.to_csv(os.path.join(paper_trader_fixtures['data_dir'], 'GC_mean_reverting_1h.csv'), index=False)
        si_mean_reverting.to_csv(os.path.join(paper_trader_fixtures['data_dir'], 'SI_mean_reverting_1h.csv'), index=False)
        gc_volatile.to_csv(os.path.join(paper_trader_fixtures['data_dir'], 'GC_volatile_1h.csv'), index=False)
        si_volatile.to_csv(os.path.join(paper_trader_fixtures['data_dir'], 'SI_volatile_1h.csv'), index=False)
        
        # Verify that scenario data was created
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'GC_mean_reverting_1h.csv'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'SI_mean_reverting_1h.csv'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'GC_volatile_1h.csv'))
        assert os.path.exists(os.path.join(paper_trader_fixtures['data_dir'], 'SI_volatile_1h.csv'))
        
        # Ensure the different scenarios have different characteristics
        gc_trending_std = gc_trending['close'].std()
        gc_volatile_std = gc_volatile['close'].std()
        
        assert gc_volatile_std > gc_trending_std  # Volatile should have higher standard deviation
    
    def test_strategy_benchmark_comparison(self, paper_trader_fixtures):
        """Test comparison between strategy and benchmark."""
        # Create sample equity curves
        trading_days = 30
        dates = pd.date_range(start=datetime.now() - timedelta(days=trading_days), periods=trading_days)
        
        # Strategy equity curve with some volatility but overall positive trend
        strategy_equity = pd.Series(index=dates, data=np.linspace(100000, 110000, trading_days) + np.random.normal(0, 500, trading_days))
        
        # Benchmark (e.g., S&P 500) with lower returns but also lower volatility
        benchmark_equity = pd.Series(index=dates, data=np.linspace(100000, 105000, trading_days) + np.random.normal(0, 300, trading_days))
        
        # Create sample trades
        strategy_trades = [
            {'entry_price': 100, 'exit_price': 105, 'quantity': 100, 'pnl': 500, 'entry_time': dates[1], 'exit_time': dates[5]},
            {'entry_price': 107, 'exit_price': 104, 'quantity': 100, 'pnl': -300, 'entry_time': dates[6], 'exit_time': dates[10]},
            {'entry_price': 103, 'exit_price': 110, 'quantity': 100, 'pnl': 700, 'entry_time': dates[11], 'exit_time': dates[15]},
            {'entry_price': 111, 'exit_price': 114, 'quantity': 100, 'pnl': 300, 'entry_time': dates[16], 'exit_time': dates[20]},
            {'entry_price': 115, 'exit_price': 113, 'quantity': 100, 'pnl': -200, 'entry_time': dates[21], 'exit_time': dates[25]}
        ]
        
        # Calculate metrics for strategy and benchmark
        strategy_metrics = PaperTradingMetrics.evaluate_strategy(strategy_equity, strategy_trades)
        
        # Create dummy trades for benchmark to allow metric calculation
        benchmark_trades = [{'pnl': benchmark_equity.iloc[-1] - benchmark_equity.iloc[0]}]
        benchmark_metrics = PaperTradingMetrics.evaluate_strategy(benchmark_equity, benchmark_trades)
        
        # Compare performance - strategy should outperform benchmark
        assert strategy_metrics['total_return'] > benchmark_metrics['total_return']
        
        # Save comparison results
        comparison = {
            'strategy': strategy_metrics,
            'benchmark': benchmark_metrics,
            'outperformance': {
                'total_return': strategy_metrics['total_return'] - benchmark_metrics['total_return'],
                'sharpe_ratio': strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
                'max_drawdown': strategy_metrics['max_drawdown'] - benchmark_metrics['max_drawdown'],
                'calmar_ratio': strategy_metrics['calmar_ratio'] - benchmark_metrics['calmar_ratio']
            }
        }
        
        # Save comparison to file
        comparison_file = os.path.join(paper_trader_fixtures['output_dir'], 'results', 'benchmark_comparison.json')
        with open(comparison_file, 'w') as f:
            # Handle numpy types for JSON serialization
            for metric_type in comparison:
                if isinstance(comparison[metric_type], dict):
                    for key, value in comparison[metric_type].items():
                        if isinstance(value, np.number):
                            comparison[metric_type][key] = float(value)
            
            json.dump(comparison, f, indent=4)
        
        assert os.path.exists(comparison_file)


# If run directly, execute tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 