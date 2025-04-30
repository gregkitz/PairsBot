#!/usr/bin/env python3
"""
Performance tests for trading system components.

This module measures the performance of signal generation, order execution,
position management, and risk calculation operations.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import tracemalloc
from functools import wraps
from contextlib import contextmanager

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

# Try importing trading components
try:
    from src.signal_generation.signal_processor import SignalProcessor
    from src.backtest.backtest_engine import BacktestEngine
    from src.paper_trading.paper_trader import PaperTrader
    from src.risk_management.position_manager import PositionManager
    TRADING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRADING_COMPONENTS_AVAILABLE = False


def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Get stats
        stats = snapshot.statistics('lineno')
        total = sum(stat.size for stat in stats)
        
        # Convert to MB
        total_mb = total / (1024 * 1024)
        
        return result, total_mb
    
    return wrapper


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


def get_data_size():
    """Get the test data size from environment variables."""
    data_size = os.environ.get("PERFORMANCE_DATA_SIZE", "medium")
    
    # Define data sizes
    sizes = {
        "small": {"days": 30, "pairs": 2},
        "medium": {"days": 90, "pairs": 5},
        "large": {"days": 365, "pairs": 10},
        "production": {"days": 730, "pairs": 20}
    }
    
    return sizes.get(data_size, sizes["medium"])


def create_synthetic_trading_data(data_size):
    """
    Create synthetic data for trading system performance tests.
    
    Parameters:
    -----------
    data_size : dict
        Dictionary with data size parameters
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with synthetic price data and pair configurations
    """
    days = data_size["days"]
    num_pairs = data_size["pairs"]
    
    # Create date range
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Create price data for pairs
    pair_data = {}
    pair_configs = []
    
    for pair_idx in range(num_pairs):
        # Generate synthetic price data for two cointegrated assets
        np.random.seed(pair_idx)  # For reproducibility
        
        # Create common trend component
        trend = np.cumsum(np.random.normal(0.0001, 0.001, days))
        
        # Create individual assets with some correlation to the trend
        asset1_prices = 100 + 50 * trend + np.cumsum(np.random.normal(0, 0.01, days))
        
        # Create asset2 that's cointegrated with asset1
        hedge_ratio = 0.7 + 0.1 * pair_idx  # Vary the hedge ratio slightly
        asset2_prices = hedge_ratio * asset1_prices + np.random.normal(0, 1, days)
        
        # Add mean-reverting noise to create spread fluctuations
        for i in range(1, days):
            # Mean-reverting component for realistic pair behavior
            if i > 0:
                spread = asset1_prices[i-1] - hedge_ratio * asset2_prices[i-1]
                # Pull back toward the mean with some factor
                reversion = 0.05 * spread
                asset1_prices[i] -= reversion / 2
                asset2_prices[i] += reversion / (2 * hedge_ratio)
        
        # Create OHLCV data for both assets
        asset1_df = pd.DataFrame({
            'open': asset1_prices * (1 + np.random.normal(0, 0.002, days)),
            'high': asset1_prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
            'low': asset1_prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
            'close': asset1_prices * (1 + np.random.normal(0, 0.002, days)),
            'volume': np.random.normal(1000000, 200000, days).astype(int)
        }, index=date_range)
        
        asset2_df = pd.DataFrame({
            'open': asset2_prices * (1 + np.random.normal(0, 0.002, days)),
            'high': asset2_prices * (1 + np.abs(np.random.normal(0, 0.005, days))),
            'low': asset2_prices * (1 - np.abs(np.random.normal(0, 0.005, days))),
            'close': asset2_prices * (1 + np.random.normal(0, 0.002, days)),
            'volume': np.random.normal(1000000, 200000, days).astype(int)
        }, index=date_range)
        
        # Ensure positive values
        asset1_df = asset1_df.clip(lower=1)
        asset2_df = asset2_df.clip(lower=1)
        asset1_df['volume'] = asset1_df['volume'].clip(lower=0)
        asset2_df['volume'] = asset2_df['volume'].clip(lower=0)
        
        # Store data
        asset1_name = f"ASSET{pair_idx*2+1}"
        asset2_name = f"ASSET{pair_idx*2+2}"
        
        pair_data[asset1_name] = asset1_df
        pair_data[asset2_name] = asset2_df
        
        # Create pair configuration
        pair_configs.append({
            'ticker1': asset1_name,
            'ticker2': asset2_name,
            'hedge_ratio': hedge_ratio,
            'half_life': 10,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'max_holding_period': 20
        })
    
    return {
        'pair_data': pair_data,
        'pair_configs': pair_configs
    }


class MockSignalProcessor:
    """Mock signal processor for testing when actual component is not available."""
    
    def __init__(self, pair_data, pair_configs):
        """
        Initialize the mock signal processor.
        
        Parameters:
        -----------
        pair_data : dict
            Dictionary of DataFrames with price data
        pair_configs : list
            List of pair configurations
        """
        self.pair_data = pair_data
        self.pair_configs = pair_configs
    
    def generate_signals(self, current_date):
        """
        Generate trading signals for all pairs at the current date.
        
        Parameters:
        -----------
        current_date : datetime.datetime
            Current date for signal generation
            
        Returns:
        --------
        dict
            Dictionary of signals for each pair
        """
        signals = {}
        
        for pair_config in self.pair_configs:
            ticker1 = pair_config['ticker1']
            ticker2 = pair_config['ticker2']
            hedge_ratio = pair_config['hedge_ratio']
            
            # Get data up to current date
            asset1_data = self.pair_data[ticker1]
            asset2_data = self.pair_data[ticker2]
            
            # Filter data
            asset1_data = asset1_data[:current_date]
            asset2_data = asset2_data[:current_date]
            
            if len(asset1_data) < 20 or len(asset2_data) < 20:
                # Not enough data
                continue
            
            # Calculate spread
            spread = asset1_data['close'] - hedge_ratio * asset2_data['close']
            
            # Calculate z-score
            window = 20
            spread_mean = spread.rolling(window=window).mean().iloc[-1]
            spread_std = spread.rolling(window=window).std().iloc[-1]
            current_spread = spread.iloc[-1]
            
            if spread_std > 0:
                z_score = (current_spread - spread_mean) / spread_std
            else:
                z_score = 0
            
            # Generate signal
            pair_name = f"{ticker1}_{ticker2}"
            
            if z_score > pair_config['entry_threshold']:
                # Short signal
                signals[pair_name] = {
                    'signal': -1,
                    'z_score': z_score,
                    'date': current_date,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'hedge_ratio': hedge_ratio,
                    'price1': asset1_data['close'].iloc[-1],
                    'price2': asset2_data['close'].iloc[-1]
                }
            elif z_score < -pair_config['entry_threshold']:
                # Long signal
                signals[pair_name] = {
                    'signal': 1,
                    'z_score': z_score,
                    'date': current_date,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'hedge_ratio': hedge_ratio,
                    'price1': asset1_data['close'].iloc[-1],
                    'price2': asset2_data['close'].iloc[-1]
                }
            elif abs(z_score) < pair_config['exit_threshold']:
                # Exit signal
                signals[pair_name] = {
                    'signal': 0,
                    'z_score': z_score,
                    'date': current_date,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'hedge_ratio': hedge_ratio,
                    'price1': asset1_data['close'].iloc[-1],
                    'price2': asset2_data['close'].iloc[-1]
                }
        
        return signals


class MockPositionManager:
    """Mock position manager for testing when actual component is not available."""
    
    def __init__(self):
        """Initialize the mock position manager."""
        self.positions = {}
        self.position_history = []
    
    def execute_signals(self, signals, current_date):
        """
        Execute trading signals.
        
        Parameters:
        -----------
        signals : dict
            Dictionary of signals for each pair
        current_date : datetime.datetime
            Current date for signal execution
            
        Returns:
        --------
        list
            List of executed trades
        """
        executed_trades = []
        
        for pair_name, signal_data in signals.items():
            ticker1 = signal_data['ticker1']
            ticker2 = signal_data['ticker2']
            hedge_ratio = signal_data['hedge_ratio']
            signal = signal_data['signal']
            price1 = signal_data['price1']
            price2 = signal_data['price2']
            
            # Check for existing position
            if pair_name in self.positions:
                # Position exists, check if exit
                position = self.positions[pair_name]
                
                if signal == 0 or (signal * position['direction'] < 0):
                    # Exit position
                    self.position_history.append({
                        'pair': pair_name,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'direction': position['direction'],
                        'entry_price1': position['entry_price1'],
                        'entry_price2': position['entry_price2'],
                        'exit_price1': price1,
                        'exit_price2': price2,
                        'pnl': self._calculate_pnl(position, price1, price2)
                    })
                    
                    executed_trades.append({
                        'pair': pair_name,
                        'action': 'exit',
                        'date': current_date,
                        'price1': price1,
                        'price2': price2
                    })
                    
                    del self.positions[pair_name]
            
            elif signal != 0:
                # No position exists, enter if signal
                self.positions[pair_name] = {
                    'entry_date': current_date,
                    'direction': signal,
                    'entry_price1': price1,
                    'entry_price2': price2,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'hedge_ratio': hedge_ratio
                }
                
                executed_trades.append({
                    'pair': pair_name,
                    'action': 'entry',
                    'direction': signal,
                    'date': current_date,
                    'price1': price1,
                    'price2': price2
                })
        
        return executed_trades
    
    def _calculate_pnl(self, position, exit_price1, exit_price2):
        """Calculate profit/loss for a position."""
        direction = position['direction']
        entry_price1 = position['entry_price1']
        entry_price2 = position['entry_price2']
        hedge_ratio = position['hedge_ratio']
        
        # Calculate PnL based on direction
        if direction == 1:  # Long position
            pnl = (exit_price1 - entry_price1) - hedge_ratio * (exit_price2 - entry_price2)
        else:  # Short position
            pnl = (entry_price1 - exit_price1) - hedge_ratio * (entry_price2 - exit_price2)
        
        return pnl


def test_signal_generation_performance(trading_data, output_dir):
    """
    Test the performance of signal generation.
    
    Parameters:
    -----------
    trading_data : dict
        Dictionary with pair data and configs
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    pair_data = trading_data['pair_data']
    pair_configs = trading_data['pair_configs']
    
    # Create signal processor (real or mock)
    if TRADING_COMPONENTS_AVAILABLE:
        try:
            signal_processor = SignalProcessor(pair_configs)
            for ticker, data in pair_data.items():
                signal_processor.add_price_data(ticker, data)
        except Exception as e:
            print(f"Error creating SignalProcessor: {e}")
            signal_processor = MockSignalProcessor(pair_data, pair_configs)
    else:
        signal_processor = MockSignalProcessor(pair_data, pair_configs)
    
    # Get all dates
    all_dates = next(iter(pair_data.values())).index
    
    # Select test dates (last 30% of available dates)
    start_idx = int(len(all_dates) * 0.7)
    test_dates = all_dates[start_idx:]
    
    # Measure signal generation performance
    generation_times = []
    signal_counts = []
    
    for date in test_dates:
        # Generate signals
        @profile_memory
        def generate_signals():
            return signal_processor.generate_signals(date)
        
        with measure_time() as gen_time:
            signals, memory_usage = generate_signals()
        
        # Record time and count
        generation_times.append(gen_time)
        signal_counts.append(len(signals))
    
    # Calculate statistics
    avg_time = sum(generation_times) / len(generation_times)
    max_time = max(generation_times)
    min_time = min(generation_times)
    avg_signal_count = sum(signal_counts) / len(signal_counts)
    
    # Create chart - signal generation time over dates
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, generation_times, 'o-')
    plt.title("Signal Generation Time")
    plt.xlabel("Date")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_generation_time.png")
    
    # Create chart - signal count over dates
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, signal_counts, 'o-')
    plt.title("Signal Count Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Signals")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_count.png")
    
    # Return results
    return {
        "avg_generation_time": avg_time,
        "max_generation_time": max_time,
        "min_generation_time": min_time,
        "avg_signal_count": avg_signal_count,
        "total_signals": sum(signal_counts),
        "memory_usage": memory_usage
    }


def test_position_management_performance(trading_data, output_dir):
    """
    Test the performance of position management.
    
    Parameters:
    -----------
    trading_data : dict
        Dictionary with pair data and configs
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    pair_data = trading_data['pair_data']
    pair_configs = trading_data['pair_configs']
    
    # Create signal processor
    signal_processor = MockSignalProcessor(pair_data, pair_configs)
    
    # Create position manager (real or mock)
    if TRADING_COMPONENTS_AVAILABLE:
        try:
            position_manager = PositionManager()
            for config in pair_configs:
                position_manager.add_pair(config)
        except Exception as e:
            print(f"Error creating PositionManager: {e}")
            position_manager = MockPositionManager()
    else:
        position_manager = MockPositionManager()
    
    # Get all dates
    all_dates = next(iter(pair_data.values())).index
    
    # Select test dates (last 30% of available dates)
    start_idx = int(len(all_dates) * 0.7)
    test_dates = all_dates[start_idx:]
    
    # Measure position management performance
    execution_times = []
    trade_counts = []
    
    for date in test_dates:
        # Generate signals
        signals = signal_processor.generate_signals(date)
        
        # Execute signals
        @profile_memory
        def execute_signals():
            return position_manager.execute_signals(signals, date)
        
        with measure_time() as exec_time:
            trades, memory_usage = execute_signals()
        
        # Record time and count
        execution_times.append(exec_time)
        trade_counts.append(len(trades))
    
    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)
    min_time = min(execution_times)
    avg_trade_count = sum(trade_counts) / len(trade_counts)
    
    # Create chart - execution time over dates
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, execution_times, 'o-')
    plt.title("Signal Execution Time")
    plt.xlabel("Date")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_execution_time.png")
    
    # Create chart - trade count over dates
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, trade_counts, 'o-')
    plt.title("Trade Count Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "trade_count.png")
    
    # Return results
    return {
        "avg_execution_time": avg_time,
        "max_execution_time": max_time,
        "min_execution_time": min_time,
        "avg_trade_count": avg_trade_count,
        "total_trades": sum(trade_counts),
        "memory_usage": memory_usage
    }


def test_scaling_performance(output_dir):
    """
    Test how trading system performance scales with number of pairs.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Create test data for different numbers of pairs
    num_pairs_list = [2, 5, 10, 20]
    days = 90  # Fixed number of days for all tests
    
    scaling_results = []
    
    for num_pairs in num_pairs_list:
        # Create test data
        data_size = {"days": days, "pairs": num_pairs}
        trading_data = create_synthetic_trading_data(data_size)
        
        # Create signal processor
        signal_processor = MockSignalProcessor(
            trading_data['pair_data'], 
            trading_data['pair_configs']
        )
        
        # Create position manager
        position_manager = MockPositionManager()
        
        # Get test dates (last 20 dates)
        all_dates = next(iter(trading_data['pair_data'].values())).index
        test_dates = all_dates[-20:]
        
        # Measure performance
        signal_times = []
        execution_times = []
        
        for date in test_dates:
            # Generate signals
            start_time = time.time()
            signals = signal_processor.generate_signals(date)
            signal_time = time.time() - start_time
            
            # Execute signals
            start_time = time.time()
            position_manager.execute_signals(signals, date)
            execution_time = time.time() - start_time
            
            signal_times.append(signal_time)
            execution_times.append(execution_time)
        
        # Calculate average times
        avg_signal_time = sum(signal_times) / len(signal_times)
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Record results
        scaling_results.append({
            "num_pairs": num_pairs,
            "avg_signal_time": avg_signal_time,
            "avg_execution_time": avg_execution_time,
            "total_time": avg_signal_time + avg_execution_time
        })
    
    # Create scaling chart - signal time vs number of pairs
    plt.figure(figsize=(10, 6))
    plt.plot(
        [r["num_pairs"] for r in scaling_results],
        [r["avg_signal_time"] for r in scaling_results],
        'o-', label="Signal Generation"
    )
    plt.plot(
        [r["num_pairs"] for r in scaling_results],
        [r["avg_execution_time"] for r in scaling_results],
        's-', label="Signal Execution"
    )
    plt.plot(
        [r["num_pairs"] for r in scaling_results],
        [r["total_time"] for r in scaling_results],
        '^-', label="Total"
    )
    plt.title("Trading System Scaling")
    plt.xlabel("Number of Pairs")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "trading_system_scaling.png")
    
    return {
        "scaling_results": scaling_results
    }


def run_tests(output_dir):
    """
    Run all trading system performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    create_directory(plots_dir)
    
    # Report component availability
    print(f"Trading Components Available: {TRADING_COMPONENTS_AVAILABLE}")
    
    # Results dictionary
    results = {
        "status": "success",
        "metrics": {
            "trading_components_available": TRADING_COMPONENTS_AVAILABLE
        },
        "test_results": {}
    }
    
    # Get data size and create test data
    data_size = get_data_size()
    print(f"Running tests with data size: {data_size}")
    
    print("Creating test data...")
    trading_data = create_synthetic_trading_data(data_size)
    
    # Run tests
    print("Testing signal generation performance...")
    results["test_results"]["signal_generation"] = test_signal_generation_performance(
        trading_data, plots_dir
    )
    
    print("Testing position management performance...")
    results["test_results"]["position_management"] = test_position_management_performance(
        trading_data, plots_dir
    )
    
    print("Testing scaling performance...")
    results["test_results"]["scaling"] = test_scaling_performance(plots_dir)
    
    # Save detailed results
    with open(output_dir / "trading_system_performance.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    # Run tests with default settings
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"performance_results/trading_system_{timestamp}")
    create_directory(output_dir)
    
    results = run_tests(output_dir)
    
    print(f"Tests completed. Results saved to {output_dir}") 