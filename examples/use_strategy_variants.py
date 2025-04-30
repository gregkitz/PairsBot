#!/usr/bin/env python
"""
Example script demonstrating how to use the different strategy variants.

This script shows how to initialize and use the various strategy variants,
including the base pairs trading strategy, time-series based strategy,
and machine learning signal generation strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add project root to path to ensure imports work from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.pairs_trading_strategy import PairsTradingStrategy
from src.strategy_variants import TimeSeriesStrategy, MLSignalStrategy, create_strategy
from src.backtest_engine import BacktestEngine
from src.visualization import plot_trades, plot_equity_curve, plot_spread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(file_path):
    """Load configuration from a JSON file."""
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def run_backtest(strategy, data, start_date=None, end_date=None, initial_capital=100000):
    """Run a backtest using the specified strategy and data."""
    # Create backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=initial_capital
    )
    
    # Run backtest
    results = engine.run(start_date=start_date, end_date=end_date)
    
    # Return results
    return results

def plot_strategy_comparison(results_dict, title='Strategy Comparison'):
    """Plot equity curves for multiple strategies for comparison."""
    plt.figure(figsize=(12, 6))
    
    for name, results in results_dict.items():
        equity_curve = results['equity_curve']
        plt.plot(equity_curve.index, equity_curve['equity'], label=name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()

def compare_strategies(data, configs, start_date=None, end_date=None, initial_capital=100000):
    """Compare multiple trading strategies using the same data."""
    results = {}
    
    for name, config in configs.items():
        logger.info(f"Running backtest for {name}...")
        
        # Create strategy
        strategy_type = config.get('type', 'pairs')
        strategy = create_strategy(strategy_type, config)
        
        # Run backtest
        result = run_backtest(strategy, data, start_date, end_date, initial_capital)
        
        # Store results
        results[name] = result
        
        # Print performance metrics
        metrics = result['metrics']
        logger.info(f"Results for {name}:")
        logger.info(f"  Total Return: {metrics['total_return']:.2%}")
        logger.info(f"  Annual Return: {metrics['annual_return']:.2%}")
        logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
    
    return results

def main():
    """Main function to run the example."""
    # Load configuration files
    pairs_config = load_config('config/strategies/pairs_strategy.json')
    time_series_config = load_config('config/strategies/time_series_pairs.json')
    ml_signals_config = load_config('config/strategies/ml_signals_pairs.json')
    
    # Load historical data
    data_processor = DataProcessor()
    
    # Get symbols from config
    all_symbols = []
    for config in [pairs_config, time_series_config, ml_signals_config]:
        for pair in config.get('pairs', []):
            all_symbols.extend([pair['leg1'], pair['leg2']])
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    
    # Load data for all symbols
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    data = {}
    
    for symbol in all_symbols:
        try:
            # Try to load from file first
            symbol_data = data_processor.load_data(symbol, start_date, end_date)
            
            if symbol_data is None or len(symbol_data) == 0:
                # If no data in file, fetch from external source
                symbol_data = data_processor.fetch_data(symbol, start_date, end_date)
            
            data[symbol] = symbol_data
            logger.info(f"Loaded data for {symbol}: {len(symbol_data)} rows")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    # Check if we have data for at least one pair
    if len(data) < 2:
        logger.error("Not enough data to proceed. Need at least two symbols.")
        return
    
    # Define the configs for the comparison
    configs = {
        'Base Pairs Strategy': pairs_config,
        'Time Series Strategy': time_series_config,
        'ML Signal Strategy': ml_signals_config
    }
    
    # Compare strategies
    backtest_start = '2022-01-01'
    backtest_end = '2023-01-01'
    
    results = compare_strategies(
        data=data,
        configs=configs,
        start_date=backtest_start,
        end_date=backtest_end,
        initial_capital=100000
    )
    
    # Plot comparison
    plot_strategy_comparison(results, title='Strategy Variant Comparison')
    
    # Plot detailed results for each strategy
    for name, result in results.items():
        # Plot trades
        plot_trades(
            equity_curve=result['equity_curve'],
            trades=result['trades'],
            title=f'{name} - Trades'
        )
        
        # Plot equity curve
        plot_equity_curve(
            equity_curve=result['equity_curve'],
            title=f'{name} - Equity Curve'
        )
        
        # Plot pair spread for the first pair if it exists
        if result['pair_analyses'] and len(result['pair_analyses']) > 0:
            first_pair = list(result['pair_analyses'].keys())[0]
            pair_analysis = result['pair_analyses'][first_pair]
            
            if 'spread_series' in pair_analysis and 'z_score_series' in pair_analysis:
                plot_spread(
                    spread_series=pair_analysis['spread_series'],
                    z_score_series=pair_analysis['z_score_series'],
                    title=f'{name} - {first_pair} Spread'
                )

if __name__ == '__main__':
    main() 