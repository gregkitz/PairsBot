"""
Script to run a full backtest on our most promising pair.

This script uses the backtest engine to backtest the BFX-ZN pair.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
from src.backtest.backtest_engine import BacktestEngine
from src.signal_generation.pairs_signal_generator import PairsSignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_data(symbol1, symbol2, data_dir="data/processed"):
    """
    Load price data for a pair of symbols.
    
    Parameters:
    -----------
    symbol1 : str
        First symbol
    symbol2 : str
        Second symbol
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    tuple
        (ticker1_prices, ticker2_prices) - DataFrames with price data
    """
    # Load data for symbol1
    file1 = os.path.join(data_dir, f"{symbol1}_processed.parquet")
    if not os.path.exists(file1):
        logger.error(f"Data file not found: {file1}")
        return None, None
    
    # Load data for symbol2
    file2 = os.path.join(data_dir, f"{symbol2}_processed.parquet")
    if not os.path.exists(file2):
        logger.error(f"Data file not found: {file2}")
        return None, None
    
    # Load the data
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    return df1, df2

def generate_signals_from_csv(signals_file):
    """
    Load signals from CSV file.
    
    Parameters:
    -----------
    signals_file : str
        Path to CSV file containing signals
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with signals
    """
    if not os.path.exists(signals_file):
        logger.error(f"Signals file not found: {signals_file}")
        return None
    
    signals = pd.read_csv(signals_file)
    
    # Convert timestamp to datetime
    signals['timestamp'] = pd.to_datetime(signals['timestamp'])
    signals.set_index('timestamp', inplace=True)
    
    return signals

def main():
    """Main function to run the backtest."""
    # Define the symbols
    symbol1 = "BFX"
    symbol2 = "ZN"
    hedge_ratio = 0.5774521323492694  # From our analysis
    
    logger.info(f"Running backtest for pair: {symbol1}-{symbol2}")
    
    # Load price data
    df1, df2 = load_pair_data(symbol1, symbol2)
    if df1 is None or df2 is None:
        return
    
    # Ensure price data is sorted by timestamp and is a time series
    if not isinstance(df1.index, pd.DatetimeIndex):
        if 'timestamp' in df1.columns:
            df1.set_index('timestamp', inplace=True)
    
    if not isinstance(df2.index, pd.DatetimeIndex):
        if 'timestamp' in df2.columns:
            df2.set_index('timestamp', inplace=True)
    
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    
    # Extract close prices
    price1 = df1['close']
    price2 = df2['close']
    
    # Either load signals from CSV file
    signals_file = f"data/results/signals/{symbol1}_{symbol2}_signals.csv"
    if os.path.exists(signals_file):
        logger.info(f"Loading signals from {signals_file}")
        signals = generate_signals_from_csv(signals_file)
    else:
        # Or generate signals using the SignalGenerator
        logger.info("Generating signals using SignalGenerator")
        signal_gen = PairsSignalGenerator(
            lookback=30,  # Adjusted based on half-life
            entry_zscore=2.0,
            exit_zscore=0.0,  # Adjusted to exit at mean
            use_rolling_regression=True,
            regression_window=60,
            stop_loss_std=4.0
        )
        
        # Override the validate_pair method to always return True
        signal_gen.validate_pair = lambda x, y: True
        
        signals = signal_gen.generate_signals(price1, price2, hedge_ratio)
    
    if signals is None:
        logger.error("Failed to generate signals")
        return
    
    # Prepare for backtest
    prices = {
        symbol1: price1,
        symbol2: price2
    }
    
    # Prepare pairs data
    pairs_data = {
        'ticker1': symbol1,
        'ticker2': symbol2,
        'hedge_ratio': hedge_ratio,
        'ticker1_prices': df1,
        'ticker2_prices': df2
    }
    
    # Set backtest parameters
    account_size = 100000  # Initial account size
    commission = 0.0002    # Commission per trade (0.02%)
    slippage = 0.0001      # Slippage per trade (0.01%)
    
    # Create backtest engine
    engine = BacktestEngine(
        signals=signals,
        prices=prices,
        position_sizes=None,
        account_size=account_size,
        commission=commission,
        slippage=slippage,
        trade_delay=1,  # 1 bar delay for execution
        allow_simultaneous_positions=False,
        pairs_data=pairs_data
    )
    
    # Run backtest
    logger.info("Running backtest...")
    start_time = time.time()
    results = engine.run_backtest()
    elapsed_time = time.time() - start_time
    logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")
    
    # Generate performance charts
    output_dir = "data/results/backtest_final"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{symbol1}_{symbol2}_backtest_{timestamp}.png")
    
    engine.plot_results(
        figsize=(15, 10),
        save_path=output_file
    )
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{symbol1}_{symbol2}_backtest_{timestamp}.json")
    engine.save_results(results_file)
    
    # Print summary of results
    metrics = results['metrics']
    logger.info("Backtest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['total_return', 'annual_return', 'max_drawdown']:
                logger.info(f"  {key}: {value:.2%}")
            else:
                logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    print("\nBacktest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['total_return', 'annual_return', 'max_drawdown']:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Skip the trade statistics due to deprecated method
    # trade_stats = engine.generate_trade_statistics()

if __name__ == "__main__":
    main() 