"""
Script to run a backtest on the most promising cointegrated pair.

This script loads the analyzed pairs data, selects the best pair based on half-life,
and runs a backtest using the existing backtest infrastructure.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import logging
import time
from src.backtest.backtest_engine import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_analysis():
    """
    Load the pair analysis results.
    
    Returns:
    --------
    pd.DataFrame
        Dataframe containing pair analysis results
    """
    analysis_file = "data/results/pairs_analysis.csv"
    if not os.path.exists(analysis_file):
        logger.error(f"Analysis file not found: {analysis_file}")
        return None
    
    return pd.read_csv(analysis_file)

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

def generate_signals(df1, df2, hedge_ratio, lookback=20, entry_zscore=2.0, exit_zscore=0.5):
    """
    Generate trading signals based on z-score of the spread.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        DataFrame with price data for the first symbol
    df2 : pd.DataFrame
        DataFrame with price data for the second symbol
    hedge_ratio : float
        Hedge ratio between the two symbols
    lookback : int
        Number of periods to use for z-score calculation
    entry_zscore : float
        Z-score threshold for entry signals
    exit_zscore : float
        Z-score threshold for exit signals
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trading signals
    """
    # Ensure both dataframes have the same index
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # Calculate spread
    log_price1 = np.log(df1['close'])
    log_price2 = np.log(df2['close'])
    spread = log_price1 - hedge_ratio * log_price2
    
    # Calculate z-score
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()
    z_score = (spread - spread_mean) / spread_std
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=spread.index)
    signals['spread'] = spread
    signals['z_score'] = z_score
    signals['position'] = 0  # 1 for long, -1 for short, 0 for flat
    
    # Generate signals based on z-score
    for i in range(lookback, len(signals)):
        if signals['position'].iloc[i-1] == 0:  # If no position
            if signals['z_score'].iloc[i] < -entry_zscore:
                signals.loc[signals.index[i], 'position'] = 1  # Go long
            elif signals['z_score'].iloc[i] > entry_zscore:
                signals.loc[signals.index[i], 'position'] = -1  # Go short
        elif signals['position'].iloc[i-1] == 1:  # If long
            if signals['z_score'].iloc[i] > -exit_zscore:
                signals.loc[signals.index[i], 'position'] = 0  # Exit long
        elif signals['position'].iloc[i-1] == -1:  # If short
            if signals['z_score'].iloc[i] < exit_zscore:
                signals.loc[signals.index[i], 'position'] = 0  # Exit short
        else:
            signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]  # Maintain position
    
    return signals

def run_backtest(symbol1, symbol2, hedge_ratio, df1, df2, signals, 
                 account_size=100000, commission=0.0002, slippage=0.0001):
    """
    Run a backtest for a pair of symbols.
    
    Parameters:
    -----------
    symbol1 : str
        First symbol
    symbol2 : str
        Second symbol
    hedge_ratio : float
        Hedge ratio between the two symbols
    df1 : pd.DataFrame
        DataFrame with price data for the first symbol
    df2 : pd.DataFrame
        DataFrame with price data for the second symbol
    signals : pd.DataFrame
        DataFrame with trading signals
    account_size : float
        Initial account size
    commission : float
        Commission per trade as percentage
    slippage : float
        Slippage per trade as percentage
        
    Returns:
    --------
    dict
        Dictionary with backtest results
    """
    # Prepare prices dictionary
    prices = {
        symbol1: df1['close'],
        symbol2: df2['close']
    }
    
    # Prepare pairs data
    pairs_data = {
        'ticker1': symbol1,
        'ticker2': symbol2,
        'hedge_ratio': hedge_ratio,
        'ticker1_prices': df1,
        'ticker2_prices': df2
    }
    
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
    results = engine.run_backtest()
    
    # Generate performance charts
    output_dir = "data/results/backtest"
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
    
    return results

def main():
    """Main function to run the backtest."""
    # Load pair analysis results
    analysis_df = load_pair_analysis()
    if analysis_df is None:
        return
    
    # Sort by half-life and select the best pair
    analysis_df = analysis_df.sort_values('half_life')
    best_pair = analysis_df.iloc[0]
    
    symbol1 = best_pair['symbol1']
    symbol2 = best_pair['symbol2']
    hedge_ratio = best_pair['hedge_ratio']
    
    logger.info(f"Selected pair: {symbol1}-{symbol2} with hedge ratio {hedge_ratio:.4f}")
    
    # Load price data
    df1, df2 = load_pair_data(symbol1, symbol2)
    if df1 is None or df2 is None:
        return
    
    # Generate signals
    signals = generate_signals(df1, df2, hedge_ratio)
    
    # Run backtest
    results = run_backtest(symbol1, symbol2, hedge_ratio, df1, df2, signals)
    
    # Print summary of results
    metrics = results.get('metrics', {})
    
    # Print available metrics
    logger.info("Backtest Results:")
    
    if 'total_return' in metrics:
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
    
    if 'cagr' in metrics:  # Check for CAGR instead of annualized_return
        logger.info(f"Annualized Return: {metrics['cagr']:.2%}")
    elif 'annualized_return' in metrics:
        logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    
    if 'sharpe_ratio' in metrics:
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in metrics:
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    if 'win_rate' in metrics:
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    
    if 'profit_factor' in metrics:
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    if 'total_trades' in metrics:
        logger.info(f"Total Trades: {metrics['total_trades']}")
    
    # Print the full metrics dictionary for debugging
    logger.info(f"All available metrics: {metrics}")
    
    print("\nBacktest Results:")
    if 'total_return' in metrics:
        print(f"Total Return: {metrics['total_return']:.2%}")
    
    if 'cagr' in metrics:
        print(f"Annualized Return: {metrics['cagr']:.2%}")
    elif 'annualized_return' in metrics:
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    
    if 'sharpe_ratio' in metrics:
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    if 'max_drawdown' in metrics:
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    if 'win_rate' in metrics:
        print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    if 'profit_factor' in metrics:
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    
    if 'total_trades' in metrics:
        print(f"Total Trades: {metrics['total_trades']}")

if __name__ == "__main__":
    main() 