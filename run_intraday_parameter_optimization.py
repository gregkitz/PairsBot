#!/usr/bin/env python3
"""
Script to optimize parameters for intraday trading across different market regimes.

This script uses the IntradayParameterOptimizer to find optimal parameter sets
for each detected market regime and creates an adaptive parameter configuration.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import argparse

from src.optimization.intraday_parameter_optimizer import IntradayParameterOptimizer
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_price_data(symbols, start_date, end_date, data_dir="data/processed"):
    """
    Load price data for multiple symbols.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to load data for
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with price data for all symbols
    """
    logger.info(f"Loading price data for {len(symbols)} symbols")
    
    # Load data for each symbol
    price_data = {}
    
    for symbol in symbols:
        # Construct file path with _processed suffix
        file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Data file for {symbol} not found: {file_path}")
            continue
        
        # Load data
        try:
            df = pd.read_parquet(file_path)
            
            # Filter by date
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            else:
                df['date'] = pd.to_datetime(df['date'])
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                df.set_index('date', inplace=True)
            
            # Store close prices
            if 'close' in df.columns:
                price_data[symbol] = df['close']
            else:
                logger.warning(f"No 'close' column found for {symbol}")
                continue
                
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            continue
    
    # Combine into a single DataFrame
    if not price_data:
        logger.error("No data loaded for any symbols")
        return None
    
    prices_df = pd.DataFrame(price_data)
    
    # Resample to daily for regime detection
    daily_prices = prices_df.resample('D').last().dropna(how='all')
    
    logger.info(f"Combined data shape: {daily_prices.shape}")
    
    return daily_prices


def load_pairs_data(pairs_file):
    """
    Load pairs data from a JSON file.
    
    Parameters:
    -----------
    pairs_file : str
        Path to pairs JSON file
        
    Returns:
    --------
    list
        List of pairs as (ticker1, ticker2, hedge_ratio) tuples
    """
    logger.info(f"Loading pairs data from {pairs_file}")
    
    try:
        with open(pairs_file, 'r') as f:
            data = json.load(f)
        
        pairs = []
        for pair in data['pairs']:
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            hedge_ratio = pair.get('hedge_ratio', 1.0)
            pairs.append((ticker1, ticker2, hedge_ratio))
        
        logger.info(f"Loaded {len(pairs)} pairs")
        return pairs
    except Exception as e:
        logger.error(f"Error loading pairs data: {e}")
        return []


def save_config(config, output_dir, filename):
    """
    Save configuration to a JSON file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_dir : str
        Output directory
    filename : str
        Output filename
        
    Returns:
    --------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{filename}_{timestamp}.json")
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Saved configuration to {output_file}")
    
    return output_file


def main(parsed_args=None):
    """
    Main function to run intraday parameter optimization.
    
    Parameters:
    -----------
    parsed_args : argparse.Namespace, optional
        Command line arguments
    """
    # Use provided args or parse from command line
    if parsed_args is None:
        parser = argparse.ArgumentParser(description='Optimize parameters for intraday trading')
        
        parser.add_argument('--pairs_file', type=str, required=True,
                            help='Path to pairs JSON file')
        parser.add_argument('--start_date', type=str, required=True,
                            help='Start date for optimization (YYYY-MM-DD)')
        parser.add_argument('--end_date', type=str, required=True,
                            help='End date for optimization (YYYY-MM-DD)')
        parser.add_argument('--n_regimes', type=int, default=3,
                            help='Number of regimes to detect')
        parser.add_argument('--lookback_window', type=int, default=60,
                            help='Lookback window for regime detection')
        parser.add_argument('--timeframe', type=str, default='1hour',
                            help='Timeframe for data')
        parser.add_argument('--commission', type=float, default=2.0,
                            help='Commission per contract')
        parser.add_argument('--slippage', type=float, default=1.0,
                            help='Slippage per contract')
        parser.add_argument('--account_size', type=float, default=25000,
                            help='Initial account size')
        parser.add_argument('--n_jobs', type=int, default=1,
                            help='Number of parallel jobs')
        parser.add_argument('--output_dir', type=str, default='output/optimization',
                            help='Directory for output files')
        parser.add_argument('--quick_mode', action='store_true',
                            help='Run a quick optimization with fewer iterations')
        
        args = parser.parse_args()
    else:
        args = parsed_args
    
    # Load pairs data
    pairs = load_pairs_data(args.pairs_file)
    
    if not pairs:
        logger.error("No pairs loaded. Exiting.")
        return
    
    # Extract all symbols from pairs
    symbols = []
    for ticker1, ticker2, _ in pairs:
        if ticker1 not in symbols:
            symbols.append(ticker1)
        if ticker2 not in symbols:
            symbols.append(ticker2)
    
    # Load price data
    prices_df = load_price_data(symbols, args.start_date, args.end_date)
    
    if prices_df is None or prices_df.empty:
        logger.error("No price data loaded. Exiting.")
        return
    
    # Create optimizer
    optimizer = IntradayParameterOptimizer(
        n_regimes=args.n_regimes if hasattr(args, 'n_regimes') else 3,
        n_jobs=args.n_jobs if hasattr(args, 'n_jobs') else 1,
        verbose=True
    )
    
    # Run optimization for each regime
    regime_results = optimizer.optimize_by_regime(
        pairs=pairs,
        prices_df=prices_df,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe if hasattr(args, 'timeframe') else '1hour',
        commission=args.commission if hasattr(args, 'commission') else 2.0,
        slippage=args.slippage if hasattr(args, 'slippage') else 1.0,
        account_size=args.account_size if hasattr(args, 'account_size') else 25000,
        lookback_window=args.lookback_window if hasattr(args, 'lookback_window') else 60
    )
    
    # Create adaptive parameter configuration
    adaptive_config = optimizer.create_adaptive_parameter_config()
    
    # Save adaptive parameter configuration
    config_file = save_config(
        adaptive_config,
        args.output_dir,
        'adaptive_parameters'
    )
    
    # Plot regime comparison
    plot_file = optimizer.plot_regime_comparison(args.output_dir)
    
    logger.info("Optimization complete!")
    logger.info(f"Saved adaptive parameter configuration to {config_file}")
    if plot_file:
        logger.info(f"Saved regime parameter comparison to {plot_file}")


if __name__ == "__main__":
    main() 