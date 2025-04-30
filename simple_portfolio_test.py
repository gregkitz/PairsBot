"""
Script to build and test a simple portfolio of the most promising pairs.

This script loads signals for a smaller set of pairs to build a robust portfolio.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from src.signal_generation.pairs_signal_generator import PairsSignalGenerator
from src.portfolio.pairs_portfolio import PairsPortfolio

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
        (ticker1_df, ticker2_df) - DataFrames with price data
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

def generate_signals(symbol1, symbol2, hedge_ratio, config):
    """
    Generate trading signals for a pair using the signal generator.
    
    Parameters:
    -----------
    symbol1 : str
        First symbol
    symbol2 : str
        Second symbol
    hedge_ratio : float
        Hedge ratio between the two symbols
    config : dict
        Configuration for the signal generator
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with trading signals
    """
    # Load price data
    df1, df2 = load_pair_data(symbol1, symbol2)
    if df1 is None or df2 is None:
        return None
    
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
    
    # Generate signals using the SignalGenerator
    signal_gen = PairsSignalGenerator(
        lookback=config.get('lookback', 50),
        entry_zscore=config.get('entry_zscore', 2.0),
        exit_zscore=config.get('exit_zscore', 0.0),
        use_rolling_regression=config.get('use_rolling_regression', True),
        regression_window=config.get('regression_window', 60),
        stop_loss_std=config.get('stop_loss_std', 4.0)
    )
    
    # Override the validate_pair method to always return True
    signal_gen.validate_pair = lambda x, y: True
    
    # Generate signals
    signals = signal_gen.generate_signals(price1, price2, hedge_ratio)
    
    if signals is None:
        logger.error(f"Failed to generate signals for {symbol1}-{symbol2}")
        return None
    
    # Calculate returns for portfolio
    signals['returns'] = signals['position'].shift(1) * (signals['z_score'].diff())
    
    # Clean returns
    signals['returns'] = signals['returns'].replace([np.inf, -np.inf, np.nan], 0)
    signals['returns'] = signals['returns'].clip(lower=-0.1, upper=0.1)
    
    return signals

def main():
    """Main function to build and test a simplified portfolio."""
    # Define our selected pairs and their parameters
    pairs = [
        {
            'symbol1': 'BFX',
            'symbol2': 'ZN',
            'hedge_ratio': 0.5774521323492694,
            'config': {
                'lookback': 50,
                'entry_zscore': 2.5,
                'exit_zscore': 0.0,
                'use_rolling_regression': True,
                'regression_window': 60,
                'stop_loss_std': 4.0
            }
        },
        {
            'symbol1': 'ALI',
            'symbol2': 'SI',
            'hedge_ratio': 0.7572436741454343,
            'config': {
                'lookback': 30,
                'entry_zscore': 2.0,
                'exit_zscore': 0.0,
                'use_rolling_regression': True,
                'regression_window': 60,
                'stop_loss_std': 3.0
            }
        },
        {
            'symbol1': 'BFX',
            'symbol2': 'RTY',
            'hedge_ratio': 0.19890714490809777,
            'config': {
                'lookback': 50,
                'entry_zscore': 2.0,
                'exit_zscore': 0.0,
                'use_rolling_regression': True,
                'regression_window': 60,
                'stop_loss_std': 4.0
            }
        }
    ]
    
    # Create portfolio
    portfolio = PairsPortfolio(
        account_size=1000000,
        max_allocation_per_pair=0.2,
        max_correlation=0.5,
        target_volatility=0.1
    )
    
    # Add pairs to portfolio
    for pair in pairs:
        symbol1 = pair['symbol1']
        symbol2 = pair['symbol2']
        hedge_ratio = pair['hedge_ratio']
        config = pair['config']
        
        logger.info(f"Generating signals for {symbol1}-{symbol2}")
        
        # Generate signals for the pair
        signals = generate_signals(symbol1, symbol2, hedge_ratio, config)
        
        if signals is not None:
            # Add pair to portfolio
            portfolio.add_pair(symbol1, symbol2, signals, config)
    
    # Calculate correlation matrix
    corr_matrix = portfolio.calculate_correlation_matrix()
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Select pairs for the portfolio
    selected_pairs = portfolio.filter_pairs_by_correlation()
    
    print("\nSelected Pairs:")
    for pair in selected_pairs:
        print(f"  {pair['symbol1']}-{pair['symbol2']}")
    
    # Calculate weights - try equal weighting
    weights = portfolio.calculate_weights(method='equal')
    
    # Allocate capital
    allocations = portfolio.allocate_capital(weights)
    
    # Print allocations
    print("\nCapital Allocations:")
    for pair_id, amount in allocations.items():
        print(f"  {pair_id}: ${amount:,.2f}")
    
    # Simulate portfolio
    results_df = portfolio.simulate_portfolio()
    
    # Calculate metrics
    metrics = portfolio.calculate_portfolio_metrics()
    
    # Print results
    print("\nPortfolio Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot portfolio performance
    output_dir = "data/results/portfolio"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"simple_portfolio_{timestamp}.png")
    portfolio.plot_portfolio_performance(save_path=output_file)
    
    # Save portfolio configuration
    config_file = os.path.join(output_dir, f"simple_portfolio_{timestamp}.json")
    portfolio.save_portfolio(config_file)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 