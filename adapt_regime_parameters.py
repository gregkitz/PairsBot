"""
Script to adapt trading parameters based on detected market regimes.

This script automatically updates the trading parameters for our portfolio
based on current market regime characteristics.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time

from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_portfolio_config(config_file):
    """
    Load portfolio configuration from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the portfolio configuration file
        
    Returns:
    --------
    dict
        Portfolio configuration
    """
    logger.info(f"Loading portfolio configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration with {len(config['pairs'])} pairs")
        return config
    except Exception as e:
        logger.error(f"Error loading portfolio configuration: {e}")
        return None

def load_pair_data(symbol, data_dir="data/processed"):
    """
    Load price data for a specific symbol.
    
    Parameters:
    -----------
    symbol : str
        Symbol to load data for
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with price data for the symbol
    """
    try:
        file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return None
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Ensure data is sorted by timestamp
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                logger.warning(f"No timestamp column found for {symbol}")
                return None
        
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded data for {symbol}: {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return None

def resample_data(df, freq='1D'):
    """
    Resample data to a specified frequency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    freq : str
        Frequency to resample to
        
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame
    """
    # Resample
    resampled = df.resample(freq).last()
    
    # Forward fill missing values
    resampled = resampled.ffill()
    
    return resampled

def detect_current_regime(symbols, n_regimes=3, lookback_window=60):
    """
    Detect the current market regime using price data for multiple symbols.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to use for regime detection
    n_regimes : int
        Number of regimes to detect
    lookback_window : int
        Lookback window for feature calculation
        
    Returns:
    --------
    tuple
        (regime_index, regime_params)
    """
    logger.info(f"Detecting market regime using {len(symbols)} symbols")
    
    # Load data for all symbols
    price_data = {}
    
    for symbol in symbols:
        df = load_pair_data(symbol)
        
        if df is not None:
            price_data[symbol] = df['close']
    
    if not price_data:
        logger.error("No data loaded for any symbols")
        return None, None
    
    # Combine into a single DataFrame
    prices_df = pd.DataFrame(price_data)
    
    # Resample to daily data for regime detection
    daily_prices = resample_data(prices_df)
    
    # Create classifier
    classifier = MarketRegimeClassifier(
        n_regimes=n_regimes,
        lookback_window=lookback_window
    )
    
    # Calculate features
    features_df = classifier.calculate_features(daily_prices)
    
    # Fit classifier
    classifier.fit(features_df)
    
    # Predict regimes
    regimes = classifier.predict(features_df)
    
    # Get current regime (most recent data point)
    current_regime = regimes.iloc[-1]
    current_params = classifier.get_regime_parameters(current_regime)
    
    logger.info(f"Detected current regime: {current_regime+1} - {current_params['regime_description']}")
    
    return current_regime, current_params

def adapt_pair_parameters(pair_config, regime_params):
    """
    Adapt pair trading parameters based on the current regime.
    
    Parameters:
    -----------
    pair_config : dict
        Configuration for a trading pair
    regime_params : dict
        Parameters for the current regime
        
    Returns:
    --------
    dict
        Updated pair configuration
    """
    # Make a copy of the original config
    updated_config = pair_config.copy()
    
    # Update trading parameters based on regime
    updated_config['config']['entry_zscore'] = regime_params['entry_zscore']
    updated_config['config']['exit_zscore'] = regime_params['exit_zscore']
    updated_config['config']['stop_loss_std'] = regime_params['stop_loss_std']
    
    # Add regime information
    updated_config['regime_info'] = {
        'regime_description': regime_params['regime_description'],
        'position_size_factor': regime_params['position_size_factor'],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return updated_config

def update_portfolio(portfolio_config, regime_params):
    """
    Update all pairs in the portfolio with regime-adapted parameters.
    
    Parameters:
    -----------
    portfolio_config : dict
        Portfolio configuration
    regime_params : dict
        Parameters for the current regime
        
    Returns:
    --------
    dict
        Updated portfolio configuration
    """
    updated_config = portfolio_config.copy()
    
    # Update each pair in the portfolio
    for i, pair in enumerate(updated_config['pairs']):
        updated_config['pairs'][i] = adapt_pair_parameters(pair, regime_params)
    
    # Add regime information at portfolio level
    updated_config['regime_info'] = {
        'regime_description': regime_params['regime_description'],
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return updated_config

def save_updated_config(updated_config, output_dir="data/results/portfolio"):
    """
    Save the updated portfolio configuration to a file.
    
    Parameters:
    -----------
    updated_config : dict
        Updated portfolio configuration
    output_dir : str
        Directory to save the updated configuration
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"regime_adapted_portfolio_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(updated_config, f, indent=2)
    
    logger.info(f"Saved updated configuration to {output_file}")
    
    return output_file

def plot_portfolio_regime(symbols, regimes, output_dir="data/results/portfolio"):
    """
    Plot the regime detection results for the portfolio.
    
    Parameters:
    -----------
    symbols : list
        List of symbols in the portfolio
    regimes : pd.Series
        Series with regime labels
    output_dir : str
        Directory to save the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load price data for all symbols
    price_data = {}
    
    for symbol in symbols:
        df = load_pair_data(symbol)
        
        if df is not None:
            price_data[symbol] = df['close']
    
    if not price_data:
        logger.error("No data loaded for any symbols")
        return None
    
    # Combine into a single DataFrame
    prices_df = pd.DataFrame(price_data)
    
    # Resample to daily data
    daily_prices = resample_data(prices_df)
    
    # Plot regime detection
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot prices in the top panel
    for symbol in daily_prices.columns:
        # Normalize prices for better comparison
        norm_price = daily_prices[symbol] / daily_prices[symbol].iloc[0]
        axes[0].plot(daily_prices.index, norm_price, label=symbol)
    
    axes[0].set_title('Portfolio Asset Prices (Normalized)')
    axes[0].set_ylabel('Price (Normalized)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot regime in the bottom panel
    cmap = plt.cm.get_cmap('viridis', len(set(regimes)))
    
    for regime in sorted(set(regimes)):
        regime_mask = regimes == regime
        axes[1].fill_between(regimes.index, regime, regime + 1, 
                           where=regime_mask, 
                           facecolor=cmap(regime), 
                           alpha=0.7,
                           label=f'Regime {regime + 1}')
    
    axes[1].set_title('Market Regimes')
    axes[1].set_ylabel('Regime')
    axes[1].set_yticks(np.arange(len(set(regimes))) + 0.5)
    axes[1].set_yticklabels([f'Regime {i+1}' for i in range(len(set(regimes)))])
    axes[1].set_xlabel('Date')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"portfolio_regimes_{timestamp}.png")
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Saved portfolio regime plot to {output_file}")
    
    return output_file

def main():
    """Main function to adapt trading parameters based on market regime."""
    # Find most recent portfolio file
    portfolio_dir = "data/results/portfolio"
    portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.json') and f.startswith('simple_portfolio_')]
    
    if not portfolio_files:
        logger.error("No portfolio configuration files found")
        return
    
    # Sort by timestamp in filename
    portfolio_files.sort(reverse=True)
    latest_portfolio = os.path.join(portfolio_dir, portfolio_files[0])
    
    logger.info(f"Using latest portfolio: {latest_portfolio}")
    
    # Load portfolio configuration
    portfolio_config = load_portfolio_config(latest_portfolio)
    
    if portfolio_config is None:
        return
    
    # Get symbols from all pairs
    all_symbols = set()
    for pair in portfolio_config['pairs']:
        all_symbols.add(pair['symbol1'])
        all_symbols.add(pair['symbol2'])
    
    # Detect current market regime
    current_regime, regime_params = detect_current_regime(list(all_symbols))
    
    if current_regime is None or regime_params is None:
        return
    
    # Update portfolio configuration with regime-adapted parameters
    updated_config = update_portfolio(portfolio_config, regime_params)
    
    # Save updated configuration
    save_updated_config(updated_config)
    
    # Print summary
    print("\nRegime-Adaptive Parameters Applied")
    print(f"Current Regime: {regime_params['regime_description']}")
    print("\nUpdated Parameters:")
    print(f"  Entry Z-Score: {regime_params['entry_zscore']:.2f}")
    print(f"  Exit Z-Score: {regime_params['exit_zscore']:.2f}")
    print(f"  Stop Loss (std): {regime_params['stop_loss_std']:.2f}")
    print(f"  Position Size Factor: {regime_params['position_size_factor']:.2f}")
    
    # Create a visualization of the portfolio with regime overlay
    classifier = MarketRegimeClassifier(
        n_regimes=3,
        lookback_window=60
    )
    
    # Load data for all symbols
    price_data = {}
    
    for symbol in all_symbols:
        df = load_pair_data(symbol)
        
        if df is not None:
            price_data[symbol] = df['close']
    
    if price_data:
        # Combine into a single DataFrame
        prices_df = pd.DataFrame(price_data)
        
        # Resample to daily data
        daily_prices = resample_data(prices_df)
        
        # Calculate features
        features_df = classifier.calculate_features(daily_prices)
        
        # Fit classifier
        classifier.fit(features_df)
        
        # Predict regimes
        regimes = classifier.predict(features_df)
        
        # Plot portfolio with regime overlay
        plot_portfolio_regime(list(all_symbols), regimes)

if __name__ == "__main__":
    main() 