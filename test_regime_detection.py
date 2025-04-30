"""
Script to test the market regime detection module.

This script loads data from our processed pairs and identifies market regimes
to adapt trading parameters accordingly.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_data(data_dir="data/processed"):
    """
    Load price data for multiple instruments.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with price data
    """
    # Find all processed parquet files
    files = [f for f in os.listdir(data_dir) if f.endswith('_processed.parquet')]
    
    if not files:
        logger.error(f"No data files found in {data_dir}")
        return None
    
    # Load data for each file
    dfs = {}
    
    for file in files:
        try:
            symbol = file.split('_')[0]
            file_path = os.path.join(data_dir, file)
            
            # Load data
            df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found in {file}")
                    continue
            
            df.sort_index(inplace=True)
            
            # Extract close price
            dfs[symbol] = df['close']
            
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    # Combine into a single DataFrame
    if dfs:
        prices_df = pd.DataFrame(dfs)
        logger.info(f"Loaded price data for {len(dfs)} symbols")
        return prices_df
    else:
        logger.error("No data loaded")
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
    # Resample each column
    resampled = df.resample(freq).last()
    
    # Forward fill missing values
    resampled = resampled.fillna(method='ffill')
    
    return resampled

def detect_regimes(prices_df, n_regimes=3, lookback_window=60):
    """
    Detect market regimes in the price data.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    n_regimes : int
        Number of regimes to detect
    lookback_window : int
        Lookback window for feature calculation
        
    Returns:
    --------
    MarketRegimeClassifier
        Fitted regime classifier
    pd.DataFrame
        DataFrame with features
    pd.Series
        Series with regime labels
    """
    # Create regime classifier
    classifier = MarketRegimeClassifier(
        n_regimes=n_regimes,
        lookback_window=lookback_window
    )
    
    # Calculate features
    features_df = classifier.calculate_features(prices_df)
    
    # Fit classifier
    classifier.fit(features_df)
    
    # Predict regimes
    regimes = classifier.predict(features_df)
    
    return classifier, features_df, regimes

def main():
    """Main function to test regime detection."""
    # Load price data
    prices_df = load_pair_data()
    
    if prices_df is None:
        return
    
    # Resample to daily data
    daily_prices = resample_data(prices_df)
    
    # Select a subset of symbols for clarity
    selected_symbols = ['ES', 'NQ', 'ZN', 'SI', 'GC']
    selected_symbols = [s for s in selected_symbols if s in daily_prices.columns]
    
    if len(selected_symbols) < 2:
        # Use the first 5 symbols if selected ones not available
        selected_symbols = daily_prices.columns[:5]
    
    daily_prices = daily_prices[selected_symbols]
    
    # Detect regimes
    classifier, features_df, regimes = detect_regimes(daily_prices, n_regimes=3, lookback_window=60)
    
    # Plot regimes
    output_dir = "data/results/regimes"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"regimes_{timestamp}.png")
    
    classifier.plot_regimes(daily_prices, regimes, main_symbols=selected_symbols, save_path=output_file)
    
    # Plot feature importance
    features_file = os.path.join(output_dir, f"feature_importance_{timestamp}.png")
    classifier.plot_feature_importance(save_path=features_file)
    
    # Print regime parameters
    print("\nRegime Parameters:")
    for i in range(classifier.n_regimes):
        params = classifier.get_regime_parameters(i)
        
        print(f"\nRegime {i+1}: {params['regime_description']}")
        print(f"  Entry Z-Score: {params['entry_zscore']:.2f}")
        print(f"  Exit Z-Score: {params['exit_zscore']:.2f}")
        print(f"  Stop Loss (std): {params['stop_loss_std']:.2f}")
        print(f"  Position Size Factor: {params['position_size_factor']:.2f}")
    
    # Count percentage of time in each regime
    regime_counts = regimes.value_counts()
    regime_pcts = regime_counts / len(regimes) * 100
    
    print("\nRegime Distribution:")
    for regime, pct in regime_pcts.items():
        print(f"  Regime {regime+1}: {pct:.1f}%")
    
    # Find current regime (most recent data point)
    current_regime = regimes.iloc[-1]
    current_params = classifier.get_regime_parameters(current_regime)
    
    print(f"\nCurrent Regime: {current_regime+1} - {current_params['regime_description']}")
    print(f"Recommended Parameters for Current Regime:")
    print(f"  Entry Z-Score: {current_params['entry_zscore']:.2f}")
    print(f"  Exit Z-Score: {current_params['exit_zscore']:.2f}")
    print(f"  Stop Loss (std): {current_params['stop_loss_std']:.2f}")
    print(f"  Position Size Factor: {current_params['position_size_factor']:.2f}")
    
    # Save results
    results_file = os.path.join(output_dir, f"regime_results_{timestamp}.csv")
    results_df = pd.DataFrame({
        'date': regimes.index,
        'regime': regimes.values
    })
    results_df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main() 