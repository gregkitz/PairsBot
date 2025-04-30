#!/usr/bin/env python
"""
Train Market Regime Classifier Model

This script loads historical data for futures contracts, trains a market regime classifier,
and saves the model for use in paper trading.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import joblib

from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_pair_data(pairs, data_dir="data/processed", start_date=None, end_date=None, resample_freq='1D'):
    """
    Load price data for specified pairs from processed data files.
    
    Parameters:
    -----------
    pairs : list
        List of pairs to load (e.g., ['ES_NQ', 'GC_SI'])
    data_dir : str
        Directory containing processed data files
    start_date : str, optional
        Start date for filtering data (YYYY-MM-DD)
    end_date : str, optional
        End date for filtering data (YYYY-MM-DD)
    resample_freq : str
        Frequency to resample data to (e.g., '1D', '1H')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with price data
    """
    # Extract unique symbols from pairs
    symbols = set()
    for pair in pairs:
        sym1, sym2 = pair.split('_')
        symbols.add(sym1)
        symbols.add(sym2)
    
    symbols = list(symbols)
    logger.info(f"Loading data for symbols: {symbols}")
    
    # Load data for each symbol
    dfs = {}
    
    for symbol in symbols:
        try:
            file_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            
            if not os.path.exists(file_path):
                logger.error(f"Data file not found: {file_path}")
                continue
            
            # Load data
            df = pd.read_parquet(file_path)
            
            # Ensure data is sorted by timestamp
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df.set_index('timestamp', inplace=True)
                else:
                    logger.warning(f"No timestamp column found for {symbol}")
                    continue
            
            df.sort_index(inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Resample to desired frequency
            if resample_freq:
                df = df.resample(resample_freq).last()
                df = df.fillna(method='ffill')
            
            # Extract close price
            dfs[symbol] = df['close']
            logger.info(f"Loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    # Combine into a single DataFrame
    if dfs:
        prices_df = pd.DataFrame(dfs)
        logger.info(f"Combined price data for {len(dfs)} symbols, shape: {prices_df.shape}")
        return prices_df
    else:
        logger.error("No data loaded")
        return None

def train_regime_classifier(prices_df, n_regimes=3, lookback_window=60, features=None):
    """
    Train a market regime classifier.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    n_regimes : int
        Number of regimes to detect
    lookback_window : int
        Lookback window for feature calculation
    features : list, optional
        List of features to use for regime detection
        
    Returns:
    --------
    MarketRegimeClassifier
        Trained regime classifier
    """
    # Default features if not provided
    if features is None:
        features = [
            'volatility',
            'trend_strength',
            'correlation',
            'mean_reversion'
        ]
    
    logger.info(f"Training market regime classifier with {n_regimes} regimes")
    logger.info(f"Using features: {features}")
    
    # Create regime classifier
    classifier = MarketRegimeClassifier(
        n_regimes=n_regimes,
        lookback_window=lookback_window,
        features=features,
        method='kmeans'
    )
    
    # Calculate features
    logger.info("Calculating features for regime detection")
    features_df = classifier.calculate_features(prices_df)
    
    # Fit classifier
    logger.info("Fitting regime classifier")
    classifier.fit(features_df)
    
    # Predict regimes
    regimes = classifier.predict(features_df)
    
    # Print regime distribution
    regime_counts = regimes.value_counts()
    regime_pcts = regime_counts / len(regimes) * 100
    
    logger.info("Regime Distribution:")
    for regime, pct in regime_pcts.items():
        logger.info(f"  Regime {regime+1}: {pct:.1f}%")
    
    # Print current regime
    current_regime = regimes.iloc[-1]
    current_params = classifier.get_regime_parameters(current_regime)
    
    logger.info(f"Current Regime: {current_regime+1} - {current_params['regime_description']}")
    logger.info("Recommended Parameters for Current Regime:")
    logger.info(f"  Entry Z-Score: {current_params['entry_zscore']:.2f}")
    logger.info(f"  Exit Z-Score: {current_params['exit_zscore']:.2f}")
    logger.info(f"  Stop Loss (std): {current_params['stop_loss_std']:.2f}")
    logger.info(f"  Position Size Factor: {current_params['position_size_factor']:.2f}")
    
    return classifier, features_df, regimes

def save_regime_classifier(classifier, output_dir="models/intraday", timestamp=None):
    """
    Save the trained regime classifier.
    
    Parameters:
    -----------
    classifier : MarketRegimeClassifier
        Trained regime classifier
    output_dir : str
        Directory to save the model
    timestamp : str, optional
        Timestamp for file naming
        
    Returns:
    --------
    str
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the classifier
    model_path = os.path.join(output_dir, "market_regime_classifier.joblib")
    joblib.dump(classifier, model_path)
    
    logger.info(f"Saved market regime classifier to {model_path}")
    
    # Save the regime plot
    plot_path = os.path.join(output_dir, f"market_regimes_{timestamp}.png")
    return model_path

def main():
    """Main function to train and save the market regime classifier."""
    # Define pairs we're using in paper trading
    pairs = ["ES_NQ", "GC_SI"]  # Add more pairs if needed
    
    # Load data
    # Use the last 3 years of data for training
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now().replace(year=datetime.now().year - 3)).strftime("%Y-%m-%d")
    
    prices_df = load_pair_data(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        resample_freq='1D'  # Daily data for regime detection
    )
    
    if prices_df is None or prices_df.empty:
        logger.error("Failed to load price data")
        return
    
    # Train regime classifier
    classifier, features_df, regimes = train_regime_classifier(
        prices_df,
        n_regimes=3,
        lookback_window=60
    )
    
    # Save classifier
    model_path = save_regime_classifier(classifier)
    
    # Generate and save plots
    output_dir = "models/intraday"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot regimes
    regimes_plot_path = os.path.join(output_dir, f"market_regimes_{timestamp}.png")
    classifier.plot_regimes(prices_df, regimes, save_path=regimes_plot_path)
    
    # Plot feature importance
    feature_plot_path = os.path.join(output_dir, f"regime_feature_importance_{timestamp}.png")
    classifier.plot_feature_importance(save_path=feature_plot_path)
    
    logger.info(f"Saved regime plot to {regimes_plot_path}")
    logger.info(f"Saved feature importance plot to {feature_plot_path}")
    
    # Save regime history
    history_path = os.path.join(output_dir, f"regime_history_{timestamp}.csv")
    regime_history = pd.DataFrame({
        'date': regimes.index,
        'regime': regimes.values
    })
    regime_history.to_csv(history_path, index=False)
    
    logger.info(f"Saved regime history to {history_path}")
    logger.info("Market regime classifier training completed successfully")

if __name__ == "__main__":
    main() 