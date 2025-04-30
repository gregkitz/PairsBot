#!/usr/bin/env python
"""
Load the trained market regime classifier model and update the intraday ML paper trader.

This script loads the saved market regime classifier model and applies it to the
paper trading system to detect the current market regime.
"""

import os
import sys
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import required components
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from src.paper_trading.intraday_ml_paper_trader import IntradayMLPaperTrader
from run_paper_trade_fixed import PaperTrader, get_historical_data, get_subscribed_symbols

def load_market_regime_model(model_path="models/intraday/market_regime_classifier.joblib"):
    """
    Load the trained market regime classifier model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    MarketRegimeClassifier
        Loaded model
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        logger.info(f"Loading market regime classifier from {model_path}")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def load_recent_data(symbols, days=60, data_dir="data/processed"):
    """
    Load recent data for regime detection.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to load data for
    days : int
        Number of days of data to load
    data_dir : str
        Directory containing processed data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with recent price data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Convert to string format
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Loading data from {start_date_str} to {end_date_str} for {symbols}")
    
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
            
            # Filter by date range
            df = df[(df.index >= start_date_str) & (df.index <= end_date_str)]
            
            # Resample to daily
            df_daily = df.resample('1D').last()
            df_daily = df_daily.fillna(method='ffill')
            
            # Extract close price
            dfs[symbol] = df_daily['close']
            logger.info(f"Loaded {len(df_daily)} days of data for {symbol}")
            
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

def detect_current_regime(model, prices_df):
    """
    Detect the current market regime.
    
    Parameters:
    -----------
    model : MarketRegimeClassifier
        Trained regime classifier model
    prices_df : pd.DataFrame
        DataFrame with recent price data
        
    Returns:
    --------
    dict
        Dictionary with regime information
    """
    if model is None or prices_df is None or prices_df.empty:
        logger.error("Missing model or data for regime detection")
        return None
    
    try:
        # Calculate features
        logger.info("Calculating features for regime detection")
        features_df = model.calculate_features(prices_df)
        
        if features_df.empty:
            logger.error("No features calculated")
            return None
        
        # Predict regimes
        logger.info("Predicting regimes")
        regimes = model.predict(features_df)
        
        # Get current regime (most recent)
        current_regime = regimes.iloc[-1]
        
        # Get regime parameters
        regime_params = model.get_regime_parameters(current_regime)
        
        # Create result
        result = {
            'regime': current_regime,
            'regime_description': regime_params['regime_description'],
            'entry_zscore': regime_params['entry_zscore'],
            'exit_zscore': regime_params['exit_zscore'],
            'stop_loss_std': regime_params['stop_loss_std'],
            'position_size_factor': regime_params['position_size_factor']
        }
        
        logger.info(f"Current Regime: {current_regime+1} - {regime_params['regime_description']}")
        logger.info(f"Entry Z-Score: {regime_params['entry_zscore']:.2f}")
        logger.info(f"Exit Z-Score: {regime_params['exit_zscore']:.2f}")
        logger.info(f"Stop Loss (std): {regime_params['stop_loss_std']:.2f}")
        logger.info(f"Position Size Factor: {regime_params['position_size_factor']:.2f}")
        
        # Plot regimes
        plot_path = os.path.join("output", "paper_trading", "current_regime.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        model.plot_regimes(prices_df, regimes, save_path=plot_path)
        
        return result
    except Exception as e:
        logger.error(f"Error detecting current regime: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to load the regime model and detect the current regime."""
    # Load the model
    model = load_market_regime_model()
    
    if model is None:
        logger.error("Failed to load market regime model")
        return
    
    # Define symbols for the pairs we're using
    symbols = ["ES", "NQ", "GC", "SI"]
    
    # Load recent data
    prices_df = load_recent_data(symbols, days=90)
    
    if prices_df is None or prices_df.empty:
        logger.error("Failed to load price data")
        return
    
    # Try to detect current regime
    current_regime = detect_current_regime(model, prices_df)
    
    # If detection fails, use the trained model's information directly
    if current_regime is None:
        logger.info("Detection failed, using information from trained model directly")
        try:
            # Get the regime information from the model attributes
            if hasattr(model, 'model') and model.model is not None:
                # Use the low volatility regime as default (regime 1)
                regime_id = 1
                regime_params = model.get_regime_parameters(regime_id)
                
                # Create the current regime info
                current_regime = {
                    'regime': regime_id,
                    'regime_description': regime_params['regime_description'],
                    'entry_zscore': regime_params['entry_zscore'],
                    'exit_zscore': regime_params['exit_zscore'],
                    'stop_loss_std': regime_params['stop_loss_std'],
                    'position_size_factor': regime_params['position_size_factor']
                }
                
                logger.info(f"Using regime {regime_id+1}: {regime_params['regime_description']}")
                logger.info(f"Entry Z-Score: {regime_params['entry_zscore']:.2f}")
                logger.info(f"Exit Z-Score: {regime_params['exit_zscore']:.2f}")
                logger.info(f"Stop Loss (std): {regime_params['stop_loss_std']:.2f}")
                logger.info(f"Position Size Factor: {regime_params['position_size_factor']:.2f}")
            else:
                # Create a default regime
                logger.info("Model lacks regime information, using default values")
                current_regime = {
                    'regime': 1,
                    'regime_description': "Low Volatility / Weak Trend",
                    'entry_zscore': 1.5,
                    'exit_zscore': 0.5,
                    'stop_loss_std': 3.0,
                    'position_size_factor': 1.0
                }
        except Exception as e:
            logger.error(f"Error extracting regime information: {e}")
            return
    
    # Save current regime to a file for the paper trader to read
    regime_path = os.path.join("models", "intraday", "current_regime.json")
    os.makedirs(os.path.dirname(regime_path), exist_ok=True)
    
    import json
    
    with open(regime_path, 'w') as f:
        json.dump(current_regime, f, indent=2)
    
    logger.info(f"Saved current regime information to {regime_path}")
    logger.info("Restart paper trading to apply the updated regime")

if __name__ == "__main__":
    main() 