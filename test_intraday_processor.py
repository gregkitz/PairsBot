"""
Test script for the IntradayDataProcessor.

This script demonstrates how to use the IntradayDataProcessor for loading
and preprocessing intraday data for pairs trading.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from src.data_processor import IntradayDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_pair_data_loading():
    """Test loading data for a pair."""
    data_dir = "data/processed"
    
    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize processor
    processor = IntradayDataProcessor(data_dir=data_dir)
    
    # Get available symbols
    symbols = processor.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair using the first two symbols
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Testing pair: {pair_id}")
    
    # Set test date range (adjust these dates based on your data)
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Load pair data
    pair_data = processor.load_pair_data(pair_id, start_date, end_date, timeframe="5min")
    
    if pair_data is None:
        logger.error("Failed to load pair data")
        return
    
    # Log data info
    logger.info(f"Loaded price data: {pair_data['prices'].shape}")
    if pair_data['volumes'] is not None:
        logger.info(f"Loaded volume data: {pair_data['volumes'].shape}")
    
    # Analyze data quality
    price_data = pair_data['prices']
    missing_data = price_data.isna().sum()
    logger.info(f"Missing values in price data:\n{missing_data}")
    
    # Create test configuration
    test_config = {
        "lookback": 20,
        "hedge_ratio": 1.0,
        "use_rolling_regression": True,
        "regression_window": 30
    }
    
    # Calculate spread
    symbol1, symbol2 = pair_id.split('_')
    spread_df = processor.calculate_spread(price_data, symbol1, symbol2, test_config)
    
    if spread_df is None:
        logger.error("Failed to calculate spread")
        return
    
    logger.info(f"Calculated spread data: {spread_df.shape}")
    
    # Visualize results
    visualize_data(symbol1, symbol2, price_data, spread_df)

def test_multiple_pairs():
    """Test loading data for multiple pairs."""
    data_dir = "data/processed"
    
    # Initialize processor
    processor = IntradayDataProcessor(data_dir=data_dir)
    
    # Get available symbols
    symbols = processor.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")
    
    if len(symbols) < 4:
        logger.error("Not enough symbols available for multiple pair analysis")
        return
    
    # Create test pairs
    pair_ids = [
        f"{symbols[0]}_{symbols[1]}",
        f"{symbols[2]}_{symbols[3]}"
    ]
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-10"
    
    # Load multiple pairs data
    pairs_data = processor.load_multiple_pairs(pair_ids, start_date, end_date, timeframe="5min")
    
    # Log results
    for pair_id, data in pairs_data.items():
        if data is not None:
            logger.info(f"Pair {pair_id}: loaded {data['prices'].shape[0]} data points")
        else:
            logger.error(f"Failed to load data for pair {pair_id}")

def test_missing_data_interpolation():
    """Test missing data interpolation."""
    data_dir = "data/processed"
    
    # Initialize processor
    processor = IntradayDataProcessor(data_dir=data_dir)
    
    # Get available symbols
    symbols = processor.get_available_symbols()
    if not symbols:
        logger.error("No symbols available")
        return
    
    # Load data for a single symbol
    symbol = symbols[0]
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    
    # Load raw data
    raw_data = processor.load_intraday_data(symbol, start_date, end_date, timeframe="5min")
    
    if raw_data is None:
        logger.error(f"Failed to load data for {symbol}")
        return
    
    # Introduce artificial missing data
    mask = raw_data.index % 5 == 0
    sparse_data = raw_data.copy()
    sparse_data.loc[mask, 'close'] = None
    
    # Interpolate missing data
    interpolated_data = processor.interpolate_missing_bars(sparse_data)
    
    # Compare
    logger.info(f"Raw data shape: {raw_data.shape}")
    logger.info(f"Sparse data missing values: {sparse_data['close'].isna().sum()}")
    logger.info(f"Interpolated data missing values: {interpolated_data['close'].isna().sum()}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(raw_data.index, raw_data['close'], label='Original')
    plt.plot(sparse_data.index, sparse_data['close'], 'o', label='Sparse', alpha=0.5)
    plt.plot(interpolated_data.index, interpolated_data['close'], '--', label='Interpolated')
    plt.title(f"Missing Data Interpolation - {symbol}")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_dir = "output/intraday_tests"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_interpolation_test.png"))
    plt.close()

def visualize_data(symbol1, symbol2, price_data, spread_df):
    """Visualize price and spread data."""
    # Create output directory
    output_dir = "output/intraday_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot prices
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(price_data.index, price_data[symbol1], label=symbol1)
    plt.plot(price_data.index, price_data[symbol2], label=symbol2)
    plt.title(f"Price Data: {symbol1} vs {symbol2}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(spread_df.index, spread_df['spread'], label='Spread')
    plt.plot(spread_df.index, spread_df['mean'], '--', label='Mean')
    plt.fill_between(
        spread_df.index,
        spread_df['mean'] - 2 * spread_df['std'],
        spread_df['mean'] + 2 * spread_df['std'],
        alpha=0.2,
        label='±2 Std Dev'
    )
    plt.title("Spread")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(spread_df.index, spread_df['zscore'], label='Z-Score')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    plt.title("Z-Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{symbol1}_{symbol2}_analysis.png"))
    plt.close()

if __name__ == "__main__":
    logger.info("Testing IntradayDataProcessor")
    
    # Test pair data loading
    test_pair_data_loading()
    
    # Test multiple pairs loading
    test_multiple_pairs()
    
    # Test missing data interpolation
    test_missing_data_interpolation()
    
    logger.info("Tests completed") 