#!/usr/bin/env python
"""
Data preprocessing script for intraday model training.

This script converts historical CSV data to the parquet format expected by the training script.
"""

import os
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_symbol(symbol, timeframe="5min", start_year=2010, end_year=2023):
    """
    Preprocess historical data for a specific symbol.
    
    Parameters:
    -----------
    symbol : str
        Symbol to preprocess
    timeframe : str
        Timeframe to use (1min, 5min, etc.)
    start_year : int
        Start year for filtering data
    end_year : int
        End year for filtering data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    logger.info(f"Preprocessing {symbol} data for {timeframe} timeframe")
    
    # Define source and destination paths
    source_path = f"data/historical/{symbol}/{timeframe}/data.csv"
    dest_dir = "data/processed"
    
    # Check if source file exists
    if not os.path.exists(source_path):
        logger.error(f"Source file not found: {source_path}")
        return None
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(source_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['datetime'])
        
        # Filter by year
        df = df[(df['timestamp'].dt.year >= start_year) & (df['timestamp'].dt.year <= end_year)]
        
        # Drop original datetime column
        df.drop('datetime', axis=1, inplace=True)
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Rename columns to match expected format
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Save processed data
        dest_path = os.path.join(dest_dir, f"{symbol}_processed.parquet")
        df.to_parquet(dest_path)
        
        logger.info(f"Saved processed data to {dest_path}: {len(df)} rows")
        
        return df
    
    except Exception as e:
        logger.error(f"Error preprocessing {symbol} data: {e}")
        return None

def main():
    """Main function to preprocess data for intraday model training."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Preprocess data for intraday model training")
    
    parser.add_argument(
        "--symbols", 
        type=str,
        default="ES,NQ,GC,SI",
        help="Comma-separated list of symbols to preprocess"
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str,
        default="5min",
        help="Timeframe to use (1min, 5min, etc.)"
    )
    
    parser.add_argument(
        "--start_year", 
        type=int,
        default=2010,
        help="Start year for filtering data"
    )
    
    parser.add_argument(
        "--end_year", 
        type=int,
        default=2023,
        help="End year for filtering data"
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Preprocess each symbol
    for symbol in tqdm(symbols, desc="Preprocessing symbols"):
        df = preprocess_symbol(symbol, args.timeframe, args.start_year, args.end_year)
        
        if df is None:
            logger.warning(f"Failed to preprocess {symbol}")
    
    logger.info("Data preprocessing complete")

if __name__ == "__main__":
    main() 