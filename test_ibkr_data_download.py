#!/usr/bin/env python
"""
Test script to verify Interactive Brokers integration for data download.
This script downloads historical data for common futures contracts.

Make sure TWS or IB Gateway is running on localhost:7497 (paper trading) before running.
"""

import os
import sys
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print version info
logger.info(f"Python version: {sys.version}")
logger.info(f"Running from: {os.getcwd()}")

# Add the src directory to the path
src_dir = Path(__file__).resolve().parent / "src"
if src_dir.exists():
    sys.path.append(str(src_dir.parent))
    logger.info(f"Added {src_dir.parent} to sys.path")
else:
    logger.warning(f"Source directory not found at {src_dir}")

# Try importing the needed modules with detailed error handling
try:
    # First import ib_insync and patch asyncio
    import ib_insync
    from ib_insync import util
    logger.info(f"Loaded ib_insync version: {ib_insync.__version__}")
    
    # Patch asyncio directly here as well
    util.patchAsyncio()
    logger.info("Applied asyncio patch")
    
    # Now import our modules
    from src.data_processor.data_loader import DataLoader, download_historical_data
    from src.data_processor.feature_calculator import generate_features
    from src.connectors.ib.ib_connector import IBConnector
    
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error during import: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

def test_ib_connection():
    """Test the connection to Interactive Brokers"""
    logger.info("Testing connection to Interactive Brokers...")
    
    try:
        # Create an IBConnector instance
        ib = IBConnector(
            host="127.0.0.1",
            port=7497,  # Paper trading port
            client_id=1001,  # Use a high client ID to avoid conflicts
            read_only=True  # Read-only for data collection
        )
        
        logger.info("IBConnector instance created")
        
        # Try to connect
        logger.info("Attempting to connect to IB...")
        connected = ib.connect()
        
        if connected:
            logger.info("Successfully connected to Interactive Brokers")
            
            try:
                # Verify connection by requesting account information
                logger.info("Requesting account information...")
                account_info = ib.get_account_info()
                
                if account_info:
                    logger.info(f"Account information retrieved: {len(account_info)} values")
                    # Log a few key account values if available
                    for key in ['NetLiquidation', 'AvailableFunds', 'TotalCashValue']:
                        if key in account_info:
                            logger.info(f"{key}: {account_info[key]}")
                else:
                    logger.warning("Retrieved empty account information")
            except Exception as e:
                logger.error(f"Error getting account information: {str(e)}")
                traceback.print_exc()
            
            # Disconnect
            logger.info("Disconnecting from Interactive Brokers...")
            ib.disconnect()
            logger.info("Disconnected from Interactive Brokers")
            
            return True
        else:
            logger.error("Failed to connect to Interactive Brokers")
            return False
    except Exception as e:
        logger.error(f"Unexpected error in test_ib_connection: {str(e)}")
        traceback.print_exc()
        return False

def test_data_download():
    """Test downloading data for futures contracts"""
    logger.info("Testing futures data download...")
    
    # Define symbols and timeframes
    symbols = ['ES', 'NQ', 'GC', 'CL']  # E-mini S&P, E-mini NASDAQ, Gold, Crude Oil
    timeframes = ['5min', '1hour', '1day']
    
    # Calculate date range for the test (last 5 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    
    # Create a DataLoader
    loader = DataLoader(
        data_dir="data/test_historical",
        ib_host="127.0.0.1",
        ib_port=7497,
        ib_client_id=1002  # Use a different client ID to avoid conflicts
    )
    
    # Test downloading data for each symbol and timeframe
    for symbol in symbols:
        for tf in timeframes:
            logger.info(f"Downloading {symbol} {tf} data...")
            
            data = loader.download_data(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and not data.empty:
                logger.info(f"Successfully downloaded {len(data)} rows for {symbol} {tf}")
                logger.info(f"Data sample:\n{data.head()}")
            else:
                logger.error(f"Failed to download data for {symbol} {tf}")

def test_full_pipeline():
    """Test the full data flow pipeline"""
    logger.info("Testing full data flow pipeline...")
    
    # Define symbols and timeframes
    symbols = ['ES']  # Just use ES to avoid connection issues
    timeframes = ['1day']  # Just use daily timeframe for simplicity
    
    # Calculate date range for the test (last 30 days)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Download data
    logger.info("Downloading historical data...")
    data_results = download_historical_data(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        data_dir="data/test_pipeline",
        parallel=False,  # Use sequential processing to avoid connection issues
        ib_host="127.0.0.1",
        ib_port=7497,
        client_id_base=10000  # Use a much higher client ID base to avoid conflicts
    )
    
    # Generate features
    logger.info("Generating technical features...")
    feature_results = generate_features(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        data_dir="data/test_pipeline",
        feature_dir="data/test_pipeline/features",
        parallel=False  # Use sequential processing for stability
    )
    
    # Check results
    for symbol in symbols:
        for tf in timeframes:
            if symbol in feature_results and tf in feature_results[symbol]:
                features = feature_results[symbol][tf]
                logger.info(f"Generated {len(features)} rows of features for {symbol} {tf}")
                logger.info(f"Feature columns: {features.columns.tolist()}")
            else:
                logger.error(f"Failed to generate features for {symbol} {tf}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Interactive Brokers data download")
    parser.add_argument("--test", choices=["connection", "download", "pipeline", "all"], 
                        default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "connection" or args.test == "all":
        test_ib_connection()
        
    if args.test == "download" or args.test == "all":
        test_data_download()
        
    if args.test == "pipeline" or args.test == "all":
        test_full_pipeline() 