"""
Script to process more futures data files.
"""

import os
import sys
import logging
from src.data_processor.futures_processor import FuturesDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Set input and output directories
    input_dir = r"data\raw\futures_full_1min_contin_adj_ratio_wiihg54"
    output_dir = r"data\processed"
    
    # List of interesting futures to process
    target_futures = [
        # Equity indices
        "ES",   # S&P 500 E-mini
        "NQ",   # Nasdaq E-mini
        "YM",   # Dow Jones E-mini
        "RTY",  # Russell 2000 E-mini
        
        # Fixed Income
        "ZN",   # 10-Year Treasury Note
        "ZF",   # 5-Year Treasury Note
        "ZT",   # 2-Year Treasury Note
        "ZB",   # 30-Year Treasury Bond
        
        # Currencies
        "6E",   # Euro FX
        "6J",   # Japanese Yen
        "6B",   # British Pound
        "6C",   # Canadian Dollar
        "6A",   # Australian Dollar
        
        # Metals
        "GC",   # Gold
        "SI",   # Silver
        "HG",   # Copper
        "PL",   # Platinum
        
        # Energy
        "CL",   # Crude Oil
        "NG",   # Natural Gas
        "RB",   # RBOB Gasoline
        "HO",   # Heating Oil
        
        # Agricultural
        "ZC",   # Corn
        "ZW",   # Wheat
        "ZS",   # Soybeans
        "KE",   # KC Wheat
    ]
    
    # Get all available files
    available_files = []
    for file in os.listdir(input_dir):
        if file.endswith('_full_1min_continuous_ratio_adjusted.txt'):
            symbol = file.split('_')[0]
            available_files.append((symbol, os.path.join(input_dir, file)))
    
    # Filter to target futures
    files_to_process = []
    for symbol, file_path in available_files:
        if symbol in target_futures:
            files_to_process.append((symbol, file_path))
    
    logger.info(f"Found {len(files_to_process)} target futures files to process")
    
    # Create processor
    processor = FuturesDataProcessor(input_dir, output_dir)
    
    # Process each file individually
    processed_files = []
    for symbol, file_path in files_to_process:
        try:
            logger.info(f"Processing {symbol} data...")
            df = processor.process_file(file_path, symbol)
            if df is not None:
                output_path = processor.save_processed_file(df, symbol)
                if output_path:
                    processed_files.append(output_path)
                    logger.info(f"Successfully processed {symbol}")
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    # Store processed files list in the processor
    processor.processed_files = processed_files
    
    # Create metadata
    metadata = processor.create_metadata()
    
    logger.info(f"Processed {len(processed_files)} files successfully")
    logger.info(f"Processed symbols: {metadata['symbols']}")

if __name__ == "__main__":
    main() 