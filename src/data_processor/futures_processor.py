"""
Futures Data Processor Module

This module provides functionality to process raw futures data files into a standardized format
for analysis and cointegration testing.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from src.utils.error_handling import (
    BaseError, DataError, DataNotFoundError, DataProcessingError, 
    DataValidationError, log_exception
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class FuturesDataProcessor:
    """
    Process futures data files into a standardized format for analysis.
    """
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize the FuturesDataProcessor with input and output directories.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing raw futures data files
        output_dir : str
            Directory to save processed data files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.processed_files = []
    
    def process_file(self, file_path, symbol=None):
        """
        Process a single futures data file.
        
        Parameters:
        -----------
        file_path : str
            Path to the raw data file
        symbol : str, optional
            Symbol to use for the futures contract. If None, extract from filename.
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
            
        Raises:
        -------
        DataNotFoundError
            If the file doesn't exist or can't be accessed
        DataValidationError
            If the data doesn't meet validation requirements
        DataProcessingError
            If there's an error during processing
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise DataNotFoundError(
                    message=f"File not found: {file_path}",
                    error_code="FILE_NOT_FOUND",
                    source="FuturesDataProcessor.process_file"
                )
                
            logger.info(f"Processing file: {file_path}")
            
            # Extract symbol from filename if not provided
            if symbol is None:
                file_name = os.path.basename(file_path)
                symbol = file_name.split('_')[0]
            
            # Read raw data
            try:
                df = pd.read_csv(file_path, header=None, 
                                names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                raise DataProcessingError(
                    message=f"Error reading CSV file: {file_path}",
                    error_code="CSV_READ_ERROR",
                    source="FuturesDataProcessor.process_file",
                    cause=e,
                    details={"symbol": symbol}
                )
            
            # Basic validation
            if df.empty:
                raise DataValidationError(
                    message=f"Empty dataframe for {file_path}",
                    error_code="EMPTY_DATA",
                    source="FuturesDataProcessor.process_file",
                    details={"symbol": symbol}
                )
                
            try:
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Check for and handle duplicate timestamps
                duplicate_count = df.index.duplicated().sum()
                if duplicate_count > 0:
                    logger.warning(f"Found {duplicate_count} duplicate timestamps in {file_path}, keeping last values")
                    df = df[~df.index.duplicated(keep='last')]
                
                # Check for missing values
                missing_values = df.isnull().sum().sum()
                if missing_values > 0:
                    logger.warning(f"Found {missing_values} missing values in {file_path}")
                    df.fillna(method='ffill', inplace=True)
                
                # Calculate returns
                df['returns'] = df['close'].pct_change()
                
                # Calculate log returns
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Calculate realized volatility (rolling 30-minute standard deviation of returns)
                df['volatility_30min'] = df['returns'].rolling(30).std()
                
                # Additional metadata
                df['date'] = df.index.date
                df['time'] = df.index.time
                df['hour'] = df.index.hour
                df['minute'] = df.index.minute
                df['day_of_week'] = df.index.dayofweek
                
                # Calculate daily high, low, open, close
                daily_data = df.groupby(df.index.date).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                
                # Add daily data back to the dataframe
                for day in daily_data.index:
                    mask = df.index.date == day
                    df.loc[mask, 'daily_open'] = daily_data.loc[day, 'open']
                    df.loc[mask, 'daily_high'] = daily_data.loc[day, 'high']
                    df.loc[mask, 'daily_low'] = daily_data.loc[day, 'low']
                    df.loc[mask, 'daily_close'] = daily_data.loc[day, 'close']
                
                return df
                
            except Exception as e:
                raise DataProcessingError(
                    message=f"Error processing data for {symbol}",
                    error_code="DATA_PROCESSING_ERROR",
                    source="FuturesDataProcessor.process_file",
                    cause=e,
                    details={"symbol": symbol}
                )
                
        except (DataNotFoundError, DataValidationError, DataProcessingError) as e:
            # These are our custom errors, just log and return None
            log_exception(e, logger)
            return None
        except Exception as e:
            # Wrap any other exceptions
            error = DataProcessingError(
                message=f"Unexpected error processing file: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                source="FuturesDataProcessor.process_file",
                cause=e,
                details={"file_path": file_path, "symbol": symbol}
            )
            log_exception(error, logger)
            return None
    
    def save_processed_file(self, df, symbol):
        """
        Save processed DataFrame to a parquet file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed DataFrame
        symbol : str
            Symbol of the futures contract
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if df is None or df.empty:
            logger.warning(f"Cannot save empty DataFrame for {symbol}")
            return None
        
        output_path = os.path.join(self.output_dir, f"{symbol}_processed.parquet")
        df.to_parquet(output_path)
        logger.info(f"Saved processed data to {output_path}")
        
        return output_path
    
    def process_directory(self, max_files=None):
        """
        Process all futures data files in the input directory.
        
        Parameters:
        -----------
        max_files : int, optional
            Maximum number of files to process
            
        Returns:
        --------
        list
            List of processed file paths
        """
        processed_files = []
        
        # Get list of files in input directory
        all_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.txt') and 'continuous' in file:
                    all_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(all_files)} futures data files to process")
        
        # Limit number of files if requested
        if max_files is not None:
            all_files = all_files[:max_files]
            logger.info(f"Limited to processing {max_files} files")
        
        # Process each file
        for file_path in all_files:
            try:
                file_name = os.path.basename(file_path)
                symbol = file_name.split('_')[0]
                
                df = self.process_file(file_path, symbol)
                
                if df is not None:
                    output_path = self.save_processed_file(df, symbol)
                    if output_path:
                        processed_files.append(output_path)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        self.processed_files = processed_files
        logger.info(f"Processed {len(processed_files)} files successfully")
        
        return processed_files
    
    def create_metadata(self):
        """
        Create metadata about processed files.
        
        Returns:
        --------
        dict
            Metadata about processed files
        """
        metadata = {
            'processed_at': datetime.now().isoformat(),
            'total_files': len(self.processed_files),
            'symbols': [os.path.basename(f).split('_')[0] for f in self.processed_files],
            'file_paths': self.processed_files
        }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata


# Example usage
if __name__ == "__main__":
    # Set input and output directories
    input_dir = r"data\raw\futures_full_1min_contin_adj_ratio_wiihg54"
    output_dir = r"data\processed"
    
    # Create processor
    processor = FuturesDataProcessor(input_dir, output_dir)
    
    # Process files
    processed_files = processor.process_directory(max_files=5)
    
    # Create metadata
    metadata = processor.create_metadata()
    
    print(f"Processed {len(processed_files)} files")
    print(f"Processed symbols: {metadata['symbols']}") 