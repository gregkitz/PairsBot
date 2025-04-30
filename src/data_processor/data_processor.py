import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime, timedelta
import re


class DataProcessor:
    """
    Processes and manages data for the pairs trading system.
    
    This class provides functionality to load and preprocess data
    from various sources.
    """
    
    def __init__(self, data_dir=None, time_filter=None):
        """
        Initialize the DataProcessor.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing historical data
        time_filter : callable, optional
            Function to filter data by time
        """
        self.data_dir = data_dir if data_dir else "data/historical"
        self.time_filter = time_filter if time_filter else (lambda x: x)
        self.data_cache = {}
    
    def get_available_tickers(self):
        """
        Get list of available tickers.
        
        Returns:
        --------
        list
            List of ticker symbols
        """
        try:
            # Find all ticker directories
            ticker_dirs = glob.glob(os.path.join(self.data_dir, '*'))
            tickers = [os.path.basename(d) for d in ticker_dirs if os.path.isdir(d)]
            return sorted(tickers)
        except Exception as e:
            logging.error(f"Error getting available tickers: {str(e)}")
            return []
    
    def get_available_timeframes(self, ticker):
        """
        Get list of available timeframes for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
            
        Returns:
        --------
        list
            List of available timeframes
        """
        try:
            # Find all timeframe directories
            timeframe_dirs = glob.glob(os.path.join(self.data_dir, ticker, '*'))
            timeframes = [os.path.basename(d) for d in timeframe_dirs if os.path.isdir(d)]
            return sorted(timeframes)
        except Exception as e:
            logging.error(f"Error getting available timeframes for {ticker}: {str(e)}")
            return []
    
    def load_data(self, ticker, timeframe="1day", start_date=None, end_date=None):
        """
        Load data for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        timeframe : str, optional
            Timeframe to load (e.g., '1min', '1day')
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pandas.DataFrame
            Data for the ticker
        """
        # Check if data is in cache
        cache_key = f"{ticker}_{timeframe}"
        if cache_key in self.data_cache:
            data = self.data_cache[cache_key]
            return self._filter_data(data, start_date, end_date)
        
        # Find the data file
        try:
            data_path = os.path.join(self.data_dir, ticker, timeframe, "data.csv")
            
            if not os.path.exists(data_path):
                logging.warning(f"Data file not found at {data_path}")
                return None
            
            # Load the data
            data = pd.read_csv(data_path)
            
            # Convert date column to datetime
            date_col = next((col for col in data.columns if re.search(r'date|time', col.lower())), None)
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                data.set_index(date_col, inplace=True)
            
            # Check for expected columns
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in expected_columns if col not in map(str.lower, data.columns)]
            
            if missing_columns:
                logging.warning(f"Missing expected columns in {data_path}: {missing_columns}")
                
                # Try to infer missing columns
                if 'close' not in map(str.lower, data.columns):
                    # Look for potential price columns
                    for col in data.columns:
                        if re.search(r'price|close|last', col.lower()):
                            data['close'] = data[col]
                            break
            
            # Convert column names to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Apply time filter
            data = self.time_filter(data)
            
            # Cache the data
            self.data_cache[cache_key] = data
            
            # Return filtered data
            return self._filter_data(data, start_date, end_date)
            
        except Exception as e:
            logging.error(f"Error loading data for {ticker} ({timeframe}): {str(e)}")
            return None
    
    def _filter_data(self, data, start_date=None, end_date=None):
        """
        Filter data by date range.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to filter
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pandas.DataFrame
            Filtered data
        """
        if data is None:
            return None
        
        # Make a copy to avoid modifying the cached data
        filtered_data = data.copy()
        
        # Filter by date range
        if start_date:
            try:
                start = pd.to_datetime(start_date)
                filtered_data = filtered_data[filtered_data.index >= start]
            except Exception as e:
                logging.warning(f"Error parsing start_date '{start_date}': {str(e)}")
        
        if end_date:
            try:
                end = pd.to_datetime(end_date)
                filtered_data = filtered_data[filtered_data.index <= end]
            except Exception as e:
                logging.warning(f"Error parsing end_date '{end_date}': {str(e)}")
        
        return filtered_data
    
    def resample_data(self, data, freq):
        """
        Resample data to a different frequency.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to resample
        freq : str
            Pandas frequency string (e.g., 'D', 'H', '15min')
            
        Returns:
        --------
        pandas.DataFrame
            Resampled data
        """
        if data is None:
            return None
        
        try:
            # Ensure data is indexed by datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                logging.warning("Data not indexed by datetime, cannot resample")
                return data
            
            # Resample OHLCV data
            resampled = pd.DataFrame()
            if 'open' in data.columns:
                resampled['open'] = data['open'].resample(freq).first()
            if 'high' in data.columns:
                resampled['high'] = data['high'].resample(freq).max()
            if 'low' in data.columns:
                resampled['low'] = data['low'].resample(freq).min()
            if 'close' in data.columns:
                resampled['close'] = data['close'].resample(freq).last()
            if 'volume' in data.columns:
                resampled['volume'] = data['volume'].resample(freq).sum()
            
            return resampled
            
        except Exception as e:
            logging.error(f"Error resampling data: {str(e)}")
            return data
    
    def load_multiple_tickers(self, tickers, timeframe="1day", start_date=None, end_date=None, column='close'):
        """
        Load data for multiple tickers and align them.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        timeframe : str, optional
            Timeframe to load (e.g., '1min', '1day')
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
        column : str, optional
            Column to extract (default: 'close')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with aligned data for all tickers
        """
        # Load data for each ticker
        data_dict = {}
        for ticker in tickers:
            ticker_data = self.load_data(ticker, timeframe, start_date, end_date)
            if ticker_data is not None and column in ticker_data.columns:
                data_dict[ticker] = ticker_data[column]
        
        if not data_dict:
            logging.warning("No data loaded for any ticker")
            return None
        
        # Create DataFrame from dictionary
        df = pd.DataFrame(data_dict)
        
        # Handle missing values
        if df.isna().any().any():
            logging.info(f"Handling missing values in multi-ticker data")
            
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Drop rows with remaining missing values
            df = df.dropna()
            
            if df.empty:
                logging.warning("All rows dropped due to missing values")
                return None
        
        return df
    
    def load_pair_data(self, ticker1, ticker2, timeframe="1day", start_date=None, end_date=None):
        """
        Load and align data for a pair of tickers.
        
        Parameters:
        -----------
        ticker1 : str
            First ticker symbol
        ticker2 : str
            Second ticker symbol
        timeframe : str, optional
            Timeframe to load (e.g., '1min', '1day')
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        dict
            Dictionary containing aligned data for the pair
        """
        # Load data for both tickers
        data1 = self.load_data(ticker1, timeframe, start_date, end_date)
        data2 = self.load_data(ticker2, timeframe, start_date, end_date)
        
        if data1 is None or data2 is None:
            logging.warning(f"Could not load data for {ticker1} or {ticker2}")
            return None
        
        # Get common dates
        common_index = data1.index.intersection(data2.index)
        
        if len(common_index) == 0:
            logging.warning(f"No common dates found for {ticker1} and {ticker2}")
            return None
        
        # Filter to common dates
        data1 = data1.loc[common_index]
        data2 = data2.loc[common_index]
        
        # Create result dictionary
        result = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'data1': data1,
            'data2': data2,
            'timeframe': timeframe,
            'start_date': common_index.min(),
            'end_date': common_index.max(),
            'num_points': len(common_index)
        }
        
        return result
    
    def clean_data(self, data):
        """
        Clean data by handling missing values and outliers.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to clean
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned data
        """
        if data is None or data.empty:
            return data
        
        # Make a copy
        cleaned = data.copy()
        
        # Handle missing values
        cleaned = cleaned.fillna(method='ffill').fillna(method='bfill')
        
        # Check for remaining NaNs
        if cleaned.isna().any().any():
            logging.warning("Data contains NaN values after filling")
            cleaned = cleaned.dropna()
        
        # Handle outliers (Z-score method)
        if 'close' in cleaned.columns:
            # Calculate returns
            returns = cleaned['close'].pct_change()
            
            # Calculate z-scores
            mean = returns.mean()
            std = returns.std()
            z_scores = (returns - mean) / std
            
            # Find outliers
            outliers = (z_scores.abs() > 5)
            
            if outliers.any():
                logging.info(f"Found {outliers.sum()} outliers in price data")
                
                # Replace outliers with previous value
                cleaned.loc[outliers, 'close'] = cleaned['close'].shift(1)
        
        return cleaned
    
    def add_features(self, data):
        """
        Add technical features to the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to add features to
            
        Returns:
        --------
        pandas.DataFrame
            Data with added features
        """
        if data is None or data.empty or 'close' not in data.columns:
            return data
        
        # Make a copy
        result = data.copy()
        
        # Add returns
        result['returns'] = result['close'].pct_change()
        
        # Add log returns
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # Add volatility features
        for window in [5, 10, 20, 60]:
            # Rolling volatility
            result[f'volatility_{window}d'] = result['returns'].rolling(window=window).std()
            
            # Rolling mean
            result[f'ma_{window}d'] = result['close'].rolling(window=window).mean()
        
        # Calculate RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        return result
    
    def get_correlation_matrix(self, tickers, timeframe="1day", start_date=None, end_date=None):
        """
        Calculate correlation matrix for multiple tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        timeframe : str, optional
            Timeframe to load (e.g., '1min', '1day')
        start_date : str, optional
            Start date in format 'YYYY-MM-DD'
        end_date : str, optional
            End date in format 'YYYY-MM-DD'
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        # Load data for multiple tickers
        data = self.load_multiple_tickers(tickers, timeframe, start_date, end_date)
        
        if data is None:
            return None
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        return corr_matrix
    
    def save_processed_data(self, data, output_path):
        """
        Save processed data to file.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to save
        output_path : str
            Path to save the data
            
        Returns:
        --------
        bool
            True if save was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Save to CSV
            data.to_csv(output_path)
            logging.info(f"Data saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")
            return False 