import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the path to import the data_loader
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data_loader import load_data

class DataProcessor:
    """
    Class to load and process data for pairs trading strategy.
    Handles data loading, resampling, and alignment for pairs analysis.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        data_dir : str, optional
            Base directory for data. If None, uses 'data' dir in project root.
        """
        if data_dir is None:
            # Use the project root/data directory
            self.data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
        else:
            self.data_dir = Path(data_dir)
        
        self.historical_dir = self.data_dir / 'historical'
        self.intraday_dir = self.data_dir / '1min'
        
        # Cache for loaded data
        self.data_cache = {}
    
    def load_daily_data(self, tickers, start_date=None, end_date=None):
        """
        Load daily data for multiple tickers and align them to the same date range.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to load
        start_date : str, optional
            Start date for the data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for the data in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with aligned price data, tickers as columns
        """
        all_data = {}
        
        for ticker in tickers:
            # Check cache first
            cache_key = f"{ticker}_daily"
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
            else:
                ticker_path = self.historical_dir / ticker / '1day' / 'data.csv'
                if ticker_path.exists():
                    df = load_data(str(ticker_path))
                    # Rename columns to lowercase if needed
                    df.columns = [col.lower() for col in df.columns]
                    # Use 'close' price for analysis
                    df = df['close'].copy()
                    # Cache the data
                    self.data_cache[cache_key] = df
                else:
                    print(f"Data not found for {ticker} at {ticker_path}")
                    continue
            
            all_data[ticker] = df
        
        # Create a DataFrame from the dictionary of Series
        if not all_data:
            return None
            
        df_aligned = pd.DataFrame(all_data)
        
        # Filter by date range if specified
        if start_date:
            df_aligned = df_aligned.loc[df_aligned.index >= start_date]
        if end_date:
            df_aligned = df_aligned.loc[df_aligned.index <= end_date]
        
        return df_aligned
    
    def load_intraday_data(self, tickers, timeframe='1min', start_date=None, end_date=None):
        """
        Load intraday data for multiple tickers.
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to load
        timeframe : str
            Timeframe to load ('1min', '5min', etc.)
        start_date : str, optional
            Start date for the data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for the data in 'YYYY-MM-DD' format
            
        Returns:
        --------
        dict
            Dictionary of DataFrames with price data, ticker symbols as keys
        """
        all_data = {}
        
        for ticker in tickers:
            # Check cache first
            cache_key = f"{ticker}_{timeframe}"
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
            else:
                # Try to find data in the intraday directory first
                if timeframe == '1min':
                    # Check the 1min directory for continuous adjusted data
                    for subdir in self.intraday_dir.glob('*'):
                        if subdir.is_dir():
                            file_pattern = f"{ticker}_full_1min_continuous_ratio_adjusted.txt"
                            for file_path in subdir.glob(file_pattern):
                                try:
                                    # Load the data - assuming format: datetime,open,high,low,close,volume
                                    df = pd.read_csv(file_path, header=None, 
                                                     names=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                                    df['datetime'] = pd.to_datetime(df['datetime'])
                                    df.set_index('datetime', inplace=True)
                                    
                                    # Cache the data
                                    self.data_cache[cache_key] = df
                                    break
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
                
                # If not found or not 1min, try the historical directory
                if cache_key not in self.data_cache:
                    ticker_path = self.historical_dir / ticker / timeframe
                    if ticker_path.exists():
                        # Try to find data.csv first
                        data_file = ticker_path / 'data.csv'
                        if data_file.exists():
                            df = load_data(str(data_file))
                        else:
                            # Try alternative file naming patterns
                            for file in ticker_path.glob(f"{ticker}_{timeframe}*.csv"):
                                df = load_data(str(file))
                                break
                            else:
                                print(f"No suitable files found for {ticker} at {ticker_path}")
                                continue
                        
                        # Cache the data
                        self.data_cache[cache_key] = df
                    else:
                        print(f"Data directory not found for {ticker} at {ticker_path}")
                        continue
            
            # Get the data from cache (if it was found and loaded)
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
                
                # Filter by date range if specified
                if start_date:
                    df = df.loc[df.index.date >= pd.to_datetime(start_date).date()]
                if end_date:
                    df = df.loc[df.index.date <= pd.to_datetime(end_date).date()]
                
                all_data[ticker] = df
        
        return all_data
    
    def resample_data(self, data, timeframe='1hour'):
        """
        Resample data to a different timeframe.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            Price data to resample, either as DataFrame or dict of DataFrames
        timeframe : str
            Target timeframe ('1hour', '4hour', etc.)
            
        Returns:
        --------
        Same type as input
            Resampled data
        """
        if isinstance(data, pd.DataFrame):
            # Handle DataFrame case (multiple tickers in columns)
            return self._resample_dataframe(data, timeframe)
        elif isinstance(data, dict):
            # Handle dict case (dict of ticker -> DataFrame)
            return {ticker: self._resample_dataframe(df, timeframe) 
                   for ticker, df in data.items()}
        else:
            raise ValueError("Data must be a DataFrame or dict of DataFrames")
    
    def _resample_dataframe(self, df, timeframe):
        """Helper method to resample a single DataFrame"""
        # Parse the timeframe string to get the rule for resampling
        rule = self._parse_timeframe(timeframe)
        
        if 'close' in df.columns:
            # OHLCV data
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        else:
            # Series data or single column
            if isinstance(df, pd.Series):
                resampled = df.resample(rule).last()
            else:
                # Try to resample all columns with 'last' rule
                resampled = df.resample(rule).last()
        
        return resampled
    
    def _parse_timeframe(self, timeframe):
        """Convert timeframe string to pandas resample rule"""
        # Map common timeframes to pandas resample rules
        timeframe_map = {
            '1min': '1T',
            '5min': '5T',
            '10min': '10T',
            '15min': '15T',
            '30min': '30T',
            '1hour': 'H',
            '4hour': '4H',
            '1day': 'D',
            '1week': 'W'
        }
        
        if timeframe in timeframe_map:
            return timeframe_map[timeframe]
        else:
            # Try to parse it directly
            return timeframe
    
    def align_pairs_data(self, data_dict, tickers=None):
        """
        Align data for multiple tickers, ensuring they all have the same timestamps.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary mapping tickers to their DataFrame of price data
        tickers : list, optional
            Specific tickers to align. If None, use all keys in data_dict.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with aligned close prices, tickers as columns
        """
        if tickers is None:
            tickers = list(data_dict.keys())
        
        # Extract close prices for each ticker
        close_series = {}
        for ticker in tickers:
            if ticker not in data_dict:
                print(f"Warning: {ticker} not found in data dictionary")
                continue
                
            df = data_dict[ticker]
            if 'close' in df.columns:
                close_series[ticker] = df['close']
            else:
                # Assume single column or Series
                close_series[ticker] = df if isinstance(df, pd.Series) else df.iloc[:, 0]
        
        # Create a DataFrame with all close prices
        all_prices = pd.DataFrame(close_series)
        
        # Forward fill missing values (within reason)
        all_prices = all_prices.fillna(method='ffill', limit=5)
        
        return all_prices 