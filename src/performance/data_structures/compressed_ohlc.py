"""
Compressed OHLC Data Structure for the Intraday Statistical Arbitrage System.

This module provides a memory-efficient compressed OHLC (Open, High, Low, Close)
data structure for storing and processing financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import bisect
from collections import OrderedDict


class CompressedOHLC:
    """
    Memory-efficient compressed OHLC (Open, High, Low, Close) data structure.
    
    This class provides a memory-efficient representation of OHLC data by:
    1. Using compact numpy arrays for data storage
    2. Supporting efficient lookups and operations
    3. Allowing conversion to pandas DataFrames when needed
    """
    
    def __init__(self, 
                data: Optional[Union[pd.DataFrame, Dict]] = None,
                compress: bool = True):
        """
        Initialize the compressed OHLC structure.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict, optional
            Initial OHLC data
        compress : bool
            Whether to compress the data in memory
        """
        # Define data types for each column
        self.dtypes = {
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'volume': np.float32
        }
        
        # Initialize data arrays
        self._timestamps = np.array([], dtype='datetime64[ns]')
        self._data = {
            'open': np.array([], dtype=self.dtypes['open']),
            'high': np.array([], dtype=self.dtypes['high']),
            'low': np.array([], dtype=self.dtypes['low']),
            'close': np.array([], dtype=self.dtypes['close']),
            'volume': np.array([], dtype=self.dtypes['volume'])
        }
        
        # Additional columns (not OHLCV)
        self._extra_columns = {}
        
        # For lookups
        self._timestamp_map = {}
        
        # Whether to use compression
        self.compress = compress
        
        # Add initial data if provided
        if data is not None:
            self.add_data(data)
    
    def add_data(self, data: Union[pd.DataFrame, Dict]) -> None:
        """
        Add OHLCV data to the structure.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            OHLCV data to add
        """
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to arrays
            if data.index.dtype.kind == 'M':
                # DataFrame has datetime index
                timestamps = data.index.values
                
                # Get OHLCV data
                ohlcv = {}
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        ohlcv[col] = data[col].values.astype(self.dtypes[col])
                    else:
                        # Use close if other values not available
                        if col != 'volume' and 'close' in data.columns:
                            ohlcv[col] = data['close'].values.astype(self.dtypes[col])
                        else:
                            # Create zeros array
                            ohlcv[col] = np.zeros(len(data), dtype=self.dtypes[col])
                
                # Get extra columns
                extra_columns = {}
                for col in data.columns:
                    if col.lower() not in ['open', 'high', 'low', 'close', 'volume']:
                        extra_columns[col] = data[col].values
                
                # Add the data
                self._add_arrays(timestamps, ohlcv, extra_columns)
            
            else:
                # DataFrame doesn't have datetime index
                if 'timestamp' not in data.columns and 'date' not in data.columns:
                    raise ValueError("DataFrame must have a datetime index or a 'timestamp'/'date' column")
                
                # Get timestamps
                if 'timestamp' in data.columns:
                    timestamps = pd.to_datetime(data['timestamp']).values
                else:
                    timestamps = pd.to_datetime(data['date']).values
                
                # Convert to DataFrame with datetime index
                df = data.set_index(pd.DatetimeIndex(timestamps))
                
                # Recursive call with datetime index
                self.add_data(df)
        
        elif isinstance(data, dict):
            # Convert dictionary to arrays
            if 'timestamp' not in data and 'date' not in data:
                raise ValueError("Dictionary must have a 'timestamp' or 'date' key")
            
            # Get timestamps
            if 'timestamp' in data:
                timestamps = pd.to_datetime(data['timestamp']).values
            else:
                timestamps = pd.to_datetime(data['date']).values
            
            # Get OHLCV data
            ohlcv = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data:
                    ohlcv[col] = np.array(data[col], dtype=self.dtypes[col])
                else:
                    # Use close if other values not available
                    if col != 'volume' and 'close' in data:
                        ohlcv[col] = np.array(data['close'], dtype=self.dtypes[col])
                    else:
                        # Create zeros array
                        ohlcv[col] = np.zeros(len(timestamps), dtype=self.dtypes[col])
            
            # Get extra columns
            extra_columns = {}
            for key, value in data.items():
                if key.lower() not in ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume']:
                    extra_columns[key] = np.array(value)
            
            # Add the data
            self._add_arrays(timestamps, ohlcv, extra_columns)
        
        else:
            raise TypeError("Data must be a pandas DataFrame or dictionary")
    
    def _add_arrays(self, 
                   timestamps: np.ndarray, 
                   ohlcv: Dict[str, np.ndarray], 
                   extra_columns: Dict[str, np.ndarray]) -> None:
        """
        Add array data to the structure.
        
        Parameters:
        -----------
        timestamps : numpy.ndarray
            Array of timestamps
        ohlcv : dict
            Dictionary of OHLCV arrays
        extra_columns : dict
            Dictionary of extra column arrays
        """
        # Get current length
        current_length = len(self._timestamps)
        
        # Combine timestamps
        self._timestamps = np.concatenate([self._timestamps, timestamps])
        
        # Combine OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in ohlcv:
                self._data[col] = np.concatenate([self._data[col], ohlcv[col]])
        
        # Combine extra columns
        for col, values in extra_columns.items():
            if col not in self._extra_columns:
                # Create new column with zeros for existing rows
                self._extra_columns[col] = np.zeros(current_length, dtype=values.dtype)
            
            # Concatenate values
            self._extra_columns[col] = np.concatenate([self._extra_columns[col], values])
        
        # Update timestamp mapping for lookups
        self._update_timestamp_map()
        
        # Sort data by timestamp
        self._sort_data()
        
        # Compress data if enabled
        if self.compress:
            self._compress_data()
    
    def _update_timestamp_map(self) -> None:
        """Update timestamp mapping for lookups."""
        self._timestamp_map = {pd.Timestamp(ts): i for i, ts in enumerate(self._timestamps)}
    
    def _sort_data(self) -> None:
        """Sort data by timestamp."""
        # Get sort indices
        sort_idx = np.argsort(self._timestamps)
        
        # Sort timestamps
        self._timestamps = self._timestamps[sort_idx]
        
        # Sort OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self._data[col] = self._data[col][sort_idx]
        
        # Sort extra columns
        for col in self._extra_columns:
            self._extra_columns[col] = self._extra_columns[col][sort_idx]
        
        # Update timestamp mapping
        self._update_timestamp_map()
    
    def _compress_data(self) -> None:
        """Compress data to reduce memory usage."""
        # Nothing to do for now - arrays are already using efficient dtypes
        # This method can be extended with more advanced compression techniques
        pass
    
    def get_at_index(self, idx: int) -> Dict[str, Any]:
        """
        Get data at a specific index.
        
        Parameters:
        -----------
        idx : int
            Index to get data for
        
        Returns:
        --------
        dict
            Dictionary with OHLCV data
        """
        if idx < 0 or idx >= len(self._timestamps):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Create result dictionary
        result = {
            'timestamp': pd.Timestamp(self._timestamps[idx]),
            'open': self._data['open'][idx],
            'high': self._data['high'][idx],
            'low': self._data['low'][idx],
            'close': self._data['close'][idx],
            'volume': self._data['volume'][idx]
        }
        
        # Add extra columns
        for col, values in self._extra_columns.items():
            result[col] = values[idx]
        
        return result
    
    def get_at_timestamp(self, 
                        timestamp: Union[datetime, pd.Timestamp, str, np.datetime64],
                        method: str = 'exact') -> Optional[Dict[str, Any]]:
        """
        Get data at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime, pandas.Timestamp, str, or numpy.datetime64
            Timestamp to get data for
        method : str
            Method for retrieval ('exact', 'before', 'after', or 'nearest')
        
        Returns:
        --------
        dict or None
            Dictionary with OHLCV data, or None if not found
        """
        # Convert to pandas Timestamp
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, datetime):
            timestamp = pd.Timestamp(timestamp)
        elif isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)
        
        # Check for exact match
        if timestamp in self._timestamp_map:
            idx = self._timestamp_map[timestamp]
            return self.get_at_index(idx)
        
        # Handle different methods if no exact match
        if method == 'exact':
            return None
        
        # Find nearest timestamp index
        timestamps = np.array([pd.Timestamp(ts) for ts in self._timestamps])
        idx = bisect.bisect_left(timestamps, timestamp)
        
        if method == 'before':
            # Get data before timestamp
            if idx > 0:
                return self.get_at_index(idx - 1)
            return None
        
        elif method == 'after':
            # Get data after timestamp
            if idx < len(timestamps):
                return self.get_at_index(idx)
            return None
        
        elif method == 'nearest':
            # Find nearest timestamp
            if idx == 0:
                return self.get_at_index(0)
            elif idx == len(timestamps):
                return self.get_at_index(len(timestamps) - 1)
            else:
                prev_ts = timestamps[idx - 1]
                next_ts = timestamps[idx]
                
                if (timestamp - prev_ts) < (next_ts - timestamp):
                    return self.get_at_index(idx - 1)
                else:
                    return self.get_at_index(idx)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact', 'before', 'after', or 'nearest'")
    
    def get_range(self, 
                start: Optional[Union[datetime, pd.Timestamp, str, np.datetime64]] = None,
                end: Optional[Union[datetime, pd.Timestamp, str, np.datetime64]] = None) -> 'CompressedOHLC':
        """
        Get a subset of data within a time range.
        
        Parameters:
        -----------
        start : datetime, pandas.Timestamp, str, or numpy.datetime64, optional
            Start timestamp (inclusive)
        end : datetime, pandas.Timestamp, str, or numpy.datetime64, optional
            End timestamp (inclusive)
        
        Returns:
        --------
        CompressedOHLC
            New CompressedOHLC object with the subset of data
        """
        # Convert to pandas Timestamp
        if start is not None:
            if isinstance(start, str):
                start = pd.Timestamp(start)
            elif isinstance(start, datetime):
                start = pd.Timestamp(start)
            elif isinstance(start, np.datetime64):
                start = pd.Timestamp(start)
        
        if end is not None:
            if isinstance(end, str):
                end = pd.Timestamp(end)
            elif isinstance(end, datetime):
                end = pd.Timestamp(end)
            elif isinstance(end, np.datetime64):
                end = pd.Timestamp(end)
        
        # Convert timestamps to numpy datetime64
        timestamps = np.array([pd.Timestamp(ts) for ts in self._timestamps])
        
        # Create mask
        if start is not None and end is not None:
            mask = (timestamps >= start) & (timestamps <= end)
        elif start is not None:
            mask = timestamps >= start
        elif end is not None:
            mask = timestamps <= end
        else:
            # No filtering, return copy of self
            return self.copy()
        
        # Filter data
        filtered_timestamps = self._timestamps[mask]
        
        filtered_ohlcv = {}
        for col in ['open', 'high', 'low', 'close', 'volume']:
            filtered_ohlcv[col] = self._data[col][mask]
        
        filtered_extra = {}
        for col, values in self._extra_columns.items():
            filtered_extra[col] = values[mask]
        
        # Create new object
        result = CompressedOHLC(compress=self.compress)
        result._timestamps = filtered_timestamps
        result._data = filtered_ohlcv
        result._extra_columns = filtered_extra
        result._update_timestamp_map()
        
        return result
    
    def resample(self, 
                freq: str, 
                method: str = 'ohlc',
                fill_method: Optional[str] = None) -> 'CompressedOHLC':
        """
        Resample the OHLC data to a regular frequency.
        
        Parameters:
        -----------
        freq : str
            Frequency for resampling (e.g., '1min', '1H', '1D')
        method : str
            Aggregation method ('ohlc', 'vwap')
        fill_method : str, optional
            Method to fill missing values ('ffill', 'bfill', None)
        
        Returns:
        --------
        CompressedOHLC
            New CompressedOHLC object with resampled data
        """
        # Convert to pandas DataFrame
        df = self.to_dataframe()
        
        # Resample
        if method == 'ohlc':
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif method == 'vwap':
            # Calculate volume-weighted average price
            df['value'] = df['close'] * df['volume']
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'value': 'sum',
                'volume': 'sum'
            })
            # Calculate VWAP
            resampled['close'] = resampled['value'] / resampled['volume'].replace(0, np.nan)
            resampled = resampled.drop('value', axis=1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ohlc' or 'vwap'")
        
        # Handle extra columns
        extra_cols = set(df.columns) - {'open', 'high', 'low', 'close', 'volume'}
        for col in extra_cols:
            if col in df.columns:
                resampled[col] = df[col].resample(freq).last()
        
        # Fill missing values if requested
        if fill_method:
            if fill_method == 'ffill':
                resampled = resampled.ffill()
            elif fill_method == 'bfill':
                resampled = resampled.bfill()
            else:
                raise ValueError(f"Unknown fill method: {fill_method}. Use 'ffill' or 'bfill'")
        
        # Create new object
        return CompressedOHLC(resampled, compress=self.compress)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame representation of the OHLC data
        """
        # Create DataFrame
        df = pd.DataFrame({
            'open': self._data['open'],
            'high': self._data['high'],
            'low': self._data['low'],
            'close': self._data['close'],
            'volume': self._data['volume']
        }, index=pd.DatetimeIndex(self._timestamps))
        
        # Add extra columns
        for col, values in self._extra_columns.items():
            df[col] = values
        
        return df
    
    def copy(self) -> 'CompressedOHLC':
        """
        Create a deep copy of the object.
        
        Returns:
        --------
        CompressedOHLC
            Deep copy of the object
        """
        result = CompressedOHLC(compress=self.compress)
        
        # Copy timestamps
        result._timestamps = self._timestamps.copy()
        
        # Copy OHLCV data
        result._data = {col: values.copy() for col, values in self._data.items()}
        
        # Copy extra columns
        result._extra_columns = {col: values.copy() for col, values in self._extra_columns.items()}
        
        # Copy timestamp map
        result._timestamp_map = self._timestamp_map.copy()
        
        return result
    
    def add_column(self, 
                  name: str, 
                  values: Union[List, np.ndarray, pd.Series]) -> None:
        """
        Add a new column to the data.
        
        Parameters:
        -----------
        name : str
            Column name
        values : list, numpy.ndarray, or pandas.Series
            Column values
        """
        # Convert to numpy array
        if isinstance(values, list):
            values = np.array(values)
        elif isinstance(values, pd.Series):
            values = values.values
        
        # Check length
        if len(values) != len(self._timestamps):
            raise ValueError(f"Values length ({len(values)}) does not match timestamps length ({len(self._timestamps)})")
        
        # Add column
        self._extra_columns[name] = values
    
    def calculate_returns(self, 
                         method: str = 'simple',
                         column: str = 'close',
                         periods: int = 1) -> None:
        """
        Calculate returns and add as a new column.
        
        Parameters:
        -----------
        method : str
            Return calculation method ('simple' or 'log')
        column : str
            Column to use for calculation
        periods : int
            Number of periods for return calculation
        """
        # Get values
        if column in self._data:
            values = self._data[column]
        elif column in self._extra_columns:
            values = self._extra_columns[column]
        else:
            raise ValueError(f"Column '{column}' not found")
        
        # Calculate returns
        if method == 'simple':
            returns = np.zeros_like(values)
            returns[periods:] = (values[periods:] / values[:-periods]) - 1
        elif method == 'log':
            returns = np.zeros_like(values)
            returns[periods:] = np.log(values[periods:] / values[:-periods])
        else:
            raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")
        
        # Add as new column
        column_name = f"{column}_{method}_return_{periods}"
        self.add_column(column_name, returns)
    
    def calculate_technical_indicators(self) -> None:
        """
        Calculate common technical indicators and add as new columns.
        """
        try:
            import talib
        except ImportError:
            raise ImportError("Please install talib: pip install TA-Lib")
        
        # Get OHLCV data
        open_prices = self._data['open']
        high_prices = self._data['high']
        low_prices = self._data['low']
        close_prices = self._data['close']
        volumes = self._data['volume']
        
        # Calculate indicators
        
        # Moving averages
        self.add_column('sma_5', talib.SMA(close_prices, timeperiod=5))
        self.add_column('sma_10', talib.SMA(close_prices, timeperiod=10))
        self.add_column('sma_20', talib.SMA(close_prices, timeperiod=20))
        self.add_column('sma_50', talib.SMA(close_prices, timeperiod=50))
        self.add_column('ema_5', talib.EMA(close_prices, timeperiod=5))
        self.add_column('ema_10', talib.EMA(close_prices, timeperiod=10))
        self.add_column('ema_20', talib.EMA(close_prices, timeperiod=20))
        
        # RSI
        self.add_column('rsi_14', talib.RSI(close_prices, timeperiod=14))
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.add_column('macd', macd)
        self.add_column('macd_signal', macd_signal)
        self.add_column('macd_hist', macd_hist)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        self.add_column('bb_upper', upper)
        self.add_column('bb_middle', middle)
        self.add_column('bb_lower', lower)
        
        # ATR
        self.add_column('atr_14', talib.ATR(
            high_prices, low_prices, close_prices, timeperiod=14
        ))
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            high_prices, low_prices, close_prices,
            fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
        )
        self.add_column('stoch_k', slowk)
        self.add_column('stoch_d', slowd)
    
    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self._timestamps)
    
    def __getitem__(self, key: Union[int, slice, datetime, pd.Timestamp, str]) -> Union[Dict[str, Any], 'CompressedOHLC']:
        """
        Get data at a specific index, slice, or timestamp.
        
        Parameters:
        -----------
        key : int, slice, datetime, pandas.Timestamp, or str
            Index, slice, or timestamp to get data for
        
        Returns:
        --------
        dict or CompressedOHLC
            Dictionary with OHLCV data, or CompressedOHLC object with sliced data
        """
        if isinstance(key, int):
            return self.get_at_index(key)
        
        elif isinstance(key, slice):
            # Get start and end indices
            start_idx = key.start if key.start is not None else 0
            end_idx = key.stop if key.stop is not None else len(self._timestamps)
            
            # Create mask
            indices = np.arange(len(self._timestamps))
            mask = (indices >= start_idx) & (indices < end_idx)
            
            # Create new object
            result = CompressedOHLC(compress=self.compress)
            result._timestamps = self._timestamps[mask]
            
            # Copy OHLCV data
            result._data = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                result._data[col] = self._data[col][mask]
            
            # Copy extra columns
            result._extra_columns = {}
            for col, values in self._extra_columns.items():
                result._extra_columns[col] = values[mask]
            
            # Update timestamp map
            result._update_timestamp_map()
            
            return result
        
        elif isinstance(key, (datetime, pd.Timestamp, str, np.datetime64)):
            return self.get_at_timestamp(key)
        
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    @classmethod
    def from_csv(cls, 
                filepath: str, 
                date_format: str = None, 
                date_column: str = 'timestamp',
                compress: bool = True) -> 'CompressedOHLC':
        """
        Create a CompressedOHLC object from a CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        date_format : str, optional
            Format string for parsing dates
        date_column : str
            Column name for the timestamp
        compress : bool
            Whether to compress the data in memory
        
        Returns:
        --------
        CompressedOHLC
            New CompressedOHLC object with data from the CSV file
        """
        # Read CSV
        if date_format:
            df = pd.read_csv(filepath, parse_dates=[date_column], date_format=date_format)
        else:
            df = pd.read_csv(filepath, parse_dates=[date_column])
        
        # Set index
        df = df.set_index(date_column)
        
        # Create object
        return cls(df, compress=compress)
    
    @classmethod
    def from_multiple_csvs(cls, 
                          filepaths: List[str], 
                          date_format: str = None, 
                          date_column: str = 'timestamp',
                          compress: bool = True) -> 'CompressedOHLC':
        """
        Create a CompressedOHLC object from multiple CSV files.
        
        Parameters:
        -----------
        filepaths : list of str
            Paths to the CSV files
        date_format : str, optional
            Format string for parsing dates
        date_column : str
            Column name for the timestamp
        compress : bool
            Whether to compress the data in memory
        
        Returns:
        --------
        CompressedOHLC
            New CompressedOHLC object with data from the CSV files
        """
        # Create empty object
        result = cls(compress=compress)
        
        # Add data from each file
        for filepath in filepaths:
            data = cls.from_csv(filepath, date_format, date_column, compress)
            
            # Add data to result
            result._add_arrays(
                data._timestamps,
                data._data,
                data._extra_columns
            )
        
        return result 