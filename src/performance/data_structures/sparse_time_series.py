"""
Sparse Time Series Data Structure for the Intraday Statistical Arbitrage System.

This module provides a memory-efficient sparse time series implementation,
optimized for financial time series with missing values or irregular intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import bisect
import collections


class SparseTimeSeries:
    """
    Memory-efficient sparse time series optimized for financial data.
    
    This class provides a sparse representation of time series data that:
    1. Uses significantly less memory for series with missing values
    2. Maintains time ordering for efficient lookup and operations
    3. Supports common time series operations and interpolation methods
    """
    
    def __init__(self, 
                data: Optional[Union[Dict, pd.Series, pd.DataFrame]] = None,
                column: Optional[str] = None,
                dtype: np.dtype = np.float64):
        """
        Initialize the sparse time series.
        
        Parameters:
        -----------
        data : dict, pandas.Series, or pandas.DataFrame, optional
            Initial data
        column : str, optional
            Column name if data is a DataFrame
        dtype : np.dtype
            Data type for values
        """
        self.dtype = dtype
        
        # Use OrderedDict to maintain insertion order for timestamps
        self._data = collections.OrderedDict()
        
        # Add initial data if provided
        if data is not None:
            self.add_data(data, column)
    
    def add_data(self, 
                data: Union[Dict, pd.Series, pd.DataFrame],
                column: Optional[str] = None) -> None:
        """
        Add data to the time series.
        
        Parameters:
        -----------
        data : dict, pandas.Series, or pandas.DataFrame
            Data to add
        column : str, optional
            Column name if data is a DataFrame
        """
        if isinstance(data, dict):
            # Handle dictionary input {timestamp: value}
            for ts, value in data.items():
                if not isinstance(ts, (datetime, pd.Timestamp)):
                    # Try to convert to timestamp
                    if isinstance(ts, str):
                        ts = pd.Timestamp(ts)
                    elif isinstance(ts, (int, float)):
                        # Assume Unix timestamp in seconds
                        ts = pd.Timestamp(datetime.fromtimestamp(ts))
                    else:
                        raise ValueError(f"Cannot convert {type(ts)} to timestamp")
                
                self._data[ts] = self.dtype(value)
        
        elif isinstance(data, pd.Series):
            # Handle pandas Series
            for ts, value in data.items():
                if not pd.isna(value):  # Skip NaN values
                    self._data[pd.Timestamp(ts)] = self.dtype(value)
        
        elif isinstance(data, pd.DataFrame):
            # Handle pandas DataFrame with a specified column
            if column is None:
                raise ValueError("Column name must be specified for DataFrame input")
            
            series = data[column]
            for ts, value in series.items():
                if not pd.isna(value):  # Skip NaN values
                    self._data[pd.Timestamp(ts)] = self.dtype(value)
        
        else:
            raise TypeError("Data must be a dict, pandas Series, or pandas DataFrame")
    
    def add_value(self, timestamp: Union[datetime, pd.Timestamp, str], value: Union[float, int, np.number]) -> None:
        """
        Add a single value to the time series.
        
        Parameters:
        -----------
        timestamp : datetime, pandas.Timestamp, or str
            Timestamp for the value
        value : float or int
            Value to add
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        elif not isinstance(timestamp, (datetime, pd.Timestamp)):
            raise TypeError("Timestamp must be a datetime, pandas.Timestamp, or string")
        
        self._data[timestamp] = self.dtype(value)
    
    def get_value(self, 
                timestamp: Union[datetime, pd.Timestamp, str],
                method: str = 'exact') -> Optional[Union[float, np.number]]:
        """
        Get a value at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime, pandas.Timestamp, or str
            Timestamp to get the value for
        method : str
            Method for value retrieval ('exact', 'ffill', 'bfill', or 'nearest')
        
        Returns:
        --------
        float or None
            Value at the specified timestamp, or None if not found
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        # Check for exact match
        if timestamp in self._data:
            return self._data[timestamp]
        
        # Handle different methods if no exact match
        if method == 'exact':
            return None
        
        # Get list of timestamps and find insertion point
        timestamps = list(self._data.keys())
        
        if not timestamps:
            return None
        
        idx = bisect.bisect_left(timestamps, timestamp)
        
        if method == 'ffill':
            # Forward fill (use previous value)
            if idx > 0:
                return self._data[timestamps[idx - 1]]
            return None
        
        elif method == 'bfill':
            # Backward fill (use next value)
            if idx < len(timestamps):
                return self._data[timestamps[idx]]
            return None
        
        elif method == 'nearest':
            # Find nearest value
            if idx == 0:
                return self._data[timestamps[0]]
            elif idx == len(timestamps):
                return self._data[timestamps[-1]]
            else:
                prev_ts = timestamps[idx - 1]
                next_ts = timestamps[idx]
                
                if (timestamp - prev_ts) < (next_ts - timestamp):
                    return self._data[prev_ts]
                else:
                    return self._data[next_ts]
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'exact', 'ffill', 'bfill', or 'nearest'")
    
    def interpolate(self, 
                   timestamp: Union[datetime, pd.Timestamp, str],
                   method: str = 'linear') -> Optional[float]:
        """
        Interpolate a value at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime, pandas.Timestamp, or str
            Timestamp to interpolate the value for
        method : str
            Interpolation method ('linear', 'quadratic', or 'cubic')
        
        Returns:
        --------
        float or None
            Interpolated value at the specified timestamp, or None if interpolation is not possible
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        # Check for exact match
        if timestamp in self._data:
            return self._data[timestamp]
        
        # Get list of timestamps and find insertion point
        timestamps = list(self._data.keys())
        
        if not timestamps:
            return None
        
        idx = bisect.bisect_left(timestamps, timestamp)
        
        # Check if we can interpolate
        if idx == 0 or idx == len(timestamps):
            return None
        
        # Get surrounding points
        prev_ts = timestamps[idx - 1]
        next_ts = timestamps[idx]
        
        prev_val = self._data[prev_ts]
        next_val = self._data[next_ts]
        
        # Convert to numeric for interpolation
        t = (timestamp - prev_ts).total_seconds()
        t_range = (next_ts - prev_ts).total_seconds()
        t_norm = t / t_range
        
        if method == 'linear':
            # Linear interpolation
            return prev_val + t_norm * (next_val - prev_val)
        
        else:
            # For more complex interpolation, we need more points
            # This is a simplification - real quadratic/cubic would need more points
            return prev_val + t_norm * (next_val - prev_val)
    
    def resample(self, 
                freq: str, 
                method: str = 'mean',
                fill_method: Optional[str] = None) -> pd.Series:
        """
        Resample the sparse time series to a regular frequency.
        
        Parameters:
        -----------
        freq : str
            Frequency for resampling (e.g., '1min', '1H', '1D')
        method : str
            Aggregation method ('mean', 'sum', 'min', 'max', etc.)
        fill_method : str, optional
            Method to fill missing values ('ffill', 'bfill', None)
        
        Returns:
        --------
        pandas.Series
            Resampled time series
        """
        # Convert to pandas Series
        series = self.to_series()
        
        # Resample
        resampler = series.resample(freq)
        
        # Apply aggregation method
        if method == 'mean':
            result = resampler.mean()
        elif method == 'sum':
            result = resampler.sum()
        elif method == 'min':
            result = resampler.min()
        elif method == 'max':
            result = resampler.max()
        elif method == 'first':
            result = resampler.first()
        elif method == 'last':
            result = resampler.last()
        elif method == 'count':
            result = resampler.count()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fill missing values if requested
        if fill_method:
            if fill_method == 'ffill':
                result = result.ffill()
            elif fill_method == 'bfill':
                result = result.bfill()
            else:
                raise ValueError(f"Unknown fill method: {fill_method}. Use 'ffill' or 'bfill'")
        
        return result
    
    def to_series(self) -> pd.Series:
        """
        Convert to pandas Series.
        
        Returns:
        --------
        pandas.Series
            Series representation of the sparse time series
        """
        return pd.Series(self._data)
    
    def to_dataframe(self, column_name: str = 'value') -> pd.DataFrame:
        """
        Convert to pandas DataFrame.
        
        Parameters:
        -----------
        column_name : str
            Name for the value column
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame representation of the sparse time series
        """
        return pd.DataFrame({column_name: self._data})
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary.
        
        Returns:
        --------
        dict
            Dictionary representation of the sparse time series
        """
        return dict(self._data)
    
    def apply(self, func: Callable) -> 'SparseTimeSeries':
        """
        Apply a function to each value in the time series.
        
        Parameters:
        -----------
        func : callable
            Function to apply
        
        Returns:
        --------
        SparseTimeSeries
            New sparse time series with transformed values
        """
        result = SparseTimeSeries(dtype=self.dtype)
        
        for ts, value in self._data.items():
            result.add_value(ts, func(value))
        
        return result
    
    def rolling(self, 
               window: int, 
               min_periods: int = None) -> 'SparseTimeSeriesRolling':
        """
        Create a rolling window view of the time series.
        
        Parameters:
        -----------
        window : int
            Size of the rolling window
        min_periods : int, optional
            Minimum number of observations required
        
        Returns:
        --------
        SparseTimeSeriesRolling
            Rolling window view
        """
        if min_periods is None:
            min_periods = window
        
        return SparseTimeSeriesRolling(self, window, min_periods)
    
    def ewm(self, 
           alpha: float = None, 
           span: float = None, 
           halflife: float = None) -> 'SparseTimeSeriesEWM':
        """
        Create an exponentially weighted window view of the time series.
        
        Parameters:
        -----------
        alpha : float, optional
            Smoothing factor
        span : float, optional
            Specify decay in terms of span
        halflife : float, optional
            Specify decay in terms of half-life
        
        Returns:
        --------
        SparseTimeSeriesEWM
            Exponentially weighted window view
        """
        return SparseTimeSeriesEWM(self, alpha=alpha, span=span, halflife=halflife)
    
    def remove(self, timestamp: Union[datetime, pd.Timestamp, str]) -> bool:
        """
        Remove a value at a specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime, pandas.Timestamp, or str
            Timestamp to remove
        
        Returns:
        --------
        bool
            True if the value was removed, False otherwise
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        if timestamp in self._data:
            del self._data[timestamp]
            return True
        
        return False
    
    def filter(self, 
              start_time: Optional[Union[datetime, pd.Timestamp, str]] = None,
              end_time: Optional[Union[datetime, pd.Timestamp, str]] = None) -> 'SparseTimeSeries':
        """
        Filter the time series by time range.
        
        Parameters:
        -----------
        start_time : datetime, pandas.Timestamp, or str, optional
            Start time (inclusive)
        end_time : datetime, pandas.Timestamp, or str, optional
            End time (inclusive)
        
        Returns:
        --------
        SparseTimeSeries
            Filtered sparse time series
        """
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)
        
        result = SparseTimeSeries(dtype=self.dtype)
        
        for ts, value in self._data.items():
            if (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time):
                result.add_value(ts, value)
        
        return result
    
    def __len__(self) -> int:
        """Return the number of values in the time series."""
        return len(self._data)
    
    def __getitem__(self, key: Union[datetime, pd.Timestamp, str]) -> Union[float, np.number]:
        """Get a value by timestamp."""
        return self.get_value(key, method='exact')
    
    def __setitem__(self, key: Union[datetime, pd.Timestamp, str], value: Union[float, int, np.number]) -> None:
        """Set a value by timestamp."""
        self.add_value(key, value)
    
    def __iter__(self):
        """Iterator over timestamp, value pairs."""
        return iter(self._data.items())
    
    def __contains__(self, timestamp: Union[datetime, pd.Timestamp, str]) -> bool:
        """Check if a timestamp exists in the time series."""
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
        
        return timestamp in self._data


class SparseTimeSeriesRolling:
    """Helper class for rolling window operations on SparseTimeSeries."""
    
    def __init__(self, series: SparseTimeSeries, window: int, min_periods: int):
        """
        Initialize the rolling window view.
        
        Parameters:
        -----------
        series : SparseTimeSeries
            The sparse time series
        window : int
            Size of the rolling window
        min_periods : int
            Minimum number of observations required
        """
        self.series = series
        self.window = window
        self.min_periods = min_periods
    
    def mean(self) -> SparseTimeSeries:
        """
        Calculate the rolling mean.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling mean values
        """
        return self._apply_rolling('mean')
    
    def std(self) -> SparseTimeSeries:
        """
        Calculate the rolling standard deviation.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling std values
        """
        return self._apply_rolling('std')
    
    def sum(self) -> SparseTimeSeries:
        """
        Calculate the rolling sum.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling sum values
        """
        return self._apply_rolling('sum')
    
    def min(self) -> SparseTimeSeries:
        """
        Calculate the rolling minimum.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling min values
        """
        return self._apply_rolling('min')
    
    def max(self) -> SparseTimeSeries:
        """
        Calculate the rolling maximum.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling max values
        """
        return self._apply_rolling('max')
    
    def median(self) -> SparseTimeSeries:
        """
        Calculate the rolling median.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling median values
        """
        return self._apply_rolling('median')
    
    def apply(self, func: Callable) -> SparseTimeSeries:
        """
        Apply a custom function to the rolling window.
        
        Parameters:
        -----------
        func : callable
            Function to apply
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with rolling function values
        """
        # Convert to pandas Series for rolling operations
        series = self.series.to_series()
        
        # Apply rolling function
        rolling_result = series.rolling(window=self.window, min_periods=self.min_periods).apply(func)
        
        # Convert back to SparseTimeSeries
        return SparseTimeSeries(rolling_result, dtype=self.series.dtype)
    
    def _apply_rolling(self, method: str) -> SparseTimeSeries:
        """Apply a rolling method to the series."""
        # Convert to pandas Series for rolling operations
        series = self.series.to_series()
        
        # Apply rolling method
        rolling = series.rolling(window=self.window, min_periods=self.min_periods)
        
        if method == 'mean':
            rolling_result = rolling.mean()
        elif method == 'std':
            rolling_result = rolling.std()
        elif method == 'sum':
            rolling_result = rolling.sum()
        elif method == 'min':
            rolling_result = rolling.min()
        elif method == 'max':
            rolling_result = rolling.max()
        elif method == 'median':
            rolling_result = rolling.median()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert back to SparseTimeSeries
        return SparseTimeSeries(rolling_result, dtype=self.series.dtype)


class SparseTimeSeriesEWM:
    """Helper class for exponentially weighted operations on SparseTimeSeries."""
    
    def __init__(self, 
                series: SparseTimeSeries, 
                alpha: float = None, 
                span: float = None,
                halflife: float = None):
        """
        Initialize the EWM view.
        
        Parameters:
        -----------
        series : SparseTimeSeries
            The sparse time series
        alpha : float, optional
            Smoothing factor
        span : float, optional
            Specify decay in terms of span
        halflife : float, optional
            Specify decay in terms of half-life
        """
        self.series = series
        
        # Check parameters
        params = [alpha, span, halflife]
        if sum(p is not None for p in params) != 1:
            raise ValueError("Exactly one of alpha, span, or halflife must be provided")
        
        self.alpha = alpha
        self.span = span
        self.halflife = halflife
    
    def mean(self) -> SparseTimeSeries:
        """
        Calculate the exponentially weighted mean.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with EWM mean values
        """
        return self._apply_ewm('mean')
    
    def std(self) -> SparseTimeSeries:
        """
        Calculate the exponentially weighted standard deviation.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with EWM std values
        """
        return self._apply_ewm('std')
    
    def var(self) -> SparseTimeSeries:
        """
        Calculate the exponentially weighted variance.
        
        Returns:
        --------
        SparseTimeSeries
            Sparse time series with EWM var values
        """
        return self._apply_ewm('var')
    
    def _apply_ewm(self, method: str) -> SparseTimeSeries:
        """Apply an EWM method to the series."""
        # Convert to pandas Series for EWM operations
        series = self.series.to_series()
        
        # Create EWM object with appropriate parameters
        ewm_kwargs = {}
        if self.alpha is not None:
            ewm_kwargs['alpha'] = self.alpha
        elif self.span is not None:
            ewm_kwargs['span'] = self.span
        elif self.halflife is not None:
            ewm_kwargs['halflife'] = self.halflife
        
        ewm = series.ewm(**ewm_kwargs)
        
        # Apply method
        if method == 'mean':
            ewm_result = ewm.mean()
        elif method == 'std':
            ewm_result = ewm.std()
        elif method == 'var':
            ewm_result = ewm.var()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert back to SparseTimeSeries
        return SparseTimeSeries(ewm_result, dtype=self.series.dtype) 