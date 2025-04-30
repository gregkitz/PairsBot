"""
Rolling Window Data Structures for the Intraday Statistical Arbitrage System.

This module provides efficient rolling window implementations for time series analysis,
optimized for performance and memory usage.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Callable, Dict, Any, Tuple
from collections import deque


class RollingWindow:
    """
    Efficient Rolling Window for time series data.
    
    This class provides an efficient implementation of a rolling window
    for time series data, with support for various statistics and custom functions.
    
    It uses an efficient algorithm that updates statistics incrementally
    as new values are added, avoiding redundant calculations.
    """
    
    def __init__(self, window_size: int, dtype: np.dtype = np.float64):
        """
        Initialize the rolling window.
        
        Parameters:
        -----------
        window_size : int
            Size of the rolling window
        dtype : np.dtype
            Data type for the window values
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        self.window_size = window_size
        self.dtype = dtype
        
        # Use deque for efficient window implementation
        self.values = deque(maxlen=window_size)
        
        # Pre-computed statistics
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = None
        self._max = None
        self._count = 0
        
        # Track if statistics are valid
        self._stats_valid = True
    
    def add(self, value: Union[float, int, np.number]) -> None:
        """
        Add a value to the rolling window.
        
        Parameters:
        -----------
        value : float or int
            Value to add to the window
        """
        # Convert to the correct dtype
        value = self.dtype(value)
        
        # If window is full, remove oldest value from statistics
        if len(self.values) == self.window_size:
            old_value = self.values[0]
            self._sum -= old_value
            self._sum_sq -= old_value * old_value
            
            # Min/max needs to be recalculated if removed value is min/max
            if self._min == old_value or self._max == old_value:
                self._stats_valid = False
        
        # Add new value to window
        self.values.append(value)
        
        # Update statistics
        self._sum += value
        self._sum_sq += value * value
        self._count = len(self.values)
        
        # Update min/max
        if not self._stats_valid:
            # Recalculate min/max
            self._min = min(self.values)
            self._max = max(self.values)
            self._stats_valid = True
        else:
            # Incremental update
            if self._min is None or value < self._min:
                self._min = value
            if self._max is None or value > self._max:
                self._max = value
    
    def add_multiple(self, values: Union[List, np.ndarray, pd.Series]) -> None:
        """
        Add multiple values to the rolling window.
        
        Parameters:
        -----------
        values : list, numpy.ndarray, or pandas.Series
            Values to add to the window
        """
        for value in values:
            self.add(value)
    
    def mean(self) -> float:
        """
        Calculate the mean of the rolling window.
        
        Returns:
        --------
        float
            Mean value
        """
        if self._count == 0:
            return np.nan
        
        return self._sum / self._count
    
    def std(self) -> float:
        """
        Calculate the standard deviation of the rolling window.
        
        Returns:
        --------
        float
            Standard deviation
        """
        if self._count <= 1:
            return np.nan
        
        # Avoid precision issues with small numbers
        var = max(0.0, (self._sum_sq - (self._sum * self._sum) / self._count) / (self._count - 1))
        return np.sqrt(var)
    
    def min(self) -> float:
        """
        Get the minimum value in the rolling window.
        
        Returns:
        --------
        float
            Minimum value
        """
        if self._count == 0:
            return np.nan
        
        return self._min
    
    def max(self) -> float:
        """
        Get the maximum value in the rolling window.
        
        Returns:
        --------
        float
            Maximum value
        """
        if self._count == 0:
            return np.nan
        
        return self._max
    
    def median(self) -> float:
        """
        Calculate the median of the rolling window.
        
        Returns:
        --------
        float
            Median value
        """
        if self._count == 0:
            return np.nan
        
        return np.median(self.values)
    
    def percentile(self, q: float) -> float:
        """
        Calculate a percentile of the rolling window.
        
        Parameters:
        -----------
        q : float
            Percentile to calculate (0-100)
        
        Returns:
        --------
        float
            Percentile value
        """
        if self._count == 0:
            return np.nan
        
        return np.percentile(self.values, q)
    
    def var(self) -> float:
        """
        Calculate the variance of the rolling window.
        
        Returns:
        --------
        float
            Variance
        """
        if self._count <= 1:
            return np.nan
        
        return (self._sum_sq - (self._sum * self._sum) / self._count) / (self._count - 1)
    
    def autocorr(self, lag: int = 1) -> float:
        """
        Calculate the autocorrelation of the rolling window.
        
        Parameters:
        -----------
        lag : int
            Lag for autocorrelation
        
        Returns:
        --------
        float
            Autocorrelation value
        """
        if self._count <= lag:
            return np.nan
        
        # Convert to numpy array for easy calculation
        values = np.array(self.values)
        
        # Calculate autocorrelation
        n = len(values)
        mean = self.mean()
        
        # Formula: sum((x_t - mean) * (x_{t-lag} - mean)) / sum((x_t - mean)^2)
        num = sum((values[lag:] - mean) * (values[:n-lag] - mean))
        den = sum((values - mean) ** 2)
        
        if den == 0:
            return np.nan
        
        return num / den
    
    def zscore(self, value: Union[float, int, np.number]) -> float:
        """
        Calculate the z-score of a value relative to the rolling window.
        
        Parameters:
        -----------
        value : float or int
            Value to calculate z-score for
        
        Returns:
        --------
        float
            Z-score
        """
        if self._count == 0:
            return np.nan
        
        std_val = self.std()
        if std_val == 0:
            return np.nan
        
        return (value - self.mean()) / std_val
    
    def apply(self, func: Callable) -> Any:
        """
        Apply a custom function to the rolling window.
        
        Parameters:
        -----------
        func : callable
            Function to apply
        
        Returns:
        --------
        any
            Result of the function
        """
        if self._count == 0:
            return np.nan
        
        return func(list(self.values))
    
    def is_full(self) -> bool:
        """
        Check if the rolling window is full.
        
        Returns:
        --------
        bool
            True if the window is full
        """
        return self._count == self.window_size
    
    def clear(self) -> None:
        """Clear the rolling window."""
        self.values.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = None
        self._max = None
        self._count = 0
        self._stats_valid = True
    
    def to_array(self) -> np.ndarray:
        """
        Convert the rolling window to a numpy array.
        
        Returns:
        --------
        numpy.ndarray
            Array containing the window values
        """
        return np.array(self.values, dtype=self.dtype)
    
    def to_list(self) -> List:
        """
        Convert the rolling window to a list.
        
        Returns:
        --------
        list
            List containing the window values
        """
        return list(self.values)
    
    def stats(self) -> Dict[str, float]:
        """
        Get all statistics of the rolling window.
        
        Returns:
        --------
        dict
            Dictionary with statistics
        """
        return {
            'mean': self.mean(),
            'std': self.std(),
            'min': self.min(),
            'max': self.max(),
            'median': self.median(),
            'var': self.var(),
            'count': self._count
        }
    
    def __len__(self) -> int:
        """Return the number of values in the window."""
        return self._count


class ExponentialRollingWindow:
    """
    Exponential Rolling Window for time series data.
    
    This class provides an implementation of an exponential moving window
    that weighs recent observations more heavily, with exponentially
    decreasing weights for older observations.
    """
    
    def __init__(self, 
                alpha: float = None, 
                span: float = None, 
                com: float = None, 
                halflife: float = None):
        """
        Initialize the exponential rolling window.
        
        Parameters:
        -----------
        alpha : float, optional
            Smoothing factor directly specified (0 < alpha <= 1)
        span : float, optional
            Specify decay in terms of span (span = 2/(alpha - 1))
        com : float, optional
            Specify decay in terms of center of mass (com = 1/alpha - 1)
        halflife : float, optional
            Specify decay in terms of half-life (halflife = log(0.5)/log(1-alpha))
        """
        # Ensure only one parameter is provided
        params = [alpha, span, com, halflife]
        if sum(p is not None for p in params) != 1:
            raise ValueError("Exactly one of alpha, span, com, or halflife must be provided")
        
        # Calculate alpha from other parameters
        if span is not None:
            alpha = 2 / (span + 1)
        elif com is not None:
            alpha = 1 / (1 + com)
        elif halflife is not None:
            alpha = 1 - np.exp(np.log(0.5) / halflife)
        
        # Validate alpha
        if not 0 < alpha <= 1:
            raise ValueError("Invalid alpha value. Must be 0 < alpha <= 1")
        
        self.alpha = alpha
        self._values = []
        self._weights = []
        
        # Pre-computed statistics
        self._mean = None
        self._var = None
        self._count = 0
        self._last_update = None
        
        # EWM state
        self._weighted_sum = 0.0
        self._weighted_sum_sq = 0.0
        self._sum_weights = 0.0
    
    def add(self, value: Union[float, int, np.number]) -> None:
        """
        Add a value to the exponential window.
        
        Parameters:
        -----------
        value : float or int
            Value to add
        """
        # Convert to float
        value = float(value)
        
        # Update state
        if self._count == 0:
            weight = 1.0
        else:
            # Decay all previous weights
            self._sum_weights *= (1 - self.alpha)
            self._weighted_sum *= (1 - self.alpha)
            self._weighted_sum_sq *= (1 - self.alpha)
            
            # Calculate new weight
            weight = self.alpha
        
        # Add new value and weight
        self._values.append(value)
        self._weights.append(weight)
        
        # Update statistics
        self._sum_weights += weight
        self._weighted_sum += weight * value
        self._weighted_sum_sq += weight * value * value
        self._count += 1
        
        # Invalidate cached statistics
        self._mean = None
        self._var = None
        self._last_update = self._count
    
    def add_multiple(self, values: Union[List, np.ndarray, pd.Series]) -> None:
        """
        Add multiple values to the exponential window.
        
        Parameters:
        -----------
        values : list, numpy.ndarray, or pandas.Series
            Values to add
        """
        for value in values:
            self.add(value)
    
    def mean(self) -> float:
        """
        Calculate the exponential weighted mean.
        
        Returns:
        --------
        float
            Exponential weighted mean
        """
        if self._count == 0:
            return np.nan
        
        if self._mean is None or self._last_update != self._count:
            self._mean = self._weighted_sum / self._sum_weights
        
        return self._mean
    
    def var(self) -> float:
        """
        Calculate the exponential weighted variance.
        
        Returns:
        --------
        float
            Exponential weighted variance
        """
        if self._count <= 1:
            return np.nan
        
        if self._var is None or self._last_update != self._count:
            mean = self.mean()
            self._var = self._weighted_sum_sq / self._sum_weights - mean * mean
        
        return self._var
    
    def std(self) -> float:
        """
        Calculate the exponential weighted standard deviation.
        
        Returns:
        --------
        float
            Exponential weighted standard deviation
        """
        var = self.var()
        if np.isnan(var):
            return np.nan
        
        return np.sqrt(var)
    
    def to_dict(self) -> Dict[str, float]:
        """
        Get statistics as a dictionary.
        
        Returns:
        --------
        dict
            Dictionary with statistics
        """
        return {
            'mean': self.mean(),
            'var': self.var(),
            'std': self.std(),
            'count': self._count,
            'alpha': self.alpha
        }
    
    def clear(self) -> None:
        """Clear the exponential window."""
        self._values = []
        self._weights = []
        self._weighted_sum = 0.0
        self._weighted_sum_sq = 0.0
        self._sum_weights = 0.0
        self._count = 0
        self._mean = None
        self._var = None
        self._last_update = None
    
    def __len__(self) -> int:
        """Return the number of values in the window."""
        return self._count 