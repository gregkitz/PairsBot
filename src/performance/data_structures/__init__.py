"""
Efficient Data Structures for the Intraday Statistical Arbitrage System.

This module provides optimized data structures for efficient storage and processing,
reducing memory usage and improving computational performance.
"""

from .sparse_time_series import SparseTimeSeries
from .compressed_ohlc import CompressedOHLC
from .rolling_window import RollingWindow, ExponentialRollingWindow

__all__ = [
    'SparseTimeSeries',
    'CompressedOHLC',
    'RollingWindow',
    'ExponentialRollingWindow'
] 