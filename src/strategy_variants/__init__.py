"""
Strategy Variants Module for the Intraday Statistical Arbitrage System.

This module provides alternative strategy implementations beyond
the core pairs trading strategy, including time-series models,
alternative signal generation methods, and multi-pair approaches.
"""

from .time_series import TimeSeriesStrategy
from .ml_signals import MLSignalStrategy
from .factory import create_strategy

__all__ = ['TimeSeriesStrategy', 'MLSignalStrategy', 'create_strategy'] 