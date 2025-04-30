"""
Paper Trading Module for the Intraday Statistical Arbitrage System.

This module provides paper trading capabilities for testing strategies
using real-time market data but simulated executions.
"""

from .paper_trader import PaperTrader

__all__ = ['PaperTrader'] 