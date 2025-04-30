"""
Strategies module for the Intraday Statistical Arbitrage System.

This module provides various trading strategies for the system.
"""

from .base import Strategy
from .multi_pair_portfolio import MultiPairPortfolio

__all__ = ['Strategy', 'MultiPairPortfolio'] 