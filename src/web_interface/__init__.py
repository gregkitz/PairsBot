"""
Web Interface Module for the Intraday Statistical Arbitrage System.

This module provides a web-based interface for monitoring strategy performance,
configuring the system, and controlling strategy execution.
"""

from .app import create_app

__all__ = ['create_app'] 