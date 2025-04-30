"""
Reporting Module for the Intraday Statistical Arbitrage System.

This module provides tools for generating comprehensive HTML reports
of backtest results with visualizations and performance metrics.
"""

from .report_generator import BacktestReportGenerator
from .metrics import calculate_performance_metrics

__all__ = ['BacktestReportGenerator', 'calculate_performance_metrics'] 