"""
Components package for the paper trading module.

This package contains modular components extracted from the monolithic
IntradayMLPaperTrader class to improve maintainability and testability.
"""

from .signal_enhancer import SignalEnhancer
from .position_manager import PositionManager
from .performance_tracker import PerformanceTracker
from .dashboard_generator import DashboardGenerator
from .alert_system import AlertSystem

__all__ = [
    'SignalEnhancer',
    'PositionManager',
    'PerformanceTracker',
    'DashboardGenerator',
    'AlertSystem'
] 