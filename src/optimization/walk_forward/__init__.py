"""
Walk-Forward Testing framework for pairs trading strategy.

This module provides tools for testing strategies using rolling windows to simulate 
real-world trading where parameters are periodically re-optimized.
"""

from .walk_forward import WalkForwardTester

__all__ = ['WalkForwardTester'] 