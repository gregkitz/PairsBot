"""
Parallel Processing Module for the Intraday Statistical Arbitrage System.

This module provides tools for executing tasks in parallel to speed up computation.
"""

from .parallel_executor import ParallelExecutor, parallel_map
from .task_pool import TaskPool

__all__ = ['ParallelExecutor', 'parallel_map', 'TaskPool'] 