"""
Parameter Optimization Framework for the Intraday Statistical Arbitrage System.

This module provides tools for optimizing the parameters of the pairs trading strategy:
- Grid Search: Exhaustive search over specified parameter values
- Genetic Algorithm: Evolutionary approach for parameter optimization
- Walk-Forward Testing: Testing strategy using rolling windows
- Intraday Parameter Optimization: Regime-specific parameter optimization for intraday trading
- Adaptive Parameter Management: Dynamic parameter adjustment based on market regimes
"""

__all__ = [
    'grid_search',
    'genetic_algorithm',
    'walk_forward',
    'intraday_parameter_optimizer',
    'adaptive_parameter_manager'
] 