"""
Visualization module for generating plots and charts.
"""

from .plotting import (
    plot_pair_prices, 
    plot_spread_zscore, 
    plot_signals_positions, 
    plot_equity_curve, 
    plot_backtest_summary
)

from .cointegration_plots import (
    plot_cointegration_relationship,
    plot_mean_reversion_zones,
    plot_residual_diagnostics,
    plot_rolling_window_analysis,
    plot_structural_breaks,
    create_cointegration_report
)

__all__ = [
    # General plotting functions
    'plot_pair_prices',
    'plot_spread_zscore',
    'plot_signals_positions',
    'plot_equity_curve',
    'plot_backtest_summary',
    
    # Cointegration visualization functions
    'plot_cointegration_relationship',
    'plot_mean_reversion_zones',
    'plot_residual_diagnostics',
    'plot_rolling_window_analysis',
    'plot_structural_breaks',
    'create_cointegration_report'
] 