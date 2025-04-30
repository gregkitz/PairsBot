"""
Strategy Factory Module for the Intraday Statistical Arbitrage System.

This module provides a factory function for creating strategy instances
based on configuration parameters.
"""

from typing import Dict, Any, Union
import logging

from src.pairs_trading_strategy import PairsTradingStrategy
from .time_series import TimeSeriesStrategy
from .ml_signals import MLSignalStrategy

# Configure logging
logger = logging.getLogger(__name__)

def create_strategy(strategy_type: str, config: Dict[str, Any]) -> Union[PairsTradingStrategy, TimeSeriesStrategy, MLSignalStrategy]:
    """
    Create a strategy instance based on the specified type and configuration.
    
    Parameters:
    -----------
    strategy_type : str
        The type of strategy to create ('pairs', 'time_series', 'ml_signals', etc.)
    config : Dict[str, Any]
        Configuration dictionary for the strategy
    
    Returns:
    --------
    Union[PairsTradingStrategy, TimeSeriesStrategy, MLSignalStrategy]
        The instantiated strategy
    
    Raises:
    -------
    ValueError
        If the strategy type is not supported
    """
    logger.info(f"Creating strategy of type: {strategy_type}")
    
    if strategy_type == 'pairs' or strategy_type == 'statistical_arbitrage':
        return PairsTradingStrategy(config)
    elif strategy_type == 'time_series':
        return TimeSeriesStrategy(config)
    elif strategy_type == 'ml_signals':
        return MLSignalStrategy(config)
    else:
        supported_types = ['pairs', 'statistical_arbitrage', 'time_series', 'ml_signals']
        raise ValueError(f"Unsupported strategy type: {strategy_type}. Supported types: {supported_types}") 