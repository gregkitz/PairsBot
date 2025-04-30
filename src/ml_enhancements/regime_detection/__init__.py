"""
Regime Detection for Pairs Trading.

This module provides tools to detect breakdowns in cointegration relationships
and regime shifts in the spread behavior.
"""

from .regime_detector import RegimeDetector
from .market_regime_classifier import MarketRegimeClassifier

__all__ = ['RegimeDetector', 'MarketRegimeClassifier'] 