"""
Spread Prediction Models for Pairs Trading.

This module provides machine learning models to predict future spread movements.
"""

from .spread_predictor import SpreadPredictor
from .model_factory import create_model, available_models

__all__ = ['SpreadPredictor', 'create_model', 'available_models'] 