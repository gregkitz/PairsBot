"""
Model Factory for Spread Prediction.

This module provides functions to create various types of machine learning models
for predicting spread movements.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Dictionary of available models
_MODEL_REGISTRY = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso,
    'elastic_net': ElasticNet,
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor,
    'svr': SVR
}

# Default hyperparameters for each model
_DEFAULT_PARAMS = {
    'linear': {},
    'ridge': {'alpha': 1.0},
    'lasso': {'alpha': 0.1},
    'elastic_net': {'alpha': 0.1, 'l1_ratio': 0.5},
    'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
    'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
    'svr': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
}


def available_models() -> List[str]:
    """
    Get a list of available model types.
    
    Returns:
    --------
    list
        List of available model names
    """
    return list(_MODEL_REGISTRY.keys())


def create_model(model_type: str, 
                scale_features: bool = True, 
                **kwargs) -> Pipeline:
    """
    Create a machine learning model for spread prediction.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create (linear, ridge, lasso, elastic_net, random_forest, gradient_boosting, svr)
    scale_features : bool
        Whether to scale features using StandardScaler
    **kwargs : dict
        Additional parameters to pass to the model constructor
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Pipeline containing scaler (if requested) and model
    
    Raises:
    -------
    ValueError
        If the model type is not supported
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_models()}")
    
    # Get default params for the model
    default_params = _DEFAULT_PARAMS.get(model_type, {})
    
    # Update with user-provided params
    model_params = {**default_params, **kwargs}
    
    # Create the model
    model_class = _MODEL_REGISTRY[model_type]
    model = model_class(**model_params)
    
    # Create pipeline with or without scaling
    if scale_features:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        pipeline = Pipeline([
            ('model', model)
        ])
    
    return pipeline 