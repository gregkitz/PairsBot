"""
Parameter Space definitions for optimization.

This module defines the parameter spaces for various optimization techniques.
"""

import numpy as np
from typing import Dict, List, Union, Any, Tuple


class ParameterSpace:
    """
    Defines the parameter space for optimization.
    
    This class provides methods for defining the search space for parameters
    and for sampling from this space.
    """
    
    def __init__(self):
        """Initialize the parameter space."""
        self.parameters = {}
        self.parameter_types = {}
    
    def add_discrete_parameter(self, name: str, values: List[Any]):
        """
        Add a discrete parameter to the space.
        
        Parameters:
        -----------
        name : str
            Name of the parameter
        values : list
            List of possible values for the parameter
        """
        self.parameters[name] = values
        self.parameter_types[name] = 'discrete'
    
    def add_continuous_parameter(self, name: str, low: float, high: float):
        """
        Add a continuous parameter to the space.
        
        Parameters:
        -----------
        name : str
            Name of the parameter
        low : float
            Lower bound of the parameter
        high : float
            Upper bound of the parameter
        """
        self.parameters[name] = (low, high)
        self.parameter_types[name] = 'continuous'
    
    def add_integer_parameter(self, name: str, low: int, high: int):
        """
        Add an integer parameter to the space.
        
        Parameters:
        -----------
        name : str
            Name of the parameter
        low : int
            Lower bound of the parameter
        high : int
            Upper bound of the parameter
        """
        self.parameters[name] = (low, high)
        self.parameter_types[name] = 'integer'
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample a random configuration from the parameter space.
        
        Returns:
        --------
        dict
            A randomly sampled parameter configuration
        """
        config = {}
        for name, param_def in self.parameters.items():
            param_type = self.parameter_types[name]
            
            if param_type == 'discrete':
                config[name] = np.random.choice(param_def)
            elif param_type == 'continuous':
                low, high = param_def
                config[name] = np.random.uniform(low, high)
            elif param_type == 'integer':
                low, high = param_def
                config[name] = np.random.randint(low, high + 1)
        
        return config
    
    def get_grid_points(self, num_points: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        Generate grid points for grid search.
        
        Parameters:
        -----------
        num_points : dict, optional
            Dictionary mapping parameter names to number of grid points for that parameter.
            If None, uses 5 points for continuous parameters and all values for discrete ones.
        
        Returns:
        --------
        list
            List of parameter configurations for grid search
        """
        if num_points is None:
            num_points = {}
        
        grid_values = {}
        for name, param_def in self.parameters.items():
            param_type = self.parameter_types[name]
            
            if param_type == 'discrete':
                grid_values[name] = param_def
            elif param_type == 'continuous':
                low, high = param_def
                n = num_points.get(name, 5)
                grid_values[name] = np.linspace(low, high, n)
            elif param_type == 'integer':
                low, high = param_def
                n = num_points.get(name, high - low + 1)
                n = min(n, high - low + 1)
                grid_values[name] = np.linspace(low, high, n, dtype=int)
        
        # Generate all combinations
        import itertools
        keys = grid_values.keys()
        grid_points = []
        for values in itertools.product(*[grid_values[key] for key in keys]):
            grid_points.append(dict(zip(keys, values)))
        
        return grid_points


def create_default_parameter_space() -> ParameterSpace:
    """
    Create a default parameter space for the pairs trading strategy.
    
    Returns:
    --------
    ParameterSpace
        Default parameter space
    """
    param_space = ParameterSpace()
    
    # Entry/exit thresholds
    param_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    param_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    
    # Risk parameters
    param_space.add_continuous_parameter('max_risk_per_trade', 0.005, 0.02)
    param_space.add_continuous_parameter('max_allocation', 0.1, 0.2)
    param_space.add_continuous_parameter('correlation_threshold', 0.3, 0.7)
    
    # Time parameters
    param_space.add_integer_parameter('max_holding_period', 60, 240)  # 1-4 hours
    
    # Strategy parameters
    param_space.add_discrete_parameter('use_kalman', [True, False])
    param_space.add_discrete_parameter('z_score_window', [20, 40, 60, 80, 100])
    
    return param_space 