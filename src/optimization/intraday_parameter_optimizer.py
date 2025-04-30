"""
Intraday Parameter Optimizer Module

This module extends the grid search optimizer to create regime-specific 
parameter sets for intraday trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
import itertools
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import logging

from src.optimization.grid_search.grid_search import GridSearchOptimizer
from src.optimization.parameter_space import ParameterSpace, create_default_parameter_space
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from src.backtest.intraday_backtest_engine import IntradayBacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class IntradayParameterOptimizer:
    """
    Intraday Parameter Optimizer with regime-specific parameter optimization.
    
    This class extends GridSearchOptimizer to create regime-specific parameter
    sets for intraday trading strategies.
    """
    
    def __init__(self, 
                 param_space: Optional[ParameterSpace] = None,
                 n_regimes: int = 3,
                 objective_function: Optional[Callable] = None,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        Initialize the intraday parameter optimizer.
        
        Parameters:
        -----------
        param_space : ParameterSpace, optional
            Parameter space to search. If None, uses default intraday parameter space.
        n_regimes : int
            Number of regimes to detect and optimize for
        objective_function : callable, optional
            Function to maximize. If None, uses Sharpe ratio.
        n_jobs : int
            Number of parallel jobs to run
        verbose : bool
            Whether to print progress
        """
        self.param_space = param_space if param_space is not None else self._create_intraday_parameter_space()
        self.n_regimes = n_regimes
        self.objective_function = objective_function
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize grid search optimizer
        self.grid_optimizer = GridSearchOptimizer(
            param_space=self.param_space,
            objective_function=self.objective_function,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Initialize regime classifier
        self.regime_classifier = None
        
        # Store results by regime
        self.regime_results = {}
        self.all_results = None
        self.regime_optimal_parameters = {}
    
    def _create_intraday_parameter_space(self) -> ParameterSpace:
        """
        Create default parameter space for intraday trading.
        
        Returns:
        --------
        ParameterSpace
            Default parameter space for intraday trading
        """
        param_space = ParameterSpace()
        
        # Entry/exit thresholds - adjusted for intraday
        param_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
        param_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
        
        # Risk parameters - more conservative for intraday
        param_space.add_continuous_parameter('max_risk_per_trade', 0.003, 0.015)
        param_space.add_continuous_parameter('max_allocation', 0.05, 0.15)
        
        # Time parameters - shorter for intraday
        param_space.add_integer_parameter('max_holding_period', 30, 180)  # 30 min to 3 hours
        
        # Strategy parameters
        param_space.add_discrete_parameter('use_kalman', [True, False])
        param_space.add_discrete_parameter('z_score_window', [10, 20, 30, 40, 50])
        
        # ML enhancement parameters
        param_space.add_discrete_parameter('use_ml_filter', [True, False])
        param_space.add_discrete_parameter('use_ml_timing', [True, False])
        param_space.add_discrete_parameter('confidence_threshold', [0.5, 0.6, 0.7, 0.8])
        
        # Volume and time filters
        param_space.add_discrete_parameter('use_volume_filter', [True, False])
        param_space.add_discrete_parameter('min_volume_percentile', [30, 40, 50, 60])
        param_space.add_discrete_parameter('avoid_first_15min', [True, False])
        
        return param_space
    
    def detect_regimes(self, 
                       prices_df: pd.DataFrame, 
                       lookback_window: int = 60) -> Tuple[MarketRegimeClassifier, pd.Series]:
        """
        Detect market regimes in the price data.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for multiple instruments
        lookback_window : int
            Lookback window for feature calculation
            
        Returns:
        --------
        tuple
            (regime_classifier, regime_series)
        """
        if self.verbose:
            logger.info(f"Detecting market regimes with {self.n_regimes} regimes")
        
        # Create regime classifier
        self.regime_classifier = MarketRegimeClassifier(
            n_regimes=self.n_regimes,
            lookback_window=lookback_window
        )
        
        # Calculate features
        features_df = self.regime_classifier.calculate_features(prices_df)
        
        # Fit classifier
        self.regime_classifier.fit(features_df)
        
        # Predict regimes
        regimes = self.regime_classifier.predict(features_df)
        
        if self.verbose:
            logger.info(f"Detected regimes: {pd.Series(regimes).value_counts().to_dict()}")
        
        return self.regime_classifier, regimes
    
    def optimize_by_regime(self,
                          pairs: List[Tuple[str, str, float]],
                          prices_df: pd.DataFrame,
                          start_date: str,
                          end_date: str,
                          timeframe: str = '1hour',
                          commission: float = 2.0,
                          slippage: float = 1.0,
                          account_size: float = 25000,
                          lookback_window: int = 60,
                          param_grid: Optional[Dict[str, List[Any]]] = None,
                          num_points: Optional[Dict[str, int]] = None) -> Dict[int, pd.DataFrame]:
        """
        Perform parameter optimization for each detected regime.
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        prices_df : pd.DataFrame
            DataFrame with price data for regime detection
        start_date : str
            Start date for backtest in 'YYYY-MM-DD' format
        end_date : str
            End date for backtest in 'YYYY-MM-DD' format
        timeframe : str
            Timeframe for data ('1min', '5min', '1hour', etc.)
        commission : float
            Commission per contract in currency units
        slippage : float
            Slippage per contract in currency units
        account_size : float
            Initial account size
        lookback_window : int
            Lookback window for regime detection
        param_grid : dict, optional
            Dictionary mapping parameter names to list of values to try.
            If None, uses grid from parameter space.
        num_points : dict, optional
            Dictionary mapping parameter names to number of grid points for that parameter.
            Used only if param_grid is None.
        
        Returns:
        --------
        dict
            Dictionary mapping regime indices to DataFrames with optimization results
        """
        # First detect regimes
        _, regimes = self.detect_regimes(prices_df, lookback_window)
        
        # Convert index to pandas DatetimeIndex if it's not already
        if not isinstance(regimes.index, pd.DatetimeIndex):
            logger.warning("Regime index is not DatetimeIndex. Converting to DatetimeIndex.")
            regimes.index = pd.DatetimeIndex(regimes.index)
        
        # Run optimization for each regime
        regime_results = {}
        all_results = []
        
        # Loop through unique regimes
        for regime in regimes.unique():
            if self.verbose:
                logger.info(f"Optimizing parameters for regime {regime}")
            
            # Get dates when this regime was active
            regime_dates = regimes[regimes == regime].index
            
            # If no dates, skip this regime
            if len(regime_dates) == 0:
                logger.warning(f"No dates found for regime {regime}, skipping")
                continue
            
            # Convert to string format for grid search
            regime_start_date = max(regime_dates.min().strftime('%Y-%m-%d'), start_date)
            regime_end_date = min(regime_dates.max().strftime('%Y-%m-%d'), end_date)
            
            # Skip if date range is invalid
            if regime_start_date >= regime_end_date:
                logger.warning(f"Invalid date range for regime {regime}: {regime_start_date} to {regime_end_date}, skipping")
                continue
            
            # Run grid search for this regime
            results_df = self.grid_optimizer.optimize(
                pairs=pairs,
                start_date=regime_start_date,
                end_date=regime_end_date,
                timeframe=timeframe,
                commission=commission,
                slippage=slippage,
                account_size=account_size,
                param_grid=param_grid,
                num_points=num_points
            )
            
            # Store results
            regime_results[regime] = results_df
            
            # Add regime column to results
            results_df = results_df.copy()
            results_df['regime'] = regime
            all_results.append(results_df)
            
            # Store optimal parameters for this regime
            if len(results_df) > 0:
                self.regime_optimal_parameters[regime] = results_df.iloc[0].drop('objective').drop('regime').to_dict()
                
                if self.verbose:
                    logger.info(f"Optimal parameters for regime {regime}:")
                    logger.info(f"Objective value: {results_df.iloc[0]['objective']:.4f}")
                    logger.info(f"Parameters: {self.regime_optimal_parameters[regime]}")
        
        # Combine all results
        if all_results:
            self.all_results = pd.concat(all_results).reset_index(drop=True)
        else:
            self.all_results = pd.DataFrame()
        
        self.regime_results = regime_results
        return regime_results
    
    def get_regime_optimal_parameters(self) -> Dict[int, Dict[str, Any]]:
        """
        Get optimal parameters for each regime.
        
        Returns:
        --------
        dict
            Dictionary mapping regime indices to optimal parameter sets
        """
        if not self.regime_optimal_parameters:
            raise ValueError("No optimization results available. Run optimize_by_regime() first.")
        
        return self.regime_optimal_parameters
    
    def create_adaptive_parameter_config(self) -> Dict[str, Any]:
        """
        Create configuration for adaptive parameters based on regimes.
        
        Returns:
        --------
        dict
            Configuration for adaptive parameters
        """
        if not self.regime_optimal_parameters:
            raise ValueError("No optimization results available. Run optimize_by_regime() first.")
        
        if self.regime_classifier is None:
            raise ValueError("No regime classifier available. Run optimize_by_regime() first.")
        
        # Create regime response mapping
        regime_responses = {}
        
        for regime, params in self.regime_optimal_parameters.items():
            # Get regime description
            regime_desc = self.regime_classifier.get_regime_description(regime)
            
            # Create simplified name based on regime description
            regime_name = "default"
            if "High Volatility" in regime_desc:
                regime_name = "high_volatility"
            elif "Low Volatility" in regime_desc:
                regime_name = "low_volatility"
            elif "Strong Trend" in regime_desc:
                regime_name = "trending"
            elif "Weak Trend" in regime_desc and "High Correlation" in regime_desc:
                regime_name = "mean_reverting"
            elif "Low Correlation" in regime_desc:
                regime_name = "low_correlation"
            
            # Extract key parameters
            entry_threshold = params.get('entry_threshold', 2.0)
            exit_threshold = params.get('exit_threshold', 0.5)
            max_risk = params.get('max_risk_per_trade', 0.01)
            max_allocation = params.get('max_allocation', 0.1)
            
            # Create regime response
            regime_responses[regime_name] = {
                "entry_zscore": entry_threshold,
                "exit_zscore": exit_threshold,
                "stop_loss_std": entry_threshold * 1.2,  # Scale based on entry
                "position_size_factor": max_allocation / 0.15,  # Normalize to 0-1 range
                "max_risk_per_trade": max_risk,
                "regime_description": regime_desc,
                "optimal_parameters": params
            }
        
        # Create adaptive parameter configuration
        adaptive_config = {
            "regime_detection": {
                "n_regimes": self.n_regimes,
                "update_frequency": 60  # minutes
            },
            "regime_responses": regime_responses,
            "update_frequency": {
                "regime_check_minutes": 15,
                "parameter_update_minutes": 30,
                "correlation_check_minutes": 10
            },
            "fallback_parameters": self._get_fallback_parameters()
        }
        
        return adaptive_config
    
    def _get_fallback_parameters(self) -> Dict[str, Any]:
        """
        Get fallback parameters based on the average of all regime parameters.
        
        Returns:
        --------
        dict
            Fallback parameters
        """
        if not self.regime_optimal_parameters:
            # Return some reasonable defaults
            return {
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "stop_loss_std": 2.5,
                "position_size_factor": 0.8,
                "max_risk_per_trade": 0.01,
                "regime_description": "Default Fallback"
            }
        
        # Calculate average of numeric parameters
        avg_params = {}
        count = {}
        
        for params in self.regime_optimal_parameters.values():
            for key, value in params.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    avg_params[key] = avg_params.get(key, 0) + value
                    count[key] = count.get(key, 0) + 1
        
        # Calculate averages
        for key in avg_params:
            if count[key] > 0:
                avg_params[key] = avg_params[key] / count[key]
        
        # Extract key parameters
        fallback = {
            "entry_zscore": avg_params.get('entry_threshold', 2.0),
            "exit_zscore": avg_params.get('exit_threshold', 0.5),
            "stop_loss_std": avg_params.get('entry_threshold', 2.0) * 1.2,
            "position_size_factor": 0.8,
            "max_risk_per_trade": avg_params.get('max_risk_per_trade', 0.01),
            "regime_description": "Average Fallback"
        }
        
        return fallback
    
    def plot_regime_comparison(self, output_dir: str = "output/optimization") -> str:
        """
        Plot comparison of optimal parameters across regimes.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output plots
            
        Returns:
        --------
        str
            Path to output plot
        """
        if not self.regime_optimal_parameters:
            raise ValueError("No optimization results available. Run optimize_by_regime() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract key parameters for comparison
        regimes = list(self.regime_optimal_parameters.keys())
        
        # Key parameters to compare
        key_params = [
            'entry_threshold', 
            'exit_threshold', 
            'max_risk_per_trade', 
            'max_allocation',
            'max_holding_period'
        ]
        
        # Filter parameters that exist in all regimes
        valid_params = []
        for param in key_params:
            if all(param in self.regime_optimal_parameters[r] for r in regimes):
                valid_params.append(param)
        
        if not valid_params:
            logger.warning("No common parameters found across regimes")
            return None
        
        # Create comparison dataframe
        comparison_data = {}
        for param in valid_params:
            comparison_data[param] = [self.regime_optimal_parameters[r][param] for r in regimes]
        
        comparison_df = pd.DataFrame(comparison_data, index=[f"Regime {r}" for r in regimes])
        
        # Plot comparison
        fig, axes = plt.subplots(len(valid_params), 1, figsize=(10, 3 * len(valid_params)))
        
        if len(valid_params) == 1:
            axes = [axes]
        
        for i, param in enumerate(valid_params):
            ax = axes[i]
            comparison_df[param].plot(kind='bar', ax=ax)
            ax.set_title(f'Optimal {param} by Regime')
            ax.set_ylabel(param)
            ax.grid(axis='y')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"regime_parameter_comparison_{timestamp}.png")
        plt.savefig(output_file)
        plt.close()
        
        logger.info(f"Saved regime parameter comparison to {output_file}")
        
        return output_file


def create_intraday_parameter_space() -> ParameterSpace:
    """
    Create a parameter space for intraday pairs trading strategy.
    
    Returns:
    --------
    ParameterSpace
        Parameter space for intraday trading
    """
    param_space = ParameterSpace()
    
    # Entry/exit thresholds - adjusted for intraday
    param_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    param_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    
    # Risk parameters - more conservative for intraday
    param_space.add_continuous_parameter('max_risk_per_trade', 0.003, 0.015)
    param_space.add_continuous_parameter('max_allocation', 0.05, 0.15)
    
    # Time parameters - shorter for intraday
    param_space.add_integer_parameter('max_holding_period', 30, 180)  # 30 min to 3 hours
    
    # Strategy parameters
    param_space.add_discrete_parameter('use_kalman', [True, False])
    param_space.add_discrete_parameter('z_score_window', [10, 20, 30, 40, 50])
    
    # ML enhancement parameters
    param_space.add_discrete_parameter('use_ml_filter', [True, False])
    param_space.add_discrete_parameter('use_ml_timing', [True, False])
    param_space.add_discrete_parameter('confidence_threshold', [0.5, 0.6, 0.7, 0.8])
    
    # Volume and time filters
    param_space.add_discrete_parameter('use_volume_filter', [True, False])
    param_space.add_discrete_parameter('min_volume_percentile', [30, 40, 50, 60])
    param_space.add_discrete_parameter('avoid_first_15min', [True, False])
    
    return param_space 