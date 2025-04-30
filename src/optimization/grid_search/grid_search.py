"""
Grid Search Optimization for Pairs Trading Strategy.

This module implements a grid search optimizer for finding optimal parameters
for the pairs trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Optional, Tuple
import itertools
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from src.pairs_trading_strategy import PairsTradingStrategy
from src.optimization.parameter_space import ParameterSpace, create_default_parameter_space


class GridSearchOptimizer:
    """
    Grid Search Optimizer for Pairs Trading Strategy.
    
    This class implements a grid search approach to find optimal parameters
    for the pairs trading strategy.
    """
    
    def __init__(self, 
                param_space: Optional[ParameterSpace] = None,
                objective_function: Optional[Callable] = None,
                n_jobs: int = 1,
                verbose: bool = True):
        """
        Initialize the grid search optimizer.
        
        Parameters:
        -----------
        param_space : ParameterSpace, optional
            Parameter space to search. If None, uses default parameter space.
        objective_function : callable, optional
            Function to maximize. If None, uses Sharpe ratio.
        n_jobs : int
            Number of parallel jobs to run
        verbose : bool
            Whether to print progress
        """
        self.param_space = param_space if param_space is not None else create_default_parameter_space()
        self.objective_function = objective_function if objective_function is not None else self._default_objective
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.results = None
        
    def _default_objective(self, strategy: PairsTradingStrategy, **kwargs) -> float:
        """
        Default objective function - Sharpe ratio.
        
        Parameters:
        -----------
        strategy : PairsTradingStrategy
            The strategy instance with results from backtesting
        
        Returns:
        --------
        float
            Sharpe ratio
        """
        # Extract performance metrics from backtest results
        if not hasattr(strategy, 'performance_metrics') or not strategy.performance_metrics:
            return -np.inf
        
        metrics = strategy.performance_metrics
        return metrics.get('sharpe_ratio', -np.inf)
    
    def optimize(self, 
                pairs: List[Tuple[str, str, float]],
                start_date: str,
                end_date: str,
                timeframe: str = '1hour',
                commission: float = 2.0,
                slippage: float = 1.0,
                account_size: float = 25000,
                param_grid: Optional[Dict[str, List[Any]]] = None,
                num_points: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Perform grid search optimization.
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
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
        param_grid : dict, optional
            Dictionary mapping parameter names to list of values to try.
            If None, uses grid from parameter space.
        num_points : dict, optional
            Dictionary mapping parameter names to number of grid points for that parameter.
            Used only if param_grid is None.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with optimization results
        """
        # Get grid points from parameter space if not provided
        if param_grid is None:
            grid_points = self.param_space.get_grid_points(num_points)
        else:
            # Convert param_grid to list of dicts
            keys = param_grid.keys()
            grid_points = []
            for values in itertools.product(*[param_grid[key] for key in keys]):
                grid_points.append(dict(zip(keys, values)))
        
        # Print optimization details
        if self.verbose:
            print(f"Grid Search Optimization")
            print(f"Number of parameter combinations: {len(grid_points)}")
            print(f"Parameters being optimized: {list(grid_points[0].keys())}")
            print(f"Start date: {start_date}, End date: {end_date}")
            print(f"Account size: ${account_size}")
            print(f"Running with {self.n_jobs} parallel jobs")
            print("=" * 50)
        
        start_time = time.time()
        
        if self.n_jobs > 1:
            # Parallel execution
            results = []
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for i, params in enumerate(grid_points):
                    future = executor.submit(
                        self._evaluate_params,
                        params=params,
                        pairs=pairs,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe,
                        commission=commission,
                        slippage=slippage,
                        account_size=account_size
                    )
                    futures.append((future, i, params))
                
                for i, (future, idx, params) in enumerate(futures):
                    try:
                        objective_value = future.result()
                        results.append({**params, 'objective': objective_value})
                        if self.verbose and (i+1) % 10 == 0:
                            print(f"Completed {i+1}/{len(grid_points)} combinations ({(i+1)/len(grid_points)*100:.1f}%)")
                    except Exception as e:
                        print(f"Error evaluating parameter set {idx}: {e}")
                        results.append({**params, 'objective': -np.inf})
        else:
            # Sequential execution
            results = []
            for i, params in enumerate(grid_points):
                objective_value = self._evaluate_params(
                    params=params,
                    pairs=pairs,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    commission=commission,
                    slippage=slippage,
                    account_size=account_size
                )
                results.append({**params, 'objective': objective_value})
                
                if self.verbose and (i+1) % 5 == 0:
                    print(f"Completed {i+1}/{len(grid_points)} combinations ({(i+1)/len(grid_points)*100:.1f}%)")
        
        end_time = time.time()
        
        # Convert to DataFrame and sort by objective value
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('objective', ascending=False).reset_index(drop=True)
        
        if self.verbose:
            print(f"Optimization completed in {end_time - start_time:.2f} seconds")
            print(f"Best objective value: {results_df.iloc[0]['objective']:.4f}")
            print(f"Best parameters: {results_df.iloc[0].drop('objective').to_dict()}")
        
        self.results = results_df
        return results_df
    
    def _evaluate_params(self, 
                        params: Dict[str, Any],
                        pairs: List[Tuple[str, str, float]],
                        start_date: str,
                        end_date: str,
                        timeframe: str,
                        commission: float,
                        slippage: float,
                        account_size: float) -> float:
        """
        Evaluate a single parameter set.
        
        Parameters:
        -----------
        params : dict
            Parameter configuration to evaluate
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        timeframe : str
            Timeframe for data
        commission : float
            Commission per contract
        slippage : float
            Slippage per contract
        account_size : float
            Initial account size
            
        Returns:
        --------
        float
            Objective function value
        """
        try:
            # Create and initialize strategy with parameters
            strategy = PairsTradingStrategy(account_size=account_size, **params)
            success = strategy.initialize_strategy(pairs=pairs, start_date=start_date, end_date=end_date, timeframe=timeframe)
            
            if not success:
                return -np.inf
            
            # Calculate spreads
            strategy.calculate_spreads(use_kalman=params.get('use_kalman', True))
            
            # Generate signals
            strategy.generate_signals()
            
            # Backtest strategy
            results = strategy.backtest_strategy(commission=commission, slippage=slippage)
            
            if results is None or not strategy.performance_metrics:
                return -np.inf
            
            # Evaluate objective
            return self.objective_function(strategy)
        
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating parameters {params}: {str(e)}")
            return -np.inf
    
    def plot_optimization_results(self, param_names=None, top_n=10, save_path=None):
        """
        Plot optimization results.
        
        Parameters:
        -----------
        param_names : list of str, optional
            Names of parameters to include in plots. If None, use all parameters.
        top_n : int
            Number of top configurations to highlight
        save_path : str, optional
            Path to save the plots
        """
        if self.results is None or self.results.empty:
            print("No optimization results to plot")
            return
        
        results = self.results.copy()
        
        if param_names is None:
            param_names = [col for col in results.columns if col != 'objective']
        
        # Scatter plots for each parameter vs objective
        n_params = len(param_names)
        fig, axs = plt.subplots(n_params, 1, figsize=(10, 4 * n_params))
        
        if n_params == 1:
            axs = [axs]
        
        for i, param in enumerate(param_names):
            ax = axs[i]
            ax.scatter(results[param], results['objective'], alpha=0.6)
            
            # Highlight top configurations
            top_configs = results.head(top_n)
            ax.scatter(top_configs[param], top_configs['objective'], color='red', s=100, alpha=0.8)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Objective Value')
            ax.set_title(f'{param} vs Objective')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'optimization_results_scatter.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Bar chart of top configurations
        top_configs = results.head(top_n).copy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create labels for each configuration
        config_labels = []
        for i, row in top_configs.iterrows():
            label = ", ".join([f"{param[:5]}={row[param]:.2f}" if isinstance(row[param], float) 
                              else f"{param[:5]}={row[param]}" 
                              for param in param_names])
            config_labels.append(f"Config {i+1}\n{label}")
        
        ax.bar(config_labels, top_configs['objective'])
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'Top {top_n} Configurations')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'top_configurations.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def save_results(self, filepath=None):
        """
        Save optimization results to CSV.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the results. If None, uses a timestamped filename.
        """
        if self.results is None or self.results.empty:
            print("No optimization results to save")
            return
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"optimization_results_{timestamp}.csv"
        
        self.results.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"Optimization results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data_processor.data_processor import DataProcessor
    from src.cointegration.pair_finder import PairFinder
    
    # Load data and find pairs
    data_processor = DataProcessor()
    pair_finder = PairFinder()
    
    # Define sample futures
    futures = ['ES', 'NQ', 'GC', 'SI', 'CL', 'NG']
    
    # Load data
    daily_data = data_processor.load_daily_data(futures, '2020-01-01', '2022-12-31')
    
    # Find pairs
    pairs_results = pair_finder.find_pairs(daily_data)
    
    # Extract pairs for optimization
    pairs = []
    for _, row in pairs_results.iterrows():
        pairs.append((row['ticker1'], row['ticker2'], row['hedge_ratio']))
    
    # Define optimization space
    param_space = ParameterSpace()
    param_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    param_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    param_space.add_continuous_parameter('max_risk_per_trade', 0.005, 0.02)
    
    # Create optimizer
    optimizer = GridSearchOptimizer(param_space=param_space, n_jobs=4)
    
    # Run optimization
    results = optimizer.optimize(
        pairs=pairs[:3],  # Use top 3 pairs
        start_date='2021-01-01',
        end_date='2022-12-31',
        timeframe='1hour',
        account_size=50000
    )
    
    # Plot results
    optimizer.plot_optimization_results()
    
    # Save results
    optimizer.save_results() 