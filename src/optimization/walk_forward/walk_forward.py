"""
Walk-Forward Testing Framework for Pairs Trading Strategy.

This module implements a walk-forward testing framework that trains on historical
windows and tests on out-of-sample periods to validate strategy robustness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Callable, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from src.pairs_trading_strategy import PairsTradingStrategy
from src.optimization.parameter_space import ParameterSpace, create_default_parameter_space
from src.optimization.grid_search.grid_search import GridSearchOptimizer
from src.optimization.genetic_algorithm.genetic_algorithm import GeneticOptimizer


class WalkForwardTester:
    """
    Walk-Forward Testing Framework for Pairs Trading Strategy.
    
    This class implements a walk-forward testing approach that:
    1. Divides data into multiple training and testing windows
    2. Optimizes parameters on each training window
    3. Tests the optimal parameters on the subsequent out-of-sample testing window
    4. Aggregates results across all testing periods
    
    This approach provides a more realistic assessment of strategy performance
    and helps prevent overfitting to historical data.
    """
    
    def __init__(self,
                optimizer_type: str = 'genetic',
                param_space: Optional[ParameterSpace] = None,
                objective_function: Optional[Callable] = None,
                window_size: int = 6,  # months
                train_size: int = 4,   # months
                test_size: int = 2,    # months
                step_size: int = 2,    # months
                optimizer_config: Optional[Dict[str, Any]] = None,
                n_jobs: int = 1,
                verbose: bool = True):
        """
        Initialize the walk-forward tester.
        
        Parameters:
        -----------
        optimizer_type : str
            Type of optimizer to use ('grid' or 'genetic')
        param_space : ParameterSpace, optional
            Parameter space to optimize. If None, uses default parameter space.
        objective_function : callable, optional
            Function to maximize. If None, uses Sharpe ratio.
        window_size : int
            Size of each window in months (train + test)
        train_size : int
            Size of training period in months
        test_size : int
            Size of testing period in months
        step_size : int
            Step size for moving windows in months
        optimizer_config : dict, optional
            Configuration for the optimizer
        n_jobs : int
            Number of parallel jobs
        verbose : bool
            Whether to print progress
        """
        self.optimizer_type = optimizer_type
        self.param_space = param_space if param_space is not None else create_default_parameter_space()
        self.objective_function = objective_function
        self.window_size = window_size
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.optimizer_config = optimizer_config if optimizer_config is not None else {}
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Store results
        self.window_results = []
        self.optimal_params = []
        self.test_performances = []
        self.aggregated_results = None
    
    def generate_windows(self, start_date: str, end_date: str) -> List[Dict[str, str]]:
        """
        Generate train/test windows for walk-forward testing.
        
        Parameters:
        -----------
        start_date : str
            Start date for the entire testing period in 'YYYY-MM-DD' format
        end_date : str
            End date for the entire testing period in 'YYYY-MM-DD' format
            
        Returns:
        --------
        list
            List of dictionaries containing train_start, train_end, test_start, test_end dates
        """
        # Convert to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        windows = []
        current_start = start_dt
        
        while True:
            # Calculate window boundaries
            train_start = current_start
            train_end = train_start + relativedelta(months=self.train_size) - timedelta(days=1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + relativedelta(months=self.test_size) - timedelta(days=1)
            
            # Check if we've reached the end of the data
            if test_end > end_dt:
                break
            
            window = {
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d')
            }
            
            windows.append(window)
            
            # Move to next window
            current_start = current_start + relativedelta(months=self.step_size)
            
            # Check if we still have enough data for another complete window
            if current_start + relativedelta(months=self.train_size + self.test_size) > end_dt:
                break
        
        return windows
    
    def _create_optimizer(self) -> Union[GridSearchOptimizer, GeneticOptimizer]:
        """
        Create an optimizer instance based on the specified type.
        
        Returns:
        --------
        optimizer
            Either GridSearchOptimizer or GeneticOptimizer
        """
        # Default config for each optimizer type
        grid_defaults = {
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
        
        genetic_defaults = {
            'population_size': 50,
            'n_generations': 10,
            'elite_size': 5,
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'tournament_size': 3,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
        
        # Merge default config with provided config
        if self.optimizer_type.lower() == 'grid':
            config = {**grid_defaults, **self.optimizer_config}
            return GridSearchOptimizer(
                param_space=self.param_space,
                objective_function=self.objective_function,
                **config
            )
        else:  # Default to genetic
            config = {**genetic_defaults, **self.optimizer_config}
            return GeneticOptimizer(
                param_space=self.param_space,
                objective_function=self.objective_function,
                **config
            )
    
    def run(self, 
           pairs: List[Tuple[str, str, float]],
           start_date: str,
           end_date: str,
           timeframe: str = '1hour',
           commission: float = 2.0,
           slippage: float = 1.0,
           account_size: float = 25000) -> pd.DataFrame:
        """
        Run walk-forward testing.
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for the entire testing period in 'YYYY-MM-DD' format
        end_date : str
            End date for the entire testing period in 'YYYY-MM-DD' format
        timeframe : str
            Timeframe for data ('1min', '5min', '1hour', etc.)
        commission : float
            Commission per contract in currency units
        slippage : float
            Slippage per contract in currency units
        account_size : float
            Initial account size
            
        Returns:
        --------
        pandas.DataFrame
            Aggregated performance metrics across all testing periods
        """
        start_time = time.time()
        
        # Generate windows
        windows = self.generate_windows(start_date, end_date)
        
        if self.verbose:
            print(f"Walk-Forward Testing")
            print(f"Optimizer type: {self.optimizer_type}")
            print(f"Number of windows: {len(windows)}")
            print(f"Training period: {self.train_size} months")
            print(f"Testing period: {self.test_size} months")
            print(f"Step size: {self.step_size} months")
            print("=" * 50)
        
        # Process each window
        for i, window in enumerate(windows):
            window_start_time = time.time()
            
            if self.verbose:
                print(f"\nWindow {i+1}/{len(windows)}")
                print(f"Train: {window['train_start']} to {window['train_end']}")
                print(f"Test: {window['test_start']} to {window['test_end']}")
            
            # Create optimizer
            optimizer = self._create_optimizer()
            
            # Optimize on training period
            if self.verbose:
                print("Optimizing parameters on training period...")
            
            best_params = optimizer.optimize(
                pairs=pairs,
                start_date=window['train_start'],
                end_date=window['train_end'],
                timeframe=timeframe,
                commission=commission,
                slippage=slippage,
                account_size=account_size
            )
            
            # Store optimal parameters for this window
            params_with_window = {
                'window': i+1,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                **best_params
            }
            self.optimal_params.append(params_with_window)
            
            # Test on out-of-sample period
            if self.verbose:
                print("Testing optimized parameters on out-of-sample period...")
            
            test_performance = self._test_parameters(
                params=best_params,
                pairs=pairs,
                start_date=window['test_start'],
                end_date=window['test_end'],
                timeframe=timeframe,
                commission=commission,
                slippage=slippage,
                account_size=account_size
            )
            
            # Store test performance for this window
            test_with_window = {
                'window': i+1,
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                **test_performance
            }
            self.test_performances.append(test_with_window)
            
            # Store complete window results
            window_result = {
                'window': i+1,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'best_params': best_params,
                'test_performance': test_performance
            }
            self.window_results.append(window_result)
            
            window_end_time = time.time()
            if self.verbose:
                print(f"Window completed in {window_end_time - window_start_time:.2f} seconds")
                print(f"Test performance: Sharpe={test_performance.get('sharpe_ratio', 'N/A'):.4f}, "
                      f"Return={test_performance.get('total_return', 'N/A'):.4f}")
        
        # Aggregate results
        self.aggregated_results = self._aggregate_results()
        
        end_time = time.time()
        
        if self.verbose:
            print("\nWalk-Forward Testing completed")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Average Sharpe ratio: {self.aggregated_results['avg_sharpe_ratio']:.4f}")
            print(f"Average return: {self.aggregated_results['avg_total_return']:.4f}")
            print(f"Win rate: {self.aggregated_results['win_rate']:.4f}")
        
        return pd.DataFrame(self.test_performances)
    
    def _test_parameters(self, 
                        params: Dict[str, Any],
                        pairs: List[Tuple[str, str, float]],
                        start_date: str,
                        end_date: str,
                        timeframe: str,
                        commission: float,
                        slippage: float,
                        account_size: float) -> Dict[str, Any]:
        """
        Test parameters on out-of-sample period.
        
        Parameters:
        -----------
        params : dict
            Parameter configuration to test
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for testing
        end_date : str
            End date for testing
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
        dict
            Performance metrics
        """
        try:
            # Create and initialize strategy with parameters
            strategy = PairsTradingStrategy(account_size=account_size, **params)
            success = strategy.initialize_strategy(pairs=pairs, start_date=start_date, end_date=end_date, timeframe=timeframe)
            
            if not success:
                return {'error': 'Failed to initialize strategy'}
            
            # Calculate spreads
            strategy.calculate_spreads(use_kalman=params.get('use_kalman', True))
            
            # Generate signals
            strategy.generate_signals()
            
            # Backtest strategy
            results = strategy.backtest_strategy(commission=commission, slippage=slippage)
            
            if results is None or not strategy.performance_metrics:
                return {'error': 'Failed to backtest strategy'}
            
            # Return performance metrics
            return strategy.performance_metrics
        
        except Exception as e:
            if self.verbose:
                print(f"Error testing parameters: {str(e)}")
            return {'error': str(e)}
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all testing periods.
        
        Returns:
        --------
        dict
            Aggregated performance metrics
        """
        if not self.test_performances:
            return {}
        
        # Extract metrics
        sharpe_ratios = [p.get('sharpe_ratio', np.nan) for p in self.test_performances 
                          if 'error' not in p and 'sharpe_ratio' in p]
        total_returns = [p.get('total_return', np.nan) for p in self.test_performances 
                          if 'error' not in p and 'total_return' in p]
        drawdowns = [p.get('max_drawdown', np.nan) for p in self.test_performances 
                      if 'error' not in p and 'max_drawdown' in p]
        winning_windows = sum(1 for r in total_returns if r > 0)
        
        # Calculate aggregated metrics
        aggregated = {
            'avg_sharpe_ratio': np.nanmean(sharpe_ratios) if sharpe_ratios else np.nan,
            'median_sharpe_ratio': np.nanmedian(sharpe_ratios) if sharpe_ratios else np.nan,
            'min_sharpe_ratio': np.nanmin(sharpe_ratios) if sharpe_ratios else np.nan,
            'max_sharpe_ratio': np.nanmax(sharpe_ratios) if sharpe_ratios else np.nan,
            'std_sharpe_ratio': np.nanstd(sharpe_ratios) if len(sharpe_ratios) > 1 else np.nan,
            
            'avg_total_return': np.nanmean(total_returns) if total_returns else np.nan,
            'median_total_return': np.nanmedian(total_returns) if total_returns else np.nan,
            'min_total_return': np.nanmin(total_returns) if total_returns else np.nan,
            'max_total_return': np.nanmax(total_returns) if total_returns else np.nan,
            'std_total_return': np.nanstd(total_returns) if len(total_returns) > 1 else np.nan,
            
            'avg_max_drawdown': np.nanmean(drawdowns) if drawdowns else np.nan,
            'median_max_drawdown': np.nanmedian(drawdowns) if drawdowns else np.nan,
            'min_max_drawdown': np.nanmin(drawdowns) if drawdowns else np.nan,
            'max_max_drawdown': np.nanmax(drawdowns) if drawdowns else np.nan,
            'std_max_drawdown': np.nanstd(drawdowns) if len(drawdowns) > 1 else np.nan,
            
            'win_rate': winning_windows / len(total_returns) if total_returns else np.nan,
            'total_windows': len(self.test_performances),
            'successful_windows': len(sharpe_ratios)
        }
        
        return aggregated
    
    def plot_results(self, save_path=None):
        """
        Plot walk-forward testing results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plots
        """
        if not self.test_performances:
            print("No walk-forward testing results to plot")
            return
        
        # Create DataFrame from test performances
        test_df = pd.DataFrame(self.test_performances)
        
        # Plot metrics across windows
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Sharpe ratio
        axs[0].plot(test_df['window'], test_df['sharpe_ratio'], 'o-', color='blue')
        axs[0].axhline(y=self.aggregated_results['avg_sharpe_ratio'], color='red', linestyle='--', 
                      label=f"Avg: {self.aggregated_results['avg_sharpe_ratio']:.2f}")
        axs[0].set_xlabel('Window')
        axs[0].set_ylabel('Sharpe Ratio')
        axs[0].set_title('Sharpe Ratio Across Windows')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Total return
        axs[1].plot(test_df['window'], test_df['total_return'], 'o-', color='green')
        axs[1].axhline(y=self.aggregated_results['avg_total_return'], color='red', linestyle='--',
                      label=f"Avg: {self.aggregated_results['avg_total_return']:.2f}")
        axs[1].set_xlabel('Window')
        axs[1].set_ylabel('Total Return')
        axs[1].set_title('Total Return Across Windows')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # Max drawdown
        axs[2].plot(test_df['window'], test_df['max_drawdown'], 'o-', color='red')
        axs[2].axhline(y=self.aggregated_results['avg_max_drawdown'], color='blue', linestyle='--',
                      label=f"Avg: {self.aggregated_results['avg_max_drawdown']:.2f}")
        axs[2].set_xlabel('Window')
        axs[2].set_ylabel('Max Drawdown')
        axs[2].set_title('Max Drawdown Across Windows')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'walk_forward_performance.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot parameter stability
        param_df = pd.DataFrame(self.optimal_params)
        param_columns = [col for col in param_df.columns if col not in ['window', 'train_start', 'train_end']]
        
        if param_columns:
            fig, axs = plt.subplots(len(param_columns), 1, figsize=(12, 4 * len(param_columns)))
            
            if len(param_columns) == 1:
                axs = [axs]
            
            for i, param in enumerate(param_columns):
                ax = axs[i]
                ax.plot(param_df['window'], param_df[param], 'o-')
                ax.set_xlabel('Window')
                ax.set_ylabel(param)
                ax.set_title(f'Evolution of {param} Across Windows')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(os.path.join(save_path, 'parameter_stability.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def save_results(self, directory=None):
        """
        Save walk-forward testing results to CSV files.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to save results. If None, uses current directory.
        """
        if not self.test_performances:
            print("No walk-forward testing results to save")
            return
        
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"walk_forward_results_{timestamp}"
            os.makedirs(directory, exist_ok=True)
        
        # Save test performances
        test_df = pd.DataFrame(self.test_performances)
        test_df.to_csv(os.path.join(directory, 'test_performances.csv'), index=False)
        
        # Save optimal parameters
        param_df = pd.DataFrame(self.optimal_params)
        param_df.to_csv(os.path.join(directory, 'optimal_parameters.csv'), index=False)
        
        # Save aggregated results
        agg_df = pd.DataFrame([self.aggregated_results])
        agg_df.to_csv(os.path.join(directory, 'aggregated_results.csv'), index=False)
        
        if self.verbose:
            print(f"Walk-forward testing results saved to {directory}")
    
    def export_strategy_report(self, directory=None):
        """
        Export a comprehensive strategy report.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to save report. If None, uses current directory.
        """
        if not self.test_performances:
            print("No walk-forward testing results for report")
            return
        
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"strategy_report_{timestamp}"
            os.makedirs(directory, exist_ok=True)
        
        # Create plots and save them
        self.plot_results(save_path=directory)
        
        # Save data files
        self.save_results(directory=directory)
        
        # Create HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pairs Trading Strategy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metric-card { background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .metric-name { font-size: 14px; color: #7f8c8d; }
                .metric-container { display: flex; flex-wrap: wrap; justify-content: space-between; }
                img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>Pairs Trading Strategy Walk-Forward Report</h1>
            
            <h2>Summary</h2>
            <div class="metric-container">
        """
        
        # Add summary metrics
        metrics = [
            ('Average Sharpe Ratio', self.aggregated_results['avg_sharpe_ratio']),
            ('Average Return', self.aggregated_results['avg_total_return']),
            ('Win Rate', self.aggregated_results['win_rate']),
            ('Average Max Drawdown', self.aggregated_results['avg_max_drawdown'])
        ]
        
        for name, value in metrics:
            html_content += f"""
                <div class="metric-card">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{value:.4f}</div>
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>Performance Across Windows</h2>
            <img src="walk_forward_performance.png" alt="Performance Metrics">
            
            <h2>Parameter Stability</h2>
            <img src="parameter_stability.png" alt="Parameter Stability">
            
            <h2>Test Performance Details</h2>
            <table>
                <tr>
                    <th>Window</th>
                    <th>Test Period</th>
                    <th>Sharpe Ratio</th>
                    <th>Total Return</th>
                    <th>Max Drawdown</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
        """
        
        # Add row for each test window
        for perf in self.test_performances:
            html_content += f"""
                <tr>
                    <td>{perf.get('window', 'N/A')}</td>
                    <td>{perf.get('test_start', 'N/A')} to {perf.get('test_end', 'N/A')}</td>
                    <td>{perf.get('sharpe_ratio', 'N/A'):.4f}</td>
                    <td>{perf.get('total_return', 'N/A'):.4f}</td>
                    <td>{perf.get('max_drawdown', 'N/A'):.4f}</td>
                    <td>{perf.get('win_rate', 'N/A'):.4f}</td>
                    <td>{perf.get('total_trades', 'N/A')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Optimal Parameters</h2>
            <table>
                <tr>
                    <th>Window</th>
                    <th>Training Period</th>
        """
        
        # Add column for each parameter
        param_keys = set()
        for params in self.optimal_params:
            for key in params.keys():
                if key not in ['window', 'train_start', 'train_end']:
                    param_keys.add(key)
        
        for key in sorted(param_keys):
            html_content += f"<th>{key}</th>"
        
        html_content += """
                </tr>
        """
        
        # Add row for each training window
        for params in self.optimal_params:
            html_content += f"""
                <tr>
                    <td>{params.get('window', 'N/A')}</td>
                    <td>{params.get('train_start', 'N/A')} to {params.get('train_end', 'N/A')}</td>
            """
            
            for key in sorted(param_keys):
                if key in params:
                    value = params[key]
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    html_content += f"<td>{value}</td>"
                else:
                    html_content += "<td>N/A</td>"
            
            html_content += "</tr>"
        
        html_content += """
            </table>
            
            <h2>Aggregated Results</h2>
            <table>
        """
        
        # Add row for each aggregated metric
        for key, value in self.aggregated_results.items():
            html_content += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value if not isinstance(value, float) else value:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(directory, 'strategy_report.html'), 'w') as f:
            f.write(html_content)
        
        if self.verbose:
            print(f"Strategy report generated at {directory}/strategy_report.html")
        
        return os.path.join(directory, 'strategy_report.html')


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
    
    # Extract pairs for testing
    pairs = []
    for _, row in pairs_results.iterrows():
        pairs.append((row['ticker1'], row['ticker2'], row['hedge_ratio']))
    
    # Define parameter space
    param_space = ParameterSpace()
    param_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    param_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    param_space.add_continuous_parameter('max_risk_per_trade', 0.005, 0.02)
    
    # Create walk-forward tester
    wft = WalkForwardTester(
        optimizer_type='genetic',
        param_space=param_space,
        window_size=6,   # 6 months total
        train_size=4,    # 4 months training
        test_size=2,     # 2 months testing
        step_size=2,     # Move 2 months at a time
        optimizer_config={
            'population_size': 20,
            'n_generations': 5,
            'n_jobs': 4
        },
        verbose=True
    )
    
    # Run walk-forward testing
    results = wft.run(
        pairs=pairs[:2],  # Use top 2 pairs
        start_date='2020-01-01',
        end_date='2022-12-31',
        timeframe='1hour',
        account_size=50000
    )
    
    # Plot results
    wft.plot_results()
    
    # Generate strategy report
    wft.export_strategy_report() 