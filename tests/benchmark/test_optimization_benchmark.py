#!/usr/bin/env python3
"""
Benchmark tests for optimization algorithms.

This module contains benchmarks for grid search and genetic algorithm optimizers,
comparing performance across different parameter spaces and problem sizes.
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Optional, Tuple
import random
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import benchmark utilities
from tests.benchmark.test_benchmark_utils import BenchmarkRunner, run_memory_profile

# Import optimization modules
try:
    from src.optimization.grid_search.grid_search import GridSearchOptimizer
    from src.optimization.genetic_algorithm.genetic_algorithm import GeneticOptimizer
    from src.optimization.parameter_space import ParameterSpace
    from src.optimization.intraday_parameter_optimizer import IntradayParameterOptimizer
except ImportError:
    # Mock classes for testing if actual modules don't exist
    class GridSearchOptimizer:
        def __init__(self, *args, **kwargs):
            pass
            
        def optimize(self, *args, **kwargs):
            # Create dummy results
            results = pd.DataFrame({
                'param1': np.random.rand(10),
                'param2': np.random.randint(1, 5, 10),
                'objective': np.random.rand(10)
            })
            return results
    
    class GeneticOptimizer:
        def __init__(self, *args, **kwargs):
            pass
            
        def optimize(self, *args, **kwargs):
            # Create dummy results
            return {'param1': np.random.rand(), 'param2': np.random.randint(1, 5)}
    
    class ParameterSpace:
        def __init__(self):
            self.parameters = {}
            self.parameter_types = {}
            
        def add_continuous_parameter(self, name, low, high):
            self.parameters[name] = (low, high)
            self.parameter_types[name] = 'continuous'
            
        def add_integer_parameter(self, name, low, high):
            self.parameters[name] = (low, high)
            self.parameter_types[name] = 'integer'
            
        def add_discrete_parameter(self, name, values):
            self.parameters[name] = values
            self.parameter_types[name] = 'discrete'
            
        def get_grid_points(self, num_points=None):
            return [{'param1': 0.5, 'param2': 2}]
            
        def sample(self):
            return {'param1': np.random.rand(), 'param2': np.random.randint(1, 5)}
    
    class IntradayParameterOptimizer:
        def __init__(self, *args, **kwargs):
            pass
            
        def optimize_by_regime(self, *args, **kwargs):
            return {0: pd.DataFrame(), 1: pd.DataFrame()}


def create_synthetic_data(n_days=100, n_assets=5):
    """
    Create synthetic price data for benchmarking.
    
    Parameters:
    -----------
    n_days : int
        Number of days in the dataset
    n_assets : int
        Number of assets to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic price data
    """
    # Create date index
    dates = pd.date_range(start='2022-01-01', periods=n_days)
    
    # Generate random price series with some correlation
    rho = 0.7  # correlation between assets
    
    # Generate correlated random series
    np.random.seed(42)  # For reproducibility
    n = n_days  # number of observations
    
    # Create prices DataFrame
    prices = pd.DataFrame(index=dates)
    
    # Create 'central' series
    central = 100 * (1 + np.random.randn(n).cumsum() * 0.01)
    
    # Create assets with correlation to central series
    for i in range(n_assets):
        # Create correlated series
        asset_return = rho * np.random.randn(n) + (1 - rho) * np.random.randn(n)
        asset_price = 50 + i*10 + (asset_return.cumsum() * (0.5 + i*0.1))
        
        # Add to DataFrame
        prices[f'asset_{i}'] = asset_price
    
    return prices


def create_synthetic_pairs(n_pairs=3):
    """
    Create synthetic pairs for benchmarking.
    
    Parameters:
    -----------
    n_pairs : int
        Number of pairs to generate
        
    Returns:
    --------
    list
        List of (ticker1, ticker2, hedge_ratio) tuples
    """
    pairs = []
    for i in range(n_pairs):
        ticker1 = f"asset_{i}"
        ticker2 = f"asset_{i+1}"
        hedge_ratio = 0.8 + 0.1 * i  # Vary the hedge ratio slightly
        pairs.append((ticker1, ticker2, hedge_ratio))
    
    return pairs


def create_parameter_spaces():
    """
    Create parameter spaces of different sizes for benchmarking.
    
    Returns:
    --------
    dict
        Dictionary mapping parameter space name to ParameterSpace instance
    """
    # Create small parameter space
    small_space = ParameterSpace()
    small_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    small_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    small_space.add_integer_parameter('lookback_period', 10, 50)
    
    # Create medium parameter space
    medium_space = ParameterSpace()
    medium_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    medium_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    medium_space.add_integer_parameter('lookback_period', 10, 50)
    medium_space.add_discrete_parameter('use_kalman', [True, False])
    medium_space.add_discrete_parameter('z_score_window', [10, 20, 30, 40, 50])
    medium_space.add_continuous_parameter('max_risk_per_trade', 0.003, 0.015)
    
    # Create large parameter space
    large_space = ParameterSpace()
    large_space.add_continuous_parameter('entry_threshold', 1.5, 3.0)
    large_space.add_continuous_parameter('exit_threshold', 0.0, 1.0)
    large_space.add_integer_parameter('lookback_period', 10, 50)
    large_space.add_discrete_parameter('use_kalman', [True, False])
    large_space.add_discrete_parameter('z_score_window', [10, 20, 30, 40, 50])
    large_space.add_continuous_parameter('max_risk_per_trade', 0.003, 0.015)
    large_space.add_continuous_parameter('max_allocation', 0.05, 0.15)
    large_space.add_integer_parameter('max_holding_period', 30, 180)
    large_space.add_discrete_parameter('use_ml_filter', [True, False])
    large_space.add_discrete_parameter('use_ml_timing', [True, False])
    large_space.add_discrete_parameter('confidence_threshold', [0.5, 0.6, 0.7, 0.8])
    
    return {
        'small': small_space,
        'medium': medium_space,
        'large': large_space
    }


def benchmark_grid_search():
    """
    Benchmark grid search optimization with different parameter spaces.
    """
    print("Benchmarking Grid Search Optimization")
    
    # Create benchmark runner
    runner = BenchmarkRunner("grid_search_optimization", repetitions=3)
    
    # Create synthetic data
    prices_df = create_synthetic_data(n_days=100, n_assets=5)
    pairs = create_synthetic_pairs(n_pairs=2)
    
    # Get parameter spaces
    parameter_spaces = create_parameter_spaces()
    
    # Set common parameters
    common_params = {
        'pairs': pairs,
        'start_date': '2022-01-01',
        'end_date': '2022-04-10',
        'timeframe': '1hour',
        'commission': 2.0,
        'slippage': 1.0,
        'account_size': 25000
    }
    
    # Run benchmarks for each parameter space
    for space_name, param_space in parameter_spaces.items():
        print(f"Testing {space_name} parameter space")
        
        # Create optimizer
        optimizer = GridSearchOptimizer(param_space=param_space, n_jobs=1, verbose=False)
        
        # Benchmark optimization
        start_time = time.time()
        try:
            results = optimizer.optimize(**common_params)
            elapsed = time.time() - start_time
            print(f"Grid search with {space_name} space completed in {elapsed:.2f} seconds")
            
            # Record result
            runner.record_result({
                'algorithm': 'grid_search',
                'parameter_space': space_name,
                'runtime': elapsed,
                'num_combinations': len(results) if isinstance(results, pd.DataFrame) else 0
            })
        except Exception as e:
            print(f"Error in grid search with {space_name} space: {e}")
    
    # Create memory profile
    try:
        print("Profiling memory usage for large parameter space")
        optimizer = GridSearchOptimizer(param_space=parameter_spaces['large'], n_jobs=1, verbose=False)
        mem_profile = run_memory_profile(optimizer.optimize, **common_params)
        
        runner.record_result({
            'algorithm': 'grid_search',
            'parameter_space': 'large',
            'memory_profile': mem_profile
        })
    except Exception as e:
        print(f"Error in memory profiling: {e}")
    
    # Save results
    runner.save_results()
    
    return runner


def benchmark_genetic_algorithm():
    """
    Benchmark genetic algorithm optimization with different parameter spaces and population sizes.
    """
    print("Benchmarking Genetic Algorithm Optimization")
    
    # Create benchmark runner
    runner = BenchmarkRunner("genetic_optimization", repetitions=3)
    
    # Create synthetic data
    prices_df = create_synthetic_data(n_days=100, n_assets=5)
    pairs = create_synthetic_pairs(n_pairs=2)
    
    # Get parameter spaces
    parameter_spaces = create_parameter_spaces()
    
    # Set common parameters
    common_params = {
        'pairs': pairs,
        'start_date': '2022-01-01',
        'end_date': '2022-04-10',
        'timeframe': '1hour',
        'commission': 2.0,
        'slippage': 1.0,
        'account_size': 25000
    }
    
    # Test population sizes
    population_sizes = [20, 50, 100]
    
    # Run benchmarks for each parameter space and population size
    for space_name, param_space in parameter_spaces.items():
        for pop_size in population_sizes:
            print(f"Testing {space_name} parameter space with population {pop_size}")
            
            # Create optimizer
            optimizer = GeneticOptimizer(
                param_space=param_space, 
                population_size=pop_size,
                n_generations=5,
                n_jobs=1,
                verbose=False
            )
            
            # Benchmark optimization
            start_time = time.time()
            try:
                results = optimizer.optimize(**common_params)
                elapsed = time.time() - start_time
                print(f"Genetic algorithm with {space_name} space and population {pop_size} completed in {elapsed:.2f} seconds")
                
                # Record result
                runner.record_result({
                    'algorithm': 'genetic',
                    'parameter_space': space_name,
                    'population_size': pop_size,
                    'runtime': elapsed
                })
            except Exception as e:
                print(f"Error in genetic algorithm with {space_name} space and population {pop_size}: {e}")
    
    # Create memory profile
    try:
        print("Profiling memory usage for large parameter space and population 50")
        optimizer = GeneticOptimizer(
            param_space=parameter_spaces['large'], 
            population_size=50,
            n_generations=5,
            n_jobs=1,
            verbose=False
        )
        mem_profile = run_memory_profile(optimizer.optimize, **common_params)
        
        runner.record_result({
            'algorithm': 'genetic',
            'parameter_space': 'large',
            'population_size': 50,
            'memory_profile': mem_profile
        })
    except Exception as e:
        print(f"Error in memory profiling: {e}")
    
    # Save results
    runner.save_results()
    
    return runner


def benchmark_regime_optimization():
    """
    Benchmark regime-specific optimization with different numbers of regimes.
    """
    print("Benchmarking Regime-Specific Optimization")
    
    # Create benchmark runner
    runner = BenchmarkRunner("regime_optimization", repetitions=2)
    
    # Create synthetic data
    prices_df = create_synthetic_data(n_days=200, n_assets=5)
    pairs = create_synthetic_pairs(n_pairs=2)
    
    # Set common parameters
    common_params = {
        'pairs': pairs,
        'prices_df': prices_df,
        'start_date': '2022-01-01',
        'end_date': '2022-07-10',
        'timeframe': '1hour',
        'commission': 2.0,
        'slippage': 1.0,
        'account_size': 25000
    }
    
    # Test different numbers of regimes
    n_regimes_list = [2, 3, 5]
    
    # Run benchmarks for each number of regimes
    for n_regimes in n_regimes_list:
        print(f"Testing with {n_regimes} regimes")
        
        # Create optimizer
        optimizer = IntradayParameterOptimizer(
            param_space=None,  # Use default
            n_regimes=n_regimes,
            n_jobs=1,
            verbose=False
        )
        
        # Benchmark optimization
        start_time = time.time()
        try:
            results = optimizer.optimize_by_regime(**common_params)
            elapsed = time.time() - start_time
            print(f"Regime optimization with {n_regimes} regimes completed in {elapsed:.2f} seconds")
            
            # Record result
            runner.record_result({
                'algorithm': 'regime_optimization',
                'n_regimes': n_regimes,
                'runtime': elapsed,
                'num_regimes_found': len(results.keys()) if results else 0
            })
        except Exception as e:
            print(f"Error in regime optimization with {n_regimes} regimes: {e}")
    
    # Create memory profile
    try:
        print("Profiling memory usage for 3 regimes")
        optimizer = IntradayParameterOptimizer(
            param_space=None,  # Use default
            n_regimes=3,
            n_jobs=1,
            verbose=False
        )
        mem_profile = run_memory_profile(optimizer.optimize_by_regime, **common_params)
        
        runner.record_result({
            'algorithm': 'regime_optimization',
            'n_regimes': 3,
            'memory_profile': mem_profile
        })
    except Exception as e:
        print(f"Error in memory profiling: {e}")
    
    # Save results
    runner.save_results()
    
    return runner


def compare_optimization_algorithms():
    """
    Compare grid search and genetic algorithm optimization performance.
    """
    print("Comparing Optimization Algorithms")
    
    # Create benchmark runner
    runner = BenchmarkRunner("optimization_comparison", repetitions=3)
    
    # Create synthetic data
    prices_df = create_synthetic_data(n_days=100, n_assets=5)
    pairs = create_synthetic_pairs(n_pairs=2)
    
    # Use medium parameter space
    param_space = create_parameter_spaces()['medium']
    
    # Set common parameters
    common_params = {
        'pairs': pairs,
        'start_date': '2022-01-01',
        'end_date': '2022-04-10',
        'timeframe': '1hour',
        'commission': 2.0,
        'slippage': 1.0,
        'account_size': 25000
    }
    
    # Create optimizers
    grid_optimizer = GridSearchOptimizer(param_space=param_space, n_jobs=1, verbose=False)
    genetic_optimizer = GeneticOptimizer(
        param_space=param_space, 
        population_size=50,
        n_generations=5,
        n_jobs=1,
        verbose=False
    )
    
    # Compare algorithms
    algorithms = [
        ('grid_search', grid_optimizer.optimize),
        ('genetic', genetic_optimizer.optimize)
    ]
    
    # Run comparison
    for name, algorithm in algorithms:
        print(f"Testing {name} algorithm")
        
        # Benchmark algorithm
        start_time = time.time()
        try:
            results = algorithm(**common_params)
            elapsed = time.time() - start_time
            print(f"{name} algorithm completed in {elapsed:.2f} seconds")
            
            # Record result
            runner.record_result({
                'algorithm': name,
                'runtime': elapsed
            })
        except Exception as e:
            print(f"Error in {name} algorithm: {e}")
    
    # Save results
    runner.save_results()
    
    return runner


def run_all_benchmarks():
    """Run all optimization benchmarks."""
    print("Running all optimization benchmarks...")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"benchmark_results/optimization_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    grid_results = benchmark_grid_search()
    genetic_results = benchmark_genetic_algorithm()
    regime_results = benchmark_regime_optimization()
    comparison_results = compare_optimization_algorithms()
    
    # Generate summary report
    summary = {
        'timestamp': timestamp,
        'grid_search': grid_results.results,
        'genetic_algorithm': genetic_results.results,
        'regime_optimization': regime_results.results,
        'comparison': comparison_results.results
    }
    
    # Save summary
    with open(output_dir / 'optimization_benchmark_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"All benchmarks completed. Results saved to {output_dir}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run optimization algorithm benchmarks")
    parser.add_argument('--benchmark', choices=['grid', 'genetic', 'regime', 'compare', 'all'],
                      default='all', help='Benchmark to run (default: all)')
    parser.add_argument('--output', type=str, default='benchmark_results',
                      help='Output directory for benchmark results')
    
    args = parser.parse_args()
    
    # Set output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run selected benchmark
    if args.benchmark == 'grid' or args.benchmark == 'all':
        benchmark_grid_search()
    
    if args.benchmark == 'genetic' or args.benchmark == 'all':
        benchmark_genetic_algorithm()
    
    if args.benchmark == 'regime' or args.benchmark == 'all':
        benchmark_regime_optimization()
    
    if args.benchmark == 'compare' or args.benchmark == 'all':
        compare_optimization_algorithms()
    
    if args.benchmark == 'all':
        run_all_benchmarks() 