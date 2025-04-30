#!/usr/bin/env python3
"""
Benchmark tests for data processing operations.

These benchmarks measure the performance of critical data processing operations
to identify potential bottlenecks and compare different implementation approaches.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import benchmark utilities
from tests.benchmark.test_benchmark_utils import BenchmarkRunner, measure_time, run_memory_profile

# Import components to test
try:
    from scripts.generate_test_data import generate_cointegrated_pair, generate_dates
    from src.utils import create_directory
except ImportError:
    print("Warning: Some modules could not be imported. Some tests might be skipped.")


def create_test_data(size=1000, pairs=5):
    """
    Create test data for benchmarking data processing operations.
    
    Parameters:
    -----------
    size : int
        Number of data points per pair
    pairs : int
        Number of pairs to generate
        
    Returns:
    --------
    dict
        Dictionary containing test data
    """
    # Generate test data directory
    test_data_dir = Path("./benchmark_data")
    create_directory(test_data_dir)
    
    # Generate dates
    dates = generate_dates(size, '1d')
    
    # Generate pairs with different parameters
    pair_data = {}
    pair_names = [
        ('CL', 'HO'), ('GC', 'SI'), ('ZC', 'ZW'),
        ('ES', 'NQ'), ('ZN', 'ZB')
    ]
    
    for i in range(min(pairs, len(pair_names))):
        ticker1, ticker2 = pair_names[i]
        hedge_ratio = 0.7 + (i * 0.05)  # Different hedge ratios
        noise_ratio = 0.2 + (i * 0.02)  # Different noise levels
        mean_rev = 0.1 + (i * 0.01)     # Different mean reversion strengths
        
        # Generate pair data
        data1, data2, spread = generate_cointegrated_pair(
            n_points=size,
            hedge_ratio=hedge_ratio,
            noise_ratio=noise_ratio,
            mean_rev_strength=mean_rev
        )
        
        # Create DataFrames
        df1 = pd.DataFrame({
            'open': data1,
            'high': data1 * (1 + np.random.uniform(0, 0.005, len(data1))),
            'low': data1 * (1 - np.random.uniform(0, 0.005, len(data1))),
            'close': data1,
            'volume': np.random.randint(1000, 10000, len(data1))
        }, index=dates)
        
        df2 = pd.DataFrame({
            'open': data2,
            'high': data2 * (1 + np.random.uniform(0, 0.005, len(data2))),
            'low': data2 * (1 - np.random.uniform(0, 0.005, len(data2))),
            'close': data2,
            'volume': np.random.randint(1000, 10000, len(data2))
        }, index=dates)
        
        pair_data[(ticker1, ticker2)] = {
            'data1': df1,
            'data2': df2,
            'spread': spread,
            'hedge_ratio': hedge_ratio
        }
        
        # Save to CSV for file-based tests
        df1.to_csv(test_data_dir / f"{ticker1}_data.csv")
        df2.to_csv(test_data_dir / f"{ticker2}_data.csv")
    
    return {
        'dates': dates,
        'pairs': pair_data,
        'data_dir': test_data_dir
    }


def calculate_spread_naive(df1, df2, hedge_ratio):
    """
    Calculate spread between two price series (naive implementation).
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        Price data for each asset
    hedge_ratio : float
        Hedge ratio between assets
        
    Returns:
    --------
    pandas.Series
        Calculated spread
    """
    return df1['close'] - hedge_ratio * df2['close']


def calculate_spread_vectorized(df1, df2, hedge_ratio):
    """
    Calculate spread between two price series (vectorized implementation).
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        Price data for each asset
    hedge_ratio : float
        Hedge ratio between assets
        
    Returns:
    --------
    pandas.Series
        Calculated spread
    """
    return df1['close'].values - hedge_ratio * df2['close'].values


def calculate_zscore_naive(spread, window=20):
    """
    Calculate z-score of spread (naive implementation with loop).
    
    Parameters:
    -----------
    spread : pandas.Series
        Spread between two assets
    window : int
        Rolling window size
        
    Returns:
    --------
    pandas.Series
        Z-score of spread
    """
    zscore = np.zeros_like(spread)
    
    for i in range(window, len(spread)):
        window_data = spread[i-window:i]
        mean = np.mean(window_data)
        std = np.std(window_data)
        zscore[i] = (spread[i] - mean) / std if std > 0 else 0
        
    return zscore


def calculate_zscore_pandas(spread, window=20):
    """
    Calculate z-score of spread (pandas implementation).
    
    Parameters:
    -----------
    spread : pandas.Series or numpy.ndarray
        Spread between two assets
    window : int
        Rolling window size
        
    Returns:
    --------
    pandas.Series
        Z-score of spread
    """
    if not isinstance(spread, pd.Series):
        spread = pd.Series(spread)
        
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (spread - rolling_mean) / rolling_std
    
    return zscore.fillna(0)


def calculate_zscore_numpy(spread, window=20):
    """
    Calculate z-score of spread (optimized numpy implementation).
    
    Parameters:
    -----------
    spread : numpy.ndarray
        Spread between two assets
    window : int
        Rolling window size
        
    Returns:
    --------
    numpy.ndarray
        Z-score of spread
    """
    # Ensure spread is a numpy array
    if isinstance(spread, pd.Series):
        spread = spread.values
    
    # Pre-allocate output array
    zscore = np.zeros_like(spread)
    
    # Use numpy's strided array view for rolling window calculation
    shape = (len(spread) - window + 1, window)
    strides = spread.strides * 2
    
    # Create a view with sliding windows
    windows = np.lib.stride_tricks.as_strided(spread[:len(spread)-window+1], 
                                             shape=shape, 
                                             strides=strides)
    
    # Calculate mean and std for each window
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1)
    
    # Avoid division by zero
    stds[stds == 0] = 1
    
    # Calculate z-scores
    zscore[window:] = (spread[window:] - means) / stds
    
    return zscore


def calculate_half_life(spread):
    """
    Calculate half-life of mean reversion for a spread.
    
    Parameters:
    -----------
    spread : numpy.ndarray or pandas.Series
        Spread between two assets
        
    Returns:
    --------
    float
        Half-life of mean reversion
    """
    # Ensure spread is a numpy array
    if isinstance(spread, pd.Series):
        spread = spread.values
    
    # Create lagged version of spread
    spread_lag = np.roll(spread, 1)
    spread_lag[0] = spread_lag[1]
    
    # Calculate delta
    delta = spread - spread_lag
    
    # Perform regression: delta = alpha + beta * spread
    X = sm.add_constant(spread_lag[1:])
    y = delta[1:]
    model = sm.OLS(y, X).fit()
    
    # Extract beta coefficient
    beta = model.params[1]
    
    # Calculate half-life
    half_life = -np.log(2) / beta if beta < 0 else 0
    
    return half_life


def run_data_processing_benchmarks():
    """Run benchmarks for data processing operations."""
    print("Starting data processing benchmarks...")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner("data_processing_benchmarks", repetitions=3)
    
    # Create test data
    print("Creating test data...")
    test_data = create_test_data(size=5000, pairs=3)
    
    # Get sample data for a pair
    pair_key = list(test_data['pairs'].keys())[0]
    df1 = test_data['pairs'][pair_key]['data1']
    df2 = test_data['pairs'][pair_key]['data2']
    hedge_ratio = test_data['pairs'][pair_key]['hedge_ratio']
    
    # Benchmark spread calculation implementations
    print("\nComparing spread calculation implementations...")
    spread_functions = [
        calculate_spread_naive,
        calculate_spread_vectorized
    ]
    
    args_list = [
        (df1, df2, hedge_ratio),
        (df1, df2, hedge_ratio)
    ]
    
    spread_comparison = runner.compare_implementations(
        spread_functions,
        args_list=args_list
    )
    
    # Use the fastest implementation for further tests
    fastest_spread_func = globals()[spread_comparison['fastest_implementation']]
    spread = fastest_spread_func(df1, df2, hedge_ratio)
    
    # Benchmark z-score calculation implementations
    print("\nComparing z-score calculation implementations...")
    zscore_functions = [
        calculate_zscore_naive,
        calculate_zscore_pandas,
        calculate_zscore_numpy
    ]
    
    args_list = [
        (spread,),
        (spread,),
        (spread,)
    ]
    
    runner.compare_implementations(
        zscore_functions,
        args_list=args_list
    )
    
    # Benchmark data loading
    print("\nBenchmarking data loading operations...")
    
    def load_csv_data(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    def load_parquet_data(file_path):
        # First check if file exists, if not create it
        if not os.path.exists(file_path):
            # Convert CSV to parquet
            df = pd.read_csv(file_path.replace('.parquet', '.csv'), 
                           index_col=0, parse_dates=True)
            df.to_parquet(file_path)
        return pd.read_parquet(file_path)
    
    # Test CSV loading
    csv_file = test_data['data_dir'] / f"{pair_key[0]}_data.csv"
    runner.run_benchmark(load_csv_data, csv_file)
    
    # Test Parquet loading (convert first if needed)
    parquet_file = test_data['data_dir'] / f"{pair_key[0]}_data.parquet"
    if not os.path.exists(parquet_file):
        df1.to_parquet(parquet_file)
    runner.run_benchmark(load_parquet_data, parquet_file)
    
    # Save benchmark results
    runner.save_results("json")
    runner.save_results("csv")
    
    return runner.results


def plot_benchmark_results(results):
    """
    Plot benchmark results for visualization.
    
    Parameters:
    -----------
    results : list
        List of benchmark result dictionaries
    """
    # Create output directory
    output_dir = Path("benchmark_results/plots")
    create_directory(output_dir)
    
    # Group results by functionality
    function_groups = {}
    for result in results:
        func_name = result['function']
        # Extract function category (e.g., calculate_spread_naive -> calculate_spread)
        category = "_".join(func_name.split("_")[:-1]) if "_" in func_name else func_name
        
        if category not in function_groups:
            function_groups[category] = []
        function_groups[category].append(result)
    
    # Plot each group
    for category, category_results in function_groups.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        names = [r['function'].split('_')[-1] for r in category_results]
        mean_times = [r['mean_time'] for r in category_results]
        stdev_times = [r['stdev_time'] for r in category_results]
        
        # Plot bar chart with error bars
        bars = ax.bar(names, mean_times, yerr=stdev_times, capsize=10)
        
        # Add labels and title
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(f'Benchmark Results: {category}')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        
        # Add value labels
        for bar, value in zip(bars, mean_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}s', ha='center', va='bottom', rotation=0)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f"{category}_benchmark.png")
        plt.close()
    
    # Plot relative speed comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    categories = []
    implementations = []
    speeds = []
    
    for category, category_results in function_groups.items():
        if len(category_results) > 1:
            # Find the fastest implementation for normalization
            min_time = min(r['mean_time'] for r in category_results)
            
            for result in category_results:
                categories.append(category)
                implementations.append(result['function'].split('_')[-1])
                speeds.append(min_time / result['mean_time'])
    
    # Create grouped bar chart
    unique_categories = list(set(categories))
    unique_implementations = list(set(implementations))
    
    bar_width = 0.8 / len(unique_implementations)
    
    for i, impl in enumerate(unique_implementations):
        impl_data = [speeds[j] for j in range(len(speeds)) 
                    if implementations[j] == impl]
        impl_categories = [categories[j] for j in range(len(categories)) 
                         if implementations[j] == impl]
        
        x = np.arange(len(unique_categories))
        impl_heights = []
        
        for cat in unique_categories:
            if cat in impl_categories:
                idx = impl_categories.index(cat)
                impl_heights.append(impl_data[idx])
            else:
                impl_heights.append(0)
        
        ax.bar(x + i * bar_width - 0.4 + bar_width/2, impl_heights, 
             width=bar_width, label=impl)
    
    # Add labels and title
    ax.set_ylabel('Relative Speed (higher is better)')
    ax.set_title('Relative Performance Comparison')
    ax.set_xticks(np.arange(len(unique_categories)))
    ax.set_xticklabels(unique_categories)
    ax.legend()
    
    # Add horizontal line at y=1
    ax.axhline(y=1, color='r', linestyle='--')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / "relative_performance.png")
    plt.close()
    
    print(f"Benchmark plots saved to {output_dir}")


if __name__ == "__main__":
    try:
        # Import statsmodels if available
        import statsmodels.api as sm
        
        # Run benchmarks
        results = run_data_processing_benchmarks()
        
        # Plot results
        plot_benchmark_results(results)
        
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install required packages: pip install pandas numpy matplotlib statsmodels")
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        import traceback
        traceback.print_exc() 