#!/usr/bin/env python3
"""
Benchmark tests for the backtest engine.

These benchmarks measure the performance of backtesting operations
to identify potential bottlenecks and performance improvements.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import benchmark utilities
from tests.benchmark.test_benchmark_utils import BenchmarkRunner, measure_time, run_memory_profile

# Import scripts for test data generation
from scripts.generate_test_data import generate_cointegrated_pair, generate_dates, save_test_data

# Import utilities
from src.utils import create_directory

# Try to import backtest components
try:
    from src.backtest.backtest_engine import BacktestEngine
except ImportError:
    # Create a simple mock BacktestEngine for testing if the real one is not available
    class BacktestEngine:
        def __init__(self, config):
            self.config = config
            
        def run_backtest(self):
            # Simulate backtesting by processing some data
            print("Running mock backtest...")
            time.sleep(0.5)  # Simulate processing time
            return {"total_trades": 10, "total_pnl": 100.0}


def create_backtest_data(output_dir="./benchmark_data", pairs=3, days=365):
    """
    Create test data for backtesting benchmarks.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save test data
    pairs : int
        Number of pairs to generate
    days : int
        Number of days of data
        
    Returns:
    --------
    dict
        Dictionary with test data information
    """
    print(f"Generating test data for {pairs} pairs with {days} days of history...")
    
    # Create output directory
    data_dir = Path(output_dir)
    create_directory(data_dir)
    
    # Define pairs to generate
    pair_configs = [
        ("CL", "HO", 0.8, 0.2, 0.15),  # Crude oil and heating oil
        ("GC", "SI", 0.6, 0.3, 0.1),   # Gold and silver
        ("ZC", "ZW", 0.75, 0.25, 0.12), # Corn and wheat
        ("ES", "NQ", 0.85, 0.15, 0.08), # S&P 500 and Nasdaq
        ("ZN", "ZB", 0.9, 0.1, 0.05)    # 10Y and 30Y Treasury
    ]
    
    # Use subset of pairs
    pair_configs = pair_configs[:pairs]
    
    # Generate data
    generated_pairs = []
    
    for ticker1, ticker2, ratio, noise, mean_rev in pair_configs:
        print(f"Generating pair: {ticker1}-{ticker2}")
        
        # Generate daily data
        dates = generate_dates(days, '1d')
        data1, data2, spread = generate_cointegrated_pair(
            n_points=days,
            hedge_ratio=ratio,
            noise_ratio=noise,
            mean_rev_strength=mean_rev
        )
        
        # Save data
        save_test_data(ticker1, ticker2, data1, data2, dates, '1d', output_dir)
        
        # Save to generated pairs list
        generated_pairs.append((ticker1, ticker2, ratio, noise, mean_rev))
    
    # Create test config files
    config_dir = data_dir / "configs"
    create_directory(config_dir)
    
    # Create configurations with different parameters
    configs = []
    
    for i, (ticker1, ticker2, ratio, noise, mean_rev) in enumerate(generated_pairs):
        # Create different configurations for benchmarking
        base_config = {
            "pair": f"{ticker1}-{ticker2}",
            "start_date": dates[0].strftime("%Y-%m-%d"),
            "end_date": dates[-1].strftime("%Y-%m-%d"),
            "hedge_ratio": ratio,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss": 3.0,
            "lookback_period": 20,
            "max_holding_period": 10,
            "data_dir": str(data_dir)
        }
        
        # Save base config
        with open(config_dir / f"{ticker1}_{ticker2}_config.json", "w") as f:
            json.dump(base_config, f, indent=2)
        
        configs.append(base_config)
        
        # Create variations for benchmarking
        variations = [
            ("short_lookback", {"lookback_period": 10}),
            ("long_lookback", {"lookback_period": 40}),
            ("tight_thresholds", {"entry_threshold": 1.5, "exit_threshold": 0.3}),
            ("wide_thresholds", {"entry_threshold": 2.5, "exit_threshold": 0.8})
        ]
        
        for name, params in variations:
            var_config = base_config.copy()
            var_config.update(params)
            
            # Save variation
            with open(config_dir / f"{ticker1}_{ticker2}_{name}_config.json", "w") as f:
                json.dump(var_config, f, indent=2)
            
            configs.append(var_config)
    
    return {
        "pairs": generated_pairs,
        "configs": configs,
        "data_dir": data_dir,
        "config_dir": config_dir
    }


def benchmark_backtest_variations(data_info):
    """
    Benchmark different backtest variations.
    
    Parameters:
    -----------
    data_info : dict
        Information about test data
        
    Returns:
    --------
    list
        Benchmark results
    """
    # Initialize benchmark runner
    runner = BenchmarkRunner("backtest_variations", repetitions=3)
    
    # Define benchmark function
    def run_backtest_with_config(config):
        engine = BacktestEngine(config)
        return engine.run_backtest()
    
    # Run benchmarks for different pairs
    for i, (ticker1, ticker2, _, _, _) in enumerate(data_info["pairs"]):
        print(f"\nBenchmarking pair: {ticker1}-{ticker2}")
        
        # Find all configs for this pair
        pair_configs = [
            config for config in data_info["configs"] 
            if config["pair"] == f"{ticker1}-{ticker2}"
        ]
        
        # Benchmark each configuration
        for config in pair_configs:
            # Extract configuration name from parameters
            if config["lookback_period"] == 10:
                name = "short_lookback"
            elif config["lookback_period"] == 40:
                name = "long_lookback"
            elif config["entry_threshold"] == 1.5:
                name = "tight_thresholds"
            elif config["entry_threshold"] == 2.5:
                name = "wide_thresholds"
            else:
                name = "base"
            
            print(f"  Running benchmark for {name} configuration...")
            runner.run_benchmark(run_backtest_with_config, config)
    
    # Save results
    runner.save_results()
    
    return runner.results


def benchmark_backtest_scaling(data_info, scale_factors=[0.25, 0.5, 1.0, 2.0, 4.0]):
    """
    Benchmark backtest performance with different data sizes.
    
    Parameters:
    -----------
    data_info : dict
        Information about test data
    scale_factors : list
        Factors to scale the data by
        
    Returns:
    --------
    list
        Benchmark results
    """
    # Initialize benchmark runner
    runner = BenchmarkRunner("backtest_scaling", repetitions=2)
    
    # Use the first pair for scaling tests
    ticker1, ticker2, ratio, noise, mean_rev = data_info["pairs"][0]
    base_config = data_info["configs"][0]
    
    # Create new directory for scaled data
    scale_dir = data_info["data_dir"] / "scaling"
    create_directory(scale_dir)
    
    # Function to run benchmark
    def run_backtest_with_data_size(config):
        engine = BacktestEngine(config)
        return engine.run_backtest()
    
    # For each scale factor
    for factor in scale_factors:
        days = int(365 * factor)
        print(f"\nBenchmarking with scale factor {factor} ({days} days)...")
        
        # Generate scaled data
        dates = generate_dates(days, '1d')
        data1, data2, spread = generate_cointegrated_pair(
            n_points=days,
            hedge_ratio=ratio,
            noise_ratio=noise,
            mean_rev_strength=mean_rev
        )
        
        # Save data with scale factor in name
        factor_dir = scale_dir / f"factor_{factor}"
        create_directory(factor_dir)
        save_test_data(ticker1, ticker2, data1, data2, dates, '1d', factor_dir)
        
        # Create config with scaled data
        scale_config = base_config.copy()
        scale_config["start_date"] = dates[0].strftime("%Y-%m-%d")
        scale_config["end_date"] = dates[-1].strftime("%Y-%m-%d")
        scale_config["data_dir"] = str(factor_dir)
        
        # Run benchmark
        runner.run_benchmark(run_backtest_with_data_size, scale_config)
    
    # Save results
    runner.save_results()
    
    return runner.results


def benchmark_memory_usage(data_info):
    """
    Benchmark memory usage during backtest.
    
    Parameters:
    -----------
    data_info : dict
        Information about test data
        
    Returns:
    --------
    dict
        Memory profiling results
    """
    print("\nProfiling memory usage...")
    
    # Use the first configuration
    config = data_info["configs"][0]
    
    # Define function for profiling
    def run_backtest():
        engine = BacktestEngine(config)
        return engine.run_backtest()
    
    # Run memory profiling
    memory_results = run_memory_profile(run_backtest)
    
    # Save results
    output_dir = Path("benchmark_results")
    create_directory(output_dir)
    
    with open(output_dir / "backtest_memory_profile.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        memory_results_json = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                             for k, v in memory_results.items()}
        json.dump(memory_results_json, f, indent=2)
    
    return memory_results


def plot_scaling_results(results):
    """
    Plot scaling benchmark results.
    
    Parameters:
    -----------
    results : list
        Benchmark results
    """
    # Create output directory
    output_dir = Path("benchmark_results/plots")
    create_directory(output_dir)
    
    # Extract data for plotting
    factors = []
    times = []
    stdevs = []
    
    for result in results:
        # Extract scale factor from function name or parameters
        if "factor_" in result.get("function", ""):
            # Extract factor from function name
            factor_str = result["function"].split("factor_")[1]
            factor = float(factor_str)
        else:
            # Default factor if not found
            factor = 1.0
        
        factors.append(factor)
        times.append(result["mean_time"])
        stdevs.append(result["stdev_time"])
    
    # Sort by factor
    sorted_data = sorted(zip(factors, times, stdevs))
    factors, times, stdevs = zip(*sorted_data) if sorted_data else ([], [], [])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot execution time vs data size
    ax.errorbar(factors, times, yerr=stdevs, fmt='o-', capsize=5)
    
    # Add labels and title
    ax.set_xlabel('Scale Factor (relative to 1 year)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Backtest Performance Scaling')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_scaling.png")
    plt.close()
    
    # Plot time per data point
    time_per_point = [t / (365 * f) for t, f in zip(times, factors)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(factors, time_per_point, 'o-')
    
    # Add labels and title
    ax.set_xlabel('Scale Factor (relative to 1 year)')
    ax.set_ylabel('Time per Data Point (seconds)')
    ax.set_title('Backtest Efficiency Scaling')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / "backtest_efficiency.png")
    plt.close()
    
    print(f"Scaling plots saved to {output_dir}")


def plot_memory_profile(memory_results):
    """
    Plot memory profiling results.
    
    Parameters:
    -----------
    memory_results : dict
        Memory profiling results
    """
    # Check if memory profiling was successful
    if "error" in memory_results:
        print(f"Memory profiling error: {memory_results['error']}")
        return
    
    # Create output directory
    output_dir = Path("benchmark_results/plots")
    create_directory(output_dir)
    
    # Extract memory usage data
    usage = memory_results.get("detailed_usage", [])
    if not usage:
        print("No detailed memory usage data available")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to MB if necessary
    usage_mb = usage if memory_results.get("baseline_memory_mb") else [u / 1024 for u in usage]
    
    # Plot memory usage over time
    ax.plot(range(len(usage_mb)), usage_mb)
    
    # Add peak and baseline lines
    baseline = memory_results.get("baseline_memory_mb", usage_mb[0])
    peak = memory_results.get("peak_memory_mb", max(usage_mb))
    
    ax.axhline(y=baseline, color='g', linestyle='--', label=f'Baseline: {baseline:.2f} MB')
    ax.axhline(y=peak, color='r', linestyle='--', label=f'Peak: {peak:.2f} MB')
    
    # Add labels and title
    ax.set_xlabel('Measurement Point')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title(f'Memory Profile for {memory_results.get("function", "Backtest")}')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / "memory_profile.png")
    plt.close()
    
    print(f"Memory profile plot saved to {output_dir}")


def run_backtest_benchmarks():
    """Run all backtest engine benchmarks."""
    try:
        print("Starting backtest engine benchmarks...")
        
        # Create test data
        data_info = create_backtest_data()
        
        # Benchmark backtest variations
        variation_results = benchmark_backtest_variations(data_info)
        
        # Benchmark backtest scaling
        scaling_results = benchmark_backtest_scaling(data_info)
        
        # Benchmark memory usage
        memory_results = benchmark_memory_usage(data_info)
        
        # Plot results
        plot_scaling_results(scaling_results)
        plot_memory_profile(memory_results)
        
        print("Backtest benchmarks completed.")
        
    except Exception as e:
        print(f"Error running backtest benchmarks: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_backtest_benchmarks() 