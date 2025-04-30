#!/usr/bin/env python3
"""
Utility functions for benchmark testing.

This module provides common functions for performance measurement and benchmarking
of critical system operations.
"""

import os
import sys
import time
import datetime
import contextlib
import csv
import json
import statistics
from functools import wraps
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Union, Tuple

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import create_directory


def timeit(func):
    """
    Decorator to measure execution time of a function.
    
    Parameters:
    -----------
    func : function
        The function to measure
        
    Returns:
    --------
    wrapper : function
        Decorated function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result, end_time - start_time
    return wrapper


@contextlib.contextmanager
def measure_time(operation_name: str = None) -> float:
    """
    Context manager for measuring execution time of a code block.
    
    Parameters:
    -----------
    operation_name : str, optional
        Name of the operation being measured
        
    Yields:
    -------
    float
        Start time of the measurement
    """
    start_time = time.time()
    yield start_time
    end_time = time.time()
    elapsed = end_time - start_time
    
    if operation_name:
        print(f"Operation '{operation_name}' took {elapsed:.4f} seconds to complete")
    else:
        print(f"Operation took {elapsed:.4f} seconds to complete")
    
    return elapsed


class BenchmarkRunner:
    """Class for running and recording benchmark tests."""
    
    def __init__(self, benchmark_name: str, output_dir: str = "benchmark_results", repetitions: int = 5):
        """
        Initialize a benchmark runner.
        
        Parameters:
        -----------
        benchmark_name : str
            Name of the benchmark
        output_dir : str
            Directory to save benchmark results
        repetitions : int
            Number of times to repeat each benchmark
        """
        self.benchmark_name = benchmark_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.repetitions = repetitions
        self.results = []
        
        # Create output directory if it doesn't exist
        create_directory(self.output_dir)
    
    def run_benchmark(self, function: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark on a given function.
        
        Parameters:
        -----------
        function : callable
            Function to benchmark
        *args, **kwargs
            Arguments to pass to the function
            
        Returns:
        --------
        dict
            Benchmark results
        """
        function_name = function.__name__
        execution_times = []
        
        print(f"Running benchmark for {function_name}...")
        
        for i in range(self.repetitions):
            print(f"  Run {i+1}/{self.repetitions}...")
            start_time = time.time()
            result = function(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"    Completed in {execution_time:.4f} seconds")
        
        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        stdev_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        # Create result dictionary
        benchmark_result = {
            "function": function_name,
            "timestamp": self.timestamp,
            "repetitions": self.repetitions,
            "execution_times": execution_times,
            "mean_time": mean_time,
            "median_time": median_time,
            "stdev_time": stdev_time,
            "min_time": min_time,
            "max_time": max_time
        }
        
        self.results.append(benchmark_result)
        
        # Print summary
        print(f"Benchmark summary for {function_name}:")
        print(f"  Mean execution time: {mean_time:.4f} seconds")
        print(f"  Median execution time: {median_time:.4f} seconds")
        print(f"  Standard deviation: {stdev_time:.4f} seconds")
        print(f"  Min execution time: {min_time:.4f} seconds")
        print(f"  Max execution time: {max_time:.4f} seconds")
        
        return benchmark_result
    
    def save_results(self, format: str = "json") -> str:
        """
        Save benchmark results to a file.
        
        Parameters:
        -----------
        format : str
            Output format, either 'json' or 'csv'
            
        Returns:
        --------
        str
            Path to the saved results file
        """
        filename = f"{self.benchmark_name}_{self.timestamp}"
        
        if format.lower() == "json":
            output_file = self.output_dir / f"{filename}.json"
            with open(output_file, "w") as f:
                json.dump(self.results, f, indent=2)
        
        elif format.lower() == "csv":
            output_file = self.output_dir / f"{filename}.csv"
            
            # Flatten the results for CSV
            flattened_results = []
            for result in self.results:
                flat_result = {
                    "function": result["function"],
                    "timestamp": result["timestamp"],
                    "repetitions": result["repetitions"],
                    "mean_time": result["mean_time"],
                    "median_time": result["median_time"],
                    "stdev_time": result["stdev_time"],
                    "min_time": result["min_time"],
                    "max_time": result["max_time"]
                }
                
                # Add individual run times
                for i, time_value in enumerate(result["execution_times"]):
                    flat_result[f"run_{i+1}"] = time_value
                
                flattened_results.append(flat_result)
            
            # Write to CSV
            with open(output_file, "w", newline="") as f:
                if flattened_results:
                    writer = csv.DictWriter(f, fieldnames=flattened_results[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_results)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
        
        print(f"Benchmark results saved to {output_file}")
        return str(output_file)
    
    def compare_implementations(self, functions: List[Callable], args_list: List[Tuple] = None, 
                               kwargs_list: List[Dict] = None) -> Dict[str, Any]:
        """
        Compare multiple implementations of the same functionality.
        
        Parameters:
        -----------
        functions : list of callables
            Functions to compare
        args_list : list of tuples, optional
            List of argument tuples for each function
        kwargs_list : list of dicts, optional
            List of keyword argument dicts for each function
            
        Returns:
        --------
        dict
            Comparison results
        """
        if args_list is None:
            args_list = [()] * len(functions)
            
        if kwargs_list is None:
            kwargs_list = [{}] * len(functions)
            
        if len(functions) != len(args_list) or len(functions) != len(kwargs_list):
            raise ValueError("Number of functions must match number of argument sets")
            
        comparison_results = []
        
        for i, (func, args, kwargs) in enumerate(zip(functions, args_list, kwargs_list)):
            print(f"\nBenchmarking implementation {i+1}: {func.__name__}")
            result = self.run_benchmark(func, *args, **kwargs)
            comparison_results.append(result)
            
        # Determine the fastest implementation
        fastest_idx = min(range(len(comparison_results)), 
                         key=lambda i: comparison_results[i]["median_time"])
        fastest = comparison_results[fastest_idx]
        
        # Calculate relative performance
        for result in comparison_results:
            result["relative_speed"] = fastest["median_time"] / result["median_time"]
            
        print("\nComparison Summary:")
        for i, result in enumerate(comparison_results):
            print(f"Implementation {i+1}: {result['function']}")
            print(f"  Median time: {result['median_time']:.4f} seconds")
            print(f"  Relative speed: {result['relative_speed']:.2f}x")
            if result["relative_speed"] == 1.0:
                print("  (Fastest implementation)")
            print("")
            
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp,
            "comparison_results": comparison_results,
            "fastest_implementation": fastest["function"]
        }


def run_memory_profile(function: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile memory usage of a function.
    
    Parameters:
    -----------
    function : callable
        Function to profile
    *args, **kwargs
        Arguments to pass to the function
        
    Returns:
    --------
    dict
        Memory profiling results
    """
    try:
        import memory_profiler
        
        # Create a memory usage function that we can use with the profiler
        def measure_memory():
            function(*args, **kwargs)
            
        # Run the memory profiler
        memory_usage = memory_profiler.memory_usage(measure_memory)
        
        # Calculate statistics
        baseline = memory_usage[0]
        peak = max(memory_usage)
        increase = peak - baseline
        
        result = {
            "function": function.__name__,
            "baseline_memory_mb": baseline,
            "peak_memory_mb": peak,
            "memory_increase_mb": increase,
            "detailed_usage": memory_usage.tolist()
        }
        
        print(f"Memory profile for {function.__name__}:")
        print(f"  Baseline memory usage: {baseline:.2f} MB")
        print(f"  Peak memory usage: {peak:.2f} MB")
        print(f"  Memory increase: {increase:.2f} MB")
        
        return result
        
    except ImportError:
        print("memory_profiler package is required for memory profiling.")
        print("Install it using: pip install memory_profiler")
        return {
            "function": function.__name__,
            "error": "memory_profiler package not installed"
        } 