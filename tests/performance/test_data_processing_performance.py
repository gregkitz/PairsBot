#!/usr/bin/env python3
"""
Performance tests for data processing components.

This module measures the performance of data loading, preprocessing, and feature engineering.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import cProfile
import pstats
import io
import tracemalloc
from functools import wraps
from contextlib import contextmanager

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)


def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Get stats
        stats = snapshot.statistics('lineno')
        total = sum(stat.size for stat in stats)
        
        # Convert to MB
        total_mb = total / (1024 * 1024)
        
        return result, total_mb
    
    return wrapper


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


def get_data_size():
    """Get the test data size from environment variables."""
    data_size = os.environ.get("PERFORMANCE_DATA_SIZE", "medium")
    
    # Define data sizes
    sizes = {
        "small": {"days": 30, "symbols": 5},
        "medium": {"days": 180, "symbols": 10},
        "large": {"days": 365, "symbols": 20},
        "production": {"days": 1825, "symbols": 50}  # 5 years, 50 symbols
    }
    
    return sizes.get(data_size, sizes["medium"])


def create_test_data(data_size):
    """
    Create synthetic test data for performance testing.
    
    Parameters:
    -----------
    data_size : dict
        Dictionary with data size parameters
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic test data
    """
    days = data_size["days"]
    symbols = data_size["symbols"]
    
    # Create date range
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Create price data
    data = {}
    
    for symbol_idx in range(symbols):
        symbol = f"SYMBOL{symbol_idx+1}"
        
        # Generate random price series with trends and volatility
        np.random.seed(symbol_idx)  # For reproducibility
        
        # Base price
        base_price = 100 + symbol_idx * 10
        
        # Trend component
        trend = np.cumsum(np.random.normal(0.0002, 0.001, days))
        
        # Volatility component
        volatility = np.random.normal(0, 0.01, days)
        
        # Combine to create price series
        prices = base_price * (1 + trend + volatility)
        
        # Create OHLCV data
        open_prices = prices * (1 + np.random.normal(0, 0.003, days))
        high_prices = prices * (1 + np.abs(np.random.normal(0, 0.007, days)))
        low_prices = prices * (1 - np.abs(np.random.normal(0, 0.007, days)))
        close_prices = prices * (1 + np.random.normal(0, 0.003, days))
        volumes = np.random.normal(1000000, 300000, days).astype(int)
        volumes = np.maximum(volumes, 0)  # Ensure non-negative
        
        # Create DataFrame for this symbol
        symbol_data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=date_range)
        
        data[symbol] = symbol_data
    
    return data


def test_data_loading_performance(test_data, output_dir):
    """
    Test the performance of data loading operations.
    
    Parameters:
    -----------
    test_data : dict
        Dictionary of DataFrames with test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    results = {
        "csv_loading": {},
        "parquet_loading": {},
        "hdf_loading": {}
    }
    
    # Create test files
    temp_dir = output_dir / "temp_data"
    create_directory(temp_dir)
    
    # Save data in different formats
    for symbol, df in test_data.items():
        df.to_csv(temp_dir / f"{symbol}.csv")
        df.to_parquet(temp_dir / f"{symbol}.parquet")
        
        # Write to HDF
        hdf_path = temp_dir / "test_data.h5"
        df.to_hdf(hdf_path, key=symbol, mode='a')
    
    # Test CSV loading
    @profile_memory
    def load_csv():
        data = {}
        for symbol in test_data.keys():
            file_path = temp_dir / f"{symbol}.csv"
            data[symbol] = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return data
    
    with measure_time() as csv_time:
        csv_data, csv_memory = load_csv()
    
    results["csv_loading"] = {
        "execution_time": csv_time,
        "memory_usage": csv_memory,
        "file_size": sum((temp_dir / f"{symbol}.csv").stat().st_size for symbol in test_data.keys()) / (1024 * 1024)
    }
    
    # Test Parquet loading
    @profile_memory
    def load_parquet():
        data = {}
        for symbol in test_data.keys():
            file_path = temp_dir / f"{symbol}.parquet"
            data[symbol] = pd.read_parquet(file_path)
        return data
    
    with measure_time() as parquet_time:
        parquet_data, parquet_memory = load_parquet()
    
    results["parquet_loading"] = {
        "execution_time": parquet_time,
        "memory_usage": parquet_memory,
        "file_size": sum((temp_dir / f"{symbol}.parquet").stat().st_size for symbol in test_data.keys()) / (1024 * 1024)
    }
    
    # Test HDF loading
    @profile_memory
    def load_hdf():
        data = {}
        hdf_path = temp_dir / "test_data.h5"
        for symbol in test_data.keys():
            data[symbol] = pd.read_hdf(hdf_path, key=symbol)
        return data
    
    with measure_time() as hdf_time:
        hdf_data, hdf_memory = load_hdf()
    
    results["hdf_loading"] = {
        "execution_time": hdf_time,
        "memory_usage": hdf_memory,
        "file_size": (temp_dir / "test_data.h5").stat().st_size / (1024 * 1024)
    }
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    formats = ["CSV", "Parquet", "HDF"]
    times = [results["csv_loading"]["execution_time"], 
             results["parquet_loading"]["execution_time"], 
             results["hdf_loading"]["execution_time"]]
    
    plt.bar(formats, times)
    plt.title("Data Loading Performance Comparison")
    plt.xlabel("File Format")
    plt.ylabel("Execution Time (seconds)")
    plt.savefig(output_dir / "data_loading_performance.png")
    
    return results


def test_feature_calculation_performance(test_data, output_dir):
    """
    Test the performance of feature calculation.
    
    Parameters:
    -----------
    test_data : dict
        Dictionary of DataFrames with test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Features to test
    features = {
        "moving_average": lambda df: df['close'].rolling(20).mean(),
        "exponential_ma": lambda df: df['close'].ewm(span=20).mean(),
        "bollinger_bands": lambda df: (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std(),
        "rsi": lambda df: calculate_rsi(df['close'], 14),
        "macd": lambda df: calculate_macd(df['close']),
        "atr": lambda df: calculate_atr(df, 14)
    }
    
    results = {}
    
    # Test each feature calculation
    for feature_name, feature_func in features.items():
        feature_results = {}
        
        # Apply to each symbol
        total_time = 0
        total_memory = 0
        
        for symbol, df in test_data.items():
            @profile_memory
            def calculate_feature():
                return feature_func(df)
            
            with measure_time() as execution_time:
                feature_data, memory_usage = calculate_feature()
            
            total_time += execution_time
            total_memory += memory_usage
        
        # Average results
        avg_time = total_time / len(test_data)
        avg_memory = total_memory / len(test_data)
        
        feature_results = {
            "execution_time": avg_time,
            "memory_usage": avg_memory
        }
        
        results[feature_name] = feature_results
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    feature_names = list(features.keys())
    times = [results[feature]["execution_time"] for feature in feature_names]
    
    plt.barh(feature_names, times)
    plt.title("Feature Calculation Performance")
    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_calculation_performance.png")
    
    return results


def calculate_rsi(series, window):
    """Calculate RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD for a price series."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def calculate_atr(df, window):
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr


def test_data_merging_performance(test_data, output_dir):
    """
    Test the performance of data merging operations.
    
    Parameters:
    -----------
    test_data : dict
        Dictionary of DataFrames with test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    results = {
        "join_all_symbols": {},
        "merge_features": {}
    }
    
    # Test joining all symbols
    @profile_memory
    def join_all_symbols():
        # Extract close prices
        close_prices = pd.DataFrame({symbol: df['close'] for symbol, df in test_data.items()})
        return close_prices
    
    with measure_time() as join_time:
        all_prices, join_memory = join_all_symbols()
    
    results["join_all_symbols"] = {
        "execution_time": join_time,
        "memory_usage": join_memory,
        "data_shape": f"{all_prices.shape[0]} rows x {all_prices.shape[1]} columns"
    }
    
    # Test merging features
    @profile_memory
    def merge_features():
        # Calculate features for first symbol
        symbol = list(test_data.keys())[0]
        df = test_data[symbol]
        
        features = pd.DataFrame(index=df.index)
        features['ma_20'] = df['close'].rolling(20).mean()
        features['ema_20'] = df['close'].ewm(span=20).mean()
        features['bb_upper'] = features['ma_20'] + df['close'].rolling(20).std() * 2
        features['bb_lower'] = features['ma_20'] - df['close'].rolling(20).std() * 2
        features['rsi'] = calculate_rsi(df['close'], 14)
        
        macd_data = calculate_macd(df['close'])
        features = features.join(macd_data)
        
        features['atr'] = calculate_atr(df, 14)
        
        return features
    
    with measure_time() as merge_time:
        features, merge_memory = merge_features()
    
    results["merge_features"] = {
        "execution_time": merge_time,
        "memory_usage": merge_memory,
        "data_shape": f"{features.shape[0]} rows x {features.shape[1]} columns"
    }
    
    return results


def run_tests(output_dir):
    """
    Run all data processing performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    create_directory(plots_dir)
    
    # Get data size
    data_size = get_data_size()
    print(f"Running tests with data size: {data_size}")
    
    # Create test data
    print("Creating test data...")
    test_data = create_test_data(data_size)
    
    # Results dictionary
    results = {
        "status": "success",
        "metrics": {
            "data_size": data_size,
            "num_symbols": len(test_data),
            "timepoints_per_symbol": len(next(iter(test_data.values())))
        },
        "test_results": {}
    }
    
    # Run tests
    print("Testing data loading performance...")
    results["test_results"]["data_loading"] = test_data_loading_performance(test_data, plots_dir)
    
    print("Testing feature calculation performance...")
    results["test_results"]["feature_calculation"] = test_feature_calculation_performance(test_data, plots_dir)
    
    print("Testing data merging performance...")
    results["test_results"]["data_merging"] = test_data_merging_performance(test_data, plots_dir)
    
    # Save detailed results
    with open(output_dir / "data_processing_performance.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Run tests with default settings
    output_dir = Path("performance_results/data_processing_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    create_directory(output_dir)
    
    results = run_tests(output_dir)
    
    print(f"Tests completed. Results saved to {output_dir}") 