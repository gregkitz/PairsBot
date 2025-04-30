# Performance Benchmark Tests

This directory contains benchmark tests for measuring and optimizing the performance of critical system operations.

## Overview

The benchmark testing framework measures execution time, memory usage, and scaling properties of various system components. This helps identify bottlenecks, optimize algorithms, and ensure the system can handle production-scale data efficiently.

## Benchmark Components

### 1. Benchmark Utilities (`test_benchmark_utils.py`)

Core utilities for measuring performance:
- `timeit` decorator for function timing
- `measure_time` context manager for code block timing
- `BenchmarkRunner` class for consistent benchmarking and reporting
- `run_memory_profile` function for memory usage measurement

### 2. Data Processing Benchmarks (`test_data_processing_benchmark.py`)

Benchmarks for data manipulation operations:
- Spread calculation algorithms comparison
- Z-score calculation implementations
- Data loading format comparison (CSV vs Parquet)
- Statistical calculations

### 3. Backtest Engine Benchmarks (`test_backtest_engine_benchmark.py`)

Benchmarks for backtesting operations:
- Backtest parameter variations
- Performance scaling with data size
- Memory usage profiling

### 4. Benchmark Runner (`test_run_benchmarks.py`)

Script for running all benchmarks and generating reports:
- Run selected or all benchmarks
- Time tracking for each benchmark
- HTML report generation with performance data

## Usage

### Running Benchmarks

To run all benchmarks:

```bash
python -m tests.benchmark.test_run_benchmarks
```

To run specific benchmarks:

```bash
python -m tests.benchmark.test_run_benchmarks --tests data backtest
```

Options:
- `--tests`: Space-separated list of test categories to run (`data`, `backtest`, `all`)
- `--output`: Directory to save benchmark results (default: `benchmark_results`)
- `--quick`: Run with fewer iterations for faster results

### Adding New Benchmarks

To add a new benchmark:

1. Use the `BenchmarkRunner` class for consistent measurement
2. Follow the pattern in existing benchmark files
3. Add your benchmark module to `test_run_benchmarks.py`

Example benchmark function:

```python
def benchmark_my_function():
    # Initialize benchmark runner
    runner = BenchmarkRunner("my_benchmark", repetitions=3)
    
    # Create test data
    test_data = create_test_data()
    
    # Define functions to benchmark
    functions = [implementation1, implementation2]
    
    # Run benchmarks
    runner.compare_implementations(functions)
    
    # Save results
    runner.save_results()
```

## Benchmark Results

Results are saved in the `benchmark_results` directory, with a timestamped subdirectory for each run. Each run includes:

- JSON and CSV files with detailed performance data
- HTML report with tables and charts
- Plot images for visual comparison

## Best Practices

1. **Control Test Data**: Generate consistent test data for reliable comparisons
2. **Multiple Implementations**: Compare different implementation approaches
3. **Data Scaling**: Test with different data sizes to identify scaling issues
4. **Repeatable Tests**: Set random seeds for reproducible results
5. **Regular Benchmarking**: Run benchmarks periodically to catch performance regressions 