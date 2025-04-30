# Performance Test Suite

This directory contains a comprehensive performance test suite for system components, focusing on measuring performance, scalability, and resource usage.

## Overview

The performance test suite measures various aspects of system performance:
- Execution time for critical operations
- Memory usage and patterns
- I/O operations and throughput
- CPU utilization
- Scaling behavior with increasing load

## Test Components

### 1. Data Processing Performance (`test_data_processing_performance.py`)

Tests the performance of data acquisition, preprocessing, and feature engineering:
- Data loading from different sources (CSV, Parquet, database)
- Feature calculation performance
- Data normalization and transformation
- Scaling behavior with increasing data volume

### 2. ML Model Training Performance (`test_model_training_performance.py`)

Tests the performance of model training pipelines:
- Training time for different model types
- Memory usage during training
- Hyperparameter optimization performance
- Feature selection performance
- Scaling behavior with increasing training data

### 3. Trading System Performance (`test_trading_system_performance.py`)

Tests the performance of the trading execution system:
- Signal generation throughput
- Order execution latency
- Position management overhead
- Risk calculation performance
- Scaling behavior with increasing number of pairs

### 4. API Performance (`test_api_performance.py`)

Tests the performance of the API endpoints:
- Request throughput
- Response time under load
- Concurrent request handling
- Result streaming performance
- Authentication overhead

### 5. End-to-End System Performance (`test_system_performance.py`)

Tests the performance of the complete system:
- End-to-end execution time 
- Resource usage during complete workflow
- System behavior under sustained load
- Recovery time after failures
- I/O bottleneck identification

## Usage

Run the complete performance test suite:

```bash
python -m tests.performance.run_performance_tests
```

Run specific performance tests:

```bash
python -m tests.performance.run_performance_tests --tests data api
```

Generate performance profiling graphs:

```bash
python -m tests.performance.run_performance_tests --profile --output profile_results
```

## Test Design Principles

The performance tests follow these design principles:

1. **Reproducibility**: Tests use consistent data and configuration for comparable results across runs
2. **Isolation**: Each test runs in isolation to avoid interference
3. **Realism**: Tests use realistic data volumes and workloads
4. **Measurability**: Clear metrics are captured for each test
5. **Scalability**: Tests can scale to measure performance at different load levels

## Performance Metrics

The following metrics are collected for each test:

- **Execution Time**: Wall clock time for operations
- **CPU Time**: Actual CPU processing time
- **Memory Usage**: Peak and average memory consumption
- **I/O Operations**: Number and size of I/O operations
- **Throughput**: Operations per second
- **Latency**: Time to first result and distribution of operation times

## Profiling Tools

The test suite uses these profiling tools:

1. `cProfile`: Python's built-in profiler for execution time analysis
2. `memory_profiler`: For detailed memory usage tracking
3. `py-spy`: For sampling-based profiling with low overhead
4. `psutil`: For system resource monitoring
5. `line_profiler`: For line-by-line execution time analysis

## Continuous Performance Testing

The performance test suite is designed to be integrated with CI/CD pipelines to:
- Detect performance regressions
- Track performance metrics over time
- Generate performance reports
- Compare performance across different configurations

## Performance Test Reports

Results are saved in the `performance_results` directory with:
- JSON files with raw metrics
- CSV files for time series analysis
- HTML reports with visualizations
- PNG plots for key metrics
- Flamegraphs for CPU and memory profiling 