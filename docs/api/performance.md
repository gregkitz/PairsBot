# Performance Optimization

The Performance Optimization modules provide tools to improve the system's computational efficiency through parallel processing, caching, and efficient data structures.

## Parallel Processing

The parallel processing module provides utilities for executing tasks in parallel to take advantage of multi-core CPUs.

```python
from src.performance.parallel import ParallelExecutor, TaskPool, parallel_map
```

### ParallelExecutor

`ParallelExecutor` manages the execution of tasks in parallel using either process-based or thread-based parallelism.

```python
from src.performance.parallel import ParallelExecutor

# Create a process-based executor
executor = ParallelExecutor(
    max_workers=4,                  # Maximum number of worker processes
    use_processes=True,             # Use process-based parallelism
    progress_callback=None,         # Optional callback for progress updates
    error_callback=None,            # Optional callback for error handling
    initialize_func=None            # Optional initialization function for workers
)

# Execute tasks in parallel
results = executor.execute(
    tasks=[                         # List of tasks to execute
        (func1, args1, kwargs1),    # Function and its arguments
        (func2, args2, kwargs2),
        # ...
    ],
    timeout=None                    # Optional timeout in seconds
)

# Or use the context manager (automatically closes resources)
with ParallelExecutor(max_workers=4) as executor:
    results = executor.execute(tasks)
```

### TaskPool

`TaskPool` manages long-running parallel tasks with automatic resource management and task prioritization.

```python
from src.performance.parallel import TaskPool, TaskPriority, TaskStatus

# Create a task pool
task_pool = TaskPool(
    max_workers=4,               # Maximum number of worker processes/threads
    use_processes=True,          # Use process-based parallelism
    max_queue_size=100,          # Maximum size of task queue
    result_ttl=3600,             # Time to live for results in seconds
    initialize_func=None         # Optional initialization function for workers
)

# Submit a task
task_id = task_pool.submit(
    func=my_function,            # Function to execute
    args=(arg1, arg2),           # Positional arguments
    kwargs={'key': 'value'},     # Keyword arguments
    priority=TaskPriority.NORMAL # Task priority (HIGH, NORMAL, LOW)
)

# Check task status
status = task_pool.get_status(task_id)  # Returns TaskStatus enum
# TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED

# Get task result (blocking call with timeout)
result = task_pool.get_result(task_id, timeout=10)

# Cancel a task
task_pool.cancel(task_id)

# Get all tasks
all_tasks = task_pool.get_tasks()

# Get all results
all_results = task_pool.get_results()

# Shutdown the task pool
task_pool.shutdown(wait=True)  # Wait for all tasks to complete
```

### parallel_map

`parallel_map` is a simpler interface for parallel execution, similar to the built-in `map` function but executed in parallel.

```python
from src.performance.parallel import parallel_map

# Execute a function on multiple inputs in parallel
results = parallel_map(
    func=my_function,        # Function to execute
    iterable=[1, 2, 3, 4],   # Input iterable
    max_workers=4,           # Maximum number of worker processes/threads
    use_processes=True,      # Use process-based parallelism
    chunksize=1,             # Chunk size for batching
    timeout=None,            # Optional timeout in seconds
    initializer=None,        # Optional initialization function
    initargs=()              # Arguments for initializer
)
```

## Caching

The caching module provides mechanisms to avoid redundant calculations by storing and reusing frequently accessed data and function results.

```python
from src.performance.caching import DataCache, memoize, timed_lru_cache, disk_cache
```

### DataCache

`DataCache` provides a flexible cache for storing data in memory or on disk with expiration.

```python
from src.performance.caching import DataCache

# Create a data cache
cache = DataCache(
    max_size_mb=100,              # Maximum memory usage in MB
    ttl=3600,                     # Time to live in seconds
    persist=True,                 # Persist cache to disk
    cache_dir='./cache',          # Directory for persistent storage
    filename='data_cache.pkl'     # Filename for persistent storage
)

# Store data in cache
cache.set('key', value, ttl=1800)  # With specific TTL

# Get data from cache
value = cache.get('key')           # Returns None if key not found or expired
value = cache.get('key', default='default value')  # With default value

# Check if key exists
exists = cache.has('key')

# Remove data from cache
cache.delete('key')

# Clear cache
cache.clear()

# Get cache statistics
stats = cache.get_stats()

# Save cache to disk
cache.save()

# Load cache from disk
cache.load()
```

### Function Caching Decorators

The module provides several decorators for caching function results:

#### memoize

`memoize` caches function results indefinitely based on arguments.

```python
from src.performance.caching import memoize

@memoize
def expensive_function(a, b):
    # Expensive computation
    return a + b

# First call computes the result
result1 = expensive_function(1, 2)  # Computes 3

# Second call with same arguments returns cached result
result2 = expensive_function(1, 2)  # Returns 3 from cache without computation

# Different arguments trigger a new computation
result3 = expensive_function(2, 3)  # Computes 5
```

#### timed_lru_cache

`timed_lru_cache` caches function results with an expiration time and maximum size.

```python
from src.performance.caching import timed_lru_cache

@timed_lru_cache(seconds=60, maxsize=128)
def get_stock_price(symbol):
    # Expensive API call
    return fetch_price_from_api(symbol)

# Cache result for 60 seconds
price1 = get_stock_price('AAPL')  # Makes API call

# Within 60 seconds, returns cached result
price2 = get_stock_price('AAPL')  # Returns from cache

# After 60 seconds, makes a new API call
# (Wait 60 seconds)
price3 = get_stock_price('AAPL')  # Makes new API call
```

#### disk_cache

`disk_cache` persists cached results to disk, allowing them to survive program restarts.

```python
from src.performance.caching import disk_cache

@disk_cache(cache_dir='./cache', ttl=3600)
def fetch_historical_data(symbol, start_date, end_date):
    # Expensive data retrieval
    return download_data(symbol, start_date, end_date)

# First call stores result on disk
data1 = fetch_historical_data('AAPL', '2022-01-01', '2022-12-31')

# Program can be restarted, and result will still be available from disk
data2 = fetch_historical_data('AAPL', '2022-01-01', '2022-12-31')  # Reads from disk
```

## Efficient Data Structures

The efficient data structures module provides optimized data structures for financial time series data.

```python
from src.performance.data_structures import SparseTimeSeries, CompressedOHLC, RollingWindow, ExponentialRollingWindow
```

### SparseTimeSeries

`SparseTimeSeries` is a memory-efficient data structure for time series with missing values or irregular intervals.

```python
from src.performance.data_structures import SparseTimeSeries
from datetime import datetime, timedelta

# Create a sparse time series
ts = SparseTimeSeries()

# Add data points (only stores non-zero/non-NaN values)
ts.add(datetime(2023, 1, 1), 100.0)
ts.add(datetime(2023, 1, 2), 101.0)
# No data for Jan 3
ts.add(datetime(2023, 1, 4), 102.0)

# Get value at specific time
value = ts.get(datetime(2023, 1, 1))  # 100.0
missing = ts.get(datetime(2023, 1, 3))  # None or default value

# Get values for a time range
values = ts.get_range(
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 1, 4),
    interval=timedelta(days=1),
    fill_method='ffill'  # Forward fill missing values
)

# Convert to pandas Series
series = ts.to_pandas()

# Get statistics
statistics = ts.get_statistics()  # min, max, mean, etc.
```

### CompressedOHLC

`CompressedOHLC` is a memory-efficient data structure for storing OHLC (Open, High, Low, Close) price data.

```python
from src.performance.data_structures import CompressedOHLC

# Create compressed OHLC data
ohlc = CompressedOHLC()

# Add a bar
ohlc.add(
    timestamp=datetime(2023, 1, 1, 9, 30),
    open=100.0,
    high=102.0,
    low=99.0,
    close=101.0,
    volume=10000
)

# Add multiple bars
ohlc.add_multiple([
    {'timestamp': datetime(2023, 1, 1, 9, 31), 'open': 101.0, 'high': 103.0, 'low': 100.5, 'close': 102.0, 'volume': 8000},
    {'timestamp': datetime(2023, 1, 1, 9, 32), 'open': 102.0, 'high': 104.0, 'low': 101.0, 'close': 103.0, 'volume': 12000}
])

# Get a bar
bar = ohlc.get(datetime(2023, 1, 1, 9, 30))

# Get bars for a time range
bars = ohlc.get_range(
    start_time=datetime(2023, 1, 1, 9, 30),
    end_time=datetime(2023, 1, 1, 9, 32)
)

# Resample to a different time frame
resampled = ohlc.resample('5min')  # 5-minute bars

# Convert to pandas DataFrame
df = ohlc.to_pandas()
```

### RollingWindow

`RollingWindow` provides an efficient implementation of rolling window calculations for time series data.

```python
from src.performance.data_structures import RollingWindow

# Create a rolling window
window = RollingWindow(
    window_size=20,  # Window size
    statistics=['mean', 'std', 'min', 'max']  # Statistics to track
)

# Add values
for price in prices:
    window.add(price)
    # Get current statistics
    current_mean = window.get_statistic('mean')
    current_std = window.get_statistic('std')

# Get all statistics
stats = window.get_statistics()  # {'mean': 100.5, 'std': 2.3, 'min': 95.0, 'max': 105.0}

# Reset window
window.reset()
```

### ExponentialRollingWindow

`ExponentialRollingWindow` calculates exponentially weighted moving statistics.

```python
from src.performance.data_structures import ExponentialRollingWindow

# Create an exponential rolling window
window = ExponentialRollingWindow(
    alpha=0.1,      # Smoothing factor
    statistics=['mean', 'var']  # Statistics to track
)

# Add values
for price in prices:
    window.add(price)
    # Get current statistics
    current_ewma = window.get_statistic('mean')  # Exponentially weighted moving average
    current_ewmv = window.get_statistic('var')   # Exponentially weighted moving variance
```

## Examples

### Parallel Backtesting

```python
from src.performance.parallel import ParallelExecutor
from src.backtesting import BacktestEngine

def run_backtest(parameters):
    """Run a backtest with specific parameters."""
    backtest_engine = BacktestEngine()
    result = backtest_engine.run(
        data=parameters['data'],
        signals=parameters['signals'],
        positions=parameters['positions']
    )
    return {
        'parameters': parameters['id'],
        'total_return': result.metrics.total_return,
        'sharpe_ratio': result.metrics.sharpe_ratio,
        'max_drawdown': result.metrics.max_drawdown
    }

# Generate parameter combinations
parameter_sets = [
    {'id': i, 'data': data, 'signals': generate_signals(data, i), 'positions': calculate_positions(i)}
    for i in range(100)  # 100 different parameter sets
]

# Run backtests in parallel
with ParallelExecutor(max_workers=8) as executor:
    results = executor.execute([
        (run_backtest, (params,), {}) for params in parameter_sets
    ])

# Find best parameters
best_result = max(results, key=lambda x: x['sharpe_ratio'])
print(f"Best parameters: {best_result['parameters']}")
print(f"Sharpe ratio: {best_result['sharpe_ratio']}")
```

### Caching Market Data

```python
from src.performance.caching import disk_cache
import pandas as pd

@disk_cache(cache_dir='./market_data_cache', ttl=86400)  # 24-hour cache
def fetch_market_data(symbol, start_date, end_date):
    """Fetch market data from an external API with caching."""
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    # Expensive API call here
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103]
    }, index=pd.date_range(start=start_date, periods=3, freq='D'))

# First call retrieves data from API and caches it
data1 = fetch_market_data('AAPL', '2023-01-01', '2023-01-31')

# Second call uses cached data
data2 = fetch_market_data('AAPL', '2023-01-01', '2023-01-31')  # No API call

# Different parameters trigger a new API call
data3 = fetch_market_data('AAPL', '2023-02-01', '2023-02-28')  # New API call
```

### Using Efficient Data Structures for Performance

```python
from src.performance.data_structures import SparseTimeSeries, RollingWindow
import pandas as pd
from datetime import datetime
import numpy as np

# Create a large time series with many missing values
dates = pd.date_range(start='2023-01-01', periods=10000, freq='5min')
values = np.random.normal(100, 5, len(dates))

# Set 80% of values to NaN (missing)
mask = np.random.random(len(dates)) < 0.8
values[mask] = np.nan

# Normal pandas Series (uses a lot of memory)
series = pd.Series(values, index=dates)
print(f"Pandas Series memory usage: {series.memory_usage() / 1024:.2f} KB")

# SparseTimeSeries (uses much less memory)
sparse_ts = SparseTimeSeries()
for i, date in enumerate(dates):
    if not np.isnan(values[i]):
        sparse_ts.add(date, values[i])

print(f"SparseTimeSeries approximate memory usage: {sparse_ts.memory_usage() / 1024:.2f} KB")

# Calculate moving average efficiently
window_size = 20
rolling_window = RollingWindow(window_size=window_size, statistics=['mean'])

# Slow way with pandas
ma_pandas = series.rolling(window=window_size).mean()

# Fast way with RollingWindow
ma_efficient = []
for i, date in enumerate(dates):
    if not np.isnan(values[i]):
        rolling_window.add(values[i])
    ma_efficient.append(rolling_window.get_statistic('mean'))

# Compare results
print("Similar results:", np.allclose(
    ma_pandas.iloc[-100:].dropna().values,
    np.array(ma_efficient[-100:])[-ma_pandas.iloc[-100:].dropna().shape[0]:],
    equal_nan=True
))
```

## Best Practices

### Parallel Processing

1. **Process vs. Threads**:
   - Use processes (`use_processes=True`) for CPU-bound tasks
   - Use threads (`use_processes=False`) for I/O-bound tasks like network requests

2. **Number of Workers**:
   - For CPU-bound tasks, use `max_workers=cpu_count()` or slightly fewer
   - For I/O-bound tasks, you can use more workers than CPU cores

3. **Error Handling**:
   - Always provide an `error_callback` for production code to catch exceptions

### Caching

1. **TTL Selection**:
   - Choose an appropriate Time-To-Live (TTL) based on how frequently the data changes
   - For static data, use a long TTL or no TTL (`memoize`)
   - For dynamic data, use a shorter TTL that matches the data refresh rate

2. **Memory Management**:
   - Set a reasonable `max_size_mb` for `DataCache` to prevent memory leaks
   - Use `disk_cache` for large datasets that need to persist between runs

3. **Cache Invalidation**:
   - Explicitly invalidate caches when data changes (`cache.delete(key)` or `cache.clear()`)

### Data Structures

1. **Choosing the Right Structure**:
   - Use `SparseTimeSeries` for time series with many missing values
   - Use `CompressedOHLC` specifically for price bar data
   - Use `RollingWindow` for efficient moving window calculations

2. **Memory Efficiency**:
   - Monitor memory usage with `memory_usage()` methods
   - For very large datasets, consider using these structures in combination with chunking or streaming 