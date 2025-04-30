# Intraday ML System: Code Optimization Strategies

This document outlines strategies for optimizing the Intraday ML System's code for improved performance, reduced latency, and better resource utilization.

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithmic Optimizations](#algorithmic-optimizations)
3. [Parallelization Strategies](#parallelization-strategies)
4. [Memory Optimization](#memory-optimization)
5. [I/O Optimization](#io-optimization)
6. [Implementation Plan](#implementation-plan)
7. [Verification and Testing](#verification-and-testing)

## Introduction

Based on the performance profiling results, this document provides specific optimization strategies for the most critical components of the Intraday ML System. The optimizations focus on:

- Refactoring inefficient algorithms
- Implementing parallel processing for data-intensive operations
- Optimizing memory usage for feature calculation
- Improving I/O efficiency

Each optimization includes:
- Target code area
- Expected performance improvement
- Implementation approach
- Potential trade-offs

## Algorithmic Optimizations

### Feature Calculation Engine

**Current Issue**: Feature calculation is computationally expensive, especially for technical indicators.

**Optimization Strategy**:
1. Replace recursive implementations with vectorized operations
2. Use numba JIT compilation for performance-critical functions
3. Implement early-stopping for feature selection
4. Optimize redundant calculations across features

**Implementation**:

```python
# BEFORE: Inefficient rolling calculation
def calculate_rolling_zscore(series, window):
    result = np.zeros(len(series))
    for i in range(window, len(series)):
        segment = series[i-window:i]
        mean = np.mean(segment)
        std = np.std(segment)
        result[i] = (series[i] - mean) / std if std > 0 else 0
    return result

# AFTER: Vectorized implementation
def calculate_rolling_zscore_optimized(series, window):
    means = series.rolling(window=window).mean()
    stds = series.rolling(window=window).std()
    zscores = (series - means) / stds
    return zscores.fillna(0)
```

**Expected Improvement**: 5-10x faster feature calculation, reducing feature engineering time from minutes to seconds for large datasets.

### Signal Processing

**Current Issue**: Signal processing uses inefficient filtering algorithms.

**Optimization Strategy**:
1. Replace iterative signal processing with vectorized operations
2. Implement more efficient regime-based filtering
3. Use specialized data structures for signal state tracking

**Implementation**:

```python
# BEFORE: Iterative signal processing
def process_signals(signals, thresholds):
    processed = np.zeros_like(signals)
    current_signal = 0
    for i in range(len(signals)):
        if abs(signals[i]) > thresholds['entry'] and current_signal == 0:
            current_signal = signals[i]
        elif abs(signals[i]) < thresholds['exit'] and current_signal != 0:
            current_signal = 0
        processed[i] = current_signal
    return processed

# AFTER: Vectorized signal processing
def process_signals_optimized(signals, thresholds):
    # Generate entry conditions
    entries = (abs(signals) > thresholds['entry']).astype(int) * signals
    
    # Create exit mask
    exits = abs(signals) < thresholds['exit']
    
    # Use cumulative logic for state tracking
    signal_states = np.zeros_like(signals)
    active_signal = 0
    
    for i in range(len(signals)):
        # If entry condition and no active signal
        if entries[i] != 0 and active_signal == 0:
            active_signal = entries[i]
        # If exit condition and have active signal
        elif exits[i] and active_signal != 0:
            active_signal = 0
        signal_states[i] = active_signal
    
    return signal_states
```

**Expected Improvement**: 2-3x faster signal processing, particularly for large datasets or high-frequency data.

### Model Inference Optimization

**Current Issue**: Model inference is not optimized for batch processing or hardware acceleration.

**Optimization Strategy**:
1. Batch inference requests for multiple models
2. Implement model quantization for faster inference
3. Use ONNX Runtime for optimized model execution

**Implementation**:

```python
# BEFORE: Individual model inference
def predict_with_models(features, models):
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(features)
    return predictions

# AFTER: Optimized batch inference
def predict_with_models_optimized(features, models, batch_size=1000):
    predictions = {}
    
    # Convert models to ONNX if not already
    onnx_models = {name: convert_to_onnx(model) if not is_onnx(model) else model 
                  for name, model in models.items()}
    
    # Create inference sessions
    sessions = {name: create_inference_session(model) for name, model in onnx_models.items()}
    
    # Process in batches
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        
        for model_name, session in sessions.items():
            batch_prediction = session.run(None, {"input": batch})[0]
            
            if model_name not in predictions:
                predictions[model_name] = np.zeros(len(features))
            
            predictions[model_name][i:i+len(batch)] = batch_prediction
            
    return predictions
```

**Expected Improvement**: 2-4x faster model inference, with significantly better performance on GPUs if available.

## Parallelization Strategies

### Feature Calculation Parallelization

**Current Issue**: Feature calculation runs sequentially across pairs and features.

**Optimization Strategy**:
1. Parallelize feature calculation across multiple pairs
2. Split computation-heavy features into parallel workers
3. Implement work-stealing task scheduling for load balancing

**Implementation**:

```python
# Add to src/ml_enhancements/feature_engineering/parallel_features.py
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np

def calculate_pair_features_parallel(pairs_data, feature_functions, max_workers=None):
    """
    Calculate features for multiple pairs in parallel
    
    Args:
        pairs_data: Dict of pair data
        feature_functions: List of feature functions to apply
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary of pair features
    """
    all_features = {}
    
    def process_pair(pair_key):
        pair_features = {}
        data = pairs_data[pair_key]
        
        for func in feature_functions:
            feature_name = func.__name__
            pair_features[feature_name] = func(data)
            
        return pair_key, pair_features
    
    # Process pairs in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(process_pair, pair_key): pair_key 
            for pair_key in pairs_data.keys()
        }
        
        for future in as_completed(future_to_pair):
            pair_key, pair_features = future.result()
            all_features[pair_key] = pair_features
    
    return all_features
```

**Command-line tool for configuration**:

```bash
# Configure parallel processing settings
python -m src.config.configure_parallelism \
  --feature-calculation-workers 8 \
  --signal-processing-workers 4 \
  --model-inference-workers 2
```

**Expected Improvement**: Near-linear scaling with the number of cores for feature calculation, potentially 4-8x faster on an 8-core system.

### Parallel Backtesting

**Current Issue**: Backtesting multiple pairs or parameter combinations runs sequentially.

**Optimization Strategy**:
1. Implement parallel backtesting for multiple pairs
2. Enable parameter sweep parallelization
3. Use shared memory for common data

**Implementation**:

```python
# Add to src/backtest/parallel_backtest.py
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from multiprocessing import shared_memory
import pickle

def run_parallel_backtests(pairs, parameters, backtest_func, max_workers=None):
    """
    Run multiple backtests in parallel
    
    Args:
        pairs: List of pairs to backtest
        parameters: Parameters for each backtest
        backtest_func: Function to run backtest
        max_workers: Maximum number of worker processes
        
    Returns:
        Dictionary of backtest results by pair
    """
    results = {}
    
    def run_single_backtest(pair, params):
        result = backtest_func(pair=pair, **params)
        return pair, result
    
    # Create tasks for parallel execution
    tasks = []
    for pair in pairs:
        for param_set in parameters:
            tasks.append((pair, param_set))
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_backtest, pair, params): (pair, params) 
            for pair, params in tasks
        }
        
        for future in as_completed(future_to_task):
            pair, result = future.result()
            if pair not in results:
                results[pair] = []
            results[pair].append(result)
    
    return results
```

**Expected Improvement**: Backtest throughput scales with the number of cores, allowing much faster parameter optimization and multi-pair analysis.

## Memory Optimization

### Efficient Data Structures

**Current Issue**: Excessive memory usage for feature storage and historical data.

**Optimization Strategy**:
1. Use memory-efficient data types (e.g., float32 instead of float64)
2. Implement sparse representations for signals
3. Use memory-mapped files for large datasets
4. Implement data streaming for historical backtests

**Implementation**:

```python
# Add to src/data/efficient_data.py
import numpy as np
import pandas as pd
import os

class MemoryEfficientDataHandler:
    """Memory efficient data handler for large datasets"""
    
    def __init__(self, base_path, use_mmap=True):
        self.base_path = base_path
        self.use_mmap = use_mmap
        self.data_cache = {}
        self.mmap_files = []
    
    def optimize_dataframe(self, df, categorical_columns=None):
        """Optimize dataframe memory usage"""
        categorical_columns = categorical_columns or []
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        for col in df.select_dtypes(include=['integer']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert categorical columns
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def load_data(self, symbol, start_date=None, end_date=None):
        """Load data with memory optimization"""
        filepath = os.path.join(self.base_path, f"{symbol}.parquet")
        
        if self.use_mmap and filepath not in self.data_cache:
            # Use memory mapping for large files
            df = pd.read_parquet(filepath, memory_map=True)
            
            # Filter by date if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            # Optimize memory usage
            df = self.optimize_dataframe(df)
            
            self.data_cache[filepath] = df
        
        return self.data_cache.get(filepath, None)
    
    def clear_cache(self):
        """Clear data cache to free memory"""
        self.data_cache.clear()
        
        # Explicitly release memory-mapped files
        for mmap_file in self.mmap_files:
            mmap_file.close()
        
        self.mmap_files = []
```

**Command-line tool for configuration**:

```bash
# Configure memory optimization settings
python -m src.config.configure_memory \
  --use-memory-mapping true \
  --float-precision reduced \
  --feature-cache-size 1000 \
  --enable-data-streaming true
```

**Expected Improvement**: 40-60% reduction in memory usage, allowing processing of larger datasets or more pairs simultaneously.

### Selective Feature Caching

**Current Issue**: All features are calculated and stored, even when not all are needed.

**Optimization Strategy**:
1. Implement lazy feature calculation
2. Cache only frequently used features
3. Use importance-based feature prioritization

**Implementation**:

```python
# Add to src/ml_enhancements/feature_engineering/lazy_features.py
class LazyFeatureCalculator:
    """Feature calculator with lazy evaluation and caching"""
    
    def __init__(self, max_cache_size=1000, prioritize_by_importance=True):
        self.feature_functions = {}
        self.feature_cache = {}
        self.max_cache_size = max_cache_size
        self.prioritize_by_importance = prioritize_by_importance
        self.feature_importance = {}
        self.access_count = {}
    
    def register_feature(self, name, function, dependencies=None):
        """Register a feature calculation function"""
        self.feature_functions[name] = {
            'function': function,
            'dependencies': dependencies or []
        }
        self.access_count[name] = 0
    
    def set_feature_importance(self, importance_dict):
        """Set feature importance for prioritization"""
        self.feature_importance = importance_dict
    
    def calculate_feature(self, name, data):
        """Calculate a feature, using cache if available"""
        # Check if in cache
        cache_key = (name, id(data))
        if cache_key in self.feature_cache:
            self.access_count[name] += 1
            return self.feature_cache[cache_key]
        
        # Calculate dependencies first
        if name in self.feature_functions:
            func_info = self.feature_functions[name]
            
            # Calculate any dependencies
            dep_values = {}
            for dep in func_info['dependencies']:
                dep_values[dep] = self.calculate_feature(dep, data)
            
            # Calculate the feature
            result = func_info['function'](data, **dep_values)
            
            # Update cache
            self.feature_cache[cache_key] = result
            self.access_count[name] += 1
            
            # Manage cache size
            self._prune_cache_if_needed()
            
            return result
        else:
            raise ValueError(f"Unknown feature: {name}")
    
    def _prune_cache_if_needed(self):
        """Remove least valuable items from cache if it's too large"""
        if len(self.feature_cache) <= self.max_cache_size:
            return
        
        # Determine which features to keep
        if self.prioritize_by_importance and self.feature_importance:
            # Prioritize by importance and access count
            scores = {name: self.feature_importance.get(name, 0) * (1 + np.log1p(count)) 
                     for name, count in self.access_count.items()}
        else:
            # Prioritize by access count only
            scores = self.access_count.copy()
        
        # Sort features by score
        sorted_features = sorted(scores.items(), key=lambda x: x[1])
        
        # Remove lowest scoring features from cache
        features_to_remove = set(name for name, _ in sorted_features[:len(self.feature_cache) - self.max_cache_size])
        
        # Remove from cache
        self.feature_cache = {k: v for k, v in self.feature_cache.items() 
                             if k[0] not in features_to_remove}
```

**Expected Improvement**: 30-50% reduction in feature calculation time for incremental updates and reduced memory usage.

## I/O Optimization

### Efficient Data Formats

**Current Issue**: Data loading and saving is I/O intensive, particularly for model serialization.

**Optimization Strategy**:
1. Use columnar data formats (Parquet/Arrow) for all data storage
2. Implement chunked reading for large datasets
3. Optimize model serialization format
4. Use compression selectively

**Implementation**:

```python
# Add to src/data/optimized_io.py
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
from joblib import dump, load

class OptimizedDataIO:
    """Optimized I/O operations for trading data"""
    
    def __init__(self, compression='snappy', chunk_size=100000):
        self.compression = compression
        self.chunk_size = chunk_size
    
    def save_dataframe(self, df, filepath, partition_cols=None):
        """Save dataframe with optimized settings"""
        # Convert to arrow table
        table = pa.Table.from_pandas(df)
        
        # Write with optimized settings
        if partition_cols:
            pq.write_to_dataset(
                table, 
                root_path=filepath, 
                partition_cols=partition_cols,
                compression=self.compression
            )
        else:
            pq.write_table(
                table, 
                filepath,
                compression=self.compression
            )
    
    def read_dataframe(self, filepath, columns=None, filters=None):
        """Read dataframe with optimized settings"""
        if os.path.isdir(filepath):
            # Read partitioned dataset
            dataset = pq.ParquetDataset(filepath, filters=filters)
            table = dataset.read(columns=columns)
        else:
            # Read single file
            table = pq.read_table(filepath, columns=columns, filters=filters)
        
        return table.to_pandas()
    
    def read_dataframe_chunked(self, filepath, callback, columns=None, filters=None):
        """Read large dataframe in chunks"""
        # Open file or dataset
        if os.path.isdir(filepath):
            dataset = pq.ParquetDataset(filepath, filters=filters)
            reader = dataset.reader()
        else:
            reader = pq.ParquetFile(filepath)
        
        # Read and process chunks
        for batch in reader.iter_batches(batch_size=self.chunk_size, columns=columns):
            df_chunk = batch.to_pandas()
            callback(df_chunk)
    
    def save_model(self, model, filepath, compress=True):
        """Save ML model with optimized format"""
        compression = 3 if compress else None
        dump(model, filepath, compress=compression, protocol=4)
    
    def load_model(self, filepath):
        """Load ML model with optimized settings"""
        return load(filepath, mmap_mode='r')
```

**Command-line tool for configuration**:

```bash
# Configure I/O optimization settings
python -m src.config.configure_io \
  --data-format parquet \
  --compression snappy \
  --chunk-size 100000 \
  --model-compression-level 3
```

**Expected Improvement**: 3-5x faster data loading and 40-60% reduction in storage requirements.

### Database Query Optimization

**Current Issue**: Database queries for historical data are inefficient.

**Optimization Strategy**:
1. Implement database indexing for common queries
2. Use database-specific optimizations (e.g., VACUUM for SQLite)
3. Implement connection pooling
4. Cache frequent queries

**Implementation**:

```python
# Add to src/db/query_optimization.py
import sqlite3
import pandas as pd
from functools import lru_cache
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

class OptimizedDatabaseAccess:
    """Database access with performance optimizations"""
    
    def __init__(self, db_path, pool_size=5, enable_caching=True):
        self.db_path = db_path
        self.pool_size = pool_size
        self.enable_caching = enable_caching
        self.connection_pool = ThreadPoolExecutor(max_workers=pool_size)
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool"""
        conn = sqlite3.connect(self.db_path)
        
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode = WAL")
        
        # Larger cache size for better performance
        conn.execute("PRAGMA cache_size = -10000")  # ~10MB cache
        
        try:
            yield conn
        finally:
            conn.close()
    
    @lru_cache(maxsize=100)
    def _cached_query(self, query, params_tuple):
        """Internal cached query execution"""
        with self.get_connection() as conn:
            return pd.read_sql(query, conn, params=params_tuple)
    
    def query_df(self, query, params=None):
        """Execute query and return dataframe with caching if enabled"""
        if self.enable_caching:
            # Convert params to hashable tuple for cache
            params_tuple = tuple(params.items()) if isinstance(params, dict) else \
                          tuple(params) if params is not None else tuple()
            
            return self._cached_query(query, params_tuple)
        else:
            with self.get_connection() as conn:
                return pd.read_sql(query, conn, params=params)
    
    def optimize_database(self):
        """Run database optimizations"""
        with self.get_connection() as conn:
            # Analyze to update statistics
            conn.execute("ANALYZE")
            
            # Vacuum to defragment
            conn.execute("VACUUM")
    
    def create_indexes(self):
        """Create common indexes for performance"""
        with self.get_connection() as conn:
            # Create index on timestamp for time-based queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp)")
            
            # Create index on symbol for symbol-based queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol)")
            
            # Create composite index for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_timestamp ON prices(symbol, timestamp)")
```

**Expected Improvement**: 2-10x faster database queries, particularly for filtering operations on large datasets.

## Implementation Plan

### Phase 1: Quick Wins (1-2 weeks)
1. Implement vectorized feature calculations (highest priority)
2. Switch to efficient data formats for I/O operations
3. Add memory optimization for core data structures

### Phase 2: Core Optimizations (2-4 weeks)
1. Implement parallelization for feature calculation
2. Add memory-mapped file support for large datasets
3. Optimize model inference with batching and ONNX runtime

### Phase 3: Advanced Optimizations (4-6 weeks)
1. Implement lazy feature calculation and caching
2. Optimize database operations
3. Add parallel backtesting framework

## Verification and Testing

For each optimization:

1. **Benchmark Before and After**:
   ```bash
   # Run benchmarks to compare performance
   python -m src.diagnostics.benchmark_optimization \
     --optimization-name "vectorized_features" \
     --iterations 10 \
     --output benchmarks/vectorized_features_results.json
   ```

2. **Verify Correctness**:
   ```bash
   # Verify output consistency
   python -m src.testing.verify_optimization \
     --optimization-name "vectorized_features" \
     --tolerance 1e-6 \
     --output verification_results.json
   ```

3. **Resource Usage Analysis**:
   ```bash
   # Analyze memory and CPU usage
   python -m src.diagnostics.analyze_resource_usage \
     --optimization-name "vectorized_features" \
     --duration 300 \
     --interval 1 \
     --output resource_usage_results.json
   ```

## Conclusion

The outlined optimization strategies address the key performance bottlenecks identified during profiling. By implementing these optimizations in the proposed phased approach, we expect to achieve:

- 5-10x faster feature calculation
- 2-4x faster model inference
- 30-60% reduction in memory usage
- 3-5x faster data loading and saving

These improvements will enable:
- Processing more pairs simultaneously
- Supporting higher frequency data
- Reducing latency for real-time signal generation
- Improving backtesting performance for faster strategy development

Regular benchmarking and verification should be performed to ensure optimizations maintain correctness while improving performance. 