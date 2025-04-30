#!/usr/bin/env python3
"""
Real-time signal optimization module for intraday trading.

This module provides optimized implementations of signal generation
and feature calculation for low-latency requirements in real-time trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import time

# Configure logging
logger = logging.getLogger(__name__)

class RealTimeSignalOptimizer:
    """
    Optimizes signal generation for real-time trading with low latency.
    
    This class implements specialized versions of signal generation algorithms
    that are optimized for low-latency requirements, using incremental updates,
    pre-computation, and efficient data structures.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the real-time signal optimizer.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the optimizer
        """
        self.config = config or {}
        
        # Set default configuration
        self.feature_cache = {}
        self.signal_cache = {}
        self.latest_prices = {}
        self.latest_times = {}
        self.latest_volumes = {}
        self.latest_features = {}
        
        # Performance metrics tracking
        self.timing_stats = {
            'feature_calculation': [],
            'signal_generation': [],
            'total_processing': []
        }
        
        # Configure optimization parameters
        self.use_incremental_updates = self.config.get('use_incremental_updates', True)
        self.use_feature_caching = self.config.get('use_feature_caching', True)
        self.parallel_processing = self.config.get('parallel_processing', False)
        self.max_lookback = self.config.get('max_lookback', 100)
        
        # Pre-allocation for calculation arrays (reduces memory allocations)
        self.preallocated_arrays = {
            'ema_weights': np.zeros(self.max_lookback),
            'rolling_window': np.zeros(self.max_lookback)
        }
        
        logger.info(f"Initialized RealTimeSignalOptimizer with configuration: {self.config}")
    
    def process_new_data(self, 
                       symbol: str, 
                       new_data: pd.DataFrame, 
                       calculate_signals: bool = True) -> Dict:
        """
        Process new data for a symbol with optimized performance.
        
        Parameters:
        -----------
        symbol : str
            The symbol identifier
        new_data : pd.DataFrame
            New price/volume data
        calculate_signals : bool
            Whether to calculate signals or just update cache
            
        Returns:
        --------
        dict
            Dictionary with processing results and timing information
        """
        start_time = time.time()
        
        # Store data in cache
        if symbol not in self.latest_prices:
            self.latest_prices[symbol] = new_data
            is_incremental = False
        else:
            # Determine if this is an incremental update or full replacement
            latest_index = self.latest_prices[symbol].index[-1]
            if new_data.index[0] > latest_index:
                # This is new data only - append
                if self.use_incremental_updates:
                    self.latest_prices[symbol] = pd.concat([
                        self.latest_prices[symbol].iloc[-self.max_lookback:], 
                        new_data
                    ])
                    is_incremental = True
                else:
                    # Full replacement
                    self.latest_prices[symbol] = new_data
                    is_incremental = False
            else:
                # Full replacement
                self.latest_prices[symbol] = new_data
                is_incremental = False
        
        # Update latest timestamp
        if not new_data.empty and isinstance(new_data.index, pd.DatetimeIndex):
            self.latest_times[symbol] = new_data.index[-1]
        
        # Extract volume if available
        if 'volume' in new_data.columns:
            if symbol not in self.latest_volumes:
                self.latest_volumes[symbol] = new_data['volume']
            else:
                if is_incremental:
                    self.latest_volumes[symbol] = pd.concat([
                        self.latest_volumes[symbol].iloc[-self.max_lookback:],
                        new_data['volume']
                    ])
                else:
                    self.latest_volumes[symbol] = new_data['volume']
        
        # Calculate features efficiently
        feature_start = time.time()
        if is_incremental and self.use_feature_caching and symbol in self.latest_features:
            # Incremental feature update
            new_features = self._calculate_incremental_features(
                symbol, new_data, self.latest_features[symbol]
            )
        else:
            # Full feature calculation
            new_features = self._calculate_features(symbol, self.latest_prices[symbol])
        
        self.latest_features[symbol] = new_features
        feature_time = time.time() - feature_start
        
        # Track timing for feature calculation
        self.timing_stats['feature_calculation'].append(feature_time)
        
        # Calculate trading signals if requested
        signals = None
        signal_time = 0
        
        if calculate_signals:
            signal_start = time.time()
            signals = self._generate_optimized_signals(symbol, new_features)
            signal_time = time.time() - signal_start
            
            # Track timing for signal generation
            self.timing_stats['signal_generation'].append(signal_time)
            
            # Cache signals
            self.signal_cache[symbol] = signals
        
        # Calculate total processing time
        total_time = time.time() - start_time
        self.timing_stats['total_processing'].append(total_time)
        
        # Log performance information
        logger.debug(f"Processing for {symbol}: features={feature_time*1000:.2f}ms, "
                    f"signals={signal_time*1000:.2f}ms, total={total_time*1000:.2f}ms")
        
        return {
            'symbol': symbol,
            'features': new_features,
            'signals': signals,
            'timing': {
                'feature_calculation_ms': feature_time * 1000,
                'signal_generation_ms': signal_time * 1000,
                'total_processing_ms': total_time * 1000
            },
            'is_incremental': is_incremental
        }
    
    def _calculate_features(self, symbol: str, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features with optimized performance.
        
        Parameters:
        -----------
        symbol : str
            Symbol identifier
        price_data : pd.DataFrame
            Price data with OHLCV columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated features
        """
        # Extract close prices
        close_prices = price_data['close'].values if 'close' in price_data.columns else price_data.values
        
        # Pre-allocate result arrays (avoid memory allocation in tight loops)
        n = len(close_prices)
        features = pd.DataFrame(index=price_data.index)
        
        # Use numpy vectorized operations for maximum performance
        # These avoid explicit Python loops which are slow
        
        # Calculate returns (faster with numpy than pandas)
        returns = np.zeros(n)
        returns[1:] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
        features['returns'] = returns
        
        # Calculate volatility efficiently
        # Use pre-allocated rolling window for better performance
        window = min(20, n)
        rolling_std = np.zeros(n)
        
        if n >= window:
            for i in range(window-1, n):
                rolling_std[i] = np.std(returns[max(0, i-window+1):i+1])
            
            features['volatility'] = rolling_std
        else:
            features['volatility'] = np.nan
        
        # EMA calculations - faster implementation using numpy
        windows = [10, 20, 50]
        for window in windows:
            if window < n:
                feature_name = f'ema_{window}'
                weights = np.exp(np.linspace(-1., 0., window))
                weights /= weights.sum()
                
                # Use numpy's convolve for fast EMA calculation
                ema = np.zeros(n)
                ema[:window] = np.nan  # Not enough data for first elements
                
                # Use faster numpy convolution
                ema[window:] = np.convolve(close_prices, weights, mode='valid')
                features[feature_name] = ema
        
        # Store in cache
        self.feature_cache[symbol] = features
        
        return features
    
    def _calculate_incremental_features(self, 
                                     symbol: str, 
                                     new_data: pd.DataFrame, 
                                     previous_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate incremental features for new data points only.
        
        Parameters:
        -----------
        symbol : str
            Symbol identifier
        new_data : pd.DataFrame
            New price data points
        previous_features : pd.DataFrame
            Previously calculated features
            
        Returns:
        --------
        pd.DataFrame
            Updated feature DataFrame
        """
        # Get the number of new data points
        n_new = len(new_data)
        
        if n_new == 0:
            return previous_features
        
        # Get full price history needed for calculations
        full_prices = self.latest_prices[symbol]
        
        # Calculate features only for new data and concatenate
        new_features = self._calculate_features(symbol, new_data)
        
        # Create correct index matching to ensure proper alignment
        aligned_features = pd.concat([
            previous_features[~previous_features.index.isin(new_features.index)],
            new_features
        ])
        
        return aligned_features
    
    def _generate_optimized_signals(self, symbol: str, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate optimized trading signals with low latency.
        
        Parameters:
        -----------
        symbol : str
            Symbol identifier
        features : pd.DataFrame
            Feature DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        # Start timing
        start_time = time.time()
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=features.index)
        signals['timestamp'] = signals.index
        signals['symbol'] = symbol
        signals['signal'] = 0  # Default to no signal
        
        # Get configuration for this symbol
        symbol_config = self.config.get('symbols', {}).get(symbol, {})
        
        # Get signal parameters
        entry_threshold = symbol_config.get('entry_threshold', 2.0)
        exit_threshold = symbol_config.get('exit_threshold', 0.5)
        
        # Check if we have required features
        required_features = ['returns', 'volatility']
        if not all(f in features.columns for f in required_features):
            logger.warning(f"Missing required features for signal generation: {symbol}")
            return signals
        
        # Get latest volatility for normalization
        latest_vol = features['volatility'].iloc[-20:].mean()
        
        # Generate signals based on normalized returns
        if latest_vol > 0:
            # Normalize returns by volatility
            normalized_returns = features['returns'] / latest_vol
            
            # Generate entry/exit signals
            long_entry = normalized_returns < -entry_threshold
            long_exit = normalized_returns > -exit_threshold
            
            short_entry = normalized_returns > entry_threshold
            short_exit = normalized_returns < exit_threshold
            
            # Apply signals
            signals['long_entry'] = long_entry.astype(int)
            signals['long_exit'] = long_exit.astype(int)
            signals['short_entry'] = short_entry.astype(int)
            signals['short_exit'] = short_exit.astype(int)
            
            # Generate final signals
            signals.loc[long_entry, 'signal'] = 1
            signals.loc[short_entry, 'signal'] = -1
            
            # Handle exits
            previous_signal = 0
            for i in range(1, len(signals)):
                current_idx = signals.index[i]
                prev_idx = signals.index[i-1]
                
                # Get previous signal
                previous_signal = signals.loc[prev_idx, 'signal']
                
                # Long exit
                if previous_signal == 1 and signals.loc[current_idx, 'long_exit']:
                    signals.loc[current_idx, 'signal'] = 0
                # Short exit
                elif previous_signal == -1 and signals.loc[current_idx, 'short_exit']:
                    signals.loc[current_idx, 'signal'] = 0
                # Continue position if no entry/exit
                elif signals.loc[current_idx, 'signal'] == 0:
                    signals.loc[current_idx, 'signal'] = previous_signal
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log execution time if over threshold
        if execution_time > 100:  # 100ms threshold for warning
            logger.warning(f"Signal generation for {symbol} took {execution_time:.2f}ms")
        
        return signals
    
    def get_latest_signals(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get the latest signals for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol identifier
            
        Returns:
        --------
        pd.DataFrame or None
            Latest signals if available
        """
        return self.signal_cache.get(symbol)
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the signal generation process.
        
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        metrics = {}
        
        for timing_type, values in self.timing_stats.items():
            if values:
                metrics[timing_type] = {
                    'avg_ms': np.mean(values) * 1000,
                    'min_ms': np.min(values) * 1000,
                    'max_ms': np.max(values) * 1000,
                    'p95_ms': np.percentile(values, 95) * 1000,
                    'count': len(values),
                    'last_ms': values[-1] * 1000 if values else None
                }
            else:
                metrics[timing_type] = {
                    'avg_ms': None,
                    'min_ms': None,
                    'max_ms': None,
                    'p95_ms': None,
                    'count': 0,
                    'last_ms': None
                }
        
        # Calculate overall metrics
        symbols_count = len(self.latest_prices)
        metrics['system'] = {
            'symbols_processed': symbols_count,
            'feature_cache_size': sum(len(df) for df in self.feature_cache.values()) if self.feature_cache else 0,
            'signal_cache_size': sum(len(df) for df in self.signal_cache.values()) if self.signal_cache else 0
        }
        
        return metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance tracking metrics."""
        for timing_type in self.timing_stats:
            self.timing_stats[timing_type] = []

def optimize_signal_generation(signal_generator, config=None):
    """
    Factory function to wrap a signal generator with real-time optimizations.
    
    Parameters:
    -----------
    signal_generator : object
        Original signal generator object
    config : dict, optional
        Configuration for the optimizer
        
    Returns:
    --------
    RealTimeSignalOptimizer
        Optimized signal generator
    """
    # Create optimizer with existing generator's configuration
    optimizer = RealTimeSignalOptimizer(config or getattr(signal_generator, 'config', {}))
    
    # Copy over necessary attributes and methods
    if hasattr(signal_generator, 'latest_prices'):
        optimizer.latest_prices = signal_generator.latest_prices
    
    return optimizer 