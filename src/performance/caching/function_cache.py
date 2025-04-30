"""
Function Caching for the Intraday Statistical Arbitrage System.

This module provides decorators for caching function results to improve performance.
"""

import os
import pickle
import hashlib
import time
import functools
import inspect
import logging
from typing import Callable, Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import threading

# Set up logging
logger = logging.getLogger(__name__)


class _TimedCache:
    """Internal class for implementing timed caching."""
    
    def __init__(self, seconds: float = 3600, maxsize: int = 128):
        """
        Initialize the timed cache.
        
        Parameters:
        -----------
        seconds : float
            Number of seconds to cache results
        maxsize : int
            Maximum number of items to cache
        """
        self.seconds = seconds
        self.maxsize = maxsize
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key):
        """Get an item from the cache."""
        with self.lock:
            # Check if key exists and not expired
            if key in self.cache and time.time() - self.timestamps[key] < self.seconds:
                return self.cache[key]
            
            # Remove expired item
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            return None
    
    def set(self, key, value):
        """Set an item in the cache."""
        with self.lock:
            # Evict oldest items if cache is full
            if self.maxsize > 0 and len(self.cache) >= self.maxsize:
                # Find oldest item
                oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
                
                # Remove oldest item
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            # Add to cache
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


def memoize(func: Callable = None, maxsize: int = 128) -> Callable:
    """
    Memoize decorator for caching function results.
    
    This decorator caches function results based on arguments,
    improving performance for repeated calls with the same inputs.
    
    Parameters:
    -----------
    func : callable
        Function to decorate
    maxsize : int
        Maximum number of results to cache (0 for unlimited)
    
    Returns:
    --------
    callable
        Decorated function
    
    Example:
    --------
    >>> @memoize(maxsize=100)
    >>> def fibonacci(n):
    >>>     if n < 2:
    >>>         return n
    >>>     return fibonacci(n-1) + fibonacci(n-2)
    """
    if func is None:
        return lambda f: memoize(f, maxsize=maxsize)
    
    cache = {}
    lock = threading.RLock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key = _create_key(args, kwargs)
        
        with lock:
            # Check if result is cached
            if key in cache:
                return cache[key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            # Evict oldest items if cache is full
            if maxsize > 0 and len(cache) >= maxsize:
                # Simple LRU implementation: remove first item
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            return result
    
    # Add cache management methods
    def clear_cache():
        with lock:
            cache.clear()
    
    def get_cache_info():
        with lock:
            return {
                'size': len(cache),
                'maxsize': maxsize
            }
    
    wrapper.clear_cache = clear_cache
    wrapper.cache_info = get_cache_info
    
    return wrapper


def timed_lru_cache(seconds: float = 3600, maxsize: int = 128) -> Callable:
    """
    Timed LRU cache decorator for caching function results with expiration.
    
    This decorator caches function results for a specified time period,
    automatically invalidating entries that have expired.
    
    Parameters:
    -----------
    seconds : float
        Number of seconds to cache results
    maxsize : int
        Maximum number of results to cache
    
    Returns:
    --------
    callable
        Decorator function
    
    Example:
    --------
    >>> @timed_lru_cache(seconds=60, maxsize=100)
    >>> def get_market_data(symbol):
    >>>     # Expensive API call
    >>>     return fetch_from_api(symbol)
    """
    def decorator(func):
        # Create cache for this function
        cache = _TimedCache(seconds=seconds, maxsize=maxsize)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = _create_key(args, kwargs)
            
            # Check if result is cached and not expired
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Add cache management methods
        wrapper.clear_cache = cache.clear
        
        def get_cache_info():
            return {
                'size': len(cache.cache),
                'maxsize': maxsize,
                'ttl': seconds
            }
        
        wrapper.cache_info = get_cache_info
        
        return wrapper
    
    return decorator


def disk_cache(directory: str, 
              expires: float = None, 
              name_prefix: str = None) -> Callable:
    """
    Disk cache decorator for caching function results to disk.
    
    This decorator caches function results to disk, which can be useful
    for expensive computations that need to persist between program runs.
    
    Parameters:
    -----------
    directory : str
        Directory to store cache files
    expires : float, optional
        Expiration time in seconds
    name_prefix : str, optional
        Prefix for cache file names
    
    Returns:
    --------
    callable
        Decorator function
    
    Example:
    --------
    >>> @disk_cache(directory='cache', expires=86400)
    >>> def load_large_dataset(file_path):
    >>>     # Expensive data loading
    >>>     return pd.read_csv(file_path)
    """
    def decorator(func):
        # Create cache directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Use function name as prefix if not specified
        prefix = name_prefix if name_prefix is not None else func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = _create_key(args, kwargs)
            
            # Create a filename from the key
            filename = f"{prefix}_{hashlib.md5(str(key).encode()).hexdigest()}.cache"
            filepath = os.path.join(directory, filename)
            
            # Check if the file exists and is not expired
            if os.path.exists(filepath):
                file_time = os.path.getmtime(filepath)
                
                # Check if expired
                if expires is None or time.time() - file_time < expires:
                    try:
                        with open(filepath, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache file: {e}")
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(result, f, protocol=4)
            except Exception as e:
                logger.warning(f"Error saving cache file: {e}")
            
            return result
        
        # Add cache management methods
        def clear_cache():
            for filename in os.listdir(directory):
                if filename.startswith(prefix) and filename.endswith('.cache'):
                    try:
                        os.remove(os.path.join(directory, filename))
                    except Exception as e:
                        logger.warning(f"Error removing cache file: {e}")
        
        def get_cache_info():
            cache_files = [f for f in os.listdir(directory) 
                         if f.startswith(prefix) and f.endswith('.cache')]
            
            total_size = sum(os.path.getsize(os.path.join(directory, f)) for f in cache_files)
            
            return {
                'size': len(cache_files),
                'total_bytes': total_size,
                'directory': directory,
                'expires': expires
            }
        
        wrapper.clear_cache = clear_cache
        wrapper.cache_info = get_cache_info
        
        return wrapper
    
    return decorator


def _create_key(args: Tuple, kwargs: Dict) -> Tuple:
    """
    Create a cache key from function arguments.
    
    Parameters:
    -----------
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments
    
    Returns:
    --------
    tuple
        Cache key
    """
    # Convert non-hashable types to hashable representations
    key_args = []
    for arg in args:
        if isinstance(arg, (list, dict, set)):
            # Convert to string representation for hashing
            key_args.append(str(sorted(arg) if isinstance(arg, set) else arg))
        else:
            key_args.append(arg)
    
    # Sort kwargs for consistent keys
    key_kwargs = sorted(kwargs.items())
    
    return tuple(key_args + key_kwargs) 