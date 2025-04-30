"""
Data Cache for the Intraday Statistical Arbitrage System.

This module provides a data cache for storing and retrieving frequently used data,
with support for expiration, persistence, and memory management.
"""

import os
import pickle
import json
import time
import threading
import logging
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class DataCache:
    """
    Data Cache for storing and retrieving frequently used data.
    
    This class provides methods for caching data in memory or on disk,
    with support for expiration, automatic eviction, and memory management.
    """
    
    def __init__(self,
                max_size: int = 1000,
                max_memory_mb: float = 500,
                expiration_seconds: Optional[float] = None,
                persistent_dir: Optional[str] = None):
        """
        Initialize the data cache.
        
        Parameters:
        -----------
        max_size : int
            Maximum number of items in the cache
        max_memory_mb : float
            Maximum memory usage in MB
        expiration_seconds : float, optional
            Default expiration time in seconds (None for no expiration)
        persistent_dir : str, optional
            Directory for persistent cache storage on disk
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.expiration_seconds = expiration_seconds
        self.persistent_dir = persistent_dir
        
        # Create persistent directory if specified
        if self.persistent_dir is not None:
            os.makedirs(self.persistent_dir, exist_ok=True)
        
        # Cache data structures
        self.cache = {}  # Main cache dictionary
        self.metadata = {}  # Metadata for cached items
        self.lock = threading.RLock()  # Thread-safe lock
        
        # Memory monitoring
        self._last_memory_check = time.time()
        self._memory_check_interval = 60  # seconds
    
    def get(self, 
           key: str, 
           default: Any = None, 
           check_expired: bool = True) -> Any:
        """
        Get an item from the cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        default : any, optional
            Default value to return if key is not found
        check_expired : bool
            Whether to check if the item has expired
            
        Returns:
        --------
        any
            Cached item or default value
        """
        with self.lock:
            # Check memory usage periodically
            self._check_memory_usage()
            
            # Check if key exists
            if key not in self.cache:
                # Try to load from disk if persistent cache is enabled
                if self.persistent_dir is not None:
                    if self._load_from_disk(key):
                        logger.debug(f"Loaded key '{key}' from disk cache")
                    else:
                        return default
                else:
                    return default
            
            # Check if expired
            if check_expired and self._is_expired(key):
                self._remove_item(key)
                return default
            
            # Update access time
            self.metadata[key]['last_accessed'] = time.time()
            
            # Return cached item
            return self.cache[key]
    
    def set(self, 
           key: str, 
           value: Any, 
           expiration_seconds: Optional[float] = None,
           persist: bool = False) -> bool:
        """
        Set an item in the cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        value : any
            Value to cache
        expiration_seconds : float, optional
            Expiration time in seconds (None for default)
        persist : bool
            Whether to persist the item to disk
            
        Returns:
        --------
        bool
            True if the item was cached successfully
        """
        with self.lock:
            # Check memory usage
            self._check_memory_usage()
            
            # Evict items if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_items()
            
            # Set expiration time
            if expiration_seconds is None:
                expiration_seconds = self.expiration_seconds
            
            expiration_time = None
            if expiration_seconds is not None:
                expiration_time = time.time() + expiration_seconds
            
            # Store item
            self.cache[key] = value
            
            # Store metadata
            self.metadata[key] = {
                'created_at': time.time(),
                'last_accessed': time.time(),
                'expiration_time': expiration_time,
                'persist': persist
            }
            
            # Persist to disk if requested
            if persist and self.persistent_dir is not None:
                self._save_to_disk(key, value)
            
            return True
    
    def contains(self, key: str, check_expired: bool = True) -> bool:
        """
        Check if an item exists in the cache.
        
        Parameters:
        -----------
        key : str
            Cache key
        check_expired : bool
            Whether to check if the item has expired
            
        Returns:
        --------
        bool
            True if the item exists and is not expired
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                # Try to load from disk if persistent cache is enabled
                if self.persistent_dir is not None:
                    if self._load_from_disk(key):
                        logger.debug(f"Loaded key '{key}' from disk cache for contains check")
                    else:
                        return False
                else:
                    return False
            
            # Check if expired
            if check_expired and self._is_expired(key):
                self._remove_item(key)
                return False
            
            return True
    
    def remove(self, key: str) -> bool:
        """
        Remove an item from the cache.
        
        Parameters:
        -----------
        key : str
            Cache key
            
        Returns:
        --------
        bool
            True if the item was removed
        """
        with self.lock:
            return self._remove_item(key)
    
    def clear(self, remove_persistent: bool = False) -> int:
        """
        Clear the cache.
        
        Parameters:
        -----------
        remove_persistent : bool
            Whether to remove persistent items from disk
            
        Returns:
        --------
        int
            Number of items removed
        """
        with self.lock:
            count = len(self.cache)
            
            # Clear memory cache
            self.cache.clear()
            
            # Get persistent keys
            persistent_keys = []
            if not remove_persistent and self.persistent_dir is not None:
                persistent_keys = [k for k, v in self.metadata.items() if v.get('persist', False)]
            
            # Clear metadata
            self.metadata.clear()
            
            # Remove persistent files if requested
            if remove_persistent and self.persistent_dir is not None:
                for filename in os.listdir(self.persistent_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.persistent_dir, filename))
            
            # Reload persistent items
            for key in persistent_keys:
                self._load_from_disk(key)
            
            logger.debug(f"Cleared cache ({count} items)")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
        --------
        dict
            Dictionary with cache statistics
        """
        with self.lock:
            # Count expired items
            expired_count = sum(1 for k in self.cache if self._is_expired(k))
            
            # Calculate memory usage
            memory_usage = self._estimate_memory_usage()
            
            # Count persistent items
            persistent_count = sum(1 for v in self.metadata.values() if v.get('persist', False))
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': memory_usage,
                'max_memory_mb': self.max_memory_mb,
                'expired_count': expired_count,
                'persistent_count': persistent_count,
                'hit_ratio': self._get_hit_ratio()
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if an item has expired."""
        if key not in self.metadata:
            return True
        
        metadata = self.metadata[key]
        
        if metadata.get('expiration_time') is None:
            return False
        
        return time.time() > metadata['expiration_time']
    
    def _remove_item(self, key: str) -> bool:
        """Remove an item from the cache and disk."""
        if key not in self.cache:
            return False
        
        # Check if item is persistent
        persist = self.metadata.get(key, {}).get('persist', False)
        
        # Remove from memory
        del self.cache[key]
        
        if key in self.metadata:
            del self.metadata[key]
        
        # Remove from disk if persistent
        if persist and self.persistent_dir is not None:
            disk_path = self._get_disk_path(key)
            if os.path.exists(disk_path):
                try:
                    os.remove(disk_path)
                except Exception as e:
                    logger.warning(f"Failed to remove persistent cache file for key '{key}': {str(e)}")
        
        return True
    
    def _evict_items(self, count: int = 1) -> int:
        """Evict items from the cache using LRU strategy."""
        if not self.cache:
            return 0
        
        # Sort items by last access time (oldest first)
        items = [(k, v.get('last_accessed', 0)) for k, v in self.metadata.items()]
        items.sort(key=lambda x: x[1])
        
        # Evict non-persistent items first
        evicted = 0
        for key, _ in items:
            if evicted >= count:
                break
            
            # Skip persistent items
            if self.metadata.get(key, {}).get('persist', False):
                continue
            
            # Remove item
            self._remove_item(key)
            evicted += 1
        
        # If still need to evict more, evict persistent items too
        if evicted < count:
            for key, _ in items:
                if evicted >= count:
                    break
                
                if key in self.cache:
                    self._remove_item(key)
                    evicted += 1
        
        return evicted
    
    def _check_memory_usage(self):
        """Check and manage memory usage."""
        # Only check periodically
        if time.time() - self._last_memory_check < self._memory_check_interval:
            return
        
        self._last_memory_check = time.time()
        
        # Estimate memory usage
        memory_usage = self._estimate_memory_usage()
        
        # If exceeding limit, evict items
        if memory_usage > self.max_memory_mb:
            # Calculate how many items to evict (at least 10% of cache)
            evict_count = max(int(len(self.cache) * 0.1), 1)
            
            # Evict items
            evicted = self._evict_items(evict_count)
            
            logger.info(f"Memory usage exceeded limit ({memory_usage:.1f} MB). "
                        f"Evicted {evicted} items.")
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the cache in MB."""
        import sys
        
        # Sample a subset of items for efficiency
        sample_size = min(100, len(self.cache))
        if sample_size == 0:
            return 0.0
        
        # Get a random sample of keys
        import random
        sample_keys = random.sample(list(self.cache.keys()), sample_size)
        
        # Calculate average size
        total_size = 0
        for key in sample_keys:
            # Estimate size of key and value
            key_size = sys.getsizeof(key)
            value_size = self._get_object_size(self.cache[key])
            total_size += key_size + value_size
        
        # Extrapolate to full cache
        avg_size = total_size / sample_size
        estimated_total = avg_size * len(self.cache)
        
        # Convert to MB
        return estimated_total / (1024 * 1024)
    
    def _get_object_size(self, obj: Any) -> int:
        """Get the size of an object in bytes."""
        import sys
        
        # Special handling for common types
        if isinstance(obj, (str, bytes, bytearray)):
            return sys.getsizeof(obj)
        elif isinstance(obj, (int, float, bool, type(None))):
            return sys.getsizeof(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, pd.Series):
            return obj.memory_usage(deep=True)
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sys.getsizeof(obj) + sum(self._get_object_size(x) for x in obj)
        elif isinstance(obj, dict):
            return sys.getsizeof(obj) + sum(self._get_object_size(k) + self._get_object_size(v) for k, v in obj.items())
        else:
            # Fall back to pickle size estimate
            try:
                return len(pickle.dumps(obj, protocol=4))
            except:
                return sys.getsizeof(obj)
    
    def _get_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        hit_count = sum(1 for v in self.metadata.values() if v.get('hit_count', 0) > 0)
        miss_count = sum(1 for v in self.metadata.values() if v.get('miss_count', 0) > 0)
        
        if hit_count + miss_count == 0:
            return 0.0
        
        return hit_count / (hit_count + miss_count)
    
    def _get_disk_path(self, key: str) -> str:
        """Get the disk path for a key."""
        # Hash the key to create a filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.persistent_dir, f"{key_hash}.cache")
    
    def _save_to_disk(self, key: str, value: Any) -> bool:
        """Save an item to disk."""
        if self.persistent_dir is None:
            return False
        
        disk_path = self._get_disk_path(key)
        
        try:
            # Create metadata for disk storage
            disk_metadata = {
                'key': key,
                'created_at': self.metadata[key]['created_at'],
                'expiration_time': self.metadata[key]['expiration_time'],
                'type': type(value).__name__
            }
            
            # Serialize value based on type
            if isinstance(value, pd.DataFrame):
                # Save metadata
                with open(f"{disk_path}.meta", 'w') as f:
                    json.dump(disk_metadata, f)
                
                # Save DataFrame as parquet
                value.to_parquet(disk_path)
            elif isinstance(value, pd.Series):
                # Update metadata
                disk_metadata['type'] = 'Series'
                
                # Save metadata
                with open(f"{disk_path}.meta", 'w') as f:
                    json.dump(disk_metadata, f)
                
                # Save Series as parquet
                value.to_frame().to_parquet(disk_path)
            else:
                # For other types, use pickle
                with open(disk_path, 'wb') as f:
                    # Save metadata and value together
                    data = {'metadata': disk_metadata, 'value': value}
                    pickle.dump(data, f, protocol=4)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save key '{key}' to disk cache: {str(e)}")
            return False
    
    def _load_from_disk(self, key: str) -> bool:
        """Load an item from disk."""
        if self.persistent_dir is None:
            return False
        
        disk_path = self._get_disk_path(key)
        meta_path = f"{disk_path}.meta"
        
        # Check if the file exists
        if not os.path.exists(disk_path):
            return False
        
        try:
            # Check if it's a parquet file (DataFrame or Series)
            if os.path.exists(meta_path):
                # Load metadata
                with open(meta_path, 'r') as f:
                    disk_metadata = json.load(f)
                
                # Check if expired
                if disk_metadata.get('expiration_time') is not None:
                    if time.time() > disk_metadata['expiration_time']:
                        # Remove expired files
                        os.remove(disk_path)
                        os.remove(meta_path)
                        return False
                
                # Load value based on type
                if disk_metadata.get('type') == 'DataFrame':
                    value = pd.read_parquet(disk_path)
                elif disk_metadata.get('type') == 'Series':
                    value = pd.read_parquet(disk_path).iloc[:, 0]
                else:
                    # Unknown type
                    return False
                
                # Update cache
                self.cache[key] = value
                
                # Update metadata
                self.metadata[key] = {
                    'created_at': disk_metadata.get('created_at', time.time()),
                    'last_accessed': time.time(),
                    'expiration_time': disk_metadata.get('expiration_time'),
                    'persist': True
                }
                
                return True
            
            else:
                # Load using pickle
                with open(disk_path, 'rb') as f:
                    data = pickle.load(f)
                
                disk_metadata = data['metadata']
                value = data['value']
                
                # Check if expired
                if disk_metadata.get('expiration_time') is not None:
                    if time.time() > disk_metadata['expiration_time']:
                        # Remove expired file
                        os.remove(disk_path)
                        return False
                
                # Update cache
                self.cache[key] = value
                
                # Update metadata
                self.metadata[key] = {
                    'created_at': disk_metadata.get('created_at', time.time()),
                    'last_accessed': time.time(),
                    'expiration_time': disk_metadata.get('expiration_time'),
                    'persist': True
                }
                
                return True
        
        except Exception as e:
            logger.error(f"Failed to load key '{key}' from disk cache: {str(e)}")
            return False 