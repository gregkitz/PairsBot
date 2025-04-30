"""
Parallel Executor for the Intraday Statistical Arbitrage System.

This module provides classes and functions for executing tasks in parallel
using process-based or thread-based parallelism.
"""

import multiprocessing as mp
import concurrent.futures
import os
import time
import logging
from typing import Callable, List, Any, Dict, Union, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
from functools import partial

# Set up logging
logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Parallel Executor for running tasks in parallel.
    
    This class provides methods for executing tasks in parallel using either
    process-based or thread-based parallelism, with support for progress tracking
    and error handling.
    """
    
    def __init__(self,
                n_jobs: int = None,
                backend: str = 'process',
                show_progress: bool = True,
                timeout: Optional[float] = None):
        """
        Initialize the parallel executor.
        
        Parameters:
        -----------
        n_jobs : int, optional
            Number of parallel jobs to run. If None, uses the number of CPU cores.
        backend : str
            Parallelism backend ('process' or 'thread')
        show_progress : bool
            Whether to show progress during execution
        timeout : float, optional
            Timeout for each task in seconds
        """
        self.n_jobs = n_jobs if n_jobs is not None else max(1, mp.cpu_count() - 1)
        self.backend = backend
        self.show_progress = show_progress
        self.timeout = timeout
        
        # Validate parameters
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer")
        
        if self.backend not in ['process', 'thread']:
            raise ValueError(f"Unknown backend: {backend}. Use 'process' or 'thread'.")
    
    def map(self, 
           func: Callable, 
           items: Iterable, 
           *args, 
           **kwargs) -> List[Any]:
        """
        Apply a function to each item in parallel.
        
        This is similar to the built-in map function, but it runs in parallel.
        
        Parameters:
        -----------
        func : callable
            Function to apply to each item
        items : iterable
            Items to process
        *args, **kwargs
            Additional arguments to pass to func
        
        Returns:
        --------
        list
            List of results
        """
        # Convert items to list to get length for progress reporting
        items_list = list(items)
        n_items = len(items_list)
        
        if n_items == 0:
            return []
        
        # Create a partial function with the additional arguments
        if args or kwargs:
            func_with_args = partial(func, *args, **kwargs)
        else:
            func_with_args = func
        
        start_time = time.time()
        
        if self.backend == 'process':
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:  # thread
            executor_class = concurrent.futures.ThreadPoolExecutor
        
        # Execute in parallel
        results = []
        with executor_class(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(func_with_args, item): i for i, item in enumerate(items_list)}
            
            # Process as completed
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                
                try:
                    # Get result with timeout if specified
                    if self.timeout is not None:
                        result = future.result(timeout=self.timeout)
                    else:
                        result = future.result()
                    
                    # Add to results
                    results.append((idx, result))
                
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Task {idx} timed out after {self.timeout} seconds")
                    results.append((idx, None))
                
                except Exception as e:
                    logger.error(f"Task {idx} generated an exception: {str(e)}")
                    results.append((idx, None))
                
                # Update progress
                completed += 1
                if self.show_progress and n_items > 1:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (n_items - completed) * avg_time
                    
                    print(f"\rProgress: {completed}/{n_items} ({completed/n_items*100:.1f}%) - "
                          f"Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s", end="")
        
        if self.show_progress and n_items > 1:
            total_time = time.time() - start_time
            print(f"\rCompleted {n_items} tasks in {total_time:.1f} seconds "
                  f"({n_items/total_time:.1f} tasks/s)")
        
        # Sort results by index
        results.sort(key=lambda x: x[0])
        
        # Return only the results, not the indices
        return [r for _, r in results]
    
    def execute(self, 
               tasks: List[Callable], 
               *args, 
               **kwargs) -> List[Any]:
        """
        Execute a list of tasks in parallel.
        
        Parameters:
        -----------
        tasks : list of callable
            Tasks to execute
        *args, **kwargs
            Additional arguments to pass to each task
        
        Returns:
        --------
        list
            List of results
        """
        # Define a wrapper function that takes a task and executes it
        def execute_task(task):
            return task(*args, **kwargs)
        
        # Use map to execute tasks in parallel
        return self.map(execute_task, tasks)


def parallel_map(func: Callable, 
                items: Iterable, 
                n_jobs: int = None, 
                backend: str = 'process', 
                show_progress: bool = True, 
                timeout: Optional[float] = None, 
                *args, 
                **kwargs) -> List[Any]:
    """
    Apply a function to each item in parallel.
    
    This is a convenience function that creates a ParallelExecutor and calls its
    map method.
    
    Parameters:
    -----------
    func : callable
        Function to apply to each item
    items : iterable
        Items to process
    n_jobs : int, optional
        Number of parallel jobs to run. If None, uses the number of CPU cores.
    backend : str
        Parallelism backend ('process' or 'thread')
    show_progress : bool
        Whether to show progress during execution
    timeout : float, optional
        Timeout for each task in seconds
    *args, **kwargs
        Additional arguments to pass to func
    
    Returns:
    --------
    list
        List of results
    """
    executor = ParallelExecutor(
        n_jobs=n_jobs,
        backend=backend,
        show_progress=show_progress,
        timeout=timeout
    )
    
    return executor.map(func, items, *args, **kwargs) 