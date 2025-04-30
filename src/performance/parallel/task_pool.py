"""
Task Pool for the Intraday Statistical Arbitrage System.

This module provides a task pool for managing long-running parallel tasks
with automatic resource management and task prioritization.
"""

import multiprocessing as mp
import threading
import queue
import time
import logging
import uuid
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor

# Set up logging
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a task in the task pool."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Task metadata for the task pool."""
    id: str
    name: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    future: Optional[Future] = None


class TaskPool:
    """
    Task Pool for managing long-running parallel tasks.
    
    This class provides methods for submitting, managing, and monitoring tasks
    with support for prioritization, cancellation, and automatic resource management.
    """
    
    def __init__(self,
                max_workers: int = None,
                backend: str = 'process',
                max_queue_size: int = 100,
                keep_completed: int = 100):
        """
        Initialize the task pool.
        
        Parameters:
        -----------
        max_workers : int, optional
            Maximum number of worker processes/threads. If None, uses CPU count.
        backend : str
            Execution backend ('process' or 'thread')
        max_queue_size : int
            Maximum size of the task queue
        keep_completed : int
            Maximum number of completed tasks to keep in history
        """
        self.max_workers = max_workers if max_workers is not None else max(1, mp.cpu_count() - 1)
        self.backend = backend
        self.max_queue_size = max_queue_size
        self.keep_completed = keep_completed
        
        # Create task queue with priority
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Dictionary to store task metadata
        self.tasks = {}
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Create executor
        if self.backend == 'process':
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:  # thread
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Create worker thread for processing tasks
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def submit(self,
              func: Callable,
              *args,
              name: str = None,
              priority: Union[TaskPriority, int, str] = TaskPriority.NORMAL,
              **kwargs) -> str:
        """
        Submit a task for execution.
        
        Parameters:
        -----------
        func : callable
            Function to execute
        *args
            Arguments to pass to the function
        name : str, optional
            Name for the task. If None, uses function name.
        priority : TaskPriority, int, or str
            Priority for the task
        **kwargs
            Keyword arguments to pass to the function
        
        Returns:
        --------
        str
            Task ID
        
        Raises:
        -------
        ValueError
            If the task queue is full
        """
        with self.lock:
            # Create task ID
            task_id = str(uuid.uuid4())
            
            # Convert priority if needed
            if isinstance(priority, int):
                priority = list(TaskPriority)[priority] if 0 <= priority < len(TaskPriority) else TaskPriority.NORMAL
            elif isinstance(priority, str):
                try:
                    priority = TaskPriority[priority.upper()]
                except KeyError:
                    priority = TaskPriority.NORMAL
            
            # Create task
            task = Task(
                id=task_id,
                name=name if name is not None else func.__name__,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=time.time()
            )
            
            # Store task
            self.tasks[task_id] = task
            
            # Add to queue with priority (negative for priority queue to give higher values higher priority)
            try:
                self.task_queue.put_nowait((-task.priority.value, task_id))
                logger.debug(f"Task {task_id} ({task.name}) submitted with priority {priority}")
                return task_id
            except queue.Full:
                del self.tasks[task_id]
                raise ValueError("Task queue is full")
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Parameters:
        -----------
        task_id : str
            ID of the task to cancel
        
        Returns:
        --------
        bool
            True if the task was cancelled, False otherwise
        """
        with self.lock:
            # Check if task exists
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return False
            
            task = self.tasks[task_id]
            
            # Check if task can be cancelled
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.warning(f"Task {task_id} is already {task.status.value}")
                return False
            
            # Cancel future if running
            if task.status == TaskStatus.RUNNING and task.future is not None:
                cancelled = task.future.cancel()
                if cancelled:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"Task {task_id} ({task.name}) cancelled")
                    return True
                else:
                    logger.warning(f"Failed to cancel task {task_id} ({task.name})")
                    return False
            
            # If pending, just mark as cancelled
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} ({task.name}) cancelled")
            return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task.
        
        Parameters:
        -----------
        task_id : str
            ID of the task
        
        Returns:
        --------
        dict or None
            Dictionary with task information, or None if not found
        """
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            # Calculate duration if applicable
            duration = None
            if task.started_at is not None:
                if task.completed_at is not None:
                    duration = task.completed_at - task.started_at
                elif task.status == TaskStatus.RUNNING:
                    duration = time.time() - task.started_at
            
            return {
                'id': task.id,
                'name': task.name,
                'status': task.status.value,
                'priority': task.priority.name,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'duration': duration,
                'result': task.result if task.status == TaskStatus.COMPLETED else None,
                'error': str(task.error) if task.error is not None else None
            }
    
    def get_result(self, task_id: str, wait: bool = False, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task.
        
        Parameters:
        -----------
        task_id : str
            ID of the task
        wait : bool
            Whether to wait for the task to complete
        timeout : float, optional
            Timeout in seconds to wait for completion
        
        Returns:
        --------
        any
            Task result, or None if not available
        
        Raises:
        -------
        Exception
            If the task failed and wait is True
        TimeoutError
            If the timeout was reached
        """
        with self.lock:
            if task_id not in self.tasks:
                return None
            
            task = self.tasks[task_id]
            
            # If not waiting or task is already done, return result
            if not wait or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                if task.status == TaskStatus.FAILED and wait:
                    raise task.error or Exception(f"Task {task_id} failed")
                return task.result
        
        # Wait for completion outside the lock to avoid deadlocks
        if wait and task.future is not None:
            try:
                result = task.future.result(timeout=timeout)
                return result
            except Exception as e:
                if isinstance(e, concurrent.futures.TimeoutError):
                    raise TimeoutError(f"Timeout waiting for task {task_id}")
                raise
        
        return None
    
    def list_tasks(self, 
                  status: Optional[Union[TaskStatus, str]] = None, 
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        List tasks in the pool.
        
        Parameters:
        -----------
        status : TaskStatus or str, optional
            Filter tasks by status
        limit : int
            Maximum number of tasks to return
        
        Returns:
        --------
        list
            List of task dictionaries
        """
        with self.lock:
            # Convert status if needed
            if isinstance(status, str):
                try:
                    status = TaskStatus(status.lower())
                except ValueError:
                    status = None
            
            # Filter and sort tasks
            tasks = list(self.tasks.values())
            
            if status is not None:
                tasks = [t for t in tasks if t.status == status]
            
            # Sort by created_at (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            
            # Limit number of tasks
            tasks = tasks[:limit]
            
            # Convert to dictionaries
            return [self.get_task(t.id) for t in tasks]
    
    def clear_completed(self, 
                       max_age: Optional[float] = None, 
                       keep_last: int = None) -> int:
        """
        Clear completed, failed, or cancelled tasks from the pool.
        
        Parameters:
        -----------
        max_age : float, optional
            Maximum age in seconds of tasks to keep
        keep_last : int, optional
            Number of recent tasks to keep
        
        Returns:
        --------
        int
            Number of tasks cleared
        """
        with self.lock:
            # Find completed tasks
            completed_ids = [
                task_id for task_id, task in self.tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            ]
            
            # Keep recent tasks if requested
            if keep_last is not None and keep_last > 0:
                # Sort by completion time (newest first)
                completed_ids.sort(
                    key=lambda tid: self.tasks[tid].completed_at or 0,
                    reverse=True
                )
                
                # Keep the most recent ones
                to_keep = completed_ids[:keep_last]
                to_remove = completed_ids[keep_last:]
            else:
                to_keep = []
                to_remove = completed_ids
            
            # Filter by age if requested
            if max_age is not None:
                current_time = time.time()
                to_remove = [
                    task_id for task_id in to_remove
                    if (self.tasks[task_id].completed_at or 0) < current_time - max_age
                ]
            
            # Remove tasks
            for task_id in to_remove:
                del self.tasks[task_id]
            
            return len(to_remove)
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the task pool.
        
        Parameters:
        -----------
        wait : bool
            Whether to wait for all tasks to complete
        """
        # Signal worker to stop
        self.stop_event.set()
        
        # Wait for worker to finish
        if wait:
            self.worker_thread.join()
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        logger.info("Task pool shut down")
    
    def _worker_loop(self):
        """Worker thread that processes tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                # Get next task from queue with timeout to check stop_event periodically
                try:
                    _, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get task
                with self.lock:
                    if task_id not in self.tasks:
                        logger.warning(f"Task {task_id} not found in pool")
                        continue
                    
                    task = self.tasks[task_id]
                    
                    # Skip cancelled tasks
                    if task.status == TaskStatus.CANCELLED:
                        self.task_queue.task_done()
                        continue
                    
                    # Mark as running
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()
                
                # Submit to executor
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                
                # Store future
                with self.lock:
                    task.future = future
                
                # Add callback to handle completion
                future.add_done_callback(lambda f, tid=task_id: self._task_completed(tid, f))
                
                logger.debug(f"Task {task_id} ({task.name}) started")
            
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                # Sleep briefly to avoid hammering in case of persistent errors
                time.sleep(0.1)
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle completion of a task."""
        with self.lock:
            if task_id not in self.tasks:
                logger.warning(f"Completed task {task_id} not found in pool")
                return
            
            task = self.tasks[task_id]
            task.completed_at = time.time()
            
            try:
                # Get result
                task.result = future.result()
                task.status = TaskStatus.COMPLETED
                logger.debug(f"Task {task_id} ({task.name}) completed successfully")
            
            except Exception as e:
                # Handle failure
                task.error = e
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task_id} ({task.name}) failed: {str(e)}")
            
            # Mark as done in the queue
            self.task_queue.task_done()
            
            # Clean up old completed tasks if needed
            if self.keep_completed > 0:
                completed_tasks = [t for t in self.tasks.values() if 
                                  t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
                
                if len(completed_tasks) > self.keep_completed:
                    # Sort by completion time (oldest first)
                    completed_tasks.sort(key=lambda t: t.completed_at or 0)
                    
                    # Remove oldest ones
                    to_remove = len(completed_tasks) - self.keep_completed
                    for t in completed_tasks[:to_remove]:
                        if t.id in self.tasks:
                            del self.tasks[t.id] 