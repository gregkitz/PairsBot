"""
Task Orchestration System for Intraday ML Trading

This module provides a robust framework for coordinating and scheduling various 
automation tasks for the intraday ML trading system with proper error handling,
logging, and retry capabilities.
"""

import logging
import time
import datetime
import os
import sys
import traceback
import schedule
from functools import wraps
from typing import Callable, Dict, List, Optional, Union, Any

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# Import project modules
from utils.logging_utils import setup_logger
from config.config_manager import ConfigManager

# Setup logger
logger = setup_logger('task_orchestrator', log_level=logging.INFO)

class RetryPolicy:
    """Defines retry behavior for tasks that fail"""
    
    def __init__(self, max_retries: int = 3, 
                 retry_delay_seconds: int = 60,
                 backoff_factor: float = 2.0,
                 exceptions_to_retry: List[Exception] = None):
        """
        Initialize retry policy
        
        Parameters:
        -----------
        max_retries : int
            Maximum number of retry attempts
        retry_delay_seconds : int
            Initial delay between retries in seconds
        backoff_factor : float
            Multiplier for delay between consecutive retries
        exceptions_to_retry : List[Exception]
            List of exceptions that should trigger a retry
        """
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.backoff_factor = backoff_factor
        self.exceptions_to_retry = exceptions_to_retry or [Exception]
        
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if retry should be attempted based on exception and attempt count"""
        if attempt >= self.max_retries:
            return False
            
        return any(isinstance(exception, exc) for exc in self.exceptions_to_retry)
        
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the current retry attempt"""
        return self.retry_delay_seconds * (self.backoff_factor ** (attempt - 1))


class Task:
    """Represents a schedulable task with error handling and retry capability"""
    
    def __init__(self, 
                 name: str,
                 function: Callable,
                 retry_policy: RetryPolicy = None,
                 dependencies: List[str] = None,
                 timeout_seconds: int = None,
                 critical: bool = False):
        """
        Initialize a task
        
        Parameters:
        -----------
        name : str
            Name of the task
        function : Callable
            Function to execute
        retry_policy : RetryPolicy
            Policy for retrying on failure
        dependencies : List[str]
            Names of tasks that must complete before this task
        timeout_seconds : int
            Maximum time allowed for task execution
        critical : bool
            If True, system will abort if task fails after all retries
        """
        self.name = name
        self.function = function
        self.retry_policy = retry_policy or RetryPolicy()
        self.dependencies = dependencies or []
        self.timeout_seconds = timeout_seconds
        self.critical = critical
        self.last_run_time = None
        self.last_run_status = None
        self.last_run_error = None
        
    def execute(self, task_context: Dict = None) -> bool:
        """
        Execute the task with proper error handling and retry logic
        
        Parameters:
        -----------
        task_context : Dict
            Context data to be passed to the function
            
        Returns:
        --------
        bool
            True if task executed successfully, False otherwise
        """
        context = task_context or {}
        start_time = time.time()
        self.last_run_time = datetime.datetime.now()
        
        logger.info(f"Starting task: {self.name}")
        
        for attempt in range(1, self.retry_policy.max_retries + 1):
            try:
                result = self.function(**context)
                execution_time = time.time() - start_time
                logger.info(f"Task {self.name} completed successfully in {execution_time:.2f}s")
                self.last_run_status = "SUCCESS"
                return True
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Task {self.name} failed on attempt {attempt}/{self.retry_policy.max_retries} "
                            f"after {execution_time:.2f}s with error: {str(e)}")
                logger.error(traceback.format_exc())
                self.last_run_error = str(e)
                
                if self.retry_policy.should_retry(e, attempt):
                    delay = self.retry_policy.get_delay(attempt)
                    logger.info(f"Retrying task {self.name} in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    break
        
        # All retries failed
        self.last_run_status = "FAILED"
        if self.critical:
            logger.critical(f"Critical task {self.name} failed after {self.retry_policy.max_retries} attempts, aborting.")
            # In production, you might want to add additional alert mechanisms here
            # such as sending emails, SMS, etc.
        else:
            logger.error(f"Task {self.name} failed after {self.retry_policy.max_retries} attempts, continuing with other tasks.")
            
        return False


class TaskOrchestrator:
    """
    Coordinates the execution of tasks based on schedule and dependencies
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the task orchestrator
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        self.tasks = {}  # name -> Task
        self.scheduled_jobs = {}  # name -> schedule.Job
        self.global_context = {}
        
        # Load configuration if provided
        if config_path:
            self.config = ConfigManager(config_path)
        else:
            self.config = None
            
    def register_task(self, task: Task) -> None:
        """
        Register a task with the orchestrator
        
        Parameters:
        -----------
        task : Task
            Task to register
        """
        if task.name in self.tasks:
            logger.warning(f"Task {task.name} already registered, overwriting")
            
        self.tasks[task.name] = task
        logger.info(f"Registered task: {task.name}")
        
    def register_function(self,
                         name: str,
                         function: Callable,
                         retry_policy: RetryPolicy = None,
                         dependencies: List[str] = None,
                         timeout_seconds: int = None,
                         critical: bool = False) -> Task:
        """
        Register a function as a task
        
        Parameters:
        -----------
        name : str
            Name of the task
        function : Callable
            Function to execute
        retry_policy : RetryPolicy
            Policy for retrying on failure
        dependencies : List[str]
            Names of tasks that must complete before this task
        timeout_seconds : int
            Maximum time allowed for task execution
        critical : bool
            If True, system will abort if task fails after all retries
            
        Returns:
        --------
        Task
            The registered task
        """
        task = Task(name, function, retry_policy, dependencies, timeout_seconds, critical)
        self.register_task(task)
        return task
    
    def schedule_task(self, task_name: str, schedule_spec: str) -> None:
        """
        Schedule a task for execution
        
        Parameters:
        -----------
        task_name : str
            Name of the task to schedule
        schedule_spec : str
            Schedule specification (e.g., 'daily at 10:00')
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not registered")
            
        task = self.tasks[task_name]
        
        # Parse schedule_spec and set up the schedule
        schedule_parts = schedule_spec.split()
        
        if schedule_parts[0] == 'daily' and schedule_parts[1] == 'at':
            job = schedule.every().day.at(schedule_parts[2]).do(self._execute_task_with_dependencies, task_name)
        elif schedule_parts[0] == 'weekly' and schedule_parts[1] == 'on':
            day = schedule_parts[2].lower()
            if schedule_parts[3] == 'at':
                time_spec = schedule_parts[4]
                
                # Map day string to schedule method
                if day == 'monday':
                    job = schedule.every().monday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'tuesday':
                    job = schedule.every().tuesday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'wednesday':
                    job = schedule.every().wednesday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'thursday':
                    job = schedule.every().thursday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'friday':
                    job = schedule.every().friday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'saturday':
                    job = schedule.every().saturday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                elif day == 'sunday':
                    job = schedule.every().sunday.at(time_spec).do(self._execute_task_with_dependencies, task_name)
                else:
                    raise ValueError(f"Invalid day of week: {day}")
        elif schedule_parts[0] == 'every' and schedule_parts[2] == 'minutes':
            interval = int(schedule_parts[1])
            job = schedule.every(interval).minutes.do(self._execute_task_with_dependencies, task_name)
        elif schedule_parts[0] == 'every' and schedule_parts[2] == 'hours':
            interval = int(schedule_parts[1])
            job = schedule.every(interval).hours.do(self._execute_task_with_dependencies, task_name)
        elif schedule_parts[0] == 'every' and schedule_parts[2] == 'seconds':
            interval = int(schedule_parts[1])
            job = schedule.every(interval).seconds.do(self._execute_task_with_dependencies, task_name)
        else:
            raise ValueError(f"Unsupported schedule specification: {schedule_spec}")
            
        self.scheduled_jobs[task_name] = job
        logger.info(f"Scheduled task {task_name}: {schedule_spec}")
        
    def set_global_context(self, context: Dict) -> None:
        """
        Set global context data available to all tasks
        
        Parameters:
        -----------
        context : Dict
            Key-value pairs to add to global context
        """
        self.global_context.update(context)
        
    def _execute_task_with_dependencies(self, task_name: str) -> bool:
        """
        Execute a task after ensuring all its dependencies are met
        
        Parameters:
        -----------
        task_name : str
            Name of the task to execute
            
        Returns:
        --------
        bool
            True if task executed successfully, False otherwise
        """
        task = self.tasks[task_name]
        
        # Check if dependencies are satisfied
        for dep_name in task.dependencies:
            if dep_name not in self.tasks:
                logger.error(f"Dependency {dep_name} for task {task_name} not registered")
                return False
                
            dep_task = self.tasks[dep_name]
            if dep_task.last_run_status != "SUCCESS":
                logger.error(f"Dependency {dep_name} for task {task_name} not satisfied (status: {dep_task.last_run_status})")
                return False
                
        # Execute the task with the global context
        return task.execute(self.global_context)
        
    def execute_task(self, task_name: str, context: Dict = None) -> bool:
        """
        Manually execute a task
        
        Parameters:
        -----------
        task_name : str
            Name of the task to execute
        context : Dict
            Additional context to merge with global context
            
        Returns:
        --------
        bool
            True if task executed successfully, False otherwise
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not registered")
            
        task_context = self.global_context.copy()
        if context:
            task_context.update(context)
            
        return self._execute_task_with_dependencies(task_name)
        
    def run_scheduler(self, block: bool = True) -> None:
        """
        Run the scheduler
        
        Parameters:
        -----------
        block : bool
            If True, will block and run continuously; if False, will run pending tasks once
        """
        logger.info("Starting task scheduler")
        
        try:
            if block:
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            else:
                schedule.run_pending()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
            logger.error(traceback.format_exc())
            
    def get_task_status(self, task_name: str = None) -> Dict:
        """
        Get status information for tasks
        
        Parameters:
        -----------
        task_name : str
            Name of specific task to get status for, or None for all tasks
            
        Returns:
        --------
        Dict
            Status information for the specified task(s)
        """
        if task_name:
            if task_name not in self.tasks:
                raise ValueError(f"Task {task_name} not registered")
                
            task = self.tasks[task_name]
            return {
                'name': task.name,
                'last_run_time': task.last_run_time,
                'last_run_status': task.last_run_status,
                'last_run_error': task.last_run_error,
                'critical': task.critical,
                'dependencies': task.dependencies
            }
        else:
            # Return status for all tasks
            return {name: {
                'name': task.name,
                'last_run_time': task.last_run_time,
                'last_run_status': task.last_run_status,
                'last_run_error': task.last_run_error,
                'critical': task.critical,
                'dependencies': task.dependencies
            } for name, task in self.tasks.items()}
            
    def load_tasks_from_config(self, config_file: str) -> None:
        """
        Load task definitions from a configuration file
        
        Parameters:
        -----------
        config_file : str
            Path to configuration file
        """
        # Implementation would load task definitions from a YAML or JSON file
        # and register them with the orchestrator
        pass


# Example usage
if __name__ == "__main__":
    # Create an orchestrator
    orchestrator = TaskOrchestrator()
    
    # Define a simple task
    def example_task(param1=None, param2=None):
        logger.info(f"Running example task with params: {param1}, {param2}")
        return True
        
    # Create a retry policy
    retry_policy = RetryPolicy(max_retries=3, retry_delay_seconds=5)
    
    # Register the task
    orchestrator.register_function(
        name="example_task",
        function=example_task,
        retry_policy=retry_policy,
        critical=True
    )
    
    # Set context
    orchestrator.set_global_context({'param1': 'value1', 'param2': 'value2'})
    
    # Schedule the task
    orchestrator.schedule_task("example_task", "every 10 seconds")
    
    # Run the scheduler
    orchestrator.run_scheduler(block=False)  # Run once
    
    # Get task status
    status = orchestrator.get_task_status("example_task")
    logger.info(f"Task status: {status}") 