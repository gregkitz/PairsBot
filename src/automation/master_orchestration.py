"""
Master Orchestration Script for Intraday ML Trading System

This script sets up and coordinates all automated processes for the intraday ML trading system,
including task scheduling, dependency management, and error handling.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Callable

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# Import project modules
from automation.task_orchestrator import TaskOrchestrator, RetryPolicy
from utils.logging_utils import setup_logger
from config.config_manager import ConfigManager

# Setup logger
logger = setup_logger('master_orchestration', log_level=logging.INFO)

class MasterOrchestration:
    """
    Manages and orchestrates all automated processes for the intraday ML trading system
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the master orchestration system
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        # Create task orchestrator
        self.orchestrator = TaskOrchestrator(config_path)
        
        # Load config if provided
        if config_path:
            self.config = ConfigManager(config_path)
        else:
            # Default configuration location
            config_dir = os.path.join(src_dir, 'config')
            default_config = os.path.join(config_dir, 'automation_config.yaml')
            if os.path.exists(default_config):
                self.config = ConfigManager(default_config)
            else:
                logger.warning(f"No configuration file found at {default_config}")
                self.config = None
                
        # Initialize task registry
        self.setup_tasks()
        
    def setup_tasks(self):
        """
        Set up all tasks for the trading system
        """
        # Register critical workflow tasks
        self._register_data_tasks()
        self._register_analysis_tasks()
        self._register_trading_tasks()
        self._register_monitoring_tasks()
        self._register_maintenance_tasks()
        
        # Set dependencies to ensure proper execution order
        self._set_task_dependencies()
        
        # Schedule tasks based on configuration or defaults
        self._schedule_tasks()
        
    def _register_data_tasks(self):
        """Register data collection and processing tasks"""
        # Data collection task
        retry_policy = RetryPolicy(max_retries=5, retry_delay_seconds=60)
        
        # Import data collection function
        try:
            from data_processor.data_loader import download_historical_data
            
            self.orchestrator.register_function(
                name="download_data",
                function=download_historical_data,
                retry_policy=retry_policy,
                critical=True
            )
            logger.info("Registered data collection task")
        except ImportError as e:
            logger.error(f"Failed to import data collection module: {str(e)}")
            
        # Feature generation task
        try:
            from data_processor.feature_calculator import generate_features
            
            self.orchestrator.register_function(
                name="generate_features",
                function=generate_features,
                retry_policy=retry_policy,
                dependencies=["download_data"],
                critical=True
            )
            logger.info("Registered feature generation task")
        except ImportError as e:
            logger.error(f"Failed to import feature calculator module: {str(e)}")
            
    def _register_analysis_tasks(self):
        """Register analysis and ML model related tasks"""
        # Register pair analysis task
        retry_policy = RetryPolicy(max_retries=3, retry_delay_seconds=120)
        
        try:
            from cointegration.pair_selector import analyze_pairs
            
            self.orchestrator.register_function(
                name="analyze_pairs",
                function=analyze_pairs,
                retry_policy=retry_policy,
                dependencies=["generate_features"],
                critical=False  # Not critical as trading can continue with existing pairs
            )
            logger.info("Registered pair analysis task")
        except ImportError as e:
            logger.error(f"Failed to import pair selector module: {str(e)}")
            
        # Register ML model training task
        try:
            from ml_enhancements.spread_prediction.spread_predictor import train_models
            
            self.orchestrator.register_function(
                name="train_ml_models",
                function=train_models,
                retry_policy=retry_policy,
                dependencies=["generate_features"],
                critical=False  # Not critical as trading can continue with existing models
            )
            logger.info("Registered ML model training task")
        except ImportError as e:
            logger.error(f"Failed to import ML model training module: {str(e)}")
            
        # Register regime detection task
        try:
            from ml_enhancements.regime_detection.regime_detector import detect_current_regime
            
            self.orchestrator.register_function(
                name="detect_regime",
                function=detect_current_regime,
                retry_policy=retry_policy,
                dependencies=["generate_features"],
                critical=True  # Critical as trading strategies depend on regime
            )
            logger.info("Registered regime detection task")
        except ImportError as e:
            logger.error(f"Failed to import regime detector module: {str(e)}")
            
        # Register parameter optimization task
        try:
            from optimization.intraday_parameter_optimizer import optimize_parameters
            
            self.orchestrator.register_function(
                name="optimize_parameters",
                function=optimize_parameters,
                retry_policy=retry_policy,
                dependencies=["detect_regime"],
                critical=False  # Not critical as trading can continue with existing parameters
            )
            logger.info("Registered parameter optimization task")
        except ImportError as e:
            logger.error(f"Failed to import parameter optimizer module: {str(e)}")
            
    def _register_trading_tasks(self):
        """Register trading execution tasks"""
        # Register paper trading task
        retry_policy = RetryPolicy(max_retries=5, retry_delay_seconds=30)
        
        try:
            from paper_trading.intraday_ml_paper_trader import run_paper_trading
            
            self.orchestrator.register_function(
                name="run_paper_trading",
                function=run_paper_trading,
                retry_policy=retry_policy,
                dependencies=["detect_regime", "generate_features"],
                critical=True  # Critical as this is the primary system function
            )
            logger.info("Registered paper trading task")
        except ImportError as e:
            logger.error(f"Failed to import paper trading module: {str(e)}")
            
    def _register_monitoring_tasks(self):
        """Register monitoring and reporting tasks"""
        # Register system health check task
        retry_policy = RetryPolicy(max_retries=3, retry_delay_seconds=30)
        
        try:
            from monitoring.system_monitor import check_system_health
            
            self.orchestrator.register_function(
                name="check_system_health",
                function=check_system_health,
                retry_policy=retry_policy,
                critical=False  # Not critical but important
            )
            logger.info("Registered system health check task")
        except ImportError as e:
            logger.error(f"Failed to import system monitor module: {str(e)}")
            
        # Register performance reporting task
        try:
            from reporting.reporting_framework import generate_performance_report
            
            self.orchestrator.register_function(
                name="generate_performance_report",
                function=generate_performance_report,
                retry_policy=retry_policy,
                dependencies=["run_paper_trading"],
                critical=False  # Not critical but important for analysis
            )
            logger.info("Registered performance reporting task")
        except ImportError as e:
            logger.error(f"Failed to import reporting framework module: {str(e)}")
            
    def _register_maintenance_tasks(self):
        """Register system maintenance tasks"""
        # Register log file cleanup task
        retry_policy = RetryPolicy(max_retries=2, retry_delay_seconds=60)
        
        def cleanup_logs():
            """Simple log cleanup function"""
            # Implementation would clean up old log files
            logger.info("Cleaning up log files")
            return True
            
        self.orchestrator.register_function(
            name="cleanup_logs",
            function=cleanup_logs,
            retry_policy=retry_policy,
            critical=False  # Not critical
        )
        logger.info("Registered log cleanup task")
        
        # Database maintenance task could be added here if needed
        
    def _set_task_dependencies(self):
        """Set up additional task dependencies beyond those specified during registration"""
        # Additional dependencies can be set here if needed
        # For example, if we need to add a dependency that wasn't known during initial registration
        pass
        
    def _schedule_tasks(self):
        """Schedule all tasks based on configuration or defaults"""
        # If we have a configuration, use it for scheduling
        if self.config:
            try:
                task_schedules = self.config.get('task_schedules', {})
                for task_name, schedule_spec in task_schedules.items():
                    if task_name in self.orchestrator.tasks:
                        self.orchestrator.schedule_task(task_name, schedule_spec)
                    else:
                        logger.warning(f"Cannot schedule unknown task: {task_name}")
            except Exception as e:
                logger.error(f"Error loading task schedules from config: {str(e)}")
                # Fall back to default scheduling
                self._set_default_schedules()
        else:
            # Use default schedules
            self._set_default_schedules()
            
    def _set_default_schedules(self):
        """Set default schedules for tasks when no configuration is available"""
        # Data tasks
        if "download_data" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("download_data", "daily at 17:00")
            
        if "generate_features" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("generate_features", "daily at 17:30")
            
        # Analysis tasks
        if "analyze_pairs" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("analyze_pairs", "weekly on sunday at 18:00")
            
        if "train_ml_models" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("train_ml_models", "weekly on sunday at 19:00")
            
        if "detect_regime" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("detect_regime", "daily at 07:30")
            
        if "optimize_parameters" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("optimize_parameters", "weekly on sunday at 20:00")
            
        # Trading tasks
        if "run_paper_trading" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("run_paper_trading", "daily at 08:00")
            
        # Monitoring tasks
        if "check_system_health" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("check_system_health", "daily at 07:00")
            
        if "generate_performance_report" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("generate_performance_report", "daily at 17:45")
            
        # Maintenance tasks
        if "cleanup_logs" in self.orchestrator.tasks:
            self.orchestrator.schedule_task("cleanup_logs", "weekly on sunday at 23:00")
            
    def start(self, block: bool = True):
        """
        Start the orchestration system
        
        Parameters:
        -----------
        block : bool
            If True, will block and run continuously; if False, will run pending tasks once
        """
        logger.info("Starting master orchestration system")
        self.orchestrator.run_scheduler(block=block)
        
    def execute_task_now(self, task_name: str, context: Dict = None):
        """
        Execute a task immediately
        
        Parameters:
        -----------
        task_name : str
            Name of the task to execute
        context : Dict
            Additional context data for the task
            
        Returns:
        --------
        bool
            True if task executed successfully, False otherwise
        """
        return self.orchestrator.execute_task(task_name, context)
        
    def get_system_status(self):
        """
        Get status information for all tasks
        
        Returns:
        --------
        Dict
            Status information for all tasks
        """
        return self.orchestrator.get_task_status()
        
    def reload_config(self, config_path: str = None):
        """
        Reload configuration
        
        Parameters:
        -----------
        config_path : str
            Path to new configuration file, or None to reload the original
        """
        if config_path:
            self.config = ConfigManager(config_path)
        elif self.config:
            self.config.reload()
            
        # Re-schedule tasks based on new config
        self._schedule_tasks()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Master orchestration for intraday ML trading system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--run-once', action='store_true', help='Run pending tasks once and exit')
    parser.add_argument('--execute', type=str, help='Execute a specific task and exit')
    parser.add_argument('--status', action='store_true', help='Print system status and exit')
    
    args = parser.parse_args()
    
    # Create master orchestration
    master = MasterOrchestration(config_path=args.config)
    
    if args.status:
        # Print status
        status = master.get_system_status()
        for task_name, task_status in status.items():
            print(f"Task: {task_name}")
            print(f"  Last run: {task_status['last_run_time']}")
            print(f"  Status: {task_status['last_run_status']}")
            if task_status['last_run_error']:
                print(f"  Error: {task_status['last_run_error']}")
        sys.exit(0)
        
    if args.execute:
        # Execute a specific task
        print(f"Executing task: {args.execute}")
        success = master.execute_task_now(args.execute)
        print(f"Task execution {'succeeded' if success else 'failed'}")
        sys.exit(0 if success else 1)
        
    # Run the system
    try:
        master.start(block=not args.run_once)
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 