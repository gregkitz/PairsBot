"""
System Startup Script for Intraday ML Trading System

This script handles the proper startup of the automation system,
including initialization, recovery from previous failures, and system checks.
"""

import os
import sys
import logging
import time
import datetime
import argparse
import json
import traceback
import subprocess
from typing import Dict, List, Any, Optional, Tuple

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)

# Import project modules
from utils.logging_utils import setup_logger
from config.config_manager import ConfigManager
from automation.automation_monitor import AutomationMonitor

# Setup logger
logger = setup_logger('system_startup', log_level=logging.INFO)

class SystemStartup:
    """
    Handles system startup, initialization, and recovery
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the system startup handler
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        self.start_time = datetime.datetime.now()
        
        # Load configuration
        if config_path:
            self.config_path = config_path
        else:
            # Default configuration location
            config_dir = os.path.join(src_dir, 'config')
            default_config = os.path.join(config_dir, 'automation_config.yaml')
            self.config_path = default_config
            
        logger.info(f"Initializing system startup with config: {self.config_path}")
        
        try:
            self.config = ConfigManager(self.config_path)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            self.config = None
            
        # Set up status directory
        self.status_dir = os.path.join(src_dir, 'logs', 'status')
        os.makedirs(self.status_dir, exist_ok=True)
        
        # Get startup settings
        self.startup_settings = self._load_startup_settings()
        
    def _load_startup_settings(self) -> Dict:
        """
        Load startup settings from config
        
        Returns:
        --------
        Dict
            Startup settings
        """
        default_settings = {
            'recovery_enabled': True,
            'max_recovery_attempts': 3,
            'system_check_timeout': 30,  # seconds
            'required_services': [],
            'required_directories': [
                os.path.join(root_dir, 'data'),
                os.path.join(root_dir, 'models'),
                os.path.join(root_dir, 'logs')
            ],
            'startup_delay': 5,  # seconds
            'notification_enabled': False
        }
        
        if not self.config:
            return default_settings
            
        # Get startup settings from config
        startup_config = self.config.get('startup', {})
        
        # Merge with defaults
        for key, value in default_settings.items():
            if key not in startup_config:
                startup_config[key] = value
                
        return startup_config
        
    def check_system_status(self) -> Dict:
        """
        Check the system status and environment
        
        Returns:
        --------
        Dict
            System status information with any issues
        """
        status = {
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'OK',
            'issues': [],
            'checks': {}
        }
        
        # 1. Check required directories
        dir_check = self._check_directories()
        status['checks']['directories'] = dir_check
        if not dir_check['status']:
            status['status'] = 'ERROR'
            status['issues'].extend(dir_check['issues'])
            
        # 2. Check required services if any
        if self.startup_settings['required_services']:
            service_check = self._check_services()
            status['checks']['services'] = service_check
            if not service_check['status']:
                status['status'] = 'ERROR'
                status['issues'].extend(service_check['issues'])
                
        # 3. Check for previous crash state
        crash_check = self._check_crash_state()
        status['checks']['crash_state'] = crash_check
        if not crash_check['status']:
            status['status'] = 'WARNING'
            status['issues'].extend(crash_check['issues'])
            
        # 4. Check system resources
        try:
            import psutil
            resource_check = {
                'status': True,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
            
            # Add warnings for high resource usage
            if resource_check['cpu_percent'] > 80:
                resource_check['status'] = False
                status['issues'].append(f"High CPU usage: {resource_check['cpu_percent']}%")
                
            if resource_check['memory_percent'] > 85:
                resource_check['status'] = False
                status['issues'].append(f"High memory usage: {resource_check['memory_percent']}%")
                
            if resource_check['disk_percent'] > 90:
                resource_check['status'] = False
                status['issues'].append(f"High disk usage: {resource_check['disk_percent']}%")
                
            status['checks']['resources'] = resource_check
            if not resource_check['status']:
                status['status'] = 'WARNING'
                
        except ImportError:
            logger.warning("psutil module not available - system resource check skipped")
            status['checks']['resources'] = {
                'status': None,
                'message': 'psutil module not available'
            }
            
        return status
        
    def _check_directories(self) -> Dict:
        """
        Check required directories exist and are writable
        
        Returns:
        --------
        Dict
            Directory check results
        """
        result = {
            'status': True,
            'checked_dirs': [],
            'issues': []
        }
        
        for dir_path in self.startup_settings['required_directories']:
            dir_status = {
                'path': dir_path,
                'exists': os.path.exists(dir_path),
                'is_dir': os.path.isdir(dir_path) if os.path.exists(dir_path) else False,
                'writable': os.access(dir_path, os.W_OK) if os.path.exists(dir_path) else False
            }
            
            result['checked_dirs'].append(dir_status)
            
            if not dir_status['exists']:
                result['status'] = False
                result['issues'].append(f"Required directory not found: {dir_path}")
                try:
                    logger.info(f"Creating missing directory: {dir_path}")
                    os.makedirs(dir_path, exist_ok=True)
                    # Check if creation was successful
                    if os.path.exists(dir_path) and os.path.isdir(dir_path):
                        logger.info(f"Successfully created directory: {dir_path}")
                    else:
                        logger.error(f"Failed to create directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Error creating directory {dir_path}: {str(e)}")
            elif not dir_status['is_dir']:
                result['status'] = False
                result['issues'].append(f"Path exists but is not a directory: {dir_path}")
            elif not dir_status['writable']:
                result['status'] = False
                result['issues'].append(f"Directory not writable: {dir_path}")
                
        return result
        
    def _check_services(self) -> Dict:
        """
        Check required services are running
        
        Returns:
        --------
        Dict
            Service check results
        """
        result = {
            'status': True,
            'checked_services': [],
            'issues': []
        }
        
        for service_name in self.startup_settings['required_services']:
            service_status = {
                'name': service_name,
                'running': False
            }
            
            try:
                # Using subprocess to check service status
                # This is Windows-specific - adjust for other OS
                cmd = f"sc query {service_name} | findstr RUNNING"
                process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                service_status['running'] = process.returncode == 0
                
                if not service_status['running']:
                    result['status'] = False
                    result['issues'].append(f"Required service not running: {service_name}")
                    
                    # Try to start the service
                    logger.info(f"Attempting to start service: {service_name}")
                    start_cmd = f"sc start {service_name}"
                    subprocess.run(start_cmd, shell=True)
                    
                    # Check again
                    time.sleep(2)
                    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if process.returncode == 0:
                        logger.info(f"Successfully started service: {service_name}")
                    else:
                        logger.error(f"Failed to start service: {service_name}")
                        
            except Exception as e:
                result['status'] = False
                service_status['error'] = str(e)
                result['issues'].append(f"Error checking service {service_name}: {str(e)}")
                
            result['checked_services'].append(service_status)
            
        return result
        
    def _check_crash_state(self) -> Dict:
        """
        Check for signs of previous crash
        
        Returns:
        --------
        Dict
            Crash state check results
        """
        result = {
            'status': True,
            'issues': []
        }
        
        # Check for lock files
        lock_file = os.path.join(self.status_dir, 'system.lock')
        if os.path.exists(lock_file):
            result['status'] = False
            result['issues'].append("System lock file found - possible previous crash")
            result['lock_file'] = lock_file
            
            # Read lock file to get information about previous run
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)
                result['previous_run'] = lock_data
            except Exception:
                result['previous_run'] = None
                
        return result
        
    def create_lock_file(self) -> None:
        """
        Create a lock file to indicate running state
        """
        lock_file = os.path.join(self.status_dir, 'system.lock')
        lock_data = {
            'start_time': self.start_time.isoformat(),
            'pid': os.getpid(),
            'hostname': socket.gethostname() if hasattr(__import__('socket'), 'gethostname') else 'unknown'
        }
        
        try:
            with open(lock_file, 'w') as f:
                json.dump(lock_data, f)
            logger.info(f"Created lock file: {lock_file}")
        except Exception as e:
            logger.error(f"Error creating lock file: {str(e)}")
            
    def remove_lock_file(self) -> None:
        """
        Remove the lock file
        """
        lock_file = os.path.join(self.status_dir, 'system.lock')
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info(f"Removed lock file: {lock_file}")
            except Exception as e:
                logger.error(f"Error removing lock file: {str(e)}")
                
    def perform_recovery(self, status: Dict) -> bool:
        """
        Perform recovery actions based on system status
        
        Parameters:
        -----------
        status : Dict
            System status information
            
        Returns:
        --------
        bool
            True if recovery was successful, False otherwise
        """
        if not self.startup_settings['recovery_enabled']:
            logger.info("Recovery is disabled - skipping")
            return True
            
        if status['status'] == 'OK':
            logger.info("No recovery needed - system status OK")
            return True
            
        logger.info(f"Performing recovery for {len(status['issues'])} issues")
        
        # Handle different recovery scenarios
        for issue in status['issues']:
            logger.info(f"Handling issue: {issue}")
            # Recovery logic would go here based on specific issues
            
        # Handle crash recovery
        if 'crash_state' in status['checks'] and not status['checks']['crash_state']['status']:
            logger.info("Performing crash recovery")
            self.recover_from_crash(status['checks']['crash_state'])
            
        # Check if all issues are resolved
        # For now, we'll just return True as a placeholder
        # In a real implementation, we would re-check the system status
        
        return True
        
    def recover_from_crash(self, crash_info: Dict) -> None:
        """
        Recover from a detected crash
        
        Parameters:
        -----------
        crash_info : Dict
            Information about the crash state
        """
        # Remove lock file
        lock_file = crash_info.get('lock_file')
        if lock_file and os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                logger.info(f"Removed crash lock file: {lock_file}")
            except Exception as e:
                logger.error(f"Error removing crash lock file: {str(e)}")
                
        # Additional recovery actions could be added here
        
    def start_automation_system(self) -> int:
        """
        Start the automation system
        
        Returns:
        --------
        int
            Exit code (0 for success, non-zero for failure)
        """
        logger.info("Starting automation system")
        
        try:
            # 1. Check system status
            status = self.check_system_status()
            
            # 2. Perform recovery if needed
            if status['status'] != 'OK':
                recovery_success = self.perform_recovery(status)
                if not recovery_success:
                    logger.error("Recovery failed - cannot continue")
                    return 1
                    
            # 3. Create lock file
            self.create_lock_file()
            
            # 4. Start the actual automation process
            return self._start_master_orchestration()
            
        except Exception as e:
            logger.error(f"Error during system startup: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
            
    def _start_master_orchestration(self) -> int:
        """
        Start the master orchestration process
        
        Returns:
        --------
        int
            Exit code (0 for success, non-zero for failure)
        """
        # Optional startup delay
        if self.startup_settings['startup_delay'] > 0:
            logger.info(f"Waiting {self.startup_settings['startup_delay']} seconds before startup...")
            time.sleep(self.startup_settings['startup_delay'])
            
        logger.info("Starting master orchestration")
        
        try:
            # Import here to avoid circular imports
            from automation.master_orchestration import MasterOrchestration
            
            # Start a monitor in a separate process
            monitor_process = None
            try:
                monitor_script = os.path.join(src_dir, 'automation', 'automation_monitor.py')
                monitor_cmd = [sys.executable, monitor_script, '--continuous']
                if self.config_path:
                    monitor_cmd.extend(['--config', self.config_path])
                    
                logger.info(f"Starting monitor process: {' '.join(monitor_cmd)}")
                
                # Start monitor process
                monitor_process = subprocess.Popen(
                    monitor_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait briefly to ensure it starts
                time.sleep(1)
                
                if monitor_process.poll() is not None:
                    # Monitor process exited immediately
                    stdout, stderr = monitor_process.communicate()
                    logger.error(f"Monitor process failed to start - return code: {monitor_process.returncode}")
                    logger.error(f"Monitor stdout: {stdout}")
                    logger.error(f"Monitor stderr: {stderr}")
                else:
                    logger.info("Monitor process started successfully")
                    
            except Exception as e:
                logger.error(f"Error starting monitor process: {str(e)}")
                monitor_process = None
                
            # Create and start master orchestration
            master = MasterOrchestration(config_path=self.config_path)
            
            # Run the orchestrator
            master.start(block=True)  # This will block until interrupted
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("System startup interrupted by user")
            self.remove_lock_file()
            return 0
        except Exception as e:
            logger.error(f"Error starting master orchestration: {str(e)}")
            logger.error(traceback.format_exc())
            self.remove_lock_file()
            return 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='System startup for intraday ML trading system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--no-recovery', action='store_true', help='Disable recovery actions')
    
    args = parser.parse_args()
    
    # Create startup handler
    startup = SystemStartup(config_path=args.config)
    
    # Disable recovery if requested
    if args.no_recovery:
        startup.startup_settings['recovery_enabled'] = False
        
    # Start the system
    exit_code = startup.start_automation_system()
    sys.exit(exit_code) 