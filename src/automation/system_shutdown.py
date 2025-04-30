"""
System Shutdown Script for Intraday ML Trading System

This script handles the proper shutdown of the automation system,
including graceful termination of processes, cleanup, and status saving.
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
import signal
import psutil
from typing import Dict, List, Any, Optional, Tuple

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)

# Import project modules
from utils.logging_utils import setup_logger
from config.config_manager import ConfigManager

# Setup logger
logger = setup_logger('system_shutdown', log_level=logging.INFO)

class SystemShutdown:
    """
    Handles system shutdown, termination, and cleanup
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the system shutdown handler
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        self.shutdown_time = datetime.datetime.now()
        
        # Load configuration
        if config_path:
            self.config_path = config_path
        else:
            # Default configuration location
            config_dir = os.path.join(src_dir, 'config')
            default_config = os.path.join(config_dir, 'automation_config.yaml')
            self.config_path = default_config
            
        logger.info(f"Initializing system shutdown with config: {self.config_path}")
        
        try:
            self.config = ConfigManager(self.config_path)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            self.config = None
            
        # Set up status directory
        self.status_dir = os.path.join(src_dir, 'logs', 'status')
        os.makedirs(self.status_dir, exist_ok=True)
        
        # Get shutdown settings
        self.shutdown_settings = self._load_shutdown_settings()
        
    def _load_shutdown_settings(self) -> Dict:
        """
        Load shutdown settings from config
        
        Returns:
        --------
        Dict
            Shutdown settings
        """
        default_settings = {
            'graceful_timeout': 30,  # seconds
            'force_kill_timeout': 10,  # seconds
            'save_state': True,
            'cleanup_temp_files': True,
            'notification_enabled': False,
            'process_names': [
                'python.exe',  # For Windows
                'python3',     # For Unix-like systems
                'python'       # Generic
            ]
        }
        
        if not self.config:
            return default_settings
            
        # Get shutdown settings from config
        shutdown_config = self.config.get('shutdown', {})
        
        # Merge with defaults
        for key, value in default_settings.items():
            if key not in shutdown_config:
                shutdown_config[key] = value
                
        return shutdown_config
        
    def find_automation_processes(self) -> List[Dict]:
        """
        Find running automation processes
        
        Returns:
        --------
        List[Dict]
            List of process information dictionaries
        """
        processes = []
        
        # Find processes by checking command line for our scripts
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                # Check if process is one of our Python processes
                if proc.info['name'] in self.shutdown_settings['process_names']:
                    cmdline = proc.info['cmdline']
                    
                    # Skip the current process
                    if os.getpid() == proc.info['pid']:
                        continue
                        
                    # Check if this is one of our automation scripts
                    is_automation = False
                    script_name = None
                    
                    if cmdline and len(cmdline) > 1:
                        for arg in cmdline[1:]:
                            if isinstance(arg, str) and 'automation' in arg and arg.endswith('.py'):
                                is_automation = True
                                script_name = os.path.basename(arg)
                                break
                                
                    if is_automation:
                        # Get process details
                        process_info = {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'script': script_name,
                            'cmdline': ' '.join(cmdline) if cmdline else '',
                            'create_time': datetime.datetime.fromtimestamp(proc.info['create_time']).isoformat(),
                            'process': proc
                        }
                        
                        processes.append(process_info)
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
        return processes
        
    def terminate_processes(self, processes: List[Dict], force: bool = False) -> List[Dict]:
        """
        Terminate automation processes
        
        Parameters:
        -----------
        processes : List[Dict]
            List of process information dictionaries
        force : bool
            Whether to force kill processes
            
        Returns:
        --------
        List[Dict]
            List of process termination results
        """
        results = []
        
        for proc_info in processes:
            result = {
                'pid': proc_info['pid'],
                'script': proc_info['script'],
                'success': False,
                'force_kill': force
            }
            
            try:
                proc = proc_info['process']
                
                if force:
                    # Force kill
                    logger.info(f"Force killing process {proc_info['pid']} ({proc_info['script']})")
                    proc.kill()
                else:
                    # Graceful termination
                    logger.info(f"Gracefully terminating process {proc_info['pid']} ({proc_info['script']})")
                    proc.terminate()
                    
                # Wait for process to terminate
                timeout = self.shutdown_settings['force_kill_timeout'] if force else self.shutdown_settings['graceful_timeout']
                
                gone, alive = psutil.wait_procs([proc], timeout=timeout)
                
                if proc in gone:
                    result['success'] = True
                    logger.info(f"Successfully terminated process {proc_info['pid']}")
                else:
                    result['success'] = False
                    logger.warning(f"Failed to terminate process {proc_info['pid']} within timeout")
                    
            except Exception as e:
                result['success'] = False
                result['error'] = str(e)
                logger.error(f"Error terminating process {proc_info['pid']}: {str(e)}")
                
            results.append(result)
            
        return results
        
    def save_system_state(self) -> bool:
        """
        Save the current system state before shutdown
        
        Returns:
        --------
        bool
            True if state was saved successfully, False otherwise
        """
        if not self.shutdown_settings['save_state']:
            logger.info("State saving disabled - skipping")
            return True
            
        state_file = os.path.join(self.status_dir, 'shutdown_state.json')
        
        try:
            # Collect state information
            state = {
                'shutdown_time': self.shutdown_time.isoformat(),
                'clean_shutdown': True,
                'processes': [],
                'status': 'OK'
            }
            
            # Get information about running processes
            processes = self.find_automation_processes()
            state['processes'] = [
                {
                    'pid': p['pid'],
                    'script': p['script'],
                    'cmdline': p['cmdline'],
                    'create_time': p['create_time']
                }
                for p in processes
            ]
            
            # Save state to file
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved system state to {state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
            return False
            
    def cleanup_temp_files(self) -> bool:
        """
        Clean up temporary files
        
        Returns:
        --------
        bool
            True if cleanup was successful, False otherwise
        """
        if not self.shutdown_settings['cleanup_temp_files']:
            logger.info("Temp file cleanup disabled - skipping")
            return True
            
        try:
            # Clean up lock file
            lock_file = os.path.join(self.status_dir, 'system.lock')
            if os.path.exists(lock_file):
                os.remove(lock_file)
                logger.info(f"Removed lock file: {lock_file}")
                
            # Additional cleanup could be added here
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
            return False
            
    def shutdown_system(self, force: bool = False) -> int:
        """
        Perform system shutdown
        
        Parameters:
        -----------
        force : bool
            Whether to force kill processes
            
        Returns:
        --------
        int
            Exit code (0 for success, non-zero for failure)
        """
        logger.info(f"Starting {'forced' if force else 'graceful'} system shutdown")
        
        try:
            # 1. Save system state
            self.save_system_state()
            
            # 2. Find automation processes
            processes = self.find_automation_processes()
            logger.info(f"Found {len(processes)} automation processes")
            
            if processes:
                # 3. Terminate processes
                results = self.terminate_processes(processes, force)
                
                # 4. Check if all processes were terminated
                all_terminated = all(r['success'] for r in results)
                
                if not all_terminated and not force:
                    # If graceful termination failed, try force kill
                    logger.warning("Graceful termination failed for some processes - trying force kill")
                    
                    # Find remaining processes
                    remaining = self.find_automation_processes()
                    
                    if remaining:
                        force_results = self.terminate_processes(remaining, True)
                        all_terminated = all(r['success'] for r in force_results)
                        
                if not all_terminated:
                    logger.error("Failed to terminate all processes")
                    
            # 5. Clean up temporary files
            self.cleanup_temp_files()
            
            logger.info("System shutdown completed")
            return 0
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='System shutdown for intraday ML trading system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--force', action='store_true', help='Force kill processes')
    parser.add_argument('--no-cleanup', action='store_true', help='Disable cleanup')
    
    args = parser.parse_args()
    
    # Create shutdown handler
    shutdown = SystemShutdown(config_path=args.config)
    
    # Disable cleanup if requested
    if args.no_cleanup:
        shutdown.shutdown_settings['cleanup_temp_files'] = False
        
    # Perform shutdown
    exit_code = shutdown.shutdown_system(force=args.force)
    sys.exit(exit_code) 