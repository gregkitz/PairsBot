"""
Automation Monitoring System

This module provides monitoring capabilities for the automation system,
including tracking task execution, detecting failures, and generating alerts.
"""

import os
import sys
import logging
import json
import datetime
import time
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Tuple

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# Import project modules
from utils.logging_utils import setup_logger
from config.config_manager import ConfigManager

# Setup logger
logger = setup_logger('automation_monitor', log_level=logging.INFO)

class AutomationMonitor:
    """
    Monitors automation tasks and system health, generating alerts when issues are detected.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the automation monitor
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
        """
        # Load configuration
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
                
        # Load monitoring settings
        self.monitoring_settings = self._load_monitoring_settings()
        
        # Set up state tracking
        self.status_history = {}
        self.alert_history = {}
        self.last_status_check = None
        
        # Set up status directory
        self.status_dir = self.monitoring_settings.get('status_dir', os.path.join(src_dir, 'logs', 'status'))
        os.makedirs(self.status_dir, exist_ok=True)
        
    def _load_monitoring_settings(self) -> Dict:
        """
        Load monitoring settings from config
        
        Returns:
        --------
        Dict
            Monitoring settings
        """
        default_settings = {
            'check_interval_seconds': 300,  # 5 minutes
            'status_dir': os.path.join(src_dir, 'logs', 'status'),
            'max_history_entries': 100,
            'alert_thresholds': {
                'task_failure_count': 3,
                'max_task_delay_minutes': 30
            },
            'email_alerts': {
                'enabled': False,
                'smtp_server': 'smtp.example.com',
                'smtp_port': 587,
                'sender': 'alerts@example.com',
                'recipients': ['admin@example.com'],
                'use_tls': True
            }
        }
        
        if not self.config:
            return default_settings
            
        # Get monitoring settings from config
        monitoring_config = self.config.get('monitoring', {})
        
        # Merge with defaults
        for key, value in default_settings.items():
            if key not in monitoring_config:
                monitoring_config[key] = value
            elif isinstance(value, dict) and isinstance(monitoring_config.get(key), dict):
                # Merge nested dictionaries
                for subkey, subvalue in value.items():
                    if subkey not in monitoring_config[key]:
                        monitoring_config[key][subkey] = subvalue
                        
        return monitoring_config
        
    def check_task_status(self, master_orchestration) -> Dict:
        """
        Check the status of all tasks
        
        Parameters:
        -----------
        master_orchestration : MasterOrchestration
            The master orchestration instance to check
            
        Returns:
        --------
        Dict
            Status information for all tasks with issues flagged
        """
        # Get the current status of all tasks
        task_status = master_orchestration.get_system_status()
        current_time = datetime.datetime.now()
        self.last_status_check = current_time
        
        # Process status and detect issues
        processed_status = {}
        alerts = []
        
        for task_name, status in task_status.items():
            processed_status[task_name] = status.copy()
            
            # Add status to history
            if task_name not in self.status_history:
                self.status_history[task_name] = []
                
            self.status_history[task_name].append({
                'time': current_time,
                'status': status['last_run_status'],
                'error': status['last_run_error']
            })
            
            # Trim history if it gets too long
            if len(self.status_history[task_name]) > self.monitoring_settings['max_history_entries']:
                self.status_history[task_name] = self.status_history[task_name][-self.monitoring_settings['max_history_entries']:]
                
            # Check for issues
            issues = self._detect_task_issues(task_name, status, current_time)
            if issues:
                processed_status[task_name]['issues'] = issues
                alerts.extend(issues)
                
        # Save processed status to file
        self._save_status(processed_status)
        
        # Handle alerts
        if alerts:
            self._handle_alerts(alerts, processed_status)
            
        return processed_status
        
    def _detect_task_issues(self, task_name: str, status: Dict, current_time: datetime.datetime) -> List[Dict]:
        """
        Detect issues with a task
        
        Parameters:
        -----------
        task_name : str
            Name of the task
        status : Dict
            Status information for the task
        current_time : datetime.datetime
            Current time
            
        Returns:
        --------
        List[Dict]
            List of issue dictionaries
        """
        issues = []
        
        # Check if task has ever run
        if status['last_run_time'] is None:
            issues.append({
                'type': 'task_never_run',
                'task': task_name,
                'message': f"Task {task_name} has never run"
            })
            return issues
            
        # Check for recent failures
        if status['last_run_status'] == "FAILED":
            failures = self._count_recent_failures(task_name)
            if failures >= self.monitoring_settings['alert_thresholds']['task_failure_count']:
                issues.append({
                    'type': 'repeated_failure',
                    'task': task_name,
                    'count': failures,
                    'message': f"Task {task_name} has failed {failures} times consecutively"
                })
            else:
                issues.append({
                    'type': 'task_failure',
                    'task': task_name,
                    'error': status['last_run_error'],
                    'message': f"Task {task_name} failed: {status['last_run_error']}"
                })
                
        # Check if critical task is delayed (only if has dependencies)
        if status['critical'] and len(status['dependencies']) == 0:
            if status['last_run_time'] is not None:
                delay_minutes = (current_time - status['last_run_time']).total_seconds() / 60
                threshold = self.monitoring_settings['alert_thresholds']['max_task_delay_minutes']
                
                if delay_minutes > threshold:
                    issues.append({
                        'type': 'task_delay',
                        'task': task_name,
                        'delay_minutes': delay_minutes,
                        'message': f"Critical task {task_name} last ran {delay_minutes:.1f} minutes ago (threshold: {threshold} minutes)"
                    })
                    
        return issues
        
    def _count_recent_failures(self, task_name: str) -> int:
        """
        Count consecutive recent failures for a task
        
        Parameters:
        -----------
        task_name : str
            Name of the task
            
        Returns:
        --------
        int
            Number of consecutive recent failures
        """
        if task_name not in self.status_history:
            return 0
            
        failures = 0
        for entry in reversed(self.status_history[task_name]):
            if entry['status'] == "FAILED":
                failures += 1
            else:
                break
                
        return failures
        
    def _save_status(self, status: Dict) -> None:
        """
        Save status information to a file
        
        Parameters:
        -----------
        status : Dict
            Status information to save
        """
        status_file = os.path.join(self.status_dir, 'automation_status.json')
        
        # Convert datetime objects to strings
        serializable_status = {}
        for task_name, task_status in status.items():
            serializable_status[task_name] = task_status.copy()
            
            if task_status['last_run_time'] is not None:
                serializable_status[task_name]['last_run_time'] = task_status['last_run_time'].isoformat()
                
        try:
            with open(status_file, 'w') as f:
                json.dump(serializable_status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status to file: {str(e)}")
            
    def _handle_alerts(self, alerts: List[Dict], full_status: Dict) -> None:
        """
        Handle alerts generated from status checks
        
        Parameters:
        -----------
        alerts : List[Dict]
            List of alert dictionaries
        full_status : Dict
            Full status information
        """
        # Log all alerts
        for alert in alerts:
            logger.warning(f"Alert: {alert['message']}")
            
        # Add alerts to history
        current_time = datetime.datetime.now()
        for alert in alerts:
            alert_key = f"{alert['type']}:{alert['task']}"
            
            if alert_key not in self.alert_history:
                self.alert_history[alert_key] = []
                
            self.alert_history[alert_key].append({
                'time': current_time,
                'alert': alert
            })
            
            # Trim history
            if len(self.alert_history[alert_key]) > self.monitoring_settings['max_history_entries']:
                self.alert_history[alert_key] = self.alert_history[alert_key][-self.monitoring_settings['max_history_entries']:]
                
        # Send email alerts if configured
        if self.monitoring_settings['email_alerts']['enabled']:
            self._send_email_alert(alerts, full_status)
            
    def _send_email_alert(self, alerts: List[Dict], full_status: Dict) -> None:
        """
        Send email alerts
        
        Parameters:
        -----------
        alerts : List[Dict]
            List of alert dictionaries
        full_status : Dict
            Full status information
        """
        email_config = self.monitoring_settings['email_alerts']
        
        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"Automation System Alert: {len(alerts)} issue(s) detected"
        
        # Build the email body
        body = f"Automation System Alert\n"
        body += f"Time: {datetime.datetime.now().isoformat()}\n"
        body += f"Host: {socket.gethostname()}\n\n"
        body += f"The following issues were detected:\n\n"
        
        for alert in alerts:
            body += f"- {alert['message']}\n"
            
        body += f"\n\nFull system status:\n\n"
        for task_name, task_status in full_status.items():
            last_run = task_status['last_run_time']
            if isinstance(last_run, datetime.datetime):
                last_run = last_run.isoformat()
                
            body += f"Task: {task_name}\n"
            body += f"  Last run: {last_run}\n"
            body += f"  Status: {task_status['last_run_status']}\n"
            if task_status['last_run_error']:
                body += f"  Error: {task_status['last_run_error']}\n"
            body += "\n"
            
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            # Set up SMTP server
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            
            if email_config.get('use_tls', False):
                server.starttls()
                
            # Login if credentials are provided
            if 'username' in email_config and 'password' in email_config:
                server.login(email_config['username'], email_config['password'])
                
            # Send the email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent email alert to {', '.join(email_config['recipients'])}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            
    def check_system_resources(self) -> Dict:
        """
        Check system resource usage
        
        Returns:
        --------
        Dict
            System resource information
        """
        import psutil
        
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check for issues
        issues = []
        
        # CPU usage threshold (80%)
        if cpu_percent > 80:
            issues.append({
                'type': 'high_cpu_usage',
                'value': cpu_percent,
                'message': f"High CPU usage: {cpu_percent}%"
            })
            
        # Memory usage threshold (85%)
        if memory.percent > 85:
            issues.append({
                'type': 'high_memory_usage',
                'value': memory.percent,
                'message': f"High memory usage: {memory.percent}%"
            })
            
        # Disk usage threshold (90%)
        if disk.percent > 90:
            issues.append({
                'type': 'high_disk_usage',
                'value': disk.percent,
                'message': f"High disk usage: {disk.percent}%"
            })
            
        resource_info = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent,
            'issues': issues
        }
        
        # Log resource usage
        logger.info(f"System resources - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
        
        # Handle alerts for resource issues
        if issues:
            self._handle_resource_alerts(issues)
            
        return resource_info
        
    def _handle_resource_alerts(self, issues: List[Dict]) -> None:
        """
        Handle system resource alerts
        
        Parameters:
        -----------
        issues : List[Dict]
            List of resource issue dictionaries
        """
        # Log all issues
        for issue in issues:
            logger.warning(f"Resource alert: {issue['message']}")
            
        # Send email if configured
        if self.monitoring_settings['email_alerts']['enabled']:
            self._send_resource_alert_email(issues)
            
    def _send_resource_alert_email(self, issues: List[Dict]) -> None:
        """
        Send email alert for resource issues
        
        Parameters:
        -----------
        issues : List[Dict]
            List of resource issue dictionaries
        """
        email_config = self.monitoring_settings['email_alerts']
        
        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender']
        msg['To'] = ', '.join(email_config['recipients'])
        msg['Subject'] = f"System Resource Alert: {len(issues)} issue(s) detected"
        
        # Build the email body
        body = f"System Resource Alert\n"
        body += f"Time: {datetime.datetime.now().isoformat()}\n"
        body += f"Host: {socket.gethostname()}\n\n"
        body += f"The following resource issues were detected:\n\n"
        
        for issue in issues:
            body += f"- {issue['message']}\n"
            
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            # Set up SMTP server
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            
            if email_config.get('use_tls', False):
                server.starttls()
                
            # Login if credentials are provided
            if 'username' in email_config and 'password' in email_config:
                server.login(email_config['username'], email_config['password'])
                
            # Send the email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Sent resource alert email to {', '.join(email_config['recipients'])}")
            
        except Exception as e:
            logger.error(f"Error sending resource alert email: {str(e)}")
            
    def run_continuous_monitoring(self, master_orchestration, interval_seconds: int = None) -> None:
        """
        Run continuous monitoring of the automation system
        
        Parameters:
        -----------
        master_orchestration : MasterOrchestration
            The master orchestration instance to monitor
        interval_seconds : int
            Monitoring interval in seconds
        """
        if interval_seconds is None:
            interval_seconds = self.monitoring_settings['check_interval_seconds']
            
        logger.info(f"Starting continuous monitoring with interval {interval_seconds} seconds")
        
        try:
            while True:
                # Check task status
                self.check_task_status(master_orchestration)
                
                # Check system resources
                try:
                    self.check_system_resources()
                except ImportError:
                    logger.warning("psutil module not available - system resource monitoring disabled")
                    
                # Sleep until next check
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {str(e)}")
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Automation monitoring for intraday ML trading system')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = AutomationMonitor(config_path=args.config)
    
    if args.continuous:
        # Import master orchestration
        try:
            from master_orchestration import MasterOrchestration
            master = MasterOrchestration(config_path=args.config)
            monitor.run_continuous_monitoring(master, args.interval)
        except ImportError:
            logger.error("Could not import MasterOrchestration - continuous monitoring disabled")
    else:
        # Run a single check of system resources
        try:
            monitor.check_system_resources()
        except ImportError:
            logger.error("psutil module not available - system resource monitoring disabled") 