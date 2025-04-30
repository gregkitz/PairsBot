"""
Windows Task Scheduler Setup Script

This script sets up Windows Task Scheduler tasks for the intraday ML trading system.
It creates scheduled tasks for the master orchestration script and various recovery tasks.
"""

import os
import sys
import argparse
import subprocess
import getpass
from typing import List, Dict, Any

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(src_dir)
sys.path.append(src_dir)

# Import project modules
from utils.logging_utils import setup_logger

# Setup logger
logger = setup_logger('windows_scheduler_setup')

def create_scheduled_task(task_name: str, 
                         script_path: str, 
                         arguments: str = "",
                         schedule_type: str = "DAILY",
                         start_time: str = "08:00",
                         days_of_week: str = None,
                         user: str = None,
                         password: str = None,
                         run_with_highest_privileges: bool = False,
                         start_in: str = None) -> bool:
    """
    Create a Windows scheduled task
    
    Parameters:
    -----------
    task_name : str
        Name of the task
    script_path : str
        Path to the script to run
    arguments : str
        Command line arguments for the script
    schedule_type : str
        Schedule type (DAILY, WEEKLY, MONTHLY, ONSTART, ONLOGON, ONIDLE)
    start_time : str
        Start time in HH:MM format
    days_of_week : str
        Days of week for WEEKLY schedule (MON, TUE, WED, THU, FRI, SAT, SUN)
    user : str
        User to run the task as
    password : str
        Password for the user
    run_with_highest_privileges : bool
        Whether to run with highest privileges
    start_in : str
        Working directory for the task
        
    Returns:
    --------
    bool
        True if task was created successfully, False otherwise
    """
    try:
        # Build the command
        python_exe = sys.executable
        if start_in is None:
            start_in = root_dir
            
        # Basic task creation
        cmd = [
            "schtasks", "/create", 
            "/tn", f"IntraDayML\\{task_name}",
            "/tr", f'"{python_exe}" "{script_path}" {arguments}',
            "/sc", schedule_type,
            "/st", start_time
        ]
        
        # Add weekly options if needed
        if schedule_type == "WEEKLY" and days_of_week:
            cmd.extend(["/d", days_of_week])
            
        # Add user if provided
        if user:
            cmd.extend(["/ru", user])
            if password:
                cmd.extend(["/rp", password])
        else:
            # Use current user
            cmd.extend(["/ru", os.getlogin()])
            
        # Add highest privileges if requested
        if run_with_highest_privileges:
            cmd.append("/rl", "HIGHEST")
            
        # Create the task
        logger.info(f"Creating scheduled task: {task_name}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=start_in)
        
        if result.returncode == 0:
            logger.info(f"Successfully created task: {task_name}")
            return True
        else:
            logger.error(f"Failed to create task {task_name}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating scheduled task {task_name}: {str(e)}")
        return False


def create_task_folder() -> bool:
    """
    Create the IntraDayML folder in Task Scheduler
    
    Returns:
    --------
    bool
        True if folder was created successfully, False otherwise
    """
    try:
        # Check if folder exists
        result = subprocess.run(
            ["schtasks", "/query", "/tn", "IntraDayML"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("IntraDayML task folder already exists")
            return True
            
        # Create the folder
        result = subprocess.run(
            ["schtasks", "/create", "/tn", "IntraDayML", "/f", "/sc", "ONCE", "/st", "00:00", "/tr", "cmd /c exit 0"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("Created IntraDayML task folder")
            return True
        else:
            logger.error(f"Failed to create task folder: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating task folder: {str(e)}")
        return False


def setup_main_tasks(user: str = None, password: str = None) -> List[str]:
    """
    Set up the main scheduled tasks for the trading system
    
    Parameters:
    -----------
    user : str
        User to run the tasks as
    password : str
        Password for the user
        
    Returns:
    --------
    List[str]
        List of task names that were created successfully
    """
    # Create the task folder
    if not create_task_folder():
        logger.error("Failed to create task folder, cannot continue")
        return []
        
    # Path to the master orchestration script
    master_script = os.path.join(src_dir, "automation", "master_orchestration.py")
    
    # Tasks to create
    tasks = [
        {
            "name": "DailyStartup",
            "script_path": master_script,
            "arguments": "--config ../config/automation_config.yaml",
            "schedule_type": "DAILY",
            "start_time": "07:00",
            "run_with_highest_privileges": True
        },
        {
            "name": "SystemHealthCheck",
            "script_path": master_script,
            "arguments": "--execute check_system_health",
            "schedule_type": "DAILY",
            "start_time": "06:30"
        },
        {
            "name": "WeeklyMaintenance",
            "script_path": master_script,
            "arguments": "--execute cleanup_logs",
            "schedule_type": "WEEKLY",
            "start_time": "23:00",
            "days_of_week": "SUN"
        },
        {
            "name": "SystemRecovery",
            "script_path": master_script,
            "arguments": "--config ../config/automation_config.yaml",
            "schedule_type": "ONSTART",
            "run_with_highest_privileges": True
        }
    ]
    
    # Create each task
    successful_tasks = []
    for task in tasks:
        if create_scheduled_task(
            task_name=task["name"],
            script_path=task["script_path"],
            arguments=task.get("arguments", ""),
            schedule_type=task.get("schedule_type", "DAILY"),
            start_time=task.get("start_time", "08:00"),
            days_of_week=task.get("days_of_week"),
            user=user,
            password=password,
            run_with_highest_privileges=task.get("run_with_highest_privileges", False),
            start_in=root_dir
        ):
            successful_tasks.append(task["name"])
            
    return successful_tasks


def remove_tasks() -> bool:
    """
    Remove all trading system tasks from Windows Task Scheduler
    
    Returns:
    --------
    bool
        True if tasks were removed successfully, False otherwise
    """
    try:
        # Get all tasks in the IntraDayML folder
        result = subprocess.run(
            ["schtasks", "/query", "/tn", "IntraDayML", "/fo", "csv"],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            logger.info("No IntraDayML tasks found to remove")
            return True
            
        # Parse the CSV output to get task names
        lines = result.stdout.splitlines()
        if len(lines) <= 1:
            logger.info("No IntraDayML tasks found to remove")
            return True
            
        # Skip the header row
        for line in lines[1:]:
            if line.strip():
                # Extract task name from CSV line
                parts = line.split(',')
                if len(parts) > 0:
                    task_path = parts[0].strip('"')
                    # Delete the task
                    delete_result = subprocess.run(
                        ["schtasks", "/delete", "/tn", task_path, "/f"],
                        capture_output=True, text=True
                    )
                    
                    if delete_result.returncode == 0:
                        logger.info(f"Removed task: {task_path}")
                    else:
                        logger.error(f"Failed to remove task {task_path}: {delete_result.stderr}")
                        
        # Try to remove the folder itself
        folder_result = subprocess.run(
            ["schtasks", "/delete", "/tn", "IntraDayML", "/f"],
            capture_output=True, text=True
        )
        
        if folder_result.returncode == 0:
            logger.info("Removed IntraDayML task folder")
        else:
            logger.warning(f"Could not remove IntraDayML task folder: {folder_result.stderr}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error removing tasks: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up Windows Task Scheduler for intraday ML trading system')
    parser.add_argument('--user', type=str, help='User to run the tasks as')
    parser.add_argument('--remove', action='store_true', help='Remove existing tasks instead of creating them')
    
    args = parser.parse_args()
    
    if args.remove:
        # Remove existing tasks
        if remove_tasks():
            print("Successfully removed trading system tasks")
        else:
            print("Failed to remove some tasks")
            sys.exit(1)
    else:
        # Set up tasks
        user = args.user or os.getlogin()
        
        # Prompt for password if a specific user was provided
        password = None
        if args.user:
            password = getpass.getpass(f"Enter password for {user}: ")
            
        # Create tasks
        successful_tasks = setup_main_tasks(user, password)
        
        if successful_tasks:
            print(f"Successfully created {len(successful_tasks)} scheduled tasks:")
            for task in successful_tasks:
                print(f"  - {task}")
        else:
            print("Failed to create any scheduled tasks")
            sys.exit(1) 