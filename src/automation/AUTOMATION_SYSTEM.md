# Intraday ML Trading System Automation

This document explains how the automation system works, including orchestration, monitoring, and startup/shutdown procedures.

## System Architecture

The automation system consists of several interconnected components:

1. **Task Orchestrator**: Core component that manages individual tasks, dependencies, and scheduling
2. **Master Orchestration**: High-level coordinator that registers and manages all system tasks
3. **Automation Monitor**: Monitors task execution and system health, generating alerts for issues
4. **System Startup**: Handles proper system initialization and recovery from failures
5. **System Shutdown**: Ensures graceful termination and cleanup of all processes
6. **Windows Task Scheduler Integration**: Schedules system startup and monitoring

## 1. Task Management

### Task Orchestrator

The `task_orchestrator.py` module provides:
- Task registration and dependency management
- Scheduling tasks based on daily, weekly, or interval specifications
- Automatic retries with exponential backoff for failed tasks
- Context passing between tasks

Example usage:
```python
# Create orchestrator
orchestrator = TaskOrchestrator('config.yaml')

# Register a task
orchestrator.register_function(
    name="download_data",
    function=download_historical_data,
    retry_policy=RetryPolicy(max_retries=5, retry_delay_seconds=60),
    critical=True
)

# Schedule task
orchestrator.schedule_task("download_data", "daily at 17:00")

# Run the scheduler
orchestrator.run_scheduler()
```

### Master Orchestration

The `master_orchestration.py` script:
- Manages the full task catalog for the system
- Sets up dependencies between system components
- Provides CLI interface for manual execution
- Handles configuration loading and reloading

## 2. System Monitoring

### Automation Monitor

The `automation_monitor.py` module:
- Tracks task execution success/failure
- Monitors system resources (CPU, memory, disk)
- Generates alerts for issues via logging and email
- Maintains status history for troubleshooting

Key monitoring capabilities:
- **Task Status Tracking**: Records execution times and results
- **Consecutive Failure Detection**: Alerts on repeated task failures
- **Resource Monitoring**: Alerts on high CPU/memory/disk usage
- **Status Persistence**: Saves status information to disk for recovery

Example usage:
```python
# Create monitor
monitor = AutomationMonitor('config.yaml')

# Run continuous monitoring
monitor.run_continuous_monitoring(master_orchestration)
```

## 3. Startup and Shutdown

### System Startup

The `system_startup.py` script:
- Performs pre-startup system checks
- Creates required directories if missing
- Recovers from previous crashes
- Starts the master orchestration process
- Launches the monitoring process

Startup sequence:
1. Load configuration
2. Check system status (directories, services, crash state)
3. Perform recovery if needed
4. Create lock file
5. Start monitoring process
6. Start master orchestration

### System Shutdown

The `system_shutdown.py` script:
- Saves system state before shutdown
- Gracefully terminates processes
- Performs forced termination if necessary
- Cleans up temporary files

Shutdown sequence:
1. Save current system state
2. Find all automation processes
3. Attempt graceful termination
4. Force kill any remaining processes
5. Clean up temporary files

## 4. Windows Task Scheduler Integration

The `windows_scheduler_setup.py` script:
- Creates a folder for intraday ML tasks
- Sets up daily startup tasks
- Configures system health check tasks
- Creates recovery tasks that run on system startup
- Manages task privileges and user context

Tasks created:
- **DailyStartup**: Runs the master orchestration daily
- **SystemHealthCheck**: Performs regular health checks
- **WeeklyMaintenance**: Cleans up logs and performs maintenance
- **SystemRecovery**: Runs after system restarts to ensure continuity

## 5. Configuration

The system is configured via `automation_config.yaml`, which contains sections for:
- Task schedules
- Retry policies
- Task dependencies
- Monitoring thresholds
- Startup/shutdown settings

Key configuration sections:
```yaml
# Task scheduling
task_schedules:
  download_data: "daily at 17:00"
  analyze_pairs: "weekly on sunday at 18:00"

# Monitoring configuration  
monitoring:
  check_interval_seconds: 300
  alert_thresholds:
    task_failure_count: 3

# Startup configuration
startup:
  recovery_enabled: true
  required_directories:
    - "data"
    - "logs"

# Shutdown configuration
shutdown:
  graceful_timeout: 30
  save_state: true
```

## 6. Resilience Features

The automation system includes several resilience features:
- **Automatic Recovery**: Detects and recovers from previous crashes
- **Retry Mechanisms**: Automatically retries failed tasks with backoff
- **Dependency Management**: Ensures tasks run in proper order
- **Process Monitoring**: Detects hung or misbehaving processes
- **State Persistence**: Maintains state between restarts
- **System Resource Monitoring**: Detects resource constraints
- **Alert Generation**: Notifies administrators of critical issues

## 7. Usage Guide

### Basic Operations

1. **Initial Setup**:
   ```
   python windows_scheduler_setup.py
   ```

2. **Manual System Start**:
   ```
   python system_startup.py --config ../config/automation_config.yaml
   ```

3. **Manual Task Execution**:
   ```
   python master_orchestration.py --execute download_data
   ```

4. **Check System Status**:
   ```
   python master_orchestration.py --status
   ```

5. **Graceful Shutdown**:
   ```
   python system_shutdown.py
   ```

### Troubleshooting

1. **Check Status Files**:
   - Examine files in the `logs/status` directory
   - `automation_status.json` contains latest task status
   - `shutdown_state.json` contains shutdown information

2. **Review Logs**:
   - Task orchestrator logs: `logs/task_orchestrator.log`
   - Monitor logs: `logs/automation_monitor.log`
   - Startup/shutdown logs: `logs/system_startup.log`, `logs/system_shutdown.log`

3. **Force Cleanup**:
   - If system is in inconsistent state: `python system_shutdown.py --force`
   - To remove all scheduled tasks: `python windows_scheduler_setup.py --remove`

4. **Restart with Recovery Disabled**:
   - In case of recovery issues: `python system_startup.py --no-recovery` 