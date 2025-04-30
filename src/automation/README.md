# Automation System for Intraday ML Trading

This directory contains components for automating the intraday ML trading system, including task scheduling, orchestration, and Windows Task Scheduler integration.

## Components

- **`task_orchestrator.py`**: Core orchestration system with retry mechanisms, dependency handling, and scheduling
- **`master_orchestration.py`**: Main entry point that sets up all tasks and their dependencies
- **`windows_scheduler_setup.py`**: Script to set up Windows Task Scheduler for automated operation

## Configuration

Automation configuration is stored in `src/config/automation_config.yaml` and includes:

- Task schedules
- Retry policies
- Task timeouts
- System dependencies
- Global context parameters
- Notification settings

## Usage

### Manual Execution

Run the master orchestration script:

```bash
python master_orchestration.py --config ../config/automation_config.yaml
```

Run a specific task:

```bash
python master_orchestration.py --execute download_data
```

Check system status:

```bash
python master_orchestration.py --status
```

### Setting Up Windows Task Scheduler

To set up automated execution on Windows:

```bash
python windows_scheduler_setup.py
```

This will create scheduled tasks for:
- Daily system startup
- System health checks
- Weekly maintenance
- System recovery after reboots

To remove all scheduled tasks:

```bash
python windows_scheduler_setup.py --remove
```

## Task Dependency Graph

The system follows this dependency graph:

```
download_data → generate_features → detect_regime → run_paper_trading → generate_performance_report
                                  ↓
                        analyze_pairs
                        train_ml_models
                        optimize_parameters
```

## Recovery and Resilience

The system includes:

- Automatic retry mechanisms for failed tasks
- Dependency checking to ensure proper execution order
- Windows Task Scheduler integration for system recovery
- Critical task identification for prioritization

## Extending

To add a new automated task:

1. Implement the task function in the appropriate module
2. Register the task in `master_orchestration.py` or add to configuration
3. Define dependencies and retry policies
4. Update the Windows Task Scheduler setup if needed

## Monitoring

Task execution status can be monitored via:

- Log files in the configured log directory
- Status commands in the master orchestration script
- Windows Task Scheduler history 