# Monitoring and Real-Time Processing Components

This document provides an overview of the monitoring and real-time processing components implemented for the intraday ML trading system. These components are essential for ensuring the system operates reliably, efficiently, and can quickly adapt to changing market conditions.

## Table of Contents

1. [Real-Time Signal Optimizer](#real-time-signal-optimizer)
2. [Alert System](#alert-system)
3. [Monitoring Dashboard](#monitoring-dashboard)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Usage Guidelines](#usage-guidelines)

## Real-Time Signal Optimizer

The `RealTimeSignalOptimizer` provides optimized implementations of signal generation algorithms for low-latency trading applications. It dramatically improves processing speed through several optimization techniques:

### Key Features

- **Incremental Data Processing**: Efficiently processes only new data points, reducing computational overhead
- **Feature Caching**: Reuses previously calculated features to minimize redundant calculations
- **Vectorized Operations**: Utilizes NumPy's vectorized operations for maximum performance
- **Pre-allocation**: Pre-allocates memory for arrays to avoid expensive runtime memory allocations
- **Performance Tracking**: Monitors and reports on execution times for all components

### Performance Metrics

The optimizer achieves the following performance targets:
- Signal generation latency < 500ms per pair
- Full system update (10 pairs) < 5 seconds
- Feature calculation < 200ms per bar per pair

### Usage Example

```python
from src.real_time.signal_optimizer import RealTimeSignalOptimizer

# Initialize optimizer
optimizer = RealTimeSignalOptimizer(config={
    'use_incremental_updates': True,
    'use_feature_caching': True,
    'parallel_processing': False,
    'max_lookback': 100
})

# Process new data
result = optimizer.process_new_data('ES_NQ', new_data_df)

# Get performance metrics
metrics = optimizer.get_performance_metrics()
print(f"Average processing time: {metrics['total_processing']['avg_ms']} ms")
```

## Alert System

The `AlertSystem` monitors critical aspects of the trading system and sends notifications when important events occur or anomalies are detected.

### Key Features

- **Regime Change Detection**: Alerts when market regime changes are detected
- **Model Degradation Monitoring**: Detects and alerts on model performance deterioration
- **Data Drift Detection**: Monitors data distributions for significant changes
- **Execution Issue Tracking**: Tracks and alerts on system execution problems
- **Multi-Channel Notifications**: Supports console, file, email, and SMS notifications
- **Alert Debouncing**: Prevents alert fatigue through intelligent alert throttling

### Alert Types

| Alert Type | Description | Default Severity |
|------------|-------------|------------------|
| Regime Change | Indicates a shift in market dynamics | Warning |
| Model Degradation | Model performance has deteriorated beyond threshold | Warning |
| Data Drift | Data statistics have shifted beyond threshold | Warning |
| Execution Issue | System execution problems detected | Error |

### Usage Example

```python
from src.monitoring.alert_system import AlertSystem

# Initialize alert system
alert_system = AlertSystem(config={
    'enabled': True,
    'channels': ['console', 'file', 'email'],
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.example.com',
        'port': 587,
        'sender': 'alerts@example.com',
        'recipients': ['trader@example.com']
    },
    'debounce': {
        'enable_debounce': True,
        'debounce_period_minutes': 60
    }
})

# Register model for monitoring
alert_system.register_model('signal_filter', {'accuracy': 0.85, 'f1': 0.82})

# Check for model degradation
alert_system.check_model_degradation('signal_filter', {'accuracy': 0.65, 'f1': 0.60})

# Update regime
alert_system.update_regime(new_regime=2, 
                         regime_data={'regime_type': 'trending', 'confidence': 0.92})
```

## Monitoring Dashboard

The `MonitoringDashboard` provides a comprehensive view of the trading system's real-time operation and performance metrics.

### Key Features

- **Real-time Performance Visualization**: Displays key performance metrics in real-time
- **System Health Monitoring**: Shows system status and resource utilization
- **Alert History**: Displays recent system alerts with relevant details
- **Regime Tracking**: Visualizes regime changes over time
- **Model Performance Tracking**: Tracks and displays model performance metrics

### Component Integration

The dashboard integrates several components:
- Real-time signal optimizer for performance tracking
- Alert system for notifications and monitoring
- Regime detector for market regime tracking
- Model retraining manager for model performance monitoring

### Usage Example

```python
from src.monitoring.monitor_dashboard import MonitoringDashboard

# Initialize dashboard
dashboard = MonitoringDashboard(
    config={
        'dashboard_update_seconds': 60,
        'model_check_minutes': 60,
        'data_drift_check_minutes': 30,
        'alerts': {
            'enabled': True,
            'channels': ['console', 'file']
        }
    },
    output_dir='output/dashboard'
)

# Start the dashboard
dashboard.start()

# Dashboard is now available at: output/dashboard/index.html
```

## Performance Benchmarking

The `benchmark_real_time.py` script provides comprehensive benchmarking of signal processing performance to ensure the system meets latency requirements.

### Key Benchmarks

- **Standard vs. Optimized Processing**: Compares the performance of standard signal processing with optimized implementations
- **Incremental vs. Full Data Processing**: Measures the benefits of incremental processing
- **Scaling Performance**: Evaluates how processing time scales with the number of symbols

### Benchmark Reports

The benchmarking script generates detailed HTML reports with:
- Performance comparison charts
- Latency metrics for all components
- Improvement percentages
- Scaling behavior analysis

### Running Benchmarks

```bash
# Run with default settings
python -m src.diagnostics.benchmark_real_time

# Run with custom settings
python -m src.diagnostics.benchmark_real_time --symbols "ES_NQ,CL_GC,ZN_ZB" --iterations 20 --output-dir output/benchmarks
```

## Usage Guidelines

### System Requirements

- Python 3.7+
- NumPy, Pandas, Matplotlib for core functionality
- SMTP server access for email alerts
- Twilio account for SMS alerts (optional)

### Configuration

The components accept JSON configuration files with the following structure:

```json
{
  "signal_processing": {
    "use_incremental_updates": true,
    "use_feature_caching": true,
    "parallel_processing": false,
    "max_lookback": 100
  },
  "alerts": {
    "enabled": true,
    "channels": ["console", "file", "email"],
    "email": {
      "enabled": true,
      "smtp_server": "smtp.example.com",
      "port": 587,
      "sender": "alerts@example.com",
      "recipients": ["trader@example.com"]
    },
    "alert_levels": {
      "model_degradation_threshold": 0.2,
      "data_drift_threshold": 0.1
    }
  },
  "dashboard_update_seconds": 60,
  "model_check_minutes": 60,
  "data_drift_check_minutes": 30
}
```

### Best Practices

1. **Performance Monitoring**: Regularly check performance metrics to ensure the system meets latency requirements
2. **Alert Configuration**: Configure alert thresholds carefully to avoid alert fatigue
3. **Resource Management**: Monitor resource utilization, especially when scaling to multiple pairs
4. **Dashboard Refresh Rate**: Adjust dashboard refresh rate based on system load and performance needs
5. **Logging**: Enable detailed logging during initial deployment and reduce to info level for production 