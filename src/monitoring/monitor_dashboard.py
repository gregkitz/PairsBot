#!/usr/bin/env python3
"""
Monitoring Dashboard for Intraday ML Trading.

This script integrates the real-time signal processing,
alert system, and dashboard functionality to provide
comprehensive monitoring for the intraday ML trading system.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server use

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Import local modules
from src.monitoring.alert_system import AlertSystem
from src.real_time.signal_optimizer import RealTimeSignalOptimizer
from src.ml_enhancements.regime_detection.regime_detector import RegimeDetector
from src.ml_enhancements.model_retraining import ModelRetrainingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'monitoring_dashboard.log'))
    ]
)
logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for intraday ML trading.
    
    This class integrates real-time signal processing, alert system,
    and performance monitoring to provide a comprehensive view
    of the trading system's operation.
    """
    
    def __init__(self, config=None, output_dir=None):
        """
        Initialize the monitoring dashboard.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration for the dashboard
        output_dir : str, optional
            Output directory for dashboard files
        """
        self.config = config or {}
        self.output_dir = output_dir or os.path.join('output', 'dashboard')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        
        # Initialize components
        self.alert_system = AlertSystem(
            config=self.config.get('alerts', {}),
            data_dir=os.path.join(self.output_dir, 'alerts')
        )
        
        self.signal_optimizer = RealTimeSignalOptimizer(
            config=self.config.get('signal_processing', {})
        )
        
        self.regime_detector = RegimeDetector()
        
        self.model_manager = ModelRetrainingManager(
            models_dir=self.config.get('models_dir', 'models/intraday'),
            output_dir=os.path.join(self.output_dir, 'model_tracking')
        )
        
        # Configure update intervals
        self.update_intervals = {
            'dashboard': self.config.get('dashboard_update_seconds', 60),
            'model_check': self.config.get('model_check_minutes', 60) * 60,
            'data_drift_check': self.config.get('data_drift_check_minutes', 30) * 60
        }
        
        # Initialize monitoring state
        self.monitoring_metrics = {
            'system_health': {
                'status': 'initializing',
                'last_update': datetime.now().isoformat(),
                'uptime_seconds': 0,
                'latency_ms': {},
                'error_count': 0
            },
            'performance': {
                'signals_generated': 0,
                'processing_times_ms': []
            },
            'models': {},
            'regimes': []
        }
        
        # Initialize running state
        self.running = False
        self.start_time = None
        self.update_threads = {}
        
        logger.info(f"Monitoring dashboard initialized in {self.output_dir}")
    
    def start(self):
        """Start the monitoring dashboard."""
        if self.running:
            logger.warning("Monitoring dashboard already running")
            return False
        
        self.running = True
        self.start_time = datetime.now()
        self.monitoring_metrics['system_health']['status'] = 'running'
        
        # Start update threads
        self._start_update_threads()
        
        logger.info("Monitoring dashboard started")
        return True
    
    def stop(self):
        """Stop the monitoring dashboard."""
        if not self.running:
            logger.warning("Monitoring dashboard not running")
            return False
        
        self.running = False
        self.monitoring_metrics['system_health']['status'] = 'stopped'
        
        # Wait for threads to stop
        for name, thread in self.update_threads.items():
            if thread.is_alive():
                logger.info(f"Waiting for {name} thread to stop...")
        
        logger.info("Monitoring dashboard stopped")
        return True
    
    def _start_update_threads(self):
        """Start all update threads."""
        # Dashboard update thread
        dashboard_thread = Thread(
            target=self._dashboard_update_loop,
            daemon=True,
            name="dashboard_updater"
        )
        dashboard_thread.start()
        self.update_threads['dashboard'] = dashboard_thread
        
        # Model check thread
        model_thread = Thread(
            target=self._model_check_loop,
            daemon=True,
            name="model_checker"
        )
        model_thread.start()
        self.update_threads['model_check'] = model_thread
        
        # Data drift check thread
        data_thread = Thread(
            target=self._data_drift_check_loop,
            daemon=True,
            name="data_drift_checker"
        )
        data_thread.start()
        self.update_threads['data_drift'] = data_thread
    
    def _dashboard_update_loop(self):
        """Main dashboard update loop."""
        logger.info("Starting dashboard update loop")
        
        while self.running:
            try:
                # Update system health metrics
                self._update_system_health()
                
                # Generate dashboard files
                self._generate_dashboard_files()
                
                # Sleep for update interval
                time.sleep(self.update_intervals['dashboard'])
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                self.alert_system.record_execution_issue(
                    issue_type="error",
                    message=f"Dashboard update error: {e}"
                )
                time.sleep(10)  # Sleep briefly to avoid rapid error loops
    
    def _model_check_loop(self):
        """Model check loop for detecting model degradation."""
        logger.info("Starting model check loop")
        
        while self.running:
            try:
                # Perform model checks
                self._check_model_performance()
                
                # Sleep for update interval
                time.sleep(self.update_intervals['model_check'])
                
            except Exception as e:
                logger.error(f"Error in model check loop: {e}")
                self.alert_system.record_execution_issue(
                    issue_type="error",
                    message=f"Model check error: {e}"
                )
                time.sleep(60)  # Longer sleep for less critical thread
    
    def _data_drift_check_loop(self):
        """Data drift check loop for detecting data distribution changes."""
        logger.info("Starting data drift check loop")
        
        while self.running:
            try:
                # Perform data drift checks
                self._check_data_drift()
                
                # Sleep for update interval
                time.sleep(self.update_intervals['data_drift_check'])
                
            except Exception as e:
                logger.error(f"Error in data drift check loop: {e}")
                self.alert_system.record_execution_issue(
                    issue_type="error",
                    message=f"Data drift check error: {e}"
                )
                time.sleep(60)  # Longer sleep for less critical thread
    
    def _update_system_health(self):
        """Update system health metrics."""
        now = datetime.now()
        uptime = (now - self.start_time).total_seconds()
        
        self.monitoring_metrics['system_health'].update({
            'last_update': now.isoformat(),
            'uptime_seconds': uptime,
            'status': 'running'
        })
        
        # Get signal processing performance metrics
        processing_metrics = self.signal_optimizer.get_performance_metrics()
        
        # Update latency metrics
        for timing_type, metrics in processing_metrics.items():
            if timing_type != 'system':
                self.monitoring_metrics['system_health']['latency_ms'][timing_type] = {
                    'avg': metrics.get('avg_ms'),
                    'p95': metrics.get('p95_ms'),
                    'last': metrics.get('last_ms')
                }
    
    def detect_regime(self, spread_data):
        """
        Detect market regime and update alert system.
        
        Parameters:
        -----------
        spread_data : pd.DataFrame
            Spread data for regime detection
        
        Returns:
        --------
        int
            Detected regime ID
        """
        # Detect regime
        try:
            regime_results = self.regime_detector.detect_spread_regime(spread_data)
            regime_id = regime_results.get('regime_id', 0)
            
            # Get regime description
            regime_type = self.regime_detector._determine_regime_type(
                regime_results.get('regime_data', pd.DataFrame())
            )
            
            # Update monitoring metrics
            self.monitoring_metrics['regimes'].append({
                'timestamp': datetime.now().isoformat(),
                'regime_id': regime_id,
                'regime_type': regime_type,
                'details': regime_results
            })
            
            # Trim history to last 100 entries
            if len(self.monitoring_metrics['regimes']) > 100:
                self.monitoring_metrics['regimes'] = self.monitoring_metrics['regimes'][-100:]
            
            # Update alert system
            self.alert_system.update_regime(
                new_regime=regime_id,
                regime_data={
                    'regime_type': regime_type,
                    'details': {
                        k: v for k, v in regime_results.items() 
                        if k not in ['regime_data']  # Exclude large dataframes
                    }
                }
            )
            
            return regime_id
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            self.alert_system.record_execution_issue(
                issue_type="error",
                message=f"Regime detection error: {e}"
            )
            return None
    
    def _check_model_performance(self):
        """Check model performance and detect degradation."""
        try:
            # Get list of registered models
            models = self.model_manager.model_registry
            
            for model_id, model_info in models.items():
                # Get current metrics
                current_metrics = model_info.get('current_metrics', {})
                if not current_metrics:
                    continue
                
                # Get baseline metrics
                baseline_metrics = model_info.get('initial_metrics', {})
                
                # Update monitoring metrics
                self.monitoring_metrics['models'][model_id] = {
                    'last_check': datetime.now().isoformat(),
                    'current_metrics': current_metrics,
                    'baseline_metrics': baseline_metrics
                }
                
                # Check for degradation with alert system
                self.alert_system.check_model_degradation(
                    model_id=model_id,
                    current_metrics=current_metrics
                )
                
                # Log check
                logger.debug(f"Checked model {model_id} performance")
        
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            self.alert_system.record_execution_issue(
                issue_type="error",
                message=f"Model performance check error: {e}"
            )
    
    def _check_data_drift(self):
        """Check for data distribution drift."""
        try:
            # For each symbol with data, calculate statistics and check for drift
            for symbol, data in self.signal_optimizer.latest_prices.items():
                if data is None or data.empty:
                    continue
                
                # Calculate current statistics
                stats = {}
                
                if 'close' in data.columns:
                    close_prices = data['close'].values
                    
                    # Calculate basic statistics
                    stats['mean'] = np.mean(close_prices)
                    stats['std'] = np.std(close_prices)
                    stats['median'] = np.median(close_prices)
                    stats['min'] = np.min(close_prices)
                    stats['max'] = np.max(close_prices)
                    
                    # Calculate return statistics
                    returns = np.diff(close_prices) / close_prices[:-1]
                    stats['return_mean'] = np.mean(returns)
                    stats['return_std'] = np.std(returns)
                    stats['return_skew'] = (
                        np.mean((returns - np.mean(returns))**3) / 
                        np.mean((returns - np.mean(returns))**2)**(3/2)
                    )
                    
                    # Create an ID for this data source
                    data_id = f"{symbol}_price"
                    
                    # Register baseline if not already registered
                    if data_id not in self.alert_system.baseline_data_stats:
                        self.alert_system.register_data_distribution(data_id, stats)
                    
                    # Check for drift
                    self.alert_system.check_data_drift(data_id, stats)
                    
                    logger.debug(f"Checked data drift for {symbol}")
        
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            self.alert_system.record_execution_issue(
                issue_type="error",
                message=f"Data drift check error: {e}"
            )
    
    def _generate_dashboard_files(self):
        """Generate dashboard HTML and visualization files."""
        try:
            # Create plots
            self._generate_performance_plots()
            
            # Save monitoring metrics to JSON
            metrics_file = os.path.join(self.output_dir, 'data', 'monitoring_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.monitoring_metrics, f, indent=2)
            
            # Save alert stats to JSON
            alert_stats_file = os.path.join(self.output_dir, 'data', 'alert_stats.json')
            with open(alert_stats_file, 'w') as f:
                json.dump(self.alert_system.get_alert_stats(), f, indent=2)
            
            # Save recent alerts to JSON
            recent_alerts_file = os.path.join(self.output_dir, 'data', 'recent_alerts.json')
            with open(recent_alerts_file, 'w') as f:
                json.dump(self.alert_system.get_recent_alerts(20), f, indent=2)
            
            # Generate HTML dashboard
            self._generate_html_dashboard()
            
            logger.debug("Generated dashboard files")
            
        except Exception as e:
            logger.error(f"Error generating dashboard files: {e}")
            self.alert_system.record_execution_issue(
                issue_type="error",
                message=f"Dashboard generation error: {e}"
            )
    
    def _generate_performance_plots(self):
        """Generate performance visualization plots."""
        plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Create latency plot
        try:
            plt.figure(figsize=(10, 6))
            
            latency_metrics = self.monitoring_metrics['system_health']['latency_ms']
            metric_types = list(latency_metrics.keys())
            
            if metric_types:
                # Extract metrics
                avg_values = [latency_metrics[t].get('avg', 0) for t in metric_types]
                p95_values = [latency_metrics[t].get('p95', 0) for t in metric_types]
                
                # Plot
                x = np.arange(len(metric_types))
                width = 0.35
                
                plt.bar(x - width/2, avg_values, width, label='Avg (ms)')
                plt.bar(x + width/2, p95_values, width, label='P95 (ms)')
                
                plt.ylabel('Latency (ms)')
                plt.title('Processing Latency by Component')
                plt.xticks(x, metric_types)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'latency.png'))
                plt.close()
        except Exception as e:
            logger.error(f"Error generating latency plot: {e}")
        
        # Create regime history plot
        try:
            if self.monitoring_metrics['regimes']:
                plt.figure(figsize=(10, 6))
                
                regimes = self.monitoring_metrics['regimes']
                timestamps = [datetime.fromisoformat(r['timestamp']) for r in regimes]
                regime_ids = [r['regime_id'] for r in regimes]
                
                plt.plot(timestamps, regime_ids, 'o-')
                plt.ylabel('Regime ID')
                plt.title('Market Regime History')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'regime_history.png'))
                plt.close()
        except Exception as e:
            logger.error(f"Error generating regime plot: {e}")
        
        # Create model performance plot
        try:
            models = self.monitoring_metrics['models']
            if models:
                for model_id, model_info in models.items():
                    if 'current_metrics' not in model_info or 'baseline_metrics' not in model_info:
                        continue
                    
                    plt.figure(figsize=(10, 6))
                    
                    current = model_info['current_metrics']
                    baseline = model_info['baseline_metrics']
                    
                    # Find metrics present in both
                    common_metrics = [m for m in current.keys() if m in baseline]
                    
                    if common_metrics:
                        # Extract values
                        current_values = [current[m] for m in common_metrics]
                        baseline_values = [baseline[m] for m in common_metrics]
                        
                        # Plot
                        x = np.arange(len(common_metrics))
                        width = 0.35
                        
                        plt.bar(x - width/2, baseline_values, width, label='Baseline')
                        plt.bar(x + width/2, current_values, width, label='Current')
                        
                        plt.ylabel('Metric Value')
                        plt.title(f'Model Performance: {model_id}')
                        plt.xticks(x, common_metrics)
                        plt.legend()
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(plots_dir, f'model_{model_id}.png'))
                        plt.close()
        except Exception as e:
            logger.error(f"Error generating model plots: {e}")
    
    def _generate_html_dashboard(self):
        """Generate HTML dashboard file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_interval = self.update_intervals['dashboard']
        
        # Get system status
        system_health = self.monitoring_metrics['system_health']
        status = system_health['status']
        uptime_seconds = system_health['uptime_seconds']
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        # Count alerts
        alert_stats = self.alert_system.get_alert_stats()
        alert_counts = alert_stats.get('alert_counts', {})
        
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Intraday ML Monitoring Dashboard</title>
            <meta http-equiv="refresh" content="{update_interval}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-container {{ display: flex; flex-wrap: wrap; }}
                .dashboard-item {{ margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 15px; }}
                .dashboard-item img {{ max-width: 100%; }}
                h1, h2, h3 {{ color: #333; }}
                .status {{ padding: 5px 10px; border-radius: 4px; display: inline-block; }}
                .status-running {{ background-color: #28a745; color: white; }}
                .status-error {{ background-color: #dc3545; color: white; }}
                .status-warning {{ background-color: #ffc107; color: black; }}
                .status-stopped {{ background-color: #6c757d; color: white; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .alert {{ margin-bottom: 5px; padding: 8px; border-radius: 4px; }}
                .alert-info {{ background-color: #d1ecf1; color: #0c5460; }}
                .alert-warning {{ background-color: #fff3cd; color: #856404; }}
                .alert-error {{ background-color: #f8d7da; color: #721c24; }}
                .timestamp {{ color: #666; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <h1>Intraday ML Monitoring Dashboard</h1>
            <p class="timestamp">Last updated: {timestamp}</p>
            
            <div class="dashboard-container">
                <!-- System Status -->
                <div class="dashboard-item" style="width: 300px;">
                    <h2>System Status</h2>
                    <p>Status: <span class="status status-{status}">{status.upper()}</span></p>
                    <p>Uptime: {uptime_str}</p>
                    <p>Alert Counts:</p>
                    <ul>
                        <li>Total Today: {alert_counts.get('daily', 0)}</li>
                        <li>Regime Changes: {alert_counts.get('regime_change', 0)}</li>
                        <li>Model Degradation: {alert_counts.get('model_degradation', 0)}</li>
                        <li>Data Drift: {alert_counts.get('data_drift', 0)}</li>
                        <li>Execution Issues: {alert_counts.get('execution_issue', 0)}</li>
                    </ul>
                    <p>Current Regime: {alert_stats.get('current_regime', 'Unknown')}</p>
                </div>
                
                <!-- Performance Visualization -->
                <div class="dashboard-item" style="width: 550px;">
                    <h2>Performance Metrics</h2>
                    <img src="plots/latency.png" alt="Latency Metrics">
                </div>
                
                <!-- Recent Alerts -->
                <div class="dashboard-item" style="width: 300px;">
                    <h2>Recent Alerts</h2>
                    <div id="alerts-container" style="max-height: 400px; overflow-y: auto;">
        """
        
        # Add recent alerts
        recent_alerts = self.alert_system.get_recent_alerts(10)
        for alert in reversed(recent_alerts):
            alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
            level = alert['level']
            message = alert['message']
            alert_type = alert['type']
            
            html += f"""
                        <div class="alert alert-{level}">
                            <strong>{alert_time} - {alert_type}:</strong> {message}
                        </div>
            """
        
        # Continue HTML
        html += """
                    </div>
                </div>
                
                <!-- Regime History -->
                <div class="dashboard-item" style="width: 550px;">
                    <h2>Market Regime History</h2>
                    <img src="plots/regime_history.png" alt="Regime History">
                </div>
                
                <!-- Model Performance -->
                <div class="dashboard-item" style="width: 300px;">
                    <h2>Model Performance</h2>
        """
        
        # Add model performance
        for model_id in self.monitoring_metrics['models'].keys():
            html += f"""
                    <h3>{model_id}</h3>
                    <img src="plots/model_{model_id}.png" alt="Model Performance">
            """
        
        # Finish HTML
        html += """
                </div>
            </div>
            
            <script>
                // Auto-scroll to bottom of alerts
                window.onload = function() {
                    var alertsContainer = document.getElementById('alerts-container');
                    alertsContainer.scrollTop = alertsContainer.scrollHeight;
                };
            </script>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(self.output_dir, 'index.html'), 'w') as f:
            f.write(html)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start monitoring dashboard')
    
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join('config', 'monitoring.json'),
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join('output', 'dashboard'),
        help='Output directory for dashboard files'
    )
    
    parser.add_argument(
        '--update-interval',
        type=int,
        default=60,
        help='Dashboard update interval in seconds'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    else:
        logger.warning(f"Configuration file not found: {args.config}")
        config = {
            'dashboard_update_seconds': args.update_interval,
            'alerts': {
                'enabled': True,
                'channels': ['console', 'file']
            }
        }
    
    # Override with command line arguments
    config['dashboard_update_seconds'] = args.update_interval
    
    # Create dashboard
    dashboard = MonitoringDashboard(
        config=config,
        output_dir=args.output_dir
    )
    
    # Start dashboard
    dashboard.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping dashboard")
        dashboard.stop()
    
    logger.info("Dashboard stopped, exiting")

if __name__ == '__main__':
    main() 