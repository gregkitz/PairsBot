#!/usr/bin/env python3
"""
Alert System for Intraday ML Trading.

This module provides a specialized alert system for monitoring critical aspects
of the intraday ML trading system, including:
- Regime changes detection
- Data and model drift monitoring
- Execution monitoring and alerting
"""

import os
import logging
import smtplib
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from threading import Thread
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logger
logger = logging.getLogger(__name__)

class AlertSystem:
    """
    Specialized alert system for intraday ML trading monitoring.
    
    This class provides alert functionality specifically designed for:
    - Detecting and alerting on market regime changes
    - Monitoring model performance degradation
    - Tracking execution issues
    - Monitoring data quality and drift
    """
    
    def __init__(self, 
                config: Optional[Dict] = None,
                data_dir: Optional[str] = None):
        """
        Initialize the alert system.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary for the alert system
        data_dir : str, optional
            Directory to store alert logs and state
        """
        self.config = config or {}
        self.data_dir = data_dir or os.path.join('output', 'alerts')
        
        # Create data directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'logs'), exist_ok=True)
        
        # Configure alert channels
        self.enabled = self.config.get('enabled', True)
        self.channels = self.config.get('channels', ['console', 'file'])
        
        # Configure email settings if needed
        self.email_config = self.config.get('email', {})
        self.email_enabled = self.email_config.get('enabled', False)
        
        # Configure SMS settings if needed
        self.sms_config = self.config.get('sms', {})
        self.sms_enabled = self.sms_config.get('enabled', False)
        
        # Set up alert level thresholds
        self.alert_levels = self.config.get('alert_levels', {
            'model_degradation_threshold': 0.2,  # 20% degradation in model metrics
            'data_drift_threshold': 0.1,         # 10% drift in data distribution
            'execution_latency_threshold': 1.0,  # 1 second execution latency
            'model_confidence_threshold': 0.6,   # Minimum model confidence
            'max_consecutive_errors': 3          # Max consecutive errors
        })
        
        # Set up debounce settings to avoid alert fatigue
        self.debounce_settings = self.config.get('debounce', {
            'enable_debounce': True,
            'debounce_period_minutes': 60,       # Only alert once per hour for same alert
            'max_alerts_per_day': 20             # Limit total alerts per day
        })
        
        # Initialize alert state
        self.alert_history = []
        self.last_alert_times = {}
        self.alert_counts = {
            'daily': 0,
            'regime_change': 0,
            'model_degradation': 0,
            'data_drift': 0,
            'execution_issue': 0
        }
        
        # Initialize monitoring state
        self.current_regime = None
        self.previous_regime = None
        self.model_metrics = {}
        self.error_count = 0
        self.consecutive_errors = 0
        
        # For model monitoring
        self.baseline_model_metrics = {}
        self.model_history = {}
        
        # For data drift monitoring
        self.baseline_data_stats = {}
        self.data_history = {}
        
        # Load existing data if available
        self._load_state()
        
        # Reset daily counts at midnight
        self._schedule_daily_reset()
        
        logger.info(f"AlertSystem initialized with config: {self.config}")
    
    def reset_daily_counters(self):
        """Reset daily alert counters."""
        self.alert_counts['daily'] = 0
        logger.info("Daily alert counters reset")
    
    def _schedule_daily_reset(self):
        """Schedule a daily reset of alert counters at midnight."""
        def reset_job():
            while True:
                # Calculate time until midnight
                now = datetime.now()
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                seconds_until_midnight = (tomorrow - now).total_seconds()
                
                # Sleep until midnight
                time.sleep(seconds_until_midnight)
                
                # Reset counters
                self.reset_daily_counters()
        
        # Start reset thread
        thread = Thread(target=reset_job, daemon=True)
        thread.start()
    
    def _load_state(self):
        """Load alert system state from disk."""
        state_file = os.path.join(self.data_dir, 'alert_system_state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                    self.alert_history = state.get('alert_history', [])
                    self.last_alert_times = state.get('last_alert_times', {})
                    self.alert_counts = state.get('alert_counts', self.alert_counts)
                    self.current_regime = state.get('current_regime')
                    self.baseline_model_metrics = state.get('baseline_model_metrics', {})
                    
                    logger.info(f"Loaded alert system state with {len(self.alert_history)} historical alerts")
            except Exception as e:
                logger.error(f"Error loading alert system state: {e}")
    
    def _save_state(self):
        """Save alert system state to disk."""
        state_file = os.path.join(self.data_dir, 'alert_system_state.json')
        try:
            state = {
                'alert_history': self.alert_history[-100:],  # Keep last 100 alerts
                'last_alert_times': self.last_alert_times,
                'alert_counts': self.alert_counts,
                'current_regime': self.current_regime,
                'baseline_model_metrics': self.baseline_model_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alert system state: {e}")
    
    def register_model(self, model_id: str, metrics: Dict[str, float]):
        """
        Register a model for monitoring.
        
        Parameters:
        -----------
        model_id : str
            Unique identifier for the model
        metrics : dict
            Baseline performance metrics for the model
        """
        self.baseline_model_metrics[model_id] = metrics
        self.model_history[model_id] = [{
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'baseline': True
        }]
        logger.info(f"Registered model {model_id} for monitoring with metrics: {metrics}")
        self._save_state()
    
    def register_data_distribution(self, data_id: str, statistics: Dict[str, float]):
        """
        Register data distribution statistics for drift monitoring.
        
        Parameters:
        -----------
        data_id : str
            Unique identifier for the data source
        statistics : dict
            Statistical properties of the data (mean, std, etc.)
        """
        self.baseline_data_stats[data_id] = statistics
        self.data_history[data_id] = [{
            'timestamp': datetime.now().isoformat(),
            'statistics': statistics,
            'baseline': True
        }]
        logger.info(f"Registered data distribution {data_id} for monitoring")
        self._save_state()
    
    def update_regime(self, new_regime: Any, regime_data: Optional[Dict] = None):
        """
        Update the current market regime and send alerts if changed.
        
        Parameters:
        -----------
        new_regime : any
            Identifier for the new regime (could be int, string, etc.)
        regime_data : dict, optional
            Additional data about the regime change
        
        Returns:
        --------
        bool
            True if a regime change was detected, False otherwise
        """
        if self.current_regime is None:
            # Initial regime setting
            self.current_regime = new_regime
            logger.info(f"Initial regime set to {new_regime}")
            return False
        
        regime_changed = self.current_regime != new_regime
        if regime_changed:
            self.previous_regime = self.current_regime
            self.current_regime = new_regime
            
            # Prepare alert data
            alert_data = regime_data or {}
            alert_data.update({
                'previous_regime': self.previous_regime,
                'new_regime': new_regime,
                'change_time': datetime.now().isoformat()
            })
            
            # Send alert
            self.send_alert(
                alert_type="regime_change",
                message=f"Market regime changed from {self.previous_regime} to {new_regime}",
                level="warning",
                data=alert_data
            )
            
            # Update state
            self.alert_counts['regime_change'] += 1
            self._save_state()
            
            return True
        
        return False
    
    def check_model_degradation(self, model_id: str, current_metrics: Dict[str, float]):
        """
        Check for model performance degradation and send alerts if needed.
        
        Parameters:
        -----------
        model_id : str
            Unique identifier for the model
        current_metrics : dict
            Current performance metrics for the model
        
        Returns:
        --------
        tuple
            (is_degraded, degradation_amount)
        """
        if model_id not in self.baseline_model_metrics:
            logger.warning(f"Model {model_id} not registered for monitoring")
            return False, 0.0
        
        # Get baseline metrics
        baseline = self.baseline_model_metrics[model_id]
        
        # Track model history
        if model_id in self.model_history:
            self.model_history[model_id].append({
                'timestamp': datetime.now().isoformat(),
                'metrics': current_metrics
            })
        
        # Calculate degradation for each metric
        degradation = {}
        for metric, baseline_value in baseline.items():
            if metric in current_metrics and baseline_value > 0:
                degradation[metric] = (baseline_value - current_metrics[metric]) / baseline_value
        
        # Check if any metric exceeds degradation threshold
        threshold = self.alert_levels['model_degradation_threshold']
        is_degraded = False
        worst_degradation = 0.0
        worst_metric = None
        
        for metric, deg_value in degradation.items():
            if deg_value > worst_degradation:
                worst_degradation = deg_value
                worst_metric = metric
                
            if deg_value > threshold:
                is_degraded = True
        
        # Send alert if degraded
        if is_degraded:
            alert_data = {
                'model_id': model_id,
                'baseline_metrics': baseline,
                'current_metrics': current_metrics,
                'degradation': degradation,
                'worst_metric': worst_metric,
                'worst_degradation': worst_degradation
            }
            
            self.send_alert(
                alert_type="model_degradation",
                message=f"Model {model_id} performance degraded by {worst_degradation:.1%} on {worst_metric}",
                level="warning",
                data=alert_data
            )
            
            # Update state
            self.alert_counts['model_degradation'] += 1
            self._save_state()
        
        return is_degraded, worst_degradation
    
    def check_data_drift(self, data_id: str, current_stats: Dict[str, float]):
        """
        Check for data distribution drift and send alerts if needed.
        
        Parameters:
        -----------
        data_id : str
            Unique identifier for the data source
        current_stats : dict
            Current statistical properties of the data
        
        Returns:
        --------
        tuple
            (is_drifted, drift_amount)
        """
        if data_id not in self.baseline_data_stats:
            logger.warning(f"Data source {data_id} not registered for monitoring")
            return False, 0.0
        
        # Get baseline stats
        baseline = self.baseline_data_stats[data_id]
        
        # Track data history
        if data_id in self.data_history:
            self.data_history[data_id].append({
                'timestamp': datetime.now().isoformat(),
                'statistics': current_stats
            })
        
        # Calculate drift for each statistic
        drift = {}
        for stat, baseline_value in baseline.items():
            if stat in current_stats and abs(baseline_value) > 1e-10:  # Avoid division by zero
                drift[stat] = abs((baseline_value - current_stats[stat]) / baseline_value)
        
        # Check if any statistic exceeds drift threshold
        threshold = self.alert_levels['data_drift_threshold']
        is_drifted = False
        worst_drift = 0.0
        worst_stat = None
        
        for stat, drift_value in drift.items():
            if drift_value > worst_drift:
                worst_drift = drift_value
                worst_stat = stat
                
            if drift_value > threshold:
                is_drifted = True
        
        # Send alert if drifted
        if is_drifted:
            alert_data = {
                'data_id': data_id,
                'baseline_stats': baseline,
                'current_stats': current_stats,
                'drift': drift,
                'worst_stat': worst_stat,
                'worst_drift': worst_drift
            }
            
            self.send_alert(
                alert_type="data_drift",
                message=f"Data {data_id} has drifted by {worst_drift:.1%} on {worst_stat}",
                level="warning",
                data=alert_data
            )
            
            # Update state
            self.alert_counts['data_drift'] += 1
            self._save_state()
        
        return is_drifted, worst_drift
    
    def record_execution_issue(self, issue_type: str, message: str, data: Optional[Dict] = None):
        """
        Record an execution issue and send alert if needed.
        
        Parameters:
        -----------
        issue_type : str
            Type of execution issue ('latency', 'error', 'connectivity', etc.)
        message : str
            Description of the issue
        data : dict, optional
            Additional data about the issue
        """
        # Increment consecutive error count for errors
        if issue_type == 'error':
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
        
        # Prepare alert data
        alert_data = data or {}
        alert_data.update({
            'issue_type': issue_type,
            'timestamp': datetime.now().isoformat(),
            'consecutive_errors': self.consecutive_errors
        })
        
        # Determine alert level based on issue type and consecutive count
        level = "info"
        if issue_type == 'error' or issue_type == 'connectivity':
            level = "error"
        elif issue_type == 'latency' or issue_type == 'order_rejection':
            level = "warning"
        
        # Check if we should send an alert based on consecutive errors
        should_alert = (
            issue_type == 'error' and 
            self.consecutive_errors >= self.alert_levels['max_consecutive_errors']
        )
        
        # Also alert on critical issues immediately
        should_alert = should_alert or issue_type in ['connectivity', 'order_rejection']
        
        # Send alert if needed
        if should_alert:
            self.send_alert(
                alert_type="execution_issue",
                message=message,
                level=level,
                data=alert_data
            )
            
            # Update state
            self.alert_counts['execution_issue'] += 1
            self._save_state()
    
    def send_alert(self, alert_type: str, message: str, level: str = "info", data: Optional[Dict] = None):
        """
        Send an alert through all configured channels.
        
        Parameters:
        -----------
        alert_type : str
            Type of alert
        message : str
            Alert message
        level : str
            Alert level ('info', 'warning', 'error')
        data : dict, optional
            Additional alert data
        
        Returns:
        --------
        bool
            True if alert was sent, False if debounced or disabled
        """
        if not self.enabled:
            return False
        
        # Check daily alert limit
        if self.alert_counts['daily'] >= self.debounce_settings.get('max_alerts_per_day', 20):
            logger.warning(f"Daily alert limit reached, alert suppressed: {alert_type} - {message}")
            return False
        
        # Check debounce
        if self._should_debounce(alert_type, data):
            logger.info(f"Alert debounced: {alert_type} - {message}")
            return False
        
        # Create alert record
        alert = {
            'type': alert_type,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]  # Keep last 1000 alerts
        
        # Update last alert time for this type
        self.last_alert_times[alert_type] = datetime.now()
        
        # Increment daily count
        self.alert_counts['daily'] += 1
        
        # Process through channels
        self._process_alert_channels(alert)
        
        # Save updated state
        self._save_state()
        
        return True
    
    def _should_debounce(self, alert_type: str, data: Optional[Dict] = None) -> bool:
        """
        Check if an alert should be debounced.
        
        Parameters:
        -----------
        alert_type : str
            Type of alert
        data : dict, optional
            Alert data for more specific debouncing
        
        Returns:
        --------
        bool
            True if the alert should be debounced, False otherwise
        """
        # Skip debounce check if disabled
        if not self.debounce_settings.get('enable_debounce', True):
            return False
        
        # Get last alert time for this type
        last_time = self.last_alert_times.get(alert_type)
        if last_time is None:
            return False
        
        # Convert string to datetime if needed
        if isinstance(last_time, str):
            last_time = datetime.fromisoformat(last_time)
        
        # Check debounce period
        debounce_minutes = self.debounce_settings.get('debounce_period_minutes', 60)
        time_since_last = (datetime.now() - last_time).total_seconds() / 60
        
        return time_since_last < debounce_minutes
    
    def _process_alert_channels(self, alert: Dict):
        """
        Process an alert through all configured channels.
        
        Parameters:
        -----------
        alert : dict
            Alert data
        """
        # Format alert message
        timestamp = datetime.fromisoformat(alert['timestamp'])
        formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{alert['level'].upper()}] {formatted_timestamp} - {alert['type']}: {alert['message']}"
        
        # Process console channel
        if 'console' in self.channels:
            if alert['level'] == 'info':
                logger.info(formatted_message)
            elif alert['level'] == 'warning':
                logger.warning(formatted_message)
            elif alert['level'] == 'error':
                logger.error(formatted_message)
        
        # Process file channel
        if 'file' in self.channels:
            log_file = os.path.join(self.data_dir, 'logs', 'alerts.log')
            with open(log_file, 'a') as f:
                f.write(f"{formatted_message}\n")
        
        # Process email channel
        if 'email' in self.channels and self.email_enabled:
            try:
                self._send_email_alert(alert)
            except Exception as e:
                logger.error(f"Error sending email alert: {e}")
        
        # Process SMS channel
        if 'sms' in self.channels and self.sms_enabled:
            try:
                self._send_sms_alert(alert)
            except Exception as e:
                logger.error(f"Error sending SMS alert: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """
        Send an email alert.
        
        Parameters:
        -----------
        alert : dict
            Alert data
        """
        # Check if email settings are configured
        if not self.email_config.get('smtp_server') or not self.email_config.get('sender'):
            logger.warning("Email settings not fully configured, skipping email alert")
            return
        
        recipients = self.email_config.get('recipients', [])
        if not recipients:
            logger.warning("No email recipients configured, skipping email alert")
            return
        
        # Create email subject
        subject = f"Trading Alert: {alert['type'].upper()} - {alert['message'][:50]}"
        if len(alert['message']) > 50:
            subject += "..."
        
        # Create email body
        body = f"""
        <html>
        <body>
            <h2>Trading System Alert</h2>
            <p><b>Type:</b> {alert['type']}</p>
            <p><b>Level:</b> {alert['level'].upper()}</p>
            <p><b>Time:</b> {alert['timestamp']}</p>
            <p><b>Message:</b> {alert['message']}</p>
            
            <h3>Additional Data:</h3>
            <pre>{json.dumps(alert['data'], indent=2)}</pre>
        </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.email_config['sender']
        msg['To'] = ', '.join(recipients)
        
        # Attach HTML body
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server
        server = smtplib.SMTP(
            self.email_config['smtp_server'], 
            self.email_config.get('port', 587)
        )
        
        server.starttls()
        
        # Login if username and password are provided
        username = self.email_config.get('username')
        password = self.email_config.get('password')
        
        if username and password:
            server.login(username, password)
        
        # Send email
        server.sendmail(
            self.email_config['sender'],
            recipients,
            msg.as_string()
        )
        
        # Close connection
        server.quit()
        
        logger.info(f"Email alert sent: {subject}")
    
    def _send_sms_alert(self, alert: Dict):
        """
        Send an SMS alert.
        
        Parameters:
        -----------
        alert : dict
            Alert data
        """
        # Skip implementation if provider not specified
        provider = self.sms_config.get('provider')
        
        if not provider:
            logger.warning("SMS provider not specified, skipping SMS alert")
            return
        
        # Simple twilio implementation (requires twilio package)
        if provider == 'twilio':
            try:
                from twilio.rest import Client
                
                account_sid = self.sms_config.get('account_sid')
                auth_token = self.sms_config.get('auth_token')
                from_number = self.sms_config.get('from_number')
                to_numbers = self.sms_config.get('to_numbers', [])
                
                if not all([account_sid, auth_token, from_number, to_numbers]):
                    logger.warning("Incomplete Twilio settings, skipping SMS alert")
                    return
                
                # Format message (keep it short for SMS)
                max_length = 140
                message = f"{alert['level'].upper()}: {alert['type']} - {alert['message']}"
                if len(message) > max_length:
                    message = message[:max_length-3] + "..."
                
                # Create client and send message
                client = Client(account_sid, auth_token)
                
                for to_number in to_numbers:
                    client.messages.create(
                        body=message,
                        from_=from_number,
                        to=to_number
                    )
                
                logger.info(f"SMS alert sent to {len(to_numbers)} recipients")
            
            except ImportError:
                logger.error("Twilio package not installed, skipping SMS alert")
            except Exception as e:
                logger.error(f"Error sending SMS alert: {e}")
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
        --------
        dict
            Alert statistics
        """
        return {
            'total_alerts': len(self.alert_history),
            'alert_counts': self.alert_counts,
            'last_alert_times': self.last_alert_times,
            'current_regime': self.current_regime,
            'previous_regime': self.previous_regime,
            'today_count': self.alert_counts['daily']
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """
        Get recent alerts.
        
        Parameters:
        -----------
        limit : int
            Maximum number of alerts to return
        
        Returns:
        --------
        list
            List of recent alerts
        """
        return self.alert_history[-limit:] 