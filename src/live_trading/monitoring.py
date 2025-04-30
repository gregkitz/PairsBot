"""
Monitoring module for the Intraday Statistical Arbitrage System.

This module provides monitoring, alerting, and health tracking functionality
for the live trading environment.
"""

import os
import json
import time
import logging
import smtplib
import threading
import requests
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np

from src.utils.error_handling import (
    BaseError, TradingError, SystemError, log_exception, 
    create_error_dict, handle_exceptions
)

# Configure logging
logger = logging.getLogger(__name__)


class TradingMonitor:
    """
    Trading monitoring system that provides:
    - Performance tracking
    - Email/SMS notifications
    - Error detection and reporting
    - System health monitoring
    - Performance dashboard data
    """
    
    def __init__(self,
                data_directory: Optional[str] = None,
                enable_email_alerts: bool = False,
                enable_sms_alerts: bool = False,
                email_settings: Optional[Dict] = None,
                sms_settings: Optional[Dict] = None,
                check_interval: int = 60,
                alert_levels: Optional[Dict] = None,
                max_errors_before_alert: int = 3,
                heartbeat_timeout: int = 60,
                debug_mode: bool = False):
        """
        Initialize the trading monitor.
        
        Parameters:
        -----------
        data_directory : str, optional
            Directory to store monitoring data. If None, a default directory will be created.
        enable_email_alerts : bool
            Whether to enable email alerts
        enable_sms_alerts : bool
            Whether to enable SMS alerts
        email_settings : dict, optional
            Email settings (smtp_server, port, username, password, sender, recipients)
        sms_settings : dict, optional
            SMS settings (provider, account_sid, auth_token, from_number, to_numbers)
        check_interval : int
            Interval in seconds to check system health
        alert_levels : dict, optional
            Alert thresholds (e.g., {'daily_loss_pct': 0.5, 'drawdown_pct': 0.3})
        max_errors_before_alert : int
            Maximum number of errors before sending an alert
        heartbeat_timeout : int
            Maximum time in seconds without heartbeat before alert
        debug_mode : bool
            Whether to run in debug mode with additional logging
        """
        # Set parameters
        self.enable_email_alerts = enable_email_alerts
        self.enable_sms_alerts = enable_sms_alerts
        self.check_interval = check_interval
        self.max_errors_before_alert = max_errors_before_alert
        self.heartbeat_timeout = heartbeat_timeout
        self.debug_mode = debug_mode
        
        # Set data directory
        if data_directory is None:
            self.data_directory = os.path.join(os.getcwd(), 'monitor_data')
        else:
            self.data_directory = data_directory
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Set email settings with defaults
        self.email_settings = {
            'smtp_server': 'smtp.gmail.com',
            'port': 587,
            'username': '',
            'password': '',
            'sender': '',
            'recipients': []
        }
        if email_settings:
            self.email_settings.update(email_settings)
        
        # Set SMS settings with defaults
        self.sms_settings = {
            'provider': 'twilio',
            'account_sid': '',
            'auth_token': '',
            'from_number': '',
            'to_numbers': []
        }
        if sms_settings:
            self.sms_settings.update(sms_settings)
        
        # Set alert levels with defaults
        self.alert_levels = {
            'daily_loss_pct': 0.5,  # Alert at 0.5% daily loss
            'drawdown_pct': 2.0,    # Alert at 2% drawdown from high
            'error_count': 3,       # Alert after 3 errors
            'missed_heartbeats': 3  # Alert after 3 missed heartbeats
        }
        if alert_levels:
            self.alert_levels.update(alert_levels)
        
        # Initialize monitoring state
        self._is_running = False
        self._monitor_thread = None
        self._last_check_time = datetime.now()
        self._last_heartbeat_time = datetime.now()
        self._error_count = 0
        self._alert_history = []
        
        # Initialize performance metrics
        self._metrics = {
            'start_time': datetime.now().isoformat(),
            'start_equity': 0.0,
            'current_equity': 0.0,
            'high_water_mark': 0.0,
            'drawdown_pct': 0.0,
            'daily_pnl': 0.0,
            'daily_pnl_pct': 0.0,
            'trade_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'error_count': 0,
            'system_status': 'offline'
        }
        
        # Initialize logs storage
        self._logs = {
            'errors': [],
            'warnings': [],
            'heartbeats': [],
            'trades': [],
            'alerts': []
        }
        
        # Load existing data if available
        self._load_data()
        
        # Configure logging level
        if self.debug_mode:
            logging.getLogger('src.live_trading.monitoring').setLevel(logging.DEBUG)
    
    def start(self) -> bool:
        """
        Start the monitoring system.
        
        Returns:
        --------
        bool
            True if successfully started, False otherwise
        """
        if self._is_running:
            logger.warning("Monitoring system is already running")
            return True
        
        try:
            logger.info("Starting trading monitoring system...")
            
            # Initialize start time
            self._metrics['start_time'] = datetime.now().isoformat()
            self._metrics['system_status'] = 'online'
            
            # Start monitoring thread
            self._is_running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
            # Log start event
            self._log_event('system', 'Monitoring system started')
            
            # Generate initial status report
            self._generate_status_report()
            
            logger.info("Trading monitoring system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the monitoring system.
        
        Returns:
        --------
        bool
            True if successfully stopped, False otherwise
        """
        if not self._is_running:
            logger.warning("Monitoring system is not running")
            return True
        
        try:
            logger.info("Stopping monitoring system...")
            
            # Stop monitoring
            self._is_running = False
            
            # Wait for monitor thread to terminate
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            # Update system status
            self._metrics['system_status'] = 'offline'
            
            # Log stop event
            self._log_event('system', 'Monitoring system stopped')
            
            # Save monitoring data
            self._save_data()
            
            # Generate final status report
            self._generate_status_report()
            
            logger.info("Monitoring system stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """
        Check if the monitoring system is running.
        
        Returns:
        --------
        bool
            True if running, False otherwise
        """
        return self._is_running
    
    def record_heartbeat(self, data: Dict) -> None:
        """
        Record a heartbeat from the trading system.
        
        Parameters:
        -----------
        data : dict
            Heartbeat data containing system status information
        """
        try:
            # Update last heartbeat time
            self._last_heartbeat_time = datetime.now()
            
            # Store heartbeat data
            heartbeat_data = {
                'time': self._last_heartbeat_time.isoformat(),
                'data': data
            }
            self._logs['heartbeats'].append(heartbeat_data)
            
            # Trim heartbeat history (keep the last 100 entries)
            if len(self._logs['heartbeats']) > 100:
                self._logs['heartbeats'] = self._logs['heartbeats'][-100:]
            
            # Update system status
            self._metrics['system_status'] = 'online'
            
            # Check if any alerts should be cleared
            self._check_alert_conditions()
        
        except Exception as e:
            logger.error(f"Error recording heartbeat: {str(e)}")
            self._error_count += 1
    
    def record_trade(self, trade_data: Dict) -> None:
        """
        Record a completed trade.
        
        Parameters:
        -----------
        trade_data : dict
            Trade data including symbol, entry/exit prices, quantities, timestamps, PnL
        """
        try:
            # Add timestamp if not provided
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            # Store trade data
            self._logs['trades'].append(trade_data)
            
            # Update trade statistics
            self._metrics['trade_count'] += 1
            
            # Calculate PnL if not provided
            if 'pnl' not in trade_data:
                # Simple calculation assuming long trades
                entry_price = trade_data.get('entry_price', 0)
                exit_price = trade_data.get('exit_price', 0)
                quantity = trade_data.get('quantity', 0)
                trade_data['pnl'] = (exit_price - entry_price) * quantity
            
            # Update win/loss statistics
            pnl = trade_data.get('pnl', 0)
            if pnl > 0:
                self._metrics['winning_trades'] += 1
            elif pnl < 0:
                self._metrics['losing_trades'] += 1
            
            # Recalculate performance metrics
            self._update_performance_metrics()
            
            # Log trade event
            self._log_event('trade', f"Trade completed: {trade_data['symbol']} with PnL: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            self._error_count += 1
    
    def record_error(self, error_data: Dict) -> None:
        """
        Record an error event.
        
        Parameters:
        -----------
        error_data : dict
            Error data including code, message, source.
            Can be either a raw dictionary or a dictionary created from a BaseError.
        """
        try:
            # Ensure we have a properly structured error dictionary
            if isinstance(error_data, BaseError):
                error_dict = error_data.to_dict()
            elif isinstance(error_data, Exception):
                error_dict = create_error_dict(error_data)
            else:
                # Add timestamp if not provided
                if 'timestamp' not in error_data:
                    error_data['timestamp'] = datetime.now().isoformat()
                
                # Ensure all required fields are present
                if 'message' not in error_data:
                    error_data['message'] = 'Unknown error'
                if 'error_code' not in error_data:
                    error_data['error_code'] = 'UNKNOWN_ERROR'
                
                error_dict = error_data
            
            # Store error data
            self._logs['errors'].append(error_dict)
            
            # Increment error count
            self._error_count += 1
            self._metrics['error_count'] += 1
            
            # Log error event
            self._log_event('error', f"Error: {error_dict.get('message', 'Unknown error')} "
                           f"[{error_dict.get('error_code', 'UNKNOWN_ERROR')}]")
            
            # Check if alert should be sent
            if self._error_count >= self.max_errors_before_alert:
                self._send_alert(
                    'error',
                    f"Multiple errors detected: {self._error_count} errors",
                    error_dict
                )
                self._error_count = 0  # Reset after alert
            
        except Exception as e:
            # Use our new logging function for consistent error reporting
            log_exception(e, logger)
    
    def record_warning(self, warning_data: Dict) -> None:
        """
        Record a warning event.
        
        Parameters:
        -----------
        warning_data : dict
            Warning data including message, source
        """
        try:
            # Add timestamp if not provided
            if 'timestamp' not in warning_data:
                warning_data['timestamp'] = datetime.now().isoformat()
            
            # Store warning data
            self._logs['warnings'].append(warning_data)
            
            # Log warning event
            self._log_event('warning', f"Warning: {warning_data.get('message', 'Unknown warning')}")
            
        except Exception as e:
            logger.error(f"Error recording warning event: {str(e)}")
    
    def update_account_values(self, account_values: Dict) -> None:
        """
        Update account values and check for alert conditions.
        
        Parameters:
        -----------
        account_values : dict
            Account values including equity, margin usage, etc.
        """
        try:
            # Extract key values
            current_equity = float(account_values.get('NetLiquidation_USD', 0))
            
            # Update start equity if not set
            if self._metrics['start_equity'] == 0.0:
                self._metrics['start_equity'] = current_equity
                self._metrics['high_water_mark'] = current_equity
            
            # Store current equity
            self._metrics['current_equity'] = current_equity
            
            # Update high water mark if higher
            if current_equity > self._metrics['high_water_mark']:
                self._metrics['high_water_mark'] = current_equity
            
            # Calculate drawdown
            if self._metrics['high_water_mark'] > 0:
                drawdown_pct = ((self._metrics['high_water_mark'] - current_equity) / 
                               self._metrics['high_water_mark']) * 100
                self._metrics['drawdown_pct'] = drawdown_pct
                
                # Check if drawdown exceeds alert threshold
                if drawdown_pct > self.alert_levels['drawdown_pct']:
                    self._send_alert(
                        'drawdown',
                        f"Drawdown alert: {drawdown_pct:.2f}% from high water mark",
                        {'current_equity': current_equity, 
                         'high_water_mark': self._metrics['high_water_mark'],
                         'drawdown_pct': drawdown_pct}
                    )
            
            # Calculate daily P&L
            daily_pnl = current_equity - self._metrics['start_equity']
            daily_pnl_pct = (daily_pnl / self._metrics['start_equity']) * 100 if self._metrics['start_equity'] > 0 else 0
            
            self._metrics['daily_pnl'] = daily_pnl
            self._metrics['daily_pnl_pct'] = daily_pnl_pct
            
            # Check if daily loss exceeds alert threshold
            if daily_pnl_pct < -self.alert_levels['daily_loss_pct']:
                self._send_alert(
                    'daily_loss',
                    f"Daily loss alert: {daily_pnl_pct:.2f}% (${daily_pnl:.2f})",
                    {'daily_pnl': daily_pnl, 'daily_pnl_pct': daily_pnl_pct}
                )
            
            # Save data periodically
            self._save_data()
            
        except Exception as e:
            logger.error(f"Error updating account values: {str(e)}")
    
    def generate_performance_report(self) -> Dict:
        """
        Generate a performance report with key metrics.
        
        Returns:
        --------
        dict
            Performance metrics
        """
        try:
            # Update performance metrics before generating report
            self._update_performance_metrics()
            
            # Create report dictionary
            report = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self._metrics,
                'system_status': {
                    'is_running': self._is_running,
                    'last_heartbeat': self._last_heartbeat_time.isoformat(),
                    'heartbeat_age_seconds': (datetime.now() - self._last_heartbeat_time).total_seconds(),
                    'error_count': self._error_count
                },
                'recent_trades': self._logs['trades'][-10:] if self._logs['trades'] else [],
                'recent_errors': self._logs['errors'][-5:] if self._logs['errors'] else [],
                'recent_alerts': self._alert_history[-5:] if self._alert_history else []
            }
            
            # Save report to file
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        logger.info("Starting monitor loop")
        
        # Initialize last check time
        self._last_check_time = datetime.now()
        
        while self._is_running:
            try:
                # Check alert conditions
                self._check_alert_conditions()
                
                # Check heartbeat
                now = datetime.now()
                if self._last_heartbeat_time is not None:
                    heartbeat_age = (now - self._last_heartbeat_time).total_seconds()
                    if heartbeat_age > self.heartbeat_timeout:
                        # Send heartbeat timeout alert
                        self._send_alert(
                            'heartbeat_timeout',
                            f"No heartbeat received for {heartbeat_age:.1f} seconds",
                            {'last_heartbeat': self._last_heartbeat_time.isoformat(),
                             'heartbeat_age': heartbeat_age}
                        )
                
                # Update performance metrics periodically
                self._update_performance_metrics()
                
                # Generate status report periodically (every 10 checks)
                now = datetime.now()
                if (now - self._last_check_time).total_seconds() > (self.check_interval * 10):
                    self._generate_status_report()
                    self._last_check_time = now
                
                # Save data periodically
                if (now - self._last_check_time).total_seconds() > (self.check_interval * 30):
                    self._save_data()
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                # Create a system error and log it
                error = SystemError(
                    message=f"Error in monitor loop: {str(e)}",
                    error_code="MONITOR_LOOP_ERROR",
                    source="TradingMonitor._monitor_loop",
                    cause=e
                )
                log_exception(error, logger)
                
                # Record the error
                self.record_error(error)
                
                # Continue with normal interval
                time.sleep(self.check_interval)
    
    @handle_exceptions(logger_obj=logger)
    def _send_alert(self, alert_type: str, message: str, data: Dict = None) -> None:
        """
        Send an alert via email and/or SMS.
        
        Parameters:
        -----------
        alert_type : str
            Type of alert (error, warning, heartbeat, etc.)
        message : str
            Alert message
        data : dict, optional
            Additional alert data
        """
        # Create alert record
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Add to alert history
        self._alert_history.append(alert)
        
        # Trim alert history (keep the last 100 entries)
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]
        
        # Log the alert
        logger.warning(f"Alert ({alert_type}): {message}")
        
        # Send email alert if enabled
        if self.enable_email_alerts:
            try:
                self._send_email_alert(alert)
            except Exception as e:
                # Create a system error and log it
                error = SystemError(
                    message=f"Failed to send email alert: {str(e)}",
                    error_code="EMAIL_ALERT_FAILURE",
                    source="TradingMonitor._send_alert",
                    cause=e
                )
                log_exception(error, logger)
        
        # Send SMS alert if enabled
        if self.enable_sms_alerts:
            try:
                self._send_sms_alert(alert)
            except Exception as e:
                # Create a system error and log it
                error = SystemError(
                    message=f"Failed to send SMS alert: {str(e)}",
                    error_code="SMS_ALERT_FAILURE",
                    source="TradingMonitor._send_alert",
                    cause=e
                )
                log_exception(error, logger)
    
    def _send_email_alert(self, alert: Dict) -> None:
        """
        Send an email alert.
        
        Parameters:
        -----------
        alert : dict
            Alert data
        """
        try:
            # Check if email settings are configured
            if not self.email_settings['smtp_server'] or not self.email_settings['sender'] or not self.email_settings['recipients']:
                logger.warning("Email settings not fully configured, skipping email alert")
                return
            
            # Create email subject
            subject = f"Trading Alert: {alert['type'].upper()} - {alert['message'][:50]}..."
            
            # Create email body
            body = f"""
            <html>
            <body>
                <h2>Trading System Alert</h2>
                <p><b>Type:</b> {alert['type']}</p>
                <p><b>Time:</b> {alert['timestamp']}</p>
                <p><b>Message:</b> {alert['message']}</p>
                
                <h3>Additional Data:</h3>
                <pre>{json.dumps(alert['data'], indent=2)}</pre>
                
                <h3>System Status:</h3>
                <p><b>Running:</b> {self._is_running}</p>
                <p><b>Equity:</b> ${self._metrics['current_equity']:.2f}</p>
                <p><b>Daily P&L:</b> ${self._metrics['daily_pnl']:.2f} ({self._metrics['daily_pnl_pct']:.2f}%)</p>
                <p><b>Drawdown:</b> {self._metrics['drawdown_pct']:.2f}%</p>
                <p><b>Error Count:</b> {self._metrics['error_count']}</p>
            </body>
            </html>
            """
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_settings['sender']
            msg['To'] = ', '.join(self.email_settings['recipients'])
            
            # Attach HTML body
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.email_settings['smtp_server'], self.email_settings['port'])
            server.starttls()
            
            # Login if username and password are provided
            if self.email_settings['username'] and self.email_settings['password']:
                server.login(self.email_settings['username'], self.email_settings['password'])
            
            # Send email
            server.sendmail(
                self.email_settings['sender'],
                self.email_settings['recipients'],
                msg.as_string()
            )
            
            # Close connection
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    def _send_sms_alert(self, alert: Dict) -> None:
        """
        Send an SMS alert.
        
        Parameters:
        -----------
        alert : dict
            Alert data
        """
        try:
            # Check if SMS settings are configured
            if self.sms_settings['provider'] != 'twilio' or not self.sms_settings['account_sid'] or not self.sms_settings['auth_token']:
                logger.warning("SMS settings not fully configured, skipping SMS alert")
                return
            
            # Check if Twilio is installed
            try:
                from twilio.rest import Client
            except ImportError:
                logger.error("Twilio package not installed, skipping SMS alert")
                return
            
            # Create message body
            body = f"ALERT: {alert['type'].upper()} - {alert['message']}"
            
            # Initialize Twilio client
            client = Client(self.sms_settings['account_sid'], self.sms_settings['auth_token'])
            
            # Send SMS to each recipient
            for to_number in self.sms_settings['to_numbers']:
                # Send message
                message = client.messages.create(
                    body=body,
                    from_=self.sms_settings['from_number'],
                    to=to_number
                )
                
                logger.info(f"SMS alert sent to {to_number}: {message.sid}")
            
        except Exception as e:
            logger.error(f"Error sending SMS alert: {str(e)}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on trade history."""
        try:
            # Calculate win rate
            total_trades = self._metrics['winning_trades'] + self._metrics['losing_trades']
            if total_trades > 0:
                self._metrics['win_rate'] = (self._metrics['winning_trades'] / total_trades) * 100
            
            # Calculate average win/loss
            if self._logs['trades']:
                wins = [t['pnl'] for t in self._logs['trades'] if t.get('pnl', 0) > 0]
                losses = [t['pnl'] for t in self._logs['trades'] if t.get('pnl', 0) < 0]
                
                self._metrics['avg_win'] = sum(wins) / len(wins) if wins else 0
                self._metrics['avg_loss'] = sum(losses) / len(losses) if losses else 0
                
                # Calculate profit factor
                total_wins = sum(wins)
                total_losses = abs(sum(losses)) if losses else 1  # Avoid division by zero
                self._metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified, using daily returns)
            if len(self._logs['trades']) > 10:
                # Group trades by day
                daily_pnls = {}
                for trade in self._logs['trades']:
                    date = trade.get('timestamp', '').split('T')[0]  # Extract date part
                    if date:
                        daily_pnls[date] = daily_pnls.get(date, 0) + trade.get('pnl', 0)
                
                # Calculate Sharpe
                if daily_pnls:
                    daily_returns = list(daily_pnls.values())
                    mean_return = np.mean(daily_returns)
                    std_return = np.std(daily_returns)
                    self._metrics['sharpe_ratio'] = mean_return / std_return if std_return > 0 else 0
            
            # Calculate max drawdown (from equity curve)
            # This is a simplified calculation, could be improved with a full equity curve
            if self._metrics['drawdown_pct'] > self._metrics.get('max_drawdown', 0):
                self._metrics['max_drawdown'] = self._metrics['drawdown_pct']
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _log_event(self, event_type: str, message: str) -> None:
        """
        Log an event.
        
        Parameters:
        -----------
        event_type : str
            Type of event
        message : str
            Event message
        """
        if event_type == 'error':
            logger.error(message)
        elif event_type == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def _generate_status_report(self) -> None:
        """Generate and save a status report."""
        try:
            # Generate performance report
            report = self.generate_performance_report()
            
            # Log status summary
            logger.info(f"Status Report: Equity=${report['metrics']['current_equity']:.2f}, "
                       f"P&L=${report['metrics']['daily_pnl']:+.2f} ({report['metrics']['daily_pnl_pct']:+.2f}%), "
                       f"Trades={report['metrics']['trade_count']}, "
                       f"Win Rate={report['metrics']['win_rate']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error generating status report: {str(e)}")
    
    def _save_report(self, report: Dict) -> None:
        """
        Save a report to disk.
        
        Parameters:
        -----------
        report : dict
            Report data
        """
        try:
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(self.data_directory, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            # Create report filename using timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(reports_dir, f'report_{timestamp}.json')
            
            # Save report to file
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also save as latest report
            latest_report_file = os.path.join(reports_dir, 'latest_report.json')
            with open(latest_report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.debug(f"Report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    def _save_data(self) -> None:
        """Save monitoring data to disk."""
        try:
            # Save metrics
            metrics_file = os.path.join(self.data_directory, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self._metrics, f, indent=2)
            
            # Save logs (limited number of entries to avoid huge files)
            logs_file = os.path.join(self.data_directory, 'logs.json')
            
            # Create a copy with limited entries
            logs_to_save = {
                'errors': self._logs['errors'][-100:],
                'warnings': self._logs['warnings'][-100:],
                'heartbeats': self._logs['heartbeats'][-20:],
                'trades': self._logs['trades'],  # Keep all trades
                'alerts': self._logs['alerts'][-50:]
            }
            
            with open(logs_file, 'w') as f:
                json.dump(logs_to_save, f, indent=2)
            
            # Save alert history
            alerts_file = os.path.join(self.data_directory, 'alerts.json')
            with open(alerts_file, 'w') as f:
                json.dump(self._alert_history[-100:], f, indent=2)
            
            logger.debug("Monitoring data saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving monitoring data: {str(e)}")
    
    def _load_data(self) -> None:
        """Load monitoring data from disk."""
        try:
            # Load metrics
            metrics_file = os.path.join(self.data_directory, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    loaded_metrics = json.load(f)
                    # Update metrics, preserving start time
                    saved_start_time = self._metrics['start_time']
                    self._metrics.update(loaded_metrics)
                    # Restore start time if this is a new session
                    if datetime.now() - datetime.fromisoformat(loaded_metrics['start_time']) > timedelta(hours=12):
                        self._metrics['start_time'] = saved_start_time
            
            # Load logs
            logs_file = os.path.join(self.data_directory, 'logs.json')
            if os.path.exists(logs_file):
                with open(logs_file, 'r') as f:
                    self._logs = json.load(f)
            
            # Load alert history
            alerts_file = os.path.join(self.data_directory, 'alerts.json')
            if os.path.exists(alerts_file):
                with open(alerts_file, 'r') as f:
                    self._alert_history = json.load(f)
            
            logger.info("Monitoring data loaded from disk")
            
        except Exception as e:
            logger.error(f"Error loading monitoring data: {str(e)}")
    
    def _check_alert_conditions(self) -> None:
        """Check if any alert conditions should be cleared."""
        pass  # Placeholder for future implementation

# Example usage:
# 
# monitor = TradingMonitor(
#     enable_email_alerts=True,
#     email_settings={
#         'smtp_server': 'smtp.gmail.com',
#         'port': 587,
#         'username': 'your-email@gmail.com',
#         'password': 'your-app-password',
#         'sender': 'your-email@gmail.com',
#         'recipients': ['alert-recipient@example.com']
#     }
# )
# 
# monitor.start()
# 
# # Record heartbeat
# monitor.record_heartbeat({
#     'is_running': True,
#     'connected': True
# })
# 
# # Record trade
# monitor.record_trade({
#     'symbol': 'ES',
#     'entry_price': 4000.50,
#     'exit_price': 4010.25,
#     'quantity': 1,
#     'entry_time': '2023-04-01T09:30:00',
#     'exit_time': '2023-04-01T11:45:00',
#     'pnl': 9.75
# })
# 
# # Generate report
# report = monitor.generate_performance_report()
# 
# # Stop when done
# monitor.stop() 