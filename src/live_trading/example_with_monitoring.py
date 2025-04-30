"""
Example script for using the LiveTrader with TradingMonitor.

This script demonstrates how to configure and use the LiveTrader class
with the TradingMonitor for monitoring, alerts, and performance tracking.

IMPORTANT: This script will connect to your Interactive Brokers account
and could potentially place real orders that involve real money. Use with caution
and only after thorough testing in a paper trading environment.
"""

import os
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from src.live_trading import LiveTrader
from src.live_trading.monitoring import TradingMonitor
from src.pairs_trading_strategy import PairsTradingStrategy
from src.connectors.ib import contract_to_symbol, symbol_to_contract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading_monitored.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MonitoredLiveTrader:
    """
    Combined LiveTrader and TradingMonitor for a full monitored live trading setup.
    """
    
    def __init__(self, 
                 ib_host: str = '127.0.0.1',
                 ib_port: int = 7496,
                 ib_client_id: int = 1,
                 account=None,
                 data_directory: str = './live_data',
                 enable_email_alerts: bool = False,
                 email_settings=None,
                 enable_sms_alerts: bool = False,
                 sms_settings=None,
                 max_daily_loss_pct: float = 1.0,
                 risk_level: str = 'low',
                 debug_mode: bool = False):
        """Initialize the monitored live trader."""
        # Create data directories
        os.makedirs(data_directory, exist_ok=True)
        trader_data_dir = os.path.join(data_directory, 'trader')
        monitor_data_dir = os.path.join(data_directory, 'monitor')
        
        # Create LiveTrader instance
        self.trader = LiveTrader(
            ib_host=ib_host,
            ib_port=ib_port,
            ib_client_id=ib_client_id,
            account=account,
            data_directory=trader_data_dir,
            use_emergency_stop=True,
            max_daily_loss_pct=max_daily_loss_pct,
            position_check_interval=10,
            confirmation_required=True,
            risk_level=risk_level,
            heartbeat_interval=30,
            auto_shutdown_time="16:00",  # 4 PM auto-shutdown
            debug_mode=debug_mode
        )
        
        # Create TradingMonitor instance
        self.monitor = TradingMonitor(
            data_directory=monitor_data_dir,
            enable_email_alerts=enable_email_alerts,
            enable_sms_alerts=enable_sms_alerts,
            email_settings=email_settings,
            sms_settings=sms_settings,
            check_interval=60,  # Check system every minute
            alert_levels={
                'daily_loss_pct': max_daily_loss_pct * 0.5,  # Alert at 50% of max loss
                'drawdown_pct': 2.0,  # Alert at 2% drawdown
                'error_count': 3,  # Alert after 3 errors
                'missed_heartbeats': 2  # Alert after 2 missed heartbeats
            },
            max_errors_before_alert=3,
            heartbeat_timeout=60,
            debug_mode=debug_mode
        )
        
        # Register event callbacks
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register LiveTrader event callbacks to monitor events."""
        self.trader.register_callback('order_status', self._on_order_status)
        self.trader.register_callback('position_change', self._on_position_change)
        self.trader.register_callback('account_update', self._on_account_update)
        self.trader.register_callback('market_data', self._on_market_data)
        self.trader.register_callback('trade', self._on_trade)
        self.trader.register_callback('error', self._on_error)
        self.trader.register_callback('heartbeat', self._on_heartbeat)
        self.trader.register_callback('emergency_stop', self._on_emergency_stop)
    
    def _on_order_status(self, data):
        """Handle order status events."""
        logger.info(f"Order {data['order_id']} status: {data.get('status', 'unknown')}")
    
    def _on_position_change(self, data):
        """Handle position change events."""
        logger.info(f"Position change: {data['symbol']} from {data['old_position']} to {data['new_position']}")
    
    def _on_account_update(self, data):
        """Handle account update events."""
        # Forward account update to monitor
        self.monitor.update_account_values(data)
        
        # Log equity update
        equity = data.get('NetLiquidation_USD', 'N/A')
        available = data.get('AvailableFunds_USD', 'N/A')
        logger.info(f"Account update: Equity=${equity}, Available=${available}")
    
    def _on_market_data(self, data):
        """Handle market data events."""
        # Process for monitoring if needed
        pass
    
    def _on_trade(self, data):
        """Handle completed trade events."""
        # Forward trade to monitor
        self.monitor.record_trade(data)
        
        # Log trade
        symbol = data.get('symbol', 'unknown')
        pnl = data.get('pnl', 0.0)
        logger.info(f"Trade completed: {symbol} with P&L: ${pnl:.2f}")
    
    def _on_error(self, data):
        """Handle error events."""
        # Forward error to monitor
        self.monitor.record_error(data)
        
        # Log error
        message = data.get('message', 'Unknown error')
        logger.error(f"Error: {message}")
    
    def _on_heartbeat(self, data):
        """Handle heartbeat events."""
        # Forward heartbeat to monitor
        self.monitor.record_heartbeat(data)
    
    def _on_emergency_stop(self, data):
        """Handle emergency stop events."""
        # Send high priority alert through monitor
        self.monitor._send_alert(
            'emergency_stop',
            f"EMERGENCY STOP TRIGGERED: {data.get('reason', 'unknown')}",
            data
        )
        
        # Log emergency stop
        logger.critical(f"EMERGENCY STOP: {data.get('reason', 'unknown')}")
    
    def start(self):
        """Start both the live trader and monitor."""
        # Start monitor first
        monitor_started = self.monitor.start()
        if not monitor_started:
            logger.error("Failed to start monitoring system")
            return False
        
        # Start trader
        trader_started = self.trader.start()
        if not trader_started:
            logger.error("Failed to start live trader")
            self.monitor.stop()
            return False
        
        logger.info("Monitored live trader started successfully")
        return True
    
    def stop(self):
        """Stop both the live trader and monitor."""
        # Stop trader first
        trader_stopped = self.trader.stop()
        
        # Stop monitor 
        monitor_stopped = self.monitor.stop()
        
        if trader_stopped and monitor_stopped:
            logger.info("Monitored live trader stopped successfully")
            return True
        else:
            logger.warning("Issues stopping monitored live trader")
            return False
    
    def is_running(self):
        """Check if both trader and monitor are running."""
        return self.trader.is_running() and self.monitor.is_running()
    
    def place_order(self, *args, **kwargs):
        """Place an order using the live trader."""
        return self.trader.place_order(*args, **kwargs)
    
    def generate_performance_report(self):
        """Generate a performance report from the monitor."""
        return self.monitor.generate_performance_report()


def main():
    """Main function to run the monitored live trading example."""
    
    # Configure email settings if desired (commented out by default)
    email_settings = {
        'smtp_server': 'smtp.gmail.com',
        'port': 587,
        'username': '',  # 'your-email@gmail.com'
        'password': '',  # 'your-app-password'
        'sender': '',    # 'your-email@gmail.com'
        'recipients': []  # ['alerts-recipient@example.com']
    }
    
    # Create the monitored live trader
    trader = MonitoredLiveTrader(
        ib_host='127.0.0.1',
        ib_port=7496,  # Use 7497 for paper trading
        ib_client_id=1,
        account=None,  # Use default account
        data_directory='./live_data',
        enable_email_alerts=False,  # Set to True to enable email alerts
        email_settings=email_settings,
        enable_sms_alerts=False,
        max_daily_loss_pct=1.0,  # 1% max daily loss
        risk_level='low',
        debug_mode=False
    )
    
    # Start the monitored trader
    if not trader.start():
        logger.error("Failed to start monitored trader")
        return
    
    try:
        # Define trading pairs
        pairs = [
            {'leg1': 'GC', 'leg2': 'SI'},  # Gold/Silver
            {'leg1': 'ES', 'leg2': 'NQ'}   # S&P 500/Nasdaq 100
        ]
        
        # Subscribe to market data for each symbol
        for pair in pairs:
            trader.trader.subscribe_market_data(pair['leg1'])
            trader.trader.subscribe_market_data(pair['leg2'])
        
        logger.info("Starting trading loop...")
        
        # Main trading loop
        while trader.is_running():
            # Generate performance report every hour
            if datetime.now().minute == 0:
                report = trader.generate_performance_report()
                logger.info(f"Hourly report: Equity=${report['metrics']['current_equity']:.2f}, "
                           f"P&L=${report['metrics']['daily_pnl']:+.2f} "
                           f"({report['metrics']['daily_pnl_pct']:+.2f}%)")
            
            # Example of how to place a market order
            # Uncomment the following lines to actually place orders
            # WARNING: This will place real orders if connected to a live account
            
            """
            # Example: Place a buy order for ES
            order_id = trader.place_order(
                symbol='ES',
                quantity=1,           # Buy 1 contract
                order_type='MKT',     # Market order
                time_in_force='DAY'   # Good for the day
            )
            
            if order_id:
                logger.info(f"Placed order {order_id} for ES")
            """
            
            # Sleep for 60 seconds before checking again
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Stop the monitored trader
        trader.stop()
        logger.info("Monitored live trader stopped")


if __name__ == "__main__":
    main() 