"""
Example script for using the LiveTrader with Interactive Brokers.

This script demonstrates how to configure and use the LiveTrader class
for executing trades with real money through the Interactive Brokers API.

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
from src.pairs_trading_strategy import PairsTradingStrategy
from src.connectors.ib import contract_to_symbol, symbol_to_contract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Callback functions for handling events
def on_position_change(data):
    """Handle position change events"""
    logger.info(f"Position change: {data['symbol']} from {data['old_position']} to {data['new_position']}")

def on_account_update(data):
    """Handle account update events"""
    equity = data.get('NetLiquidation_USD', 'N/A')
    available = data.get('AvailableFunds_USD', 'N/A')
    logger.info(f"Account update: Equity=${equity}, Available=${available}")

def on_market_data(data):
    """Handle market data events"""
    symbol = data['symbol']
    price = data['data'].get('last', 'N/A')
    logger.debug(f"Market data: {symbol} last price: {price}")

def on_order_status(data):
    """Handle order status events"""
    logger.info(f"Order status: {data['order_id']} - {data['status']}")
    
def on_emergency_stop(data):
    """Handle emergency stop events"""
    logger.warning(f"EMERGENCY STOP TRIGGERED: {data['reason']} - Loss: ${data['daily_loss']:.2f}")

def on_error(data):
    """Handle error events"""
    logger.error(f"Error: {data['message']}")

def on_heartbeat(data):
    """Handle heartbeat events"""
    logger.debug(f"Heartbeat: system running={data['is_running']}, connected={data['connected']}")

def main():
    """Main function to run the live trading example"""
    
    # Create the live trader instance
    live_trader = LiveTrader(
        ib_host='127.0.0.1',            # Local IB Gateway/TWS
        ib_port=7496,                   # TWS port (7496 for live, 7497 for paper)
        ib_client_id=1,                 # Client ID
        account=None,                   # Use default account
        data_directory='./live_data',   # Directory to save trading data
        use_emergency_stop=True,        # Enable emergency stop
        max_daily_loss_pct=1.0,         # 1% max daily loss
        position_check_interval=10,     # Check positions every 10 seconds
        confirmation_required=True,     # Require confirmation for orders
        risk_level='low',               # Low risk level
        heartbeat_interval=30,          # 30-second heartbeat
        auto_shutdown_time="16:00",     # Automatically shutdown at 4 PM
        debug_mode=False                # Disable debug mode
    )
    
    # Register callbacks for events
    live_trader.register_callback('position_change', on_position_change)
    live_trader.register_callback('account_update', on_account_update)
    live_trader.register_callback('market_data', on_market_data)
    live_trader.register_callback('order_status', on_order_status)
    live_trader.register_callback('emergency_stop', on_emergency_stop)
    live_trader.register_callback('error', on_error)
    live_trader.register_callback('heartbeat', on_heartbeat)
    
    # Start the live trader
    if not live_trader.start():
        logger.error("Failed to start live trader")
        return
    
    try:
        # Define trading pairs
        pairs = [
            {'leg1': 'GC', 'leg2': 'SI'},  # Gold/Silver
            {'leg1': 'ES', 'leg2': 'NQ'}   # S&P 500/Nasdaq 100
        ]
        
        # Subscribe to market data for each symbol
        for pair in pairs:
            live_trader.subscribe_market_data(pair['leg1'])
            live_trader.subscribe_market_data(pair['leg2'])
        
        logger.info("Starting trading loop...")
        
        # Main trading loop
        while live_trader.is_running():
            # Check for trading opportunities
            for pair in pairs:
                # Get market data for each leg
                leg1_data = live_trader.get_market_data(pair['leg1'])
                leg2_data = live_trader.get_market_data(pair['leg2'])
                
                # Skip if we don't have data for both legs
                if not leg1_data or not leg2_data:
                    continue
                
                # Example of how to place a market order
                # Uncomment the following lines to actually place orders
                # WARNING: This will place real orders if connected to a live account
                
                """
                # Example: Place a buy order for leg1
                order_id = live_trader.place_order(
                    symbol=pair['leg1'],
                    quantity=1,           # Buy 1 contract
                    order_type='MKT',     # Market order
                    time_in_force='DAY'   # Good for the day
                )
                
                if order_id:
                    logger.info(f"Placed order {order_id} for {pair['leg1']}")
                """
                
                # Get current positions
                positions = live_trader.get_positions()
                
                # Log current positions
                for symbol, position in positions.items():
                    logger.info(f"Current position: {symbol} - {position['position']}")
            
            # Check account values
            account_values = live_trader.get_account_values()
            
            # Sleep for 60 seconds before checking again
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Stop the live trader
        live_trader.stop()
        logger.info("Live trader stopped")

if __name__ == "__main__":
    main() 