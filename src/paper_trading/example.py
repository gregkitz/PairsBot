"""
Example script for using the PaperTrader with Interactive Brokers.

This script demonstrates how to:
1. Connect to IB for market data
2. Set up a paper trading environment
3. Place orders and track positions
4. Print account and position updates
"""

import os
import time
import logging
import signal
import sys
from datetime import datetime

from src.paper_trading import PaperTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
paper_trader = None
keep_running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the paper trader."""
    global keep_running
    logger.info("Stopping paper trader...")
    keep_running = False


def on_account_update(account_data):
    """Handle account updates."""
    logger.info(f"Account update: Equity=${account_data['equity']:.2f}, "
               f"Cash=${account_data['cash']:.2f}, "
               f"Unrealized P&L=${account_data['unrealized_pnl']:.2f}")


def on_position_change(symbol, position_data):
    """Handle position changes."""
    if not position_data:
        logger.info(f"Position closed: {symbol}")
        return
    
    logger.info(f"Position update: {symbol} - Quantity: {position_data['quantity']}, "
               f"Avg Cost: ${position_data['avg_cost']:.2f}")


def on_order_status(order_data):
    """Handle order status updates."""
    logger.info(f"Order update: {order_data['symbol']} {order_data['action']} - "
               f"Status: {order_data['status']}, "
               f"Filled: {order_data['filled_quantity']}/{order_data['quantity']}")


def on_trade(trade_data):
    """Handle trade updates."""
    logger.info(f"Trade executed: {trade_data['symbol']} {trade_data['action']} "
               f"{trade_data['quantity']} @ ${trade_data['price']:.2f}")


def on_market_data(symbol, market_data):
    """Handle market data updates."""
    # Exclude ticker object which can't be easily formatted
    data = {k: v for k, v in market_data.items() if k != 'ticker'}
    logger.debug(f"Market data update: {symbol} - {data}")


def run_example():
    """Run the paper trading example."""
    global paper_trader
    
    # Set up signal handling for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting paper trading example...")
    
    try:
        # Create paper trader
        paper_trader = PaperTrader(
            initial_capital=100000.0,  # $100,000 starting capital
            ib_host='127.0.0.1',       # IB TWS/Gateway host
            ib_port=7497,              # IB TWS paper trading port (7497 for TWS paper)
            ib_client_id=1,            # IB client ID
            data_directory='paper_trading_data',  # Directory to store data
            commission_model='ibkr_pro',  # IBKR Pro commission model
            slippage_model='fixed',    # Fixed percentage slippage
            slippage_factor=0.0001,    # 0.01% slippage
            auto_shutdown_time=None    # No auto-shutdown
        )
        
        # Register callbacks
        paper_trader.add_callback('account_update', on_account_update)
        paper_trader.add_callback('position_change', on_position_change)
        paper_trader.add_callback('order_status', on_order_status)
        paper_trader.add_callback('trade', on_trade)
        paper_trader.add_callback('market_data', on_market_data)
        
        # Start paper trader
        if not paper_trader.start():
            logger.error("Failed to start paper trader")
            return
        
        # Subscribe to market data
        symbols = ['AAPL-STK-SMART', 'MSFT-STK-SMART', 'ES-202306-FUT-GLOBEX']
        for symbol in symbols:
            logger.info(f"Subscribing to market data for {symbol}")
            paper_trader.subscribe_market_data(symbol)
        
        # Let market data initialize
        logger.info("Waiting for market data to initialize...")
        time.sleep(5)
        
        # Place some example orders
        logger.info("Placing example orders...")
        
        # Buy 100 shares of AAPL at market
        aapl_order_id = paper_trader.place_order(
            symbol='AAPL-STK-SMART',
            action='BUY',
            quantity=100,
            order_type='MKT'
        )
        
        if aapl_order_id:
            logger.info(f"AAPL market buy order placed with ID: {aapl_order_id}")
        
        # Let order execute
        time.sleep(5)
        
        # Place a limit order for MSFT
        msft_price = paper_trader.get_price('MSFT-STK-SMART')
        if msft_price:
            # Limit buy 10% below current price
            limit_price = msft_price * 0.9
            
            msft_order_id = paper_trader.place_order(
                symbol='MSFT-STK-SMART',
                action='BUY',
                quantity=50,
                order_type='LMT',
                limit_price=limit_price
            )
            
            if msft_order_id:
                logger.info(f"MSFT limit buy order placed with ID: {msft_order_id}")
        
        # Place a bracket order for ES futures
        es_price = paper_trader.get_price('ES-202306-FUT-GLOBEX')
        if es_price:
            # Bracket order: Entry at market, profit target 1% above, stop loss 0.5% below
            bracket_order = paper_trader.place_bracket_order(
                symbol='ES-202306-FUT-GLOBEX',
                action='BUY',
                quantity=1,
                entry_order_type='MKT',
                profit_price=es_price * 1.01,
                stop_price=es_price * 0.995
            )
            
            if bracket_order['entry']:
                logger.info(f"ES bracket order placed: Entry ID {bracket_order['entry']}, "
                           f"Profit ID {bracket_order['profit']}, Stop ID {bracket_order['stop']}")
        
        # Keep the script running
        logger.info("Paper trading example running. Press Ctrl+C to exit...")
        
        while keep_running:
            # Check for completed limit order
            if 'msft_order_id' in locals() and paper_trader.get_order(msft_order_id)['status'] != 'FILLED':
                # Cancel the MSFT limit order after 30 seconds
                if (datetime.now() - paper_trader.get_order(msft_order_id)['creation_time']).total_seconds() > 30:
                    logger.info(f"Canceling MSFT limit order {msft_order_id}")
                    paper_trader.cancel_order(msft_order_id)
            
            # Wait for user to interrupt
            time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error in paper trading example: {str(e)}")
    
    finally:
        # Stop paper trader
        if paper_trader and paper_trader.is_running():
            paper_trader.stop()
        
        logger.info("Paper trading example finished")


if __name__ == "__main__":
    run_example() 