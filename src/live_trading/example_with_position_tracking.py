"""
Example script for using the LiveTrader with PositionTracker.

This script demonstrates how to configure and use the LiveTrader class
with the PositionTracker for position management, P&L calculation, and
trade tracking.

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

from src.live_trading import LiveTrader, PositionTracker
from src.pairs_trading_strategy import PairsTradingStrategy
from src.connectors.ib import contract_to_symbol, symbol_to_contract

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading_tracked.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TrackedLiveTrader:
    """
    LiveTrader extended with PositionTracker for comprehensive position management.
    """
    
    def __init__(self, 
                 ib_host: str = '127.0.0.1',
                 ib_port: int = 7496,
                 ib_client_id: int = 1,
                 account=None,
                 data_directory: str = './live_data',
                 use_emergency_stop: bool = True,
                 max_daily_loss_pct: float = 1.0,
                 risk_level: str = 'low',
                 commission_model=None,
                 slippage_model=None,
                 debug_mode: bool = False):
        """Initialize the tracked live trader."""
        # Create data directories
        os.makedirs(data_directory, exist_ok=True)
        trader_data_dir = os.path.join(data_directory, 'trader')
        position_data_dir = os.path.join(data_directory, 'positions')
        
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
        
        # Create PositionTracker instance
        self.tracker = PositionTracker(
            data_directory=position_data_dir,
            commission_model=commission_model,
            slippage_model=slippage_model,
            debug_mode=debug_mode
        )
        
        # Register event callbacks
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register LiveTrader event callbacks to track positions and trades."""
        self.trader.register_callback('order_status', self._on_order_status)
        self.trader.register_callback('position_change', self._on_position_change)
        self.trader.register_callback('account_update', self._on_account_update)
        self.trader.register_callback('trade', self._on_trade)
        self.trader.register_callback('error', self._on_error)
    
    def _on_order_status(self, data):
        """Handle order status events."""
        order_id = data.get('order_id', 'unknown')
        status = data.get('status', 'unknown')
        symbol = data.get('symbol', 'unknown')
        
        logger.info(f"Order {order_id} status: {status} for {symbol}")
        
        # Handle filled orders
        if status == 'FILLED':
            # Check if this is an entry or exit order
            # This would require more context in a real implementation
            pass
    
    def _on_position_change(self, data):
        """Handle position change events."""
        symbol = data.get('symbol', 'unknown')
        old_pos = data.get('old_position', 0)
        new_pos = data.get('new_position', 0)
        
        logger.info(f"Position change: {symbol} from {old_pos} to {new_pos}")
        
        # Handle new positions
        if old_pos == 0 and new_pos != 0:
            # This is a new position
            # In real implementation, we'd need price data which should come from trade events
            pass
        
        # Handle closed positions
        elif old_pos != 0 and new_pos == 0:
            # Position was closed
            pass
    
    def _on_account_update(self, data):
        """Handle account update events."""
        equity = data.get('NetLiquidation_USD', 'N/A')
        available = data.get('AvailableFunds_USD', 'N/A')
        
        logger.info(f"Account update: Equity=${equity}, Available=${available}")
    
    def _on_trade(self, data):
        """Handle completed trade events."""
        symbol = data.get('symbol', 'unknown')
        quantity = data.get('quantity', 0)
        price = data.get('price', 0.0)
        trade_time = data.get('time', datetime.now().isoformat())
        order_id = data.get('order_id', None)
        trade_type = data.get('trade_type', 'unknown')  # 'entry' or 'exit'
        
        logger.info(f"Trade: {symbol} {quantity} @ {price}, Type: {trade_type}")
        
        # Handle trades in position tracker
        if trade_type == 'entry':
            # Create a new position in the tracker
            position_id = self.tracker.create_position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=trade_time,
                order_id=order_id
            )
            logger.info(f"Created position {position_id} in tracker")
            
        elif trade_type == 'exit':
            # Find corresponding position(s) to close
            positions = self.tracker.get_position_by_symbol(symbol)
            active_positions = [p for p in positions if p['status'] == 'open']
            
            if active_positions:
                # Close each position separately (in real implementation, we might be more selective)
                for position in active_positions:
                    self.tracker.close_position(
                        position_id=position['position_id'],
                        exit_price=price,
                        exit_time=trade_time,
                        exit_order_id=order_id
                    )
                    logger.info(f"Closed position {position['position_id']} in tracker")
    
    def _on_error(self, data):
        """Handle error events."""
        message = data.get('message', 'Unknown error')
        logger.error(f"Error: {message}")
    
    def start(self):
        """Start the tracked live trader."""
        # Start trader
        trader_started = self.trader.start()
        if not trader_started:
            logger.error("Failed to start live trader")
            return False
        
        logger.info("Tracked live trader started successfully")
        return True
    
    def stop(self):
        """Stop the tracked live trader."""
        # Stop trader
        trader_stopped = self.trader.stop()
        
        if trader_stopped:
            logger.info("Tracked live trader stopped successfully")
            return True
        else:
            logger.warning("Issues stopping tracked live trader")
            return False
    
    def is_running(self):
        """Check if trader is running."""
        return self.trader.is_running()
    
    def place_order(self, *args, **kwargs):
        """Place an order using the live trader."""
        return self.trader.place_order(*args, **kwargs)
    
    def get_performance_summary(self, start_date=None, end_date=None):
        """Get performance summary from the position tracker."""
        return self.tracker.get_performance_summary(start_date, end_date)
    
    def get_positions(self):
        """Get all active positions from both IB and the tracker."""
        ib_positions = self.trader.get_positions()
        tracker_positions = self.tracker.get_active_positions()
        
        # In a full implementation, we'd reconcile these positions
        return {
            'ib_positions': ib_positions,
            'tracker_positions': tracker_positions
        }
    
    def create_pair_position(self,
                            pair_id: str,
                            leg1_symbol: str,
                            leg1_quantity: float,
                            leg1_price: float,
                            leg2_symbol: str,
                            leg2_quantity: float,
                            leg2_price: float):
        """Create a pair position using the position tracker and execute orders."""
        # First, create the pair in the tracker
        self.tracker.create_pair_position(
            pair_id=pair_id,
            leg1_symbol=leg1_symbol,
            leg1_quantity=leg1_quantity,
            leg1_price=leg1_price,
            leg2_symbol=leg2_symbol,
            leg2_quantity=leg2_quantity,
            leg2_price=leg2_price,
            entry_time=datetime.now(),
            metadata={'strategy': 'pairs_trading'}
        )
        
        # Then place the orders with the trader
        # Note: In real implementation, we'd handle the order IDs and status callbacks
        leg1_order_id = self.trader.place_order(
            symbol=leg1_symbol,
            quantity=leg1_quantity,
            order_type='MKT'
        )
        
        leg2_order_id = self.trader.place_order(
            symbol=leg2_symbol,
            quantity=leg2_quantity,
            order_type='MKT'
        )
        
        logger.info(f"Created pair position {pair_id} and placed orders for both legs")
        
        return {
            'pair_id': pair_id,
            'leg1_order_id': leg1_order_id,
            'leg2_order_id': leg2_order_id
        }
    
    def close_pair_position(self, 
                          pair_id: str,
                          leg1_exit_price: float,
                          leg2_exit_price: float):
        """Close a pair position using the position tracker and execute orders."""
        # First, get the pair position from the tracker
        pair = self.tracker.get_pair_position(pair_id)
        
        if not pair:
            logger.warning(f"Pair position {pair_id} not found")
            return None
        
        # Get the individual positions in the pair
        leg_positions = []
        for position_id in pair['positions']:
            pos = self.tracker.get_position(position_id)
            if pos:
                leg_positions.append(pos)
        
        if len(leg_positions) != 2:
            logger.warning(f"Pair {pair_id} doesn't have exactly 2 legs")
            return None
        
        # Close the pair in the tracker
        self.tracker.close_pair_position(
            pair_id=pair_id,
            leg1_exit_price=leg1_exit_price,
            leg2_exit_price=leg2_exit_price,
            exit_time=datetime.now()
        )
        
        # Place orders to close the positions
        leg1_close_order_id = None
        leg2_close_order_id = None
        
        for leg in leg_positions:
            # Close with opposite position
            if leg['metadata'].get('leg') == 'leg1':
                leg1_close_order_id = self.trader.place_order(
                    symbol=leg['symbol'],
                    quantity=-leg['quantity'],  # Opposite sign to close
                    order_type='MKT'
                )
            elif leg['metadata'].get('leg') == 'leg2':
                leg2_close_order_id = self.trader.place_order(
                    symbol=leg['symbol'],
                    quantity=-leg['quantity'],  # Opposite sign to close
                    order_type='MKT'
                )
        
        logger.info(f"Closed pair position {pair_id} and placed orders to close both legs")
        
        return {
            'pair_id': pair_id,
            'leg1_close_order_id': leg1_close_order_id,
            'leg2_close_order_id': leg2_close_order_id
        }


def main():
    """Main function to run the tracked live trading example."""
    
    # Define commission model
    commission_model = {
        'per_contract': 0.85,  # $0.85 per contract
        'per_order': 0.0,      # $0.00 per order
        'percent': 0.0,        # 0% of trade value
        'minimum': 0.0         # $0.00 minimum commission
    }
    
    # Define slippage model
    slippage_model = {
        'fixed_ticks': 1,      # 1 tick of slippage
        'percent': 0.0,        # 0% of price
        'market_impact': 0.0   # 0% market impact factor
    }
    
    # Create the tracked live trader
    trader = TrackedLiveTrader(
        ib_host='127.0.0.1',
        ib_port=7496,  # Use 7497 for paper trading
        ib_client_id=1,
        account=None,  # Use default account
        data_directory='./live_data',
        use_emergency_stop=True,
        max_daily_loss_pct=1.0,  # 1% max daily loss
        risk_level='low',
        commission_model=commission_model,
        slippage_model=slippage_model,
        debug_mode=False
    )
    
    # Start the trader
    if not trader.start():
        logger.error("Failed to start trader")
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
            # Example of how to create a pair position
            # Uncomment the following lines to actually create positions
            # WARNING: This will place real orders if connected to a live account
            
            """
            # Example: Create a pair position
            pair_result = trader.create_pair_position(
                pair_id=f"PAIR_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                leg1_symbol='GC',
                leg1_quantity=1,
                leg1_price=1800.50,  # Estimated price, will use market price
                leg2_symbol='SI',
                leg2_quantity=-10,   # Short 10 contracts of silver
                leg2_price=22.75     # Estimated price, will use market price
            )
            
            # Store the pair ID for later reference
            if pair_result:
                current_pair_id = pair_result['pair_id']
            """
            
            # Example of how to close a pair position
            # Uncomment the following lines to actually close positions
            # WARNING: This will place real orders if connected to a live account
            
            """
            # Example: Close a pair position
            if 'current_pair_id' in locals():
                close_result = trader.close_pair_position(
                    pair_id=current_pair_id,
                    leg1_exit_price=1810.25,  # Estimated price, will use market price
                    leg2_exit_price=22.50     # Estimated price, will use market price
                )
                
                # Remove the pair ID reference
                if close_result:
                    del current_pair_id
            """
            
            # Generate performance summary every hour
            if datetime.now().minute == 0:
                summary = trader.get_performance_summary()
                logger.info(f"Performance Summary: "
                          f"Trades={summary['total_trades']}, "
                          f"Win Rate={summary['win_rate']:.1f}%, "
                          f"P&L=${summary['total_pnl']:+.2f}, "
                          f"Net P&L=${summary['total_net_pnl']:+.2f}")
            
            # Sleep for 60 seconds before checking again
            time.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, shutting down...")
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
    finally:
        # Stop the trader
        trader.stop()
        logger.info("Tracked live trader stopped")


if __name__ == "__main__":
    main() 