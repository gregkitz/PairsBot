"""
Live Trader for the Intraday Statistical Arbitrage System.

This module provides a live trading environment that connects to Interactive Brokers
for real market data and actual order execution.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid

from ..connectors.ib import IBConnector, contract_to_symbol, symbol_to_contract

# Configure logging
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live Trading implementation for the Intraday Statistical Arbitrage System.
    
    This class provides a live trading environment that connects to Interactive Brokers
    for real market data and order execution, allowing for actual trading of strategies.
    """
    
    def __init__(self,
                ib_host: str = '127.0.0.1',
                ib_port: int = 7496,  # 7496 for TWS Live
                ib_client_id: int = 1,
                account: Optional[str] = None,
                data_directory: Optional[str] = None,
                use_emergency_stop: bool = True,
                max_daily_loss_pct: float = 1.0,
                position_check_interval: int = 10,
                confirmation_required: bool = True,
                risk_level: str = 'low',
                heartbeat_interval: int = 30,
                auto_shutdown_time: Optional[str] = None,
                debug_mode: bool = False):
        """
        Initialize the live trading environment.
        
        Parameters:
        -----------
        ib_host : str
            Interactive Brokers TWS/Gateway hostname or IP address
        ib_port : int
            Interactive Brokers TWS/Gateway port (7496 for TWS Live)
        ib_client_id : int
            Interactive Brokers client ID
        account : str, optional
            IB account ID. If None, the first account will be used
        data_directory : str, optional
            Directory to store trading data (positions, orders, trades)
        use_emergency_stop : bool
            Whether to use emergency stop on excessive loss
        max_daily_loss_pct : float
            Maximum allowed daily loss as percentage of account equity
        position_check_interval : int
            Interval in seconds to check positions
        confirmation_required : bool
            Whether to require confirmation before executing orders
        risk_level : str
            Risk level ('low', 'medium', 'high') affecting position sizing
        heartbeat_interval : int
            Interval in seconds to send heartbeat signals
        auto_shutdown_time : str, optional
            Time to automatically shutdown trading (format: "HH:MM")
        debug_mode : bool
            Whether to run in debug mode with additional logging
        """
        # Set parameters
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self.account = account
        self.use_emergency_stop = use_emergency_stop
        self.max_daily_loss_pct = max_daily_loss_pct
        self.position_check_interval = position_check_interval
        self.confirmation_required = confirmation_required
        self.risk_level = risk_level
        self.heartbeat_interval = heartbeat_interval
        self.auto_shutdown_time = auto_shutdown_time
        self.debug_mode = debug_mode
        
        # Set data directory
        if data_directory is None:
            self.data_directory = os.path.join(os.getcwd(), 'live_trading_data')
        else:
            self.data_directory = data_directory
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Create IB connector for market data and execution
        self.ib_connector = IBConnector(
            host=ib_host,
            port=ib_port,
            client_id=ib_client_id,
            account=account,
            read_only=False,  # Live trading requires read_only=False
            auto_reconnect=True
        )
        
        # Initialize trading state
        self._is_running = False
        self._event_thread = None
        self._position_check_thread = None
        self._heartbeat_thread = None
        self._last_event_time = datetime.now()
        self._trading_paused = False
        self._emergency_stop_triggered = False
        self._pending_confirmation = {}
        
        # Initialize market data and orders
        self._market_data = {}
        self._pending_orders = {}
        self._executed_orders = {}
        self._positions = {}
        self._account_values = {}
        self._daily_pnl = 0.0
        self._start_equity = 0.0
        
        # Initialize event callbacks
        self._callbacks = {
            'order_status': [],
            'position_change': [],
            'account_update': [],
            'market_data': [],
            'trade': [],
            'error': [],
            'heartbeat': [],
            'emergency_stop': []
        }
        
        # Initialize risk metrics
        self._risk_metrics = {
            'max_position_size': 0.0,
            'max_daily_loss': 0.0,
            'current_risk_exposure': 0.0,
            'daily_high_water_mark': 0.0,
        }
        
        # Load existing data if available
        self._load_data()
        
        # Setup auto-shutdown if specified
        if self.auto_shutdown_time:
            self._setup_auto_shutdown()
            
        # Configure logging level
        if self.debug_mode:
            logging.getLogger('src.live_trading').setLevel(logging.DEBUG)
            logging.getLogger('src.connectors.ib').setLevel(logging.DEBUG)
    
    def start(self) -> bool:
        """
        Start the live trading environment.
        
        Returns:
        --------
        bool
            True if successfully started, False otherwise
        """
        if self._is_running:
            logger.warning("Live trader is already running")
            return True
        
        try:
            # Connect to IB
            if not self.ib_connector.is_connected():
                logger.info("Connecting to Interactive Brokers...")
                if not self.ib_connector.connect():
                    logger.error("Failed to connect to Interactive Brokers")
                    return False
            
            # Get initial account values
            account_info = self.ib_connector.get_account_info()
            if not account_info:
                logger.error("Failed to retrieve account information")
                return False
            
            # Store account values
            self._account_values = account_info
            self._start_equity = float(account_info.get('NetLiquidation_USD', 0))
            self._risk_metrics['max_daily_loss'] = self._start_equity * self.max_daily_loss_pct / 100
            self._risk_metrics['daily_high_water_mark'] = self._start_equity
            
            logger.info(f"Starting equity: ${self._start_equity:.2f}")
            
            # Set risk parameters based on risk level
            self._set_risk_parameters()
            
            # Verify account has enough buying power
            if float(account_info.get('AvailableFunds_USD', 0)) < self._start_equity * 0.1:
                logger.warning("Account has low available funds (less than 10% of equity)")
            
            # Start event processing thread
            self._is_running = True
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._event_thread.start()
            
            # Start position check thread
            self._position_check_thread = threading.Thread(target=self._position_check_loop, daemon=True)
            self._position_check_thread.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
            
            logger.info("Live trader started successfully")
            
            # Notify callbacks
            self._notify_callbacks('account_update', self._account_values)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting live trader: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the live trading environment.
        
        Returns:
        --------
        bool
            True if successfully stopped, False otherwise
        """
        if not self._is_running:
            logger.warning("Live trader is not running")
            return True
        
        try:
            logger.info("Stopping live trader...")
            
            # Stop event processing
            self._is_running = False
            
            # Wait for event thread to terminate
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=5.0)
            
            # Wait for position check thread to terminate
            if self._position_check_thread and self._position_check_thread.is_alive():
                self._position_check_thread.join(timeout=5.0)
            
            # Wait for heartbeat thread to terminate
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5.0)
            
            # Cancel all pending orders
            self._cancel_all_pending_orders()
            
            # Disconnect from IB
            if self.ib_connector.is_connected():
                self.ib_connector.disconnect()
            
            # Save data
            self._save_data()
            
            logger.info("Live trader stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping live trader: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """
        Check if the live trader is running.
        
        Returns:
        --------
        bool
            True if running, False otherwise
        """
        return self._is_running
    
    def pause_trading(self) -> bool:
        """
        Pause all trading activity but keep market data flowing.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._is_running:
            logger.warning("Live trader is not running")
            return False
        
        try:
            logger.info("Pausing trading activity...")
            self._trading_paused = True
            
            # Cancel all pending orders
            self._cancel_all_pending_orders()
            
            logger.info("Trading activity paused")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing trading activity: {str(e)}")
            return False
    
    def resume_trading(self) -> bool:
        """
        Resume trading activity after being paused.
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._is_running:
            logger.warning("Live trader is not running")
            return False
        
        if not self._trading_paused:
            logger.warning("Trading is not paused")
            return True
        
        try:
            logger.info("Resuming trading activity...")
            self._trading_paused = False
            logger.info("Trading activity resumed")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming trading activity: {str(e)}")
            return False
    
    def is_paused(self) -> bool:
        """
        Check if trading is paused.
        
        Returns:
        --------
        bool
            True if paused, False otherwise
        """
        return self._trading_paused
    
    def _cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders."""
        try:
            # Get all open orders
            open_orders = self.ib_connector.get_active_orders()
            
            if not open_orders:
                logger.info("No open orders to cancel")
                return
            
            logger.info(f"Cancelling {len(open_orders)} open orders...")
            
            # Cancel each order
            for order_id in open_orders:
                self.ib_connector.cancel_order(order_id)
                
                # Remove from pending orders
                if order_id in self._pending_orders:
                    del self._pending_orders[order_id]
            
            logger.info("All open orders cancelled")
            
        except Exception as e:
            logger.error(f"Error cancelling all pending orders: {str(e)}")
    
    def _set_risk_parameters(self) -> None:
        """Set risk parameters based on risk level."""
        # Calculate max position size based on risk level
        if self.risk_level == 'low':
            self._risk_metrics['max_position_size'] = self._start_equity * 0.02  # 2% max position size
        elif self.risk_level == 'medium':
            self._risk_metrics['max_position_size'] = self._start_equity * 0.05  # 5% max position size
        elif self.risk_level == 'high':
            self._risk_metrics['max_position_size'] = self._start_equity * 0.10  # 10% max position size
        else:
            logger.warning(f"Unknown risk level: {self.risk_level}, using 'low' as default")
            self._risk_metrics['max_position_size'] = self._start_equity * 0.02  # 2% max position size
        
        logger.info(f"Risk level: {self.risk_level}")
        logger.info(f"Max position size: ${self._risk_metrics['max_position_size']:.2f}")
        logger.info(f"Max daily loss: ${self._risk_metrics['max_daily_loss']:.2f}")
    
    def _load_data(self) -> None:
        """Load existing trading data from disk."""
        try:
            # Load positions
            positions_file = os.path.join(self.data_directory, 'positions.json')
            if os.path.exists(positions_file):
                with open(positions_file, 'r') as f:
                    self._positions = json.load(f)
                logger.info(f"Loaded {len(self._positions)} positions from disk")
            
            # Load executed orders
            orders_file = os.path.join(self.data_directory, 'orders.json')
            if os.path.exists(orders_file):
                with open(orders_file, 'r') as f:
                    self._executed_orders = json.load(f)
                logger.info(f"Loaded {len(self._executed_orders)} executed orders from disk")
                
        except Exception as e:
            logger.error(f"Error loading trading data: {str(e)}")
    
    def _save_data(self) -> None:
        """Save trading data to disk."""
        try:
            # Save positions
            positions_file = os.path.join(self.data_directory, 'positions.json')
            with open(positions_file, 'w') as f:
                json.dump(self._positions, f, indent=2)
            
            # Save executed orders
            orders_file = os.path.join(self.data_directory, 'orders.json')
            with open(orders_file, 'w') as f:
                json.dump(self._executed_orders, f, indent=2)
                
            # Save daily summary
            today = datetime.now().strftime('%Y-%m-%d')
            summary_file = os.path.join(self.data_directory, f'summary_{today}.json')
            summary = {
                'date': today,
                'start_equity': self._start_equity,
                'end_equity': float(self._account_values.get('NetLiquidation_USD', 0)),
                'daily_pnl': self._daily_pnl,
                'daily_pnl_pct': self._daily_pnl / self._start_equity if self._start_equity > 0 else 0,
                'position_count': len(self._positions),
                'trade_count': sum(1 for order in self._executed_orders.values() if order.get('execution_time', '').startswith(today)),
                'emergency_stop_triggered': self._emergency_stop_triggered
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.debug("Trading data saved to disk")
        
        except Exception as e:
            logger.error(f"Error saving trading data: {str(e)}")
    
    def _setup_auto_shutdown(self) -> None:
        """Setup auto-shutdown based on specified time."""
        if not self.auto_shutdown_time:
            return
        
        try:
            # Parse shutdown time
            hour, minute = map(int, self.auto_shutdown_time.split(':'))
            
            # Get current time
            now = datetime.now()
            
            # Set shutdown time for today
            shutdown_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If shutdown time is in the past, set it for tomorrow
            if shutdown_time < now:
                shutdown_time += timedelta(days=1)
            
            # Calculate seconds until shutdown
            seconds_until_shutdown = (shutdown_time - now).total_seconds()
            
            # Setup timer for auto-shutdown
            threading.Timer(seconds_until_shutdown, self.stop).start()
            
            logger.info(f"Auto-shutdown scheduled for {shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"Error setting up auto-shutdown: {str(e)}")
    
    def _event_loop(self) -> None:
        """Main event processing loop."""
        while self._is_running:
            try:
                # Process pending IB events
                self.ib_connector.process_pending_events()
                
                # Update account values periodically (every 60 seconds)
                now = datetime.now()
                if (now - self._last_event_time).total_seconds() > 60:
                    self._update_account_values()
                    self._check_emergency_stop()
                    self._last_event_time = now
                
                # Save data periodically (every 5 minutes)
                if (now - self._last_event_time).total_seconds() > 300:
                    self._save_data()
                
                # Sleep a bit to avoid high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in event loop: {str(e)}")
                # Continue the loop to maintain operation
    
    def _position_check_loop(self) -> None:
        """Periodic position check loop."""
        while self._is_running:
            try:
                # Skip if trading is paused
                if not self._trading_paused:
                    # Get current positions from IB
                    ib_positions = self.ib_connector.get_positions()
                    
                    # Update our internal positions
                    for symbol, position in ib_positions.items():
                        self._update_position(symbol, position)
                    
                    # Check for positions that exist in our records but not in IB
                    for symbol in list(self._positions.keys()):
                        if symbol not in ib_positions:
                            # Position has been closed
                            self._update_position(symbol, 0)
                
                # Sleep for the specified interval
                time.sleep(self.position_check_interval)
                
            except Exception as e:
                logger.error(f"Error in position check loop: {str(e)}")
                # Sleep and continue the loop
                time.sleep(5)
    
    def _heartbeat_loop(self) -> None:
        """Heartbeat loop to detect system health."""
        while self._is_running:
            try:
                # Emit heartbeat event
                self._notify_callbacks('heartbeat', {
                    'timestamp': datetime.now().isoformat(),
                    'is_running': self._is_running,
                    'is_paused': self._trading_paused,
                    'emergency_stop': self._emergency_stop_triggered,
                    'connected': self.ib_connector.is_connected()
                })
                
                # Check IB connection
                if not self.ib_connector.is_connected():
                    logger.warning("IB connection lost, attempting to reconnect...")
                    self.ib_connector.connect()
                
                # Sleep for the specified interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                # Sleep and continue the loop
                time.sleep(5)
    
    def place_order(self, 
                   symbol: str, 
                   quantity: int, 
                   order_type: str = 'MKT', 
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'DAY',
                   outside_rth: bool = False,
                   order_ref: Optional[str] = None) -> Optional[str]:
        """
        Place an order for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to trade (e.g., 'ES')
        quantity : int
            The quantity to trade (positive for buy, negative for sell)
        order_type : str
            The order type ('MKT', 'LMT', 'STP', 'STP LMT')
        limit_price : float, optional
            The limit price for limit orders
        stop_price : float, optional
            The stop price for stop orders
        time_in_force : str
            Time in force ('DAY', 'GTC', 'IOC', 'GTD')
        outside_rth : bool
            Whether the order can be executed outside regular trading hours
        order_ref : str, optional
            A reference string for the order
            
        Returns:
        --------
        str or None
            The order ID if successful, None otherwise
        """
        if not self._is_running:
            logger.error("Cannot place order: Live trader is not running")
            return None
            
        if self._trading_paused:
            logger.error("Cannot place order: Trading is paused")
            return None
            
        if self._emergency_stop_triggered:
            logger.error("Cannot place order: Emergency stop is triggered")
            return None
            
        # Validate order parameters
        if order_type in ['LMT', 'STP LMT'] and limit_price is None:
            logger.error("Limit price is required for limit orders")
            return None
            
        if order_type in ['STP', 'STP LMT'] and stop_price is None:
            logger.error("Stop price is required for stop orders")
            return None
            
        try:
            # Generate order reference if not provided
            if order_ref is None:
                order_ref = f"LIVETRADER_{uuid.uuid4().hex[:8]}"
                
            # Create contract
            contract = symbol_to_contract(symbol)
            
            # Create order
            order_id = self.ib_connector.place_order(
                contract=contract,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                outside_rth=outside_rth,
                order_ref=order_ref
            )
            
            if not order_id:
                logger.error(f"Failed to place order for {symbol}")
                return None
                
            # Store pending order
            self._pending_orders[order_id] = {
                'symbol': symbol,
                'quantity': quantity,
                'order_type': order_type,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'outside_rth': outside_rth,
                'order_ref': order_ref,
                'status': 'PENDING',
                'creation_time': datetime.now().isoformat(),
                'execution_time': None,
                'fill_price': None,
                'filled_quantity': 0
            }
            
            logger.info(f"Order placed: {order_id} for {symbol} - {quantity} @ {order_type}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return None
    
    def _update_position(self, symbol: str, position: float) -> None:
        """
        Update internal position tracking.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the position
        position : float
            The current position size
        """
        # Check if position exists
        if symbol in self._positions:
            # Update position
            old_position = self._positions[symbol]['position']
            self._positions[symbol]['position'] = position
            self._positions[symbol]['last_update'] = datetime.now().isoformat()
            
            # Notify position change if there's a difference
            if old_position != position:
                self._notify_callbacks('position_change', {
                    'symbol': symbol,
                    'old_position': old_position,
                    'new_position': position,
                    'timestamp': datetime.now().isoformat()
                })
        else:
            # Create new position entry
            self._positions[symbol] = {
                'symbol': symbol,
                'position': position,
                'created_at': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat()
            }
            
            # Notify position change
            self._notify_callbacks('position_change', {
                'symbol': symbol,
                'old_position': 0,
                'new_position': position,
                'timestamp': datetime.now().isoformat()
            })
        
        # Remove position if zero
        if position == 0 and symbol in self._positions:
            del self._positions[symbol]
    
    def _update_account_values(self) -> None:
        """Update account values and check daily P&L."""
        try:
            # Get account info from IB
            account_info = self.ib_connector.get_account_info()
            
            if not account_info:
                logger.warning("Failed to update account values")
                return
                
            # Store previous equity for comparison
            prev_equity = float(self._account_values.get('NetLiquidation_USD', self._start_equity))
            
            # Update account values
            self._account_values = account_info
            
            # Calculate current equity
            current_equity = float(account_info.get('NetLiquidation_USD', 0))
            
            # Update daily P&L
            self._daily_pnl = current_equity - self._start_equity
            
            # Update high water mark if current equity is higher
            if current_equity > self._risk_metrics['daily_high_water_mark']:
                self._risk_metrics['daily_high_water_mark'] = current_equity
            
            # Update current risk exposure
            self._risk_metrics['current_risk_exposure'] = abs(self._daily_pnl)
            
            # Log equity changes
            if abs(current_equity - prev_equity) > 0.01:
                logger.info(f"Account equity: ${current_equity:.2f} (${current_equity - prev_equity:+.2f})")
                logger.info(f"Daily P&L: ${self._daily_pnl:+.2f} ({self._daily_pnl / self._start_equity:+.2%})")
            
            # Notify account update
            self._notify_callbacks('account_update', self._account_values)
            
        except Exception as e:
            logger.error(f"Error updating account values: {str(e)}")
    
    def _check_emergency_stop(self) -> None:
        """Check if emergency stop should be triggered based on daily loss."""
        if not self.use_emergency_stop:
            return
            
        try:
            # Check if emergency stop already triggered
            if self._emergency_stop_triggered:
                return
                
            # Calculate current daily loss
            current_equity = float(self._account_values.get('NetLiquidation_USD', 0))
            daily_loss = self._start_equity - current_equity
            
            # Check if daily loss exceeds maximum allowed
            if daily_loss > self._risk_metrics['max_daily_loss']:
                logger.warning(f"EMERGENCY STOP TRIGGERED: Daily loss (${daily_loss:.2f}) exceeds maximum (${self._risk_metrics['max_daily_loss']:.2f})")
                
                # Set emergency stop flag
                self._emergency_stop_triggered = True
                
                # Pause trading
                self.pause_trading()
                
                # Cancel all pending orders
                self._cancel_all_pending_orders()
                
                # Notify emergency stop
                self._notify_callbacks('emergency_stop', {
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'daily_loss_exceeded',
                    'daily_loss': daily_loss,
                    'max_daily_loss': self._risk_metrics['max_daily_loss']
                })
                
        except Exception as e:
            logger.error(f"Error checking emergency stop: {str(e)}")
    
    def register_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Register a callback function for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            The event type to register for ('order_status', 'position_change', 
            'account_update', 'market_data', 'trade', 'error', 'heartbeat', 'emergency_stop')
        callback : callable
            The callback function to call when the event occurs
            
        Returns:
        --------
        bool
            True if registered successfully, False otherwise
        """
        if event_type not in self._callbacks:
            logger.error(f"Unknown event type: {event_type}")
            return False
            
        if not callable(callback):
            logger.error("Callback must be callable")
            return False
            
        self._callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for {event_type}")
        return True
    
    def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """
        Notify all registered callbacks for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            The event type to notify
        data : Any
            The data to pass to the callbacks
        """
        if event_type not in self._callbacks:
            return
            
        for callback in self._callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {str(e)}")
    
    def handle_market_data(self, symbol: str, data: Dict) -> None:
        """
        Handle market data updates for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol for the market data
        data : dict
            The market data
        """
        # Store market data
        self._market_data[symbol] = {
            **data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Notify market data update
        self._notify_callbacks('market_data', {
            'symbol': symbol,
            'data': self._market_data[symbol]
        })
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to get market data for
            
        Returns:
        --------
        dict or None
            The market data if available, None otherwise
        """
        return self._market_data.get(symbol)
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get all current positions.
        
        Returns:
        --------
        dict
            A dictionary of positions by symbol
        """
        return self._positions
    
    def get_pending_orders(self) -> Dict[str, Dict]:
        """
        Get all pending orders.
        
        Returns:
        --------
        dict
            A dictionary of pending orders by order ID
        """
        return self._pending_orders
    
    def get_executed_orders(self) -> Dict[str, Dict]:
        """
        Get all executed orders.
        
        Returns:
        --------
        dict
            A dictionary of executed orders by order ID
        """
        return self._executed_orders
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.
        
        Returns:
        --------
        dict
            A dictionary of risk metrics
        """
        return self._risk_metrics
    
    def get_account_values(self) -> Dict[str, str]:
        """
        Get current account values.
        
        Returns:
        --------
        dict
            A dictionary of account values
        """
        return self._account_values
    
    def subscribe_market_data(self, symbol: str) -> bool:
        """
        Subscribe to market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to subscribe to
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create contract
            contract = symbol_to_contract(symbol)
            
            # Subscribe to market data
            success = self.ib_connector.subscribe_market_data(contract)
            
            if success:
                logger.info(f"Subscribed to market data for {symbol}")
            else:
                logger.error(f"Failed to subscribe to market data for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error subscribing to market data for {symbol}: {str(e)}")
            return False
    
    def unsubscribe_market_data(self, symbol: str) -> bool:
        """
        Unsubscribe from market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            The symbol to unsubscribe from
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Create contract
            contract = symbol_to_contract(symbol)
            
            # Unsubscribe from market data
            success = self.ib_connector.unsubscribe_market_data(contract)
            
            if success:
                logger.info(f"Unsubscribed from market data for {symbol}")
            else:
                logger.error(f"Failed to unsubscribe from market data for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error unsubscribing from market data for {symbol}: {str(e)}")
            return False 