"""
Paper Trader for the Intraday Statistical Arbitrage System.

This module provides a paper trading environment that uses real market data
from Interactive Brokers but simulates order execution and position management.
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

# Use relative import for package structure
from ..connectors.ib import IBConnector, contract_to_symbol, symbol_to_contract
from src.utils.error_handling import (
    BaseError, TradingError, OrderExecutionError, handle_exceptions, log_exception
)

# Configure logging
logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper Trading implementation for the Intraday Statistical Arbitrage System.
    
    This class provides a paper trading environment that mimics real trading
    but with simulated order execution, allowing for strategy testing without
    risking real capital.
    """
    
    def __init__(self,
                initial_capital: float = 100000.0,
                ib_host: str = '127.0.0.1',
                ib_port: int = 7497,
                ib_client_id: int = 1,
                data_directory: Optional[str] = None,
                commission_model: str = 'ibkr_pro',
                slippage_model: str = 'fixed',
                slippage_factor: float = 0.0001,
                latency_model: str = 'fixed',
                latency_ms: int = 100,
                auto_shutdown_time: Optional[str] = None):
        """
        Initialize the paper trading environment.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital in the paper trading account
        ib_host : str
            Interactive Brokers TWS/Gateway hostname or IP address
        ib_port : int
            Interactive Brokers TWS/Gateway port
        ib_client_id : int
            Interactive Brokers client ID
        data_directory : str, optional
            Directory to store paper trading data (positions, orders, trades)
        commission_model : str
            Commission model to use ('ibkr_pro', 'ibkr_lite', 'flat', 'none')
        slippage_model : str
            Slippage model to use ('fixed', 'variable', 'none')
        slippage_factor : float
            Slippage factor (percentage for fixed, std dev multiplier for variable)
        latency_model : str
            Latency model to use ('fixed', 'variable', 'none')
        latency_ms : int
            Latency in milliseconds for fixed latency model
        auto_shutdown_time : str, optional
            Time to automatically shutdown paper trading (format: "HH:MM")
        """
        # Set parameters
        self.initial_capital = initial_capital
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.slippage_factor = slippage_factor
        self.latency_model = latency_model
        self.latency_ms = latency_ms
        self.auto_shutdown_time = auto_shutdown_time
        
        # Set data directory
        if data_directory is None:
            self.data_directory = os.path.join(os.getcwd(), 'paper_trading_data')
        else:
            self.data_directory = data_directory
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Create IB connector for market data
        self.ib_connector = IBConnector(
            host=ib_host,
            port=ib_port,
            client_id=ib_client_id,
            read_only=True,  # We'll only use IB for market data, not execution
            auto_reconnect=True
        )
        
        # Initialize account data
        self._account = {
            'cash': initial_capital,
            'equity': initial_capital,
            'margin_used': 0.0,
            'pnl_day': 0.0,
            'pnl_total': 0.0,
            'unrealized_pnl': 0.0,
            'initial_capital': initial_capital,
            'starting_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Initialize positions, orders, and trades
        self._positions = {}  # Symbol -> Position
        self._orders = {}     # Order ID -> Order
        self._trades = []     # List of executed trades
        
        # Initialize market data subscriptions
        self._market_data = {}  # Symbol -> Market Data
        
        # Initialize event processing
        self._is_running = False
        self._event_thread = None
        self._last_event_time = datetime.now()
        
        # Initialize event callbacks
        self._callbacks = {
            'order_status': [],
            'position_change': [],
            'account_update': [],
            'market_data': [],
            'trade': [],
            'error': []
        }
        
        # Load existing data if available
        self._load_data()
        
        # Setup auto-shutdown if specified
        if self.auto_shutdown_time:
            self._setup_auto_shutdown() 

    def _load_data(self) -> None:
        """Load existing paper trading data from disk."""
        try:
            # Load account data
            account_file = os.path.join(self.data_directory, 'account.json')
            if os.path.exists(account_file):
                with open(account_file, 'r') as f:
                    self._account = json.load(f)
                logger.info(f"Loaded account data: Capital=${self._account['equity']:.2f}")
            
            # Load positions
            positions_file = os.path.join(self.data_directory, 'positions.json')
            if os.path.exists(positions_file):
                with open(positions_file, 'r') as f:
                    self._positions = json.load(f)
                logger.info(f"Loaded {len(self._positions)} positions")
            
            # Load trades
            trades_file = os.path.join(self.data_directory, 'trades.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self._trades = json.load(f)
                logger.info(f"Loaded {len(self._trades)} historical trades")
            
        except Exception as e:
            logger.error(f"Error loading paper trading data: {str(e)}")
    
    def _save_data(self) -> None:
        """Save paper trading data to disk."""
        try:
            # Save account data
            account_file = os.path.join(self.data_directory, 'account.json')
            with open(account_file, 'w') as f:
                json.dump(self._account, f, indent=2)
            
            # Save positions
            positions_file = os.path.join(self.data_directory, 'positions.json')
            with open(positions_file, 'w') as f:
                json.dump(self._positions, f, indent=2)
            
            # Save trades
            trades_file = os.path.join(self.data_directory, 'trades.json')
            with open(trades_file, 'w') as f:
                json.dump(self._trades, f, indent=2)
            
            logger.debug("Paper trading data saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving paper trading data: {str(e)}")
    
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
    
    def start(self) -> bool:
        """
        Start the paper trading environment.
        
        Returns:
        --------
        bool
            True if successfully started, False otherwise
        """
        if self._is_running:
            logger.warning("Paper trader is already running")
            return True
        
        try:
            # Connect to IB
            if not self.ib_connector.is_connected():
                if not self.ib_connector.connect():
                    logger.error("Failed to connect to IB")
                    return False
            
            # Start event processing thread
            self._is_running = True
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            self._event_thread.start()
            
            logger.info("Paper trader started")
            
            # Notify account update callbacks
            self._notify_callbacks('account_update', self._account)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting paper trader: {str(e)}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the paper trading environment.
        
        Returns:
        --------
        bool
            True if successfully stopped, False otherwise
        """
        if not self._is_running:
            logger.warning("Paper trader is not running")
            return True
        
        try:
            # Stop event processing
            self._is_running = False
            
            # Wait for event thread to terminate
            if self._event_thread and self._event_thread.is_alive():
                self._event_thread.join(timeout=5.0)
            
            # Disconnect from IB
            if self.ib_connector.is_connected():
                self.ib_connector.disconnect()
            
            # Save data
            self._save_data()
            
            logger.info("Paper trader stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping paper trader: {str(e)}")
            return False
    
    def is_running(self) -> bool:
        """
        Check if the paper trader is running.
        
        Returns:
        --------
        bool
            True if running, False otherwise
        """
        return self._is_running
    
    def _event_loop(self) -> None:
        """Main event processing loop."""
        while self._is_running:
            try:
                # Process pending IB events
                self.ib_connector.process_pending_events()
                
                # Process paper orders
                self._process_orders()
                
                # Update account values
                self._update_account()
                
                # Save data periodically (every 5 minutes)
                now = datetime.now()
                if (now - self._last_event_time).total_seconds() > 300:
                    self._save_data()
                    self._last_event_time = now
                
                # Sleep a bit to avoid high CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in event loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error 

    def _notify_callbacks(self, event_type: str, *args, **kwargs) -> None:
        """
        Notify callbacks for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type
        *args, **kwargs
            Arguments to pass to callbacks
        """
        if event_type not in self._callbacks:
            return
        
        for callback in self._callbacks[event_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {str(e)}")
    
    def add_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Add a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (order_status, position_change, account_update, market_data, trade, error)
        callback : Callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self._callbacks:
            logger.error(f"Invalid event type: {event_type}")
            return False
        
        self._callbacks[event_type].append(callback)
        return True
    
    def remove_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Remove a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (order_status, position_change, account_update, market_data, trade, error)
        callback : Callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self._callbacks:
            logger.error(f"Invalid event type: {event_type}")
            return False
        
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            return True
        
        return False
    
    # -------------------------------------------------------------------------
    # Market Data Methods
    # -------------------------------------------------------------------------
    
    def subscribe_market_data(self, symbol: str) -> bool:
        """
        Subscribe to real-time market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to subscribe to
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._is_running:
            logger.error("Paper trader is not running")
            return False
        
        try:
            # Check if already subscribed
            if symbol in self._market_data:
                logger.debug(f"Already subscribed to {symbol}")
                return True
            
            # Get market data from IB
            ticker = self.ib_connector.get_market_data(symbol, subscribe=True)
            
            if not ticker:
                logger.error(f"Failed to subscribe to market data for {symbol}")
                return False
            
            # Store market data
            self._market_data[symbol] = {
                'ticker': ticker,
                'last_price': ticker.last if hasattr(ticker, 'last') and ticker.last else ticker.close if hasattr(ticker, 'close') else None,
                'bid': ticker.bid if hasattr(ticker, 'bid') and ticker.bid else None,
                'ask': ticker.ask if hasattr(ticker, 'ask') and ticker.ask else None,
                'volume': ticker.volume if hasattr(ticker, 'volume') and ticker.volume else 0,
                'time': datetime.now(),
                'subscription_time': datetime.now()
            }
            
            logger.info(f"Subscribed to market data for {symbol}")
            
            # Setup IB callback for market data updates
            self.ib_connector.add_callback('market_data', self._on_market_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to market data for {symbol}: {str(e)}")
            return False
    
    def unsubscribe_market_data(self, symbol: str) -> bool:
        """
        Unsubscribe from real-time market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to unsubscribe from
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._is_running:
            logger.error("Paper trader is not running")
            return False
        
        try:
            # Check if subscribed
            if symbol not in self._market_data:
                logger.debug(f"Not subscribed to {symbol}")
                return True
            
            # Cancel market data from IB
            self.ib_connector.cancel_market_data(symbol)
            
            # Remove from market data
            del self._market_data[symbol]
            
            logger.info(f"Unsubscribed from market data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from market data for {symbol}: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str, auto_subscribe: bool = True) -> Dict[str, Any]:
        """
        Get current market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get market data for
        auto_subscribe : bool
            If True, automatically subscribe if not already subscribed
        
        Returns:
        --------
        Dict[str, Any]
            Market data or empty dict if not available
        """
        if not self._is_running:
            logger.error("Paper trader is not running")
            return {}
        
        try:
            # Check if subscribed
            if symbol not in self._market_data:
                if not auto_subscribe:
                    logger.warning(f"Not subscribed to {symbol}")
                    return {}
                
                # Subscribe to market data
                if not self.subscribe_market_data(symbol):
                    logger.error(f"Failed to subscribe to market data for {symbol}")
                    return {}
            
            # Check if market data is stale (older than 5 seconds)
            if (datetime.now() - self._market_data[symbol]['time']).total_seconds() > 5:
                # Get fresh market data from IB
                ticker = self.ib_connector.get_market_data(symbol, subscribe=False)
                
                if ticker:
                    # Update market data
                    self._market_data[symbol].update({
                        'ticker': ticker,
                        'last_price': ticker.last if hasattr(ticker, 'last') and ticker.last else ticker.close if hasattr(ticker, 'close') else self._market_data[symbol]['last_price'],
                        'bid': ticker.bid if hasattr(ticker, 'bid') and ticker.bid else self._market_data[symbol]['bid'],
                        'ask': ticker.ask if hasattr(ticker, 'ask') and ticker.ask else self._market_data[symbol]['ask'],
                        'volume': ticker.volume if hasattr(ticker, 'volume') and ticker.volume else self._market_data[symbol]['volume'],
                        'time': datetime.now()
                    })
            
            # Return copy of market data (without ticker object)
            result = self._market_data[symbol].copy()
            result.pop('ticker', None)
            return result
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return {}
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get quote for
        
        Returns:
        --------
        Dict[str, Any]
            Quote data with bid, ask, last price or empty dict if not available
        """
        market_data = self.get_market_data(symbol)
        
        if not market_data:
            return {}
        
        return {
            'bid': market_data.get('bid'),
            'ask': market_data.get('ask'),
            'last_price': market_data.get('last_price'),
            'time': market_data.get('time')
        }
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get price for
        
        Returns:
        --------
        Optional[float]
            Current price or None if not available
        """
        quote = self.get_quote(symbol)
        
        if not quote:
            return None
        
        # Use last price if available, otherwise midpoint of bid/ask
        if quote.get('last_price'):
            return quote['last_price']
        elif quote.get('bid') and quote.get('ask'):
            return (quote['bid'] + quote['ask']) / 2
        else:
            return None
    
    def _on_market_data(self, symbol: str, ticker: Any) -> None:
        """
        Handle market data updates from IB.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        ticker : Any
            Ticker object
        """
        if symbol not in self._market_data:
            return
        
        # Update market data
        self._market_data[symbol].update({
            'ticker': ticker,
            'last_price': ticker.last if hasattr(ticker, 'last') and ticker.last else ticker.close if hasattr(ticker, 'close') else self._market_data[symbol]['last_price'],
            'bid': ticker.bid if hasattr(ticker, 'bid') and ticker.bid else self._market_data[symbol]['bid'],
            'ask': ticker.ask if hasattr(ticker, 'ask') and ticker.ask else self._market_data[symbol]['ask'],
            'volume': ticker.volume if hasattr(ticker, 'volume') and ticker.volume else self._market_data[symbol]['volume'],
            'time': datetime.now()
        })
        
        # Notify callbacks
        self._notify_callbacks('market_data', symbol, self._market_data[symbol])
    
    # -------------------------------------------------------------------------
    # Order Methods
    # -------------------------------------------------------------------------
    
    def place_order(self,
                  symbol: str,
                  action: str,
                  quantity: float,
                  order_type: str = 'MKT',
                  limit_price: Optional[float] = None,
                  stop_price: Optional[float] = None,
                  time_in_force: str = 'GTC',
                  order_id: Optional[str] = None) -> Optional[str]:
        """
        Place a simulated order.
        
        Parameters:
        -----------
        symbol : str
            Symbol to place order for
        action : str
            Order action ('BUY' or 'SELL')
        quantity : float
            Order quantity
        order_type : str
            Order type (e.g., 'MKT', 'LMT', 'STP', 'STP LMT')
        limit_price : float, optional
            Limit price for limit orders
        stop_price : float, optional
            Stop price for stop orders
        time_in_force : str
            Time in force (e.g., 'GTC', 'DAY', 'IOC')
        order_id : str, optional
            Order ID. If not specified, a unique ID will be generated.
        
        Returns:
        --------
        Optional[str]
            Order ID if successful, None otherwise
        
        Raises:
        -------
        OrderExecutionError
            If order cannot be placed due to validation errors or system issues
        """
        if not self._is_running:
            error = OrderExecutionError(
                message="Paper trader is not running",
                error_code="TRADER_INACTIVE",
                source="PaperTrader.place_order"
            )
            log_exception(error, logger)
            return None
        
        try:
            # Validate order parameters
            if not symbol:
                raise OrderExecutionError(
                    message="Symbol is required",
                    error_code="INVALID_SYMBOL",
                    source="PaperTrader.place_order"
                )
            
            if action not in ['BUY', 'SELL']:
                raise OrderExecutionError(
                    message=f"Invalid action: {action}",
                    error_code="INVALID_ACTION",
                    source="PaperTrader.place_order",
                    details={"action": action}
                )
            
            if quantity <= 0:
                raise OrderExecutionError(
                    message=f"Invalid quantity: {quantity}",
                    error_code="INVALID_QUANTITY",
                    source="PaperTrader.place_order",
                    details={"quantity": quantity}
                )
            
            if order_type not in ['MKT', 'LMT', 'STP', 'STP LMT']:
                raise OrderExecutionError(
                    message=f"Invalid order type: {order_type}",
                    error_code="INVALID_ORDER_TYPE",
                    source="PaperTrader.place_order",
                    details={"order_type": order_type}
                )
            
            if order_type in ['LMT', 'STP LMT'] and limit_price is None:
                raise OrderExecutionError(
                    message="Limit price is required for limit orders",
                    error_code="MISSING_LIMIT_PRICE",
                    source="PaperTrader.place_order"
                )
            
            if order_type in ['STP', 'STP LMT'] and stop_price is None:
                raise OrderExecutionError(
                    message="Stop price is required for stop orders",
                    error_code="MISSING_STOP_PRICE",
                    source="PaperTrader.place_order"
                )
            
            # Get current price for the symbol
            current_price = self.get_price(symbol)
            if current_price is None:
                # Subscribe to market data if not available
                self.subscribe_market_data(symbol)
                current_price = self.get_price(symbol)
                
                if current_price is None:
                    raise OrderExecutionError(
                        message=f"Cannot get current price for {symbol}",
                        error_code="PRICE_UNAVAILABLE",
                        source="PaperTrader.place_order",
                        details={"symbol": symbol}
                    )
            
            # Generate order ID if not provided
            if order_id is None:
                order_id = f"ORD_{uuid.uuid4().hex[:12]}"
            
            # Create order record
            order = {
                'id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type,
                'limit_price': limit_price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'status': 'PENDING',
                'is_active': True,
                'creation_time': datetime.now(),
                'last_update_time': datetime.now(),
                'filled_quantity': 0,
                'avg_fill_price': 0.0,
                'error_message': None
            }
            
            # Add to orders
            self._orders[order_id] = order
            
            logger.info(f"Order placed: {action} {quantity} {symbol} ({order_type}) - ID: {order_id}")
            
            # Notify callbacks
            self._notify_callbacks('order_status', order)
            
            return order_id
            
        except OrderExecutionError as e:
            # This is already a custom error, just log it
            log_exception(e, logger)
            return None
        except Exception as e:
            # Wrap generic exceptions in a custom error
            error = OrderExecutionError(
                message=f"Error placing order: {str(e)}",
                error_code="ORDER_EXECUTION_FAILED",
                source="PaperTrader.place_order",
                cause=e,
                details={
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "order_type": order_type
                }
            )
            log_exception(error, logger)
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a simulated order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self._is_running:
            logger.error("Paper trader is not running")
            return False
        
        try:
            # Check if order exists
            if order_id not in self._orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            # Check if order can be canceled
            if self._orders[order_id]['status'] in ['FILLED', 'CANCELLED', 'REJECTED']:
                logger.warning(f"Order {order_id} cannot be canceled (status: {self._orders[order_id]['status']})")
                return False
            
            # Cancel order
            self._orders[order_id]['status'] = 'CANCELLED'
            self._orders[order_id]['is_active'] = False
            self._orders[order_id]['last_update_time'] = datetime.now()
            
            logger.info(f"Order {order_id} canceled")
            
            # Notify callbacks
            self._notify_callbacks('order_status', self._orders[order_id])
            
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            return False
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details.
        
        Parameters:
        -----------
        order_id : str
            Order ID
        
        Returns:
        --------
        Dict[str, Any]
            Order details or empty dict if not found
        """
        if order_id not in self._orders:
            return {}
        
        return self._orders[order_id].copy()
    
    def get_orders(self, symbol: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get all orders, optionally filtered by symbol and/or status.
        
        Parameters:
        -----------
        symbol : str, optional
            Filter by symbol
        status : str, optional
            Filter by status
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of orders, keyed by order ID
        """
        result = {}
        
        for order_id, order in self._orders.items():
            # Apply filters
            if symbol and order['symbol'] != symbol:
                continue
            
            if status and order['status'] != status:
                continue
            
            result[order_id] = order.copy()
        
        return result
    
    def place_bracket_order(self,
                         symbol: str,
                         action: str,
                         quantity: float,
                         entry_order_type: str = 'MKT',
                         entry_price: Optional[float] = None,
                         profit_price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = 'GTC') -> Dict[str, Optional[str]]:
        """
        Place a bracket order (entry, profit target, stop loss).
        
        Parameters:
        -----------
        symbol : str
            Symbol to place order for
        action : str
            Order action ('BUY' or 'SELL')
        quantity : float
            Order quantity
        entry_order_type : str
            Entry order type (e.g., 'MKT', 'LMT')
        entry_price : float, optional
            Entry price for limit orders
        profit_price : float, optional
            Price for profit target order
        stop_price : float, optional
            Price for stop loss order
        time_in_force : str
            Time in force (e.g., 'GTC', 'DAY')
        
        Returns:
        --------
        Dict[str, Optional[str]]
            Dictionary with entry, profit, and stop order IDs
        """
        if not self._is_running:
            logger.error("Paper trader is not running")
            return {'entry': None, 'profit': None, 'stop': None}
        
        try:
            # Validate action
            if action.upper() not in ['BUY', 'SELL']:
                logger.error(f"Invalid action: {action}")
                return {'entry': None, 'profit': None, 'stop': None}
            
            # Calculate exit action
            exit_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
            
            # Place entry order
            entry_id = self.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=entry_order_type,
                limit_price=entry_price if entry_order_type == 'LMT' else None,
                time_in_force=time_in_force
            )
            
            if not entry_id:
                logger.error("Failed to place entry order")
                return {'entry': None, 'profit': None, 'stop': None}
            
            # Set parent-child relationship
            self._orders[entry_id]['is_parent'] = True
            self._orders[entry_id]['children'] = []
            
            # Place profit target order if specified
            profit_id = None
            if profit_price is not None:
                profit_id = self.place_order(
                    symbol=symbol,
                    action=exit_action,
                    quantity=quantity,
                    order_type='LMT',
                    limit_price=profit_price,
                    time_in_force=time_in_force
                )
                
                if profit_id:
                    # Set as child order and initially inactive
                    self._orders[profit_id]['is_child'] = True
                    self._orders[profit_id]['parent_id'] = entry_id
                    self._orders[profit_id]['is_active'] = False
                    self._orders[profit_id]['status'] = 'INACTIVE'
                    self._orders[entry_id]['children'].append(profit_id)
            
            # Place stop loss order if specified
            stop_id = None
            if stop_price is not None:
                stop_id = self.place_order(
                    symbol=symbol,
                    action=exit_action,
                    quantity=quantity,
                    order_type='STP',
                    stop_price=stop_price,
                    time_in_force=time_in_force
                )
                
                if stop_id:
                    # Set as child order and initially inactive
                    self._orders[stop_id]['is_child'] = True
                    self._orders[stop_id]['parent_id'] = entry_id
                    self._orders[stop_id]['is_active'] = False
                    self._orders[stop_id]['status'] = 'INACTIVE'
                    self._orders[entry_id]['children'].append(stop_id)
            
            logger.info(f"Bracket order placed: {action} {quantity} {symbol}")
            return {'entry': entry_id, 'profit': profit_id, 'stop': stop_id}
            
        except Exception as e:
            logger.error(f"Error placing bracket order: {str(e)}")
            return {'entry': None, 'profit': None, 'stop': None}
    
    def _process_orders(self) -> None:
        """Process pending orders."""
        if not self._is_running:
            return
        
        try:
            # Process active orders
            for order_id, order in list(self._orders.items()):
                if not order['is_active'] or order['status'] not in ['PENDING', 'PARTIALLY_FILLED']:
                    continue
                
                # Check if we have market data for the symbol
                if order['symbol'] not in self._market_data:
                    self.subscribe_market_data(order['symbol'])
                    continue
                
                # Get current price
                current_price = self.get_price(order['symbol'])
                if current_price is None:
                    continue
                
                # Simulate execution based on order type
                self._simulate_execution(order_id, current_price)
                
        except Exception as e:
            logger.error(f"Error processing orders: {str(e)}")

    def _simulate_execution(self, order_id: str, current_price: float) -> None:
        """
        Simulate order execution.
        
        Parameters:
        -----------
        order_id : str
            Order ID
        current_price : float
            Current price
        """
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        
        # Check order type
        if order['order_type'] == 'MKT':
            # Market orders execute immediately at current price
            self._execute_order(order_id, current_price, order['quantity'] - order['filled_quantity'])
            
        elif order['order_type'] == 'LMT':
            # Limit orders execute when price is favorable
            if (order['action'] == 'BUY' and current_price <= order['limit_price']) or \
               (order['action'] == 'SELL' and current_price >= order['limit_price']):
                self._execute_order(order_id, order['limit_price'], order['quantity'] - order['filled_quantity'])
                
        elif order['order_type'] == 'STP':
            # Stop orders convert to market orders when price hits stop
            if (order['action'] == 'BUY' and current_price >= order['stop_price']) or \
               (order['action'] == 'SELL' and current_price <= order['stop_price']):
                self._execute_order(order_id, current_price, order['quantity'] - order['filled_quantity'])
                
        elif order['order_type'] == 'STP LMT':
            # Stop-limit orders convert to limit orders when price hits stop
            if (order['action'] == 'BUY' and current_price >= order['stop_price']) or \
               (order['action'] == 'SELL' and current_price <= order['stop_price']):
                # Convert to limit order
                order['order_type'] = 'LMT'
                order['status'] = 'PENDING'  # Re-check as limit order
                self._orders[order_id] = order

    def _execute_order(self, order_id: str, price: float, quantity: float) -> None:
        """
        Execute a simulated order.
        
        Parameters:
        -----------
        order_id : str
            Order ID
        price : float
            Execution price
        quantity : float
            Quantity to execute
        """
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        
        # Apply slippage to price
        if self.slippage_model == 'fixed':
            # Fixed percentage slippage
            slippage = price * self.slippage_factor
            if order['action'] == 'BUY':
                # Buy orders get worse prices (higher)
                price += slippage
            else:
                # Sell orders get worse prices (lower)
                price -= slippage
        elif self.slippage_model == 'variable':
            # Variable slippage based on symbol volatility (simplified)
            # In a real implementation, we would use historical volatility
            slippage = price * self.slippage_factor * np.random.random()
            if order['action'] == 'BUY':
                price += slippage
            else:
                price -= slippage
        
        # Calculate commission
        commission = self._calculate_commission(order['symbol'], price, quantity)
        
        # Update order status
        order['filled_quantity'] += quantity
        
        # Calculate average fill price including this execution
        if order['filled_quantity'] > 0:
            # Weighted average of previous fills and current fill
            prev_qty = order['filled_quantity'] - quantity
            if prev_qty > 0:
                prev_avg_price = order['avg_fill_price']
                order['avg_fill_price'] = (prev_avg_price * prev_qty + price * quantity) / order['filled_quantity']
            else:
                order['avg_fill_price'] = price
        
        # Update order status
        if order['filled_quantity'] >= order['quantity']:
            order['status'] = 'FILLED'
            order['is_active'] = False
        else:
            order['status'] = 'PARTIALLY_FILLED'
        
        order['last_update_time'] = datetime.now()
        
        # Create trade record
        trade = {
            'order_id': order_id,
            'symbol': order['symbol'],
            'action': order['action'],
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'time': datetime.now()
        }
        
        # Add to trades list
        self._trades.append(trade)
        
        # Update position
        self._update_position(order['symbol'], order['action'], quantity, price, commission)
        
        # Activate child orders if this order is filled and is a parent
        if order['status'] == 'FILLED' and order.get('is_parent', False):
            for child_id in order.get('children', []):
                if child_id in self._orders:
                    self._orders[child_id]['is_active'] = True
                    self._orders[child_id]['status'] = 'PENDING'
        
        # Notify callbacks
        self._notify_callbacks('order_status', order)
        self._notify_callbacks('trade', trade)

    def _update_position(self, symbol: str, action: str, quantity: float, price: float, commission: float) -> None:
        """
        Update position based on a trade.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        action : str
            Action ('BUY' or 'SELL')
        quantity : float
            Quantity
        price : float
            Price
        commission : float
            Commission
        """
        # Convert action to position adjustment
        position_change = quantity if action == 'BUY' else -quantity
        
        # Update position
        if symbol in self._positions:
            # Add to existing position
            self._positions[symbol]['quantity'] += position_change
            
            # Update average cost for buys (increasing position)
            if (position_change > 0 and self._positions[symbol]['quantity'] > 0) or \
               (position_change < 0 and self._positions[symbol]['quantity'] < 0):
                prev_qty = self._positions[symbol]['quantity'] - position_change
                prev_cost = self._positions[symbol]['avg_cost']
                
                if prev_qty != 0:
                    # Weighted average of previous cost and current price
                    self._positions[symbol]['avg_cost'] = \
                        (prev_cost * abs(prev_qty) + price * abs(position_change)) / abs(self._positions[symbol]['quantity'])
                else:
                    self._positions[symbol]['avg_cost'] = price
            
            # If position crosses zero, reset average cost
            if (prev_qty > 0 and self._positions[symbol]['quantity'] <= 0) or \
               (prev_qty < 0 and self._positions[symbol]['quantity'] >= 0):
                self._positions[symbol]['avg_cost'] = price
            
            # Remove position if quantity is zero
            if self._positions[symbol]['quantity'] == 0:
                del self._positions[symbol]
        else:
            # Create new position
            self._positions[symbol] = {
                'quantity': position_change,
                'avg_cost': price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        
        # Update account
        cost = price * abs(position_change) + commission
        self._account['cash'] -= cost if action == 'BUY' else -cost
        
        # Notify position change
        self._notify_callbacks('position_change', symbol, self._positions.get(symbol, {}))

    def _calculate_commission(self, symbol: str, price: float, quantity: float) -> float:
        """
        Calculate commission for a trade.
        
        Parameters:
        -----------
        symbol : str
            Symbol
        price : float
            Price
        quantity : float
            Quantity
        
        Returns:
        --------
        float
            Commission amount
        """
        if self.commission_model == 'none':
            return 0.0
        
        # Get contract details
        contract_details = self.ib_connector.get_contract_details(symbol)
        is_stock = contract_details.get('contract_type') == 'Stock' if contract_details else True
        
        # Calculate commission based on model
        if self.commission_model == 'ibkr_pro':
            if is_stock:
                # IBKR Pro stock commission: min($1.00, 0.005 * shares)
                return min(1.00, 0.005 * quantity)
            else:
                # IBKR Pro futures commission: ~$0.85 per contract
                return 0.85 * quantity
        elif self.commission_model == 'ibkr_lite':
            if is_stock:
                # IBKR Lite stock commission: $0 for US stocks
                return 0.0
            else:
                # IBKR Lite futures commission: same as Pro
                return 0.85 * quantity
        elif self.commission_model == 'flat':
            # Flat commission per trade
            return 1.0
        
        return 0.0

    def _update_account(self) -> None:
        """Update account values based on current positions and market data."""
        if not self._is_running:
            return
        
        try:
            # Calculate equity, margin used, and unrealized P&L
            equity = self._account['cash']
            margin_used = 0.0
            unrealized_pnl = 0.0
            
            # Update positions with current market prices
            for symbol, position in list(self._positions.items()):
                # Get current price
                current_price = self.get_price(symbol)
                if current_price is None:
                    continue
                
                # Calculate position value and unrealized P&L
                position_value = current_price * position['quantity']
                position_cost = position['avg_cost'] * position['quantity']
                
                # Update position
                position['market_price'] = current_price
                position['market_value'] = position_value
                position['unrealized_pnl'] = position_value - position_cost
                
                # Update totals
                equity += position_value
                margin_used += abs(position_value)
                unrealized_pnl += position['unrealized_pnl']
            
            # Update account values
            self._account['equity'] = equity
            self._account['margin_used'] = margin_used
            self._account['unrealized_pnl'] = unrealized_pnl
            self._account['last_update_time'] = datetime.now()
            
            # Notify account update
            self._notify_callbacks('account_update', self._account)
            
        except Exception as e:
            logger.error(f"Error updating account: {str(e)}") 