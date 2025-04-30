"""
Interactive Brokers (IB) Connector for the Intraday Statistical Arbitrage System.

This module provides a high-level connector to Interactive Brokers for
market data retrieval, order execution, and account management using the
ib_insync library.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ib_insync import IB, Contract, Order, Trade, Ticker, util
from ib_insync import Stock, Future, Forex, Index, BarData

# Patch asyncio to enable ib_insync to work in synchronous code
util.patchAsyncio()

from .ib_utils import contract_to_symbol, symbol_to_contract, parse_contract_details, is_regular_trading_hours

# Configure logging
logger = logging.getLogger(__name__)


class IBConnector:
    """
    Interactive Brokers (IB) Connector for the Intraday Statistical Arbitrage System.
    
    This class provides a high-level interface to Interactive Brokers for
    market data retrieval, order execution, and account management.
    """
    
    def __init__(self,
                host: str = '127.0.0.1',
                port: int = 7497,  # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway Paper, 4001 for Gateway Live
                client_id: int = 1,
                account: Optional[str] = None,
                timeout: int = 30,
                read_only: bool = False,
                max_retry_count: int = 3,
                retry_wait_time: int = 5,
                auto_reconnect: bool = True,
                use_async: bool = False):
        """
        Initialize the IB connector.
        
        Parameters:
        -----------
        host : str
            IB TWS/Gateway hostname or IP address
        port : int
            IB TWS/Gateway port
        client_id : int
            Client ID for IB connection
        account : str, optional
            IB account ID. If None, the first account will be used
        timeout : int
            Connection timeout in seconds
        read_only : bool
            If True, no orders will be placed (for testing)
        max_retry_count : int
            Maximum number of connection retry attempts
        retry_wait_time : int
            Wait time between retry attempts in seconds
        auto_reconnect : bool
            If True, automatically reconnect on disconnection
        use_async : bool
            If True, use async/await pattern (requires event loop)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self.timeout = timeout
        self.read_only = read_only
        self.max_retry_count = max_retry_count
        self.retry_wait_time = retry_wait_time
        self.auto_reconnect = auto_reconnect
        self.use_async = use_async
        
        # Create IB connection object
        self.ib = IB()
        
        # Keep track of subscribed market data
        self._market_data_subscriptions = {}
        
        # Keep track of active orders
        self._active_orders = {}
        
        # Keep track of positions
        self._positions = {}
        
        # Keep track of contracts by symbol
        self._contract_details_by_symbol = {}
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Connect flag
        self._is_connected = False
        
        # Keep track of event callbacks
        self._callbacks = {
            'order_status': [],
            'position_change': [],
            'account_update': [],
            'market_data': [],
            'connection_state': [],
            'error': []
        }
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway.
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        if self._is_connected:
            logger.warning("Already connected to IB")
            return True
        
        retry_count = 0
        while retry_count < self.max_retry_count:
            try:
                logger.info(f"Connecting to IB at {self.host}:{self.port} (client ID: {self.client_id})")
                
                # Connect to IB
                self.ib.connect(
                    self.host, 
                    self.port,
                    clientId=self.client_id,
                    timeout=self.timeout,
                    readonly=self.read_only
                )
                
                # Check if connected
                if not self.ib.isConnected():
                    logger.error("Failed to connect to IB")
                    retry_count += 1
                    time.sleep(self.retry_wait_time)
                    continue
                
                # Set account if specified, otherwise use first account
                if not self.account:
                    accounts = self.ib.managedAccounts()
                    if accounts:
                        self.account = accounts[0]
                        logger.info(f"Using account: {self.account}")
                    else:
                        logger.error("No accounts available")
                        return False
                
                # Skip the problematic reqAccountUpdates call
                # self.ib.reqAccountUpdates(True, self.account)
                
                # Update positions manually
                self._update_positions()
                
                # Set connected flag
                self._is_connected = True
                logger.info("Successfully connected to IB")
                
                # Notify connection state callbacks
                self._notify_callbacks('connection_state', True)
                
                return True
                
            except Exception as e:
                logger.error(f"Error connecting to IB: {str(e)}")
                retry_count += 1
                if retry_count < self.max_retry_count:
                    logger.info(f"Retrying connection in {self.retry_wait_time} seconds...")
                    time.sleep(self.retry_wait_time)
                else:
                    logger.error(f"Failed to connect after {self.max_retry_count} attempts")
                    return False
        
        return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers TWS/Gateway."""
        if self._is_connected:
            logger.info("Disconnecting from IB")
            
            # Cancel all market data subscriptions
            for contract in list(self._market_data_subscriptions.values()):
                try:
                    self.ib.cancelMktData(contract)
                except Exception as e:
                    logger.error(f"Error canceling market data: {str(e)}")
            
            # Disconnect from IB
            self.ib.disconnect()
            
            # Set connected flag
            self._is_connected = False
            logger.info("Disconnected from IB")
            
            # Notify connection state callbacks
            self._notify_callbacks('connection_state', False)
    
    def is_connected(self) -> bool:
        """
        Check if connected to Interactive Brokers.
        
        Returns:
        --------
        bool
            True if connected, False otherwise
        """
        return self._is_connected and self.ib.isConnected()
    
    def _setup_event_handlers(self) -> None:
        """Setup IB event handlers."""
        # Order status updates
        self.ib.orderStatusEvent += self._on_order_status
        
        # Execution details
        self.ib.execDetailsEvent += self._on_execution_details
        
        # Position updates
        self.ib.positionEvent += self._on_position
        
        # Account updates
        self.ib.accountValueEvent += self._on_account_value
        
        # Connection lost
        self.ib.disconnectedEvent += self._on_disconnected
        
        # Error handling
        self.ib.errorEvent += self._on_error
        
        # Market data updates
        self.ib.pendingTickersEvent += self._on_pending_tickers
    
    def _update_positions(self) -> None:
        """Update positions dictionary."""
        try:
            # Request positions
            positions = self.ib.reqPositions()
            
            # Clear positions dict
            self._positions = {}
            
            # Update positions dict
            for position in positions:
                symbol = contract_to_symbol(position.contract)
                self._positions[symbol] = {
                    'symbol': symbol,
                    'contract': position.contract,
                    'position': position.position,
                    'avg_cost': position.avgCost
                }
            
            logger.info(f"Positions updated: {len(self._positions)} active positions")
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of account values
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        try:
            # Use a very simple approach just to get basic account info
            # Without relying on problematic API calls
            result = {}
            
            # Just return the account ID if we have it
            if self.account:
                result['Account'] = self.account
                
            # Get available accounts
            accounts = self.ib.managedAccounts()
            if accounts:
                result['ManagedAccounts'] = ','.join(accounts)
                
            # Try to get client ID and connection info
            result['ClientId'] = self.client_id
            result['Connected'] = self.is_connected()
            result['Host'] = self.host
            result['Port'] = self.port
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of positions, keyed by symbol
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        # Update positions first
        self._update_positions()
        
        return self._positions
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get position for
        
        Returns:
        --------
        Dict[str, Any]
            Position information or empty dict if no position
        """
        positions = self.get_positions()
        return positions.get(symbol, {})
    
    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Get contract details for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get contract details for
        
        Returns:
        --------
        Dict[str, Any]
            Contract details or empty dict if no details found
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        # Check if we already have contract details
        if symbol in self._contract_details_by_symbol:
            return self._contract_details_by_symbol[symbol]
        
        try:
            # Convert symbol to contract
            contract = symbol_to_contract(symbol)
            
            # Request contract details
            details_list = self.ib.reqContractDetails(contract)
            
            if not details_list:
                logger.warning(f"No contract details found for {symbol}")
                return {}
            
            # Parse details
            details = parse_contract_details(details_list[0])
            
            # Cache details
            self._contract_details_by_symbol[symbol] = details
            
            return details
            
        except Exception as e:
            logger.error(f"Error getting contract details for {symbol}: {str(e)}")
            return {}
    
    def get_market_data(self, symbol: str, subscribe: bool = True) -> Optional[Ticker]:
        """
        Get real-time market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get market data for
        subscribe : bool
            If True, subscribe to market data updates
        
        Returns:
        --------
        Optional[Ticker]
            Market data ticker or None if not available
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            # Check if already subscribed
            if symbol in self._market_data_subscriptions:
                return self.ib.ticker(self._market_data_subscriptions[symbol])
            
            # Convert symbol to contract
            contract = symbol_to_contract(symbol)
            
            # Request market data
            ticker = self.ib.reqMktData(contract)
            
            # Wait a bit for data to arrive
            for _ in range(10):  # Wait up to 1 second
                if ticker.time:
                    break
                time.sleep(0.1)
                self.ib.sleep(0)
            
            # Subscribe if requested
            if subscribe:
                self._market_data_subscriptions[symbol] = contract
            else:
                self.ib.cancelMktData(contract)
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def cancel_market_data(self, symbol: str) -> bool:
        """
        Cancel market data subscription for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to cancel market data for
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return False
        
        try:
            if symbol in self._market_data_subscriptions:
                contract = self._market_data_subscriptions[symbol]
                self.ib.cancelMktData(contract)
                del self._market_data_subscriptions[symbol]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error canceling market data for {symbol}: {str(e)}")
            return False
    
    def get_historical_data(self, 
                          symbol: str,
                          start: Optional[datetime] = None,
                          end: Optional[datetime] = None,
                          duration: str = '1 D',
                          bar_size: str = '1 min',
                          what_to_show: str = 'TRADES',
                          use_rth: bool = True) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get historical data for
        start : datetime, optional
            Start datetime for historical data. If specified, duration is ignored.
        end : datetime, optional
            End datetime for historical data. If not specified, current time is used.
        duration : str
            Duration of historical data (e.g., '1 D', '1 W', '1 M')
        bar_size : str
            Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')
        what_to_show : str
            Type of data to retrieve (e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK')
        use_rth : bool
            If True, only include data from regular trading hours
        
        Returns:
        --------
        pd.DataFrame
            DataFrame of historical data with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return pd.DataFrame()
        
        try:
            # Convert symbol to contract
            contract = symbol_to_contract(symbol)
            
            # Set end time if not specified
            if end is None:
                end = datetime.now()
            
            # Set up request parameters
            if start is not None:
                # Calculate duration from start to end
                delta = end - start
                days = delta.days
                seconds = delta.seconds
                
                if days > 365:
                    duration = f"{days // 365} Y"
                elif days > 30:
                    duration = f"{days // 30} M"
                elif days > 0:
                    duration = f"{days} D"
                else:
                    duration = f"{seconds // 3600} H"
                
                # Adjust end time for IB API which treats it as exclusive
                end = end + timedelta(seconds=1)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth
            )
            
            if not bars:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(bars)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Set index to date
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def place_order(self, 
                  symbol: str,
                  action: str,
                  quantity: float,
                  order_type: str = 'MKT',
                  limit_price: Optional[float] = None,
                  stop_price: Optional[float] = None,
                  time_in_force: str = 'GTC',
                  outside_rth: bool = False,
                  order_id: Optional[int] = None,
                  parent_id: Optional[int] = None,
                  transmit: bool = True,
                  **kwargs) -> Optional[Trade]:
        """
        Place an order.
        
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
        outside_rth : bool
            If True, allow order execution outside regular trading hours
        order_id : int, optional
            Order ID. If not specified, a unique ID will be generated.
        parent_id : int, optional
            Parent order ID for bracket orders
        transmit : bool
            If True, transmit the order to IB
        **kwargs : dict
            Additional order parameters
        
        Returns:
        --------
        Optional[Trade]
            Trade object if successful, None otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        if self.read_only:
            logger.warning("In read-only mode, order not placed")
            return None
        
        try:
            # Convert symbol to contract
            contract = symbol_to_contract(symbol)
            
            # Validate action
            if action.upper() not in ['BUY', 'SELL']:
                logger.error(f"Invalid action: {action}")
                return None
            
            # Create order object
            order = Order()
            order.action = action.upper()
            order.totalQuantity = quantity
            order.orderType = order_type.upper()
            order.tif = time_in_force.upper()
            order.outsideRth = outside_rth
            
            # Set order ID if specified
            if order_id is not None:
                order.orderId = order_id
            
            # Set parent ID if specified
            if parent_id is not None:
                order.parentId = parent_id
            
            # Set transmit flag
            order.transmit = transmit
            
            # Set prices if specified
            if limit_price is not None:
                order.lmtPrice = limit_price
            
            if stop_price is not None:
                order.auxPrice = stop_price
            
            # Add additional parameters
            for key, value in kwargs.items():
                setattr(order, key, value)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            logger.info(f"Order placed: {action} {quantity} {symbol} ({order_type})")
            
            # Track active order
            if trade and trade.order and trade.order.orderId:
                self._active_orders[trade.order.orderId] = {
                    'trade': trade,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': order_type,
                    'time': datetime.now()
                }
            
            return trade
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def place_bracket_order(self,
                          symbol: str,
                          action: str,
                          quantity: float,
                          entry_order_type: str = 'MKT',
                          entry_price: Optional[float] = None,
                          profit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          time_in_force: str = 'GTC',
                          outside_rth: bool = False,
                          **kwargs) -> List[Optional[Trade]]:
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
            Entry order type (e.g., 'MKT', 'LMT', 'STP', 'STP LMT')
        entry_price : float, optional
            Entry price for limit orders
        profit_price : float, optional
            Price for profit target order
        stop_price : float, optional
            Price for stop loss order
        time_in_force : str
            Time in force (e.g., 'GTC', 'DAY', 'IOC')
        outside_rth : bool
            If True, allow order execution outside regular trading hours
        **kwargs : dict
            Additional order parameters
        
        Returns:
        --------
        List[Optional[Trade]]
            List of trade objects [entry, profit, stop] if successful
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return [None, None, None]
        
        if self.read_only:
            logger.warning("In read-only mode, bracket order not placed")
            return [None, None, None]
        
        try:
            # Get next order ID
            next_id = self.ib.client.getReqId()
            
            # Validate action
            if action.upper() not in ['BUY', 'SELL']:
                logger.error(f"Invalid action: {action}")
                return [None, None, None]
            
            # Calculate opposite action for exit orders
            exit_action = 'SELL' if action.upper() == 'BUY' else 'BUY'
            
            # Create bracket order
            bracket = self.ib.bracketOrder(
                contract=symbol_to_contract(symbol),
                action=action.upper(),
                quantity=quantity,
                limitPrice=entry_price,
                takeProfitPrice=profit_price,
                stopLossPrice=stop_price,
                outsideRth=outside_rth,
                tif=time_in_force.upper(),
                **kwargs
            )
            
            # Modify entry order if needed
            parent = bracket[0]
            if entry_order_type != 'LMT':
                parent.orderType = entry_order_type
                if entry_order_type == 'MKT':
                    parent.lmtPrice = 0
            
            # Place orders
            trades = []
            for order in bracket:
                trade = self.ib.placeOrder(symbol_to_contract(symbol), order)
                trades.append(trade)
                
                if trade and trade.order and trade.order.orderId:
                    self._active_orders[trade.order.orderId] = {
                        'trade': trade,
                        'symbol': symbol,
                        'action': action if trade == trades[0] else exit_action,
                        'quantity': quantity,
                        'order_type': trade.order.orderType,
                        'time': datetime.now()
                    }
            
            logger.info(f"Bracket order placed: {action} {quantity} {symbol}")
            
            return trades
            
        except Exception as e:
            logger.error(f"Error placing bracket order: {str(e)}")
            return [None, None, None]
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : int
            Order ID to cancel
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return False
        
        if self.read_only:
            logger.warning("In read-only mode, order not canceled")
            return False
        
        try:
            # Check if order exists
            if order_id not in self._active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                
                # Check if exists in IB
                for trade in self.ib.openTrades():
                    if trade.order.orderId == order_id:
                        self.ib.cancelOrder(trade.order)
                        logger.info(f"Order {order_id} canceled")
                        return True
                
                return False
            
            # Get trade object
            trade = self._active_orders[order_id]['trade']
            
            # Cancel order
            self.ib.cancelOrder(trade.order)
            
            logger.info(f"Order {order_id} canceled")
            
            # Remove from active orders
            del self._active_orders[order_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order: {str(e)}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all active orders.
        
        Returns:
        --------
        bool
            True if all orders canceled successfully, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return False
        
        if self.read_only:
            logger.warning("In read-only mode, orders not canceled")
            return False
        
        try:
            # Cancel orders through IB API
            self.ib.reqGlobalCancel()
            
            # Clear active orders
            self._active_orders = {}
            
            logger.info("All orders canceled")
            
            return True
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            return False
    
    def get_order_status(self, order_id: int) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Parameters:
        -----------
        order_id : int
            Order ID to get status for
        
        Returns:
        --------
        Dict[str, Any]
            Order status information or empty dict if not found
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        try:
            # Check active orders
            if order_id in self._active_orders:
                trade = self._active_orders[order_id]['trade']
                
                return {
                    'order_id': order_id,
                    'symbol': self._active_orders[order_id]['symbol'],
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'filled': trade.orderStatus.filled,
                    'remaining': trade.orderStatus.remaining,
                    'status': trade.orderStatus.status,
                    'avg_fill_price': trade.orderStatus.avgFillPrice,
                    'whyHeld': trade.orderStatus.whyHeld,
                    'time': self._active_orders[order_id]['time']
                }
            
            # Check IB open trades
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    symbol = contract_to_symbol(trade.contract)
                    
                    return {
                        'order_id': order_id,
                        'symbol': symbol,
                        'action': trade.order.action,
                        'quantity': trade.order.totalQuantity,
                        'filled': trade.orderStatus.filled,
                        'remaining': trade.orderStatus.remaining,
                        'status': trade.orderStatus.status,
                        'avg_fill_price': trade.orderStatus.avgFillPrice,
                        'whyHeld': trade.orderStatus.whyHeld,
                        'time': datetime.now()  # Don't have original time
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return {}
    
    def get_active_orders(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all active orders.
        
        Returns:
        --------
        Dict[int, Dict[str, Any]]
            Dictionary of active orders, keyed by order ID
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return {}
        
        try:
            # Update active orders from IB
            for trade in self.ib.openTrades():
                order_id = trade.order.orderId
                
                # Skip if already in active orders
                if order_id in self._active_orders:
                    continue
                
                symbol = contract_to_symbol(trade.contract)
                
                self._active_orders[order_id] = {
                    'trade': trade,
                    'symbol': symbol,
                    'action': trade.order.action,
                    'quantity': trade.order.totalQuantity,
                    'order_type': trade.order.orderType,
                    'time': datetime.now()  # Don't have original time
                }
            
            # Create result dict with order status
            result = {}
            for order_id, order_info in self._active_orders.items():
                result[order_id] = self.get_order_status(order_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting active orders: {str(e)}")
            return {}
    
    def run_event_loop(self, timeout: float = 0.1) -> None:
        """
        Run the IB event loop for a specified time.
        
        Parameters:
        -----------
        timeout : float
            Time to wait for events in seconds
        """
        if not self.is_connected():
            return
        
        self.ib.sleep(timeout)
    
    def process_pending_events(self) -> None:
        """Process any pending IB events (non-blocking)."""
        if not self.is_connected():
            return
        
        self.ib.sleep(0)
    
    def add_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Add a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (order_status, position_change, account_update, market_data, connection_state, error)
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
            Event type (order_status, position_change, account_update, market_data, connection_state, error)
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
    
    def _on_order_status(self, trade: Trade) -> None:
        """
        Handle order status updates.
        
        Parameters:
        -----------
        trade : Trade
            Trade object
        """
        order_id = trade.order.orderId
        
        # Update active orders
        if order_id in self._active_orders:
            self._active_orders[order_id]['trade'] = trade
        
        # Check if order completed or canceled
        if trade.orderStatus.status in ['Filled', 'Cancelled']:
            # Remove from active orders
            if order_id in self._active_orders:
                del self._active_orders[order_id]
        
        # Notify callbacks
        self._notify_callbacks('order_status', trade)
    
    def _on_execution_details(self, trade: Trade, fill: Any) -> None:
        """
        Handle execution details.
        
        Parameters:
        -----------
        trade : Trade
            Trade object
        fill : Any
            Execution details
        """
        # Update positions
        self._update_positions()
    
    def _on_position(self, position: Any) -> None:
        """
        Handle position updates.
        
        Parameters:
        -----------
        position : Any
            Position object
        """
        symbol = contract_to_symbol(position.contract)
        
        self._positions[symbol] = {
            'symbol': symbol,
            'contract': position.contract,
            'position': position.position,
            'avg_cost': position.avgCost
        }
        
        # Notify callbacks
        self._notify_callbacks('position_change', symbol, position.position, position.avgCost)
    
    def _on_account_value(self, value: Any) -> None:
        """
        Handle account value updates.
        
        Parameters:
        -----------
        value : Any
            Account value object
        """
        # Notify callbacks
        self._notify_callbacks('account_update', value.tag, value.value, value.currency)
    
    def _on_disconnected(self) -> None:
        """Handle connection loss."""
        logger.warning("Disconnected from IB")
        
        # Set connected flag
        self._is_connected = False
        
        # Notify callbacks
        self._notify_callbacks('connection_state', False)
        
        # Attempt to reconnect if auto-reconnect enabled
        if self.auto_reconnect:
            logger.info("Attempting to reconnect...")
            self.connect()
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """
        Handle errors.
        
        Parameters:
        -----------
        reqId : int
            Request ID
        errorCode : int
            Error code
        errorString : str
            Error message
        contract : Contract
            Contract object, if applicable
        """
        # Log error
        if errorCode in [2104, 2106, 2158]:  # Common informational messages
            logger.debug(f"IB Info ({errorCode}): {errorString}")
        else:
            symbol = contract_to_symbol(contract) if contract else None
            logger.error(f"IB Error ({errorCode}): {errorString} - ReqID: {reqId}, Symbol: {symbol}")
        
        # Notify callbacks
        self._notify_callbacks('error', reqId, errorCode, errorString, contract)
    
    def _on_pending_tickers(self, tickers: List[Ticker]) -> None:
        """
        Handle market data updates.
        
        Parameters:
        -----------
        tickers : List[Ticker]
            List of updated tickers
        """
        for ticker in tickers:
            # Find symbol for this ticker
            symbol = None
            for sym, contract in self._market_data_subscriptions.items():
                if contract.conId == ticker.contract.conId:
                    symbol = sym
                    break
            
            if not symbol:
                continue
            
            # Notify callbacks
            self._notify_callbacks('market_data', symbol, ticker)
    
    def __del__(self) -> None:
        """Clean up when object is deleted."""
        if self._is_connected:
            self.disconnect() 