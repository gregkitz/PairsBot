"""
Mock Exchange for testing purposes.

This module provides a mock implementation of an exchange
for unit and integration testing without connecting to a real exchange.
"""

import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

class OrderType(Enum):
    """Order types supported by the mock exchange."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    """Order statuses supported by the mock exchange."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderSide(Enum):
    """Order sides supported by the mock exchange."""
    BUY = "BUY"
    SELL = "SELL"

class MockExchange:
    """
    Mock implementation of an exchange for testing purposes.
    
    This class simulates an exchange that can accept and execute orders,
    maintain a simple order book, and provide market data for testing.
    """
    
    def __init__(self, market_data_provider=None, latency_ms: int = 100, slippage_pct: float = 0.001):
        """
        Initialize the mock exchange.
        
        Parameters:
        -----------
        market_data_provider : object, optional
            Provider of market data (if None, use random data)
        latency_ms : int
            Simulated latency in milliseconds
        slippage_pct : float
            Simulated slippage as percentage of price
        """
        self.market_data_provider = market_data_provider
        self.latency_ms = latency_ms
        self.slippage_pct = slippage_pct
        
        # Internal state
        self.orders = {}  # Order ID -> Order dict
        self.positions = {}  # Symbol -> Position dict
        self.order_book = {}  # Symbol -> {bids: [...], asks: [...]}
        self.trade_history = []  # List of executed trades
        self.callbacks = {
            'order_update': [],
            'position_update': [],
            'execution': []
        }
        
        # Account data
        self.account = {
            'cash': 100000.0,
            'equity': 100000.0,
            'margin_used': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0
        }
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get market data for
        
        Returns:
        --------
        Dict[str, Any]
            Market data with price, bid, ask, etc.
        """
        # If we have a market data provider, use it
        if self.market_data_provider is not None:
            try:
                # Get latest price
                latest_price = self.market_data_provider.get_latest_price(symbol)
                
                # Generate bid/ask spread
                spread = latest_price * 0.001  # 0.1% spread
                bid = latest_price - spread / 2
                ask = latest_price + spread / 2
                
                return {
                    'symbol': symbol,
                    'price': latest_price,
                    'bid': bid,
                    'ask': ask,
                    'volume': 1000,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception:
                # Fall back to random data
                pass
        
        # Generate random data
        base_price = 100.0 + hash(symbol) % 900  # 100-1000 range
        spread = base_price * 0.001  # 0.1% spread
        
        return {
            'symbol': symbol,
            'price': base_price,
            'bid': base_price - spread / 2,
            'ask': base_price + spread / 2,
            'volume': 1000,
            'timestamp': datetime.now().isoformat()
        }
    
    def place_order(self, symbol: str, side: OrderSide, quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'DAY') -> str:
        """
        Place an order on the exchange.
        
        Parameters:
        -----------
        symbol : str
            Symbol to trade
        side : OrderSide
            Order side (BUY or SELL)
        quantity : float
            Order quantity
        order_type : OrderType
            Type of order (MARKET, LIMIT, STOP, STOP_LIMIT)
        limit_price : float, optional
            Limit price for LIMIT and STOP_LIMIT orders
        stop_price : float, optional
            Stop price for STOP and STOP_LIMIT orders
        time_in_force : str
            Time in force (DAY, GTC, IOC, FOK)
        
        Returns:
        --------
        str
            Order ID
        """
        # Generate unique order ID
        order_id = str(uuid.uuid4())
        
        # Get current market data
        market_data = self.get_market_data(symbol)
        
        # Create order
        order = {
            'id': order_id,
            'symbol': symbol,
            'side': side.value if isinstance(side, OrderSide) else side,
            'quantity': quantity,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'order_type': order_type.value if isinstance(order_type, OrderType) else order_type,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'time_in_force': time_in_force,
            'status': OrderStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'fills': [],
            'avg_fill_price': None,
            'total_commission': 0.0
        }
        
        # Store order
        self.orders[order_id] = order
        
        # Process the order (simulating some latency)
        self._process_order(order_id)
        
        return order_id
    
    def _process_order(self, order_id: str) -> None:
        """
        Process an order (internal method).
        
        Parameters:
        -----------
        order_id : str
            Order ID to process
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        symbol = order['symbol']
        
        # Update order status
        order['status'] = OrderStatus.OPEN.value
        order['updated_at'] = datetime.now().isoformat()
        
        # Notify callbacks
        self._notify_order_update(order)
        
        # Get current market data
        market_data = self.get_market_data(symbol)
        
        # Determine if we can fill the order
        can_fill = False
        fill_price = None
        
        if order['order_type'] == OrderType.MARKET.value:
            # Market orders always fill
            can_fill = True
            
            # Add slippage for market orders
            if order['side'] == OrderSide.BUY.value:
                fill_price = market_data['ask'] * (1 + self.slippage_pct)
            else:  # SELL
                fill_price = market_data['bid'] * (1 - self.slippage_pct)
                
        elif order['order_type'] == OrderType.LIMIT.value:
            # Limit orders fill if limit price is favorable
            if order['side'] == OrderSide.BUY.value:
                can_fill = order['limit_price'] >= market_data['ask']
                fill_price = min(order['limit_price'], market_data['ask'])
            else:  # SELL
                can_fill = order['limit_price'] <= market_data['bid']
                fill_price = max(order['limit_price'], market_data['bid'])
                
        elif order['order_type'] == OrderType.STOP.value:
            # Stop orders become market orders when triggered
            if order['side'] == OrderSide.BUY.value:
                can_fill = market_data['price'] >= order['stop_price']
                fill_price = market_data['ask'] * (1 + self.slippage_pct)
            else:  # SELL
                can_fill = market_data['price'] <= order['stop_price']
                fill_price = market_data['bid'] * (1 - self.slippage_pct)
                
        elif order['order_type'] == OrderType.STOP_LIMIT.value:
            # Stop-limit orders become limit orders when triggered
            if order['side'] == OrderSide.BUY.value:
                stop_triggered = market_data['price'] >= order['stop_price']
                can_fill = stop_triggered and order['limit_price'] >= market_data['ask']
                fill_price = min(order['limit_price'], market_data['ask'])
            else:  # SELL
                stop_triggered = market_data['price'] <= order['stop_price']
                can_fill = stop_triggered and order['limit_price'] <= market_data['bid']
                fill_price = max(order['limit_price'], market_data['bid'])
        
        # Fill the order if possible
        if can_fill:
            self._fill_order(order_id, fill_price)
    
    def _fill_order(self, order_id: str, fill_price: float, fill_quantity: Optional[float] = None) -> None:
        """
        Fill an order (internal method).
        
        Parameters:
        -----------
        order_id : str
            Order ID to fill
        fill_price : float
            Price at which to fill the order
        fill_quantity : float, optional
            Quantity to fill (if None, fill entire remaining quantity)
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Determine fill quantity
        if fill_quantity is None:
            fill_quantity = order['remaining_quantity']
        else:
            fill_quantity = min(fill_quantity, order['remaining_quantity'])
        
        # Calculate commission (flat fee per filled order)
        commission = 1.0  # $1 per fill
        
        # Create fill
        fill = {
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update order
        order['fills'].append(fill)
        order['filled_quantity'] += fill_quantity
        order['remaining_quantity'] -= fill_quantity
        order['total_commission'] += commission
        
        # Calculate average fill price
        total_value = sum(f['price'] * f['quantity'] for f in order['fills'])
        total_quantity = sum(f['quantity'] for f in order['fills'])
        order['avg_fill_price'] = total_value / total_quantity if total_quantity > 0 else None
        
        # Update order status
        if order['remaining_quantity'] <= 0:
            order['status'] = OrderStatus.FILLED.value
        else:
            order['status'] = OrderStatus.PARTIALLY_FILLED.value
        
        order['updated_at'] = datetime.now().isoformat()
        
        # Update position
        self._update_position(order, fill)
        
        # Update account
        self._update_account(order, fill)
        
        # Add to trade history
        trade = {
            'order_id': order_id,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': fill['timestamp']
        }
        self.trade_history.append(trade)
        
        # Notify callbacks
        self._notify_order_update(order)
        self._notify_execution(trade)
    
    def _update_position(self, order: Dict[str, Any], fill: Dict[str, Any]) -> None:
        """
        Update position after a fill (internal method).
        
        Parameters:
        -----------
        order : Dict[str, Any]
            Order that was filled
        fill : Dict[str, Any]
            Fill details
        """
        symbol = order['symbol']
        
        # Create position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': 0.0,
                'avg_price': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        # Calculate position change
        quantity_change = fill['quantity']
        if order['side'] == OrderSide.SELL.value:
            quantity_change = -quantity_change
        
        # Update position
        old_position = position.copy()
        old_quantity = position['quantity']
        new_quantity = old_quantity + quantity_change
        
        # If position direction is changing, realize PnL
        if (old_quantity > 0 and new_quantity <= 0) or (old_quantity < 0 and new_quantity >= 0):
            # Closing old position
            close_qty = min(abs(old_quantity), abs(quantity_change))
            pnl = close_qty * (fill['price'] - position['avg_price'])
            if old_quantity < 0:
                pnl = -pnl  # Reverse for short positions
            
            position['realized_pnl'] += pnl
            
            # New position from remaining quantity
            remaining_qty = abs(quantity_change) - abs(old_quantity)
            if remaining_qty > 0:
                position['quantity'] = remaining_qty if quantity_change > 0 else -remaining_qty
                position['avg_price'] = fill['price']
            else:
                position['quantity'] = 0
                position['avg_price'] = 0
        else:
            # Adding to existing position
            position['avg_price'] = (position['avg_price'] * old_quantity + 
                                  fill['price'] * quantity_change) / new_quantity
            position['quantity'] = new_quantity
        
        # Update unrealized PnL
        current_price = self.get_market_data(symbol)['price']
        position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_price'])
        if position['quantity'] < 0:
            position['unrealized_pnl'] = -position['unrealized_pnl']
        
        # Notify callbacks if position changed
        if position != old_position:
            self._notify_position_update(position)
    
    def _update_account(self, order: Dict[str, Any], fill: Dict[str, Any]) -> None:
        """
        Update account after a fill (internal method).
        
        Parameters:
        -----------
        order : Dict[str, Any]
            Order that was filled
        fill : Dict[str, Any]
            Fill details
        """
        # Calculate impact on cash
        amount = fill['quantity'] * fill['price']
        
        if order['side'] == OrderSide.BUY.value:
            self.account['cash'] -= amount
        else:  # SELL
            self.account['cash'] += amount
        
        # Subtract commission
        self.account['cash'] -= fill['commission']
        
        # Update margin used based on positions
        self.account['margin_used'] = sum(
            abs(pos['quantity'] * self.get_market_data(pos['symbol'])['price'])
            for pos in self.positions.values()
        ) / 4  # Assuming 25% margin requirement
        
        # Update unrealized PnL
        self.account['unrealized_pnl'] = sum(
            pos['unrealized_pnl'] for pos in self.positions.values()
        )
        
        # Update equity
        self.account['equity'] = self.account['cash'] + self.account['unrealized_pnl']
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Parameters:
        -----------
        order_id : str
            Order ID to cancel
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Can't cancel filled orders
        if order['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
            return False
        
        # Update order status
        order['status'] = OrderStatus.CANCELLED.value
        order['updated_at'] = datetime.now().isoformat()
        
        # Notify callbacks
        self._notify_order_update(order)
        
        return True
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.
        
        Parameters:
        -----------
        order_id : str
            Order ID to get
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            Order details if found, None otherwise
        """
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Parameters:
        -----------
        symbol : str, optional
            Symbol to filter by
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of open orders
        """
        open_statuses = [OrderStatus.PENDING.value, OrderStatus.OPEN.value, OrderStatus.PARTIALLY_FILLED.value]
        
        if symbol is None:
            return [o for o in self.orders.values() if o['status'] in open_statuses]
        
        return [o for o in self.orders.values() 
                if o['status'] in open_statuses and o['symbol'] == symbol]
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get position for
        
        Returns:
        --------
        Optional[Dict[str, Any]]
            Position details if found, None otherwise
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of all positions
        """
        return list(self.positions.values())
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
        --------
        Dict[str, Any]
            Account information
        """
        return self.account
    
    def add_callback(self, event_type: str, callback: callable) -> bool:
        """
        Add a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (order_update, position_update, execution)
        callback : callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self.callbacks:
            return False
        
        self.callbacks[event_type].append(callback)
        return True
    
    def remove_callback(self, event_type: str, callback: callable) -> bool:
        """
        Remove a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (order_update, position_update, execution)
        callback : callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self.callbacks:
            return False
        
        if callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            return True
        
        return False
    
    def _notify_order_update(self, order: Dict[str, Any]) -> None:
        """
        Notify order update callbacks.
        
        Parameters:
        -----------
        order : Dict[str, Any]
            Updated order
        """
        for callback in self.callbacks['order_update']:
            try:
                callback(order)
            except Exception as e:
                print(f"Error in order update callback: {e}")
    
    def _notify_position_update(self, position: Dict[str, Any]) -> None:
        """
        Notify position update callbacks.
        
        Parameters:
        -----------
        position : Dict[str, Any]
            Updated position
        """
        for callback in self.callbacks['position_update']:
            try:
                callback(position)
            except Exception as e:
                print(f"Error in position update callback: {e}")
    
    def _notify_execution(self, execution: Dict[str, Any]) -> None:
        """
        Notify execution callbacks.
        
        Parameters:
        -----------
        execution : Dict[str, Any]
            Execution details
        """
        for callback in self.callbacks['execution']:
            try:
                callback(execution)
            except Exception as e:
                print(f"Error in execution callback: {e}") 