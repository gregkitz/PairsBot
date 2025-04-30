"""
Mock IB Connector for testing purposes.

This module provides a mock implementation of the Interactive Brokers connector
for unit and integration testing without requiring an actual IB connection.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

class MockIBConnector:
    """
    Mock implementation of IBConnector for testing purposes.
    
    This class mimics the behavior of the IBConnector class but uses
    simulated data rather than connecting to the Interactive Brokers API.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, 
                client_id: int = 1, read_only: bool = True,
                auto_reconnect: bool = True):
        """
        Initialize the mock IB connector.
        
        Parameters:
        -----------
        host : str
            Hostname or IP address (not used in mock)
        port : int
            Port number (not used in mock)
        client_id : int
            Client ID (not used in mock)
        read_only : bool
            Whether to allow order submission or only data access
        auto_reconnect : bool
            Whether to auto-reconnect on disconnection (not used in mock)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.read_only = read_only
        self.auto_reconnect = auto_reconnect
        
        # Internal state
        self._connected = False
        self._market_data = {}
        self._contract_details = {}
        self._orders = {}
        self._callbacks = {
            'market_data': [],
            'order_status': [],
            'position': [],
            'error': []
        }
        
        # Load mock data if available
        self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data from fixture files if available."""
        fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fixtures')
        
        # Load contract details
        contract_details_path = os.path.join(fixtures_dir, 'contract_details.json')
        if os.path.exists(contract_details_path):
            with open(contract_details_path, 'r') as f:
                self._contract_details = json.load(f)
        
        # Load market data
        market_data_path = os.path.join(fixtures_dir, 'market_data.json')
        if os.path.exists(market_data_path):
            with open(market_data_path, 'r') as f:
                self._market_data = json.load(f)
    
    def connect(self) -> bool:
        """
        Simulate connecting to IB.
        
        Returns:
        --------
        bool
            True if connection is successful
        """
        self._connected = True
        return True
    
    def disconnect(self) -> bool:
        """
        Simulate disconnecting from IB.
        
        Returns:
        --------
        bool
            True if disconnect is successful
        """
        self._connected = False
        return True
    
    def is_connected(self) -> bool:
        """
        Check if the connector is connected.
        
        Returns:
        --------
        bool
            True if connected, False otherwise
        """
        return self._connected
    
    def process_pending_events(self) -> None:
        """Simulate processing pending IB events."""
        pass
    
    def get_market_data(self, symbol: str, subscribe: bool = False) -> Dict[str, Any]:
        """
        Get market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get market data for
        subscribe : bool
            Whether to subscribe to real-time updates
        
        Returns:
        --------
        Dict[str, Any]
            Market data dictionary
        """
        if not self._connected:
            return {}
        
        # Get market data from preloaded data or generate synthetic
        if symbol in self._market_data:
            data = self._market_data[symbol]
        else:
            # Generate synthetic market data
            base_price = hash(symbol) % 100 + 50  # Deterministic but different per symbol
            data = {
                'symbol': symbol,
                'last_price': base_price + np.random.normal(0, 0.5),
                'bid': base_price - 0.1,
                'ask': base_price + 0.1,
                'volume': int(np.random.randint(100, 10000)),
                'open': base_price - 1.0,
                'high': base_price + 1.0,
                'low': base_price - 1.5,
                'close': base_price,
                'time': datetime.now().isoformat()
            }
            self._market_data[symbol] = data
        
        # Simulate real-time updates if subscribing
        if subscribe:
            for callback in self._callbacks.get('market_data', []):
                try:
                    callback(symbol, data)
                except Exception as e:
                    print(f"Error in market data callback: {str(e)}")
        
        return data
    
    def get_historical_data(self, symbol: str, start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           bar_size: str = '1 day',
                           what_to_show: str = 'TRADES') -> pd.DataFrame:
        """
        Get historical market data.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get historical data for
        start_date : Union[str, datetime]
            Start date
        end_date : Union[str, datetime]
            End date
        bar_size : str
            Bar size (e.g., '1 day', '1 hour', '5 mins')
        what_to_show : str
            Type of data to retrieve
        
        Returns:
        --------
        pd.DataFrame
            Historical market data
        """
        if not self._connected:
            return pd.DataFrame()
        
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate a date range based on bar size
        if bar_size == '1 day':
            freq = 'D'
        elif bar_size == '1 hour':
            freq = 'H'
        elif bar_size == '5 mins':
            freq = '5min'
        else:
            freq = 'D'  # Default to daily
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate synthetic prices (with some randomness but deterministic based on symbol)
        base_price = hash(symbol) % 100 + 50
        np.random.seed(hash(symbol))  # Make it deterministic based on symbol
        
        # Generate synthetic price series with a slight drift
        drift = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + drift)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure High ≥ Open ≥ Low and High ≥ Close ≥ Low
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        return df
    
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
            Contract details
        """
        if not self._connected:
            return {}
        
        # Get contract details from preloaded data or generate synthetic
        if symbol in self._contract_details:
            return self._contract_details[symbol]
        
        # Extract asset class and exchange from symbol
        if symbol.startswith('ES'):
            asset_class = 'FUT'
            exchange = 'CME'
            name = 'E-mini S&P 500'
            multiplier = 50.0
        elif symbol.startswith('NQ'):
            asset_class = 'FUT'
            exchange = 'CME'
            name = 'E-mini NASDAQ 100'
            multiplier = 20.0
        elif symbol.startswith('CL'):
            asset_class = 'FUT'
            exchange = 'NYMEX'
            name = 'Crude Oil'
            multiplier = 1000.0
        elif symbol.startswith('GC'):
            asset_class = 'FUT'
            exchange = 'COMEX'
            name = 'Gold'
            multiplier = 100.0
        elif symbol.startswith('ZB'):
            asset_class = 'FUT'
            exchange = 'CBOT'
            name = '30-Year U.S. Treasury Bond'
            multiplier = 1000.0
        else:
            asset_class = 'STK'
            exchange = 'SMART'
            name = symbol
            multiplier = 1.0
        
        # Generate synthetic contract details
        contract_details = {
            'symbol': symbol,
            'name': name,
            'contract_type': asset_class,
            'exchange': exchange,
            'currency': 'USD',
            'multiplier': multiplier,
            'min_tick': 0.25 if asset_class == 'FUT' else 0.01,
            'expiration': (datetime.now() + timedelta(days=90)).strftime('%Y%m%d') if asset_class == 'FUT' else None
        }
        
        self._contract_details[symbol] = contract_details
        return contract_details
    
    def place_order(self, symbol: str, action: str, quantity: float,
                   order_type: str = 'MKT', limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
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
            Order type (e.g., 'MKT', 'LMT', 'STP')
        limit_price : float, optional
            Limit price for limit orders
        stop_price : float, optional
            Stop price for stop orders
        
        Returns:
        --------
        str
            Order ID
        """
        if not self._connected:
            return ""
        
        if self.read_only:
            raise ValueError("Cannot place orders in read-only mode")
        
        # Generate unique order ID
        order_id = f"mock_{len(self._orders) + 1}"
        
        # Create order
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_type': order_type,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'status': 'Submitted',
            'time': datetime.now().isoformat()
        }
        
        self._orders[order_id] = order
        
        # Simulate order fill in the next event loop
        # In a real implementation, this would happen asynchronously
        self._simulate_order_fill(order_id)
        
        return order_id
    
    def _simulate_order_fill(self, order_id: str) -> None:
        """
        Simulate order fill.
        
        Parameters:
        -----------
        order_id : str
            Order ID to fill
        """
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        
        # Get latest market data
        market_data = self.get_market_data(order['symbol'])
        
        # Determine fill price
        if order['order_type'] == 'MKT':
            # Market orders fill at current price with some slippage
            slippage = 0.01 * market_data.get('last_price', 100.0)
            if order['action'] == 'BUY':
                fill_price = market_data.get('last_price', 100.0) + slippage
            else:
                fill_price = market_data.get('last_price', 100.0) - slippage
        elif order['order_type'] == 'LMT':
            # Limit orders fill at limit price or better
            if order['action'] == 'BUY':
                fill_price = min(market_data.get('last_price', 100.0), order['limit_price'])
            else:
                fill_price = max(market_data.get('last_price', 100.0), order['limit_price'])
        else:
            # Default fill at market price
            fill_price = market_data.get('last_price', 100.0)
        
        # Update order status
        order['status'] = 'Filled'
        order['fill_price'] = fill_price
        order['fill_time'] = datetime.now().isoformat()
        
        # Notify callbacks
        for callback in self._callbacks.get('order_status', []):
            try:
                callback(order)
            except Exception as e:
                print(f"Error in order status callback: {str(e)}")
    
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
        if not self._connected or self.read_only:
            return False
        
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        
        # Cannot cancel filled orders
        if order['status'] == 'Filled':
            return False
        
        # Update order status
        order['status'] = 'Cancelled'
        order['cancel_time'] = datetime.now().isoformat()
        
        # Notify callbacks
        for callback in self._callbacks.get('order_status', []):
            try:
                callback(order)
            except Exception as e:
                print(f"Error in order status callback: {str(e)}")
        
        return True
    
    def add_callback(self, event_type: str, callback: callable) -> bool:
        """
        Add a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (e.g., 'market_data', 'order_status')
        callback : callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self._callbacks:
            return False
        
        self._callbacks[event_type].append(callback)
        return True
    
    def remove_callback(self, event_type: str, callback: callable) -> bool:
        """
        Remove a callback for a specific event type.
        
        Parameters:
        -----------
        event_type : str
            Event type (e.g., 'market_data', 'order_status')
        callback : callable
            Callback function
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if event_type not in self._callbacks:
            return False
        
        if callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            return True
        
        return False


def contract_to_symbol(contract: Any) -> str:
    """
    Convert IB contract to symbol string.
    
    Parameters:
    -----------
    contract : Any
        IB contract object
    
    Returns:
    --------
    str
        Symbol string
    """
    # In mock implementation, just return a string representation
    if hasattr(contract, 'symbol'):
        return contract.symbol
    elif isinstance(contract, dict) and 'symbol' in contract:
        return contract['symbol']
    else:
        return str(contract)


def symbol_to_contract(symbol: str) -> Any:
    """
    Convert symbol string to IB contract.
    
    Parameters:
    -----------
    symbol : str
        Symbol string
    
    Returns:
    --------
    Any
        Mock IB contract object
    """
    # In mock implementation, just return a dict representing a contract
    return {'symbol': symbol} 