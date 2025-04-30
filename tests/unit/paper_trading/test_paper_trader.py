"""
Unit tests for the PaperTrader class.

This module contains tests for the PaperTrader class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil

from src.paper_trading.paper_trader import PaperTrader
from tests.mocks.mock_ib_connector import MockIBConnector

@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for paper trading data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_ib_connector():
    """Fixture providing a mock IB connector."""
    connector = MockIBConnector()
    connector.connect()
    return connector

@pytest.fixture
def paper_trader(temp_dir, monkeypatch):
    """Fixture providing a PaperTrader instance with mocked dependencies."""
    # Mock the IBConnector import in paper_trader module
    monkeypatch.setattr("src.paper_trading.paper_trader.IBConnector", MockIBConnector)
    
    trader = PaperTrader(
        initial_capital=100000.0,
        ib_host='127.0.0.1',
        ib_port=7497,
        ib_client_id=1,
        data_directory=temp_dir,
        commission_model='flat',
        slippage_model='fixed',
        slippage_factor=0.0001
    )
    return trader

class TestPaperTrader:
    """Test class for PaperTrader."""
    
    def test_initialization(self, paper_trader, temp_dir):
        """Test that the paper trader is initialized correctly."""
        assert paper_trader.initial_capital == 100000.0
        assert paper_trader.commission_model == 'flat'
        assert paper_trader.slippage_model == 'fixed'
        assert paper_trader.slippage_factor == 0.0001
        assert paper_trader.data_directory == temp_dir
        
        # Check account initialization
        assert paper_trader._account['cash'] == 100000.0
        assert paper_trader._account['equity'] == 100000.0
        assert paper_trader._account['margin_used'] == 0.0
        assert paper_trader._account['pnl_day'] == 0.0
        assert paper_trader._account['unrealized_pnl'] == 0.0
    
    def test_start_stop(self, paper_trader):
        """Test starting and stopping the paper trader."""
        # Start paper trader
        result = paper_trader.start()
        assert result is True
        assert paper_trader.is_running() is True
        
        # Stop paper trader
        result = paper_trader.stop()
        assert result is True
        assert paper_trader.is_running() is False
    
    def test_market_data_subscription(self, paper_trader):
        """Test market data subscription."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        result = paper_trader.subscribe_market_data('ESM23')
        assert result is True
        
        # Get market data
        market_data = paper_trader.get_market_data('ESM23')
        assert isinstance(market_data, dict)
        assert 'last_price' in market_data
        assert 'bid' in market_data
        assert 'ask' in market_data
        assert 'volume' in market_data
        
        # Unsubscribe from market data
        result = paper_trader.unsubscribe_market_data('ESM23')
        assert result is True
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_order_placement(self, paper_trader):
        """Test order placement and execution."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Place a market order
        order_id = paper_trader.place_order(
            symbol='ESM23',
            action='BUY',
            quantity=1.0,
            order_type='MKT'
        )
        
        assert order_id is not None
        
        # Get order
        order = paper_trader.get_order(order_id)
        assert order['symbol'] == 'ESM23'
        assert order['action'] == 'BUY'
        assert order['quantity'] == 1.0
        assert order['order_type'] == 'MKT'
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Check order status
        order = paper_trader.get_order(order_id)
        assert order['status'] in ['FILLED', 'PARTIALLY_FILLED', 'PENDING']
        
        # Check position
        if order['status'] == 'FILLED':
            positions = paper_trader._positions
            assert 'ESM23' in positions
            assert positions['ESM23']['quantity'] == 1.0
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_limit_order(self, paper_trader):
        """Test limit order placement."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Get current price
        price = paper_trader.get_price('ESM23')
        assert price is not None
        
        # Place a limit order below market price (should fill)
        limit_price = price * 1.05  # 5% above market price for buy
        order_id = paper_trader.place_order(
            symbol='ESM23',
            action='BUY',
            quantity=1.0,
            order_type='LMT',
            limit_price=limit_price
        )
        
        assert order_id is not None
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Check order status
        order = paper_trader.get_order(order_id)
        assert order['status'] in ['FILLED', 'PARTIALLY_FILLED', 'PENDING']
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_stop_order(self, paper_trader):
        """Test stop order placement."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Get current price
        price = paper_trader.get_price('ESM23')
        assert price is not None
        
        # Place a stop order above market price (should trigger for buy)
        stop_price = price * 0.95  # 5% below market price for sell
        order_id = paper_trader.place_order(
            symbol='ESM23',
            action='SELL',
            quantity=1.0,
            order_type='STP',
            stop_price=stop_price
        )
        
        assert order_id is not None
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Check order status
        order = paper_trader.get_order(order_id)
        assert order['status'] in ['FILLED', 'PARTIALLY_FILLED', 'PENDING']
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_bracket_order(self, paper_trader):
        """Test bracket order placement."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Get current price
        price = paper_trader.get_price('ESM23')
        assert price is not None
        
        # Place a bracket order
        bracket_orders = paper_trader.place_bracket_order(
            symbol='ESM23',
            action='BUY',
            quantity=1.0,
            entry_order_type='MKT',
            profit_price=price * 1.05,  # 5% profit
            stop_price=price * 0.95  # 5% stop loss
        )
        
        assert 'entry' in bracket_orders
        assert bracket_orders['entry'] is not None
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Check entry order status
        entry_order = paper_trader.get_order(bracket_orders['entry'])
        assert entry_order['status'] in ['FILLED', 'PARTIALLY_FILLED', 'PENDING']
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_account_update(self, paper_trader):
        """Test account update after trades."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Get initial equity
        initial_equity = paper_trader._account['equity']
        
        # Place a market order
        order_id = paper_trader.place_order(
            symbol='ESM23',
            action='BUY',
            quantity=1.0,
            order_type='MKT'
        )
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Check if order is filled
        order = paper_trader.get_order(order_id)
        if order['status'] == 'FILLED':
            # Check that equity has changed
            current_equity = paper_trader._account['equity']
            assert current_equity != initial_equity
            
            # Check that position is tracked
            assert 'ESM23' in paper_trader._positions
            
            # Check that unrealized PnL is calculated
            assert 'unrealized_pnl' in paper_trader._account
        
        # Stop paper trader
        paper_trader.stop()
    
    def test_save_and_load_data(self, paper_trader, temp_dir):
        """Test saving and loading paper trading data."""
        # Start paper trader
        paper_trader.start()
        
        # Subscribe to market data
        paper_trader.subscribe_market_data('ESM23')
        
        # Place a market order
        order_id = paper_trader.place_order(
            symbol='ESM23',
            action='BUY',
            quantity=1.0,
            order_type='MKT'
        )
        
        # Wait for order processing
        import time
        time.sleep(0.5)
        
        # Stop paper trader (should save data)
        paper_trader.stop()
        
        # Check that data files were created
        account_file = os.path.join(temp_dir, 'account.json')
        positions_file = os.path.join(temp_dir, 'positions.json')
        trades_file = os.path.join(temp_dir, 'trades.json')
        
        assert os.path.exists(account_file)
        assert os.path.exists(positions_file)
        assert os.path.exists(trades_file)
        
        # Create new paper trader with same data directory
        new_trader = PaperTrader(
            initial_capital=100000.0,
            ib_host='127.0.0.1',
            ib_port=7497,
            ib_client_id=1,
            data_directory=temp_dir
        )
        
        # Check that data was loaded
        if 'ESM23' in paper_trader._positions:
            assert 'ESM23' in new_trader._positions
            assert new_trader._positions['ESM23']['quantity'] == paper_trader._positions['ESM23']['quantity'] 