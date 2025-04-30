"""
Integration tests for the PositionManager with IntradayMLPaperTrader.

This module tests the integration between the PositionManager component
and the IntradayMLPaperTrader class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.paper_trading.intraday_ml_paper_trader import IntradayMLPaperTrader
from src.paper_trading.components.position_manager import PositionManager


@pytest.fixture
def mock_ml_config():
    """Fixture providing a mock ML configuration."""
    return {
        'trading_plan': {
            'trading_parameters': {
                'position_size': 0.1,
                'max_positions': 3
            }
        },
        'pairs': [
            {
                'symbol1': 'GC',
                'symbol2': 'SI',
                'hedge_ratio': 1.5,
                'z_entry': 2.0,
                'z_exit': 0.5,
                'stop_loss_z': 3.0,
                'max_holding_period': 180
            }
        ]
    }


@pytest.fixture
def mock_paper_trader():
    """Fixture providing a mock paper trader."""
    paper_trader = Mock()
    
    # Mock methods
    paper_trader.get_position.return_value = None
    paper_trader.get_account_value.return_value = 100000.0
    paper_trader.get_last_price.return_value = 100.0
    paper_trader.place_order.return_value = "order_123"
    
    return paper_trader


@pytest.fixture
def mock_intraday_ml_system():
    """Fixture providing a mock ML system."""
    ml_system = Mock()
    
    # Mock data processing
    ml_system.process_data.return_value = pd.DataFrame({
        'zscore': np.sin(np.linspace(0, 10, 100)) * 2.0,
        'signal': np.zeros(100),
        'quality': np.ones(100) * 0.8
    })
    
    # Mock signal enhancement
    ml_system.enhance_signals.return_value = (
        pd.DataFrame({'signal': [1]}),
        {'confidence': 0.85}
    )
    
    return ml_system


@pytest.fixture
def intraday_trader(mock_paper_trader, mock_ml_config, mock_intraday_ml_system):
    """Fixture providing an IntradayMLPaperTrader with mocked dependencies."""
    with patch('src.paper_trading.intraday_ml_paper_trader.PaperTrader', return_value=mock_paper_trader):
        with patch('src.paper_trading.intraday_ml_paper_trader.IntradayMLSystem', return_value=mock_intraday_ml_system):
            trader = IntradayMLPaperTrader(
                initial_capital=100000.0,
                ib_host='localhost',
                ib_port=7497,
                ml_config=mock_ml_config,
                test_mode=True
            )
            
            # Mock the _load_pairs_from_config method
            trader._load_pairs_from_config = MagicMock(return_value=mock_ml_config['pairs'])
            
            # Setup the trader
            trader.setup()
            
            return trader


class TestPositionManagerIntegration:
    """Integration tests for PositionManager with IntradayMLPaperTrader."""
    
    def test_position_manager_initialization(self, intraday_trader):
        """Test that the PositionManager is properly initialized in the IntradayMLPaperTrader."""
        # Verify PositionManager is initialized
        assert intraday_trader.position_manager is not None
        assert isinstance(intraday_trader.position_manager, PositionManager)
        
        # Verify PositionManager has the paper trader reference
        assert intraday_trader.position_manager.paper_trader == intraday_trader.paper_trader
    
    def test_trading_pairs_setup(self, intraday_trader, mock_ml_config):
        """Test that trading pairs are properly set up in the PositionManager."""
        # Get the pair ID
        pair = mock_ml_config['pairs'][0]
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        
        # Verify pair is in PositionManager
        assert pair_id in intraday_trader.position_manager.pair_configs
        
        # Verify pair configuration is correct
        pair_config = intraday_trader.position_manager.pair_configs[pair_id]
        assert pair_config['pair_id'] == pair_id
        assert pair_config['leg1'] == pair['symbol1']
        assert pair_config['leg2'] == pair['symbol2']
        assert pair_config['hedge_ratio'] == pair['hedge_ratio']
        assert pair_config['z_entry'] == pair['z_entry']
        assert pair_config['z_exit'] == pair['z_exit']
        assert pair_config['stop_loss_z'] == pair['stop_loss_z']
        assert pair_config['max_holding_period'] == pair['max_holding_period']
    
    def test_position_functions_delegation(self, intraday_trader, mock_ml_config):
        """Test that position-related functions delegate to the PositionManager."""
        # Create a spy on the PositionManager methods
        get_position_spy = MagicMock(wraps=intraday_trader.position_manager.get_position)
        exit_position_spy = MagicMock(wraps=intraday_trader.position_manager._exit_position)
        enter_position_spy = MagicMock(wraps=intraday_trader.position_manager._enter_position)
        
        # Replace the methods with the spies
        intraday_trader.position_manager.get_position = get_position_spy
        intraday_trader.position_manager._exit_position = exit_position_spy
        intraday_trader.position_manager._enter_position = enter_position_spy
        
        # Get the pair
        pair = mock_ml_config['pairs'][0]
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        
        # Call the trader's position methods
        intraday_trader._get_current_pair_position(pair)
        intraday_trader._close_pair_position(pair)
        intraday_trader._open_pair_position(pair, 1)
        
        # Verify the PositionManager methods were called
        get_position_spy.assert_called_once_with(pair_id)
        exit_position_spy.assert_called_once_with(pair_id, "manual")
        
        # Verify the _enter_position call
        enter_position_spy.assert_called_once()
        args, kwargs = enter_position_spy.call_args
        assert args[0] == pair_id
        assert args[1] == 1
        assert args[2] == intraday_trader.position_manager.pair_configs[pair_id]
    
    def test_apply_trading_signals(self, intraday_trader, mock_ml_config):
        """Test that applying trading signals delegates to the PositionManager."""
        # Create a spy on the PositionManager execute_signals method
        execute_signals_spy = MagicMock(return_value={"executed": 1, "errors": 0})
        intraday_trader.position_manager.execute_signals = execute_signals_spy
        
        # Create test signals
        pair = mock_ml_config['pairs'][0]
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        signals = {
            pair_id: pd.DataFrame({'signal': [1]})
        }
        
        # Apply the signals
        intraday_trader._apply_trading_signals(signals)
        
        # Verify the PositionManager method was called
        execute_signals_spy.assert_called_once()
        args, kwargs = execute_signals_spy.call_args
        assert args[0] == pair_id
        pd.testing.assert_frame_equal(args[1], signals[pair_id])
    
    def test_monitor_positions(self, intraday_trader):
        """Test that position monitoring delegates to the PositionManager."""
        # Create a spy on the PositionManager monitor_all_positions method
        monitor_spy = MagicMock(return_value={
            "stop_losses_triggered": 0,
            "take_profits_triggered": 0,
            "holding_limits_triggered": 0,
            "correlation_breakdowns": 0,
            "positions_checked": 1,
            "total_positions": 1,
            "position_metrics": {}
        })
        intraday_trader.position_manager.monitor_all_positions = monitor_spy
        
        # Create test market data
        market_data = {
            "GC_SI": pd.DataFrame({
                'zscore': [1.0],
                'spread': [2.5],
                'GC': [1800],
                'SI': [25]
            })
        }
        
        # Monitor positions
        intraday_trader._monitor_positions(market_data)
        
        # Verify the PositionManager method was called
        monitor_spy.assert_called_once_with(market_data) 