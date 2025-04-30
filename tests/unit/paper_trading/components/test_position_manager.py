"""
Unit tests for the PositionManager component.

This module contains tests to verify the functionality of the PositionManager component.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.paper_trading.components.position_manager import PositionManager

# Constants for testing
TEST_PAIR_ID = 'GC_SI'
TEST_LEG1 = 'GC'
TEST_LEG2 = 'SI'


@pytest.fixture
def mock_paper_trader():
    """Fixture providing a mock paper trader."""
    paper_trader = Mock()
    
    # Mock the get_position method
    paper_trader.get_position.return_value = {
        'quantity': 1,
        'entry_price': 100.0,
        'current_price': 105.0,
        'unrealized_pnl': 5.0
    }
    
    # Mock the place_order method
    paper_trader.place_order.return_value = "order_123"
    
    # Mock the get_account_value method
    paper_trader.get_account_value.return_value = 100000.0
    
    # Mock the get_last_price method
    paper_trader.get_last_price.return_value = 100.0
    
    return paper_trader


@pytest.fixture
def test_pair_config():
    """Fixture providing a test pair configuration."""
    return {
        'pair_id': TEST_PAIR_ID,
        'leg1': TEST_LEG1,
        'leg2': TEST_LEG2,
        'hedge_ratio': 1.5,
        'leg1_multiplier': 1.0,
        'leg2_multiplier': 1.0,
        'z_entry': 2.0,
        'z_exit': 0.5,
        'stop_loss_z': 3.0,
        'max_holding_period': 180,
        'min_correlation': 0.5,
        'half_life': 24
    }


@pytest.fixture
def position_manager(mock_paper_trader):
    """Fixture providing a PositionManager instance with mocked dependencies."""
    config = {
        'default_position_size': 0.1,
        'risk_per_trade': 0.02
    }
    return PositionManager(paper_trader=mock_paper_trader, config=config)


@pytest.fixture
def position_manager_with_pair(position_manager, test_pair_config):
    """Fixture providing a PositionManager with a test pair added."""
    position_manager.add_pair(test_pair_config)
    return position_manager


@pytest.fixture
def test_market_data():
    """Fixture providing test market data."""
    # Create sample market data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=10), 
                         periods=100, freq='5min')
    
    data = pd.DataFrame({
        'timestamp': dates,
        TEST_LEG1: np.linspace(1800, 2000, 100),  # Gold prices
        TEST_LEG2: np.linspace(20, 25, 100),      # Silver prices
        'spread': np.linspace(2, 3, 100) + np.sin(np.linspace(0, 10, 100)) * 0.2,
        'zscore': np.sin(np.linspace(0, 10, 100)) * 2.0
    })
    
    data.set_index('timestamp', inplace=True)
    return data


class TestPositionManager:
    """Tests for the PositionManager class."""
    
    def test_initialization(self, position_manager, mock_paper_trader):
        """Test that the PositionManager initializes correctly."""
        assert position_manager.paper_trader == mock_paper_trader
        assert isinstance(position_manager.config, dict)
        assert position_manager.positions == {}
        assert position_manager.position_history == []
        assert position_manager.pair_configs == {}
    
    def test_add_pair(self, position_manager, test_pair_config):
        """Test adding a trading pair to the manager."""
        result = position_manager.add_pair(test_pair_config)
        
        assert result is True
        assert TEST_PAIR_ID in position_manager.pair_configs
        assert position_manager.pair_configs[TEST_PAIR_ID] == test_pair_config
    
    def test_add_pair_without_id(self, position_manager):
        """Test adding a pair without an ID fails."""
        result = position_manager.add_pair({})
        
        assert result is False
        assert len(position_manager.pair_configs) == 0
    
    def test_get_positions(self, position_manager_with_pair):
        """Test getting all positions."""
        # Initially there are no positions
        positions = position_manager_with_pair.get_positions()
        assert positions == {}
        
        # Add a position manually for testing
        test_position = {'pair_id': TEST_PAIR_ID, 'direction': 1}
        position_manager_with_pair.positions[TEST_PAIR_ID] = test_position
        
        # Now get_positions should return the test position
        positions = position_manager_with_pair.get_positions()
        assert TEST_PAIR_ID in positions
        assert positions[TEST_PAIR_ID] == test_position
        
        # Ensure the returned value is a copy
        positions[TEST_PAIR_ID] = "modified"
        assert position_manager_with_pair.positions[TEST_PAIR_ID] == test_position
    
    def test_get_position(self, position_manager_with_pair):
        """Test getting a specific position."""
        # Initially there is no position
        position = position_manager_with_pair.get_position(TEST_PAIR_ID)
        assert position is None
        
        # Add a position manually for testing
        test_position = {'pair_id': TEST_PAIR_ID, 'direction': 1}
        position_manager_with_pair.positions[TEST_PAIR_ID] = test_position
        
        # Now get_position should return the test position
        position = position_manager_with_pair.get_position(TEST_PAIR_ID)
        assert position == test_position
    
    def test_get_position_history(self, position_manager_with_pair):
        """Test getting position history."""
        # Initially there is no history
        history = position_manager_with_pair.get_position_history()
        assert history == []
        
        # Add a position to history manually for testing
        test_position = {'pair_id': TEST_PAIR_ID, 'direction': 1, 'status': 'closed'}
        position_manager_with_pair.position_history.append(test_position)
        
        # Now get_position_history should return the test position
        history = position_manager_with_pair.get_position_history()
        assert len(history) == 1
        assert history[0] == test_position
        
        # Ensure the returned value is a copy
        history[0] = "modified"
        assert position_manager_with_pair.position_history[0] == test_position
    
    def test_execute_signals_new_position(self, position_manager_with_pair, monkeypatch):
        """Test executing signals for a new position."""
        # Create test signals
        signals = pd.Series([1], index=[datetime.now()])
        
        # Mock the _enter_position method
        mock_enter = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_enter_position", mock_enter)
        
        # Execute signals
        result = position_manager_with_pair.execute_signals(TEST_PAIR_ID, signals)
        
        # Check result
        assert result["executed"] == 1
        assert result["errors"] == 0
        mock_enter.assert_called_once_with(TEST_PAIR_ID, 1, position_manager_with_pair.pair_configs[TEST_PAIR_ID])
    
    def test_execute_signals_exit_position(self, position_manager_with_pair, monkeypatch):
        """Test executing signals to exit a position."""
        # Create test signals
        signals = pd.Series([0], index=[datetime.now()])
        
        # Add a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1
        }
        
        # Mock the _exit_position method
        mock_exit = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        
        # Execute signals
        result = position_manager_with_pair.execute_signals(TEST_PAIR_ID, signals)
        
        # Check result
        assert result["executed"] == 1
        assert result["errors"] == 0
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "signal_exit")
    
    def test_execute_signals_reversal(self, position_manager_with_pair, monkeypatch):
        """Test executing signals to reverse a position."""
        # Create test signals
        signals = pd.Series([-1], index=[datetime.now()])
        
        # Add a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1
        }
        
        # Mock the _exit_position and _enter_position methods
        mock_exit = MagicMock(return_value=True)
        mock_enter = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        monkeypatch.setattr(position_manager_with_pair, "_enter_position", mock_enter)
        
        # Execute signals
        result = position_manager_with_pair.execute_signals(TEST_PAIR_ID, signals)
        
        # Check result
        assert result["executed"] == 2
        assert result["errors"] == 0
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "signal_reversal")
        mock_enter.assert_called_once_with(TEST_PAIR_ID, -1, position_manager_with_pair.pair_configs[TEST_PAIR_ID])
    
    def test_enter_position(self, position_manager_with_pair, mock_paper_trader):
        """Test entering a position."""
        # Enter a long position
        result = position_manager_with_pair._enter_position(
            TEST_PAIR_ID, 
            1, 
            position_manager_with_pair.pair_configs[TEST_PAIR_ID]
        )
        
        # Check result
        assert result is True
        assert TEST_PAIR_ID in position_manager_with_pair.positions
        assert position_manager_with_pair.positions[TEST_PAIR_ID]['direction'] == 1
        assert position_manager_with_pair.positions[TEST_PAIR_ID]['status'] == 'open'
        
        # Check orders were placed
        assert mock_paper_trader.place_order.call_count == 2
    
    def test_exit_position(self, position_manager_with_pair, mock_paper_trader):
        """Test exiting a position."""
        # Create a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now(),
            'leg1': {
                'symbol': TEST_LEG1,
                'action': 'BUY',
                'quantity': 1.0,
                'order_id': 'order_123'
            },
            'leg2': {
                'symbol': TEST_LEG2,
                'action': 'SELL',
                'quantity': 1.5,
                'order_id': 'order_456'
            },
            'status': 'open'
        }
        
        # Exit the position
        result = position_manager_with_pair._exit_position(TEST_PAIR_ID, "test")
        
        # Check result
        assert result is True
        assert TEST_PAIR_ID not in position_manager_with_pair.positions
        assert len(position_manager_with_pair.position_history) == 1
        assert position_manager_with_pair.position_history[0]['status'] == 'closed'
        assert position_manager_with_pair.position_history[0]['exit_reason'] == 'test'
        
        # Check orders were placed
        assert mock_paper_trader.place_order.call_count == 2
    
    def test_check_stop_losses(self, position_manager_with_pair, test_market_data, monkeypatch):
        """Test checking stop losses."""
        # Create a test position (long)
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        # Mock _exit_position
        mock_exit = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        
        # Test with z-score at -2.0 (not triggering stop loss)
        data = test_market_data.copy()
        data.loc[data.index[-1], 'zscore'] = -2.0
        result = position_manager_with_pair.check_stop_losses(TEST_PAIR_ID, data)
        assert result is False
        mock_exit.assert_not_called()
        
        # Test with z-score at -3.5 (triggering stop loss)
        data = test_market_data.copy()
        data.loc[data.index[-1], 'zscore'] = -3.5
        result = position_manager_with_pair.check_stop_losses(TEST_PAIR_ID, data)
        assert result is True
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "stop_loss")
    
    def test_check_take_profits(self, position_manager_with_pair, test_market_data, monkeypatch):
        """Test checking take profits."""
        # Create a test position (long)
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now(),
            'status': 'open'
        }
        
        # Mock _exit_position
        mock_exit = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        
        # Test with z-score at 1.0 (not triggering take profit)
        data = test_market_data.copy()
        data.loc[data.index[-1], 'zscore'] = 1.0
        result = position_manager_with_pair.check_take_profits(TEST_PAIR_ID, data)
        assert result is False
        mock_exit.assert_not_called()
        
        # Test with z-score at 0.3 (triggering take profit)
        data = test_market_data.copy()
        data.loc[data.index[-1], 'zscore'] = 0.3
        result = position_manager_with_pair.check_take_profits(TEST_PAIR_ID, data)
        assert result is True
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "take_profit")
    
    def test_check_holding_limits(self, position_manager_with_pair, monkeypatch):
        """Test checking holding time limits."""
        # Create a test position with entry time 3 hours ago
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now() - timedelta(hours=3),
            'status': 'open'
        }
        
        # Mock _exit_position
        mock_exit = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        
        # Check holding limits (should trigger with max_holding_period = 180 min = 3 hours)
        result = position_manager_with_pair.check_holding_limits(TEST_PAIR_ID)
        assert result is True
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "max_holding_period")
    
    def test_adjust_position_size(self, position_manager_with_pair):
        """Test adjusting position size based on volatility."""
        # Create a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now(),
            'original_size': 1.0,
            'size': 1.0,
            'status': 'open'
        }
        
        # Test with volatility ratio of 2.0 (should adjust)
        result = position_manager_with_pair.adjust_position_size(
            TEST_PAIR_ID, 
            {'current_volatility': 0.2, 'baseline_volatility': 0.1}
        )
        
        assert result is True
        assert position_manager_with_pair.positions[TEST_PAIR_ID]['size'] == 0.5  # 1.0 / 2.0
        assert len(position_manager_with_pair.positions[TEST_PAIR_ID]['size_adjustments']) == 1
        
        # Test with small volatility change (should not adjust)
        result = position_manager_with_pair.adjust_position_size(
            TEST_PAIR_ID, 
            {'current_volatility': 0.105, 'baseline_volatility': 0.1}
        )
        
        assert result is False  # No adjustment needed
    
    def test_track_position_performance(self, position_manager_with_pair, test_market_data, mock_paper_trader):
        """Test tracking position performance metrics."""
        # Create a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now() - timedelta(hours=1),
            'entry_spread': 2.5,
            'entry_zscore': 2.0,
            'leg1': {'symbol': TEST_LEG1},
            'leg2': {'symbol': TEST_LEG2},
            'status': 'open'
        }
        
        # Track performance
        metrics = position_manager_with_pair.track_position_performance(TEST_PAIR_ID, test_market_data)
        
        # Check metrics
        assert metrics['pair_id'] == TEST_PAIR_ID
        assert metrics['leg1_pnl'] == 5.0
        assert metrics['leg2_pnl'] == 5.0
        assert metrics['total_pnl'] == 10.0
        assert 'holding_time_minutes' in metrics
        assert 'current_spread' in metrics
        assert 'current_zscore' in metrics
        
        # Check position was updated
        assert position_manager_with_pair.positions[TEST_PAIR_ID]['current_pnl'] == 10.0
        assert 'performance_history' in position_manager_with_pair.positions[TEST_PAIR_ID]
        assert len(position_manager_with_pair.positions[TEST_PAIR_ID]['performance_history']) == 1
    
    def test_analyze_position_risk(self, position_manager_with_pair, test_market_data):
        """Test analyzing position risk metrics."""
        # Create a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now() - timedelta(hours=1),
            'entry_zscore': 2.0,
            'holding_time_minutes': 60,
            'status': 'open'
        }
        
        # Set up test data with specific z-score
        data = test_market_data.copy()
        data.loc[data.index[-1], 'zscore'] = 1.0
        
        # Analyze risk
        metrics = position_manager_with_pair.analyze_position_risk(TEST_PAIR_ID, data)
        
        # Check metrics
        assert metrics['pair_id'] == TEST_PAIR_ID
        assert metrics['direction'] == 1
        assert metrics['current_zscore'] == 1.0
        assert metrics['distance_to_stop_loss'] == 4.0  # |1.0 - (-3.0)|
        assert metrics['distance_to_take_profit'] == 0.5  # |1.0 - 0.5|
        assert metrics['risk_reward_ratio'] == 0.125  # 0.5/4.0
        assert 'time_decay_factor' in metrics
        
        # Check position was updated
        assert 'risk_metrics' in position_manager_with_pair.positions[TEST_PAIR_ID]
    
    def test_check_correlation_breakdown(self, position_manager_with_pair, test_market_data, monkeypatch):
        """Test checking for correlation breakdown."""
        # Create a test position
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now(),
            'leg1': {'symbol': TEST_LEG1},
            'leg2': {'symbol': TEST_LEG2},
            'status': 'open'
        }
        
        # Mock _exit_position
        mock_exit = MagicMock(return_value=True)
        monkeypatch.setattr(position_manager_with_pair, "_exit_position", mock_exit)
        
        # Check correlation with normal data (should be high correlation)
        result = position_manager_with_pair.check_correlation_breakdown(TEST_PAIR_ID, test_market_data)
        assert result is False
        mock_exit.assert_not_called()
        
        # Create data with low correlation
        low_corr_data = test_market_data.copy()
        low_corr_data[TEST_LEG2] = np.random.randn(len(low_corr_data)) * 10 + 22  # Uncorrelated data
        
        # Check correlation with uncorrelated data
        result = position_manager_with_pair.check_correlation_breakdown(TEST_PAIR_ID, low_corr_data)
        assert result is True
        mock_exit.assert_called_once_with(TEST_PAIR_ID, "correlation_breakdown")
        
        # Check that position was updated with correlation
        assert 'current_correlation' in position_manager_with_pair.positions[TEST_PAIR_ID]
    
    def test_get_position_summary(self, position_manager_with_pair):
        """Test getting position summary."""
        # Create test positions
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now() - timedelta(hours=2),
            'holding_time_minutes': 120,
            'current_pnl': 50.0,
            'leg1': {'quantity': 1, 'price': 1800},
            'leg2': {'quantity': -1.5, 'price': 22},
            'status': 'open'
        }
        
        position_manager_with_pair.positions['TEST_PAIR_2'] = {
            'pair_id': 'TEST_PAIR_2',
            'direction': -1,
            'entry_time': datetime.now() - timedelta(hours=1),
            'holding_time_minutes': 60,
            'current_pnl': -20.0,
            'leg1': {'quantity': -1, 'price': 100},
            'leg2': {'quantity': 2, 'price': 50},
            'status': 'open'
        }
        
        # Get summary
        summary = position_manager_with_pair.get_position_summary()
        
        # Check summary
        assert summary['position_count'] == 2
        assert summary['unrealized_pnl'] == 30.0  # 50.0 - 20.0
        assert summary['total_exposure'] > 0
        assert TEST_PAIR_ID in summary['positions']
        assert 'TEST_PAIR_2' in summary['positions']
        assert summary['positions'][TEST_PAIR_ID]['direction'] == 1
        assert summary['positions']['TEST_PAIR_2']['direction'] == -1
    
    def test_monitor_all_positions(self, position_manager_with_pair, test_market_data, monkeypatch):
        """Test monitoring all positions."""
        # Create test positions
        position_manager_with_pair.positions[TEST_PAIR_ID] = {
            'pair_id': TEST_PAIR_ID,
            'direction': 1,
            'entry_time': datetime.now() - timedelta(hours=2),
            'leg1': {'symbol': TEST_LEG1},
            'leg2': {'symbol': TEST_LEG2},
            'status': 'open'
        }
        
        # Mock the various check functions
        mock_track = MagicMock(return_value={'total_pnl': 50.0})
        mock_analyze = MagicMock(return_value={'risk_reward_ratio': 0.5})
        mock_stop_loss = MagicMock(return_value=False)
        mock_take_profit = MagicMock(return_value=False)
        mock_holding = MagicMock(return_value=False)
        mock_correlation = MagicMock(return_value=False)
        
        monkeypatch.setattr(position_manager_with_pair, "track_position_performance", mock_track)
        monkeypatch.setattr(position_manager_with_pair, "analyze_position_risk", mock_analyze)
        monkeypatch.setattr(position_manager_with_pair, "check_stop_losses", mock_stop_loss)
        monkeypatch.setattr(position_manager_with_pair, "check_take_profits", mock_take_profit)
        monkeypatch.setattr(position_manager_with_pair, "check_holding_limits", mock_holding)
        monkeypatch.setattr(position_manager_with_pair, "check_correlation_breakdown", mock_correlation)
        
        # Create market data dictionary
        market_data = {TEST_PAIR_ID: test_market_data}
        
        # Monitor positions
        results = position_manager_with_pair.monitor_all_positions(market_data)
        
        # Check results
        assert results['total_positions'] == 1
        assert results['positions_checked'] == 1
        assert results['stop_losses_triggered'] == 0
        assert results['take_profits_triggered'] == 0
        assert results['holding_limits_triggered'] == 0
        assert results['correlation_breakdowns'] == 0
        assert TEST_PAIR_ID in results['position_metrics']
        assert 'performance' in results['position_metrics'][TEST_PAIR_ID]
        assert 'risk' in results['position_metrics'][TEST_PAIR_ID]
        assert 'checks' in results['position_metrics'][TEST_PAIR_ID]
        
        # Verify all check functions were called
        mock_track.assert_called_once()
        mock_analyze.assert_called_once()
        mock_stop_loss.assert_called_once()
        mock_take_profit.assert_called_once()
        mock_holding.assert_called_once()
        mock_correlation.assert_called_once() 