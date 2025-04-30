"""
Unit tests for basic risk controls in the MVTS approach.

This module contains tests for the stop-loss and maximum holding period
functionality that are essential risk controls for the Minimal Viable Trading
Strategy (MVTS) implementation.

These tests verify that the risk management components properly enforce:
- Stop-loss limits to prevent excessive losses
- Maximum holding period to prevent position staleness
- Daily loss limits for proper risk management
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.risk_management.risk_manager import RiskManager


@pytest.fixture
def basic_risk_manager():
    """Create a basic risk manager with MVTS configuration."""
    return RiskManager(
        max_zscore=3.0,             # z-score based stop loss
        max_drawdown=0.02,          # 2% max drawdown as percentage of position
        max_holding_period=10       # 10-day maximum holding period
    )


@pytest.fixture
def mock_trade_data():
    """Create mock trade data for testing risk controls."""
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Generate mock z-scores (gradually increasing)
    zscore_data = np.linspace(-0.5, 4.0, 20)  # Crosses the 3.0 threshold
    
    # Generate mock drawdown data (gradually increasing)
    drawdown_data = np.linspace(0.005, 0.03, 20)  # Crosses the 0.02 threshold
    
    # Generate mock trades with different holding periods
    trades = []
    
    # Trade 1: Exceeds max holding period
    trades.append({
        'trade_id': 'T001',
        'entry_date': dates[0],
        'current_date': dates[11],  # 11 days > 10 max holding period
        'position': 1,
        'entry_price': 100.0,
        'current_price': 101.5,
        'pnl': 150.0,
        'pnl_pct': 0.015,
        'max_drawdown': 0.01,
        'current_zscore': 1.5
    })
    
    # Trade 2: Exceeds max z-score
    trades.append({
        'trade_id': 'T002',
        'entry_date': dates[5],
        'current_date': dates[10],  # 5 days holding
        'position': -1,
        'entry_price': 50.0,
        'current_price': 49.0,
        'pnl': 100.0,
        'pnl_pct': 0.02,
        'max_drawdown': 0.01,
        'current_zscore': 3.5      # > 3.0 max zscore
    })
    
    # Trade 3: Exceeds max drawdown
    trades.append({
        'trade_id': 'T003',
        'entry_date': dates[8],
        'current_date': dates[10],  # 2 days holding
        'position': 1,
        'entry_price': 75.0,
        'current_price': 73.0,
        'pnl': -200.0,
        'pnl_pct': -0.027,         # > 0.02 max drawdown
        'max_drawdown': 0.027,
        'current_zscore': 1.2
    })
    
    # Trade 4: No violations
    trades.append({
        'trade_id': 'T004',
        'entry_date': dates[9],
        'current_date': dates[10],  # 1 day holding
        'position': -1,
        'entry_price': 120.0,
        'current_price': 118.0,
        'pnl': 200.0,
        'pnl_pct': 0.017,
        'max_drawdown': 0.005,
        'current_zscore': 1.8
    })
    
    # Create daily portfolio snapshots
    portfolio_snapshots = {date: [] for date in dates}
    
    # Add trades to their respective dates
    for trade in trades:
        # Add the trade to snapshots from entry_date to current_date
        current_idx = dates.get_loc(trade['current_date'])
        entry_idx = dates.get_loc(trade['entry_date'])
        
        for i in range(entry_idx, current_idx + 1):
            # Create a copy to avoid modifying the original
            trade_copy = trade.copy()
            # Adjust date-specific fields
            days_held = (dates[i] - trade['entry_date']).days
            trade_copy['current_date'] = dates[i]
            trade_copy['days_held'] = days_held
            
            # Add to the appropriate snapshot
            portfolio_snapshots[dates[i]].append(trade_copy)
    
    return {
        'trades': trades,
        'portfolio_snapshots': portfolio_snapshots,
        'dates': dates,
        'zscores': pd.Series(zscore_data, index=dates, name='zscore'),
        'drawdowns': pd.Series(drawdown_data, index=dates, name='drawdown')
    }


@pytest.fixture
def mock_daily_pnl():
    """Create mock daily P&L data for testing daily loss limits."""
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
    
    # Generate random daily P&L with a few days exceeding threshold
    np.random.seed(42)  # For reproducibility
    daily_pnl = np.random.normal(500, 2000, 20)
    
    # Set a few days to have large losses
    daily_pnl[5] = -4500   # Exceeds daily limit
    daily_pnl[12] = -3800  # Exceeds daily limit
    
    return pd.Series(daily_pnl, index=dates, name='daily_pnl')


class TestBasicRiskControls:
    """Test class for basic risk controls in the MVTS approach."""
    
    def test_max_holding_period(self, basic_risk_manager, mock_trade_data):
        """Test that maximum holding period control works correctly."""
        # Get test data
        trades = mock_trade_data['trades']
        
        # Check each trade for max holding period violation
        for trade in trades:
            days_held = (trade['current_date'] - trade['entry_date']).days
            
            # Create a mock signal for this trade
            signal = {
                'action': 'hold',
                'reason': 'continue'
            }
            
            # Create mock pair info with days held
            pair_info = {
                'days_held': days_held,
                'current_zscore': trade['current_zscore']
            }
            
            # Create mock portfolio state
            portfolio_state = {
                'max_drawdown': trade['max_drawdown']
            }
            
            # Apply risk controls
            adjusted_signal = basic_risk_manager.adjust_signal(
                signal, pair_info, portfolio_state
            )
            
            # Check if max holding period should trigger exit
            if days_held > basic_risk_manager.max_holding_period:
                assert adjusted_signal['action'] == 'close'
                assert 'holding_period' in adjusted_signal['reason']
            elif trade['current_zscore'] > basic_risk_manager.max_zscore:
                assert adjusted_signal['action'] == 'close'
                assert 'zscore' in adjusted_signal['reason']
            elif trade['max_drawdown'] > basic_risk_manager.max_drawdown:
                assert adjusted_signal['action'] == 'close'
                assert 'drawdown' in adjusted_signal['reason']
            else:
                # No risk violations, signal should be unchanged
                assert adjusted_signal['action'] == 'hold'
                assert adjusted_signal['reason'] == 'continue'
    
    def test_stop_loss(self, basic_risk_manager, mock_trade_data):
        """Test that stop-loss control works correctly."""
        # Get test data
        trades = mock_trade_data['trades']
        
        for trade in trades:
            # Create a mock signal
            signal = {
                'action': 'hold',
                'reason': 'continue'
            }
            
            # Create mock pair info
            pair_info = {
                'days_held': (trade['current_date'] - trade['entry_date']).days,
                'current_zscore': trade['current_zscore'],
                'drawdown': trade['max_drawdown'],
                'trade_id': trade['trade_id']
            }
            
            # Create mock portfolio state
            portfolio_state = {
                'max_drawdown': trade['max_drawdown']
            }
            
            # Apply risk controls
            adjusted_signal = basic_risk_manager.adjust_signal(
                signal, pair_info, portfolio_state
            )
            
            # Check if stopout conditions would trigger
            if trade['max_drawdown'] > basic_risk_manager.max_drawdown:
                assert adjusted_signal['action'] == 'close'
                assert 'drawdown' in adjusted_signal['reason']
                print(f"Trade {trade['trade_id']} correctly stopped out due to drawdown")
            
            # For z-score based stop-loss
            if abs(trade['current_zscore']) > basic_risk_manager.max_zscore:
                assert adjusted_signal['action'] == 'close'
                assert 'zscore' in adjusted_signal['reason']
                print(f"Trade {trade['trade_id']} correctly stopped out due to z-score")
    
    def test_max_z_score_stop(self, basic_risk_manager):
        """Test that the z-score threshold stop works correctly."""
        # Create mock signals and pair info with different z-scores
        test_cases = [
            {'zscore': 1.0, 'should_close': False},
            {'zscore': 2.0, 'should_close': False},
            {'zscore': 2.9, 'should_close': False},
            {'zscore': 3.0, 'should_close': False},  # Exactly at threshold
            {'zscore': 3.1, 'should_close': True},   # Just over threshold
            {'zscore': 4.0, 'should_close': True},   # Clearly over threshold
            {'zscore': -1.0, 'should_close': False},
            {'zscore': -2.0, 'should_close': False},
            {'zscore': -3.1, 'should_close': True},  # Over threshold (negative)
            {'zscore': -4.0, 'should_close': True}   # Clearly over threshold (negative)
        ]
        
        for case in test_cases:
            # Create a mock signal
            signal = {
                'action': 'hold',
                'reason': 'continue'
            }
            
            # Create mock pair info with the test z-score
            pair_info = {
                'current_zscore': case['zscore'],
                'days_held': 5  # Below max holding period
            }
            
            # Create mock portfolio state
            portfolio_state = {
                'max_drawdown': 0.01  # Below max drawdown
            }
            
            # Apply risk controls
            adjusted_signal = basic_risk_manager.adjust_signal(
                signal, pair_info, portfolio_state
            )
            
            # Check if the signal was correctly adjusted based on z-score
            if case['should_close']:
                assert adjusted_signal['action'] == 'close', f"Failed for z-score {case['zscore']}"
                assert 'zscore' in adjusted_signal['reason']
            else:
                assert adjusted_signal['action'] == 'hold', f"Failed for z-score {case['zscore']}"
                assert adjusted_signal['reason'] == 'continue'
    
    def test_daily_loss_limit(self, basic_risk_manager, mock_daily_pnl):
        """Test that daily loss limits are enforced correctly."""
        # Skip this test if daily_loss_limit is not implemented
        if not hasattr(basic_risk_manager, 'daily_loss_limit'):
            pytest.skip("daily_loss_limit not implemented in RiskManager yet")
        
        # Set a daily loss limit
        basic_risk_manager.daily_loss_limit = 3000  # $3000 daily loss limit
        
        # Test daily P&L against limit
        dates_to_check = mock_daily_pnl.index
        
        for date in dates_to_check:
            daily_pnl = mock_daily_pnl[date]
            
            # Check if trading should be halted based on daily loss
            should_halt = basic_risk_manager.check_daily_loss_limit(daily_pnl)
            
            # Verify correct halt signal
            if daily_pnl < -basic_risk_manager.daily_loss_limit:
                assert should_halt is True, f"Should halt trading on {date} with P&L {daily_pnl}"
            else:
                assert should_halt is False, f"Should not halt trading on {date} with P&L {daily_pnl}"
    
    def test_integrated_risk_controls(self, basic_risk_manager, mock_trade_data):
        """Test that multiple risk controls work together correctly."""
        # Get test data
        snapshots = mock_trade_data['portfolio_snapshots']
        dates = mock_trade_data['dates']
        
        # Test each day's portfolio against risk controls
        for date in dates:
            trades = snapshots[date]
            
            for trade in trades:
                # Create a mock signal
                signal = {
                    'action': 'hold',
                    'reason': 'continue'
                }
                
                # Create mock pair info
                pair_info = {
                    'days_held': trade.get('days_held', 0),
                    'current_zscore': trade.get('current_zscore', 0),
                    'trade_id': trade.get('trade_id', '')
                }
                
                # Create mock portfolio state
                portfolio_state = {
                    'max_drawdown': trade.get('max_drawdown', 0)
                }
                
                # Apply risk controls
                adjusted_signal = basic_risk_manager.adjust_signal(
                    signal, pair_info, portfolio_state
                )
                
                # Check if any risk control should trigger exit
                should_exit = False
                exit_reason = []
                
                if pair_info['days_held'] > basic_risk_manager.max_holding_period:
                    should_exit = True
                    exit_reason.append('holding_period')
                
                if abs(pair_info['current_zscore']) > basic_risk_manager.max_zscore:
                    should_exit = True
                    exit_reason.append('zscore')
                
                if portfolio_state['max_drawdown'] > basic_risk_manager.max_drawdown:
                    should_exit = True
                    exit_reason.append('drawdown')
                
                # Verify signal reflects the expected decision
                if should_exit:
                    assert adjusted_signal['action'] == 'close', f"Failed for trade {trade['trade_id']} on {date}"
                    # Check that at least one of the expected reasons is in the actual reason
                    assert any(reason in adjusted_signal['reason'] for reason in exit_reason), \
                           f"Failed for trade {trade['trade_id']} on {date}, expected one of {exit_reason}"
                else:
                    assert adjusted_signal['action'] == 'hold', f"Failed for trade {trade['trade_id']} on {date}"
                    assert adjusted_signal['reason'] == 'continue' 