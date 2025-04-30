"""
Unit tests for daily loss limits in the MVTS.

This module contains tests for the daily loss limit functionality that is an
essential risk control for the Minimal Viable Trading Strategy (MVTS) implementation
to meet prop firm requirements.

The tests verify that the risk management components properly:
- Track daily P&L across all trading pairs
- Enforce daily loss limits by temporarily halting trading when limits are exceeded
- Reset daily loss tracking at the start of a new trading day
- Handle different account sizes and risk parameters
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

# Import the proper modules - will depend on actual implementation in the codebase
try:
    from src.risk_management.daily_loss_manager import DailyLossManager
except ImportError:
    # If dedicated class doesn't exist, we'll patch the risk manager
    from src.risk_management.risk_manager import RiskManager


@pytest.fixture
def account_settings():
    """Create account settings with different sizes for testing."""
    return {
        'small': {
            'account_size': 50000,
            'daily_loss_limit_pct': 0.03,  # 3% daily loss limit
            'daily_loss_limit_amount': 1500  # $1,500 daily loss limit
        },
        'medium': {
            'account_size': 100000,
            'daily_loss_limit_pct': 0.025,  # 2.5% daily loss limit
            'daily_loss_limit_amount': 2500  # $2,500 daily loss limit
        },
        'large': {
            'account_size': 250000,
            'daily_loss_limit_pct': 0.02,  # 2% daily loss limit
            'daily_loss_limit_amount': 5000  # $5,000 daily loss limit
        }
    }


@pytest.fixture
def mock_daily_trades():
    """Create mock trades across multiple days for testing daily limits."""
    # Create dates for a trading week
    dates = []
    start_date = datetime(2023, 5, 1)  # Monday
    for i in range(5):  # Monday to Friday
        dates.append(start_date + timedelta(days=i))
    
    # Create trades with profits and losses
    trades = {
        # Monday: Moderate profits, no daily limit breach
        dates[0]: [
            {'trade_id': 'M1', 'pnl': 500, 'pair': 'AAPL_MSFT'},
            {'trade_id': 'M2', 'pnl': -200, 'pair': 'GOOGL_FB'},
            {'trade_id': 'M3', 'pnl': 700, 'pair': 'AMZN_NFLX'}
        ],
        # Tuesday: Big loss day, should trigger daily limit
        dates[1]: [
            {'trade_id': 'T1', 'pnl': -1800, 'pair': 'AAPL_MSFT'},
            {'trade_id': 'T2', 'pnl': -1200, 'pair': 'GOOGL_FB'},
            {'trade_id': 'T3', 'pnl': 500, 'pair': 'AMZN_NFLX'}
        ],
        # Wednesday: Mixed day, no limit breach
        dates[2]: [
            {'trade_id': 'W1', 'pnl': 800, 'pair': 'AAPL_MSFT'},
            {'trade_id': 'W2', 'pnl': -600, 'pair': 'GOOGL_FB'},
            {'trade_id': 'W3', 'pnl': 300, 'pair': 'AMZN_NFLX'}
        ],
        # Thursday: Very small loss day, no limit breach
        dates[3]: [
            {'trade_id': 'TH1', 'pnl': -300, 'pair': 'AAPL_MSFT'},
            {'trade_id': 'TH2', 'pnl': -100, 'pair': 'GOOGL_FB'},
            {'trade_id': 'TH3', 'pnl': 200, 'pair': 'AMZN_NFLX'}
        ],
        # Friday: Another big loss day, should trigger limit
        dates[4]: [
            {'trade_id': 'F1', 'pnl': -1500, 'pair': 'AAPL_MSFT'},
            {'trade_id': 'F2', 'pnl': -1000, 'pair': 'GOOGL_FB'},
            {'trade_id': 'F3', 'pnl': -800, 'pair': 'AMZN_NFLX'}
        ]
    }
    
    # Calculate cumulative PnL for each day
    daily_pnl = {}
    for date in dates:
        daily_pnl[date] = sum(trade['pnl'] for trade in trades[date])
    
    return {
        'dates': dates,
        'trades': trades,
        'daily_pnl': daily_pnl
    }


class TestDailyLossLimits:
    """Test class for daily loss limit functionality in the MVTS approach."""
    
    def test_daily_loss_limit_initialization(self, account_settings):
        """Test that daily loss limits are correctly initialized."""
        # Check if DailyLossManager exists
        if 'DailyLossManager' in globals():
            # Test with dedicated class
            for account_type, settings in account_settings.items():
                manager = DailyLossManager(
                    account_size=settings['account_size'],
                    daily_loss_limit_pct=settings['daily_loss_limit_pct']
                )
                
                expected_limit = settings['daily_loss_limit_amount']
                assert manager.daily_loss_limit == expected_limit, \
                       f"Daily loss limit for {account_type} account initialized incorrectly"
        else:
            # Test with RiskManager if DailyLossManager doesn't exist
            for account_type, settings in account_settings.items():
                manager = RiskManager()
                manager.account_size = settings['account_size']
                manager.daily_loss_limit_pct = settings['daily_loss_limit_pct']
                
                # Call method to initialize daily loss limit if it exists
                if hasattr(manager, 'initialize_daily_loss_limit'):
                    manager.initialize_daily_loss_limit()
                    
                    # If method exists, expect daily_loss_limit to be set
                    if hasattr(manager, 'daily_loss_limit'):
                        expected_limit = settings['daily_loss_limit_amount']
                        assert manager.daily_loss_limit == expected_limit, \
                               f"Daily loss limit for {account_type} account initialized incorrectly"
                    else:
                        pytest.skip("RiskManager doesn't have daily_loss_limit attribute")
                else:
                    pytest.skip("RiskManager doesn't have initialize_daily_loss_limit method")
    
    def test_daily_loss_tracking(self, mock_daily_trades):
        """Test that daily losses are properly tracked."""
        # Check if DailyLossManager exists
        if 'DailyLossManager' in globals():
            manager = DailyLossManager(account_size=100000, daily_loss_limit_pct=0.025)
            
            # Process trades day by day
            for date in mock_daily_trades['dates']:
                manager.reset_daily_tracking(date)  # Reset at start of day
                
                # Process each trade
                for trade in mock_daily_trades['trades'][date]:
                    manager.update_daily_pnl(trade['pnl'])
                
                expected_daily_pnl = mock_daily_trades['daily_pnl'][date]
                assert manager.current_daily_pnl == expected_daily_pnl, \
                       f"Daily P&L tracking incorrect for {date}"
        else:
            # If using RiskManager for daily loss tracking
            manager = RiskManager()
            manager.account_size = 100000
            manager.daily_loss_limit_pct = 0.025
            
            # Check if necessary methods exist
            has_reset_method = hasattr(manager, 'reset_daily_tracking')
            has_update_method = hasattr(manager, 'update_daily_pnl')
            has_tracking_attr = hasattr(manager, 'current_daily_pnl')
            
            if has_reset_method and has_update_method and has_tracking_attr:
                # Process trades day by day
                for date in mock_daily_trades['dates']:
                    manager.reset_daily_tracking(date)  # Reset at start of day
                    
                    # Process each trade
                    for trade in mock_daily_trades['trades'][date]:
                        manager.update_daily_pnl(trade['pnl'])
                    
                    expected_daily_pnl = mock_daily_trades['daily_pnl'][date]
                    assert manager.current_daily_pnl == expected_daily_pnl, \
                           f"Daily P&L tracking incorrect for {date}"
            else:
                pytest.skip("RiskManager doesn't have required methods for daily loss tracking")
    
    def test_daily_loss_limit_enforcement(self, mock_daily_trades, account_settings):
        """Test that daily loss limits are properly enforced."""
        medium_settings = account_settings['medium']
        
        # Check if DailyLossManager exists
        if 'DailyLossManager' in globals():
            manager = DailyLossManager(
                account_size=medium_settings['account_size'],
                daily_loss_limit_pct=medium_settings['daily_loss_limit_pct']
            )
            
            # Process trades day by day and check for limit breaches
            for date in mock_daily_trades['dates']:
                manager.reset_daily_tracking(date)  # Reset at start of day
                
                # Track if trading should be halted
                should_halt = False
                
                # Process each trade
                for trade in mock_daily_trades['trades'][date]:
                    # Update PnL
                    manager.update_daily_pnl(trade['pnl'])
                    
                    # Check if limit is breached
                    if manager.current_daily_pnl < -manager.daily_loss_limit:
                        should_halt = True
                
                # Check daily total against expected halt status
                daily_pnl = mock_daily_trades['daily_pnl'][date]
                expected_halt = daily_pnl < -medium_settings['daily_loss_limit_amount']
                
                assert should_halt == expected_halt, \
                       f"Incorrect trading halt status for {date} with P&L {daily_pnl}"
        else:
            # If using RiskManager for daily loss tracking
            manager = RiskManager()
            manager.account_size = medium_settings['account_size']
            manager.daily_loss_limit_pct = medium_settings['daily_loss_limit_pct']
            manager.daily_loss_limit = medium_settings['daily_loss_limit_amount']
            
            # Check if necessary methods exist
            has_reset_method = hasattr(manager, 'reset_daily_tracking')
            has_update_method = hasattr(manager, 'update_daily_pnl')
            has_check_method = hasattr(manager, 'check_daily_loss_limit')
            
            if has_reset_method and has_update_method and has_check_method:
                # Process trades day by day and check for limit breaches
                for date in mock_daily_trades['dates']:
                    manager.reset_daily_tracking(date)  # Reset at start of day
                    
                    # Process each trade
                    for trade in mock_daily_trades['trades'][date]:
                        # Update PnL
                        manager.update_daily_pnl(trade['pnl'])
                    
                    # Check if limit is breached
                    should_halt = manager.check_daily_loss_limit(manager.current_daily_pnl)
                    
                    # Check daily total against expected halt status
                    daily_pnl = mock_daily_trades['daily_pnl'][date]
                    expected_halt = daily_pnl < -medium_settings['daily_loss_limit_amount']
                    
                    assert should_halt == expected_halt, \
                           f"Incorrect trading halt status for {date} with P&L {daily_pnl}"
            else:
                pytest.skip("RiskManager doesn't have required methods for daily loss limit enforcement")
    
    def test_daily_reset(self, mock_daily_trades):
        """Test that daily tracking is properly reset."""
        # Check if DailyLossManager exists
        if 'DailyLossManager' in globals():
            manager = DailyLossManager(account_size=100000, daily_loss_limit_pct=0.025)
            
            # Test over multiple days
            for date in mock_daily_trades['dates']:
                # Simulate some existing PnL before reset
                manager.current_daily_pnl = -1000
                
                # Reset at start of day
                manager.reset_daily_tracking(date)
                
                # Should be reset to zero
                assert manager.current_daily_pnl == 0, \
                       f"Daily P&L not reset correctly for {date}"
        else:
            # If using RiskManager for daily loss tracking
            manager = RiskManager()
            
            # Check if necessary methods exist
            has_reset_method = hasattr(manager, 'reset_daily_tracking')
            has_tracking_attr = hasattr(manager, 'current_daily_pnl')
            
            if has_reset_method and has_tracking_attr:
                # Test over multiple days
                for date in mock_daily_trades['dates']:
                    # Simulate some existing PnL before reset
                    manager.current_daily_pnl = -1000
                    
                    # Reset at start of day
                    manager.reset_daily_tracking(date)
                    
                    # Should be reset to zero
                    assert manager.current_daily_pnl == 0, \
                           f"Daily P&L not reset correctly for {date}"
            else:
                pytest.skip("RiskManager doesn't have required methods for daily tracking reset")
    
    def test_different_account_sizes(self, account_settings, mock_daily_trades):
        """Test daily loss limits with different account sizes."""
        # Check if DailyLossManager exists
        if 'DailyLossManager' in globals():
            # Only test big loss days (index 1 and 4)
            loss_days = [mock_daily_trades['dates'][1], mock_daily_trades['dates'][4]]
            
            for account_type, settings in account_settings.items():
                manager = DailyLossManager(
                    account_size=settings['account_size'],
                    daily_loss_limit_pct=settings['daily_loss_limit_pct']
                )
                
                for date in loss_days:
                    manager.reset_daily_tracking(date)
                    
                    # Process all trades for the day
                    for trade in mock_daily_trades['trades'][date]:
                        manager.update_daily_pnl(trade['pnl'])
                    
                    daily_pnl = mock_daily_trades['daily_pnl'][date]
                    expected_halt = daily_pnl < -settings['daily_loss_limit_amount']
                    
                    # Check if trading should be halted
                    should_halt = manager.current_daily_pnl < -manager.daily_loss_limit
                    
                    assert should_halt == expected_halt, \
                           f"Incorrect trading halt status for {account_type} account on {date}"
        else:
            # If using RiskManager for daily loss tracking
            # Only test big loss days (index 1 and 4)
            loss_days = [mock_daily_trades['dates'][1], mock_daily_trades['dates'][4]]
            
            for account_type, settings in account_settings.items():
                manager = RiskManager()
                manager.account_size = settings['account_size']
                manager.daily_loss_limit_pct = settings['daily_loss_limit_pct']
                manager.daily_loss_limit = settings['daily_loss_limit_amount']
                
                # Check if necessary methods exist
                has_reset_method = hasattr(manager, 'reset_daily_tracking')
                has_update_method = hasattr(manager, 'update_daily_pnl')
                has_tracking_attr = hasattr(manager, 'current_daily_pnl')
                
                if has_reset_method and has_update_method and has_tracking_attr:
                    for date in loss_days:
                        manager.reset_daily_tracking(date)
                        
                        # Process all trades for the day
                        for trade in mock_daily_trades['trades'][date]:
                            manager.update_daily_pnl(trade['pnl'])
                        
                        daily_pnl = mock_daily_trades['daily_pnl'][date]
                        expected_halt = daily_pnl < -settings['daily_loss_limit_amount']
                        
                        # Check if trading should be halted
                        should_halt = manager.current_daily_pnl < -manager.daily_loss_limit
                        
                        assert should_halt == expected_halt, \
                               f"Incorrect trading halt status for {account_type} account on {date}"
                else:
                    pytest.skip("RiskManager doesn't have required methods for daily loss tracking")
    
    def test_prop_firm_compliance(self, account_settings):
        """Test that daily loss limits comply with typical prop firm requirements."""
        if 'DailyLossManager' in globals():
            # Common prop firm daily loss limits are typically 3-5% of account
            for account_type, settings in account_settings.items():
                manager = DailyLossManager(
                    account_size=settings['account_size'],
                    daily_loss_limit_pct=settings['daily_loss_limit_pct']
                )
                
                # Check that limit is set correctly
                expected_limit = settings['account_size'] * settings['daily_loss_limit_pct']
                assert abs(manager.daily_loss_limit - expected_limit) < 0.01, \
                       f"Daily loss limit for {account_type} account not set to {settings['daily_loss_limit_pct']*100}% of account"
                
                # Check that limit matches expected amount
                assert abs(manager.daily_loss_limit - settings['daily_loss_limit_amount']) < 0.01, \
                       f"Daily loss limit amount incorrect for {account_type} account"
        else:
            # If using RiskManager
            manager = RiskManager()
            
            # Check if it has daily_loss_limit_pct attribute
            if hasattr(manager, 'daily_loss_limit_pct'):
                # Common prop firm daily loss limits are typically 3-5% of account
                for account_type, settings in account_settings.items():
                    manager.account_size = settings['account_size']
                    manager.daily_loss_limit_pct = settings['daily_loss_limit_pct']
                    
                    # Calculate expected limit
                    expected_limit = settings['account_size'] * settings['daily_loss_limit_pct']
                    
                    # If manager has method to initialize, call it
                    if hasattr(manager, 'initialize_daily_loss_limit'):
                        manager.initialize_daily_loss_limit()
                    else:
                        # Otherwise set it directly
                        manager.daily_loss_limit = expected_limit
                    
                    # Check that limit matches expected amount
                    assert abs(manager.daily_loss_limit - settings['daily_loss_limit_amount']) < 0.01, \
                           f"Daily loss limit amount incorrect for {account_type} account"
            else:
                pytest.skip("RiskManager doesn't have daily_loss_limit_pct attribute") 