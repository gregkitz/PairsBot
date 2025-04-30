"""
Tests for the IntradayBacktestEngine class.

This module tests the functionality of the IntradayBacktestEngine including
time-of-day constraints, transaction cost modeling, and other intraday-specific features.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.backtest.intraday_backtest_engine import IntradayBacktestEngine

logging.basicConfig(level=logging.ERROR)

class TestIntradayBacktestEngine(unittest.TestCase):
    """Test cases for the IntradayBacktestEngine class."""
    
    def setUp(self):
        """Set up test data for each test."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-01 16:00:00', freq='5min')
        
        # Create price data for two instruments
        self.prices = {
            'SPY': pd.DataFrame({
                'open': np.linspace(400, 410, len(dates)) + np.random.normal(0, 0.5, len(dates)),
                'high': np.linspace(400, 410, len(dates)) + np.random.normal(0, 1.0, len(dates)),
                'low': np.linspace(400, 410, len(dates)) + np.random.normal(0, 1.0, len(dates)),
                'close': np.linspace(400, 410, len(dates)) + np.random.normal(0, 0.5, len(dates)),
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates),
            'QQQ': pd.DataFrame({
                'open': np.linspace(300, 310, len(dates)) + np.random.normal(0, 0.5, len(dates)),
                'high': np.linspace(300, 310, len(dates)) + np.random.normal(0, 1.0, len(dates)),
                'low': np.linspace(300, 310, len(dates)) + np.random.normal(0, 1.0, len(dates)),
                'close': np.linspace(300, 310, len(dates)) + np.random.normal(0, 0.5, len(dates)),
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        }
        
        # Create sample signals data
        self.signals = pd.DataFrame({
            'SPY': [0] * 10 + [1] * 20 + [0] * 10 + [-1] * 20 + [0] * (len(dates) - 60)
        }, index=dates)
        
        # Create volume data
        self.volume_data = {
            'SPY': pd.DataFrame({
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates),
            'QQQ': pd.DataFrame({
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        }
        
        # Test intraday params
        self.intraday_params = {
            "max_holding_period": 60,  # 1 hour in minutes
            "time_filters": {
                "avoid_first_15min": True,
                "avoid_lunch_hour": True,
                "high_liquidity_windows": [
                    {"start": "09:45", "end": "11:30"},
                    {"start": "13:30", "end": "15:45"}
                ]
            },
            "exit_buffer_minutes": 15  # Exit 15 minutes before market close
        }
        
        # Test transaction cost model
        self.transaction_cost_model = {
            "commission_model": "ibkr_pro",
            "commission_params": {
                "per_contract": 0.85,
                "per_share": 0.005,
                "minimum": 1.0
            },
            "slippage_model": "fixed",
            "slippage_params": {
                "fixed_percentage": 0.0001  # 0.01%
            }
        }
    
    def test_initialization(self):
        """Test that the engine initializes correctly with default parameters."""
        engine = IntradayBacktestEngine(
            signals=self.signals,
            prices=self.prices,
            account_size=100000
        )
        
        self.assertEqual(engine.initial_account_size, 100000)
        self.assertEqual(engine.intraday_params["max_holding_period"], 180)
        self.assertEqual(engine.transaction_cost_model["commission_model"], "ibkr_pro")
    
    def test_initialization_with_custom_params(self):
        """Test that the engine initializes correctly with custom parameters."""
        engine = IntradayBacktestEngine(
            signals=self.signals,
            prices=self.prices,
            account_size=100000,
            intraday_params=self.intraday_params,
            transaction_cost_model=self.transaction_cost_model
        )
        
        self.assertEqual(engine.intraday_params["max_holding_period"], 60)
        self.assertEqual(engine.transaction_cost_model["slippage_model"], "fixed")
    
    def test_apply_max_holding_period(self):
        """Test that positions are closed after the maximum holding period."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            intraday_params={
                "max_holding_period": 30  # 30 minutes max holding
            }
        )
        
        # Create signals with long holding periods
        dates = pd.date_range(start='2023-01-01 10:00:00', end='2023-01-01 12:00:00', freq='5min')
        signals = pd.DataFrame({
            'SPY': [1] * len(dates)  # Always in position
        }, index=dates)
        
        # Apply max holding period constraint
        constrained_signals = engine._apply_max_holding_period(signals)
        
        # Should be in position for 30 minutes (6 bars), then out
        self.assertEqual(constrained_signals.iloc[0, 0], 1)  # First bar in position
        self.assertEqual(constrained_signals.iloc[5, 0], 1)  # Last bar in position (5th bar = 25 minutes)
        self.assertEqual(constrained_signals.iloc[6, 0], 0)  # Position closed after 30 minutes
    
    def test_apply_time_filters(self):
        """Test that time-of-day filters are correctly applied."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            intraday_params={
                "time_filters": {
                    "avoid_first_15min": True,
                    "avoid_lunch_hour": True
                }
            }
        )
        
        # Create signals that span the entire trading day
        dates = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-01 16:00:00', freq='5min')
        
        # Create signals with all entries at different times of day
        signals = pd.DataFrame({
            'SPY': [0] * len(dates)
        }, index=dates)
        
        # Add entry signals at different times
        signals.loc['2023-01-01 09:35:00', 'SPY'] = 1  # First 15 minutes
        signals.loc['2023-01-01 10:00:00', 'SPY'] = 1  # After first 15 minutes
        signals.loc['2023-01-01 12:10:00', 'SPY'] = 1  # Lunch hour
        signals.loc['2023-01-01 14:00:00', 'SPY'] = 1  # After lunch hour
        
        # Apply time filters
        constrained_signals = engine._apply_time_filters(signals)
        
        # Check if signals were filtered correctly
        self.assertEqual(constrained_signals.loc['2023-01-01 09:35:00', 'SPY'], 0)  # Filtered out (first 15 min)
        self.assertEqual(constrained_signals.loc['2023-01-01 10:00:00', 'SPY'], 1)  # Kept (after first 15 min)
        self.assertEqual(constrained_signals.loc['2023-01-01 12:10:00', 'SPY'], 0)  # Filtered out (lunch hour)
        self.assertEqual(constrained_signals.loc['2023-01-01 14:00:00', 'SPY'], 1)  # Kept (after lunch hour)
    
    def test_market_close_constraint(self):
        """Test that positions are closed before market close."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            intraday_params={
                "exit_buffer_minutes": 15  # Exit 15 minutes before close
            },
            market_hours={
                "start": "09:30",
                "end": "16:00"
            }
        )
        
        # Create signals that span the entire trading day
        dates = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-01 16:00:00', freq='5min')
        
        # Create signals with a position that spans the market close
        signals = pd.DataFrame({
            'SPY': [0] * len(dates)
        }, index=dates)
        
        # Add a position from 15:00 to the end of the day
        time_mask = (signals.index >= '2023-01-01 15:00:00')
        signals.loc[time_mask, 'SPY'] = 1
        
        # Apply market close constraint
        constrained_signals = engine._apply_market_close_constraint(signals)
        
        # Check if positions are closed before market close
        self.assertEqual(constrained_signals.loc['2023-01-01 15:40:00', 'SPY'], 1)  # Still open before exit buffer
        self.assertEqual(constrained_signals.loc['2023-01-01 15:45:00', 'SPY'], 0)  # Closed at exit buffer time
        self.assertEqual(constrained_signals.loc['2023-01-01 15:50:00', 'SPY'], 0)  # Remains closed
    
    def test_commission_calculation(self):
        """Test that commission is calculated correctly."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            transaction_cost_model={
                "commission_model": "ibkr_pro",
                "commission_params": {
                    "per_contract": 0.85,
                    "per_share": 0.005,
                    "minimum": 1.0
                }
            }
        )
        
        # Test stock commission
        spy_commission = engine._calculate_commission("SPY", 400.0, 100)
        # Should be max(1.0, 100 * 0.005) = max(1.0, 0.5) = 1.0 (minimum commission)
        self.assertEqual(spy_commission, 1.0)
        
        # Test stock commission with larger quantity
        spy_commission_large = engine._calculate_commission("SPY", 400.0, 300)
        # Should be max(1.0, 300 * 0.005) = max(1.0, 1.5) = 1.5
        self.assertEqual(spy_commission_large, 1.5)
        
        # Test futures commission
        es_commission = engine._calculate_commission("ES", 4000.0, 2)
        # Should be 2 * 0.85 = 1.7
        self.assertEqual(es_commission, 1.7)
    
    def test_slippage_calculation(self):
        """Test that slippage is calculated correctly."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            transaction_cost_model={
                "slippage_model": "fixed",
                "slippage_params": {
                    "fixed_percentage": 0.0001  # 0.01%
                }
            }
        )
        
        # Test fixed slippage
        timestamp = datetime(2023, 1, 1, 10, 0, 0)
        slippage = engine._calculate_slippage(timestamp, "SPY", 400.0, 100)
        # Should be 400 * 100 * 0.0001 = 4.0
        self.assertEqual(slippage, 4.0)
        
        # Test slippage with different model
        engine.transaction_cost_model = {
            "slippage_model": "variable",
            "slippage_params": {
                "base_percentage": 0.0001,
                "max_factor": 1.0  # Set to 1.0 for deterministic testing
            }
        }
        
        # Mock the random function for deterministic testing
        np.random.seed(42)
        slippage_variable = engine._calculate_slippage(timestamp, "SPY", 400.0, 100)
        
        # Variable slippage with random factor between 0 and 1
        # Should be approximately 400 * 100 * 0.0001 * random_factor
        self.assertGreaterEqual(slippage_variable, 0.0)
        self.assertLessEqual(slippage_variable, 4.0)  # Maximum would be with factor=1.0
    
    def test_transaction_cost_impact(self):
        """Test that transaction costs impact the trade PnL correctly."""
        engine = IntradayBacktestEngine(
            signals=None,
            prices=self.prices,
            account_size=100000,
            transaction_cost_model={
                "commission_model": "flat",
                "commission_params": {
                    "flat_fee": 1.0
                },
                "slippage_model": "fixed",
                "slippage_params": {
                    "fixed_percentage": 0.0001  # 0.01%
                }
            }
        )
        
        # Create a sample trade
        timestamp = datetime(2023, 1, 1, 10, 0, 0)
        trade = {
            'type': 'entry',
            'symbol1': 'SPY',
            'qty1': 100,
            'price1': 400.0,
            'pnl': 0.0  # Initial PnL before costs
        }
        
        # Mock prices data
        prices = {
            'SPY': {
                'close': 400.0
            }
        }
        
        # Calculate transaction costs
        costs = engine._calculate_transaction_costs(timestamp, trade, prices)
        
        # Check costs
        self.assertEqual(costs['commission'], 1.0)  # Flat fee
        self.assertEqual(costs['slippage'], 4.0)  # 400 * 100 * 0.0001
        self.assertEqual(costs['total'], 5.0)  # Total cost
    
    def test_full_backtest(self):
        """Test a full backtest run with intraday constraints."""
        engine = IntradayBacktestEngine(
            signals=self.signals,
            prices=self.prices,
            account_size=100000,
            intraday_params=self.intraday_params,
            transaction_cost_model=self.transaction_cost_model,
            volume_data=self.volume_data
        )
        
        # Run the backtest
        results = engine.run_backtest()
        
        # Check if backtest ran successfully
        self.assertIsNotNone(results)
        self.assertIn('equity_curve', results)
        self.assertIn('intraday_metrics', results)
        
        # Check if intraday metrics were recorded
        self.assertIn('transaction_costs', results['intraday_metrics'])
        self.assertIn('forced_exits', results['intraday_metrics'])
        self.assertIn('time_violations', results['intraday_metrics'])
    
    def test_detailed_metrics(self):
        """Test that detailed metrics are calculated correctly."""
        engine = IntradayBacktestEngine(
            signals=self.signals,
            prices=self.prices,
            account_size=100000,
            intraday_params=self.intraday_params,
            transaction_cost_model=self.transaction_cost_model
        )
        
        # Run backtest first to populate metrics
        engine.run_backtest()
        
        # Manually add some transaction costs for testing
        engine.intraday_metrics["transaction_costs"]["commission"] = [1.0, 2.0, 3.0]
        engine.intraday_metrics["transaction_costs"]["slippage"] = [0.5, 1.0, 1.5]
        
        # Manually add some forced exits for testing
        engine.intraday_metrics["forced_exits"] = [
            {"timestamp": datetime(2023, 1, 1, 15, 45), "instrument": "SPY", "reason": "market_close"},
            {"timestamp": datetime(2023, 1, 1, 11, 30), "instrument": "SPY", "reason": "max_holding_period"}
        ]
        
        # Calculate detailed metrics
        detailed_metrics = engine.calculate_detailed_metrics()
        
        # Check metrics
        self.assertIn('transaction_costs', detailed_metrics)
        self.assertEqual(detailed_metrics['transaction_costs']['commission'], 6.0)
        self.assertEqual(detailed_metrics['transaction_costs']['slippage'], 3.0)
        self.assertEqual(detailed_metrics['transaction_costs']['total'], 9.0)
        
        self.assertIn('constraints', detailed_metrics)
        self.assertEqual(detailed_metrics['constraints']['forced_exits'], 2)
        
        # Check forced exits by reason
        reasons = detailed_metrics['constraints']['forced_exits_by_reason']
        self.assertEqual(reasons['market_close'], 1)
        self.assertEqual(reasons['max_holding_period'], 1)


if __name__ == '__main__':
    unittest.main() 