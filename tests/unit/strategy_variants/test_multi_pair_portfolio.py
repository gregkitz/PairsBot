"""
Unit tests for the multi-pair portfolio strategy.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.multi_pair_portfolio import MultiPairPortfolio
from src.asset_classes.base import Asset
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.risk_manager import RiskManager
from src.signal_generation.signal_generator import SignalGenerator
from src.pair_trading.pair_finder import PairFinder
from src.spread_analytics.spread_analyzer import SpreadAnalyzer

class MockAsset(Asset):
    """Mock asset implementation for testing."""
    
    def get_data(self, start_date, end_date):
        return pd.DataFrame()
    
    def get_metadata(self):
        return {"symbol": self.symbol}
    
    def get_current_price(self):
        return 100.0
    
    def get_trading_hours(self):
        return {"trading_hours": "9:30-16:00"}

class TestMultiPairPortfolio(unittest.TestCase):
    """Test cases for MultiPairPortfolio strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.position_sizer = MagicMock(spec=PositionSizer)
        self.position_sizer.calculate_position_size.return_value = 5000
        
        self.risk_manager = MagicMock(spec=RiskManager)
        self.risk_manager.adjust_signal.return_value = {"action": "hold"}
        
        self.signal_generator = MagicMock(spec=SignalGenerator)
        self.signal_generator.generate_signal.return_value = {"action": "hold"}
        
        self.pair_finder = MagicMock(spec=PairFinder)
        self.spread_analyzer = MagicMock(spec=SpreadAnalyzer)
        
        # Sample assets
        self.assets = [
            MockAsset(symbol=f"ASSET{i}", name=f"Test Asset {i}") 
            for i in range(10)
        ]
        
        # Set up sample pairs
        self.sample_pairs = []
        for i in range(5):
            self.sample_pairs.append({
                'asset1': self.assets[i*2],
                'asset2': self.assets[i*2+1],
                'hedge_ratio': 0.8 + (i * 0.1),
                'p_value': 0.01 + (i * 0.01),
                'half_life': 5 + i,
                'spread_volatility': 0.02 + (i * 0.005)
            })
        
        self.pair_finder.find_pairs.return_value = self.sample_pairs
        
        # Set up spread analyzer mock
        self.spread_analyzer.calculate_spread.return_value = {
            'current_zscore': 2.5,
            'current_spread': 0.5,
            'mean': 0.0,
            'std': 0.2
        }
        
        self.spread_analyzer.get_historical_spread.return_value = pd.Series(
            np.random.normal(0, 1, 100),
            index=pd.date_range(end=datetime.now(), periods=100, freq='D')
        )
        
        # Create the strategy
        self.strategy = MultiPairPortfolio(
            max_pairs=3,
            max_allocation_per_pair=0.2,
            total_risk_allocation=0.01,
            correlation_threshold=0.3,
            position_sizer=self.position_sizer,
            risk_manager=self.risk_manager,
            signal_generator=self.signal_generator,
            pair_finder=self.pair_finder,
            spread_analyzer=self.spread_analyzer
        )
    
    def test_initialization(self):
        """Test proper initialization of the strategy."""
        self.assertEqual(self.strategy.max_pairs, 3)
        self.assertEqual(self.strategy.max_allocation_per_pair, 0.2)
        self.assertEqual(self.strategy.total_risk_allocation, 0.01)
        self.assertEqual(self.strategy.correlation_threshold, 0.3)
        self.assertEqual(len(self.strategy.active_pairs), 0)
    
    def test_find_tradable_pairs(self):
        """Test finding tradable pairs."""
        # Setup
        for i, pair in enumerate(self.sample_pairs):
            pair['current_zscore'] = 2.0 + (i * 0.2)
        
        # Execute
        tradable_pairs = self.strategy.find_tradable_pairs(
            universe=self.assets,
            lookback_period=30,
            min_half_life=2,
            max_half_life=15,
            p_value_threshold=0.05,
            min_zscore=2.0
        )
        
        # Verify
        self.pair_finder.find_pairs.assert_called_once()
        self.spread_analyzer.calculate_spread.assert_called()
        self.assertEqual(len(tradable_pairs), len(self.sample_pairs))
    
    def test_optimize_portfolio(self):
        """Test portfolio optimization."""
        # Setup
        for i, pair in enumerate(self.sample_pairs):
            pair['current_zscore'] = 2.0 + (i * 0.2)
        
        # Execute
        optimized_pairs = self.strategy.optimize_portfolio(
            tradable_pairs=self.sample_pairs,
            account_value=100000
        )
        
        # Verify
        self.assertLessEqual(len(optimized_pairs), self.strategy.max_pairs)
        self.assertTrue(all('allocation' in pair for pair in optimized_pairs))
        
        # Check allocation limits
        total_allocation = sum(pair['allocation'] for pair in optimized_pairs)
        self.assertLessEqual(total_allocation, self.strategy.total_risk_allocation)
    
    def test_rebalance_portfolio(self):
        """Test portfolio rebalancing."""
        # Setup - no active pairs initially
        self.assertEqual(len(self.strategy.active_pairs), 0)
        
        # Execute rebalance
        rebalance_actions = self.strategy.rebalance_portfolio(
            universe=self.assets,
            account_value=100000
        )
        
        # Verify new pairs are added
        self.assertIn('open', rebalance_actions)
        self.assertTrue(len(rebalance_actions['open']) > 0)
        self.assertTrue(len(self.strategy.active_pairs) > 0)
        
        # Execute another rebalance with modified pairs
        # Modify the mock to return different pairs
        modified_pairs = self.sample_pairs[1:3]  # Different subset
        self.pair_finder.find_pairs.return_value = modified_pairs
        
        rebalance_actions = self.strategy.rebalance_portfolio(
            universe=self.assets,
            account_value=100000
        )
        
        # Verify there are close actions
        self.assertIn('close', rebalance_actions)
    
    def test_update(self):
        """Test strategy update function."""
        # Setup - add some active pairs
        for i in range(2):
            pair_id = f"ASSET{i*2}_ASSET{i*2+1}"
            self.strategy.active_pairs[pair_id] = self.sample_pairs[i]
        
        # Execute
        current_data = {
            'timestamp': datetime.now(),
            'account_value': 100000,
            'universe': self.assets
        }
        
        actions = self.strategy.update(current_data)
        
        # Verify
        self.assertIn('signals', actions)
        self.assertEqual(len(actions['signals']), len(self.strategy.active_pairs))
        self.assertEqual(self.signal_generator.generate_signal.call_count, len(self.strategy.active_pairs))
    
    def test_generate_orders(self):
        """Test order generation."""
        # Setup - add some active pairs
        for i in range(2):
            pair_id = f"ASSET{i*2}_ASSET{i*2+1}"
            pair = self.sample_pairs[i].copy()
            pair['leg1_position'] = 10
            pair['leg2_position'] = -8
            self.strategy.active_pairs[pair_id] = pair
        
        # Create mock signals
        signals = {
            'signals': [
                {
                    'pair_id': 'ASSET0_ASSET1',
                    'signal': {'action': 'close', 'reason': 'target_reached'},
                    'pair_info': self.strategy.active_pairs['ASSET0_ASSET1']
                }
            ],
            'rebalance': {
                'close': [],
                'open': [self.sample_pairs[2]],
                'adjust': []
            }
        }
        
        # Execute
        orders = self.strategy.generate_orders(signals)
        
        # Verify
        self.assertTrue(len(orders) > 0)
        
        # Check for close orders
        close_orders = [order for order in orders if order['action'] == 'close']
        self.assertTrue(len(close_orders) > 0)
        
        # Check for open orders
        open_orders = [order for order in orders if order['action'] in ['buy', 'sell']]
        self.assertTrue(len(open_orders) > 0)
    
    @patch('src.strategies.multi_pair_portfolio.datetime')
    def test_should_rebalance(self, mock_datetime):
        """Test rebalance timing logic."""
        # Setup
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Case 1: No previous rebalance
        self.strategy.last_rebalance_time = None
        self.assertTrue(self.strategy.should_rebalance(now))
        
        # Case 2: Recent rebalance
        self.strategy.last_rebalance_time = now - timedelta(hours=12)
        self.strategy.rebalance_frequency = '1D'  # Daily rebalance
        self.assertFalse(self.strategy.should_rebalance(now))
        
        # Case 3: Old rebalance
        self.strategy.last_rebalance_time = now - timedelta(days=2)
        self.strategy.rebalance_frequency = '1D'
        self.assertTrue(self.strategy.should_rebalance(now)) 