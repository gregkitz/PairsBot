"""
Unit tests for transaction cost models.

This test suite validates the implementation of transaction cost models including:
- Exchange fee calculation
- Slippage models
- Market impact models
- Transaction cost analysis
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the module under test
from src.execution.transaction_costs import (
    ExchangeType, SlippageModel, ExchangeFeeConfig, ExchangeFeeModel,
    FixedSlippageModel, VolumeBasedSlippageModel, VolatilityAdjustedSlippageModel,
    MarketImpactModel, CustomSlippageModel, TransactionCostAnalyzer,
    create_futures_exchange_fee_model, create_stock_exchange_fee_model, create_crypto_exchange_fee_model
)


class TestExchangeFeeModel(unittest.TestCase):
    """Test cases for exchange fee model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a basic exchange fee configuration
        self.config = ExchangeFeeConfig(
            maker_fee=1.0,    # 1.0 bps (0.01%)
            taker_fee=3.0,    # 3.0 bps (0.03%)
            minimum_fee=1.0,  # $1.0 minimum fee
            maximum_fee=100.0,  # $100 maximum fee
            fixed_fees={'ES': 1.5},  # Fixed fee for ES futures
            tier_levels=[
                (1000000, 2.5),  # $1M monthly volume: 2.5 bps taker fee
                (5000000, 2.0),  # $5M monthly volume: 2.0 bps taker fee
            ],
            rebates={'SPY': 0.2}  # 0.2 bps rebate for SPY
        )
        
        # Create exchange fee model
        self.fee_model = ExchangeFeeModel(ExchangeType.FUTURES_EXCHANGE, self.config)
        
    def test_taker_fee_calculation(self):
        """Test taker fee calculation."""
        # Test with small order (should use minimum fee)
        order_value = 100.0
        fee = self.fee_model.calculate_fee(order_value, is_maker=False)
        self.assertEqual(fee, 1.0)  # Should be minimum fee
        
        # Test with larger order
        order_value = 100000.0
        fee = self.fee_model.calculate_fee(order_value, is_maker=False)
        expected_fee = order_value * (self.config.taker_fee / 10000)
        self.assertAlmostEqual(fee, expected_fee)
        
        # Test with very large order (should use maximum fee)
        order_value = 10000000.0
        fee = self.fee_model.calculate_fee(order_value, is_maker=False)
        self.assertEqual(fee, 100.0)  # Should be maximum fee
    
    def test_maker_fee_calculation(self):
        """Test maker fee calculation."""
        # Test with maker order
        order_value = 100000.0
        fee = self.fee_model.calculate_fee(order_value, is_maker=True)
        expected_fee = order_value * (self.config.maker_fee / 10000)
        self.assertAlmostEqual(fee, expected_fee)
    
    def test_tier_based_fee(self):
        """Test tier-based fee calculation."""
        order_value = 100000.0
        
        # Test with monthly volume that hits first tier
        fee = self.fee_model.calculate_fee(
            order_value, is_maker=False, monthly_volume=1500000)
        expected_fee = order_value * (2.5 / 10000)
        self.assertAlmostEqual(fee, expected_fee)
        
        # Test with monthly volume that hits second tier
        fee = self.fee_model.calculate_fee(
            order_value, is_maker=False, monthly_volume=6000000)
        expected_fee = order_value * (2.0 / 10000)
        self.assertAlmostEqual(fee, expected_fee)
    
    def test_fixed_fee(self):
        """Test fixed fee for specific instruments."""
        order_value = 100000.0
        
        # Test with symbol that has fixed fee
        fee = self.fee_model.calculate_fee(
            order_value, is_maker=False, symbol='ES')
        self.assertEqual(fee, 1.5)
    
    def test_rebate(self):
        """Test rebate for specific instruments."""
        order_value = 100000.0
        
        # Test with symbol that has rebate for maker orders
        fee = self.fee_model.calculate_fee(
            order_value, is_maker=True, symbol='SPY')
        expected_fee = max(0, order_value * (self.config.maker_fee / 10000) - order_value * (0.2 / 10000))
        self.assertAlmostEqual(fee, expected_fee)


class TestFixedSlippageModel(unittest.TestCase):
    """Test cases for fixed slippage model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create fixed slippage model with 5 basis points
        self.slippage_model = FixedSlippageModel(basis_points=5.0)
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        price = 100.0
        quantity = 10
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(price, quantity)
        
        # Expected slippage: price * quantity * (basis_points / 10000)
        expected_slippage = price * quantity * (5.0 / 10000)
        self.assertAlmostEqual(slippage, expected_slippage)


class TestVolumeBasedSlippageModel(unittest.TestCase):
    """Test cases for volume-based slippage model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create volume-based slippage model
        self.slippage_model = VolumeBasedSlippageModel(
            base_slippage_bps=5.0, volume_impact_factor=0.5)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'volume': [10000, 12000, 9000, 11000, 10500, 9800, 10200, 9700, 10800, 11200]
        })
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        price = 100.0
        quantity = 1000  # 10% of average volume
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(price, quantity, self.market_data)
        
        # Expected calculation:
        avg_volume = self.market_data['volume'].mean()  # ~10420
        volume_ratio = quantity / avg_volume  # ~0.096
        impact_multiplier = np.power(volume_ratio, 0.5)  # ~0.31
        adjusted_slippage_bps = 5.0 * (1 + impact_multiplier)  # ~6.55
        expected_slippage = price * quantity * (adjusted_slippage_bps / 10000)
        
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_error_when_missing_volume(self):
        """Test error when volume data is missing."""
        price = 100.0
        quantity = 1000
        
        # Create market data without volume
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5)],
            'price': [100, 101, 102, 101, 100]
        })
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.slippage_model.calculate_slippage(price, quantity, invalid_data)


class TestVolatilityAdjustedSlippageModel(unittest.TestCase):
    """Test cases for volatility-adjusted slippage model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create volatility-adjusted slippage model
        self.slippage_model = VolatilityAdjustedSlippageModel(
            base_slippage_bps=5.0, volatility_impact_factor=1.5)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(30)],
            'close': [100 + i * 0.1 for i in range(30)]
        })
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        price = 100.0
        quantity = 10
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(price, quantity, self.market_data)
        
        # Calculate expected result
        returns = self.market_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        volatility_multiplier = np.power(volatility, 1.5)
        adjusted_slippage_bps = 5.0 * volatility_multiplier
        expected_slippage = price * quantity * (adjusted_slippage_bps / 10000)
        
        self.assertAlmostEqual(slippage, expected_slippage, places=2)
    
    def test_error_when_insufficient_data(self):
        """Test error when insufficient price data is provided."""
        price = 100.0
        quantity = 10
        
        # Create market data with only one row (insufficient for volatility)
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close': [100]
        })
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.slippage_model.calculate_slippage(price, quantity, invalid_data)


class TestMarketImpactModel(unittest.TestCase):
    """Test cases for market impact model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create market impact model
        self.impact_model = MarketImpactModel(
            alpha=0.1, beta=0.3, gamma=0.5, 
            temporary_impact_factor=1.0, permanent_impact_factor=0.5)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(30)],
            'close': [100 + i * 0.1 for i in range(30)],
            'volume': [10000 for _ in range(30)]
        })
    
    def test_market_impact_calculation(self):
        """Test market impact calculation."""
        price = 100.0
        quantity = 1000  # 10% of volume
        
        # Calculate market impact
        impact = self.impact_model.calculate_market_impact(
            price, quantity, self.market_data, is_buy=True)
        
        # Validate result structure
        self.assertIn('temporary_impact', impact)
        self.assertIn('permanent_impact', impact)
        self.assertIn('total_impact', impact)
        self.assertIn('impact_bps', impact)
        
        # Check relationships between impact components
        self.assertGreater(impact['temporary_impact'], 0)
        self.assertGreater(impact['permanent_impact'], 0)
        self.assertEqual(impact['total_impact'], 
                         impact['temporary_impact'] + impact['permanent_impact'])
    
    def test_direction_adjustment(self):
        """Test direction adjustment for buy/sell orders."""
        price = 100.0
        quantity = 1000
        
        # Calculate impact for buy order
        buy_impact = self.impact_model.calculate_market_impact(
            price, quantity, self.market_data, is_buy=True)
        
        # Calculate impact for sell order
        sell_impact = self.impact_model.calculate_market_impact(
            price, quantity, self.market_data, is_buy=False)
        
        # Buy impact should be positive, sell impact should be negative
        self.assertGreater(buy_impact['total_impact'], 0)
        self.assertLess(sell_impact['total_impact'], 0)
        
        # Absolute values should be the same
        self.assertAlmostEqual(abs(buy_impact['total_impact']), 
                              abs(sell_impact['total_impact']), places=6)


class TestTransactionCostAnalyzer(unittest.TestCase):
    """Test cases for transaction cost analyzer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create exchange fee model
        fee_config = ExchangeFeeConfig(
            maker_fee=1.0, taker_fee=3.0, minimum_fee=1.0)
        fee_model = ExchangeFeeModel(ExchangeType.FUTURES_EXCHANGE, fee_config)
        
        # Create slippage model
        slippage_model = FixedSlippageModel(basis_points=5.0)
        
        # Create market impact model
        impact_model = MarketImpactModel()
        
        # Create transaction cost analyzer
        self.analyzer = TransactionCostAnalyzer(
            exchange_fee_model=fee_model,
            slippage_model=slippage_model,
            market_impact_model=impact_model
        )
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(30)],
            'close': [100 + i * 0.1 for i in range(30)],
            'volume': [10000 for _ in range(30)]
        })
        
        # Create test trades
        self.trades = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5)],
            'price': [100, 101, 102, 101, 100],
            'quantity': [10, 5, 8, 12, 7],
            'is_buy': [True, False, True, True, False],
            'is_maker': [False, True, False, False, True],
            'symbol': ['ES', 'ES', 'NQ', 'ES', 'NQ']
        })
    
    def test_calculate_transaction_costs(self):
        """Test transaction cost calculation for a single trade."""
        price = 100.0
        quantity = 10
        
        # Calculate transaction costs
        costs = self.analyzer.calculate_transaction_costs(
            price, quantity, self.market_data, is_buy=True, is_maker=False)
        
        # Validate result structure
        self.assertIn('order_value', costs)
        self.assertIn('exchange_fee', costs)
        self.assertIn('slippage', costs)
        self.assertIn('impact_cost', costs)
        self.assertIn('total_cost', costs)
        self.assertIn('cost_percentage', costs)
        
        # Check calculations
        self.assertEqual(costs['order_value'], price * quantity)
        self.assertEqual(costs['total_cost'], 
                        costs['exchange_fee'] + costs['slippage'] + costs['impact_cost'])
    
    def test_analyze_strategy_costs(self):
        """Test analysis of transaction costs for a series of trades."""
        # Analyze strategy costs
        analysis = self.analyzer.analyze_strategy_costs(self.trades, self.market_data)
        
        # Validate result structure
        self.assertIn('total_trades', analysis)
        self.assertIn('total_volume', analysis)
        self.assertIn('total_value', analysis)
        self.assertIn('total_cost', analysis)
        self.assertIn('total_cost_percentage', analysis)
        self.assertIn('fee_cost', analysis)
        self.assertIn('slippage_cost', analysis)
        self.assertIn('impact_cost', analysis)
        
        # Check basic calculations
        self.assertEqual(analysis['total_trades'], len(self.trades))
        self.assertEqual(analysis['total_volume'], self.trades['quantity'].sum())


class TestPreConfiguredModels(unittest.TestCase):
    """Test cases for pre-configured models."""
    
    def test_futures_exchange_fee_model(self):
        """Test pre-configured futures exchange fee model."""
        model = create_futures_exchange_fee_model()
        self.assertEqual(model.exchange_type, ExchangeType.FUTURES_EXCHANGE)
        
        # Test basic fee calculation
        fee = model.calculate_fee(100000.0, is_maker=False)
        self.assertGreater(fee, 0)
    
    def test_stock_exchange_fee_model(self):
        """Test pre-configured stock exchange fee model."""
        model = create_stock_exchange_fee_model()
        self.assertEqual(model.exchange_type, ExchangeType.STOCK_EXCHANGE)
        
        # Test maker rebate
        fee = model.calculate_fee(100000.0, is_maker=True)
        self.assertGreaterEqual(fee, 0)  # Should not go below zero even with rebate
    
    def test_crypto_exchange_fee_model(self):
        """Test pre-configured crypto exchange fee model."""
        model = create_crypto_exchange_fee_model()
        self.assertEqual(model.exchange_type, ExchangeType.CRYPTO_EXCHANGE)
        
        # Test higher fees typical of crypto exchanges
        fee = model.calculate_fee(100000.0, is_maker=False)
        self.assertGreater(fee, 0)


class TestCustomSlippageModel(unittest.TestCase):
    """Test cases for custom slippage model."""
    
    def test_custom_slippage_function(self):
        """Test custom slippage function."""
        # Define a custom slippage function
        def custom_slippage(price, quantity, market_data):
            return price * quantity * 0.001  # 10 bps fixed slippage
        
        # Create custom slippage model
        model = CustomSlippageModel(slippage_function=custom_slippage)
        
        # Test slippage calculation
        price = 100.0
        quantity = 10
        slippage = model.calculate_slippage(price, quantity)
        
        # Expected result
        expected_slippage = price * quantity * 0.001
        self.assertEqual(slippage, expected_slippage)


if __name__ == '__main__':
    unittest.main() 