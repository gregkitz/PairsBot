"""
Unit tests for execution algorithms.

This test suite validates the implementation of execution algorithms including:
- Market execution algorithm
- TWAP (Time-Weighted Average Price) algorithm
- VWAP (Volume-Weighted Average Price) algorithm
- Adaptive execution algorithm
- Iceberg execution algorithm
- Execution simulator
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the module under test
from src.execution.execution_algorithms import (
    ExecutionType, ExecutionConfig, TWAPConfig, VWAPConfig, AdaptiveConfig,
    ExecutionAlgorithm, MarketExecutionAlgorithm, TWAPExecutionAlgorithm,
    VWAPExecutionAlgorithm, AdaptiveExecutionAlgorithm, IcebergExecutionAlgorithm,
    ExecutionSimulator
)


class TestMarketExecutionAlgorithm(unittest.TestCase):
    """Test cases for market execution algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create market execution algorithm
        self.algorithm = MarketExecutionAlgorithm()
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'price': [100 + i * 0.1 for i in range(10)],
            'volume': [1000 * (i + 1) for i in range(10)]
        })
    
    def test_generate_execution_plan(self):
        """Test generation of market execution plan."""
        quantity = 100
        
        # Generate execution plan
        plan = self.algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check plan structure
        self.assertEqual(len(plan), 1)  # Market orders execute in a single slice
        self.assertIn('timestamp', plan.columns)
        self.assertIn('quantity', plan.columns)
        self.assertIn('price', plan.columns)
        self.assertIn('type', plan.columns)
        
        # Check plan details
        self.assertEqual(plan['quantity'].iloc[0], quantity)
        self.assertEqual(plan['price'].iloc[0], self.market_data['price'].iloc[-1])
        self.assertEqual(plan['type'].iloc[0], 'market')
    
    def test_validation_error(self):
        """Test error when invalid market data is provided."""
        # Create invalid market data (missing price)
        invalid_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(5)],
            'volume': [1000 * (i + 1) for i in range(5)]
        })
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.algorithm.generate_execution_plan(100, invalid_data)


class TestTWAPExecutionAlgorithm(unittest.TestCase):
    """Test cases for TWAP execution algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create TWAP configuration
        self.config = TWAPConfig(
            num_slices=5,
            random_variance=0.1,
            max_participation=0.2
        )
        
        # Create TWAP execution algorithm
        self.algorithm = TWAPExecutionAlgorithm(config=self.config)
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)],
            'price': [100 + i * 0.1 for i in range(10)],
            'volume': [1000 * (i + 1) for i in range(10)]
        })
    
    def test_generate_execution_plan(self):
        """Test generation of TWAP execution plan."""
        quantity = 100
        
        # Generate execution plan
        plan = self.algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check plan structure
        self.assertEqual(len(plan), self.config.num_slices)
        self.assertIn('timestamp', plan.columns)
        self.assertIn('quantity', plan.columns)
        self.assertIn('price', plan.columns)
        self.assertIn('type', plan.columns)
        
        # Check quantities sum to total
        self.assertAlmostEqual(plan['quantity'].sum(), quantity, places=6)
        
        # Check timestamps are reasonably spaced
        timestamps = plan['timestamp'].tolist()
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i-1], timestamps[i])  # Earlier timestamps first
    
    def test_no_randomization(self):
        """Test TWAP with no randomization."""
        # Create TWAP configuration without randomization
        config = TWAPConfig(
            num_slices=5,
            random_variance=0.0,
            max_participation=0.2
        )
        
        # Create TWAP execution algorithm
        algorithm = TWAPExecutionAlgorithm(config=config)
        
        quantity = 100
        
        # Generate execution plan
        plan = algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check even distribution across slices
        expected_quantity_per_slice = quantity / config.num_slices
        for i in range(config.num_slices - 1):  # All but last slice should be equal
            self.assertAlmostEqual(plan['quantity'].iloc[i], expected_quantity_per_slice, places=6)


class TestVWAPExecutionAlgorithm(unittest.TestCase):
    """Test cases for VWAP execution algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create VWAP configuration
        self.config = VWAPConfig(
            max_participation=0.2,
            num_slices=4
        )
        
        # Create VWAP execution algorithm
        self.algorithm = VWAPExecutionAlgorithm(config=self.config)
        
        # Create test market data with different volumes by hour
        now = datetime.now()
        timestamps = []
        volumes = []
        prices = []
        
        for i in range(24):  # 24 hours of data
            hour_timestamps = [now.replace(hour=i, minute=j*15) for j in range(4)]
            timestamps.extend(hour_timestamps)
            
            # Create different volume patterns by hour
            if i < 8:  # Low volume hours
                hour_volumes = [500 + j*100 for j in range(4)]
            elif 8 <= i < 16:  # High volume hours
                hour_volumes = [2000 + j*200 for j in range(4)]
            else:  # Medium volume hours
                hour_volumes = [1000 + j*100 for j in range(4)]
                
            volumes.extend(hour_volumes)
            prices.extend([100 + i*0.1 + j*0.01 for j in range(4)])
        
        self.market_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
    
    def test_generate_execution_plan(self):
        """Test generation of VWAP execution plan."""
        quantity = 100
        
        # Generate execution plan
        plan = self.algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check plan structure
        self.assertGreater(len(plan), 0)
        self.assertIn('timestamp', plan.columns)
        self.assertIn('quantity', plan.columns)
        self.assertIn('price', plan.columns)
        self.assertIn('type', plan.columns)
        self.assertIn('participation_rate', plan.columns)
        
        # Check quantities sum to total
        self.assertAlmostEqual(plan['quantity'].sum(), quantity, places=6)
        
        # Check high volume hours get more quantity
        if len(plan) > 1:
            hour_quantities = {}
            for i, row in plan.iterrows():
                hour = row['timestamp'].hour
                if hour not in hour_quantities:
                    hour_quantities[hour] = 0
                hour_quantities[hour] += row['quantity']
            
            # Find high and low volume hours
            high_vol_hours = [h for h in hour_quantities.keys() if 8 <= h < 16]
            low_vol_hours = [h for h in hour_quantities.keys() if h < 8]
            
            if high_vol_hours and low_vol_hours:
                avg_high_vol_quantity = sum(hour_quantities[h] for h in high_vol_hours) / len(high_vol_hours)
                avg_low_vol_quantity = sum(hour_quantities[h] for h in low_vol_hours) / len(low_vol_hours)
                
                # High volume hours should get more quantity than low volume hours
                self.assertGreater(avg_high_vol_quantity, avg_low_vol_quantity)
    
    def test_volume_profile_estimation(self):
        """Test volume profile estimation."""
        # Test volume profile estimation method
        volume_profile = self.algorithm._estimate_volume_profile(self.market_data)
        
        # Should have entries for each hour
        unique_hours = self.market_data['timestamp'].dt.hour.unique()
        for hour in unique_hours:
            self.assertIn(hour, volume_profile)
        
        # Probabilities should sum to approximately 1
        self.assertAlmostEqual(sum(volume_profile.values()), 1.0, places=6)
        
        # High volume hours should have higher probabilities
        high_vol_hours = [h for h in volume_profile.keys() if 8 <= h < 16]
        low_vol_hours = [h for h in volume_profile.keys() if h < 8]
        
        if high_vol_hours and low_vol_hours:
            avg_high_vol_prob = sum(volume_profile[h] for h in high_vol_hours) / len(high_vol_hours)
            avg_low_vol_prob = sum(volume_profile[h] for h in low_vol_hours) / len(low_vol_hours)
            
            # High volume hours should have higher probabilities
            self.assertGreater(avg_high_vol_prob, avg_low_vol_prob)


class TestAdaptiveExecutionAlgorithm(unittest.TestCase):
    """Test cases for adaptive execution algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create adaptive configuration
        self.config = AdaptiveConfig(
            min_participation=0.05,
            max_participation=0.3,
            volatility_factor=1.2,
            urgency=0.6
        )
        
        # Create adaptive execution algorithm
        self.algorithm = AdaptiveExecutionAlgorithm(config=self.config)
        
        # Create test market data with varying volatility
        now = datetime.now()
        timestamps = []
        prices = []
        
        # First period - low volatility
        base_price = 100.0
        for i in range(20):
            timestamps.append(now - timedelta(minutes=i))
            prices.append(base_price + np.sin(i/10) * 0.05)  # Low volatility
        
        # Second period - higher volatility
        base_price = 100.0
        for i in range(20, 40):
            timestamps.append(now - timedelta(minutes=i))
            prices.append(base_price + np.sin(i/10) * 0.2)  # Higher volatility
        
        self.market_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices
        })
        
        # Add bid/ask data for spread calculation
        self.market_data_with_spread = self.market_data.copy()
        self.market_data_with_spread['bid'] = self.market_data_with_spread['price'] * 0.999
        self.market_data_with_spread['ask'] = self.market_data_with_spread['price'] * 1.001
        self.market_data_with_spread['mid'] = self.market_data_with_spread['price']
    
    def test_generate_execution_plan(self):
        """Test generation of adaptive execution plan."""
        quantity = 100
        
        # Generate execution plan
        plan = self.algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check plan structure
        self.assertGreater(len(plan), 0)
        self.assertIn('timestamp', plan.columns)
        self.assertIn('quantity', plan.columns)
        self.assertIn('price', plan.columns)
        self.assertIn('type', plan.columns)
        self.assertIn('participation_rate', plan.columns)
        
        # Check quantities sum to total
        self.assertAlmostEqual(plan['quantity'].sum(), quantity, places=6)
        
        # Check order types
        self.assertTrue(any(plan['type'] == 'limit') or any(plan['type'] == 'market'))
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Calculate volatility for low volatility period
        low_vol = self.algorithm._calculate_volatility(self.market_data.head(20))
        
        # Calculate volatility for higher volatility period
        high_vol = self.algorithm._calculate_volatility(self.market_data.tail(20))
        
        # Higher volatility period should have higher volatility
        self.assertGreater(high_vol, low_vol)
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        # Calculate spread with bid/ask data
        spread = self.algorithm._calculate_spread(self.market_data_with_spread)
        
        # Spread should be around 0.2% (0.002)
        self.assertAlmostEqual(spread, 0.002, places=4)
        
        # Test default spread when bid/ask not available
        default_spread = self.algorithm._calculate_spread(self.market_data)
        self.assertEqual(default_spread, 0.0001)  # Should return default 1 bps


class TestIcebergExecutionAlgorithm(unittest.TestCase):
    """Test cases for iceberg execution algorithm."""
    
    def setUp(self):
        """Set up test environment."""
        # Create configuration
        self.config = ExecutionConfig(
            max_participation=0.2
        )
        
        # Create iceberg execution algorithm
        self.visible_quantity = 10.0
        self.algorithm = IcebergExecutionAlgorithm(
            visible_quantity=self.visible_quantity,
            config=self.config
        )
        
        # Create test market data
        self.market_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(30)],
            'price': [100 + i * 0.1 for i in range(30)],
            'volume': [1000 for _ in range(30)]
        })
    
    def test_generate_execution_plan(self):
        """Test generation of iceberg execution plan."""
        quantity = 100
        
        # Generate execution plan
        plan = self.algorithm.generate_execution_plan(quantity, self.market_data)
        
        # Check plan structure
        expected_slices = int(np.ceil(quantity / self.visible_quantity))
        self.assertEqual(len(plan), expected_slices)
        
        self.assertIn('timestamp', plan.columns)
        self.assertIn('quantity', plan.columns)
        self.assertIn('price', plan.columns)
        self.assertIn('type', plan.columns)
        self.assertIn('display_quantity', plan.columns)
        
        # Check quantities sum to total
        self.assertAlmostEqual(plan['quantity'].sum(), quantity, places=6)
        
        # Check visible quantity
        for i in range(len(plan) - 1):  # All but last slice
            self.assertEqual(plan['quantity'].iloc[i], self.visible_quantity)
            self.assertEqual(plan['display_quantity'].iloc[i], self.visible_quantity)
        
        # Last slice might be partial
        self.assertLessEqual(plan['quantity'].iloc[-1], self.visible_quantity)


class TestExecutionSimulator(unittest.TestCase):
    """Test cases for execution simulator."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test market data
        now = datetime.now()
        self.market_data = pd.DataFrame({
            'timestamp': [now - timedelta(minutes=i) for i in range(60)],
            'price': [100 + np.sin(i/10) * 1.0 for i in range(60)],
            'volume': [1000 + 500 * np.sin(i/15) for i in range(60)]
        })
        
        # Create execution simulator
        self.simulator = ExecutionSimulator(
            market_data=self.market_data
        )
        
        # Create example execution plan (TWAP)
        timestamps = [now - timedelta(minutes=i*10) for i in range(5)]
        self.execution_plan = pd.DataFrame({
            'timestamp': timestamps,
            'quantity': [20.0] * 5,
            'price': [100.0, 99.5, 100.2, 100.5, 99.8],
            'type': ['market', 'limit', 'limit', 'market', 'limit'],
            'participation_rate': [0.1] * 5
        })
    
    def test_simulate_execution(self):
        """Test execution simulation."""
        # Simulate execution
        results = self.simulator.simulate_execution(self.execution_plan)
        
        # Check results structure
        self.assertEqual(len(results), len(self.execution_plan))
        self.assertIn('planned_time', results.columns)
        self.assertIn('actual_time', results.columns)
        self.assertIn('planned_price', results.columns)
        self.assertIn('actual_price', results.columns)
        self.assertIn('planned_quantity', results.columns)
        self.assertIn('executed_quantity', results.columns)
        self.assertIn('order_type', results.columns)
        self.assertIn('executed', results.columns)
        self.assertIn('slippage', results.columns)
        self.assertIn('slippage_value', results.columns)
        self.assertIn('latency', results.columns)
        
        # Check market orders are executed
        market_orders = results[results['order_type'] == 'market']
        self.assertTrue(all(market_orders['executed']))
        
        # Check latency is applied
        for i, row in results.iterrows():
            if row['executed']:
                self.assertGreater(row['actual_time'], row['planned_time'])
                self.assertGreater(row['latency'], 0.0)
    
    def test_evaluate_execution(self):
        """Test execution evaluation."""
        # Simulate execution
        results = self.simulator.simulate_execution(self.execution_plan)
        
        # Evaluate execution
        evaluation = self.simulator.evaluate_execution(results)
        
        # Check evaluation structure
        self.assertIn('fill_rate', evaluation)
        self.assertIn('average_slippage_bps', evaluation)
        self.assertIn('total_slippage_value', evaluation)
        self.assertIn('average_latency', evaluation)
        self.assertIn('implementation_shortfall', evaluation)
        self.assertIn('executed_quantity', evaluation)
        self.assertIn('planned_quantity', evaluation)
        
        # Check fill rate calculation
        expected_fill_rate = results['executed_quantity'].sum() / results['planned_quantity'].sum()
        self.assertEqual(evaluation['fill_rate'], expected_fill_rate)
        
        # Check planned and executed quantities
        self.assertEqual(evaluation['planned_quantity'], self.execution_plan['quantity'].sum())
        self.assertEqual(evaluation['executed_quantity'], results['executed_quantity'].sum())
    
    def test_custom_models(self):
        """Test custom models for simulator."""
        # Define custom models
        def custom_spread_model(data):
            return data['price'] * 0.002  # 20 bps spread
        
        def custom_volatility_model(data):
            return 0.02  # Fixed 2% volatility
        
        def custom_latency_model():
            return 0.5  # Fixed 500ms latency
        
        # Create simulator with custom models
        simulator = ExecutionSimulator(
            market_data=self.market_data,
            spread_model=custom_spread_model,
            volatility_model=custom_volatility_model,
            latency_model=custom_latency_model
        )
        
        # Simulate execution
        results = simulator.simulate_execution(self.execution_plan)
        
        # Check custom latency is applied
        for i, row in results.iterrows():
            if row['executed']:
                self.assertAlmostEqual(row['latency'], 0.5, places=6)
        
        # Market orders should have higher slippage due to wider spread
        market_orders = results[results['order_type'] == 'market']
        if not market_orders.empty:
            for i, row in market_orders.iterrows():
                if row['executed']:
                    self.assertGreater(abs(row['slippage']), 0.0005)  # Should have significant slippage


if __name__ == '__main__':
    unittest.main() 