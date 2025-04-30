#!/usr/bin/env python3
"""
Unit tests for backtest tasks.

These tests validate the functionality of the task system for running backtests,
ensuring that backtest tasks are submitted, processed, and results are stored correctly.
"""

import os
import sys
import unittest
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import modules to test
try:
    from tasks.backtest_tasks import run_backtest_task, get_backtest_status, get_backtest_results
    from src.backtest.backtest_engine import BacktestEngine
except ImportError:
    # Use mock classes if actual implementations don't exist yet
    run_backtest_task = MagicMock()
    get_backtest_status = MagicMock()
    get_backtest_results = MagicMock()
    BacktestEngine = MagicMock()


class TestBacktestTasks(unittest.TestCase):
    """Test suite for backtest tasks."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test configuration
        self.test_config = {
            "pair": "CL-HO",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "hedge_ratio": 0.8,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "stop_loss": 3.0,
            "lookback_period": 20,
            "max_holding_period": 10
        }
        
        # Create test pairs data
        self.test_pairs = [
            {
                "asset1": "CL",
                "asset2": "HO",
                "hedge_ratio": 0.8,
                "half_life": 7,
                "zscore_entry": 2.0,
                "zscore_exit": 0.5,
                "lookback_period": 20
            }
        ]
        
        # Save test configuration to file
        config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(self.test_config, f)
        
        self.config_path = config_path
        
        # Generate test results
        self.test_results = {
            "task_id": "test_task_123",
            "status": "completed",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "trades": [
                {
                    "entry_time": "2022-03-15T10:30:00",
                    "exit_time": "2022-03-15T14:45:00",
                    "entry_price_1": 100.0,
                    "entry_price_2": 120.0,
                    "exit_price_1": 101.0,
                    "exit_price_2": 119.5,
                    "quantity_1": 1,
                    "quantity_2": -0.8,
                    "pnl": 1.6,
                    "exit_reason": "target"
                }
            ],
            "metrics": {
                "total_trades": 1,
                "winning_trades": 1,
                "losing_trades": 0,
                "total_pnl": 1.6,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.5
            }
        }
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    @patch("tasks.backtest_tasks.run_backtest_task.delay")
    def test_submit_backtest_task(self, mock_delay):
        """Test that a backtest task can be submitted."""
        # Configure mock
        mock_task_id = "test_task_123"
        mock_delay.return_value.id = mock_task_id
        
        # Import here to avoid circular import issues with mocks
        from tasks.backtest_tasks import submit_backtest_task
        
        # Submit task
        task_id = submit_backtest_task(self.config_path)
        
        # Verify task was submitted
        self.assertEqual(task_id, mock_task_id)
        mock_delay.assert_called_once_with(self.config_path)
    
    @patch("src.backtest.backtest_engine.BacktestEngine")
    def test_run_backtest_task(self, mock_backtest_engine):
        """Test running a backtest task."""
        # Configure mock
        mock_instance = mock_backtest_engine.return_value
        mock_instance.run_backtest.return_value = self.test_results["metrics"]
        mock_instance.get_trades.return_value = self.test_results["trades"]
        
        # Run backtest task
        result = run_backtest_task(self.config_path)
        
        # Verify backtest was run with correct parameters
        mock_backtest_engine.assert_called_once()
        mock_instance.run_backtest.assert_called_once()
        
        # Verify result format
        self.assertIn("metrics", result)
        self.assertIn("trades", result)
        self.assertEqual(result["metrics"], self.test_results["metrics"])
        self.assertEqual(result["trades"], self.test_results["trades"])
    
    @patch("tasks.backtest_tasks.AsyncResult")
    def test_get_backtest_status(self, mock_async_result):
        """Test getting backtest task status."""
        # Configure mock
        mock_instance = mock_async_result.return_value
        mock_instance.state = "SUCCESS"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = True
        
        # Get status
        status = get_backtest_status("test_task_123")
        
        # Verify status retrieval
        mock_async_result.assert_called_once_with("test_task_123")
        self.assertEqual(status, "SUCCESS")
    
    @patch("tasks.backtest_tasks.AsyncResult")
    def test_get_backtest_results(self, mock_async_result):
        """Test getting backtest results."""
        # Configure mock
        mock_instance = mock_async_result.return_value
        mock_instance.state = "SUCCESS"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = True
        mock_instance.result = self.test_results
        
        # Get results
        results = get_backtest_results("test_task_123")
        
        # Verify results retrieval
        mock_async_result.assert_called_once_with("test_task_123")
        self.assertEqual(results, self.test_results)
    
    @patch("tasks.backtest_tasks.AsyncResult")
    def test_backtest_failure_handling(self, mock_async_result):
        """Test handling of failed backtest tasks."""
        # Configure mock for failed task
        mock_instance = mock_async_result.return_value
        mock_instance.state = "FAILURE"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = False
        mock_instance.failed.return_value = True
        mock_instance.traceback = "Test traceback"
        
        # Import here to avoid circular import issues with mocks
        from tasks.backtest_tasks import get_backtest_error
        
        # Get error
        error = get_backtest_error("test_task_123")
        
        # Verify error retrieval
        mock_async_result.assert_called_once_with("test_task_123")
        self.assertIn("error", error)
        self.assertIn("traceback", error)
    
    def test_backtest_with_real_data(self):
        """Test running a backtest with real test data.
        
        This test uses the test data generation script to create test data,
        then runs a real backtest on that data.
        """
        try:
            # Skip if BacktestEngine is mocked
            if isinstance(BacktestEngine, MagicMock):
                self.skipTest("Actual BacktestEngine implementation not available")
            
            # First check if the test data exists
            test_data_path = "data/historical/CL/1day/data.csv"
            if not os.path.exists(test_data_path):
                # Generate test data
                from scripts.generate_test_data import main as generate_data
                generate_data()
                
                # Verify data was generated
                self.assertTrue(os.path.exists(test_data_path), 
                               "Test data generation failed")
            
            # Run the backtest
            engine = BacktestEngine(self.test_config)
            results = engine.run_backtest()
            
            # Basic validation of results
            self.assertIsNotNone(results)
            self.assertIn("total_trades", results)
            self.assertIn("total_pnl", results)
            
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Required modules not available: {e}")


if __name__ == "__main__":
    unittest.main() 