#!/usr/bin/env python3
"""
Unit tests for optimization tasks.

These tests validate the functionality of the task system for running parameter optimizations,
ensuring that optimization tasks are submitted, processed, and results are stored correctly.
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
    from tasks.optimization_tasks import run_optimization_task, get_optimization_status, get_optimization_results
    from src.optimization.parameter_optimizer import ParameterOptimizer
except ImportError:
    # Use mock classes if actual implementations don't exist yet
    run_optimization_task = MagicMock()
    get_optimization_status = MagicMock()
    get_optimization_results = MagicMock()
    ParameterOptimizer = MagicMock()


class TestOptimizationTasks(unittest.TestCase):
    """Test suite for optimization tasks."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test configuration
        self.test_config = {
            "pair": "CL-HO",
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "optimization_method": "grid",
            "optimization_metric": "sharpe_ratio",
            "parameters": {
                "hedge_ratio": [0.7, 0.8, 0.9],
                "entry_threshold": [1.5, 2.0, 2.5],
                "exit_threshold": [0.3, 0.5, 0.7],
                "lookback_period": [10, 20, 30]
            },
            "max_iterations": 100,
            "crossover_rate": 0.7,
            "mutation_rate": 0.2
        }
        
        # Save test configuration to file
        config_path = os.path.join(self.temp_dir.name, "test_optimization_config.json")
        with open(config_path, "w") as f:
            json.dump(self.test_config, f)
        
        self.config_path = config_path
        
        # Generate test results
        self.test_results = {
            "task_id": "test_optimization_123",
            "status": "completed",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "best_parameters": {
                "hedge_ratio": 0.8,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "lookback_period": 20
            },
            "optimization_history": [
                {
                    "iteration": 0,
                    "parameters": {
                        "hedge_ratio": 0.7,
                        "entry_threshold": 1.5,
                        "exit_threshold": 0.3,
                        "lookback_period": 10
                    },
                    "metrics": {
                        "sharpe_ratio": 0.8,
                        "total_pnl": 5.2,
                        "max_drawdown": 2.1
                    }
                },
                {
                    "iteration": 1,
                    "parameters": {
                        "hedge_ratio": 0.8,
                        "entry_threshold": 2.0,
                        "exit_threshold": 0.5,
                        "lookback_period": 20
                    },
                    "metrics": {
                        "sharpe_ratio": 1.2,
                        "total_pnl": 7.5,
                        "max_drawdown": 1.8
                    }
                }
            ],
            "best_metrics": {
                "sharpe_ratio": 1.2,
                "total_pnl": 7.5,
                "max_drawdown": 1.8,
                "total_trades": 25,
                "winning_trades": 15,
                "losing_trades": 10
            }
        }
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    @patch("tasks.optimization_tasks.run_optimization_task.delay")
    def test_submit_optimization_task(self, mock_delay):
        """Test that an optimization task can be submitted."""
        # Configure mock
        mock_task_id = "test_optimization_123"
        mock_delay.return_value.id = mock_task_id
        
        # Import here to avoid circular import issues with mocks
        from tasks.optimization_tasks import submit_optimization_task
        
        # Submit task
        task_id = submit_optimization_task(self.config_path)
        
        # Verify task was submitted
        self.assertEqual(task_id, mock_task_id)
        mock_delay.assert_called_once_with(self.config_path)
    
    @patch("src.optimization.parameter_optimizer.ParameterOptimizer")
    def test_run_optimization_task(self, mock_optimizer):
        """Test running an optimization task."""
        # Configure mock
        mock_instance = mock_optimizer.return_value
        mock_instance.optimize.return_value = {
            "best_parameters": self.test_results["best_parameters"],
            "optimization_history": self.test_results["optimization_history"],
            "best_metrics": self.test_results["best_metrics"]
        }
        
        # Run optimization task
        result = run_optimization_task(self.config_path)
        
        # Verify optimizer was run with correct parameters
        mock_optimizer.assert_called_once()
        mock_instance.optimize.assert_called_once()
        
        # Verify result format
        self.assertIn("best_parameters", result)
        self.assertIn("optimization_history", result)
        self.assertIn("best_metrics", result)
        self.assertEqual(result["best_parameters"], self.test_results["best_parameters"])
        self.assertEqual(result["best_metrics"], self.test_results["best_metrics"])
    
    @patch("tasks.optimization_tasks.AsyncResult")
    def test_get_optimization_status(self, mock_async_result):
        """Test getting optimization task status."""
        # Configure mock
        mock_instance = mock_async_result.return_value
        mock_instance.state = "SUCCESS"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = True
        
        # Get status
        status = get_optimization_status("test_optimization_123")
        
        # Verify status retrieval
        mock_async_result.assert_called_once_with("test_optimization_123")
        self.assertEqual(status, "SUCCESS")
    
    @patch("tasks.optimization_tasks.AsyncResult")
    def test_get_optimization_results(self, mock_async_result):
        """Test getting optimization results."""
        # Configure mock
        mock_instance = mock_async_result.return_value
        mock_instance.state = "SUCCESS"
        mock_instance.ready.return_value = True
        mock_instance.successful.return_value = True
        mock_instance.result = self.test_results
        
        # Get results
        results = get_optimization_results("test_optimization_123")
        
        # Verify results retrieval
        mock_async_result.assert_called_once_with("test_optimization_123")
        self.assertEqual(results, self.test_results)
    
    @patch("tasks.optimization_tasks.AsyncResult")
    def test_optimization_progress(self, mock_async_result):
        """Test getting optimization progress."""
        # Configure mock for in-progress task
        mock_instance = mock_async_result.return_value
        mock_instance.state = "PROGRESS"
        mock_instance.ready.return_value = False
        mock_instance.info = {
            "current": 25,
            "total": 100,
            "status": "optimizing",
            "current_best": {
                "parameters": self.test_results["best_parameters"],
                "metrics": self.test_results["best_metrics"]
            }
        }
        
        # Import here to avoid circular import issues with mocks
        from tasks.optimization_tasks import get_optimization_progress
        
        # Get progress
        progress = get_optimization_progress("test_optimization_123")
        
        # Verify progress retrieval
        mock_async_result.assert_called_once_with("test_optimization_123")
        self.assertIn("current", progress)
        self.assertIn("total", progress)
        self.assertIn("percentage", progress)
        self.assertEqual(progress["percentage"], 25)
        self.assertIn("current_best", progress)
    
    def test_optimization_method_selection(self):
        """Test selection of optimization methods."""
        try:
            # Skip if ParameterOptimizer is mocked
            if isinstance(ParameterOptimizer, MagicMock):
                self.skipTest("Actual ParameterOptimizer implementation not available")
            
            # Test each optimization method
            for method in ["grid", "genetic", "bayesian"]:
                # Create configuration with specified method
                config = self.test_config.copy()
                config["optimization_method"] = method
                
                # Save modified configuration
                method_config_path = os.path.join(self.temp_dir.name, f"test_{method}_config.json")
                with open(method_config_path, "w") as f:
                    json.dump(config, f)
                
                # Create optimizer and verify method selection
                optimizer = ParameterOptimizer(method_config_path)
                self.assertEqual(optimizer.method, method)
                
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_optimization_with_real_data(self):
        """Test running an optimization with real test data.
        
        This test uses the test data generation script to create test data,
        then runs a real optimization on that data.
        """
        try:
            # Skip if ParameterOptimizer is mocked
            if isinstance(ParameterOptimizer, MagicMock):
                self.skipTest("Actual ParameterOptimizer implementation not available")
            
            # First check if the test data exists
            test_data_path = "data/historical/CL/1day/data.csv"
            if not os.path.exists(test_data_path):
                # Generate test data
                from scripts.generate_test_data import main as generate_data
                generate_data()
                
                # Verify data was generated
                self.assertTrue(os.path.exists(test_data_path), 
                               "Test data generation failed")
            
            # Run the optimization with reduced parameter space for quick testing
            quick_config = self.test_config.copy()
            quick_config["parameters"] = {
                "entry_threshold": [1.5, 2.0],
                "exit_threshold": [0.5, 0.7]
            }
            quick_config["max_iterations"] = 4
            
            # Save modified configuration
            quick_config_path = os.path.join(self.temp_dir.name, "quick_test_config.json")
            with open(quick_config_path, "w") as f:
                json.dump(quick_config, f)
            
            # Run optimization
            optimizer = ParameterOptimizer(quick_config_path)
            results = optimizer.optimize()
            
            # Basic validation of results
            self.assertIsNotNone(results)
            self.assertIn("best_parameters", results)
            self.assertIn("best_metrics", results)
            self.assertGreaterEqual(len(results["optimization_history"]), 1)
            
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Required modules not available: {e}")


if __name__ == "__main__":
    unittest.main() 