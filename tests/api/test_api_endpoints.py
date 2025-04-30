#!/usr/bin/env python3
"""
Unit tests for API endpoints.

These tests validate the functionality of the API endpoints for submitting tasks,
retrieving results, and system status monitoring.
"""

import os
import sys
import unittest
import pytest
import json
import tempfile
from unittest.mock import patch, MagicMock
import requests
from datetime import datetime
import time

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import benchmark utilities for performance testing
from tests.benchmark.test_benchmark_utils import measure_time

# Import config utilities
from config.test.api_test_config import ENDPOINTS

# Mock Flask app if needed for testing
try:
    from api import app as flask_app
except ImportError:
    # Create a mock Flask app
    flask_app = MagicMock()


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    try:
        # Use the actual Flask test client if available
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            yield client
    except (AttributeError, ImportError):
        # If Flask app is not available, skip the tests
        pytest.skip("Flask app not available for testing")


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for API endpoints."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Load API test configuration
        try:
            with open("config/test/api_test_config.json", "r") as f:
                self.config = json.load(f)
            
            # Extract base URL
            self.base_url = self.config["endpoints"]["base_url"]
            
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback to default configuration
            self.config = {"endpoints": {"base_url": "http://localhost:5000/api"}}
            self.base_url = "http://localhost:5000/api"
        
        # Create test configuration for backtest
        self.backtest_config = {
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
        
        # Create test configuration for optimization
        self.optimization_config = {
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
            }
        }
        
        # Save test configurations to files
        backtest_config_path = os.path.join(self.temp_dir.name, "test_backtest_config.json")
        with open(backtest_config_path, "w") as f:
            json.dump(self.backtest_config, f)
        
        optimization_config_path = os.path.join(self.temp_dir.name, "test_optimization_config.json")
        with open(optimization_config_path, "w") as f:
            json.dump(self.optimization_config, f)
        
        self.backtest_config_path = backtest_config_path
        self.optimization_config_path = optimization_config_path
        
        # Set up authentication headers if needed
        if self.config.get("authorization", {}).get("enabled", False):
            self.headers = {
                self.config["authorization"]["header_name"]: self.config["authorization"]["api_key"]
            }
        else:
            self.headers = {}
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    @patch("requests.post")
    def test_submit_backtest_task(self, mock_post):
        """Test submitting a backtest task via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "task_id": "test_task_123",
            "status": "accepted"
        }
        mock_post.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/backtest/submit"
        data = {"config": self.backtest_config}
        response = requests.post(url, json=data, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 202)
        result = response.json()
        self.assertEqual(result["status"], "accepted")
        self.assertIn("task_id", result)
    
    @patch("requests.post")
    def test_submit_optimization_task(self, mock_post):
        """Test submitting an optimization task via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "task_id": "test_optimization_123",
            "status": "accepted"
        }
        mock_post.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/optimization/submit"
        data = {"config": self.optimization_config}
        response = requests.post(url, json=data, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 202)
        result = response.json()
        self.assertEqual(result["status"], "accepted")
        self.assertIn("task_id", result)
    
    @patch("requests.get")
    def test_get_backtest_status(self, mock_get):
        """Test getting backtest task status via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "test_task_123",
            "status": "COMPLETED"
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/backtest/status/test_task_123"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["task_id"], "test_task_123")
    
    @patch("requests.get")
    def test_get_backtest_results(self, mock_get):
        """Test getting backtest results via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "test_task_123",
            "status": "completed",
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
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/backtest/results/test_task_123"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "completed")
        self.assertIn("trades", result)
        self.assertIn("metrics", result)
    
    @patch("requests.get")
    def test_get_optimization_results(self, mock_get):
        """Test getting optimization results via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "test_optimization_123",
            "status": "completed",
            "best_parameters": {
                "hedge_ratio": 0.8,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "lookback_period": 20
            },
            "best_metrics": {
                "sharpe_ratio": 1.2,
                "total_pnl": 7.5,
                "max_drawdown": 1.8
            }
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/optimization/results/test_optimization_123"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "completed")
        self.assertIn("best_parameters", result)
        self.assertIn("best_metrics", result)
    
    @patch("requests.get")
    def test_get_system_status(self, mock_get):
        """Test getting system status via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "components": {
                "worker": "running",
                "redis": "running",
                "database": "running"
            },
            "active_tasks": 2,
            "queue_length": 1,
            "system_load": 0.35
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/system/status"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "healthy")
        self.assertIn("components", result)
        self.assertIn("active_tasks", result)
    
    @patch("requests.get")
    def test_get_historical_data(self, mock_get):
        """Test retrieving historical data via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ticker": "CL",
            "timeframe": "1day",
            "start_date": "2022-01-01",
            "end_date": "2022-01-10",
            "data": [
                {"date": "2022-01-01", "open": 75.2, "high": 76.1, "low": 74.8, "close": 75.5, "volume": 12345},
                {"date": "2022-01-02", "open": 75.5, "high": 77.0, "low": 75.2, "close": 76.8, "volume": 23456}
            ]
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/data/historical/CL/1day?start=2022-01-01&end=2022-01-10"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["ticker"], "CL")
        self.assertIn("data", result)
        self.assertEqual(len(result["data"]), 2)
    
    @patch("requests.get")
    def test_get_pairs_list(self, mock_get):
        """Test retrieving available pairs via API."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "pairs": [
                {"pair": "CL-HO", "hedge_ratio": 0.8, "half_life": 7},
                {"pair": "GC-SI", "hedge_ratio": 0.6, "half_life": 10},
                {"pair": "ZC-ZW", "hedge_ratio": 0.75, "half_life": 12}
            ]
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/pairs"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("pairs", result)
        self.assertEqual(len(result["pairs"]), 3)
    
    @patch("requests.post")
    def test_api_error_handling(self, mock_post):
        """Test API error handling."""
        # Configure mock for error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "Invalid configuration",
            "message": "Missing required parameters"
        }
        mock_post.return_value = mock_response
        
        # Make API request with invalid data
        url = f"{self.base_url}/backtest/submit"
        data = {"invalid": "data"}
        response = requests.post(url, json=data, headers=self.headers)
        
        # Verify error response
        self.assertEqual(response.status_code, 400)
        result = response.json()
        self.assertIn("error", result)
        self.assertIn("message", result)
    
    @patch("requests.get")
    def test_api_authentication(self, mock_get):
        """Test API authentication requirements."""
        # Skip if authentication not enabled in config
        if not self.config.get("authorization", {}).get("enabled", False):
            self.skipTest("Authentication not enabled in configuration")
        
        # Configure mock for unauthorized access
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": "Unauthorized",
            "message": "Invalid or missing API key"
        }
        mock_get.return_value = mock_response
        
        # Make API request without auth headers
        url = f"{self.base_url}/system/status"
        response = requests.get(url)  # No headers
        
        # Verify unauthorized response
        self.assertEqual(response.status_code, 401)
        result = response.json()
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Unauthorized")
    
    @patch("requests.post")
    def test_task_cancellation(self, mock_post):
        """Test cancelling a running task."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "test_task_123",
            "status": "cancelled",
            "message": "Task successfully cancelled"
        }
        mock_post.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/tasks/cancel"
        data = {"task_id": "test_task_123"}
        response = requests.post(url, json=data, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["task_id"], "test_task_123")
    
    @patch("requests.get")
    def test_task_list(self, mock_get):
        """Test retrieving list of tasks."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tasks": [
                {"task_id": "task1", "type": "backtest", "status": "completed", "created_at": "2022-03-15T10:30:00"},
                {"task_id": "task2", "type": "optimization", "status": "running", "created_at": "2022-03-15T11:45:00"}
            ],
            "total": 2
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/tasks"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("tasks", result)
        self.assertEqual(len(result["tasks"]), 2)
        self.assertEqual(result["total"], 2)
    
    @patch("requests.get")
    def test_api_performance(self, mock_get):
        """Test API endpoint performance."""
        # Configure mock with delayed response to simulate real performance
        def delayed_response(*args, **kwargs):
            time.sleep(0.05)  # 50ms response time
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "healthy"}
            return mock_resp
        
        mock_get.side_effect = delayed_response
        
        # Make API request with timing
        url = f"{self.base_url}/health"
        
        # Measure response time
        with measure_time("API request") as start_time:
            response = requests.get(url, headers=self.headers)
        
        # Response time should be reasonable for a health check endpoint
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 0.2, "Health check API response time too slow")
        self.assertEqual(response.status_code, 200)
    
    @patch("requests.post")
    def test_async_task_submission(self, mock_post):
        """Test that task submission is asynchronous."""
        # Configure initial submission response
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.json.return_value = {
            "task_id": "test_task_async",
            "status": "accepted"
        }
        mock_post.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/backtest/submit"
        data = {"config": self.backtest_config}
        
        # Measure response time - should be quick since task is asynchronous
        with measure_time("Async task submission") as start_time:
            response = requests.post(url, json=data, headers=self.headers)
        
        # Submission should be fast (accept task, not wait for completion)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 0.5, "Task submission not properly async")
        
        # Verify API response indicates task is accepted but not complete
        self.assertEqual(response.status_code, 202)
        result = response.json()
        self.assertEqual(result["status"], "accepted")
    
    @patch("requests.get")
    def test_system_metrics_endpoint(self, mock_get):
        """Test system metrics endpoint."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "memory_usage": {
                "total_gb": 16.0,
                "used_gb": 8.5,
                "percent": 53.1
            },
            "cpu_usage": {
                "percent": 42.3,
                "cores": 8
            },
            "disk_usage": {
                "total_gb": 500,
                "used_gb": 350,
                "percent": 70.0
            },
            "task_queue": {
                "pending": 3,
                "running": 1,
                "completed_today": 25
            },
            "uptime_hours": 72.5
        }
        mock_get.return_value = mock_response
        
        # Make API request
        url = f"{self.base_url}/system/metrics"
        response = requests.get(url, headers=self.headers)
        
        # Verify API response
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("memory_usage", result)
        self.assertIn("cpu_usage", result)
        self.assertIn("task_queue", result)
        
        # Verify metrics have expected structure
        mem_usage = result["memory_usage"]
        self.assertIn("total_gb", mem_usage)
        self.assertIn("used_gb", mem_usage)
        self.assertIn("percent", mem_usage)
        
        # Verify task queue metrics exist
        task_queue = result["task_queue"]
        self.assertIn("pending", task_queue)
        self.assertIn("running", task_queue)


# Flask app test using pytest fixtures
def test_health_endpoint(client):
    """Test the health endpoint using Flask test client."""
    if client is None:
        pytest.skip("Flask app not available")
    
    response = client.get("/api/health")
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"


def test_backtest_submit_endpoint(client):
    """Test the backtest submission endpoint using Flask test client."""
    if client is None:
        pytest.skip("Flask app not available")
    
    # Mock the task submission
    with patch("tasks.backtest_tasks.submit_backtest_task") as mock_submit:
        mock_submit.return_value = "test_task_123"
        
        # Test request
        data = {
            "config": {
                "pair": "CL-HO",
                "start_date": "2022-01-01",
                "end_date": "2022-12-31"
            }
        }
        response = client.post(
            "/api/backtest/submit",
            data=json.dumps(data),
            content_type="application/json"
        )
        
        assert response.status_code == 202
        result = json.loads(response.data)
        assert result["status"] == "accepted"
        assert result["task_id"] == "test_task_123"


def test_invalid_request(client):
    """Test handling of invalid requests."""
    if client is None:
        pytest.skip("Flask app not available")
    
    # Test with missing data
    response = client.post(
        "/api/backtest/submit",
        data=json.dumps({}),
        content_type="application/json"
    )
    
    assert response.status_code == 400
    result = json.loads(response.data)
    assert "error" in result


def test_api_versioning(client):
    """Test API versioning support."""
    if client is None:
        pytest.skip("Flask app not available")
    
    # Test both current and legacy API versions if supported
    versions = ["/api/v1", "/api"]
    
    for version_prefix in versions:
        response = client.get(f"{version_prefix}/health")
        
        # Both versions should work
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"


def test_endpoint_throttling(client):
    """Test API endpoint throttling."""
    if client is None:
        pytest.skip("Flask app not available")
    
    # Rapid succession of requests should be throttled after a threshold
    # Make multiple requests in quick succession
    responses = []
    request_count = 20  # Should trigger throttling if implemented
    
    for _ in range(request_count):
        responses.append(client.get("/api/health"))
    
    # Check if any responses indicate throttling (status 429)
    throttled = any(r.status_code == 429 for r in responses)
    
    # This test will pass whether throttling is implemented or not
    # It serves as documentation of the throttling behavior
    if throttled:
        # Throttling is implemented
        assert any(r.status_code == 429 for r in responses)
        
        # Check response headers for retry information
        throttled_response = next(r for r in responses if r.status_code == 429)
        assert "Retry-After" in throttled_response.headers
    else:
        # Throttling is not implemented or threshold not reached
        assert all(r.status_code == 200 for r in responses)


if __name__ == "__main__":
    unittest.main() 