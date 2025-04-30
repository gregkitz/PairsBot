#!/usr/bin/env python3
"""
Integration tests for the containerized distributed processing system.

These tests verify that Docker containers can start correctly, communicate with each
other, and maintain persistence as needed for the trading system.
"""

import os
import sys
import unittest
import subprocess
import time
import requests
import json
import docker
import pytest
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import create_directory


class ContainerIntegrationTests(unittest.TestCase):
    """Test the integration of Docker containers for the trading system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment by ensuring Docker is running."""
        cls.client = docker.from_env()
        
        # Check if Docker is available
        try:
            cls.client.ping()
        except Exception as e:
            pytest.skip(f"Docker is not available: {e}")
            
        # Create directories needed for testing
        test_data_dir = Path("./data/test_container")
        create_directory(test_data_dir)
        
        # Create test data for persistence testing
        with open(test_data_dir / "test_file.json", "w") as f:
            json.dump({"test_key": "test_value"}, f)
            
        # Start the containers using docker-compose
        try:
            cls.compose_process = subprocess.run(
                ["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d"],
                check=True,
                capture_output=True
            )
            
            # Give containers time to start
            time.sleep(10)
            
            # Get container IDs
            cls.containers = {
                container.name: container
                for container in cls.client.containers.list()
                if container.status == "running"
            }
            
            if not cls.containers:
                pytest.skip("No containers found running. Check docker-compose configuration.")
        
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to start containers via docker-compose: {e.stderr.decode()}")
            
    @classmethod
    def tearDownClass(cls):
        """Clean up by stopping all containers."""
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker/docker-compose.yml", "down"],
                check=True,
                capture_output=True
            )
        except Exception as e:
            print(f"Warning: Failed to stop containers: {e}")
    
    def test_container_running_status(self):
        """Test that all containers are running."""
        expected_containers = ["quant-trader-app", "quant-trader-worker", "redis", "postgres"]
        
        for container_name in expected_containers:
            # Find container with a name containing the expected name
            matching_containers = [c for c_name, c in self.containers.items() if container_name in c_name]
            
            self.assertTrue(
                len(matching_containers) > 0,
                f"Container '{container_name}' is not running"
            )
            
            # Verify container is healthy
            for container in matching_containers:
                container.reload()  # Refresh container data
                self.assertEqual(
                    container.status,
                    "running",
                    f"Container {container.name} is not running (status: {container.status})"
                )
    
    def test_api_container_responds(self):
        """Test that the API container responds to requests."""
        try:
            # Try to connect to the API service
            response = requests.get("http://localhost:5000/api/health", timeout=5)
            
            self.assertEqual(
                response.status_code,
                200,
                f"API health endpoint returned unexpected status: {response.status_code}"
            )
            
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            
        except requests.RequestException as e:
            self.fail(f"Failed to connect to API container: {e}")
    
    def test_worker_container_connects_to_redis(self):
        """Test that the worker container can connect to Redis."""
        # Find worker container
        worker_containers = [c for c_name, c in self.containers.items() if "worker" in c_name]
        
        if not worker_containers:
            self.fail("Worker container not found")
            
        worker = worker_containers[0]
        
        # Execute command in worker container to check Redis connection
        exit_code, output = worker.exec_run(
            cmd="python -c \"import redis; r = redis.Redis(host='redis', port=6379); print(r.ping())\"",
            stderr=False
        )
        
        self.assertEqual(exit_code, 0, f"Worker failed to connect to Redis: {output.decode()}")
        self.assertIn("True", output.decode(), "Redis ping did not return True")
    
    def test_container_volume_persistence(self):
        """Test that container volumes persist data."""
        # Find application container
        app_containers = [c for c_name, c in self.containers.items() if "app" in c_name]
        
        if not app_containers:
            self.fail("App container not found")
            
        app = app_containers[0]
        
        # Create a test file in the mounted volume
        test_file_path = "/app/data/test_persistence.txt"
        test_content = f"Test data {time.time()}"
        
        exit_code, output = app.exec_run(
            cmd=f"bash -c \"echo '{test_content}' > {test_file_path}\"",
            stderr=False
        )
        
        self.assertEqual(exit_code, 0, f"Failed to create test file: {output.decode()}")
        
        # Restart the container
        app.restart()
        time.sleep(5)  # Give container time to restart
        
        # Check if the file still exists with the same content
        exit_code, output = app.exec_run(
            cmd=f"cat {test_file_path}",
            stderr=False
        )
        
        self.assertEqual(exit_code, 0, f"Failed to read test file after restart: {output.decode()}")
        self.assertEqual(
            output.decode().strip(),
            test_content,
            "Content in persisted file doesn't match the original"
        )
    
    def test_communication_between_containers(self):
        """Test that containers can communicate with each other."""
        # Find application container
        app_containers = [c for c_name, c in self.containers.items() if "app" in c_name]
        
        if not app_containers:
            self.fail("App container not found")
            
        app = app_containers[0]
        
        # Execute a command in the app container to submit a task to Celery
        test_task_code = """
import sys
import time
from celery import Celery

# Setup Celery
celery_app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

# Define a simple test task
@celery_app.task
def test_task(x, y):
    return x + y

# Submit the task and wait for the result
result = test_task.delay(4, 4)
try:
    value = result.get(timeout=10)
    print(f"Task result: {value}")
    sys.exit(0 if value == 8 else 1)
except Exception as e:
    print(f"Task failed: {e}")
    sys.exit(1)
"""
        
        # Write the test code to a file in the container
        exit_code, output = app.exec_run(
            cmd=f"bash -c \"echo '{test_task_code}' > /tmp/test_task.py\"",
            stderr=False
        )
        
        self.assertEqual(exit_code, 0, f"Failed to create test script: {output.decode()}")
        
        # Execute the test script
        exit_code, output = app.exec_run(
            cmd="python /tmp/test_task.py",
            stderr=True
        )
        
        self.assertEqual(exit_code, 0, f"Container communication test failed: {output.decode()}")
        self.assertIn("Task result: 8", output.decode(), "Task didn't return the expected result")


@pytest.fixture(scope="session")
def docker_environment():
    """Pytest fixture for Docker environment setup and teardown."""
    # Setup
    client = docker.from_env()
    
    try:
        client.ping()
    except Exception as e:
        pytest.skip(f"Docker is not available: {e}")
    
    try:
        subprocess.run(
            ["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d"],
            check=True,
            capture_output=True
        )
        
        # Wait for containers to start
        time.sleep(10)
        
        yield client
        
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to start containers: {e.stderr.decode()}")
        yield None
    
    finally:
        # Teardown
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker/docker-compose.yml", "down"],
                check=True,
                capture_output=True
            )
        except Exception as e:
            print(f"Warning: Failed to stop containers: {e}")


@pytest.mark.parametrize("container_name", ["app", "worker", "redis", "postgres"])
def test_container_logs_for_errors(docker_environment, container_name):
    """Test that container logs do not contain errors."""
    if not docker_environment:
        pytest.skip("Docker environment not available")
    
    # Find container with name containing the specified pattern
    containers = docker_environment.containers.list()
    matching_containers = [
        container for container in containers 
        if container_name in container.name and container.status == "running"
    ]
    
    if not matching_containers:
        pytest.fail(f"No running containers found with name containing '{container_name}'")
    
    container = matching_containers[0]
    
    # Get container logs
    logs = container.logs(tail=100).decode("utf-8")
    
    # Check for common error indicators
    error_indicators = [
        "ERROR", "Error:", "Exception", "CRITICAL", "FATAL",
        "Traceback (most recent call last)", "failed", "Failed"
    ]
    
    # Filter out known non-fatal warnings or debug messages
    benign_messages = [
        "DEBUG", "[INFO]", "WARNING: Connection pool is full"
    ]
    
    for line in logs.splitlines():
        # Skip benign messages
        if any(benign in line for benign in benign_messages):
            continue
            
        # Check for error indicators
        assert not any(indicator in line for indicator in error_indicators), \
            f"Found error in {container_name} logs: {line}"


if __name__ == "__main__":
    unittest.main() 