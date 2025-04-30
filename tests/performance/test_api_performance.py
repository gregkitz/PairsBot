#!/usr/bin/env python3
"""
Performance tests for API endpoints.

This module measures the performance of API endpoints, including response time,
throughput, concurrency handling, and scaling behavior.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import tracemalloc
from functools import wraps
from contextlib import contextmanager
import asyncio
import aiohttp
import requests
import statistics
import concurrent.futures
from urllib.parse import urljoin
import psutil
import resource

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

# Try importing API components
try:
    from src.api.main import app as fastapi_app
    from fastapi.testclient import TestClient
    API_COMPONENTS_AVAILABLE = True
except ImportError:
    API_COMPONENTS_AVAILABLE = False


def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Get stats
        stats = snapshot.statistics('lineno')
        total = sum(stat.size for stat in stats)
        
        # Convert to MB
        total_mb = total / (1024 * 1024)
        
        return result, total_mb
    
    return wrapper


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


@contextmanager
def measure_system_resources():
    """Context manager for measuring system resource utilization."""
    # Start resource measurement
    process = psutil.Process(os.getpid())
    start_cpu_percent = process.cpu_percent()
    start_mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Store system-wide stats
    start_system_cpu = psutil.cpu_percent()
    start_system_mem = psutil.virtual_memory().percent
    
    yield
    
    # End resource measurement
    end_cpu_percent = process.cpu_percent()
    end_mem_usage = process.memory_info().rss / (1024 * 1024)  # MB
    
    # End system-wide stats
    end_system_cpu = psutil.cpu_percent()
    end_system_mem = psutil.virtual_memory().percent
    
    # Calculate differences
    cpu_usage = end_cpu_percent - start_cpu_percent
    mem_usage_diff = end_mem_usage - start_mem_usage
    system_cpu_diff = end_system_cpu - start_system_cpu
    system_mem_diff = end_system_mem - start_system_mem
    
    return {
        "process_cpu_percent": cpu_usage,
        "process_memory_mb": end_mem_usage,
        "process_memory_diff_mb": mem_usage_diff,
        "system_cpu_percent": end_system_cpu,
        "system_cpu_diff": system_cpu_diff,
        "system_memory_percent": end_system_mem,
        "system_memory_diff": system_mem_diff
    }


def classify_error(status_code=None, exception=None):
    """
    Classify error type based on status code or exception.
    
    Parameters:
    -----------
    status_code : int, optional
        HTTP status code
    exception : Exception, optional
        Exception that occurred
        
    Returns:
    --------
    str
        Classification of the error
    """
    if status_code:
        if 400 <= status_code < 500:
            if status_code == 401:
                return "auth_error"
            elif status_code == 403:
                return "permission_error"
            elif status_code == 404:
                return "not_found"
            elif status_code == 429:
                return "rate_limited"
            else:
                return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown_status"
    
    if exception:
        if isinstance(exception, requests.exceptions.Timeout):
            return "timeout"
        elif isinstance(exception, requests.exceptions.ConnectionError):
            return "connection_error"
        elif isinstance(exception, requests.exceptions.RequestException):
            return "request_error"
        elif isinstance(exception, aiohttp.ClientError):
            return "async_client_error"
        elif isinstance(exception, asyncio.TimeoutError):
            return "async_timeout"
        else:
            return "unknown_error"
    
    return "unclassified_error"


def get_test_config():
    """Get test configuration from environment variables."""
    # Define defaults
    config = {
        "api_url": os.environ.get("API_URL", "http://localhost:8000"),
        "test_intensity": os.environ.get("API_TEST_INTENSITY", "medium"),
        "concurrency_levels": [1, 5, 10, 25, 50],
        "request_timeout": 30.0,
        "repetitions": 3,
        "enable_auth": False,
        "enable_resource_profiling": os.environ.get("ENABLE_RESOURCE_PROFILING", "true").lower() == "true",
        "performance_thresholds": {
            "max_latency_ms": float(os.environ.get("MAX_LATENCY_MS", "500")),
            "min_requests_per_second": float(os.environ.get("MIN_REQUESTS_PER_SECOND", "10")),
            "max_error_rate": float(os.environ.get("MAX_ERROR_RATE", "0.05"))
        }
    }
    
    # Define test intensities
    intensities = {
        "light": {
            "requests_per_endpoint": 10,
            "duration_seconds": 5,
            "concurrency_levels": [1, 5, 10]
        },
        "medium": {
            "requests_per_endpoint": 50,
            "duration_seconds": 10,
            "concurrency_levels": [1, 5, 10, 25]
        },
        "heavy": {
            "requests_per_endpoint": 200,
            "duration_seconds": 30,
            "concurrency_levels": [1, 5, 10, 25, 50, 100]
        }
    }
    
    # Update config with intensity-specific settings
    intensity = intensities.get(config["test_intensity"], intensities["medium"])
    config.update(intensity)
    
    return config


class APIPerformanceTester:
    """Class for testing API performance."""
    
    def __init__(self, base_url, enable_auth=False, auth_token=None, enable_resource_profiling=False):
        """
        Initialize API performance tester.
        
        Parameters:
        -----------
        base_url : str
            Base URL of the API
        enable_auth : bool
            Whether to enable authentication
        auth_token : str, optional
            Authentication token to use
        enable_resource_profiling : bool
            Whether to enable resource profiling
        """
        self.base_url = base_url
        self.enable_auth = enable_auth
        self.auth_token = auth_token
        self.enable_resource_profiling = enable_resource_profiling
        
        # Initialize test client if API components are available
        self.client = None
        if API_COMPONENTS_AVAILABLE:
            self.client = TestClient(fastapi_app)
        
        # Initialize error tracking
        self.error_counts = {}
    
    def get_headers(self):
        """
        Get request headers.
        
        Returns:
        --------
        dict
            Headers for API requests
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.enable_auth and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        return headers
    
    def get_endpoints(self):
        """
        Get list of endpoints to test.
        
        Returns:
        --------
        list
            List of endpoint dictionaries with path, method, and payload
        """
        # Define endpoints to test
        endpoints = [
            {
                "name": "health",
                "path": "/health",
                "method": "GET",
                "payload": None
            },
            {
                "name": "system_status",
                "path": "/system/status",
                "method": "GET",
                "payload": None
            },
            {
                "name": "list_tasks",
                "path": "/tasks",
                "method": "GET",
                "payload": None
            }
        ]
        
        # Add task-specific endpoints if available
        try:
            # These are more complex endpoints that might not be available
            # or might require special authentication
            endpoints.extend([
                {
                    "name": "backtest",
                    "path": "/tasks/backtest",
                    "method": "POST",
                    "payload": {
                        "pair": "CL-HO",
                        "start_date": "2022-01-01",
                        "end_date": "2022-03-31",
                        "parameters": {
                            "entry_threshold": 2.0,
                            "exit_threshold": 0.5
                        }
                    }
                },
                {
                    "name": "optimization",
                    "path": "/tasks/optimize",
                    "method": "POST",
                    "payload": {
                        "pair": "CL-HO",
                        "start_date": "2022-01-01",
                        "end_date": "2022-03-31",
                        "parameters": {
                            "entry_threshold_range": [1.5, 3.0],
                            "exit_threshold_range": [0.1, 1.0]
                        }
                    }
                },
                {
                    "name": "pair_analysis",
                    "path": "/analysis/pair",
                    "method": "POST",
                    "payload": {
                        "pair": "CL-HO",
                        "start_date": "2022-01-01",
                        "end_date": "2022-03-31"
                    }
                }
            ])
        except Exception:
            # Just use the basic endpoints if there's any issue
            pass
        
        return endpoints
    
    async def test_endpoint_async(self, session, endpoint, num_requests):
        """
        Test a single endpoint asynchronously multiple times.
        
        Parameters:
        -----------
        session : aiohttp.ClientSession
            Async HTTP session
        endpoint : dict
            Endpoint configuration
        num_requests : int
            Number of requests to make
            
        Returns:
        --------
        dict
            Test results with response times and error counts
        """
        url = urljoin(self.base_url, endpoint["path"])
        method = endpoint["method"]
        payload = endpoint["payload"]
        
        response_times = []
        errors = 0
        error_types = {}
        
        for _ in range(num_requests):
            start_time = time.time()
            
            try:
                if method == "GET":
                    async with session.get(url, headers=self.get_headers()) as response:
                        await response.text()
                        status = response.status
                elif method == "POST":
                    async with session.post(url, json=payload, headers=self.get_headers()) as response:
                        await response.text()
                        status = response.status
                else:
                    continue
                
                # Record time only for successful responses
                if 200 <= status < 300:
                    response_times.append(time.time() - start_time)
                else:
                    errors += 1
                    error_type = classify_error(status_code=status)
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            except Exception as e:
                errors += 1
                error_type = classify_error(exception=e)
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "response_times": response_times,
            "errors": errors,
            "error_types": error_types
        }
    
    async def run_concurrent_tests_async(self, endpoint, concurrency, num_requests_per_worker):
        """
        Run concurrent tests against an endpoint.
        
        Parameters:
        -----------
        endpoint : dict
            Endpoint configuration
        concurrency : int
            Number of concurrent requests
        num_requests_per_worker : int
            Number of requests per worker
            
        Returns:
        --------
        dict
            Test results
        """
        # Track resources if enabled
        resource_usage = None
        if self.enable_resource_profiling:
            with measure_system_resources() as resources:
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for _ in range(concurrency):
                        task = self.test_endpoint_async(session, endpoint, num_requests_per_worker)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                resource_usage = resources
        else:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for _ in range(concurrency):
                    task = self.test_endpoint_async(session, endpoint, num_requests_per_worker)
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
        
        # Combine results
        all_response_times = []
        total_errors = 0
        combined_error_types = {}
        
        for result in results:
            all_response_times.extend(result["response_times"])
            total_errors += result["errors"]
            
            # Combine error types
            for error_type, count in result.get("error_types", {}).items():
                combined_error_types[error_type] = combined_error_types.get(error_type, 0) + count
        
        # Calculate statistics
        stats = {}
        if all_response_times:
            stats = {
                "min": min(all_response_times),
                "max": max(all_response_times),
                "mean": statistics.mean(all_response_times),
                "median": statistics.median(all_response_times),
                "p95": np.percentile(all_response_times, 95),
                "p99": np.percentile(all_response_times, 99),
                "total_requests": concurrency * num_requests_per_worker,
                "successful_requests": len(all_response_times),
                "error_count": total_errors,
                "error_rate": total_errors / (concurrency * num_requests_per_worker),
                "error_types": combined_error_types,
                "requests_per_second": len(all_response_times) / (sum(all_response_times) / concurrency)
            }
        
        # Add resource usage if available
        if resource_usage:
            stats["resource_usage"] = resource_usage
        
        return stats
    
    def test_endpoint_sync(self, endpoint, num_requests):
        """
        Test a single endpoint synchronously multiple times.
        
        Parameters:
        -----------
        endpoint : dict
            Endpoint configuration
        num_requests : int
            Number of requests to make
            
        Returns:
        --------
        dict
            Test results
        """
        url = urljoin(self.base_url, endpoint["path"])
        method = endpoint["method"]
        payload = endpoint["payload"]
        
        response_times = []
        errors = 0
        error_types = {}
        
        for _ in range(num_requests):
            start_time = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(url, headers=self.get_headers(), timeout=30)
                elif method == "POST":
                    response = requests.post(url, json=payload, headers=self.get_headers(), timeout=30)
                else:
                    continue
                
                # Record time only for successful responses
                if 200 <= response.status_code < 300:
                    response_times.append(time.time() - start_time)
                else:
                    errors += 1
                    error_type = classify_error(status_code=response.status_code)
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            except Exception as e:
                errors += 1
                error_type = classify_error(exception=e)
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "response_times": response_times,
            "errors": errors,
            "error_types": error_types
        }
    
    def run_concurrent_tests_sync(self, endpoint, concurrency, num_requests_per_worker):
        """
        Run concurrent tests against an endpoint synchronously.
        
        Parameters:
        -----------
        endpoint : dict
            Endpoint configuration
        concurrency : int
            Number of concurrent requests
        num_requests_per_worker : int
            Number of requests per worker
            
        Returns:
        --------
        dict
            Test results
        """
        all_response_times = []
        total_errors = 0
        combined_error_types = {}
        
        # Track resources if enabled
        resource_usage = None
        if self.enable_resource_profiling:
            with measure_system_resources() as resources:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for _ in range(concurrency):
                        future = executor.submit(self.test_endpoint_sync, endpoint, num_requests_per_worker)
                        futures.append(future)
                    
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        all_response_times.extend(result["response_times"])
                        total_errors += result["errors"]
                        
                        # Combine error types
                        for error_type, count in result.get("error_types", {}).items():
                            combined_error_types[error_type] = combined_error_types.get(error_type, 0) + count
                resource_usage = resources
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for _ in range(concurrency):
                    future = executor.submit(self.test_endpoint_sync, endpoint, num_requests_per_worker)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    all_response_times.extend(result["response_times"])
                    total_errors += result["errors"]
                    
                    # Combine error types
                    for error_type, count in result.get("error_types", {}).items():
                        combined_error_types[error_type] = combined_error_types.get(error_type, 0) + count
        
        # Calculate statistics
        stats = {}
        if all_response_times:
            stats = {
                "min": min(all_response_times),
                "max": max(all_response_times),
                "mean": statistics.mean(all_response_times),
                "median": statistics.median(all_response_times),
                "p95": np.percentile(all_response_times, 95),
                "p99": np.percentile(all_response_times, 99),
                "total_requests": concurrency * num_requests_per_worker,
                "successful_requests": len(all_response_times),
                "error_count": total_errors,
                "error_rate": total_errors / (concurrency * num_requests_per_worker),
                "error_types": combined_error_types,
                "requests_per_second": len(all_response_times) / (sum(all_response_times) / concurrency)
            }
        
        # Add resource usage if available
        if resource_usage:
            stats["resource_usage"] = resource_usage
        
        return stats
    
    def test_endpoint_with_client(self, endpoint, num_requests):
        """
        Test a single endpoint with the FastAPI test client.
        
        Parameters:
        -----------
        endpoint : dict
            Endpoint configuration
        num_requests : int
            Number of requests to make
            
        Returns:
        --------
        dict
            Test results
        """
        if not self.client:
            return {"response_times": [], "errors": num_requests, "error_types": {"client_unavailable": num_requests}}
        
        method = endpoint["method"]
        path = endpoint["path"]
        payload = endpoint["payload"]
        
        response_times = []
        errors = 0
        error_types = {}
        
        # Track memory usage if profiling is enabled
        memory_usage = None
        if self.enable_resource_profiling:
            with measure_system_resources() as resources:
                for _ in range(num_requests):
                    start_time = time.time()
                    
                    try:
                        if method == "GET":
                            response = self.client.get(path, headers=self.get_headers())
                        elif method == "POST":
                            response = self.client.post(path, json=payload, headers=self.get_headers())
                        else:
                            continue
                        
                        # Record time only for successful responses
                        if 200 <= response.status_code < 300:
                            response_times.append(time.time() - start_time)
                        else:
                            errors += 1
                            error_type = classify_error(status_code=response.status_code)
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    except Exception as e:
                        errors += 1
                        error_type = classify_error(exception=e)
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                memory_usage = resources
        else:
            for _ in range(num_requests):
                start_time = time.time()
                
                try:
                    if method == "GET":
                        response = self.client.get(path, headers=self.get_headers())
                    elif method == "POST":
                        response = self.client.post(path, json=payload, headers=self.get_headers())
                    else:
                        continue
                    
                    # Record time only for successful responses
                    if 200 <= response.status_code < 300:
                        response_times.append(time.time() - start_time)
                    else:
                        errors += 1
                        error_type = classify_error(status_code=response.status_code)
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                
                except Exception as e:
                    errors += 1
                    error_type = classify_error(exception=e)
                    error_types[error_type] = error_types.get(error_type, 0) + 1
        
        result = {
            "response_times": response_times,
            "errors": errors,
            "error_types": error_types
        }
        
        # Add resource usage if available
        if memory_usage:
            result["resource_usage"] = memory_usage
        
        return result
    
    def test_single_endpoint_performance(self, endpoint, config):
        """
        Test the performance of a single endpoint.
        
        Parameters:
        -----------
        endpoint : dict
            Endpoint configuration
        config : dict
            Test configuration
            
        Returns:
        --------
        dict
            Test results
        """
        print(f"Testing endpoint: {endpoint['name']} - {endpoint['method']} {endpoint['path']}")
        
        # Test latency for a single request
        single_request_results = self.test_endpoint_sync(endpoint, 1)
        latency = 0
        if single_request_results["response_times"]:
            latency = single_request_results["response_times"][0]
        
        print(f"  Single request latency: {latency:.4f} seconds")
        
        # Convert to milliseconds for easier reading
        latency_ms = latency * 1000
        
        # Test throughput at different concurrency levels
        concurrency_results = {}
        
        for concurrency in config["concurrency_levels"]:
            print(f"  Testing with concurrency level: {concurrency}")
            
            # Use synchronous testing for lower concurrency levels
            # and async for higher levels to avoid resource exhaustion
            if concurrency <= 10:
                stats = self.run_concurrent_tests_sync(
                    endpoint, 
                    concurrency, 
                    config["requests_per_endpoint"] // concurrency
                )
            else:
                stats = asyncio.run(self.run_concurrent_tests_async(
                    endpoint, 
                    concurrency, 
                    config["requests_per_endpoint"] // concurrency
                ))
            
            concurrency_results[concurrency] = stats
            
            # Calculate requests per second
            requests_per_second = stats.get("requests_per_second", 0)
            if "mean" in stats and stats["successful_requests"] > 0:
                if "requests_per_second" not in stats:
                    requests_per_second = stats["successful_requests"] / (stats["mean"] * concurrency)
            
            print(f"    Mean response time: {stats.get('mean', 0):.4f} seconds")
            print(f"    Requests per second: {requests_per_second:.2f}")
            
            # If resource profiling is enabled, report it
            if "resource_usage" in stats:
                ru = stats["resource_usage"]
                print(f"    CPU usage: {ru.get('process_cpu_percent', 0):.2f}%")
                print(f"    Memory usage: {ru.get('process_memory_mb', 0):.2f} MB")
            
            # Print error distribution if errors occurred
            if stats.get("error_count", 0) > 0:
                print(f"    Error rate: {stats.get('error_rate', 0):.2%}")
                for error_type, count in stats.get("error_types", {}).items():
                    print(f"      {error_type}: {count}")
            
            # Check against performance thresholds
            if "performance_thresholds" in config:
                thresholds = config["performance_thresholds"]
                
                # Check latency threshold (mean response time in ms)
                mean_latency_ms = stats.get("mean", 0) * 1000
                if mean_latency_ms > thresholds.get("max_latency_ms", float("inf")):
                    print(f"    WARNING: Mean latency ({mean_latency_ms:.2f} ms) exceeds threshold "
                          f"({thresholds.get('max_latency_ms')} ms)")
                
                # Check throughput threshold
                if requests_per_second < thresholds.get("min_requests_per_second", 0):
                    print(f"    WARNING: Throughput ({requests_per_second:.2f} req/s) below threshold "
                          f"({thresholds.get('min_requests_per_second')} req/s)")
                
                # Check error rate threshold
                error_rate = stats.get("error_rate", 0)
                if error_rate > thresholds.get("max_error_rate", 1.0):
                    print(f"    WARNING: Error rate ({error_rate:.2%}) exceeds threshold "
                          f"({thresholds.get('max_error_rate'):.2%})")
        
        return {
            "endpoint": endpoint["name"],
            "path": endpoint["path"],
            "method": endpoint["method"],
            "latency": latency,
            "latency_ms": latency_ms,
            "concurrency_results": concurrency_results,
            "meets_thresholds": self.check_against_thresholds(endpoint["name"], latency_ms, concurrency_results, config)
        }
    
    def check_against_thresholds(self, endpoint_name, latency_ms, concurrency_results, config):
        """
        Check performance results against thresholds.
        
        Parameters:
        -----------
        endpoint_name : str
            Name of the endpoint
        latency_ms : float
            Single-request latency in milliseconds
        concurrency_results : dict
            Results from concurrency tests
        config : dict
            Test configuration
            
        Returns:
        --------
        dict
            Threshold satisfaction results
        """
        if "performance_thresholds" not in config:
            return {"status": "no_thresholds"}
        
        thresholds = config["performance_thresholds"]
        results = {
            "latency": True,
            "throughput": True,
            "error_rate": True,
            "overall": True
        }
        
        # Check latency threshold
        if latency_ms > thresholds.get("max_latency_ms", float("inf")):
            results["latency"] = False
            results["overall"] = False
        
        # Check concurrency results
        for concurrency, stats in concurrency_results.items():
            # Skip if not enough data
            if not stats or stats.get("successful_requests", 0) == 0:
                continue
            
            # Calculate requests per second
            requests_per_second = stats.get("requests_per_second", 0)
            if "mean" in stats and not requests_per_second:
                requests_per_second = stats["successful_requests"] / (stats["mean"] * concurrency)
            
            # Check throughput threshold
            if requests_per_second < thresholds.get("min_requests_per_second", 0):
                results["throughput"] = False
                results["overall"] = False
            
            # Check error rate threshold
            error_rate = stats.get("error_rate", 0)
            if error_rate > thresholds.get("max_error_rate", 1.0):
                results["error_rate"] = False
                results["overall"] = False
        
        return results
    
    def test_fastapi_client_performance(self, endpoints, config):
        """
        Test the performance using the FastAPI test client.
        
        Parameters:
        -----------
        endpoints : list
            List of endpoint configurations
        config : dict
            Test configuration
            
        Returns:
        --------
        dict
            Test results
        """
        if not API_COMPONENTS_AVAILABLE:
            return {"status": "skipped", "reason": "API components not available"}
        
        print("Testing with FastAPI test client")
        
        results = {}
        
        for endpoint in endpoints:
            print(f"Testing endpoint: {endpoint['name']} - {endpoint['method']} {endpoint['path']}")
            
            # Test with a single request
            single_result = self.test_endpoint_with_client(endpoint, 1)
            latency = 0
            if single_result["response_times"]:
                latency = single_result["response_times"][0]
            
            # Test with multiple requests
            multi_result = self.test_endpoint_with_client(endpoint, config["requests_per_endpoint"])
            
            # Calculate statistics
            stats = {}
            if multi_result["response_times"]:
                stats = {
                    "min": min(multi_result["response_times"]),
                    "max": max(multi_result["response_times"]),
                    "mean": statistics.mean(multi_result["response_times"]),
                    "median": statistics.median(multi_result["response_times"]),
                    "p95": np.percentile(multi_result["response_times"], 95),
                    "p99": np.percentile(multi_result["response_times"], 99),
                    "total_requests": config["requests_per_endpoint"],
                    "successful_requests": len(multi_result["response_times"]),
                    "error_count": multi_result["errors"],
                    "error_rate": multi_result["errors"] / config["requests_per_endpoint"],
                    "error_types": multi_result.get("error_types", {})
                }
                
                # Calculate requests per second
                if stats["mean"] > 0:
                    stats["requests_per_second"] = stats["successful_requests"] / stats["mean"]
            
            # Add resource usage if available
            if "resource_usage" in multi_result:
                stats["resource_usage"] = multi_result["resource_usage"]
            
            # Check against performance thresholds
            meets_thresholds = None
            if "performance_thresholds" in config and stats:
                thresholds = config["performance_thresholds"]
                meets_thresholds = {
                    "latency": (stats.get("mean", 0) * 1000) <= thresholds.get("max_latency_ms", float("inf")),
                    "throughput": stats.get("requests_per_second", 0) >= thresholds.get("min_requests_per_second", 0),
                    "error_rate": stats.get("error_rate", 0) <= thresholds.get("max_error_rate", 1.0)
                }
                meets_thresholds["overall"] = all(meets_thresholds.values())
            
            results[endpoint["name"]] = {
                "latency": latency,
                "latency_ms": latency * 1000,
                "stats": stats,
                "meets_thresholds": meets_thresholds
            }
            
            # Print results and warnings
            print(f"  Latency: {latency * 1000:.2f} ms")
            print(f"  Mean response time: {stats.get('mean', 0) * 1000:.2f} ms")
            print(f"  Requests per second: {stats.get('requests_per_second', 0):.2f}")
            
            # Print resource usage if available
            if "resource_usage" in stats:
                ru = stats["resource_usage"]
                print(f"  CPU usage: {ru.get('process_cpu_percent', 0):.2f}%")
                print(f"  Memory usage: {ru.get('process_memory_mb', 0):.2f} MB")
            
            # Print error distribution if errors occurred
            if stats.get("error_count", 0) > 0:
                print(f"  Error rate: {stats.get('error_rate', 0):.2%}")
                for error_type, count in stats.get("error_types", {}).items():
                    print(f"    {error_type}: {count}")
            
            # Print threshold warnings
            if meets_thresholds and not meets_thresholds["overall"]:
                print("  WARNING: Endpoint does not meet performance thresholds:")
                for metric, result in meets_thresholds.items():
                    if metric != "overall" and not result:
                        threshold_value = config["performance_thresholds"].get(
                            f"{'max' if metric == 'latency' or metric == 'error_rate' else 'min'}_{metric}",
                            "N/A"
                        )
                        print(f"    {metric}: Failed to meet threshold ({threshold_value})")
        
        return results
    
    def run_endpoint_tests(self, config):
        """
        Run performance tests for all endpoints.
        
        Parameters:
        -----------
        config : dict
            Test configuration
            
        Returns:
        --------
        dict
            Test results
        """
        endpoints = self.get_endpoints()
        results = {}
        
        for endpoint in endpoints:
            result = self.test_single_endpoint_performance(endpoint, config)
            results[endpoint["name"]] = result
        
        # Summarize overall performance
        summary = self.summarize_results(results)
        print("\nPerformance Test Summary:")
        print(f"  Total endpoints tested: {summary['total_endpoints']}")
        print(f"  Endpoints meeting all thresholds: {summary['endpoints_meeting_thresholds']}")
        print(f"  Endpoints with latency issues: {summary['endpoints_with_latency_issues']}")
        print(f"  Endpoints with throughput issues: {summary['endpoints_with_throughput_issues']}")
        print(f"  Endpoints with error rate issues: {summary['endpoints_with_error_issues']}")
        
        # Add summary to results
        results["summary"] = summary
        
        return results
    
    def summarize_results(self, results):
        """
        Create a summary of test results.
        
        Parameters:
        -----------
        results : dict
            Test results
            
        Returns:
        --------
        dict
            Summary statistics
        """
        summary = {
            "total_endpoints": len(results),
            "endpoints_meeting_thresholds": 0,
            "endpoints_with_latency_issues": 0,
            "endpoints_with_throughput_issues": 0,
            "endpoints_with_error_issues": 0,
            "endpoint_statuses": {}
        }
        
        for endpoint_name, result in results.items():
            endpoint_status = {
                "latency_ms": result.get("latency_ms", 0),
                "meets_all_thresholds": False
            }
            
            meets_thresholds = result.get("meets_thresholds", {})
            if meets_thresholds:
                endpoint_status["meets_all_thresholds"] = meets_thresholds.get("overall", False)
                
                if endpoint_status["meets_all_thresholds"]:
                    summary["endpoints_meeting_thresholds"] += 1
                
                if not meets_thresholds.get("latency", True):
                    summary["endpoints_with_latency_issues"] += 1
                    endpoint_status["latency_issue"] = True
                
                if not meets_thresholds.get("throughput", True):
                    summary["endpoints_with_throughput_issues"] += 1
                    endpoint_status["throughput_issue"] = True
                
                if not meets_thresholds.get("error_rate", True):
                    summary["endpoints_with_error_issues"] += 1
                    endpoint_status["error_rate_issue"] = True
            
            summary["endpoint_statuses"][endpoint_name] = endpoint_status
        
        return summary
    
    def create_results_visualization(self, results, output_dir):
        """
        Create visualizations of test results.
        
        Parameters:
        -----------
        results : dict
            Test results
        output_dir : Path
            Directory to save visualizations
        """
        # Create latency comparison chart
        plt.figure(figsize=(12, 6))
        endpoints = []
        latencies = []
        
        for endpoint_name, result in results.items():
            if endpoint_name == "summary":
                continue
                
            if "latency" in result:
                endpoints.append(endpoint_name)
                latencies.append(result["latency"] * 1000)  # Convert to ms
        
        # Create bar chart with color coding based on thresholds
        bars = plt.bar(endpoints, latencies)
        
        # Color the bars based on threshold satisfaction
        for i, (endpoint_name, result) in enumerate(zip(endpoints, [results[ep] for ep in endpoints])):
            if "meets_thresholds" in result and "latency" in result["meets_thresholds"]:
                if result["meets_thresholds"]["latency"]:
                    bars[i].set_color('green')
                else:
                    bars[i].set_color('red')
        
        plt.title("API Endpoint Latency Comparison (ms)")
        plt.xlabel("Endpoint")
        plt.ylabel("Latency (milliseconds)")
        plt.grid(True, axis='y')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "endpoint_latency_comparison.png")
        
        # Create concurrency impact charts for each endpoint
        for endpoint_name, result in results.items():
            if endpoint_name == "summary":
                continue
                
            if "concurrency_results" in result:
                concurrency_results = result["concurrency_results"]
                
                if not concurrency_results:
                    continue
                
                plt.figure(figsize=(12, 6))
                concurrency_levels = list(concurrency_results.keys())
                mean_times = [concurrency_results[c].get("mean", 0) * 1000 for c in concurrency_levels]  # Convert to ms
                
                plt.plot(concurrency_levels, mean_times, 'o-')
                plt.title(f"Impact of Concurrency on Response Time - {endpoint_name}")
                plt.xlabel("Concurrency Level")
                plt.ylabel("Mean Response Time (milliseconds)")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f"concurrency_impact_{endpoint_name}.png")
                
                # Create throughput chart
                plt.figure(figsize=(12, 6))
                throughputs = [concurrency_results[c].get("requests_per_second", 0) for c in concurrency_levels]
                
                plt.plot(concurrency_levels, throughputs, 'o-')
                plt.title(f"Throughput vs Concurrency - {endpoint_name}")
                plt.xlabel("Concurrency Level")
                plt.ylabel("Requests per Second")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(output_dir / f"throughput_{endpoint_name}.png")
        
        # Create error rate comparison (if there are errors)
        has_errors = False
        for endpoint_name, result in results.items():
            if endpoint_name == "summary":
                continue
                
            for concurrency, stats in result.get("concurrency_results", {}).items():
                if stats.get("error_count", 0) > 0:
                    has_errors = True
                    break
            
            if has_errors:
                break
        
        if has_errors:
            plt.figure(figsize=(12, 6))
            endpoints = []
            error_rates = []
            
            for endpoint_name, result in results.items():
                if endpoint_name == "summary":
                    continue
                    
                if "concurrency_results" in result:
                    # Use the highest concurrency level for comparison
                    max_concurrency = max(result["concurrency_results"].keys())
                    stats = result["concurrency_results"][max_concurrency]
                    
                    endpoints.append(endpoint_name)
                    error_rates.append(stats.get("error_rate", 0) * 100)  # Convert to percentage
            
            bars = plt.bar(endpoints, error_rates)
            
            # Color the bars based on threshold satisfaction
            threshold_value = None
            for config_key, result in results.items():
                if "meets_thresholds" in result and result["meets_thresholds"]:
                    # Use the first endpoint that has threshold data
                    threshold_value = 100 * result["meets_thresholds"].get("max_error_rate", 0.05)
                    break
            
            if threshold_value is not None:
                plt.axhline(y=threshold_value, color='r', linestyle='-', label=f'Threshold ({threshold_value:.1f}%)')
                plt.legend()
            
            plt.title("API Endpoint Error Rate Comparison")
            plt.xlabel("Endpoint")
            plt.ylabel("Error Rate (%)")
            plt.grid(True, axis='y')
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "endpoint_error_rate_comparison.png")
            
        # Create resource usage chart if profiling was enabled
        has_resource_data = False
        for endpoint_name, result in results.items():
            if endpoint_name == "summary":
                continue
                
            for concurrency, stats in result.get("concurrency_results", {}).items():
                if "resource_usage" in stats:
                    has_resource_data = True
                    break
            
            if has_resource_data:
                break
        
        if has_resource_data:
            plt.figure(figsize=(12, 6))
            endpoints = []
            cpu_usage = []
            memory_usage = []
            
            for endpoint_name, result in results.items():
                if endpoint_name == "summary":
                    continue
                    
                if "concurrency_results" in result:
                    # Use the highest concurrency level for comparison
                    max_concurrency = max(result["concurrency_results"].keys())
                    stats = result["concurrency_results"][max_concurrency]
                    
                    if "resource_usage" in stats:
                        endpoints.append(endpoint_name)
                        cpu_usage.append(stats["resource_usage"].get("process_cpu_percent", 0))
                        memory_usage.append(stats["resource_usage"].get("process_memory_mb", 0))
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # CPU usage subplot
            ax1.bar(endpoints, cpu_usage)
            ax1.set_title("CPU Usage by Endpoint")
            ax1.set_xlabel("Endpoint")
            ax1.set_ylabel("CPU Usage (%)")
            ax1.grid(True, axis='y')
            ax1.set_xticklabels(endpoints, rotation=45, ha="right")
            
            # Memory usage subplot
            ax2.bar(endpoints, memory_usage)
            ax2.set_title("Memory Usage by Endpoint")
            ax2.set_xlabel("Endpoint")
            ax2.set_ylabel("Memory Usage (MB)")
            ax2.grid(True, axis='y')
            ax2.set_xticklabels(endpoints, rotation=45, ha="right")
            
            plt.tight_layout()
            plt.savefig(output_dir / "endpoint_resource_usage.png")


def test_api_performance(config, output_dir):
    """
    Test API performance.
    
    Parameters:
    -----------
    config : dict
        Test configuration
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    tester = APIPerformanceTester(
        base_url=config["api_url"],
        enable_auth=config["enable_auth"],
        enable_resource_profiling=config.get("enable_resource_profiling", False)
    )
    
    # Run endpoint tests
    endpoint_results = tester.run_endpoint_tests(config)
    
    # Run FastAPI client tests if available
    client_results = tester.test_fastapi_client_performance(
        tester.get_endpoints(), 
        config
    )
    
    # Create visualizations
    tester.create_results_visualization(endpoint_results, output_dir)
    
    # Generate performance insights
    performance_insights = generate_performance_insights(endpoint_results, client_results)
    
    # Save insights to file
    with open(output_dir / "performance_insights.md", "w") as f:
        f.write(performance_insights)
    
    # Return combined results
    return {
        "endpoint_tests": endpoint_results,
        "client_tests": client_results,
        "config": config,
        "insights": performance_insights
    }


def generate_performance_insights(endpoint_results, client_results):
    """
    Generate performance insights from test results.
    
    Parameters:
    -----------
    endpoint_results : dict
        Results from endpoint tests
    client_results : dict
        Results from client tests
        
    Returns:
    --------
    str
        Markdown-formatted performance insights
    """
    summary = endpoint_results.get("summary", {})
    
    insights = [
        "# API Performance Test Insights\n",
        f"## Summary\n",
        f"- Total endpoints tested: {summary.get('total_endpoints', 0)}",
        f"- Endpoints meeting all thresholds: {summary.get('endpoints_meeting_thresholds', 0)}",
        f"- Endpoints with latency issues: {summary.get('endpoints_with_latency_issues', 0)}",
        f"- Endpoints with throughput issues: {summary.get('endpoints_with_throughput_issues', 0)}",
        f"- Endpoints with error rate issues: {summary.get('endpoints_with_error_issues', 0)}\n",
        "## Endpoint Performance\n"
    ]
    
    # Add per-endpoint insights
    for endpoint_name, result in endpoint_results.items():
        if endpoint_name == "summary":
            continue
            
        insights.append(f"### {endpoint_name} (`{result.get('method', '')} {result.get('path', '')}`)\n")
        
        # Add latency information
        insights.append(f"- **Latency**: {result.get('latency_ms', 0):.2f} ms")
        
        # Add concurrency information if available
        if "concurrency_results" in result:
            # Get the highest concurrency level
            max_concurrency = max(result["concurrency_results"].keys())
            stats = result["concurrency_results"][max_concurrency]
            
            insights.append(f"- **Max Throughput**: {stats.get('requests_per_second', 0):.2f} requests/second at concurrency {max_concurrency}")
            insights.append(f"- **Error Rate**: {stats.get('error_rate', 0) * 100:.2f}%")
            
            # Add error type distribution if there are errors
            if stats.get("error_count", 0) > 0 and "error_types" in stats:
                insights.append("- **Error Types**:")
                for error_type, count in stats["error_types"].items():
                    insights.append(f"  - {error_type}: {count} occurrences")
            
            # Add resource usage if available
            if "resource_usage" in stats:
                ru = stats["resource_usage"]
                insights.append("- **Resource Usage**:")
                insights.append(f"  - CPU: {ru.get('process_cpu_percent', 0):.2f}%")
                insights.append(f"  - Memory: {ru.get('process_memory_mb', 0):.2f} MB")
        
        # Add threshold satisfaction information
        if "meets_thresholds" in result:
            meets_thresholds = result["meets_thresholds"]
            
            if meets_thresholds.get("overall", False):
                insights.append("- **Performance Assessment**: Meets all performance thresholds ✅")
            else:
                insights.append("- **Performance Assessment**: Does not meet all performance thresholds ❌")
                
                # Add specific threshold failures
                issues = []
                if not meets_thresholds.get("latency", True):
                    issues.append("Latency above threshold")
                if not meets_thresholds.get("throughput", True):
                    issues.append("Throughput below threshold")
                if not meets_thresholds.get("error_rate", True):
                    issues.append("Error rate above threshold")
                
                if issues:
                    insights.append("  - Issues: " + ", ".join(issues))
        
        insights.append("\n")
    
    # Add client test insights if available and different from endpoint results
    if isinstance(client_results, dict) and client_results.get("status") != "skipped":
        insights.append("## FastAPI Test Client Results\n")
        
        for endpoint_name, result in client_results.items():
            insights.append(f"### {endpoint_name}\n")
            
            # Add latency information
            insights.append(f"- **Latency**: {result.get('latency_ms', 0):.2f} ms")
            
            # Add statistics if available
            if "stats" in result:
                stats = result["stats"]
                insights.append(f"- **Mean Response Time**: {stats.get('mean', 0) * 1000:.2f} ms")
                insights.append(f"- **P95 Response Time**: {stats.get('p95', 0) * 1000:.2f} ms")
                insights.append(f"- **P99 Response Time**: {stats.get('p99', 0) * 1000:.2f} ms")
                insights.append(f"- **Requests per Second**: {stats.get('requests_per_second', 0):.2f}")
                insights.append(f"- **Error Rate**: {stats.get('error_rate', 0) * 100:.2f}%")
            
            insights.append("\n")
    
    # Add recommendations
    insights.append("## Recommendations\n")
    
    # Base recommendations on test results
    if summary.get("endpoints_with_latency_issues", 0) > 0:
        insights.append("- **Latency Improvement**: Consider optimizing slow endpoints:")
        for endpoint_name, status in summary.get("endpoint_statuses", {}).items():
            if status.get("latency_issue", False):
                insights.append(f"  - `{endpoint_name}`")
    
    if summary.get("endpoints_with_throughput_issues", 0) > 0:
        insights.append("- **Throughput Improvement**: Consider scaling or optimizing endpoints with throughput issues:")
        for endpoint_name, status in summary.get("endpoint_statuses", {}).items():
            if status.get("throughput_issue", False):
                insights.append(f"  - `{endpoint_name}`")
    
    if summary.get("endpoints_with_error_issues", 0) > 0:
        insights.append("- **Error Handling**: Investigate and fix endpoints with high error rates:")
        for endpoint_name, status in summary.get("endpoint_statuses", {}).items():
            if status.get("error_rate_issue", False):
                insights.append(f"  - `{endpoint_name}`")
    
    return "\n".join(insights)


def run_tests(output_dir):
    """
    Run all API performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    create_directory(plots_dir)
    
    # Get test configuration
    config = get_test_config()
    print(f"API URL: {config['api_url']}")
    print(f"Test intensity: {config['test_intensity']}")
    print(f"Resource profiling: {'Enabled' if config.get('enable_resource_profiling', False) else 'Disabled'}")
    
    # Results dictionary
    results = {
        "status": "success",
        "metrics": {
            "api_url": config["api_url"],
            "api_components_available": API_COMPONENTS_AVAILABLE
        },
        "test_results": {}
    }
    
    # Run tests
    print("Running API performance tests...")
    try:
        api_results = test_api_performance(config, plots_dir)
        results["test_results"] = api_results
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    # Save detailed results
    with open(output_dir / "api_performance.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    if "test_results" in results and "endpoint_tests" in results["test_results"]:
        summary = results["test_results"]["endpoint_tests"].get("summary", {})
        print("\nTest Summary:")
        print(f"Total endpoints tested: {summary.get('total_endpoints', 0)}")
        print(f"Endpoints meeting thresholds: {summary.get('endpoints_meeting_thresholds', 0)}")
        print(f"Endpoints with performance issues: {summary.get('total_endpoints', 0) - summary.get('endpoints_meeting_thresholds', 0)}")
    
    return results


if __name__ == "__main__":
    # Run tests with default settings
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"performance_results/api_{timestamp}")
    create_directory(output_dir)
    
    results = run_tests(output_dir)
    
    print(f"Tests completed. Results saved to {output_dir}") 