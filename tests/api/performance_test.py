#!/usr/bin/env python3
"""
Performance testing for API endpoints.

This script runs performance tests against API endpoints to measure response times,
throughput, and error rates under various load conditions.
"""

import os
import sys
import time
import json
import argparse
import asyncio
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from config
try:
    from config.test.api_test_config import get_endpoint, get_auth_headers
except ImportError:
    # Fallback default functions if config module not available
    BASE_URL = "http://localhost:5000/api"
    def get_endpoint(name, *args):
        endpoints = {
            "health": "/health",
            "backtest_submit": "/backtest/submit",
            "pairs": "/pairs"
        }
        return f"{BASE_URL}{endpoints.get(name, '/')}"
    
    def get_auth_headers():
        return {}

# Import utilities
try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

# Try to import aiohttp for async testing
try:
    import aiohttp
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("Warning: aiohttp not available. Async tests will be skipped.")

# Try to import requests for sync testing
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Sync tests will be skipped.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="API performance testing tool")
    
    parser.add_argument("--endpoint", choices=["health", "pairs", "backtest", "all"], 
                      default="health", help="Endpoint to test")
    parser.add_argument("--method", choices=["GET", "POST"], default="GET", 
                      help="HTTP method to use")
    parser.add_argument("--users", type=int, default=10, 
                      help="Number of simulated users")
    parser.add_argument("--requests", type=int, default=100, 
                      help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=5, 
                      help="Number of concurrent requests")
    parser.add_argument("--output", default="performance_results", 
                      help="Directory to save results")
    parser.add_argument("--plot", action="store_true", 
                      help="Generate performance plots")
    parser.add_argument("--delay", type=float, default=0, 
                      help="Delay between requests in seconds")
    
    return parser.parse_args()


class PerformanceTest:
    """Base class for API performance testing."""
    
    def __init__(self, endpoint, method="GET", data=None, headers=None):
        """
        Initialize performance test.
        
        Parameters:
        -----------
        endpoint : str
            API endpoint to test
        method : str
            HTTP method (GET, POST)
        data : dict, optional
            Data to send in request body
        headers : dict, optional
            Headers to include in request
        """
        self.endpoint = endpoint
        self.method = method
        self.data = data or {}
        self.headers = headers or {}
        self.headers.update(get_auth_headers())
        self.results = []
    
    def run_test(self, num_requests, concurrency=1, delay=0):
        """
        Run performance test.
        
        Parameters:
        -----------
        num_requests : int
            Number of requests to send
        concurrency : int
            Number of concurrent requests
        delay : float
            Delay between requests in seconds
            
        Returns:
        --------
        dict
            Test results
        """
        raise NotImplementedError("Subclasses must implement run_test method")
    
    def calculate_statistics(self):
        """
        Calculate performance statistics from results.
        
        Returns:
        --------
        dict
            Performance statistics
        """
        if not self.results:
            return {
                "min_time": 0,
                "max_time": 0,
                "mean_time": 0,
                "median_time": 0,
                "p95_time": 0,
                "p99_time": 0,
                "std_dev": 0,
                "requests_per_second": 0,
                "success_rate": 0,
                "error_rate": 0,
                "sample_size": 0
            }
        
        # Extract timings and success flags
        timings = [r["time"] for r in self.results]
        successes = [r["success"] for r in self.results]
        
        # Calculate statistics
        stats = {
            "min_time": min(timings),
            "max_time": max(timings),
            "mean_time": statistics.mean(timings),
            "median_time": statistics.median(timings),
            "p95_time": np.percentile(timings, 95),
            "p99_time": np.percentile(timings, 99),
            "std_dev": statistics.stdev(timings) if len(timings) > 1 else 0,
            "requests_per_second": len(timings) / sum(timings),
            "success_rate": sum(successes) / len(successes) * 100,
            "error_rate": (len(successes) - sum(successes)) / len(successes) * 100,
            "sample_size": len(timings)
        }
        
        return stats
    
    def plot_results(self, output_dir):
        """
        Generate performance plots from results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        if not self.results:
            return
        
        # Create output directory
        plots_dir = Path(output_dir) / "plots"
        create_directory(plots_dir)
        
        # Extract data
        timings = [r["time"] for r in self.results]
        timestamps = [r["timestamp"] - self.results[0]["timestamp"] for r in self.results]
        
        # Plot response time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(timings, bins=30, alpha=0.7)
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'Response Time Distribution for {self.endpoint}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plots_dir / f"{self.endpoint.replace('/', '_')}_response_distribution.png")
        plt.close()
        
        # Plot response time over time
        plt.figure(figsize=(12, 6))
        plt.scatter(timestamps, timings, alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Response Time (seconds)')
        plt.title(f'Response Times Over Test Duration for {self.endpoint}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plots_dir / f"{self.endpoint.replace('/', '_')}_response_time.png")
        plt.close()
        
        # Calculate and plot moving average
        window = min(10, len(timings))
        if window > 1:
            moving_avg = [
                statistics.mean(timings[max(0, i-window):i+1]) 
                for i in range(len(timings))
            ]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, moving_avg)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Moving Average Response Time (seconds)')
            plt.title(f'Response Time Trend for {self.endpoint}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plots_dir / f"{self.endpoint.replace('/', '_')}_moving_average.png")
            plt.close()
        
        print(f"Plots saved to {plots_dir}")
    
    def save_results(self, output_dir):
        """
        Save test results to file.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        # Create output directory
        create_directory(output_dir)
        
        # Get statistics
        stats = self.calculate_statistics()
        
        # Create result data
        result_data = {
            "endpoint": self.endpoint,
            "method": self.method,
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "details": self.results
        }
        
        # Save to file
        filename = f"{self.endpoint.replace('/', '_')}_{self.method}_{int(time.time())}.json"
        with open(Path(output_dir) / filename, "w") as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Results saved to {Path(output_dir) / filename}")
        
        # Return statistics for summary
        return stats


class SyncPerformanceTest(PerformanceTest):
    """Synchronous performance test using requests library."""
    
    def __init__(self, endpoint, method="GET", data=None, headers=None):
        """Initialize synchronous performance test."""
        super().__init__(endpoint, method, data, headers)
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for SyncPerformanceTest")
    
    def make_request(self):
        """
        Make a single request and record results.
        
        Returns:
        --------
        dict
            Request result including success flag and response time
        """
        start_time = time.time()
        result = {
            "timestamp": start_time,
            "success": False,
            "time": 0,
            "status_code": None,
            "error": None
        }
        
        try:
            if self.method == "GET":
                response = requests.get(
                    self.endpoint, 
                    headers=self.headers,
                    timeout=10
                )
            elif self.method == "POST":
                response = requests.post(
                    self.endpoint, 
                    json=self.data,
                    headers=self.headers,
                    timeout=10
                )
            else:
                result["error"] = f"Unsupported method: {self.method}"
                result["time"] = time.time() - start_time
                return result
            
            # Record results
            end_time = time.time()
            result["time"] = end_time - start_time
            result["status_code"] = response.status_code
            result["success"] = 200 <= response.status_code < 300
            
            # Include response data for debugging if needed
            # result["response"] = response.text
            
        except Exception as e:
            end_time = time.time()
            result["time"] = end_time - start_time
            result["error"] = str(e)
        
        return result
    
    def run_test(self, num_requests, concurrency=1, delay=0):
        """
        Run synchronous performance test with ThreadPoolExecutor.
        
        Parameters:
        -----------
        num_requests : int
            Number of requests to send
        concurrency : int
            Number of concurrent requests
        delay : float
            Delay between requests in seconds
            
        Returns:
        --------
        dict
            Test results
        """
        self.results = []
        
        print(f"Running test: {self.method} {self.endpoint}")
        print(f"Requests: {num_requests}, Concurrency: {concurrency}")
        
        # Define batch size based on concurrency
        batch_size = min(concurrency, num_requests)
        
        # Prepare for execution
        total_start_time = time.time()
        
        # Run requests in batches
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for i in range(0, num_requests, batch_size):
                # Submit batch of requests
                batch_count = min(batch_size, num_requests - i)
                futures = [executor.submit(self.make_request) for _ in range(batch_count)]
                
                # Process results as they complete
                for future in futures:
                    result = future.result()
                    self.results.append(result)
                    
                    # Print progress
                    if (len(self.results) % 10 == 0) or (len(self.results) == num_requests):
                        success_rate = sum(r["success"] for r in self.results) / len(self.results) * 100
                        print(f"Completed {len(self.results)}/{num_requests} requests ({success_rate:.1f}% success)")
                
                # Apply delay if specified
                if delay > 0 and i + batch_size < num_requests:
                    time.sleep(delay)
        
        total_time = time.time() - total_start_time
        
        # Calculate and print summary statistics
        stats = self.calculate_statistics()
        
        print("\nTest Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Requests per second: {stats['requests_per_second']:.2f}")
        print(f"Mean response time: {stats['mean_time'] * 1000:.2f} ms")
        print(f"Median response time: {stats['median_time'] * 1000:.2f} ms")
        print(f"95th percentile: {stats['p95_time'] * 1000:.2f} ms")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        
        return stats


class AsyncPerformanceTest(PerformanceTest):
    """Asynchronous performance test using aiohttp library."""
    
    def __init__(self, endpoint, method="GET", data=None, headers=None):
        """Initialize asynchronous performance test."""
        super().__init__(endpoint, method, data, headers)
        
        if not ASYNC_AVAILABLE:
            raise ImportError("aiohttp library is required for AsyncPerformanceTest")
    
    async def make_request(self, session):
        """
        Make a single request and record results.
        
        Parameters:
        -----------
        session : aiohttp.ClientSession
            HTTP session for making requests
            
        Returns:
        --------
        dict
            Request result
        """
        start_time = time.time()
        result = {
            "timestamp": start_time,
            "success": False,
            "time": 0,
            "status_code": None,
            "error": None
        }
        
        try:
            if self.method == "GET":
                async with session.get(self.endpoint, headers=self.headers) as response:
                    # Read response to ensure request is complete
                    await response.text()
                    
                    # Record results
                    end_time = time.time()
                    result["time"] = end_time - start_time
                    result["status_code"] = response.status
                    result["success"] = 200 <= response.status < 300
            
            elif self.method == "POST":
                async with session.post(self.endpoint, json=self.data, headers=self.headers) as response:
                    # Read response to ensure request is complete
                    await response.text()
                    
                    # Record results
                    end_time = time.time()
                    result["time"] = end_time - start_time
                    result["status_code"] = response.status
                    result["success"] = 200 <= response.status < 300
            
            else:
                result["error"] = f"Unsupported method: {self.method}"
                result["time"] = time.time() - start_time
        
        except Exception as e:
            end_time = time.time()
            result["time"] = end_time - start_time
            result["error"] = str(e)
        
        return result
    
    async def run_batch(self, session, count, semaphore, delay=0):
        """
        Run a batch of requests with rate limiting via semaphore.
        
        Parameters:
        -----------
        session : aiohttp.ClientSession
            Session for making requests
        count : int
            Number of requests to make
        semaphore : asyncio.Semaphore
            Semaphore for limiting concurrency
        delay : float
            Delay between requests in seconds
        """
        for i in range(count):
            async with semaphore:
                result = await self.make_request(session)
                self.results.append(result)
                
                # Print progress
                if (len(self.results) % 10 == 0) or (len(self.results) == count):
                    success_count = sum(1 for r in self.results if r["success"])
                    success_rate = (success_count / len(self.results)) * 100
                    print(f"Completed {len(self.results)}/{count} requests ({success_rate:.1f}% success)")
                
                if delay > 0:
                    await asyncio.sleep(delay)
    
    async def _run_test_async(self, num_requests, concurrency=1, delay=0):
        """
        Run asynchronous performance test.
        
        Parameters:
        -----------
        num_requests : int
            Number of requests to send
        concurrency : int
            Maximum number of concurrent requests
        delay : float
            Delay between requests in seconds
        """
        self.results = []
        
        print(f"Running async test: {self.method} {self.endpoint}")
        print(f"Requests: {num_requests}, Concurrency: {concurrency}")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        # Create session for all requests
        async with aiohttp.ClientSession() as session:
            await self.run_batch(session, num_requests, semaphore, delay)
    
    def run_test(self, num_requests, concurrency=1, delay=0):
        """
        Run performance test using asyncio.
        
        Parameters:
        -----------
        num_requests : int
            Number of requests to send
        concurrency : int
            Number of concurrent requests
        delay : float
            Delay between requests in seconds
            
        Returns:
        --------
        dict
            Test results
        """
        # Record start time
        total_start_time = time.time()
        
        # Run the async test
        asyncio.run(self._run_test_async(num_requests, concurrency, delay))
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Calculate and print summary statistics
        stats = self.calculate_statistics()
        
        print("\nTest Results:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Requests per second: {stats['requests_per_second']:.2f}")
        print(f"Mean response time: {stats['mean_time'] * 1000:.2f} ms")
        print(f"Median response time: {stats['median_time'] * 1000:.2f} ms")
        print(f"95th percentile: {stats['p95_time'] * 1000:.2f} ms")
        print(f"Success rate: {stats['success_rate']:.2f}%")
        
        return stats


def create_endpoint_test_data(endpoint, method):
    """
    Create test data for specific endpoint.
    
    Parameters:
    -----------
    endpoint : str
        Endpoint name
    method : str
        HTTP method
        
    Returns:
    --------
    tuple
        (endpoint_url, data, method)
    """
    # Default empty data
    data = {}
    
    if endpoint == "health":
        url = get_endpoint("health")
        method = "GET"
    
    elif endpoint == "pairs":
        url = get_endpoint("pairs")
        method = "GET"
    
    elif endpoint == "backtest":
        url = get_endpoint("backtest_submit")
        method = "POST"
        # Create test backtest configuration
        data = {
            "config": {
                "pair": "CL-HO",
                "start_date": "2022-01-01",
                "end_date": "2022-01-31",
                "hedge_ratio": 0.8,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "lookback_period": 20
            }
        }
    
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    return url, data, method


def main():
    """Run the API performance test with specified parameters."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    create_directory(output_dir)
    
    # Save test configuration
    with open(output_dir / "test_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # List of endpoints to test
    if args.endpoint == "all":
        endpoints = ["health", "pairs", "backtest"]
    else:
        endpoints = [args.endpoint]
    
    # Store results for summary
    all_results = {}
    
    # Run tests for each endpoint
    for endpoint in endpoints:
        print(f"\n=== Testing endpoint: {endpoint} ===\n")
        
        # Get endpoint URL and test data
        url, data, method = create_endpoint_test_data(endpoint, args.method)
        
        try:
            # Choose test class based on availability
            if ASYNC_AVAILABLE:
                test = AsyncPerformanceTest(url, method, data)
            elif REQUESTS_AVAILABLE:
                test = SyncPerformanceTest(url, method, data)
            else:
                print("Error: Neither aiohttp nor requests is available. Cannot run tests.")
                return
            
            # Run the test
            stats = test.run_test(args.requests, args.concurrency, args.delay)
            
            # Save detailed results
            test.save_results(output_dir)
            
            # Generate plots if requested
            if args.plot:
                test.plot_results(output_dir)
            
            # Store summary stats
            all_results[endpoint] = {
                "url": url,
                "method": method,
                "stats": stats
            }
            
        except Exception as e:
            print(f"Error testing {endpoint}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary report
    summary = {
        "timestamp": timestamp,
        "configuration": vars(args),
        "results": all_results
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n=== Test Summary ===\n")
    print(f"Test completed at: {timestamp}")
    print(f"Results saved to: {output_dir}")
    
    for endpoint, result in all_results.items():
        stats = result["stats"]
        print(f"\n{endpoint} ({result['method']} {result['url']}):")
        print(f"  Requests/second: {stats['requests_per_second']:.2f}")
        print(f"  Mean response: {stats['mean_time'] * 1000:.2f} ms")
        print(f"  95th percentile: {stats['p95_time'] * 1000:.2f} ms")
        print(f"  Success rate: {stats['success_rate']:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError running performance tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 