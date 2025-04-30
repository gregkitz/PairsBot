#!/usr/bin/env python3
"""
Runner for performance tests.

This script runs the performance test suite and generates reports.
"""

import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run performance tests for system components")
    
    parser.add_argument("--tests", nargs="+", choices=[
        "data", "model", "trading", "api", "system", "all"
    ], default=["all"], help="Performance tests to run (default: all)")
    
    parser.add_argument("--output", type=str, default="performance_results",
                      help="Directory to save performance results (default: performance_results)")
    
    parser.add_argument("--profile", action="store_true",
                      help="Generate detailed performance profiles")
    
    parser.add_argument("--quick", action="store_true",
                      help="Run with smaller data samples for quick testing")
    
    parser.add_argument("--data-size", type=str, choices=["small", "medium", "large", "production"],
                      default="medium", help="Size of test data to use (default: medium)")
    
    parser.add_argument("--repeats", type=int, default=3,
                      help="Number of times to repeat each test for consistent results (default: 3)")
    
    parser.add_argument("--timeout", type=int, default=3600,
                      help="Timeout for the entire test suite in seconds (default: 3600)")
    
    parser.add_argument("--report-format", type=str, choices=["json", "html", "both"],
                      default="both", help="Format for performance reports (default: both)")
    
    return parser.parse_args()


def setup_environment(args):
    """
    Set up the test environment.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create output directory
    create_directory(args.output)
    
    # Set environment variables
    if args.quick:
        os.environ["PERFORMANCE_QUICK_MODE"] = "1"
    
    os.environ["PERFORMANCE_DATA_SIZE"] = args.data_size
    os.environ["PERFORMANCE_REPEATS"] = str(args.repeats)
    
    # Return the test directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output) / timestamp
    create_directory(result_dir)
    
    return result_dir


def run_data_processing_performance_tests(output_dir):
    """
    Run data processing performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save performance results
    
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Running Data Processing Performance Tests ===\n")
    
    try:
        from tests.performance.test_data_processing_performance import run_tests
        results = run_tests(output_dir)
        return results
    except ImportError:
        print("Data processing performance tests not implemented yet.")
        return {"status": "not_implemented"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_model_training_performance_tests(output_dir):
    """
    Run model training performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save performance results
    
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Running ML Model Training Performance Tests ===\n")
    
    try:
        from tests.performance.test_model_training_performance import run_tests
        results = run_tests(output_dir)
        return results
    except ImportError:
        print("Model training performance tests not implemented yet.")
        return {"status": "not_implemented"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_trading_system_performance_tests(output_dir):
    """
    Run trading system performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save performance results
    
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Running Trading System Performance Tests ===\n")
    
    try:
        from tests.performance.test_trading_system_performance import run_tests
        results = run_tests(output_dir)
        return results
    except ImportError:
        print("Trading system performance tests not implemented yet.")
        return {"status": "not_implemented"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_api_performance_tests(output_dir):
    """
    Run API performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save performance results
    
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Running API Performance Tests ===\n")
    
    try:
        from tests.performance.test_api_performance import run_tests
        results = run_tests(output_dir)
        return results
    except ImportError:
        print("API performance tests not implemented yet.")
        return {"status": "not_implemented"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def run_system_performance_tests(output_dir):
    """
    Run end-to-end system performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save performance results
    
    Returns:
    --------
    dict
        Test results
    """
    print("\n=== Running End-to-End System Performance Tests ===\n")
    
    try:
        from tests.performance.test_system_performance import run_tests
        results = run_tests(output_dir)
        return results
    except ImportError:
        print("System performance tests not implemented yet.")
        return {"status": "not_implemented"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def generate_performance_report(results, output_dir, format="both"):
    """
    Generate a performance report.
    
    Parameters:
    -----------
    results : dict
        Test results
    output_dir : Path
        Directory to save performance results
    format : str
        Report format (json, html, or both)
    """
    # Save raw JSON results
    if format in ["json", "both"]:
        with open(output_dir / "performance_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Generate HTML report
    if format in ["html", "both"]:
        html_content = generate_html_report(results)
        with open(output_dir / "performance_report.html", "w") as f:
            f.write(html_content)


def generate_html_report(results):
    """
    Generate an HTML report from performance results.
    
    Parameters:
    -----------
    results : dict
        Test results
    
    Returns:
    --------
    str
        HTML report content
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .summary { background-color: #f5f5f5; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .chart { margin: 20px 0; }
            .error { color: red; }
            .not-implemented { color: #888; font-style: italic; }
            .metric { margin-bottom: 10px; }
            .metric-name { font-weight: bold; }
            .metric-value { float: right; }
        </style>
    </head>
    <body>
        <h1>Performance Test Results</h1>
        <div class="section summary">
            <h2>Summary</h2>
            <p>Timestamp: {timestamp}</p>
            <p>Total Run Time: {total_time:.2f} seconds</p>
            
            <table>
                <tr>
                    <th>Test Category</th>
                    <th>Status</th>
                    <th>Run Time (s)</th>
                </tr>
    """.format(
        timestamp=results.get("timestamp", "Unknown"),
        total_time=results.get("total_runtime", 0)
    )
    
    # Add summary rows
    categories = ["data_processing", "model_training", "trading_system", "api", "system"]
    for category in categories:
        if category in results:
            category_data = results[category]
            status = category_data.get("status", "Unknown")
            runtime = category_data.get("runtime", 0)
            
            html += f"""
                <tr>
                    <td>{category.replace('_', ' ').title()}</td>
                    <td>{status}</td>
                    <td>{runtime:.2f}</td>
                </tr>
            """
    
    html += """
            </table>
        </div>
    """
    
    # Add detailed sections for each category
    for category in categories:
        if category in results:
            category_data = results[category]
            status = category_data.get("status", "Unknown")
            
            html += f"""
        <div class="section">
            <h2>{category.replace('_', ' ').title()} Performance</h2>
            """
            
            if status == "not_implemented":
                html += """
                <p class="not-implemented">These tests have not been implemented yet.</p>
                """
            elif status == "error":
                html += f"""
                <p class="error">Error running tests: {category_data.get('error', 'Unknown error')}</p>
                """
            else:
                # Add detailed metrics if available
                if "metrics" in category_data:
                    metrics = category_data["metrics"]
                    html += """
                <h3>Performance Metrics</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    """
                    
                    for metric_name, metric_value in metrics.items():
                        html += f"""
                    <tr>
                        <td>{metric_name.replace('_', ' ').title()}</td>
                        <td>{metric_value}</td>
                    </tr>
                        """
                    
                    html += """
                </table>
                    """
                
                # Add test results if available
                if "test_results" in category_data:
                    test_results = category_data["test_results"]
                    html += """
                <h3>Test Results</h3>
                <table>
                    <tr>
                        <th>Test Name</th>
                        <th>Execution Time (s)</th>
                        <th>Memory Usage (MB)</th>
                        <th>Status</th>
                    </tr>
                    """
                    
                    for test_name, test_data in test_results.items():
                        html += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td>{test_data.get('execution_time', 0):.4f}</td>
                        <td>{test_data.get('memory_usage', 0):.2f}</td>
                        <td>{test_data.get('status', 'Unknown')}</td>
                    </tr>
                        """
                    
                    html += """
                </table>
                    """
            
            html += """
        </div>
            """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    output_dir = setup_environment(args)
    
    # Start timing
    start_time = time.time()
    
    # Results dictionary
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "configuration": {
            "tests": args.tests,
            "data_size": args.data_size,
            "profile": args.profile,
            "repeats": args.repeats,
            "quick_mode": args.quick
        }
    }
    
    # Run selected tests
    if "all" in args.tests or "data" in args.tests:
        data_start = time.time()
        data_results = run_data_processing_performance_tests(output_dir)
        data_time = time.time() - data_start
        results["data_processing"] = data_results
        results["data_processing"]["runtime"] = data_time
    
    if "all" in args.tests or "model" in args.tests:
        model_start = time.time()
        model_results = run_model_training_performance_tests(output_dir)
        model_time = time.time() - model_start
        results["model_training"] = model_results
        results["model_training"]["runtime"] = model_time
    
    if "all" in args.tests or "trading" in args.tests:
        trading_start = time.time()
        trading_results = run_trading_system_performance_tests(output_dir)
        trading_time = time.time() - trading_start
        results["trading_system"] = trading_results
        results["trading_system"]["runtime"] = trading_time
    
    if "all" in args.tests or "api" in args.tests:
        api_start = time.time()
        api_results = run_api_performance_tests(output_dir)
        api_time = time.time() - api_start
        results["api"] = api_results
        results["api"]["runtime"] = api_time
    
    if "all" in args.tests or "system" in args.tests:
        system_start = time.time()
        system_results = run_system_performance_tests(output_dir)
        system_time = time.time() - system_start
        results["system"] = system_results
        results["system"]["runtime"] = system_time
    
    # Calculate total runtime
    total_time = time.time() - start_time
    results["total_runtime"] = total_time
    
    # Generate report
    generate_performance_report(results, output_dir, args.report_format)
    
    print(f"\nPerformance tests completed in {total_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Set timeout handler
    import signal
    
    def timeout_handler(signum, frame):
        print("\nPerformance tests timed out.")
        sys.exit(2)
    
    # Parse arguments first to get timeout value
    args = parse_arguments()
    
    # Set the alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nPerformance tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running performance tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cancel the alarm
        signal.alarm(0) 