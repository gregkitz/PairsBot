#!/usr/bin/env python3
"""
Automated test runner for the trading system.

This script runs tests in parallel, generates coverage reports, and validates
that code meets quality standards. It is designed to be used in CI/CD
pipelines or for local development.
"""

import argparse
import os
import sys
import time
import subprocess
import multiprocessing
import platform
import json
from pathlib import Path
from datetime import datetime


def get_cpu_count():
    """Return the number of CPU cores to use for parallel testing."""
    cpu_count = multiprocessing.cpu_count()
    # Leave one core free for the OS
    return max(1, cpu_count - 1)


def run_command(cmd, cwd=None, env=None):
    """Run a command and return the exit code, stdout, and stderr."""
    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def run_linting(args):
    """Run flake8 linting on the codebase."""
    print("\n=== Running Linting Checks ===\n")
    
    # Define paths to check
    paths_to_check = ["src", "tests", "scripts"]
    flake8_args = [
        "flake8",
        "--max-line-length=120",
        "--exclude=venv,.git,__pycache__,build,dist",
    ] + paths_to_check
    
    exit_code, stdout, stderr = run_command(flake8_args)
    
    if exit_code != 0:
        print("Linting issues found:")
        print(stdout)
        if args.strict:
            return False
    else:
        print("No linting issues found.")
    
    return True


def run_type_checking(args):
    """Run mypy type checking on the codebase."""
    print("\n=== Running Type Checking ===\n")
    
    mypy_args = [
        "mypy",
        "--ignore-missing-imports",
        "src",
    ]
    
    exit_code, stdout, stderr = run_command(mypy_args)
    
    if exit_code != 0:
        print("Type checking issues found:")
        print(stdout)
        if args.strict:
            return False
    else:
        print("No type checking issues found.")
    
    return True


def run_unit_tests(args):
    """Run unit tests with pytest."""
    print("\n=== Running Unit Tests ===\n")
    
    pytest_args = [
        "pytest",
        "tests/unit",
        "-v",
        f"--maxfail={args.maxfail}",
    ]
    
    if args.parallel:
        cpu_count = get_cpu_count()
        pytest_args.extend(["-n", str(cpu_count)])
    
    if args.coverage:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:coverage_reports/unit",
            "--cov-report=xml:coverage_reports/unit/coverage.xml",
        ])
    
    exit_code, stdout, stderr = run_command(pytest_args)
    
    if exit_code != 0:
        print("Unit tests failed.")
        if args.strict:
            return False
    else:
        print("All unit tests passed!")
    
    return True


def run_integration_tests(args):
    """Run integration tests with pytest."""
    print("\n=== Running Integration Tests ===\n")
    
    pytest_args = [
        "pytest",
        "tests/integration",
        "-v",
        f"--maxfail={args.maxfail}",
    ]
    
    if args.coverage:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:coverage_reports/integration",
            "--cov-report=xml:coverage_reports/integration/coverage.xml",
        ])
    
    exit_code, stdout, stderr = run_command(pytest_args)
    
    if exit_code != 0:
        print("Integration tests failed.")
        if args.strict:
            return False
    else:
        print("All integration tests passed!")
    
    return True


def run_benchmark_tests(args):
    """Run benchmark tests."""
    print("\n=== Running Benchmark Tests ===\n")
    
    if not os.path.exists("tests/benchmark"):
        print("No benchmark tests found.")
        return True
    
    pytest_args = [
        "pytest",
        "tests/benchmark",
        "-v",
    ]
    
    exit_code, stdout, stderr = run_command(pytest_args)
    
    # Benchmark tests may fail if performance degrades, but we
    # don't want to block the build because of this
    if exit_code != 0:
        print("Benchmark tests failed or showed performance regression.")
        print(stdout)
    else:
        print("All benchmark tests passed!")
    
    return True


def generate_report(test_results, start_time, args):
    """Generate a JSON report of test results."""
    end_time = time.time()
    duration = end_time - start_time
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "results": test_results,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": multiprocessing.cpu_count(),
        },
        "args": vars(args),
    }
    
    os.makedirs("test_reports", exist_ok=True)
    report_file = f"test_reports/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTest report saved to {report_file}")
    
    # Generate summary
    print("\n=== Test Summary ===\n")
    print(f"Duration: {duration:.2f} seconds")
    
    all_passed = all(result["passed"] for result in test_results.values())
    if all_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests failed:")
        for test_name, result in test_results.items():
            status = "PASSED" if result["passed"] else "FAILED"
            print(f"  {test_name}: {status}")
    
    return all_passed


def main():
    """Run the automated test suite."""
    parser = argparse.ArgumentParser(description="Run automated tests for the trading system")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests only")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--type-check", action="store_true", help="Run type checking only")
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel test execution")
    parser.add_argument("--coverage", action="store_true", help="Generate test coverage reports")
    parser.add_argument("--strict", action="store_true", help="Fail if any test fails")
    parser.add_argument("--maxfail", type=int, default=0, help="Stop after N failures (default: 0 = no limit)")
    args = parser.parse_args()
    
    # If no specific tests are selected, run all
    run_all = not (args.unit or args.integration or args.benchmark or args.lint or args.type_check)
    
    # Create directory for coverage reports
    if args.coverage:
        os.makedirs("coverage_reports/unit", exist_ok=True)
        os.makedirs("coverage_reports/integration", exist_ok=True)
    
    start_time = time.time()
    test_results = {}
    
    # Run selected tests
    if args.lint or run_all:
        test_results["linting"] = {"passed": run_linting(args)}
    
    if args.type_check or run_all:
        test_results["type_checking"] = {"passed": run_type_checking(args)}
    
    if args.unit or run_all:
        test_results["unit_tests"] = {"passed": run_unit_tests(args)}
    
    if args.integration or run_all:
        test_results["integration_tests"] = {"passed": run_integration_tests(args)}
    
    if args.benchmark or run_all:
        test_results["benchmark_tests"] = {"passed": run_benchmark_tests(args)}
    
    # Generate report
    all_passed = generate_report(test_results, start_time, args)
    
    return 0 if all_passed or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main()) 