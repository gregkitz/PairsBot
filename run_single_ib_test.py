#!/usr/bin/env python
"""
Script to run a single Interactive Brokers integration test.

This script runs a single test from the IB integration tests, 
with more verbose output and error handling.

Usage examples:
    # Run the basic connectivity test
    ./run_single_ib_test.py
    
    # Run a specific test
    ./run_single_ib_test.py test_equity_asset_integration
    
    # Run all tests
    ./run_single_ib_test.py all
    
    # Show help and available tests
    ./run_single_ib_test.py -h
"""

import os
import sys
import unittest
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of available tests
AVAILABLE_TESTS = [
    'test_connector_connectivity',
    'test_equity_asset_integration',
    'test_futures_asset_integration',
]

def main():
    """Main function to run a single IB integration test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a single IB integration test.')
    parser.add_argument(
        'test_name', 
        choices=AVAILABLE_TESTS + ['all'],
        nargs='?',
        default='test_connector_connectivity',
        help='Name of the test to run'
    )
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Import the test class
    from tests.integration.test_ib_integration import TestIBIntegration
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    if args.test_name == 'all':
        # Add all tests
        logger.info("Running all IB integration tests")
        for test_name in AVAILABLE_TESTS:
            test_suite.addTest(TestIBIntegration(test_name))
    else:
        # Add the specified test
        logger.info(f"Running test: {args.test_name}")
        test_suite.addTest(TestIBIntegration(args.test_name))
    
    # Run the test
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    print("-" * 80)
    print("Interactive Brokers Integration Test Runner")
    print("")
    print("Before running this script:")
    print("1. Make sure TWS is running and logged in")
    print("2. Ensure TWS API is enabled and listening on port 7497")
    print("3. Be ready to approve API connection in TWS")
    print("-" * 80)
    print("")
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running test: {e}")
        sys.exit(1) 