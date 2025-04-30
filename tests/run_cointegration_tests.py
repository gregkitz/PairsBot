#!/usr/bin/env python3
"""
Script to run cointegration tests with academic-grade test data.

This script generates the necessary test data and runs the cointegration test suite.
"""

import os
import sys
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def ensure_test_data_exists():
    """Ensure the academic test data exists, generate it if it doesn't."""
    fixtures_dir = os.path.join('tests', 'fixtures', 'cointegration_test_data')
    metadata_file = os.path.join(fixtures_dir, 'metadata.json')
    
    if not os.path.exists(fixtures_dir) or not os.path.exists(metadata_file):
        print("Generating academic test data for cointegration testing...")
        from scripts.generate_test_data import generate_academic_test_dataset
        generate_academic_test_dataset(fixtures_dir)
        print("Test data generation complete.")
    else:
        print("Using existing test data from", fixtures_dir)

def run_tests(args):
    """Run the cointegration tests."""
    # Ensure test data exists
    ensure_test_data_exists()
    
    # Construct pytest command
    pytest_args = ['pytest']
    
    # Add verbose flag if requested
    if args.verbose:
        pytest_args.append('-v')
    
    # Add specific test file
    pytest_args.append('tests/unit/cointegration/test_cointegration_methods.py')
    
    # Run the tests
    print("Running cointegration tests with command:", " ".join(pytest_args))
    result = subprocess.run(pytest_args)
    
    # Return the exit code
    return result.returncode

def main():
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser(description='Run cointegration tests with academic-grade data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--regenerate-data', action='store_true', help='Force regeneration of test data')
    
    args = parser.parse_args()
    
    # If regeneration is requested, delete existing data
    if args.regenerate_data:
        fixtures_dir = os.path.join('tests', 'fixtures', 'cointegration_test_data')
        import shutil
        if os.path.exists(fixtures_dir):
            print(f"Removing existing test data from {fixtures_dir}")
            shutil.rmtree(fixtures_dir)
    
    # Run the tests
    exit_code = run_tests(args)
    
    # Exit with the same code as pytest
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 