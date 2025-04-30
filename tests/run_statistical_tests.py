#!/usr/bin/env python3
"""
Script to run advanced statistical method tests.

This script provides a convenient way to run the statistical tests with various options.
"""

import os
import sys
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def run_tests(args):
    """Run the statistical method tests."""
    # Construct pytest command
    pytest_args = ['pytest']
    
    # Add verbose flag if requested
    if args.verbose:
        pytest_args.append('-v')
    
    # Add specific test files
    if args.advanced_only:
        pytest_args.append('tests/unit/cointegration/test_advanced_statistical_methods.py')
    elif args.integrated_only:
        pytest_args.append('tests/unit/cointegration/test_integrated_statistical_methods.py')
    elif args.specific:
        pytest_args.append(args.specific)
    else:
        # Run all statistical method tests
        pytest_args.extend([
            'tests/unit/cointegration/test_statistical_methods.py',
            'tests/unit/cointegration/test_advanced_statistical_methods.py',
            'tests/unit/cointegration/test_integrated_statistical_methods.py'
        ])
    
    # Add keyword filter if specified
    if args.keyword:
        pytest_args.extend(['-k', args.keyword])
    
    # Add xvs option for coverage
    if args.coverage:
        pytest_args.extend(['--cov=src.cointegration', '--cov-report=term'])
    
    # Add option to generate HTML report
    if args.html_report:
        report_dir = args.html_report_dir or 'test_reports/statistical_methods'
        os.makedirs(report_dir, exist_ok=True)
        pytest_args.extend(['--html', f'{report_dir}/report.html', '--self-contained-html'])
    
    # Run pytest with the constructed arguments
    print(f"Running tests with command: {' '.join(pytest_args)}")
    result = subprocess.run(pytest_args)
    
    return result.returncode

def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run statistical method tests')
    
    # Test selection options
    parser.add_argument('--advanced-only', action='store_true', 
                        help='Run only advanced statistical method tests')
    parser.add_argument('--integrated-only', action='store_true', 
                        help='Run only integrated statistical method tests')
    parser.add_argument('--specific', type=str, 
                        help='Run a specific test file')
    parser.add_argument('-k', '--keyword', type=str, 
                        help='Only run tests matching the given keyword expression')
    
    # Output options
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--coverage', action='store_true', 
                        help='Generate coverage report')
    parser.add_argument('--html-report', action='store_true', 
                        help='Generate HTML test report')
    parser.add_argument('--html-report-dir', type=str, 
                        help='Directory to store HTML report (default: test_reports/statistical_methods)')
    
    args = parser.parse_args()
    
    # Run the tests
    exit_code = run_tests(args)
    
    # Exit with the same code as pytest
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 