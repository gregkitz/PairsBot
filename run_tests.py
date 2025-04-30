#!/usr/bin/env python
"""
Run tests for the Intraday Statistical Arbitrage System.
"""

import os
import sys
import unittest

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_tests():
    """Run all the tests in the tests directory."""
    test_path = os.path.join(project_root, 'tests')
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_path)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 