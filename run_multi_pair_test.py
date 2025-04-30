#!/usr/bin/env python3
"""
Run the MultiPairPortfolio unit tests directly.
"""

import os
import sys
import unittest

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Define the test file to run
test_file = 'tests/unit/strategies/test_multi_pair_portfolio.py'

if __name__ == "__main__":
    # Run the test file directly
    test_module = unittest.defaultTestLoader.discover(os.path.dirname(test_file), 
                                                     pattern=os.path.basename(test_file))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_module)
    
    # Return non-zero exit code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 