#!/usr/bin/env python
"""
Script to run Interactive Brokers integration tests.

This script runs the integration tests that require a running TWS instance.
"""

import os
import sys
import unittest
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the IB integration tests."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Print information
    logger.info("Running Interactive Brokers Integration Tests")
    logger.info("Make sure that TWS is running on localhost:7497 (Paper Trading)")
    logger.info("Project root: %s", project_root)
    
    # Discover and run the tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.join(project_root, 'tests', 'integration'),
        pattern='test_ib_*.py'
    )
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(main()) 