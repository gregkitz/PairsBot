"""
Script to test our regime scheduler for a short period.

This script demonstrates the regime scheduler with more frequent updates
for testing purposes.
"""

import time
import logging
import subprocess
import schedule
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_regime_update():
    """Run the regime update script."""
    logger.info("Running regime update...")
    
    try:
        # Run the script as a subprocess
        result = subprocess.run(
            ["python", "adapt_regime_parameters.py"],
            capture_output=True,
            text=True
        )
        
        # Log the output
        if result.returncode == 0:
            logger.info("Regime update completed successfully")
        else:
            logger.error(f"Regime update failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running regime update: {e}")
        return False

def main():
    """Run a short test of the scheduler."""
    # Run immediately
    run_regime_update()
    
    # Schedule to run every 30 seconds for testing
    schedule.every(30).seconds.do(run_regime_update)
    
    logger.info("Test scheduler started - will run for 1 minute")
    logger.info("Updates scheduled every 30 seconds")
    
    # Run for 1 minute
    end_time = time.time() + 60
    
    try:
        while time.time() < end_time:
            schedule.run_pending()
            time.sleep(1)
        
        logger.info("Test completed")
    except KeyboardInterrupt:
        logger.info("Test stopped by user")
    except Exception as e:
        logger.error(f"Error during test: {e}")

if __name__ == "__main__":
    main() 