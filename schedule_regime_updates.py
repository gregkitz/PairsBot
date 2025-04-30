"""
Script to schedule regular updates for our regime detection.

This script sets up a scheduler to run the adapt_regime_parameters.py script
at regular intervals to keep our trading parameters updated with the latest
market regime.
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from datetime import datetime, timedelta
import schedule

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/regime_scheduler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_regime_update():
    """
    Run the adapt_regime_parameters.py script to update trading parameters.
    """
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
            logger.info(f"Regime update completed successfully")
            logger.debug(f"Output: {result.stdout}")
        else:
            logger.error(f"Regime update failed with code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running regime update: {e}")
        return False

def setup_schedule(update_interval=24):
    """
    Set up the scheduler to run updates at the specified interval.
    
    Parameters:
    -----------
    update_interval : int
        Interval in hours between updates
    """
    # Run immediately on startup
    run_regime_update()
    
    if update_interval == 24:
        # Schedule to run daily at a specific time (e.g., midnight)
        schedule.every().day.at("00:00").do(run_regime_update)
        logger.info("Scheduled regime updates to run daily at midnight")
    else:
        # Schedule to run at the specified interval
        schedule.every(update_interval).hours.do(run_regime_update)
        logger.info(f"Scheduled regime updates to run every {update_interval} hours")
    
    # Calculate when the next run will occur
    next_run = schedule.next_run()
    logger.info(f"Next update scheduled for: {next_run}")

def main():
    """Main function to set up the scheduler."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Schedule regime detection updates")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=24, 
        help="Interval in hours between updates (default: 24)"
    )
    parser.add_argument(
        "--daemon", 
        action="store_true", 
        help="Run as a background daemon"
    )
    
    args = parser.parse_args()
    
    # Set up the schedule
    setup_schedule(args.interval)
    
    # Run the scheduler
    logger.info(f"Starting scheduler with update interval of {args.interval} hours")
    logger.info("Press Ctrl+C to exit")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")

if __name__ == "__main__":
    main() 