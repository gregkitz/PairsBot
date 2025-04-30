#!/usr/bin/env python
"""
Minimal debug script for testing IB TWS connectivity.

This script isolates just the connection part to diagnose TWS connectivity issues.
"""

import sys
import time
import logging
import threading
from ib_insync import IB, util

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for timeout
timed_out = False

def timeout_function():
    """Function to run when timeout occurs."""
    global timed_out
    timed_out = True
    print("\n\nTIMEOUT: Script execution timed out after 30 seconds")
    print("Check if TWS is showing a permission dialog that might be hidden")
    print("Forcing program exit...")
    sys.exit(1)

def main():
    """Connect to IB TWS with minimal code."""
    global timed_out
    
    # Set up timeout
    timeout = 30  # 30 seconds
    timer = threading.Timer(timeout, timeout_function)
    timer.daemon = True
    timer.start()
    
    print("\n=== IB TWS Connection Debug ===\n")
    print("Attempting to connect to TWS...")
    
    # Create IB connection object
    ib = IB()
    
    try:
        # Try to connect
        print("Step 1: Initiating connection...")
        ib.connect('localhost', 7497, clientId=999, timeout=10)
        print("Step 2: Connected successfully!")
        
        # Check connection 
        if ib.isConnected():
            print("Step 3: Connection verified")
        else:
            print("Step 3: Connection verification failed")
            return 1
            
        # Get account information
        print("Step 4: Requesting account list...")
        accounts = ib.managedAccounts()
        print(f"Step 5: Accounts found: {accounts}")
        
        # Try to get a market data point
        print("Step 6: Testing market data query...")
        try:
            contract = ib.qualifyContracts(ib.Stock('AAPL', 'SMART', 'USD'))[0]
            print(f"Step 7: Contract details received for AAPL")
            
            # Request market data
            print("Step 8: Requesting market data (5 second timeout)...")
            ticker = ib.reqMktData(contract)
            
            # Wait briefly for data to arrive
            print("Waiting for market data...")
            for i in range(5):
                if timed_out:
                    break
                ib.sleep(1)
                print(f"  Ticker last price: {ticker.last}, bid: {ticker.bid}, ask: {ticker.ask}")
                
            print("Step 9: Market data request completed")
        except Exception as e:
            print(f"Market data error: {e}")
        
        # Disconnect
        print("Step 10: Disconnecting...")
        ib.disconnect()
        print("Step 11: Successfully disconnected")
        
        # Cancel the timeout timer
        timer.cancel()
        
        return 0
        
    except Exception as e:
        print(f"Connection error: {e}")
        if ib.isConnected():
            ib.disconnect()
            
        # Cancel the timeout timer
        timer.cancel()
        
        return 1

if __name__ == "__main__":
    print("This script tests minimal connectivity to Interactive Brokers TWS")
    print("IMPORTANT: Make sure TWS is running and check for permission dialogs")
    print("The script will timeout after 30 seconds if it gets stuck")
    print("=" * 70)
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1) 