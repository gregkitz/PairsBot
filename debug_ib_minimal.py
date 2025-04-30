#!/usr/bin/env python
"""
Minimal debug script for testing IB TWS connectivity without account operations.

This script tests only the absolute basics of TWS connectivity with minimal operations.
"""

import sys
import time
import threading
from ib_insync import IB, util

print("=" * 80)
print("Minimal Interactive Brokers TWS Connection Test")
print("This will ONLY establish a connection with no additional operations")
print("=" * 80)

# Global timeout flag
timed_out = False

def timeout_function():
    """Function to run when timeout occurs."""
    global timed_out
    timed_out = True
    print("\n\nTIMEOUT: Script execution timed out after 15 seconds")
    print("Check if TWS is showing a permission dialog that might be hidden")
    sys.exit(1)

# Start timeout timer
timer = threading.Timer(15, timeout_function)
timer.daemon = True
timer.start()

try:
    # Create IB connection object
    ib = IB()
    
    # Try to connect with minimal timeout
    print("\nStep 1: Connecting to TWS (localhost:7497, client ID: 123)...")
    ib.connect('localhost', 7497, clientId=123, timeout=5)
    
    # Check connection
    if ib.isConnected():
        print("Step 2: Successfully connected to TWS!")
        print("Step 3: Waiting 3 seconds to see if connection remains stable...")
        
        # Wait a bit to see if connection remains stable
        for i in range(3):
            if not ib.isConnected():
                print(f"ERROR: Lost connection after {i} seconds")
                sys.exit(1)
            time.sleep(1)
            print(f"Connection still active ({i+1}/3 seconds)")
        
        print("Step 4: Connection remained stable for 3 seconds")
        
        # Try to get time from TWS (minimal operation)
        print("Step 5: Getting current server time from TWS...")
        try:
            current_time = ib.reqCurrentTime()
            print(f"Step 6: Server time: {current_time}")
            print("\nSUCCESS: All basic connectivity tests passed!")
        except Exception as e:
            print(f"ERROR: Failed to get server time: {e}")
            sys.exit(1)
    else:
        print("ERROR: Failed to establish connection")
        sys.exit(1)
    
    # Disconnect
    print("\nStep 7: Disconnecting...")
    ib.disconnect()
    print("Step 8: Successfully disconnected")
    
    # Cancel the timeout timer
    timer.cancel()
    
    print("\nAll tests completed successfully!")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR: {e}")
    if 'ib' in locals() and ib.isConnected():
        ib.disconnect()
    sys.exit(1) 