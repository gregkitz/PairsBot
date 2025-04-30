#!/usr/bin/env python
"""
Basic Interactive Brokers connection test with multiple attempts.

This script tries different client IDs and configurations to establish a connection.
"""

import sys
import time
import threading
from ib_insync import IB, util

print("=" * 80)
print("Interactive Brokers TWS Connection Test - Multiple Attempts")
print("This script will try different configurations to connect to TWS")
print("=" * 80)

# Global timeout flag
timed_out = False

def timeout_function():
    """Function to run when timeout occurs."""
    global timed_out
    timed_out = True
    print("\n\nTIMEOUT: Script execution timed out after 60 seconds")
    print("Check if TWS is showing a permission dialog that might be hidden")
    sys.exit(1)

# Start timeout timer
timer = threading.Timer(60, timeout_function)
timer.daemon = True
timer.start()

# Different client IDs to try
client_ids = [1, 2, 3, 999, 1000, 1001]

# Different port numbers to try
ports = [7497, 7496]

# Try different combinations of client IDs and ports
for port in ports:
    print(f"\n==== Testing port {port} ====")
    for client_id in client_ids:
        try:
            # Create IB connection object
            ib = IB()
            
            # Try to connect 
            print(f"\nAttempt: port={port}, client_id={client_id}")
            ib.connect('localhost', port, clientId=client_id, timeout=5, readonly=True)
            
            # If we get here, connection was successful
            if ib.isConnected():
                print(f"SUCCESS! Connected with port={port}, client_id={client_id}")
                
                # Try to get server time
                try:
                    current_time = ib.reqCurrentTime()
                    print(f"Server time: {current_time}")
                except Exception as e:
                    print(f"Failed to get server time: {e}")
                
                # Success, no need to try more combinations
                print("\nDisconnecting...")
                ib.disconnect()
                timer.cancel()
                
                print("\n==== CONNECTION SUMMARY ====")
                print(f"Successfully connected with:")
                print(f"- Host: localhost")
                print(f"- Port: {port}")
                print(f"- Client ID: {client_id}")
                print(f"- Read-only: True")
                print("\nUse these settings in your tests!")
                
                sys.exit(0)
            
        except Exception as e:
            # Connection failed, try next combination
            print(f"Failed: {str(e)}")
            if 'ib' in locals() and ib.isConnected():
                ib.disconnect()
            continue

# If we get here, all combinations failed
print("\n==== SUMMARY ====")
print("All connection attempts failed!")
print("\nPlease check the following:")
print("1. TWS is running and you're logged in")
print("2. API access is enabled in TWS (Edit > Global Configuration > API > Settings)")
print("3. The correct port is set in TWS (7497 for Paper Trading)")
print("4. No trusted IP restrictions are in place")
print("5. No other applications are using the same client ID")
print("6. There are no hidden permission dialogs")

timer.cancel()
sys.exit(1) 