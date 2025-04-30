#!/usr/bin/env python
"""
Test script for the fixed IB connector.

This script tests the fixed IB connector with a simple connection.
"""

import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test the fixed IB connector."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add the project root to the Python path
    sys.path.insert(0, str(project_root))
    
    # Import the fixed connector
    from src.connectors.ib.fixed_connector import FixedIBConnector
    
    # Create the connector
    connector = FixedIBConnector(
        host="localhost",
        port=7497,
        client_id=1,
        timeout=10,
        read_only=True
    )
    
    try:
        # Connect to TWS
        print("\n==== Testing connection to TWS ====")
        connected = connector.connect()
        if not connected:
            print("\nConnection failed. Attempting to trigger permission dialogs...")
            print("IMPORTANT: Watch for dialog boxes in TWS and approve them")
            print("After approving, you'll need to run this script again")
            
            # Try to trigger permissions even though we're not connected
            try:
                connector.ib.connect('localhost', 7497, clientId=2, timeout=5, readonly=True)
                connector.trigger_api_permissions()
            except Exception as e:
                print(f"Could not trigger permissions: {e}")
                
            return 1
        
        print("Successfully connected to TWS")
        
        # Try to trigger permission dialogs if needed
        print("\n==== Triggering permission dialogs ====")
        print("IMPORTANT: Watch for dialog boxes in TWS and approve them")
        connector.trigger_api_permissions()
        
        # Get server time
        print("\n==== Testing server time ====")
        server_time = connector.get_server_time()
        print(f"Server time: {server_time}")
        
        if server_time:
            print("\nYOUR CONNECTION IS WORKING! You can now run the integration tests")
        else:
            print("\nCould not get server time. Permissions might not be fully granted.")
            print("Check TWS for permission dialogs and run this script again.")
        
        return 0 if server_time else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        # Disconnect from TWS
        if 'connector' in locals() and connector.is_connected():
            connector.disconnect()

if __name__ == "__main__":
    print("=" * 70)
    print("Testing fixed IB connector")
    print("Make sure TWS is running on localhost:7497")
    print("=" * 70)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1) 