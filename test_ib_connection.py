#!/usr/bin/env python3
"""Test connection to Interactive Brokers."""

from ib_insync import IB
import time

if __name__ == "__main__":
    # Create IB connection
    ib = IB()
    
    try:
        # Try to connect
        print("Attempting to connect to IB...")
        ib.connect('localhost', 7496, clientId=12345)
        
        # Check connection status
        if ib.isConnected():
            print("Successfully connected to IB!")
            
            # Get IB server time to verify connection is working
            ib_time = ib.reqCurrentTime()
            print(f"IB server time: {ib_time}")
            
            # Try to get some basic account info
            try:
                account_values = ib.accountValues()
                print(f"Account values retrieved: {len(account_values)}")
            except Exception as acc_err:
                print(f"Could not get account values: {acc_err}")
                
            # Try to get contract details for a future
            try:
                from ib_insync import Future
                contract = Future(symbol='ES', exchange='CME', currency='USD')
                contracts = ib.reqContractDetails(contract)
                print(f"Found {len(contracts)} contract details for ES")
                if contracts:
                    print(f"First contract: {contracts[0].contract}")
            except Exception as contract_err:
                print(f"Could not get contract details: {contract_err}")
        else:
            print("Failed to connect to IB - no active connection")
    except Exception as e:
        print(f"Error connecting to IB: {e}")
    finally:
        # Disconnect if we're connected
        if ib.isConnected():
            print("Disconnecting from IB")
            ib.disconnect()
        else:
            print("No active connection to disconnect") 