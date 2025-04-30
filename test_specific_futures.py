#!/usr/bin/env python3
"""
Test script for specific futures contracts with different exchanges and specific contract months.
This tests a variety of approaches since the previous tests with generic approaches failed.
"""

import sys
import os
import logging
import time
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Contract, Future, ContFuture, util
    logger.info("Successfully imported ib_insync")
except ImportError:
    logger.error("Failed to import ib_insync. Please install it with: pip install ib_insync")
    sys.exit(1)

def test_futures_with_various_exchanges(ib, symbol, exchanges):
    """Test futures with various exchanges."""
    for exchange in exchanges:
        try:
            logger.info(f"Testing {symbol} with exchange {exchange}")
            
            # Try with both Future and Contract approaches
            # 1. Try with Future
            future = Future(symbol=symbol, exchange=exchange)
            future_details = ib.reqContractDetails(future)
            
            if future_details:
                logger.info(f"SUCCESS with Future for {symbol} on {exchange}!")
                logger.info(f"Found {len(future_details)} contracts")
                for i, detail in enumerate(future_details[:3]):
                    logger.info(f"Contract {i+1}: {detail.contract}")
            else:
                logger.info(f"No contracts found with Future for {symbol} on {exchange}")
            
            # 2. Try with Contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = exchange
            contract.currency = "USD"
            
            contract_details = ib.reqContractDetails(contract)
            
            if contract_details:
                logger.info(f"SUCCESS with Contract for {symbol} on {exchange}!")
                logger.info(f"Found {len(contract_details)} contracts")
                for i, detail in enumerate(contract_details[:3]):
                    logger.info(f"Contract {i+1}: {detail.contract}")
            else:
                logger.info(f"No contracts found with Contract for {symbol} on {exchange}")
                
        except Exception as e:
            logger.error(f"Error testing {symbol} on {exchange}: {e}")

def test_specific_contract_months(ib, symbol, exchange, months):
    """Test specific contract months."""
    for month in months:
        try:
            logger.info(f"Testing {symbol} for month {month} on {exchange}")
            
            # Try with both Future and Contract approaches
            # 1. Try with Future
            future = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=month)
            future_details = ib.reqContractDetails(future)
            
            if future_details:
                logger.info(f"SUCCESS with Future for {symbol} for month {month} on {exchange}!")
                logger.info(f"Found {len(future_details)} contracts")
                for i, detail in enumerate(future_details[:3]):
                    logger.info(f"Contract {i+1}: {detail.contract}")
            else:
                logger.info(f"No contracts found with Future for {symbol} for month {month} on {exchange}")
            
            # 2. Try with Contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = exchange
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = month
            
            contract_details = ib.reqContractDetails(contract)
            
            if contract_details:
                logger.info(f"SUCCESS with Contract for {symbol} for month {month} on {exchange}!")
                logger.info(f"Found {len(contract_details)} contracts")
                for i, detail in enumerate(contract_details[:3]):
                    logger.info(f"Contract {i+1}: {detail.contract}")
            else:
                logger.info(f"No contracts found with Contract for {symbol} for month {month} on {exchange}")
                
        except Exception as e:
            logger.error(f"Error testing {symbol} for month {month} on {exchange}: {e}")

def test_many_different_contract_specifications(ib):
    """Try many different contract specifications that might work."""
    
    # Test different symbols and contract specs
    tests = [
        # ES tests
        {'symbol': 'ES', 'secType': 'FUT', 'exchange': 'CME', 'currency': 'USD'},
        {'symbol': 'ES', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        {'symbol': 'ES', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD', 'lastTradeDateOrContractMonth': '202506'},
        {'symbol': 'ES', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD', 'lastTradeDateOrContractMonth': '202509'},
        {'symbol': 'ES', 'secType': 'FUT', 'exchange': 'CME', 'currency': 'USD', 'localSymbol': 'ESM5'},
        {'symbol': 'ES', 'secType': 'CONTFUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        {'symbol': 'ES', 'secType': 'CONTFUT', 'exchange': 'CME', 'currency': 'USD'},
        
        # NQ tests
        {'symbol': 'NQ', 'secType': 'FUT', 'exchange': 'CME', 'currency': 'USD'},
        {'symbol': 'NQ', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        {'symbol': 'NQ', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD', 'lastTradeDateOrContractMonth': '202506'},
        {'symbol': 'NQ', 'secType': 'CONTFUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        
        # GC tests
        {'symbol': 'GC', 'secType': 'FUT', 'exchange': 'NYMEX', 'currency': 'USD'},
        {'symbol': 'GC', 'secType': 'FUT', 'exchange': 'COMEX', 'currency': 'USD'},
        {'symbol': 'GC', 'secType': 'FUT', 'exchange': 'COMEX', 'currency': 'USD', 'lastTradeDateOrContractMonth': '202506'},
        {'symbol': 'GC', 'secType': 'CONTFUT', 'exchange': 'NYMEX', 'currency': 'USD'},
        {'symbol': 'GC', 'secType': 'CONTFUT', 'exchange': 'COMEX', 'currency': 'USD'},
        
        # SI tests
        {'symbol': 'SI', 'secType': 'FUT', 'exchange': 'NYMEX', 'currency': 'USD'},
        {'symbol': 'SI', 'secType': 'FUT', 'exchange': 'COMEX', 'currency': 'USD'},
        {'symbol': 'SI', 'secType': 'FUT', 'exchange': 'COMEX', 'currency': 'USD', 'lastTradeDateOrContractMonth': '202506'},
        {'symbol': 'SI', 'secType': 'CONTFUT', 'exchange': 'NYMEX', 'currency': 'USD'},
        {'symbol': 'SI', 'secType': 'CONTFUT', 'exchange': 'COMEX', 'currency': 'USD'},
        
        # Microsized futures
        {'symbol': 'MES', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        {'symbol': 'MNQ', 'secType': 'FUT', 'exchange': 'GLOBEX', 'currency': 'USD'},
        {'symbol': 'MGC', 'secType': 'FUT', 'exchange': 'COMEX', 'currency': 'USD'},
    ]
    
    # Test each contract
    successes = []
    for i, test in enumerate(tests):
        try:
            logger.info(f"Test {i+1}/{len(tests)}: Testing {test}")
            
            contract = Contract()
            for key, value in test.items():
                setattr(contract, key, value)
            
            details = ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"SUCCESS! Found {len(details)} contracts for {test}")
                for j, detail in enumerate(details[:2]):  # Show first 2 contracts
                    logger.info(f"Contract {j+1}: {detail.contract}")
                successes.append((i+1, test, len(details)))
            else:
                logger.info(f"No contracts found for {test}")
            
            # Pause between requests to avoid overwhelming IB
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error testing {test}: {e}")
    
    # Show summary of successes
    logger.info("\n=== SUCCESSFUL TESTS SUMMARY ===\n")
    if successes:
        for i, test, count in successes:
            logger.info(f"Test {i}: {test} - Found {count} contracts")
    else:
        logger.info("No successful tests! None of the contracts could be found.")
    
    # Print recommendation
    logger.info("\n=== RECOMMENDATION ===\n")
    if successes:
        best_test = max(successes, key=lambda x: x[2])
        logger.info(f"Best contract specification: Test {best_test[0]}")
        logger.info(f"Contract details: {best_test[1]}")
        logger.info(f"Found {best_test[2]} contracts")
        
        contract_args = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in best_test[1].items()])
        logger.info(f"\nRecommended code:")
        logger.info(f"from ib_insync import Contract")
        logger.info(f"contract = Contract({contract_args})")
        logger.info(f"details = ib.reqContractDetails(contract)")
        logger.info(f"if details:")
        logger.info(f"    front_month = details[0].contract")
    else:
        logger.info("No successful contract specifications found.")
        logger.info("Please check your IB connection and data subscriptions.")

def main():
    """Main test function."""
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description='Test specific futures contracts with IB.')
    parser.add_argument('--host', default='127.0.0.1', help='IB Gateway/TWS host')
    parser.add_argument('--port', type=int, default=7496, help='IB Gateway/TWS port')
    parser.add_argument('--client-id', type=int, default=12345, help='Client ID for IB connection')
    args = parser.parse_args()
    
    # Connect to IB
    ib = IB()
    try:
        logger.info(f"Connecting to IB at {args.host}:{args.port} (client ID: {args.client_id})")
        ib.connect(args.host, args.port, clientId=args.client_id)
        logger.info(f"Connected: {ib.isConnected()}")
        
        if not ib.isConnected():
            logger.error("Failed to connect to IB")
            return
            
        # Try many different contract specifications to see what works
        logger.info("Testing many different contract specifications...")
        test_many_different_contract_specifications(ib)
        
    except Exception as e:
        logger.error(f"Error during tests: {e}")
    finally:
        # Disconnect from IB
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB")

if __name__ == "__main__":
    main() 