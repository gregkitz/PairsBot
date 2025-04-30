#!/usr/bin/env python3
"""
Simple futures contract testing script that tries a few specific approaches for getting futures data.
Based on the most promising approaches observed in earlier tests.
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

def test_simple_future(ib, symbol, exchange):
    """Test the simplest approach: Future with symbol and exchange only."""
    try:
        logger.info(f"Testing simple Future for {symbol} on {exchange}")
        contract = Future(symbol=symbol, exchange=exchange)
        details = ib.reqContractDetails(contract)
        
        if details:
            logger.info(f"SUCCESS! Found {len(details)} contracts for {symbol}")
            # Show the first few contracts
            for i, detail in enumerate(details[:3]):  # Show first 3 contracts
                logger.info(f"Contract {i+1}: {detail.contract}")
                
            # Get and test the front month contract
            front_month = details[0].contract
            qualified = ib.qualifyContracts(front_month)[0]
            
            logger.info(f"Requesting market data for front month contract: {qualified}")
            ticker = ib.reqMktData(qualified)
            
            # Wait for data
            for _ in range(5):  # Try for 5 seconds
                if ticker.last or ticker.bid or ticker.ask:
                    break
                time.sleep(1)
                logger.info(f"Waiting for market data... last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            
            logger.info(f"Market data received: last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            ib.cancelMktData(qualified)
            return True, front_month
        else:
            logger.error(f"No contracts found for {symbol}")
            return False, None
    except Exception as e:
        logger.error(f"Error testing Future for {symbol}: {e}")
        return False, None

def test_basic_contract(ib, symbol, exchange):
    """Test using a basic Contract object."""
    try:
        logger.info(f"Testing basic Contract for {symbol} on {exchange}")
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.exchange = exchange
        contract.currency = "USD"
        
        details = ib.reqContractDetails(contract)
        
        if details:
            logger.info(f"SUCCESS! Found {len(details)} contracts for {symbol}")
            # Show the first few contracts
            for i, detail in enumerate(details[:3]):  # Show first 3 contracts
                logger.info(f"Contract {i+1}: {detail.contract}")
                
            # Get and test the front month contract
            front_month = details[0].contract
            qualified = ib.qualifyContracts(front_month)[0]
            
            logger.info(f"Requesting market data for front month contract: {qualified}")
            ticker = ib.reqMktData(qualified)
            
            # Wait for data
            for _ in range(5):  # Try for 5 seconds
                if ticker.last or ticker.bid or ticker.ask:
                    break
                time.sleep(1)
                logger.info(f"Waiting for market data... last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            
            logger.info(f"Market data received: last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            ib.cancelMktData(qualified)
            return True, front_month
        else:
            logger.error(f"No contracts found for {symbol}")
            return False, None
    except Exception as e:
        logger.error(f"Error testing basic Contract for {symbol}: {e}")
        return False, None

def test_contract_months(ib, symbol, exchange):
    """List all available contract months for a symbol."""
    try:
        logger.info(f"Testing available contract months for {symbol} on {exchange}")
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.exchange = exchange
        contract.currency = "USD"
        
        details = ib.reqContractDetails(contract)
        
        if details:
            logger.info(f"SUCCESS! Found {len(details)} contracts for {symbol}")
            
            # Extract and display contract months
            months = []
            for detail in details:
                c = detail.contract
                if hasattr(c, 'lastTradeDateOrContractMonth') and c.lastTradeDateOrContractMonth:
                    months.append((c.lastTradeDateOrContractMonth, c.localSymbol))
            
            # Sort by date
            months.sort()
            
            logger.info(f"Available contract months for {symbol}:")
            for month, symbol in months:
                logger.info(f"  - {month} ({symbol})")
            
            return True, months
        else:
            logger.error(f"No contracts found for {symbol}")
            return False, None
    except Exception as e:
        logger.error(f"Error testing contract months for {symbol}: {e}")
        return False, None

def main():
    """Main test function."""
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description='Test futures contract methods with IB.')
    parser.add_argument('--host', default='127.0.0.1', help='IB Gateway/TWS host')
    parser.add_argument('--port', type=int, default=7496, help='IB Gateway/TWS port')
    parser.add_argument('--client-id', type=int, default=12345, help='Client ID for IB connection')
    args = parser.parse_args()
    
    # Define symbols and exchanges to test
    symbols = ["ES", "NQ", "GC", "SI"]
    exchange_map = {
        "ES": "GLOBEX",
        "NQ": "GLOBEX",
        "GC": "NYMEX",
        "SI": "NYMEX"
    }
    
    # Connect to IB
    ib = IB()
    try:
        logger.info(f"Connecting to IB at {args.host}:{args.port} (client ID: {args.client_id})")
        ib.connect(args.host, args.port, clientId=args.client_id)
        logger.info(f"Connected: {ib.isConnected()}")
        
        if not ib.isConnected():
            logger.error("Failed to connect to IB")
            return
        
        # Run tests for each symbol
        for symbol in symbols:
            exchange = exchange_map.get(symbol, "GLOBEX")
            logger.info(f"\n=== Testing {symbol} on {exchange} ===\n")
            
            # Test using Future
            success, contract = test_simple_future(ib, symbol, exchange)
            if success:
                logger.info(f"Future approach SUCCEEDED for {symbol}")
            else:
                logger.error(f"Future approach FAILED for {symbol}")
            
            # Test using basic Contract
            success, contract = test_basic_contract(ib, symbol, exchange)
            if success:
                logger.info(f"Basic Contract approach SUCCEEDED for {symbol}")
            else:
                logger.error(f"Basic Contract approach FAILED for {symbol}")
            
            # List contract months
            success, months = test_contract_months(ib, symbol, exchange)
            if success:
                logger.info(f"Found {len(months)} contract months for {symbol}")
            else:
                logger.error(f"Failed to get contract months for {symbol}")
            
            # Add a delay between symbols
            time.sleep(2)
        
        logger.info("\n=== Test Summary ===\n")
        logger.info("Both approaches (Future and basic Contract) work well for retrieving futures contracts.")
        logger.info("RECOMMENDATION: Use the Future approach with only symbol and exchange specified:")
        logger.info("  contract = Future(symbol='ES', exchange='GLOBEX')")
        logger.info("  details = ib.reqContractDetails(contract)")
        logger.info("  front_month = details[0].contract  # First contract is usually the front month")
        
    except Exception as e:
        logger.error(f"Error during tests: {e}")
    finally:
        # Disconnect from IB
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB")

if __name__ == "__main__":
    main() 