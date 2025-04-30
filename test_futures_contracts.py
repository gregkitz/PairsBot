#!/usr/bin/env python3
"""
Test script to determine which method of futures contract specification works best with IB.
This script tries multiple different approaches to see which ones succeed in retrieving contract
details and market data from Interactive Brokers.
"""

import sys
import os
import logging
import time
import json
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'futures_contract_test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Contract, Future, ContFuture, util
    logger.info("Successfully imported ib_insync")
except ImportError:
    logger.error("Failed to import ib_insync. Please install it with: pip install ib_insync")
    sys.exit(1)

# Create output directory
os.makedirs('test_results', exist_ok=True)

class FuturesContractTester:
    """Test different methods of specifying futures contracts with Interactive Brokers."""

    def __init__(self, host='127.0.0.1', port=7496, client_id=12345):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.results = {}
        self.symbols = ["ES", "NQ", "GC", "SI"]  # Common futures
        self.exchange_map = {
            "ES": "GLOBEX",  # E-mini S&P 500
            "NQ": "GLOBEX",  # E-mini NASDAQ-100
            "GC": "NYMEX",   # Gold
            "SI": "NYMEX",   # Silver
            "CL": "NYMEX",   # Crude Oil
            "ZB": "ECBOT",   # 30-Year U.S. Treasury Bond
            "ZN": "ECBOT"    # 10-Year U.S. Treasury Note
        }
        
    def connect(self):
        """Connect to Interactive Brokers."""
        try:
            logger.info(f"Connecting to IB at {self.host}:{self.port} (client ID: {self.client_id})")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"Connected: {self.ib.isConnected()}")
            return self.ib.isConnected()
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")
    
    def run_tests(self):
        """Run all tests for each symbol."""
        if not self.ib.isConnected():
            logger.error("Not connected to IB. Cannot run tests.")
            return
        
        results_by_symbol = {}
        
        # Test each symbol
        for symbol in self.symbols:
            logger.info(f"Running tests for {symbol}")
            symbol_results = {
                "empty_future": self.test_empty_future(symbol),
                "empty_future_with_exchange": self.test_empty_future_with_exchange(symbol),
                "current_year_month": self.test_current_year_month(symbol),
                "next_year_month": self.test_next_year_month(symbol),
                "continuous_future": self.test_continuous_future(symbol),
                "contract_months": self.get_available_contract_months(symbol),
                "simple_contract": self.test_simple_contract(symbol)
            }
            results_by_symbol[symbol] = symbol_results
            
            # Adding a delay between symbols to avoid rate limiting
            time.sleep(2)
        
        self.results = results_by_symbol
        self.save_results()
        return results_by_symbol
    
    def test_empty_future(self, symbol):
        """Test Future contract with only symbol specified."""
        try:
            logger.info(f"Testing empty Future for {symbol} (no exchange specified)")
            contract = Future(symbol=symbol)
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} contracts for {symbol}")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No contracts found for {symbol}")
                return {
                    "success": False,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing empty Future for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_empty_future_with_exchange(self, symbol):
        """Test Future contract with symbol and exchange specified."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            logger.info(f"Testing Future for {symbol} with exchange {exchange}")
            
            contract = Future(symbol=symbol, exchange=exchange)
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} contracts for {symbol} on {exchange}")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No contracts found for {symbol} on {exchange}")
                return {
                    "success": False,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing Future with exchange for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_current_year_month(self, symbol):
        """Test Future contract with current year/month."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            
            # Get current year/month in format YYYYMM
            current_date = datetime.datetime.now()
            year_month = current_date.strftime("%Y%m")
            
            logger.info(f"Testing Future for {symbol} with current year/month {year_month}")
            
            contract = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=year_month)
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} contracts for {symbol} with month {year_month}")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "year_month": year_month,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "year_month": year_month,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No contracts found for {symbol} with month {year_month}")
                return {
                    "success": False,
                    "year_month": year_month,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing Future with current year/month for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_next_year_month(self, symbol):
        """Test Future contract with next year/month."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            
            # Get next month in format YYYYMM
            current_date = datetime.datetime.now()
            next_month = current_date.replace(day=1) + datetime.timedelta(days=32)
            next_month = next_month.replace(day=1)
            year_month = next_month.strftime("%Y%m")
            
            logger.info(f"Testing Future for {symbol} with next year/month {year_month}")
            
            contract = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=year_month)
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} contracts for {symbol} with month {year_month}")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "year_month": year_month,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "year_month": year_month,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No contracts found for {symbol} with month {year_month}")
                return {
                    "success": False,
                    "year_month": year_month,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing Future with next year/month for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_continuous_future(self, symbol):
        """Test ContFuture contract."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            logger.info(f"Testing ContFuture for {symbol} on {exchange}")
            
            contract = ContFuture(symbol=symbol, exchange=exchange)
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} continuous contracts for {symbol}")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No continuous contracts found for {symbol}")
                return {
                    "success": False,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing ContFuture for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_simple_contract(self, symbol):
        """Test with a simple generic Contract object."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            logger.info(f"Testing simple Contract for {symbol} on {exchange}")
            
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = exchange
            contract.currency = "USD"
            
            # Try to get contract details
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Success! Found {len(details)} contracts for {symbol} using simple Contract")
                first_contract = details[0].contract
                logger.info(f"First contract: {first_contract}")
                
                # Try to get market data for the first contract
                try:
                    qualified_contract = self.ib.qualifyContracts(first_contract)[0]
                    ticker = self.ib.reqMktData(qualified_contract)
                    time.sleep(3)  # Wait for market data
                    has_market_data = ticker.last is not None or ticker.bid is not None or ticker.ask is not None
                    logger.info(f"Market data available: {has_market_data}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
                    self.ib.cancelMktData(qualified_contract)
                    
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": has_market_data,
                        "market_data": {"last": ticker.last, "bid": ticker.bid, "ask": ticker.ask}
                    }
                except Exception as e:
                    logger.error(f"Failed to get market data: {e}")
                    return {
                        "success": True,
                        "num_contracts": len(details),
                        "first_contract": str(first_contract),
                        "has_market_data": False,
                        "market_data_error": str(e)
                    }
            else:
                logger.warning(f"No contracts found for {symbol} with simple Contract")
                return {
                    "success": False,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error testing simple Contract for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_contract_months(self, symbol):
        """Get all available contract months for a symbol."""
        try:
            exchange = self.exchange_map.get(symbol, "GLOBEX")
            logger.info(f"Retrieving available contract months for {symbol} on {exchange}")
            
            # Create a simple contract to get all available months
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "FUT"
            contract.exchange = exchange
            contract.currency = "USD"
            
            # Request contract details - this should return all available contracts
            details = self.ib.reqContractDetails(contract)
            
            if details:
                logger.info(f"Found {len(details)} available contract months for {symbol}")
                
                # Extract contract months
                contract_months = []
                for detail in details:
                    c = detail.contract
                    if hasattr(c, 'lastTradeDateOrContractMonth') and c.lastTradeDateOrContractMonth:
                        contract_months.append(c.lastTradeDateOrContractMonth)
                
                # Sort months
                contract_months.sort()
                logger.info(f"Available contract months for {symbol}: {contract_months}")
                
                return {
                    "success": True,
                    "contract_months": contract_months,
                    "num_months": len(contract_months)
                }
            else:
                logger.warning(f"No contracts found for {symbol}")
                return {
                    "success": False,
                    "error": "No contracts found"
                }
                
        except Exception as e:
            logger.error(f"Error getting contract months for {symbol}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_results(self):
        """Save results to JSON file."""
        try:
            filename = os.path.join('test_results', f'futures_contract_test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            
            # Create a custom encoder for handling non-serializable data
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime.datetime, datetime.date)):
                        return obj.isoformat()
                    elif isinstance(obj, (float, int)) and (obj == float('inf') or obj == float('-inf') or obj != obj):  # Check for NaN, Inf
                        return str(obj)
                    try:
                        return super().default(obj)
                    except:
                        return str(obj)  # Convert anything else to string
            
            # Save with pretty-printing
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, cls=CustomEncoder)
            
            logger.info(f"Results saved to {filename}")
            
            # Also create a summary file that's easier to read
            summary_filename = os.path.join('test_results', f'summary_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(summary_filename, 'w') as f:
                f.write("=== FUTURES CONTRACT TEST RESULTS ===\n\n")
                
                for symbol, tests in self.results.items():
                    f.write(f"SYMBOL: {symbol}\n")
                    f.write("="*40 + "\n")
                    
                    # Check which tests worked
                    successful_methods = []
                    
                    for test_name, result in tests.items():
                        if test_name == "contract_months":
                            continue  # Skip this for summary
                            
                        if result.get("success", False):
                            successful_methods.append(test_name)
                            f.write(f"✓ {test_name}: SUCCESS - Found {result.get('num_contracts', 0)} contracts\n")
                            
                            if "has_market_data" in result and result["has_market_data"]:
                                f.write(f"  ✓ Has market data: last={result.get('market_data', {}).get('last')}\n")
                            elif "has_market_data" in result:
                                f.write(f"  ✗ No market data available\n")
                                
                            if "first_contract" in result:
                                f.write(f"  First contract: {result['first_contract']}\n")
                        else:
                            f.write(f"✗ {test_name}: FAILED - {result.get('error', 'Unknown error')}\n")
                    
                    f.write("\n")
                    
                    # Show contract months if available
                    if "contract_months" in tests and tests["contract_months"].get("success", False):
                        months = tests["contract_months"].get("contract_months", [])
                        f.write(f"Available contract months ({len(months)}):\n")
                        f.write(", ".join(months) + "\n")
                    
                    f.write("\nRECOMMENDED METHOD(S):\n")
                    if successful_methods:
                        for method in successful_methods:
                            f.write(f"- {method}\n")
                    else:
                        f.write("NONE - All methods failed\n")
                    
                    f.write("\n\n")
                
                # Overall recommendation
                f.write("=== OVERALL RECOMMENDATION ===\n")
                f.write("Based on the test results, here are the recommended methods to use:\n\n")
                
                all_successes = {}
                for symbol, tests in self.results.items():
                    for test_name, result in tests.items():
                        if test_name == "contract_months":
                            continue
                        if result.get("success", False):
                            all_successes[test_name] = all_successes.get(test_name, 0) + 1
                
                if all_successes:
                    best_method = max(all_successes.items(), key=lambda x: x[1])
                    f.write(f"BEST METHOD: {best_method[0]} (worked for {best_method[1]}/{len(self.symbols)} symbols)\n\n")
                    
                    f.write("Success rates for each method:\n")
                    for method, count in sorted(all_successes.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- {method}: {count}/{len(self.symbols)} symbols ({count/len(self.symbols)*100:.1f}%)\n")
                else:
                    f.write("No successful methods found. Contact Interactive Brokers support.\n")
            
            logger.info(f"Summary saved to {summary_filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description='Test different methods of specifying futures contracts with IB.')
    parser.add_argument('--host', default='127.0.0.1', help='IB Gateway/TWS host')
    parser.add_argument('--port', type=int, default=7496, help='IB Gateway/TWS port')
    parser.add_argument('--client-id', type=int, default=12345, help='Client ID for IB connection')
    args = parser.parse_args()
    
    # Run the tests
    tester = FuturesContractTester(host=args.host, port=args.port, client_id=args.client_id)
    
    try:
        if tester.connect():
            results = tester.run_tests()
            logger.info("All tests completed.")
            
            # Print the most successful method
            all_successes = {}
            for symbol, tests in results.items():
                for test_name, result in tests.items():
                    if test_name == "contract_months":
                        continue
                    if result.get("success", False):
                        all_successes[test_name] = all_successes.get(test_name, 0) + 1
            
            if all_successes:
                best_method = max(all_successes.items(), key=lambda x: x[1])
                logger.info(f"RECOMMENDED METHOD: {best_method[0]} (worked for {best_method[1]}/{len(tester.symbols)} symbols)")
                
                # Print full results to console
                logger.info("\nSUCCESS RATES:")
                for method, count in sorted(all_successes.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"- {method}: {count}/{len(tester.symbols)} symbols ({count/len(tester.symbols)*100:.1f}%)")
            else:
                logger.info("No successful methods found. Check your IB connection and data subscriptions.")
        else:
            logger.error("Failed to connect to IB. Make sure IB Gateway/TWS is running with API connections enabled.")
    finally:
        tester.disconnect()

if __name__ == "__main__":
    main() 