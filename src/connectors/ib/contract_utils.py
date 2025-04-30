"""
Interactive Brokers Contract Utilities

This module provides utility functions for creating and managing
Interactive Brokers contract objects for various instrument types.
"""

import logging
import datetime
from typing import Optional, Dict, Tuple, Union, List
import re
import time
import threading
import concurrent.futures

# Import IB API classes if available
try:
    from ib_insync import Contract, Future, ContFuture, Option, Stock, Forex, util
    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False
    # Define stub classes for type hints that won't create runtime errors
    class Contract:
        pass
    class Future:
        pass
    class ContFuture:
        pass

logger = logging.getLogger(__name__)

# Mapping of futures symbols to their exchanges
FUTURES_EXCHANGE_MAP = {
    # US Indices
    "ES": "CME",     # E-mini S&P 500
    "NQ": "CME",     # E-mini NASDAQ-100
    "YM": "CBOT",    # E-mini Dow
    "RTY": "CME",    # E-mini Russell 2000
    "MES": "CME",    # Micro E-mini S&P 500
    "MNQ": "CME",    # Micro E-mini NASDAQ-100
    "MYM": "CBOT",   # Micro E-mini Dow
    "M2K": "CME",    # Micro E-mini Russell 2000
    
    # Metals
    "GC": "COMEX",   # Gold
    "SI": "COMEX",   # Silver
    "HG": "COMEX",   # Copper
    "PL": "NYMEX",   # Platinum
    "PA": "NYMEX",   # Palladium
    "MGC": "COMEX",  # Micro Gold
    "SIL": "COMEX",  # Micro Silver
    
    # Energy
    "CL": "NYMEX",   # Crude Oil
    "NG": "NYMEX",   # Natural Gas
    "RB": "NYMEX",   # Gasoline
    "HO": "NYMEX",   # Heating Oil
    "MCL": "NYMEX",  # Micro Crude Oil
    
    # Interest Rates
    "ZB": "CBOT",    # 30-Year T-Bond
    "ZN": "CBOT",    # 10-Year T-Note
    "ZF": "CBOT",    # 5-Year T-Note
    "ZT": "CBOT",    # 2-Year T-Note
    
    # Currencies
    "6E": "CME",     # Euro FX
    "6J": "CME",     # Japanese Yen
    "6B": "CME",     # British Pound
    "6A": "CME",     # Australian Dollar
    "6C": "CME",     # Canadian Dollar
    
    # Agricultural
    "ZC": "CBOT",    # Corn
    "ZW": "CBOT",    # Wheat
    "ZS": "CBOT",    # Soybeans
    "ZM": "CBOT",    # Soybean Meal
    "ZL": "CBOT",    # Soybean Oil
    
    # European Futures
    "FDAX": "EUREX", # DAX
    "FESX": "EUREX", # Euro Stoxx 50
    "FGBL": "EUREX", # Euro Bund
}

def get_exchange_for_symbol(symbol: str) -> str:
    """
    Returns the appropriate exchange for a given futures symbol.
    
    Args:
        symbol (str): The futures symbol
        
    Returns:
        str: The exchange for the symbol, or "SMART" if not found
    """
    # Strip off any prefixes that might be in the symbol (like 'M' for micro)
    base_symbol = re.sub(r'^[A-Z]', '', symbol) if re.match(r'^[A-Z][A-Z]+$', symbol) else symbol
    
    # Check if the symbol is in our map
    if symbol in FUTURES_EXCHANGE_MAP:
        return FUTURES_EXCHANGE_MAP[symbol]
    elif base_symbol in FUTURES_EXCHANGE_MAP:
        return FUTURES_EXCHANGE_MAP[base_symbol]
    
    # Default to SMART routing if not found
    logger.warning(f"Exchange not found for symbol {symbol}, defaulting to SMART")
    return "SMART"

def create_futures_contract(
    symbol: str, 
    expiry: Optional[str] = None, 
    exchange: Optional[str] = None, 
    currency: str = "USD"
) -> Contract:
    """
    Create a futures contract with the specified attributes.
    
    Args:
        symbol (str): The futures symbol
        expiry (str, optional): Expiry date in YYYYMM or YYYYMMDD format
        exchange (str, optional): Exchange name (if None, determined from symbol)
        currency (str, optional): Currency code (default: USD)
        
    Returns:
        Contract: An IB contract object
    """
    if not HAS_IB_INSYNC:
        logger.error("ib_insync is not installed. Cannot create futures contract.")
        return None
    
    # Determine exchange if not provided
    if exchange is None:
        exchange = get_exchange_for_symbol(symbol)
    
    # Create the contract
    contract = Future(symbol=symbol, exchange=exchange, currency=currency)
    
    # Add expiry if provided
    if expiry:
        # Format expiry appropriately
        if len(expiry) == 6:  # YYYYMM format
            contract.lastTradeDateOrContractMonth = expiry
        elif len(expiry) == 8:  # YYYYMMDD format
            contract.lastTradeDateOrContractMonth = expiry
        else:
            # Try to parse and format the date
            try:
                dt = datetime.datetime.strptime(expiry, "%Y-%m-%d")
                contract.lastTradeDateOrContractMonth = dt.strftime("%Y%m%d")
            except ValueError:
                logger.warning(f"Invalid expiry format: {expiry}, using as-is")
                contract.lastTradeDateOrContractMonth = expiry
    
    return contract

def create_continuous_futures_contract(
    symbol: str, 
    exchange: Optional[str] = None, 
    currency: str = "USD"
) -> Contract:
    """
    Create a continuous futures contract for the specified symbol.
    
    Args:
        symbol (str): The futures symbol
        exchange (str, optional): Exchange name (if None, determined from symbol)
        currency (str, optional): Currency code (default: USD)
        
    Returns:
        Contract: An IB continuous futures contract
    """
    if not HAS_IB_INSYNC:
        logger.error("ib_insync is not installed. Cannot create continuous futures contract.")
        return None
    
    # Determine exchange if not provided
    if exchange is None:
        exchange = get_exchange_for_symbol(symbol)
    
    # Create the continuous contract
    contract = ContFuture(symbol=symbol, exchange=exchange, currency=currency)
    
    return contract

def contract_search_with_timeout(
    ib_instance, 
    symbol: str, 
    sec_type: str = "FUT",
    exchange: Optional[str] = None,
    currency: str = "USD",
    timeout: int = 60
) -> List[Contract]:
    """
    Search for contracts with a timeout to prevent hanging.
    
    Args:
        ib_instance: The IB client instance
        symbol (str): The contract symbol
        sec_type (str): Security type (default: "FUT")
        exchange (str, optional): Exchange name
        currency (str): Currency code (default: USD)
        timeout (int): Timeout in seconds (default: 60)
        
    Returns:
        List[Contract]: List of matching contracts or empty list on timeout
    """
    if not HAS_IB_INSYNC:
        logger.error("ib_insync is not installed. Cannot search for contracts.")
        return []
    
    # Determine exchange if not provided and it's a futures contract
    if exchange is None and sec_type == "FUT":
        exchange = get_exchange_for_symbol(symbol)
    
    # Create a basic contract for the search
    contract = Contract(symbol=symbol, secType=sec_type, 
                      exchange=exchange if exchange else "SMART", 
                      currency=currency)
    
    # Set up for threaded execution with timeout
    result = []
    search_completed = threading.Event()
    
    def target():
        try:
            nonlocal result
            result = ib_instance.reqContractDetails(contract)
            search_completed.set()
        except Exception as e:
            logger.error(f"Error in contract search: {e}")
            search_completed.set()
    
    # Start the search in a thread
    thread = threading.Thread(target=target)
    thread.start()
    
    # Wait with timeout
    is_completed = search_completed.wait(timeout=timeout)
    
    if not is_completed:
        logger.warning(f"Contract search timed out after {timeout} seconds for {symbol}")
        return []
    
    return result

def reqContractDetails_with_timeout(
    ib_instance, 
    contract: Contract, 
    timeout: int = 60
) -> List:
    """
    Request contract details with a timeout to prevent hanging.
    
    Args:
        ib_instance: The IB client instance
        contract (Contract): The contract to get details for
        timeout (int): Timeout in seconds (default: 60)
        
    Returns:
        List: Contract details or empty list on timeout
    """
    if not HAS_IB_INSYNC:
        logger.error("ib_insync is not installed. Cannot request contract details.")
        return []
    
    # Set up for threaded execution with timeout
    result = []
    request_completed = threading.Event()
    
    def target():
        try:
            nonlocal result
            result = ib_instance.reqContractDetails(contract)
            request_completed.set()
        except Exception as e:
            logger.error(f"Error in reqContractDetails: {e}")
            request_completed.set()
    
    # Start the request in a thread
    thread = threading.Thread(target=target)
    thread.start()
    
    # Wait with timeout
    is_completed = request_completed.wait(timeout=timeout)
    
    if not is_completed:
        logger.warning(f"reqContractDetails timed out after {timeout} seconds for {contract.symbol}")
        return []
    
    return result

def parse_contract_string(contract_str: str) -> Contract:
    """
    Parse a contract string in the format "SYMBOL-EXPIRY-TYPE-EXCHANGE".
    
    Args:
        contract_str (str): Contract string in the format "SYMBOL-EXPIRY-TYPE-EXCHANGE"
        
    Returns:
        Contract: IB contract object
    """
    if not HAS_IB_INSYNC:
        logger.error("ib_insync is not installed. Cannot parse contract string.")
        return None
    
    try:
        # Parse the contract string
        parts = contract_str.split("-")
        
        if len(parts) >= 4:
            symbol = parts[0]
            expiry = parts[1]
            sec_type = parts[2]
            exchange = parts[3]
            currency = parts[4] if len(parts) > 4 else "USD"
            
            if sec_type == "FUT":
                # Create a futures contract
                contract = create_futures_contract(
                    symbol=symbol,
                    expiry=expiry,
                    exchange=exchange,
                    currency=currency
                )
            elif sec_type == "STK":
                # Create a stock contract
                contract = Stock(symbol=symbol, exchange=exchange, currency=currency)
            else:
                # Create a generic contract
                contract = Contract(
                    symbol=symbol,
                    secType=sec_type,
                    lastTradeDateOrContractMonth=expiry,
                    exchange=exchange,
                    currency=currency
                )
            
            return contract
        else:
            logger.error(f"Invalid contract string format: {contract_str}")
            return None
    except Exception as e:
        logger.error(f"Error parsing contract string '{contract_str}': {e}")
        return None 