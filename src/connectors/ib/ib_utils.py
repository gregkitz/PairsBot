"""
Utility functions for working with Interactive Brokers (IB) contracts.

This module provides helper functions for converting between IB contract
objects and symbol strings, and other utility functions for working with
the IB API.
"""

import re
from typing import Dict, List, Optional, Tuple, Union, Any
from ib_insync import Contract, Future, Stock, Forex, Index


def symbol_to_contract(symbol: str) -> Contract:
    """
    Convert a symbol string to an IB contract object.
    
    Parameters:
    -----------
    symbol : str
        Symbol string in format 'ES-202206-FUT-GLOBEX'
        or 'AAPL-STK-SMART', or other supported formats
    
    Returns:
    --------
    Contract
        IB contract object
    
    Raises:
    -------
    ValueError
        If the symbol format is invalid or unsupported
    """
    parts = symbol.split('-')
    
    if len(parts) < 2:
        # Try to infer the type from the symbol if not explicitly provided
        if re.match(r'^[A-Z]{6}$', symbol):  # Forex pairs are typically 6 letters
            return Forex(symbol[:3], symbol[3:])
        elif symbol.startswith('^'):  # Common convention for indices
            return Index(symbol[1:], 'SMART', 'USD')
        else:
            # Default to a stock on SMART exchange
            return Stock(symbol, 'SMART', 'USD')
    
    # Handle different contract types based on the format
    if len(parts) >= 3 and parts[-2] == 'FUT':
        # Futures contract: ES-202206-FUT-GLOBEX
        root_symbol = parts[0]
        expiry = parts[1]
        exchange = parts[-1] if len(parts) == 4 else 'GLOBEX'
        
        return Future(root_symbol, expiry, exchange)
    
    elif len(parts) >= 2 and parts[-1] == 'STK':
        # Stock: AAPL-STK or AAPL-STK-SMART
        ticker = parts[0]
        exchange = parts[2] if len(parts) >= 3 else 'SMART'
        currency = parts[3] if len(parts) >= 4 else 'USD'
        
        return Stock(ticker, exchange, currency)
    
    elif len(parts) >= 2 and parts[-1] == 'CASH':
        # Forex: EUR-USD-CASH or EURUSD-CASH
        if len(parts[0]) == 6:  # Combined format like 'EURUSD'
            base = parts[0][:3]
            quote = parts[0][3:]
        else:  # Split format like 'EUR-USD'
            base = parts[0]
            quote = parts[1]
        
        return Forex(base, quote)
    
    elif len(parts) >= 2 and parts[-1] == 'IND':
        # Index: SPX-IND or SPX-IND-CBOE
        ticker = parts[0]
        exchange = parts[2] if len(parts) >= 3 else 'SMART'
        currency = parts[3] if len(parts) >= 4 else 'USD'
        
        return Index(ticker, exchange, currency)
    
    else:
        raise ValueError(f"Unsupported symbol format: {symbol}")


def contract_to_symbol(contract: Contract) -> str:
    """
    Convert an IB contract object to a symbol string.
    
    Parameters:
    -----------
    contract : Contract
        IB contract object
    
    Returns:
    --------
    str
        Symbol string in format 'ES-202206-FUT-GLOBEX'
        or 'AAPL-STK-SMART', or other supported formats
    
    Raises:
    -------
    ValueError
        If the contract type is unsupported
    """
    if isinstance(contract, Future):
        return f"{contract.symbol}-{contract.lastTradeDateOrContractMonth}-FUT-{contract.exchange}"
    
    elif isinstance(contract, Stock):
        return f"{contract.symbol}-STK-{contract.exchange}"
    
    elif isinstance(contract, Forex):
        return f"{contract.symbol}{contract.currency}-CASH"
    
    elif isinstance(contract, Index):
        return f"{contract.symbol}-IND-{contract.exchange}"
    
    else:
        raise ValueError(f"Unsupported contract type: {type(contract)}")


def parse_contract_details(details: Dict) -> Dict[str, Any]:
    """
    Parse contract details returned by IB API into a more usable format.
    
    Parameters:
    -----------
    details : Dict
        Contract details dictionary from IB API
    
    Returns:
    --------
    Dict[str, Any]
        Parsed contract details
    """
    result = {}
    
    # Extract basic information
    contract = details.contract
    result['symbol'] = contract_to_symbol(contract)
    result['conId'] = contract.conId
    result['exchange'] = contract.exchange
    result['currency'] = contract.currency
    
    # Extract contract-specific details
    if isinstance(contract, Future):
        result['contract_type'] = 'Future'
        result['expiry'] = contract.lastTradeDateOrContractMonth
        result['multiplier'] = float(details.multiplier) if details.multiplier else 1.0
        result['min_tick'] = details.minTick
        
    elif isinstance(contract, Stock):
        result['contract_type'] = 'Stock'
        result['industry'] = details.industry
        result['category'] = details.category
        result['min_tick'] = details.minTick
        
    elif isinstance(contract, Forex):
        result['contract_type'] = 'Forex'
        result['base_currency'] = contract.symbol
        result['quote_currency'] = contract.currency
        result['min_tick'] = details.minTick
        
    elif isinstance(contract, Index):
        result['contract_type'] = 'Index'
        
    # Add trading hours and timezone
    if hasattr(details, 'timeZoneId'):
        result['timezone'] = details.timeZoneId
    
    if hasattr(details, 'tradingHours'):
        result['trading_hours'] = details.tradingHours
    
    if hasattr(details, 'liquidHours'):
        result['liquid_hours'] = details.liquidHours
    
    return result


def is_regular_trading_hours(contract: Contract) -> bool:
    """
    Check if the current time is within regular trading hours for the contract.
    
    Parameters:
    -----------
    contract : Contract
        IB contract object
    
    Returns:
    --------
    bool
        True if current time is within regular trading hours, False otherwise
    """
    from datetime import datetime
    import pytz
    
    # This function would need to parse the trading hours from IB
    # and compare with the current time. This is a simplified version.
    # A full implementation would need to query contract details and parse the tradingHours field.
    
    # For now, we'll assume standard US market hours (9:30 AM - 4:00 PM Eastern Time)
    if isinstance(contract, Stock) and contract.exchange in ['SMART', 'NYSE', 'NASDAQ', 'ARCA', 'BATS']:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's between 9:30 AM and 4:00 PM
        market_open = eastern.localize(datetime(now.year, now.month, now.day, 9, 30))
        market_close = eastern.localize(datetime(now.year, now.month, now.day, 16, 0))
        
        return market_open <= now <= market_close
    
    # For futures, forex, and other instruments, we would need more complex logic
    # based on the specific trading hours of each exchange and instrument
    return True  # Default to True for other instruments 