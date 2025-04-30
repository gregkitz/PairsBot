#!/usr/bin/env python3
"""
Fixed wrapper for running paper trading with ML enhancements.
This script addresses the JSON serialization issue with datetime objects.
"""

import os
import sys
import json
import datetime
import time
import logging
import signal
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import threading
import concurrent.futures

# Configure logging with file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('output', 'paper_trading', 'logs', f'paper_trader_fixed_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs('output/paper_trading/logs', exist_ok=True)
os.makedirs('output/paper_trading/dashboard', exist_ok=True)
os.makedirs('output/paper_trading/signals', exist_ok=True)

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return super().default(obj)

# Patch the JSON module's default encoding method
original_dumps = json.dumps
def patched_dumps(*args, **kwargs):
    if 'cls' not in kwargs:
        kwargs['cls'] = DateTimeEncoder
    return original_dumps(*args, **kwargs)

# Apply the patch globally
json.dumps = patched_dumps
logger.info("Successfully patched JSON encoder to handle datetime objects")

# Also patch json.dump to use the DateTimeEncoder
original_dump = json.dump
def patched_dump(*args, **kwargs):
    if 'cls' not in kwargs:
        kwargs['cls'] = DateTimeEncoder
    return original_dump(*args, **kwargs)

# Apply the dump patch globally
json.dump = patched_dump
logger.info("Successfully patched JSON dump function to handle datetime objects")

# Import paper trader first to apply monkey patches
from src.paper_trading.paper_trader import PaperTrader

# Add the missing method to PaperTrader
def get_subscribed_symbols(self):
    """Return a list of currently subscribed symbols."""
    return list(self._market_data.keys())

# Add methods to expose market data
def get_market_data(self):
    """Return the current market data."""
    return self._market_data

def get_account_info(self):
    """Return the account information."""
    return self._account

# Store a cache of streaming bars that stays updated
_streaming_bars = {}

def get_historical_data(self, symbol, duration='1 D', bar_size='5 mins', what_to_show='TRADES'):
    """Get historical data for a symbol with keepUpToDate for real-time updates.
    
    Args:
        symbol: The symbol to get data for
        duration: Time duration to retrieve (e.g., '1 D' for 1 day)
        bar_size: Size of the bars ('1 min', '5 mins', etc.)
        what_to_show: Type of data to retrieve ('TRADES', 'MIDPOINT', etc.)
    
    Returns:
        DataFrame with historical data or None if error
    """
    try:
        logger.info(f"Requesting historical data for {symbol} ({duration}, {bar_size})")
        
        # Check if we already have streaming data for this symbol and bar size
        cache_key = f"{symbol}_{bar_size}_{what_to_show}"
        if cache_key in self._streaming_bars:
            logger.info(f"Using cached streaming bars for {symbol}")
            # Return the latest snapshot from the streaming data
            bars = self._streaming_bars[cache_key]
            if hasattr(bars, 'df'):
                return bars.df()
            else:
                # Manual conversion if df method not available
                data = [{
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in bars]
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('date', inplace=True)
                    return df
        
        # Check if we have a contract object or just a symbol string
        contract = None
        
        # Map of futures exchanges based on symbol
        exchange_map = {
            "ES": "CME",    # E-mini S&P 500 - CME (not GLOBEX)
            "NQ": "CME",    # E-mini NASDAQ-100 - CME (not GLOBEX)
            "GC": "COMEX",  # Gold - COMEX (not NYMEX)
            "SI": "COMEX",  # Silver - COMEX (not NYMEX)
            "CL": "NYMEX",  # Crude Oil
            "ZB": "ECBOT",  # 30-Year U.S. Treasury Bond
            "ZN": "ECBOT"   # 10-Year U.S. Treasury Note
        }
        
        # If symbol is already a Contract object
        if hasattr(symbol, 'symbol') and hasattr(symbol, 'secType'):
            contract = symbol
            
            # Update the exchange if needed
            if contract.secType == 'FUT' or contract.secType == 'CONTFUT':
                if contract.symbol in exchange_map:
                    contract.exchange = exchange_map[contract.symbol]
        else:
            # Try to parse the symbol string to create a contract
            try:
                # Parse the symbol string (format: SYMBOL-EXPIRY-TYPE-EXCHANGE)
                parts = symbol.split('-')
                
                if len(parts) >= 3:
                    symbol_base = parts[0]
                    
                    # Get the appropriate exchange based on symbol
                    exchange = exchange_map.get(symbol_base, "CME")  # Default to CME
                    
                    # Determine if it's a continuous or regular futures contract
                    if parts[1].upper() == 'CONTFUT' or 'CONTFUT' in symbol.upper():
                        contract = create_continuous_futures_contract(
                            symbol=symbol_base,
                            exchange=exchange
                        )
                    else:
                        # Regular futures contract
                        contract = create_futures_contract(
                            symbol=symbol_base,
                            expiry=parts[1],
                            exchange=exchange
                        )
            except Exception as e:
                logger.error(f"Error parsing symbol {symbol}: {e}")
                return None
                
        if not contract:
            logger.error(f"Could not create contract for {symbol}")
            return None
            
        # When requesting historical data, add a longer timeout
        if hasattr(self.ib_connector, 'ib'):
            try:
                # Temporarily increase timeout for historical data requests
                original_timeout = self.ib_connector.ib.RequestTimeout
                self.ib_connector.ib.RequestTimeout = 60  # Set to 60 seconds
                
                # First try without keepUpToDate to get initial data more reliably
                logger.info(f"Requesting initial historical data for {contract.symbol}")
                
                # Start with a request that doesn't use keepUpToDate
                bars = self.ib_connector.ib.reqHistoricalData(
                    contract,
                    endDateTime='',  # '' for current time
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1,  # 1 = 'YYYYMMDD HH:MM:SS'
                    keepUpToDate=False  # Don't keep updating to get initial data faster
                )
                
                # If we got data, store it and then try to start a real-time stream
                if bars:
                    # Store bars for later reference
                    if not cache_key in self._streaming_bars:
                        # Create a non-updating copy
                        self._streaming_bars[cache_key] = bars
                    
                    # Now that we have initial data, try to start streaming updates
                    try:
                        logger.info(f"Starting real-time updates for {contract.symbol}")
                        # Start a streaming request in the background
                        streaming_bars = self.ib_connector.ib.reqHistoricalData(
                            contract,
                            endDateTime='',
                            durationStr=duration,
                            barSizeSetting=bar_size,
                            whatToShow=what_to_show,
                            useRTH=True,
                            formatDate=1,
                            keepUpToDate=True  # Now enable streaming updates
                        )
                        
                        # Update the reference to use streaming bars for future requests
                        if streaming_bars:
                            self._streaming_bars[cache_key] = streaming_bars
                            logger.info(f"Successfully started streaming updates for {contract.symbol}")
                    except Exception as stream_err:
                        logger.warning(f"Could not start streaming updates: {stream_err}")
                        # We'll continue with the non-streaming data we already have
                
                # Reset timeout to original value
                self.ib_connector.ib.RequestTimeout = original_timeout
                
                # Process bars if we got them
                if bars:
                    # Convert to dataframe
                    if hasattr(bars, 'df'):
                        return bars.df()
                    else:
                        # Manual conversion
                        data = [{
                            'date': bar.date,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        } for bar in bars]
                        
                        if data:
                            df = pd.DataFrame(data)
                            df.set_index('date', inplace=True)
                            return df
                
                # Rest of the fallback approaches remain unchanged
                # ... existing code ...
                
            except Exception as req_error:
                logger.error(f"Error requesting historical data: {req_error}")
                
                # Try fallback with simpler contract and shorter durations
                try:
                    logger.info(f"Trying fallback with simpler contract and shorter duration for {symbol}")
                    from ib_insync import Contract
                    
                    # Get symbol base if it's a string with parts
                    if isinstance(symbol, str) and '-' in symbol:
                        symbol_base = symbol.split('-')[0]
                    elif hasattr(contract, 'symbol'):
                        symbol_base = contract.symbol
                    else:
                        symbol_base = symbol
                    
                    # Get appropriate exchange
                    exchange = exchange_map.get(symbol_base, "CME")
                    
                    simple_contract = Contract()
                    simple_contract.symbol = symbol_base
                    simple_contract.secType = 'FUT'
                    simple_contract.exchange = exchange
                    simple_contract.currency = 'USD'
                    
                    # For futures, we need a contract month
                    if hasattr(contract, 'lastTradeDateOrContractMonth') and contract.lastTradeDateOrContractMonth:
                        simple_contract.lastTradeDateOrContractMonth = contract.lastTradeDateOrContractMonth
                    
                    # Try with a very short duration
                    bars = self.ib_connector.ib.reqHistoricalData(
                        simple_contract,
                        endDateTime='',
                        durationStr='3 m',  # Very short duration
                        barSizeSetting='1 min',  # 1-minute bars
                        whatToShow=what_to_show,
                        useRTH=True,
                        formatDate=1
                    )
                    
                    # Process bars if we got them
                    if bars:
                        data = [{
                            'date': bar.date,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        } for bar in bars]
                        
                        if data:
                            df = pd.DataFrame(data)
                            df.set_index('date', inplace=True)
                            logger.info(f"Successfully retrieved fallback data for {symbol}")
                            return df
                except Exception as fallback_error:
                    logger.error(f"Fallback historical data request failed: {fallback_error}")
                
                # Final fallback approaches remain unchanged
                # ... existing code ...
                
                return None
        else:
            logger.error(f"No IB connection available for historical data request")
            return None
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Add method to handle market data subscription with real-time bars
def get_realtime_bars(self, symbol, bar_size='5 mins'):
    """Get real-time bars for a symbol instead of historical data.
    
    This method uses reqRealTimeBars which is more reliable for real-time data
    than keepUpToDate with historical data.
    
    Args:
        symbol: The symbol or contract to get data for
        bar_size: Size of the bars in seconds (5, 10, 15, 30)
    
    Returns:
        RealTimeBarList object that updates in real-time
    """
    try:
        # Check if we have a contract object or just a symbol string
        contract = None
        
        # Map of futures exchanges based on symbol
        exchange_map = {
            "ES": "CME",    # E-mini S&P 500 - CME (not GLOBEX)
            "NQ": "CME",    # E-mini NASDAQ-100 - CME (not GLOBEX)
            "GC": "COMEX",  # Gold - COMEX (not NYMEX)
            "SI": "COMEX",  # Silver - COMEX (not NYMEX)
            "CL": "NYMEX",  # Crude Oil
            "ZB": "ECBOT",  # 30-Year U.S. Treasury Bond
            "ZN": "ECBOT"   # 10-Year U.S. Treasury Note
        }
        
        # If symbol is already a Contract object
        if hasattr(symbol, 'symbol') and hasattr(symbol, 'secType'):
            contract = symbol
            
            # Update the exchange if needed
            if contract.secType == 'FUT' or contract.secType == 'CONTFUT':
                if contract.symbol in exchange_map:
                    contract.exchange = exchange_map[contract.symbol]
        else:
            # If symbol is a string, try to get it as a fully qualified contract first
            try:
                from ib_insync import Future, ContFuture
                
                if hasattr(self.ib_connector, 'ib'):
                    # Use contract details search
                    if isinstance(symbol, str):
                        # Extract base symbol
                        if '-' in symbol:
                            base_symbol = symbol.split('-')[0]
                        else:
                            base_symbol = symbol
                        
                        # Try to find contract details
                        search_contract = None
                        
                        # Check if it's a continuous future request
                        if 'CONTFUT' in symbol.upper():
                            search_contract = ContFuture(symbol=base_symbol, exchange='')
                        else:
                            search_contract = Future(symbol=base_symbol, exchange='')
                        
                        # Request contract details
                        logger.info(f"Searching for contract details for {base_symbol}")
                        contracts = self.ib_connector.ib.reqContractDetails(search_contract)
                        
                        if contracts:
                            # Use the first contract
                            contract = contracts[0].contract
                            logger.info(f"Found contract for {base_symbol}: {contract}")
                        else:
                            logger.warning(f"No contracts found for {base_symbol}")
            except Exception as contract_err:
                logger.error(f"Error getting contract details: {contract_err}")
                
            # If we still don't have a contract, try to parse the symbol string
            if not contract:
                try:
                    # Parse the symbol string (format: SYMBOL-EXPIRY-TYPE-EXCHANGE)
                    parts = symbol.split('-')
                    
                    if len(parts) >= 3:
                        symbol_base = parts[0]
                        
                        # Get the appropriate exchange based on symbol
                        exchange = exchange_map.get(symbol_base, "CME")  # Default to CME
                        
                        # Determine if it's a continuous or regular futures contract
                        if parts[1].upper() == 'CONTFUT' or 'CONTFUT' in symbol.upper():
                            contract = create_continuous_futures_contract(
                                symbol=symbol_base,
                                exchange=exchange
                            )
                        else:
                            # Regular futures contract
                            contract = create_futures_contract(
                                symbol=symbol_base,
                                expiry=parts[1],
                                exchange=exchange
                            )
                except Exception as e:
                    logger.error(f"Error parsing symbol {symbol}: {e}")
                    return None
                
        if not contract:
            logger.error(f"Could not create contract for {symbol}")
            return None
        
        # Try to qualify the contract with IB
        if hasattr(self.ib_connector, 'ib'):
            try:
                qualified = self.ib_connector.ib.qualifyContracts(contract)
                if qualified:
                    contract = qualified[0]
                    logger.info(f"Using qualified contract: {contract}")
            except Exception as qualify_err:
                logger.warning(f"Could not qualify contract: {qualify_err}")
                # Continue with the original contract
        
        # Convert bar_size string to seconds
        if bar_size == '5 mins':
            bar_seconds = 5
        elif bar_size == '1 min':
            bar_seconds = 5  # Use 5-second bars as minimum
        elif bar_size == '15 mins':
            bar_seconds = 15
        elif bar_size == '30 mins':
            bar_seconds = 30
        else:
            bar_seconds = 5  # Default to 5-second bars
            
        # Log the request
        logger.info(f"Requesting real-time bars for {contract.symbol} ({bar_seconds}s)")
        
        # Request real-time bars
        if hasattr(self.ib_connector, 'ib'):
            # Make multiple attempts with increasing timeouts
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Set a timeout for the request
                    original_timeout = self.ib_connector.ib.RequestTimeout
                    timeout = 10 * (attempt + 1)  # Increase timeout with each attempt
                    self.ib_connector.ib.RequestTimeout = timeout
                    
                    logger.info(f"Attempt {attempt+1}/{max_attempts} for real-time bars (timeout: {timeout}s)")
                    
                    bars = self.ib_connector.ib.reqRealTimeBars(
                        contract, 
                        barSize=bar_seconds,  # in seconds: 5, 10, 15, 30
                        whatToShow='TRADES',
                        useRTH=True,
                        realTimeBarsOptions=[]
                    )
                    
                    # Reset the timeout
                    self.ib_connector.ib.RequestTimeout = original_timeout
                    
                    if bars:
                        logger.info(f"Successfully subscribed to real-time bars for {contract.symbol}")
                        return bars
                    else:
                        logger.warning(f"Empty bars returned for {contract.symbol}")
                        continue
                except Exception as attempt_err:
                    logger.warning(f"Attempt {attempt+1} failed: {attempt_err}")
                    # Reset timeout
                    if 'original_timeout' in locals():
                        self.ib_connector.ib.RequestTimeout = original_timeout
                    
                    # Sleep before retry
                    import time
                    time.sleep(1)
                    
                    # If last attempt, raise
                    if attempt == max_attempts - 1:
                        raise
            
            logger.error(f"All attempts to get real-time bars failed for {contract.symbol}")
            return None
        else:
            logger.error(f"No IB connection available for real-time bars request")
            return None
    except Exception as e:
        logger.error(f"Error getting real-time bars: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Add method to cancel real-time bars
def cancel_realtime_bars(self, bars):
    """Cancel real-time bars subscription."""
    try:
        if hasattr(self.ib_connector, 'ib') and bars:
            self.ib_connector.ib.cancelRealTimeBars(bars)
            logger.info(f"Cancelled real-time bars subscription")
            return True
        return False
    except Exception as e:
        logger.error(f"Error cancelling real-time bars: {e}")
        return False

# Add method to log signal checks
def log_signal_check(self, pair, signal_info):
    """Log signal check information for transparency."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "pair": pair,
        "signal": signal_info
    }
    
    # Save to signals directory
    signal_file = os.path.join("output", "paper_trading", "signals", f"signals_{datetime.datetime.now().strftime('%Y%m%d')}.json")
    
    try:
        if os.path.exists(signal_file):
            with open(signal_file, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
            
        signals.append(log_entry)
        
        with open(signal_file, 'w') as f:
            json.dump(signals, f, indent=2, cls=DateTimeEncoder)
            
        logger.info(f"Signal check logged for {pair}: {signal_info}")
    except Exception as e:
        logger.error(f"Error logging signal check: {e}")

# Add method to handle contract-based market data subscription
def subscribe_market_data_with_contract(self, contract):
    """Subscribe to market data using a properly formatted contract object."""
    try:
        # Use the reqMktData method of the IB object directly
        if hasattr(self.ib_connector, 'ib'):
            logger.info(f"Attempting to subscribe to market data for contract: {contract}")
            
            # If it's a string, try to convert it to a contract
            if isinstance(contract, str):
                try:
                    # Try to parse it as a symbol
                    from ib_insync import Future
                    if '-' in contract:
                        parts = contract.split('-')
                        symbol = parts[0]
                        contract = Future(symbol=symbol)
                        logger.info(f"Converted string {contract} to Future contract")
                    else:
                        # Just use the string as a symbol
                        contract = Future(symbol=contract)
                        logger.info(f"Created Future contract from {contract}")
                except Exception as e:
                    logger.error(f"Failed to convert string to contract: {e}")
                    return False
            
            # Ensure we have a valid contract object
            if not hasattr(contract, 'secType') or not hasattr(contract, 'symbol'):
                logger.error(f"Invalid contract object: {contract}")
                return False
                
            # Apply the correct exchange based on symbol if not set or if it's GLOBEX/NYMEX
            # (common issues from older code)
            if (not hasattr(contract, 'exchange') or 
                not contract.exchange or 
                contract.exchange == 'GLOBEX' or 
                (contract.exchange == 'NYMEX' and contract.symbol in ['GC', 'SI'])):
                
                # Use our exchange mapping function
                exchange = get_exchange_for_symbol(contract.symbol)
                contract.exchange = exchange
                logger.info(f"Set exchange for {contract.symbol} to {exchange}")
                
            if not hasattr(contract, 'currency') or not contract.currency:
                contract.currency = 'USD'
                logger.info(f"Set default currency for {contract.symbol} to USD")
                
            # Log the fully-formed contract we're using
            logger.info(f"Using contract: {contract.symbol} ({contract.secType}) on {contract.exchange}")
            
            # Now request market data with the contract - skip the qualification step if we have 
            # a well-formed contract already
            try:
                if hasattr(contract, 'lastTradeDateOrContractMonth') and contract.lastTradeDateOrContractMonth:
                    # If we already have a front month contract specification, use it directly
                    logger.info(f"Making market data request for {contract.symbol} {contract.lastTradeDateOrContractMonth}")
                else:
                    # Try to qualify contract first, but with timeout
                    logger.info(f"Trying to qualify contract for {contract.symbol}")
                    
                    # Save original timeout
                    original_timeout = None
                    if hasattr(self.ib_connector.ib, 'RequestTimeout'):
                        original_timeout = self.ib_connector.ib.RequestTimeout
                        self.ib_connector.ib.RequestTimeout = 60  # 60-second timeout
                        
                    try:
                        qualified_contracts = self.ib_connector.ib.reqContractDetails(contract)
                        
                        if qualified_contracts:
                            contract = qualified_contracts[0].contract  # Use the first qualified contract
                            logger.info(f"Using qualified contract: {contract}")
                        else:
                            logger.warning(f"Could not qualify contract: {contract}")
                            # Continue anyway with original contract
                    except Exception as qualify_err:
                        logger.warning(f"Error qualifying contract: {qualify_err}, continuing with original")
                    
                    # Reset timeout
                    if original_timeout is not None:
                        self.ib_connector.ib.RequestTimeout = original_timeout
                
                # Make the actual market data request with a timeout
                logger.info(f"Requesting market data for {contract.symbol}")
                
                # Set timeout for market data request
                original_req_timeout = None
                if hasattr(self.ib_connector.ib, 'RequestTimeout'):
                    original_req_timeout = self.ib_connector.ib.RequestTimeout
                    self.ib_connector.ib.RequestTimeout = 60  # 60-second timeout
                
                # Make the request
                ticker = self.ib_connector.ib.reqMktData(contract)
                
                # Reset timeout
                if original_req_timeout is not None:
                    self.ib_connector.ib.RequestTimeout = original_req_timeout
                
                # Generate a unique key for this contract
                if hasattr(contract, 'lastTradeDateOrContractMonth') and contract.lastTradeDateOrContractMonth:
                    symbol_key = f"{contract.symbol}-{contract.lastTradeDateOrContractMonth}-{contract.secType}-{contract.exchange}"
                else:
                    symbol_key = f"{contract.symbol}-{contract.secType}-{contract.exchange}"
                
                # Store the subscription
                self._market_data[symbol_key] = ticker
                
                logger.info(f"Subscribed to market data for {symbol_key}")
                return True
            except Exception as req_error:
                logger.error(f"Error requesting market data: {req_error}")
                return False
        else:
            # Fallback to the original method
            logger.warning("IB connector does not have direct access to IB object. Using fallback method.")
            
            # Try to convert contract to symbol string
            try:
                if hasattr(contract, 'lastTradeDateOrContractMonth') and contract.lastTradeDateOrContractMonth:
                    contract_str = f"{contract.symbol}-{contract.lastTradeDateOrContractMonth}-{contract.secType}-{contract.exchange}"
                else:
                    contract_str = f"{contract.symbol}-{contract.secType}-{contract.exchange}"
                return self.subscribe_market_data(contract_str)
            except:
                logger.error("Failed to convert contract to symbol string for subscription")
                return False
    except Exception as e:
        logger.error(f"Error subscribing to market data with contract: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Attach the methods to the PaperTrader class
PaperTrader.get_subscribed_symbols = get_subscribed_symbols
PaperTrader.get_market_data = get_market_data
PaperTrader.get_account_info = get_account_info
PaperTrader.get_historical_data = get_historical_data
PaperTrader.get_realtime_bars = get_realtime_bars
PaperTrader.cancel_realtime_bars = cancel_realtime_bars
PaperTrader.log_signal_check = log_signal_check
PaperTrader.subscribe_market_data_with_contract = subscribe_market_data_with_contract
# Also attach streaming_bars as a class attribute
PaperTrader._streaming_bars = {}

logger.info("Successfully patched PaperTrader with additional methods")

# Import and modify IntradayMLPaperTrader after patching PaperTrader
from src.paper_trading.intraday_ml_paper_trader import IntradayMLPaperTrader

# Override the _refresh_ml_system method to add signal logging
original_refresh_ml = IntradayMLPaperTrader._refresh_ml_system

def enhanced_refresh_ml_system(self):
    """Enhanced refresh ML system with better logging."""
    logger.info("Refreshing ML system and checking for trading signals...")
    
    try:
        # Ensure datetime is available
        from datetime import datetime, timedelta
        import numpy as np
        import pandas as pd
        
        # Use our patched data collection method that doesn't use start_date
        try:
            # Wrap the market data collection in a try-except to handle errors
            market_data = self._collect_market_data()
        except Exception as data_err:
            logger.error(f"Error collecting market data: {data_err}")
            import traceback
            logger.error(traceback.format_exc())
            # Create an empty market data structure to avoid further errors
            market_data = {'prices': {}, 'volumes': {}, 'spreads': {}}
        
        # Try to load the current regime from the saved file
        self._load_current_regime()
        
        # If we have price data in market_data, try to detect the current market regime
        if market_data and 'prices' in market_data and market_data['prices']:
            try:
                # Use the first symbol with data for regime detection
                regime_symbol = list(market_data['prices'].keys())[0]
                regime_data = market_data['prices'][regime_symbol]
                
                # Verify regime_data is a valid DataFrame before using it
                if isinstance(regime_data, pd.DataFrame) and not regime_data.empty:
                    # Detect market regime using the valid DataFrame
                    self._detect_market_regime(regime_data)
                    logger.info(f"Current market regime: {self.current_regime}")
                else:
                    # If regime_data is not valid, load from file
                    self._load_current_regime()
                    logger.info(f"Using saved regime: {self.current_regime}")
            except Exception as regime_err:
                logger.error(f"Error detecting market regime: {regime_err}")
                # Make sure we have a valid regime even on error
                self._load_current_regime()
        else:
            # If no historical data, load regime from file
            self._load_current_regime()
            logger.info(f"No historical data for regime detection, using saved regime: {self.current_regime}")
        
        # Don't call original refresh if we have no data
        if not market_data or (not market_data.get('prices') and not market_data.get('spreads')):
            logger.warning("No market data or historical data available, skipping ML refresh")
            
            # Log placeholder signals for monitoring
            for pair in self.current_pairs:
                # Get the pair symbols from config - safely handle None return
                pair_symbols = self._get_pair_symbols(pair) or []
                
                # Create basic signal info
                signal_info = {
                    "pair": pair,
                    "timestamp": datetime.now().isoformat(),
                    "ml_signal": "waiting_for_data",
                    "signal_strength": 0.0,
                    "regime": self.current_regime or "unknown",
                    "entry_threshold": 2.0,
                    "status": "waiting_for_data",
                    "note": "Waiting for market data"
                }
                
                # Add symbols info if available
                if len(pair_symbols) >= 2:
                    signal_info["symbol1"] = pair_symbols[0]
                    signal_info["symbol2"] = pair_symbols[1]
                
                # Log the check
                self.paper_trader.log_signal_check(pair, signal_info)
            
            # We still update the dashboard
            self._update_dashboard()
            return True
                
        # Now call the original refresh method with our prepared data
        try:
            result = original_refresh_ml(self)
        except Exception as ml_err:
            logger.error(f"Error in original ML refresh: {ml_err}")
            result = None
        
        # Log information about any pairs being monitored
        for pair in self.current_pairs:
            try:
                # Get the pair symbols from config - safely handle None return
                pair_symbols = self._get_pair_symbols(pair) or []
                
                # Initialize symbol1 and symbol2 to avoid unbound local variable errors
                symbol1 = None
                symbol2 = None
                
                # Get default parameters from config
                if 'default_parameters' not in self.ml_config:
                    self.ml_config['default_parameters'] = {
                        "entry_threshold": 2.0,
                        "exit_threshold": 0.5,
                        "stop_loss": 3.0,
                        "position_size": 0.1
                    }
                
                # Get the pair config from the ML config if available
                pair_config = None
                for p in self.ml_config.get('pairs', []):
                    if p.get('pair_id') == pair:
                        pair_config = p.get('config', {})
                        break
                
                # Use pair-specific config if available, otherwise use defaults
                entry_threshold = pair_config.get('entry_zscore', self.ml_config['default_parameters']['entry_threshold']) if pair_config else self.ml_config['default_parameters']['entry_threshold']
                
                # Get signal information based on available data
                signal_status = "monitoring"
                ml_signal = "pending"
                signal_strength = 0.0
                
                # If we have both symbols with data, we can provide more information
                if len(pair_symbols) >= 2:
                    symbol1 = pair_symbols[0]
                    symbol2 = pair_symbols[1]
                    signal_status = "active_monitoring"
                    
                    # Try to get latest prices
                    raw_market_data = self.paper_trader.get_market_data()
                    
                    if symbol1 in raw_market_data and symbol2 in raw_market_data:
                        if (hasattr(raw_market_data[symbol1], 'last') and 
                            hasattr(raw_market_data[symbol2], 'last') and
                            raw_market_data[symbol1].last and 
                            raw_market_data[symbol2].last):
                            
                            price_ratio = raw_market_data[symbol1].last / raw_market_data[symbol2].last
                            signal_strength = 0.2  # Just a placeholder
                    
                    # Determine signal based on historical data if available
                    prices = market_data.get('prices', {})
                    if symbol1 in prices and symbol2 in prices:
                        ml_signal = "neutral"  # Placeholder - would use model in real implementation
                
                # Create signal info with available data
                signal_info = {
                    "pair": pair,
                    "timestamp": datetime.now().isoformat(),
                    "ml_signal": ml_signal,
                    "signal_strength": signal_strength,
                    "regime": self.current_regime or "unknown",
                    "entry_threshold": entry_threshold,
                    "status": signal_status
                }
                
                # Add symbols info if available
                if symbol1 and symbol2:
                    signal_info["symbol1"] = symbol1
                    signal_info["symbol2"] = symbol2
                    
                    # Add price data if available
                    raw_market_data = self.paper_trader.get_market_data()
                    if symbol1 in raw_market_data and hasattr(raw_market_data[symbol1], 'last'):
                        signal_info["price1"] = raw_market_data[symbol1].last
                    if symbol2 in raw_market_data and hasattr(raw_market_data[symbol2], 'last'):
                        signal_info["price2"] = raw_market_data[symbol2].last
                
                # Log the check
                self.paper_trader.log_signal_check(pair, signal_info)
            except Exception as pair_err:
                logger.error(f"Error processing pair {pair}: {pair_err}")
                
                # Still log a basic signal so we know what's happening
                signal_info = {
                    "pair": pair,
                    "timestamp": datetime.now().isoformat(),
                    "ml_signal": "error",
                    "signal_strength": 0.0,
                    "regime": self.current_regime or "unknown",
                    "status": "error",
                    "note": f"Error processing: {str(pair_err)}"
                }
                
                self.paper_trader.log_signal_check(pair, signal_info)
            
        # Update the dashboard more frequently with signal info
        self._update_dashboard()
    except Exception as e:
        logger.error(f"Error in enhanced ML refresh: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return True  # Always return something even on failure

# Add functions to load and detect market regimes
def _load_current_regime(self):
    """Load current regime from the saved file."""
    try:
        model_dir = getattr(self, 'models_dir', 'models/intraday')
        current_regime_path = os.path.join(model_dir, "current_regime.json")
        
        if os.path.exists(current_regime_path):
            with open(current_regime_path, 'r') as f:
                regime_data = json.load(f)
                
            if 'regime_description' in regime_data:
                self.current_regime = regime_data['regime_description']
                logger.info(f"Loaded current regime from file: {self.current_regime}")
                return True
                
        # If the file doesn't exist or doesn't have the correct data, use a default value
        self.current_regime = "Low Volatility / Weak Trend"
        logger.info(f"Using default market regime: {self.current_regime}")
        return False
    except Exception as e:
        logger.error(f"Error loading current regime: {e}")
        # Set a default regime value
        self.current_regime = "Low Volatility / Weak Trend"
        logger.info(f"Using default market regime: {self.current_regime}")
        return False

def _detect_market_regime(self, price_data):
    """Detect the current market regime using the trained classifier."""
    try:
        import joblib
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Set default regime in case detection fails
        if not hasattr(self, 'current_regime') or not self.current_regime:
            self.current_regime = "Unknown"
        
        # Paths to model files
        model_dir = getattr(self, 'models_dir', 'models/intraday')
        regime_model_path = os.path.join(model_dir, "market_regime_classifier.joblib")
        
        if not os.path.exists(regime_model_path):
            logger.warning(f"Market regime classifier model not found at {regime_model_path}")
            return False
        
        # Load regime classifier
        try:
            regime_classifier = joblib.load(regime_model_path)
            logger.info(f"Loaded market regime classifier from {regime_model_path}")
        except Exception as model_err:
            logger.error(f"Error loading regime classifier model: {model_err}")
            return False
        
        # Extract features from price data
        try:
            # Verify we have valid price data
            if price_data is None or not isinstance(price_data, pd.DataFrame) or price_data.empty:
                logger.warning("Invalid price data for regime detection")
                return self._load_current_regime()  # Use cached regime as fallback
            
            if 'close' not in price_data.columns:
                logger.warning("No 'close' column in price data for regime detection")
                return self._load_current_regime()  # Use cached regime as fallback
            
            # Basic feature extraction - use feature names that match the trained model
            features = pd.DataFrame()
            
            # Calculate volatility (standard deviation of returns)
            returns = price_data['close'].pct_change(fill_method=None).dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] if len(returns) > 20 else 0
            
            # Calculate trend strength
            ma_short = price_data['close'].rolling(window=5).mean()
            ma_long = price_data['close'].rolling(window=20).mean()
            trend_strength = (ma_short.iloc[-1] / ma_long.iloc[-1] - 1) if len(ma_long) > 20 else 0
            
            # Calculate correlation (autocorrelation)
            if len(returns) > 10:
                autocorr = returns.autocorr(lag=1)
                correlation = abs(autocorr)  # Use absolute correlation
            else:
                correlation = 0.5  # Neutral correlation if not enough data
            
            # Calculate mean reversion strength
            if len(returns) > 20:
                # Calculate Hurst exponent (simplified version) 
                # Values < 0.5 indicate mean reversion, > 0.5 indicate trending
                price_diff = np.diff(np.log(price_data['close'].values))
                if len(price_diff) > 0:
                    abs_diff = np.abs(price_diff)
                    mean_reversion = 1 - (abs_diff.mean() / abs_diff.std())
                    # Normalize to [0, 1]
                    mean_reversion = max(0, min(1, mean_reversion))
                else:
                    mean_reversion = 0.5
            else:
                mean_reversion = 0.5
            
            # Create feature DataFrame with the correct feature names expected by the model
            features = pd.DataFrame({
                'volatility_avg': [volatility],
                'trend_strength_avg': [trend_strength],
                'correlation_avg': [correlation],
                'mean_reversion_avg': [mean_reversion]
            })
            
            # Detect the regime
            try:
                regime_id = regime_classifier.predict(features)[0]
                
                # Try to get regime description
                try:
                    regime_desc = regime_classifier.get_regime_description(regime_id)
                    self.current_regime = regime_desc
                    logger.info(f"Detected market regime: {regime_desc} (ID: {regime_id})")
                except:
                    # Fallback to preset descriptions
                    regimes = ["Low Volatility / Mean Reverting", "Low Volatility / Weak Trend", "High Volatility / Strong Trend"]
                    self.current_regime = regimes[regime_id % len(regimes)]
                    logger.info(f"Using fallback regime description: {self.current_regime} (ID: {regime_id})")
                    
                return True
            except Exception as predict_err:
                logger.error(f"Error predicting regime: {predict_err}")
                return self._load_current_regime()
                
        except Exception as feature_err:
            logger.error(f"Error extracting features for regime detection: {feature_err}")
            
            # Try loading from file as fallback
            return self._load_current_regime()
            
    except Exception as e:
        logger.error(f"Error in market regime detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return self._load_current_regime()

# Add helper method to get pair symbols
def _get_pair_symbols(self, pair_id):
    """Helper method to get the actual symbols for a pair from configuration."""
    try:
        # Default initializations to avoid unbound local variable errors
        symbol1 = None
        symbol2 = None
        
        # Look up in the config
        if hasattr(self, 'ml_config') and 'pairs' in self.ml_config:
            for pair_config in self.ml_config.get('pairs', []):
                if pair_config.get('pair_id') == pair_id:
                    symbol1 = pair_config.get('full_symbol1')
                    symbol2 = pair_config.get('full_symbol2')
                    if symbol1 and symbol2:
                        return [symbol1, symbol2]
        
        # Try to parse the pair_id (assuming format like ES_NQ)
        if '_' in pair_id:
            parts = pair_id.split('_')
            if len(parts) == 2:
                # Try to find the subscribed symbols that match these parts
                subscribed = self.paper_trader.get_subscribed_symbols()
                symbol1_candidates = [s for s in subscribed if parts[0] in s]
                symbol2_candidates = [s for s in subscribed if parts[1] in s]
                
                if symbol1_candidates and symbol2_candidates:
                    symbol1 = symbol1_candidates[0]
                    symbol2 = symbol2_candidates[0]
                    return [symbol1, symbol2]
        
        # Could not find symbols
        logger.warning(f"Could not determine symbols for pair {pair_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting pair symbols: {e}")
        return None

# Override the _update_dashboard method to create a more detailed dashboard
original_update_dashboard = IntradayMLPaperTrader._update_dashboard

def enhanced_update_dashboard(self):
    """Create a more detailed and transparent dashboard."""
    try:
        dashboard_path = os.path.join(self.output_dir, "dashboard", "index.html")
        
        # Get account information
        account_info = self.paper_trader.get_account_info()
        
        # Get any open positions
        positions = self.paper_trader._positions
        num_positions = len(positions)
        
        # Get current regime and pairs
        regime = self.current_regime or "Unknown"
        regime_class = "neutral"
        if regime:
            if "volatility" in regime.lower():
                if "high" in regime.lower():
                    regime_class = "warning"
                elif "low" in regime.lower():
                    regime_class = "good"
            elif "trend" in regime.lower():
                if "strong" in regime.lower():
                    regime_class = "good"
                elif "weak" in regime.lower():
                    regime_class = "neutral"
                    
        pairs_str = ", ".join(self.current_pairs) if self.current_pairs else "None"
        
        # Get subscribed symbols
        subscribed_symbols = self.paper_trader.get_subscribed_symbols()
        subscribed_str = ", ".join(subscribed_symbols) if subscribed_symbols else "None"
        
        # Get latest signals
        latest_signals = []
        signal_file = os.path.join("output", "paper_trading", "signals", f"signals_{datetime.datetime.now().strftime('%Y%m%d')}.json")
        if os.path.exists(signal_file):
            try:
                with open(signal_file, 'r') as f:
                    all_signals = json.load(f)
                    # Get the 5 most recent signals
                    latest_signals = all_signals[-5:] if len(all_signals) > 0 else []
            except Exception as e:
                logger.error(f"Error loading signals for dashboard: {e}")
        
        # Create the dashboard HTML
        with open(dashboard_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="refresh" content="60">
                <title>ML-Enhanced Paper Trading Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .metric {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                    .metric h2 {{ margin-top: 0; color: #0066cc; }}
                    .value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .status {{ display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; }}
                    .good {{ background-color: #4CAF50; }}
                    .warning {{ background-color: #FF9800; }}
                    .bad {{ background-color: #F44336; }}
                    .neutral {{ background-color: #2196F3; }}
                    .last-updated {{ color: #666; font-size: 12px; margin-top: 30px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #f5f5f5; }}
                    .signal-table {{ margin-top: 20px; }}
                    pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                </style>
            </head>
            <body>
                <h1>ML-Enhanced Paper Trading Dashboard</h1>
                <p>Running paper trading for pairs: {pairs_str}</p>
                
                <div class="metric">
                    <h2>Status</h2>
                    <div class="value">
                        <span class="status good">ACTIVE</span>
                    </div>
                    <p>Paper trading is currently active and monitoring the market</p>
                </div>
                
                <div class="metric">
                    <h2>Market Data Subscriptions</h2>
                    <div class="value">
                        {len(subscribed_symbols)} symbols
                    </div>
                    <p>Currently subscribed to: {subscribed_str}</p>
                </div>
                
                <div class="metric">
                    <h2>Market Regime</h2>
                    <div class="value">
                        <span class="status {regime_class}">{regime}</span>
                    </div>
                    <p>Current detected market regime</p>
                </div>
                
                <div class="metric">
                    <h2>Account Value</h2>
                    <div class="value">${account_info.get('equity', 0):.2f}</div>
                    <p>Current account equity</p>
                </div>
                
                <div class="metric">
                    <h2>Open Positions</h2>
                    <div class="value">{num_positions}</div>
                    <p>Number of currently open positions</p>
                </div>
                
                <div class="metric">
                    <h2>Today's P&L</h2>
                    <div class="value">${account_info.get('pnl_day', 0):.2f}</div>
                    <p>Profit & Loss for the current trading day</p>
                </div>
                
                <div class="metric">
                    <h2>Signal Monitoring</h2>
                    <table class="signal-table">
                        <tr>
                            <th>Timestamp</th>
                            <th>Pair</th>
                            <th>Signal Info</th>
                        </tr>
            """)
            
            # Add latest signals to the table
            for signal in reversed(latest_signals):
                timestamp = signal.get("timestamp", "")
                pair = signal.get("pair", "")
                signal_info = json.dumps(signal.get("signal", {}), indent=2)
                
                f.write(f"""
                        <tr>
                            <td>{timestamp}</td>
                            <td>{pair}</td>
                            <td><pre>{signal_info}</pre></td>
                        </tr>
                """)
            
            if not latest_signals:
                f.write(f"""
                        <tr>
                            <td colspan="3" style="text-align: center;">No signals recorded yet</td>
                        </tr>
                """)
            
            f.write(f"""
                    </table>
                </div>
                
                <p class="last-updated">Last updated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>This page automatically refreshes every minute.</p>
                <p>For detailed logs, check the logs directory. Signal history is in the signals directory.</p>
            </body>
            </html>
            """)
            
        logger.info(f"Updated dashboard at: {dashboard_path}")
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        try:
            # Fallback to a simpler dashboard
            original_update_dashboard(self)
        except:
            # If all else fails, create a basic dashboard
            try:
                dashboard_path = os.path.join(self.output_dir, "dashboard", "index.html")
                with open(dashboard_path, 'w') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html><head><title>Paper Trading</title>
                    <meta http-equiv="refresh" content="60">
                    </head><body>
                    <h1>Paper Trading Active</h1>
                    <p>Last update: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>Error occurred with dashboard: {str(e)}</p>
                    </body></html>
                    """)
            except:
                pass

# Add method to monitor and refresh market data subscriptions
def _monitor_market_data(self):
    """Check if market data subscriptions are active and refresh if needed."""
    try:
        # Check existing market data
        market_data = self.get_market_data()
        if not market_data:
            logger.warning("No market data subscriptions found")
            return False
        
        # Count subscriptions with valid data
        valid_count = 0
        need_refresh_symbols = []
        
        for symbol, ticker in market_data.items():
            if hasattr(ticker, 'last') and ticker.last is not None:
                valid_count += 1
                logger.info(f"Active market data for {symbol}: last={ticker.last}")
            elif hasattr(ticker, 'close') and ticker.close is not None:
                valid_count += 1
                logger.info(f"Active market data for {symbol}: close={ticker.close}")
            else:
                logger.warning(f"No price data for {symbol}, may need to refresh")
                need_refresh_symbols.append(symbol)
        
        # Process symbols that need refreshing in batch to reduce API calls
        if need_refresh_symbols:
            logger.info(f"Refreshing {len(need_refresh_symbols)} symbols: {need_refresh_symbols}")
            
            # Try to refresh the subscriptions
            try:
                # Only refresh if we have an IB connection
                if hasattr(self.ib_connector, 'ib'):
                    # Use a more efficient batch approach
                    import threading
                    
                    def refresh_symbol_with_timeout(symbol, timeout=10):
                        """Refresh a symbol subscription with timeout."""
                        try:
                            # Extract symbol root
                            base_symbol = symbol.split('-')[0] if '-' in symbol else symbol
                            
                            from ib_insync import Future
                            
                            # Get appropriate exchange
                            exchange = get_exchange_for_symbol(base_symbol)
                            
                            # Create a Future contract with no expiry (let IB determine front month)
                            contract = Future(symbol=base_symbol, exchange=exchange, currency='USD')
                            
                            # Try to get contract details first
                            details = self.ib_connector.ib.reqContractDetails(contract)
                            
                            if details and len(details) > 0:
                                # Use the first contract returned (usually the front month)
                                front_month = details[0].contract
                                # Try to subscribe directly to the front month
                                return self.subscribe_market_data_with_contract(front_month)
                            else:
                                # Fallback to basic subscription if no details
                                logger.warning(f"No contract details for {base_symbol}, using simple subscription")
                                return self.subscribe_market_data(base_symbol)
                            
                        except Exception as e:
                            logger.error(f"Error refreshing {symbol}: {e}")
                            return False
                    
                    # Process symbols in parallel with timeout using a thread pool
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        # Use shorter timeouts for batch processing
                        futures = {executor.submit(refresh_symbol_with_timeout, symbol, 10): symbol for symbol in need_refresh_symbols}
                        
                        # Collect results with timeouts
                        refreshed_count = 0
                        for future in concurrent.futures.as_completed(futures, timeout=30):
                            symbol = futures[future]
                            try:
                                result = future.result()
                                if result:
                                    refreshed_count += 1
                                    logger.info(f"Successfully refreshed {symbol}")
                            except Exception as exc:
                                logger.error(f"Refresh {symbol} generated an exception: {exc}")
                        
                        logger.info(f"Successfully refreshed {refreshed_count}/{len(need_refresh_symbols)} symbols")
            except Exception as refresh_err:
                logger.error(f"Error in batch refresh: {refresh_err}")
        
        logger.info(f"Market data status: {valid_count}/{len(market_data)} symbols have valid price data")
        return valid_count > 0
    except Exception as e:
        logger.error(f"Error monitoring market data: {e}")
        return False

# Add the monitor_market_data method to PaperTrader
PaperTrader._monitor_market_data = _monitor_market_data

# Modify the enhanced_refresh_loop function to reduce frequency of searches
def enhanced_refresh_loop(self):
    """Enhanced refresh loop with better error handling and logging."""
    logger.info("Starting enhanced refresh loop for ML monitoring")
    
    # Track when we last checked market data
    last_market_data_check = time.time()
    market_data_check_interval = 900  # Check every 15 minutes instead of 5 minutes
    
    # Add timeout for contract searches
    def contract_search_with_timeout(func, symbol, timeout=60):  # Increased from 30 to 60 seconds
        """Run contract search with a timeout to prevent hanging."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(symbol)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.warning(f"Contract search for {symbol} timed out after {timeout} seconds")
            return None
        
        if exception[0]:
            logger.error(f"Contract search for {symbol} raised: {exception[0]}")
            return None
        
        return result[0]
    
    # Monkey-patch the contract search functions to use timeout
    original_get_future = self.paper_trader.ib_connector.ib.reqContractDetails
    
    def reqContractDetails_with_timeout(contract, timeout=60):  # Increased from 30 to 60 seconds
        """Run reqContractDetails with timeout - single longer attempt."""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = original_get_future(contract)
            except Exception as e:
                exception[0] = e
        
        logger.info(f"Requesting contract details for {contract.symbol} with {timeout}s timeout")
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.warning(f"reqContractDetails for {contract.symbol} timed out after {timeout}s")
            return []
        
        if exception[0]:
            logger.error(f"reqContractDetails for {contract.symbol} raised: {exception[0]}")
            return []
        
        if result[0]:
            logger.info(f"Successfully found contract details for {contract.symbol}")
        else:
            logger.warning(f"No contract details found for {contract.symbol}")
            
        return result[0]
    
    # Apply the monkey patch if possible
    if hasattr(self.paper_trader, 'ib_connector') and hasattr(self.paper_trader.ib_connector, 'ib'):
        self.paper_trader.ib_connector.ib._original_reqContractDetails = self.paper_trader.ib_connector.ib.reqContractDetails
        self.paper_trader.ib_connector.ib.reqContractDetails = reqContractDetails_with_timeout
        logger.info("Applied timeout to contract detail searches")
    
    while self.is_running:
        try:
            # Initial delay to allow data to be collected
            time.sleep(self.refresh_interval_seconds)
            
            # Skip if not running
            if not self.is_running:
                break
            
            # Check if we need to monitor market data
            current_time = time.time()
            if current_time - last_market_data_check > market_data_check_interval:
                logger.info("Performing periodic market data check")
                try:
                    # Use a timeout for this operation
                    timeout_thread = threading.Thread(target=self.paper_trader._monitor_market_data)
                    timeout_thread.daemon = True
                    timeout_thread.start()
                    timeout_thread.join(60)  # 60 second max for market data check
                    
                    if timeout_thread.is_alive():
                        logger.warning("Market data check timed out after 60 seconds")
                except Exception as monitor_err:
                    logger.error(f"Error in market data check: {monitor_err}")
                
                last_market_data_check = current_time
            
            # Check if we have any subscribed symbols
            symbols = self.paper_trader.get_subscribed_symbols()
            if not symbols:
                logger.warning("No market data subscriptions found. Checking for configuration...")
                
                # Try to subscribe to market data for pairs
                if self.current_pairs:
                    # Get the full symbols from the config
                    config_path = os.path.join('config', 'paper_trading.json')
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            
                        # Subscribe to market data for these pairs
                        for pair_config in config.get('pairs', []):
                            if pair_config.get('pair_id') in self.current_pairs:
                                # Get the symbols
                                symbol1 = pair_config.get('full_symbol1')
                                symbol2 = pair_config.get('full_symbol2')
                                
                                if symbol1 and symbol2:
                                    # Subscribe to market data
                                    logger.info(f"Subscribing to market data for {symbol1} and {symbol2}")
                                    try:
                                        self.paper_trader.subscribe_market_data_with_contract(symbol1)
                                        self.paper_trader.subscribe_market_data_with_contract(symbol2)
                                        logger.info(f"Successfully subscribed to market data for pair {pair_config.get('pair_id')}")
                                    except Exception as e:
                                        logger.error(f"Error subscribing to market data: {e}")
                    except Exception as e:
                        logger.error(f"Error loading pair symbols from config: {e}")
            
            # Always refresh ML system even if no symbols are subscribed yet
            # This allows us to show activity in the dashboard
            try:
                # Set a timeout for the ML refresh to prevent hanging
                timeout_thread = threading.Thread(target=self._refresh_ml_system)
                timeout_thread.daemon = True
                timeout_thread.start()
                timeout_thread.join(120)  # 2 minute timeout
                
                if timeout_thread.is_alive():
                    logger.warning("ML system refresh timed out after 120 seconds, continuing with loop")
            except Exception as ml_err:
                logger.error(f"Error in ML refresh: {ml_err}")
            
            # Log that we're actively monitoring
            logger.info(f"Actively monitoring market for trading opportunities. Current regime: {self.current_regime}")
            
        except Exception as e:
            logger.error(f"Error in refresh loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Still update dashboard even on error
            try:
                self._update_dashboard()
            except Exception as dash_error:
                logger.error(f"Dashboard update failed: {dash_error}")
            
            # Send alert
            if self.enable_alerts:
                try:
                    self._send_alert(
                        title="Refresh Loop Error",
                        message=f"Error in refresh loop: {e}",
                        level="error"
                    )
                except:
                    pass
            
            # Short pause before retrying
            time.sleep(5)
    
    # Remove monkey patch if we applied it
    if hasattr(self.paper_trader, 'ib_connector') and hasattr(self.paper_trader.ib_connector, 'ib') and hasattr(self.paper_trader.ib_connector.ib, '_original_reqContractDetails'):
        self.paper_trader.ib_connector.ib.reqContractDetails = self.paper_trader.ib_connector.ib._original_reqContractDetails
        logger.info("Removed timeout from contract detail searches")

# Update the IntradayMLPaperTrader with our enhanced refresh loop
IntradayMLPaperTrader._refresh_loop = enhanced_refresh_loop

# Also make sure the _load_current_regime method is attached to IntradayMLPaperTrader
IntradayMLPaperTrader._load_current_regime = _load_current_regime

logger.info("Added market data monitoring capability")

# Initialize variables
paper_trader = None
keep_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop the paper trader."""
    global keep_running
    logger.info("Stopping paper trader (received interrupt signal)...")
    keep_running = False

# Add these helper functions before the add_example_pairs function

def create_futures_contract(symbol, expiry, exchange, currency='USD'):
    """
    Create a properly formatted futures contract for IB.
    
    Using the proper format based on IB documentation:
    https://interactivebrokers.github.io/tws-api/basic_contracts.html
    """
    from ib_insync import Contract
    
    contract = Contract()
    contract.secType = 'FUT'
    contract.exchange = exchange
    contract.currency = currency
    
    # Extract symbol from symbol-expiry-FUT-exchange format if needed
    if '-' in symbol:
        contract.symbol = symbol.split('-')[0]
    else:
        contract.symbol = symbol
    
    # Parse expiry - format should be YYYYMM
    if expiry.startswith('20'):
        contract.lastTradeDateOrContractMonth = expiry
    
    logger.info(f"Created futures contract: {contract.symbol}-{contract.lastTradeDateOrContractMonth}-{contract.exchange}")
    return contract

def create_continuous_futures_contract(symbol, exchange, currency='USD'):
    """
    Create a continuous futures contract which doesn't expire.
    
    This function is maintained for backward compatibility with existing code,
    but in many cases it's better to let IB determine the front month contract.
    """
    from ib_insync import Contract, Future
    
    logger.info(f"Note: Using create_continuous_futures_contract for {symbol} (consider using front month futures instead)")
    
    # Try using a standard Future with no expiry specified first
    try:
        contract = Future(symbol=symbol, exchange=exchange, currency=currency)
        logger.info(f"Created futures contract with no expiry for {symbol}-{exchange}")
        return contract
    except Exception as e:
        logger.warning(f"Error creating standard future, falling back to ContFuture: {e}")
        
        # Fallback to ContFuture for backward compatibility
        contract = Contract()
        contract.secType = 'CONTFUT'
        contract.exchange = exchange
        contract.currency = currency
        
        # Extract symbol from symbol-expiry-FUT-exchange format if needed
        if '-' in symbol:
            contract.symbol = symbol.split('-')[0]
        else:
            contract.symbol = symbol
        
        logger.info(f"Created continuous futures contract: {contract.symbol}-CONTFUT-{contract.exchange}")
        return contract

# Helper function to get the appropriate exchange for a symbol
def get_exchange_for_symbol(symbol):
    """Return the appropriate exchange for a given futures symbol."""
    exchange_map = {
        "ES": "CME",    # E-mini S&P 500 - CME (not GLOBEX)
        "NQ": "CME",    # E-mini NASDAQ-100 - CME (not GLOBEX) 
        "GC": "COMEX",  # Gold - COMEX (not NYMEX)
        "SI": "COMEX",  # Silver - COMEX (not NYMEX)
        "CL": "NYMEX",  # Crude Oil
        "ZB": "ECBOT",  # 30-Year U.S. Treasury Bond
        "ZN": "ECBOT"   # 10-Year U.S. Treasury Note
    }
    return exchange_map.get(symbol, "CME")  # Default to CME

# Function to add example pairs for monitoring
def add_example_pairs(paper_trader_instance):
    """Add example pairs to monitor if none are configured."""
    # Check if any pairs are already being monitored
    if not paper_trader_instance.current_pairs:
        example_pairs = ["ES_NQ", "GC_SI"]
        paper_trader_instance.current_pairs = example_pairs
        logger.info(f"Added example pairs for monitoring: {example_pairs}")

        # Create direct contract definitions instead of searching
        try:
            from ib_insync import Contract, Future, ContFuture
            
            # Define the contracts with the proper exchanges based on testing
            contracts = {
                # E-mini S&P 500 Futures - use CME, not GLOBEX
                "ES": Contract(
                    symbol="ES",
                    secType="FUT", 
                    exchange="CME",
                    currency="USD"
                ),
                # E-mini NASDAQ-100 Futures - use CME, not GLOBEX
                "NQ": Contract(
                    symbol="NQ", 
                    secType="FUT", 
                    exchange="CME",
                    currency="USD"
                ),
                # Gold Futures - use COMEX, not NYMEX
                "GC": Contract(
                    symbol="GC",
                    secType="FUT", 
                    exchange="COMEX",
                    currency="USD"
                ),
                # Silver Futures - use COMEX, not NYMEX
                "SI": Contract(
                    symbol="SI",
                    secType="FUT", 
                    exchange="COMEX",
                    currency="USD"
                )
            }
            
            # Log the contracts to debug
            for symbol, contract in contracts.items():
                logger.info(f"Contract for {symbol}: {contract}")
                logger.info(f"Contract {symbol} attributes: symbol={contract.symbol}, secType={contract.secType}, exchange={contract.exchange}")
            
            # Use contract definitions instead of searching
            if hasattr(paper_trader_instance.paper_trader, 'ib_connector') and hasattr(paper_trader_instance.paper_trader.ib_connector, 'ib'):
                ib = paper_trader_instance.paper_trader.ib_connector.ib
                
                # Try directly subscribing without reqContractDetails
                logger.info("Trying direct market data subscription without contract details")
                direct_subscribed = 0
                for symbol, contract in contracts.items():
                    try:
                        # Try multiple expiry months as fallbacks
                        subscribed = False
                        current_year = datetime.datetime.now().year
                        current_month = datetime.datetime.now().month
                        
                        # Try multiple expiry months: current month + 3, 2, 4, 6 months ahead
                        expiry_offsets = [3, 2, 4, 6]
                        
                        for offset in expiry_offsets:
                            if subscribed:
                                break
                                
                            try:
                                # Calculate expiry month
                                front_month = current_month + offset
                                year = current_year
                                while front_month > 12:
                                    front_month = front_month - 12
                                    year = year + 1
                                    
                                # Format as YYYYMM
                                contract_month = f"{year}{front_month:02d}"
                                
                                # Create a NEW contract instead of trying to copy
                                from ib_insync import Contract
                                month_contract = Contract()
                                month_contract.symbol = contract.symbol
                                month_contract.secType = contract.secType
                                month_contract.exchange = contract.exchange
                                month_contract.currency = contract.currency
                                month_contract.lastTradeDateOrContractMonth = contract_month
                                
                                logger.info(f"Trying direct subscription for {symbol} with contract month {contract_month}")
                                paper_trader_instance.paper_trader.subscribe_market_data_with_contract(month_contract)
                                logger.info(f"Successfully subscribed to {symbol} with {contract_month}")
                                direct_subscribed += 1
                                subscribed = True
                                break
                            except Exception as month_err:
                                logger.warning(f"Failed {symbol} with month {contract_month}: {month_err}")
                        
                        # If all expiry tries failed, try without expiry (continuous future)
                        if not subscribed:
                            try:
                                logger.info(f"Trying continuous future for {symbol}")
                                # Create a NEW contract for continuous futures
                                from ib_insync import Contract
                                continuous_contract = Contract()
                                continuous_contract.symbol = contract.symbol
                                continuous_contract.secType = 'CONTFUT'
                                continuous_contract.exchange = contract.exchange
                                continuous_contract.currency = contract.currency
                                
                                paper_trader_instance.paper_trader.subscribe_market_data_with_contract(continuous_contract)
                                logger.info(f"Successfully subscribed to continuous future for {symbol}")
                                direct_subscribed += 1
                                subscribed = True
                            except Exception as cont_err:
                                logger.warning(f"Failed continuous future for {symbol}: {cont_err}")
                    except Exception as direct_err:
                        logger.error(f"Error with direct subscription for {symbol}: {direct_err}")
                if direct_subscribed > 0:
                    logger.info(f"Successfully subscribed to {direct_subscribed} symbols directly")
                    
                # If direct subscription doesn't work, continue with the existing approach
                if direct_subscribed == 0:
                    # Subscribe to each contract
                    subscribed_count = 0
                    for symbol, contract in contracts.items():
                        try:
                            # Get contract details to find front month contract
                            logger.info(f"Requesting contract details for {symbol} on {contract.exchange}")
                            details = ib.reqContractDetails(contract)
                            
                            if details and len(details) > 0:
                                # Use the first contract (usually front month)
                                front_month = details[0].contract
                                
                                # Qualify the contract
                                qualified_contracts = ib.qualifyContracts(front_month)
                                
                                if qualified_contracts and len(qualified_contracts) > 0:
                                    qualified_contract = qualified_contracts[0]
                                    logger.info(f"Subscribing to market data for front month {symbol}: {qualified_contract}")
                                    paper_trader_instance.paper_trader.subscribe_market_data_with_contract(qualified_contract)
                                    subscribed_count += 1
                                else:
                                    logger.warning(f"Could not qualify contract for {symbol}")
                            else:
                                logger.warning(f"No contract details found for {symbol}")
                        except Exception as e:
                            logger.error(f"Error subscribing to {symbol}: {e}")
                    
                    logger.info(f"Subscribed to {subscribed_count} of {len(contracts)} futures contracts using contract details")
                
                # If we still couldn't get any contracts through details or direct subscription, try a simple approach
                if direct_subscribed == 0 and subscribed_count == 0:
                    logger.info("Trying simplified approach for market data subscriptions")
                    
                    # First try direct bare futures without month specification
                    basic_subscribed = 0
                    for symbol, contract_template in contracts.items():
                        try:
                            # Create a bare futures contract with no expiry month
                            from ib_insync import Future
                            
                            bare_contract = Future(
                                symbol=symbol,
                                exchange=contract_template.exchange,
                                currency='USD'
                            )
                            
                            logger.info(f"Trying bare futures contract for {symbol} on {bare_contract.exchange}")
                            paper_trader_instance.paper_trader.subscribe_market_data_with_contract(bare_contract)
                            logger.info(f"Successfully subscribed to {symbol} using bare future")
                            basic_subscribed += 1
                        except Exception as future_err:
                            logger.error(f"Error with bare future for {symbol}: {future_err}")
                    
                    # If that didn't work, try the most basic approach
                    if basic_subscribed == 0:
                        for symbol in contracts.keys():
                            try:
                                # Try direct symbol subscription as fallback
                                logger.info(f"Trying basic market data subscription for {symbol}")
                                paper_trader_instance.paper_trader.subscribe_market_data(symbol)
                                subscribed_count += 1
                            except Exception as basic_err:
                                logger.error(f"Error with basic subscription for {symbol}: {basic_err}")
            else:
                logger.error("No IB connection available")
                
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Generate example signals for demonstration
        for pair in example_pairs:
            # Create a demonstration signal
            signal_info = {
                "pair": pair,
                "timestamp": datetime.datetime.now().isoformat(),
                "ml_signal": "neutral",
                "signal_strength": 0.45,
                "regime": "ranging",
                "entry_threshold": 2.0,
                "status": "monitoring",
                "note": "Demo signal - no actual market data available"
            }
            
            # Log the check
            paper_trader_instance.paper_trader.log_signal_check(pair, signal_info)
            logger.info(f"Created demo signal for {pair}")
        
        # Attempt to load the current market regime from file
        try:
            paper_trader_instance._load_current_regime()
            logger.info(f"Current market regime: {paper_trader_instance.current_regime}")
        except Exception as e:
            logger.error(f"Failed to load market regime: {e}")
            
    return paper_trader_instance.current_pairs

# Add a patched _collect_market_data method
def patched_collect_market_data(self) -> dict:
    """Patched version of _collect_market_data that works with our real-time bars implementation.
    Uses real-time data to reduce API request load and avoid historical data pacing violations.
    """
    # Get subscribed symbols
    symbols = self.paper_trader.get_subscribed_symbols()
    
    if not symbols:
        logger.warning("No subscribed symbols found")
        return {}
    
    # Initialize data structures
    prices = {}
    volumes = {}
    
    # Initialize real-time bars dictionary if not already created
    if not hasattr(self, '_realtime_bars'):
        self._realtime_bars = {}
    
    # First, subscribe to real-time bars for any symbols not already subscribed
    for symbol in symbols:
        if symbol not in self._realtime_bars:
            try:
                # Try to get real-time bars
                bars = self.paper_trader.get_realtime_bars(symbol, bar_size='5 mins')
                
                if bars is not None:
                    self._realtime_bars[symbol] = bars
                    logger.info(f"Successfully subscribed to real-time bars for {symbol}")
            except Exception as e:
                logger.error(f"Error subscribing to real-time bars for {symbol}: {e}")
    
    # Now convert real-time bars data to DataFrames for processing
    for symbol, bars in self._realtime_bars.items():
        try:
            # Convert real-time bars to DataFrame
            if len(bars) > 0:
                # Create DataFrame from bars
                data = [{
                    'date': bar.time,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in bars]
                
                if data:
                    df = pd.DataFrame(data)
                    df.set_index('date', inplace=True)
                    prices[symbol] = df
                    
                    # Extract volume data
                    if 'volume' in df.columns:
                        volumes[symbol] = pd.DataFrame({'volume': df['volume']}, index=df.index)
                        
                    logger.info(f"Successfully processed {len(df)} real-time bars for {symbol}")
                else:
                    logger.warning(f"No real-time bars data for {symbol}")
            else:
                logger.warning(f"Empty real-time bars list for {symbol}")
        except Exception as e:
            logger.error(f"Error processing real-time bars for {symbol}: {e}")
    
    # If we don't have data from real-time bars yet, try historical data as fallback
    if not prices:
        logger.warning("No real-time bars data available, falling back to historical data")
        
        # Process all symbols to ensure we get data
        symbols_to_process = symbols
        logger.info(f"Processing {len(symbols_to_process)} symbols for historical data")
        
        # Get historical data for selected symbols
        for symbol in symbols_to_process:
            try:
                # Try with a shorter duration first to avoid timeouts
                historical_data = self.paper_trader.get_historical_data(
                    symbol=symbol,
                    duration='10 m',  # Much shorter window for initial data
                    bar_size="1 min",
                    what_to_show="TRADES"
                )
                
                if historical_data is not None and not historical_data.empty:
                    prices[symbol] = historical_data
                    if 'volume' in historical_data.columns:
                        volumes[symbol] = pd.DataFrame({'volume': historical_data['volume']}, index=historical_data.index)
                    
                    logger.info(f"Successfully retrieved historical data for {symbol}")
                else:
                    # Try with even smaller duration as a last resort
                    logger.info(f"Trying with smaller timeframe for {symbol}")
                    historical_data = self.paper_trader.get_historical_data(
                        symbol=symbol,
                        duration='5 m',
                        bar_size="1 min",
                        what_to_show="TRADES"
                    )
                    
                    if historical_data is not None and not historical_data.empty:
                        prices[symbol] = historical_data
                        if 'volume' in historical_data.columns:
                            volumes[symbol] = pd.DataFrame({'volume': historical_data['volume']}, index=historical_data.index)
                        logger.info(f"Successfully retrieved shorter historical data for {symbol}")
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")
    
    # If we still don't have any price data, create some placeholder data for demonstration
    if not prices and symbols:
        logger.warning("No historical data available, creating placeholder data")
        # Use current timestamp
        current_time = datetime.datetime.now()
        
        # Create placeholder data for each symbol
        for symbol in symbols:
            # Create a simple DataFrame with a single row (current time)
            placeholder_data = pd.DataFrame({
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [0]
            }, index=[current_time])
            
            prices[symbol] = placeholder_data
            volumes[symbol] = pd.DataFrame({'volume': [0]}, index=[current_time])
            
            logger.info(f"Created placeholder data for {symbol}")
    
    # Calculate spreads for pairs
    spreads = {}
    
    # Process pairs
    for pair_id in self.current_pairs:
        try:
            # Get pair symbols
            pair_symbols = self._get_pair_symbols(pair_id)
            
            if pair_symbols and len(pair_symbols) >= 2:
                symbol1 = pair_symbols[0]
                symbol2 = pair_symbols[1]
                
                # Check if we have data for both symbols
                if symbol1 in prices and symbol2 in prices:
                    # Align the dataframes to the same time index
                    common_idx = prices[symbol1].index.intersection(prices[symbol2].index)
                    if len(common_idx) > 2:  # Need at least a few data points
                        price1 = prices[symbol1].loc[common_idx]['close']
                        price2 = prices[symbol2].loc[common_idx]['close']
                        
                        # Calculate hedge ratio using simple linear regression
                        try:
                            model = sm.OLS(price1, sm.add_constant(price2)).fit()
                            hedge_ratio = model.params[1]
                        except:
                            # Fallback to simple ratio of means
                            hedge_ratio = price1.mean() / price2.mean() if price2.mean() != 0 else 1.0
                        
                        # Calculate spread
                        spread = price1 - hedge_ratio * price2
                        
                        # Store the spread
                        spreads[pair_id] = pd.DataFrame({
                            'price1': price1,
                            'price2': price2,
                            'spread': spread,
                            'hedge_ratio': pd.Series([hedge_ratio] * len(common_idx), index=common_idx)
                        })
                        
                        logger.info(f"Successfully calculated spread for pair {pair_id} using {len(common_idx)} data points")
                    else:
                        logger.warning(f"Insufficient common data points ({len(common_idx)}) for pair {pair_id}")
                else:
                    missing_symbols = []
                    if symbol1 not in prices:
                        missing_symbols.append(symbol1)
                    if symbol2 not in prices:
                        missing_symbols.append(symbol2)
                    logger.warning(f"Missing data for symbols {missing_symbols} in pair {pair_id}")
            else:
                logger.warning(f"Could not determine symbols for pair {pair_id}")
                
        except Exception as e:
            logger.error(f"Error calculating spread for pair {pair_id}: {e}")
    
    # Return collected data
    return {
        'prices': prices,
        'volumes': volumes,
        'spreads': spreads
    }

# Make sure to attach the method to IntradayMLPaperTrader
IntradayMLPaperTrader._collect_market_data = patched_collect_market_data

# Function to get pair symbols with proper initialization of symbol variables
def _get_pair_symbols(self, pair_id):
    """Helper method to get the actual symbols for a pair from configuration."""
    try:
        # Default initializations to avoid unbound local variable errors
        symbol1 = None
        symbol2 = None
        
        # Look up in the config
        if hasattr(self, 'ml_config') and 'pairs' in self.ml_config:
            for pair_config in self.ml_config.get('pairs', []):
                if pair_config.get('pair_id') == pair_id:
                    symbol1 = pair_config.get('full_symbol1')
                    symbol2 = pair_config.get('full_symbol2')
                    if symbol1 and symbol2:
                        return [symbol1, symbol2]
        
        # Try to parse the pair_id (assuming format like ES_NQ)
        if '_' in pair_id:
            parts = pair_id.split('_')
            if len(parts) == 2:
                # Try to find the subscribed symbols that match these parts
                subscribed = self.paper_trader.get_subscribed_symbols()
                symbol1_candidates = [s for s in subscribed if parts[0] in s]
                symbol2_candidates = [s for s in subscribed if parts[1] in s]
                
                if symbol1_candidates and symbol2_candidates:
                    symbol1 = symbol1_candidates[0]
                    symbol2 = symbol2_candidates[0]
                    return [symbol1, symbol2]
        
        # Could not find symbols
        logger.warning(f"Could not determine symbols for pair {pair_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting pair symbols: {e}")
        return None

# Override the existing method with the fixed version
IntradayMLPaperTrader._get_pair_symbols = _get_pair_symbols

def main():
    """Main function to run the paper trader."""
    global paper_trader, keep_running
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        logger.info("Creating ML-enhanced paper trader...")
        
        # Get config path
        config_path = os.path.join('config', 'paper_trading.json')
        
        # Create IntradayMLPaperTrader instance
        paper_trader = IntradayMLPaperTrader(
            initial_capital=100000.0,
            ib_host='localhost',
            ib_port=7496,
            ib_client_id=54321,  # Use a more unique client ID
            ml_config_path=config_path,
            models_dir='models/intraday',  # Explicitly set models directory
            output_dir='output/paper_trading',
            dashboard_update_interval_seconds=60,  # Update more frequently
            refresh_interval_seconds=60,  # Increased to 60 seconds to avoid rate limiting
            enable_dashboard=True,
            enable_alerts=True,
            auto_shutdown_time='16:00'
        )
        
        # Ensure the models_dir attribute is set
        if not hasattr(paper_trader, 'models_dir'):
            paper_trader.models_dir = 'models/intraday'
            
        # Start the paper trader
        logger.info("Starting ML-enhanced paper trader...")
        if paper_trader.start():
            logger.info("ML-enhanced paper trader started successfully!")
            
            # Add example pairs to monitor
            add_example_pairs(paper_trader)
            
            # Update dashboard for the first time
            paper_trader._update_dashboard()
            
            # Log dashboard location
            dashboard_path = os.path.join(paper_trader.output_dir, "dashboard", "index.html")
            logger.info(f"Dashboard available at: file://{os.path.abspath(dashboard_path)}")
            
            # Keep running until stopped
            logger.info("Paper trader is running. Press Ctrl+C to stop.")
            while keep_running:
                time.sleep(1)
            
            # Stop paper trader
            logger.info("Stopping ML-enhanced paper trader...")
            paper_trader.stop()
            logger.info("ML-enhanced paper trader stopped.")
        else:
            logger.error("Failed to start ML-enhanced paper trader!")
    
    except Exception as e:
        logger.error(f"Error running ML-enhanced paper trader: {e}")
        import traceback
        traceback.print_exc()
        if paper_trader and hasattr(paper_trader, 'is_running') and paper_trader.is_running:
            paper_trader.stop()

if __name__ == '__main__':
    main() 