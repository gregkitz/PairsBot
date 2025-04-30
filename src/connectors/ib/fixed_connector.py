"""
Fixed version of the Interactive Brokers (IB) Connector for testing.

This module provides a high-level connector to Interactive Brokers for
market data retrieval and testing integration, with a fix for the API call
that was causing issues.
"""

import time
import logging
from typing import Dict, Any

from ib_insync import IB, Contract, util, Stock
from ib_insync import Future, Forex, Index, BarData

# Configure logging
logger = logging.getLogger(__name__)


class FixedIBConnector:
    """
    Fixed version of the IBConnector for testing integration with TWS.
    
    This class simplifies the IBConnector and fixes issues with the API calls.
    """
    
    def __init__(self,
                host: str = 'localhost',
                port: int = 7497,
                client_id: int = 1,
                timeout: int = 30,
                read_only: bool = True):
        """
        Initialize the IB connector.
        
        Parameters:
        -----------
        host : str
            IB TWS/Gateway hostname or IP address
        port : int
            IB TWS/Gateway port
        client_id : int
            Client ID for IB connection
        timeout : int
            Connection timeout in seconds
        read_only : bool
            If True, no orders will be placed (for testing)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.read_only = read_only
        
        # Create IB connection object
        self.ib = IB()
        
        # Connect flag
        self._is_connected = False
    
    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS/Gateway.
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        if self._is_connected:
            logger.warning("Already connected to IB")
            return True
        
        try:
            logger.info(f"Connecting to IB at {self.host}:{self.port} (client ID: {self.client_id})")
            logger.info("IMPORTANT: Check if TWS is displaying any permission dialogs")
            
            # Connect to IB with simplified parameters and shorter timeout
            self.ib.connect(
                self.host, 
                self.port,
                clientId=self.client_id,
                timeout=5,  # Use a shorter timeout to fail fast
                readonly=self.read_only
            )
            
            # Check if connected
            if not self.ib.isConnected():
                logger.error("Failed to connect to IB")
                return False
            
            # Verify connection is stable
            logger.info("Connected to TWS, waiting to verify stability...")
            self.ib.sleep(1)
            
            if not self.ib.isConnected():
                logger.error("Connection was unstable and disconnected")
                return False
            
            # Set connected flag
            self._is_connected = True
            logger.info("Successfully connected to IB")
            
            # Print a message about API synchronization
            logger.info("Note: If you see this but then disconnect, check for API permission dialogs in TWS")
            
            return True
                
        except Exception as e:
            logger.error(f"Error connecting to IB: {str(e)}")
            
            # Check for common errors
            if "timeout" in str(e).lower():
                logger.error("Timeout occurred. This often happens when TWS requires permission approval.")
                logger.error("Check for permission dialogs in the TWS application.")
            elif "connection reset" in str(e).lower():
                logger.error("Connection reset. This could mean TWS actively rejected the connection.")
                logger.error("Make sure API access is enabled in TWS: Edit > Global Configuration > API > Settings")
            
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Interactive Brokers TWS/Gateway."""
        if self._is_connected:
            logger.info("Disconnecting from IB")
            
            # Disconnect from IB
            self.ib.disconnect()
            
            # Set connected flag
            self._is_connected = False
            logger.info("Disconnected from IB")
    
    def is_connected(self) -> bool:
        """
        Check if connected to Interactive Brokers.
        
        Returns:
        --------
        bool
            True if connected, False otherwise
        """
        return self._is_connected and self.ib.isConnected()
    
    def get_server_time(self):
        """
        Get the current server time from TWS.
        
        Returns:
        --------
        datetime
            Current server time
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            return self.ib.reqCurrentTime()
        except Exception as e:
            logger.error(f"Error getting server time: {str(e)}")
            return None
    
    def create_contract(self, symbol: str, sec_type: str = 'STK', 
                      exchange: str = 'SMART', currency: str = 'USD',
                      expiry: str = '', strike: float = 0.0, right: str = '') -> Contract:
        """
        Create a contract object for TWS.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the instrument
        sec_type : str
            Security type (STK, FUT, OPT, IND, CASH, etc.)
        exchange : str
            Exchange name
        currency : str
            Currency
        expiry : str
            Expiration date (for futures and options)
        strike : float
            Strike price (for options)
        right : str
            Option right (C or P)
            
        Returns:
        --------
        Contract
            Contract object
        """
        try:
            if sec_type == 'STK':
                contract = Stock(symbol, exchange, currency)
            elif sec_type == 'FUT':
                if expiry.lower() == 'front':
                    # Get front month contract
                    contracts = self.ib.reqContractDetails(
                        Future(symbol=symbol, exchange=exchange, currency=currency)
                    )
                    if not contracts:
                        return None
                    # Sort by expiry date and get the first one
                    contracts.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)
                    contract = contracts[0].contract
                else:
                    contract = Future(symbol=symbol, exchange=exchange, 
                                     currency=currency, lastTradeDateOrContractMonth=expiry)
            elif sec_type == 'CASH':
                # For currencies, symbol should be like 'EUR.USD'
                base, quote = symbol.split('.')
                contract = Forex(pair=f'{base}{quote}')
            elif sec_type == 'IND':
                contract = Index(symbol, exchange, currency)
            else:
                # Generic contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = sec_type
                contract.exchange = exchange
                contract.currency = currency
                if expiry:
                    contract.lastTradeDateOrContractMonth = expiry
                if strike:
                    contract.strike = strike
                if right:
                    contract.right = right
                    
            return contract
        
        except Exception as e:
            logger.error(f"Error creating contract: {str(e)}")
            return None
    
    def get_market_data(self, symbol: str, sec_type: str = 'STK', 
                      exchange: str = 'SMART', currency: str = 'USD',
                      timeout: float = 5.0) -> Dict[str, Any]:
        """
        Get real-time market data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the instrument
        sec_type : str
            Security type (STK, FUT, OPT, IND, CASH, etc.)
        exchange : str
            Exchange name
        currency : str
            Currency
        timeout : float
            Timeout in seconds to wait for market data
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with market data
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            # Create contract
            contract = self.create_contract(symbol, sec_type, exchange, currency)
            if not contract:
                logger.error(f"Could not create contract for {symbol}")
                return None
            
            # Request market data
            ticker = self.ib.reqMktData(contract)
            
            # Wait for data to arrive
            timeout_time = time.time() + timeout
            while time.time() < timeout_time:
                self.ib.sleep(0.1)
                if ticker.last or ticker.bid or ticker.ask:
                    break
            
            # Create result dictionary
            result = {
                'symbol': symbol,
                'last': ticker.last,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'close': ticker.close,
                'volume': ticker.volume,
                'time': ticker.time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None
    
    def get_historical_data(self, symbol: str, duration: str = '1 D',
                          bar_size: str = '1 min', what_to_show: str = 'TRADES',
                          sec_type: str = 'STK', exchange: str = 'SMART', 
                          currency: str = 'USD') -> Any:
        """
        Get historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the instrument
        duration : str
            Time duration to go back
        bar_size : str
            Size of each bar
        what_to_show : str
            Type of data to retrieve
        sec_type : str
            Security type
        exchange : str
            Exchange name
        currency : str
            Currency
            
        Returns:
        --------
        DataFrame
            DataFrame with historical data
        """
        if not self.is_connected():
            logger.error("Not connected to IB")
            return None
        
        try:
            # Create contract
            contract = self.create_contract(symbol, sec_type, exchange, currency)
            if not contract:
                logger.error(f"Could not create contract for {symbol}")
                return None
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            # Convert to dataframe
            df = util.df(bars)
            if df is not None and not df.empty:
                df['symbol'] = symbol
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def trigger_api_permissions(self) -> None:
        """
        Attempt to trigger API permission dialogs in TWS.
        
        This method makes various API calls that typically require permission,
        which can help identify if permissions are the issue.
        """
        logger.info("Attempting to trigger API permission dialogs...")
        
        try:
            # 1. Request market data (requires market data permissions)
            logger.info("1. Requesting AAPL market data...")
            contract = Stock('AAPL', 'SMART', 'USD')
            self.ib.reqMktData(contract)
            self.ib.sleep(1)
            
            # 2. Request account summary (requires account permissions)
            logger.info("2. Requesting account summary...")
            self.ib.reqAccountSummary(1, 'All', 'AccountType,NetLiquidation')
            self.ib.sleep(1)
            
            # 3. Request managed accounts (requires basic API permissions)
            logger.info("3. Requesting managed accounts...")
            accounts = self.ib.managedAccounts()
            logger.info(f"Managed accounts: {accounts}")
            
            # If we get here, permissions are likely granted
            logger.info("API calls completed without timeout, permissions may be properly set")
            
        except Exception as e:
            logger.error(f"Error while triggering permissions: {str(e)}")
            logger.error("If TWS showed permission dialogs, approve them and try connecting again") 