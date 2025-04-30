#!/usr/bin/env python
"""
Interactive Brokers Data Download Script for Intraday ML Trading System

This script downloads and updates historical market data for all configured trading pairs
and instruments. It's designed to be run daily after market close as part of the
automated data pipeline.

The script will:
1. Connect to Interactive Brokers TWS/Gateway
2. Download historical data for configured symbols and timeframes
3. Store data in the standard data directories
4. Generate log information about the process

NOTE: NinjaTrader Integration
For more reliable data, consider integrating with NinjaTrader which offers:
- More stable API access
- Better historical data retrieval
- Simpler contract specifications
To integrate with NinjaTrader, you would need to:
1. Create a NinjaTrader connection class similar to IBConnector
2. Implement the data downloading via NinjaTrader's API or exported files
3. Update this script to use the NinjaTrader connector
"""

import os
import sys
import time
import logging
import argparse
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yaml

# Set up the path to include the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs", "data_download.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_download")

# Import the required modules
try:
    from src.data_processor.data_loader import download_historical_data
    from src.connectors.ib.ib_connector import IBConnector
    from ib_insync import Contract, Future, Stock, Forex, Index
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    sys.exit(1)

def generate_unique_client_id(base=1000):
    """
    Generate a unique client ID based on current timestamp and random number
    
    Parameters:
    -----------
    base : int
        Base client ID value
        
    Returns:
    --------
    int
        Unique client ID
    """
    # Use milliseconds since epoch + random number to ensure uniqueness
    timestamp = int(time.time() * 1000) % 10000  # Last 4 digits of current timestamp in ms
    random_num = random.randint(0, 999)  # Random 3-digit number
    
    # Combine base + timestamp + random to create unique ID
    unique_id = base + timestamp + random_num
    
    # Ensure it's within the acceptable range (must be positive and < 1,000,000,000)
    unique_id = abs(unique_id) % 999999999
    
    return unique_id

def load_config(config_path=None):
    """
    Load the configuration for data download
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Default config path if not specified
    if not config_path:
        config_path = os.path.join(project_root, "config", "data_config.yaml")
    
    # Check if config exists
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default settings")
        return {
            "data_dir": os.path.join(project_root, "data", "historical"),
            "ib_host": "127.0.0.1",
            "ib_port": 7497,  # Default to paper trading port
            "client_id_base": 10000,  # Use a higher base client ID to avoid conflicts
            "symbols": ["ES", "NQ", "CL", "GC"],  # Default symbols
            "timeframes": ["1min", "5min", "1hour", "1day"],  # Default timeframes
            "days_to_download": 5,  # Default days to download
            "use_synthetic_data": False,  # Use synthetic data if IB fails
            "contracts": {
                "ES": {"symbol": "ES", "secType": "FUT", "exchange": "CME", "currency": "USD"},
                "NQ": {"symbol": "NQ", "secType": "FUT", "exchange": "CME", "currency": "USD"},
                "CL": {"symbol": "CL", "secType": "FUT", "exchange": "NYMEX", "currency": "USD"},
                "GC": {"symbol": "GC", "secType": "FUT", "exchange": "COMEX", "currency": "USD"}
            }
        }
    
    # Load config using direct YAML loading
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            data_config = config.get("data_download", {})
        
        logger.info(f"Loaded configuration from {config_path}")
        return data_config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def test_ib_connection(host, port, client_id):
    """
    Test the connection to Interactive Brokers
    
    Parameters:
    -----------
    host : str
        IB TWS/Gateway hostname or IP
    port : int
        IB TWS/Gateway port
    client_id : int
        Client ID for IB connection
        
    Returns:
    --------
    bool
        True if connected successfully, False otherwise
    """
    logger.info(f"Testing connection to Interactive Brokers at {host}:{port} with client ID {client_id}")
    
    try:
        # Create an IBConnector instance
        ib = IBConnector(
            host=host,
            port=port,
            client_id=client_id,
            read_only=True
        )
        
        # Try to connect
        connected = ib.connect()
        
        if connected:
            logger.info("Successfully connected to Interactive Brokers")
            # Disconnect
            ib.disconnect()
            return True
        else:
            logger.error("Failed to connect to Interactive Brokers")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Interactive Brokers: {str(e)}")
        return False

def create_contract(symbol, contract_specs=None):
    """
    Create an IB contract for the given symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol to create contract for
    contract_specs : dict, optional
        Contract specifications (symbol, secType, exchange, currency, etc.)
        
    Returns:
    --------
    Contract
        IB contract object
    """
    if not contract_specs:
        # Default to stock
        return Stock(symbol, 'SMART', 'USD')
    
    sec_type = contract_specs.get('secType', 'STK')
    
    if sec_type == 'FUT':
        # For futures, we need to find the active contract
        symbol = contract_specs.get('symbol', symbol)
        exchange = contract_specs.get('exchange', 'CME')
        currency = contract_specs.get('currency', 'USD')
        expiry = contract_specs.get('expiry', '')
        
        # Create a Future contract
        contract = Future(symbol, expiry, exchange)
        contract.currency = currency
        
        # If no expiry is specified, this will request the front month contract
        return contract
    
    elif sec_type == 'STK':
        # For stocks
        exchange = contract_specs.get('exchange', 'SMART')
        currency = contract_specs.get('currency', 'USD')
        return Stock(symbol, exchange, currency)
    
    elif sec_type == 'CASH':
        # For forex
        base = contract_specs.get('base', symbol[:3])
        quote = contract_specs.get('quote', symbol[3:])
        return Forex(base, quote)
    
    elif sec_type == 'IND':
        # For indices
        exchange = contract_specs.get('exchange', 'SMART')
        currency = contract_specs.get('currency', 'USD')
        return Index(symbol, exchange, currency)
    
    else:
        # Generic contract
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = contract_specs.get('exchange', 'SMART')
        contract.currency = contract_specs.get('currency', 'USD')
        return contract

def generate_synthetic_data(symbol, timeframe, start_date, end_date, volatility=None):
    """
    Generate synthetic market data for testing
    
    Parameters:
    -----------
    symbol : str
        Symbol to generate data for
    timeframe : str
        Timeframe of the data (1min, 5min, etc.)
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    volatility : float, optional
        Volatility to use for price generation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with synthetic OHLCV data
    """
    logger.info(f"Generating synthetic data for {symbol} at {timeframe} timeframe")
    
    # Set random seed based on symbol for consistent data
    np.random.seed(hash(symbol) % 2**32)
    
    # Create date range based on timeframe
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    if timeframe == '1min':
        # Trading hours only (6:30 AM to 1:15 PM Chicago time)
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Monday to Friday
                for hour in range(6, 14):
                    minute_limit = 60
                    if hour == 13:
                        minute_limit = 15  # Only until 1:15 PM
                    for minute in range(0, minute_limit):
                        if hour == 6 and minute < 30:
                            continue  # Start at 6:30 AM
                        dates.append(current_dt.replace(hour=hour, minute=minute))
            current_dt += timedelta(days=1)
        
        freq = None
    elif timeframe == '5min':
        # Trading hours only, 5-minute intervals
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Monday to Friday
                for hour in range(6, 14):
                    minute_limit = 60
                    if hour == 13:
                        minute_limit = 15  # Only until 1:15 PM
                    for minute in range(0, minute_limit, 5):
                        if hour == 6 and minute < 30:
                            continue  # Start at 6:30 AM
                        dates.append(current_dt.replace(hour=hour, minute=minute))
            current_dt += timedelta(days=1)
        
        freq = None
    elif timeframe == '15min':
        # Trading hours only, 15-minute intervals
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Monday to Friday
                for hour in range(6, 14):
                    minute_limit = 60
                    if hour == 13:
                        minute_limit = 15  # Only until 1:15 PM
                    for minute in range(0, minute_limit, 15):
                        if hour == 6 and minute < 30:
                            continue  # Start at 6:30 AM
                        dates.append(current_dt.replace(hour=hour, minute=minute))
            current_dt += timedelta(days=1)
        
        freq = None
    elif timeframe == '1hour':
        # Trading hours only, hourly intervals
        dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Monday to Friday
                for hour in range(7, 14):  # 7 AM to 1 PM
                    dates.append(current_dt.replace(hour=hour, minute=0))
            current_dt += timedelta(days=1)
        
        freq = None
    elif timeframe == '1day':
        # Business days only
        dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        freq = 'B'
    else:
        # Default to daily
        dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        freq = 'B'
    
    # Set initial price and volatility based on symbol
    if symbol == 'ES':
        initial_price = 5000.0
        vol = volatility or 0.008
    elif symbol == 'NQ':
        initial_price = 15000.0
        vol = volatility or 0.01
    elif symbol == 'YM':
        initial_price = 35000.0
        vol = volatility or 0.007
    elif symbol == 'RTY':
        initial_price = 2000.0
        vol = volatility or 0.009
    elif symbol == 'CL':
        initial_price = 75.0
        vol = volatility or 0.015
    elif symbol == 'GC':
        initial_price = 2000.0
        vol = volatility or 0.005
    else:
        initial_price = 100.0
        vol = volatility or 0.01
    
    # Generate prices using geometric Brownian motion
    n = len(dates)
    dt = 1.0
    mu = 0.0002  # Drift term
    
    # Generate random steps
    W = np.random.standard_normal(size=n)
    W = np.cumsum(W)*np.sqrt(dt)  # Brownian motion
    
    # Generate prices
    X = (mu-0.5*vol**2)*dt + vol*W
    S = initial_price*np.exp(X)  # Geometric Brownian motion
    
    # Generate OHLC data with some randomness
    data = []
    for i in range(n):
        if i == 0:
            # First bar
            open_price = initial_price
        else:
            # Use previous close as next open
            open_price = data[i-1]['close']
        
        close = S[i]
        
        # Add some randomness to high and low
        intrabar_vol = vol * np.random.uniform(0.5, 1.5)
        high = max(open_price, close) * (1 + intrabar_vol * np.random.random())
        low = min(open_price, close) * (1 - intrabar_vol * np.random.random())
        
        # Generate volume with some randomness
        volume = int(np.random.lognormal(10, 1) * (1 + 0.1 * np.random.randn()))
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    logger.info(f"Generated {len(df)} rows of synthetic data for {symbol} {timeframe}")
    return df

def download_data(config):
    """
    Download data based on configuration
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Extract configuration
        data_dir = config.get("data_dir", os.path.join(project_root, "data", "historical"))
        ib_host = config.get("ib_host", "127.0.0.1")
        ib_port = config.get("ib_port", 7497)
        client_id_base = config.get("client_id_base", 10000)
        symbols = config.get("symbols", ["ES", "NQ", "CL", "GC"])
        timeframes = config.get("timeframes", ["1min", "5min", "1hour", "1day"])
        days_to_download = config.get("days_to_download", 5)
        contracts = config.get("contracts", {})
        use_synthetic = config.get("use_synthetic_data", False)
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_to_download)).strftime("%Y-%m-%d")
        
        # Log the download parameters
        logger.info(f"Downloading data for {symbols} with timeframes {timeframes}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Dictionary to hold results
        results = {symbol: {} for symbol in symbols}
        success = True
        
        # Try to use IB first
        ib_success = False
        try:
            # Generate a unique client ID for this run
            unique_client_id = generate_unique_client_id(client_id_base)
            logger.info(f"Using unique client ID: {unique_client_id}")
            logger.info(f"Attempting to use Interactive Brokers for data")
        
            # Create IBConnector
            ib = IBConnector(
                host=ib_host,
                port=ib_port,
                client_id=unique_client_id,
                read_only=True
            )
            
            # Try to connect
            connected = ib.connect()
            if connected:
                logger.info("Successfully connected to Interactive Brokers")
                
                try:
                    # For each symbol and timeframe, download data using the same connection
                    for symbol in symbols:
                        # Get contract specs if available
                        contract_specs = contracts.get(symbol, None)
                        logger.info(f"Processing {symbol} with contract specs: {contract_specs}")
                        
                        # For futures, we need to qualify the contract first
                        qualified_contract = None
                        if contract_specs and contract_specs.get('secType', '') == 'FUT':
                            # Create a basic future contract
                            future_symbol = contract_specs.get('symbol', symbol)
                            exchange = contract_specs.get('exchange', 'CME')
                            currency = contract_specs.get('currency', 'USD')
                            
                            contract = Future(future_symbol, '', exchange)
                            contract.currency = currency
                            
                            # First, get the front-month contract
                            logger.info(f"Qualifying future contract for {symbol}...")
                            try:
                                # Use reqContractDetails to get details for this contract
                                contract_details = ib.ib.reqContractDetails(contract)
                                
                                if contract_details:
                                    # Sort by expiry date and get the front month
                                    contract_details.sort(key=lambda x: x.contractDetails.contract.lastTradeDateOrContractMonth)
                                    qualified_contract = contract_details[0].contract
                                    logger.info(f"Found front month contract for {symbol}: {qualified_contract.localSymbol}")
                                else:
                                    logger.error(f"No contract details found for {symbol}")
                            except Exception as e:
                                logger.error(f"Error qualifying contract for {symbol}: {str(e)}")
                        
                        for tf in timeframes:
                            logger.info(f"Downloading {symbol} {tf} data...")
                            
                            # Download data
                            try:
                                # Map timeframe to IB bar size
                                bar_size_map = {
                                    "1min": "1 min",
                                    "5min": "5 mins",
                                    "15min": "15 mins",
                                    "1hour": "1 hour",
                                    "1day": "1 day"
                                }
                                
                                bar_size = bar_size_map.get(tf, "1 day")
                                
                                # Use the qualified contract if available, otherwise create a new one
                                if qualified_contract:
                                    contract = qualified_contract
                                else:
                                    # Create a basic contract (non-futures)
                                    contract = create_contract(symbol, contract_specs)
                                
                                # Request historical data directly using ib_insync
                                bars = ib.ib.reqHistoricalData(
                                    contract=contract,
                                    endDateTime=pd.Timestamp(end_date).to_pydatetime(),
                                    durationStr=f"{days_to_download} D",
                                    barSizeSetting=bar_size,
                                    whatToShow='TRADES',
                                    useRTH=True,
                                    formatDate=1
                                )
                                
                                if bars:
                                    # Convert to DataFrame
                                    from ib_insync import util
                                    data = util.df(bars)
                                    
                                    # Rename columns to match expected format
                                    data = data.rename(columns={
                                        'date': 'date',
                                        'open': 'open',
                                        'high': 'high',
                                        'low': 'low',
                                        'close': 'close',
                                        'volume': 'volume'
                                    })
                                    
                                    # Set date as index
                                    data.set_index('date', inplace=True)
                                    
                                    results[symbol][tf] = data
                                    
                                    logger.info(f"Successfully downloaded {len(data)} rows for {symbol} {tf}")
                                    
                                    # Save data to CSV
                                    symbol_dir = Path(data_dir) / symbol / tf
                                    symbol_dir.mkdir(parents=True, exist_ok=True)
                                    data.to_csv(symbol_dir / "data.csv")
                                else:
                                    logger.error(f"No data returned for {symbol} {tf} from IB")
                                    if use_synthetic:
                                        logger.info(f"Generating synthetic data for {symbol} {tf}")
                                        data = generate_synthetic_data(symbol, tf, start_date, end_date)
                                        results[symbol][tf] = data
                                        
                                        # Save synthetic data to CSV
                                        symbol_dir = Path(data_dir) / symbol / tf
                                        symbol_dir.mkdir(parents=True, exist_ok=True)
                                        data.to_csv(symbol_dir / "data.csv")
                                    else:
                                        results[symbol][tf] = pd.DataFrame()
                                        success = False
                            except Exception as e:
                                logger.error(f"Error downloading {symbol} {tf} data from IB: {str(e)}")
                                if use_synthetic:
                                    logger.info(f"Generating synthetic data for {symbol} {tf}")
                                    data = generate_synthetic_data(symbol, tf, start_date, end_date)
                                    results[symbol][tf] = data
                                    
                                    # Save synthetic data to CSV
                                    symbol_dir = Path(data_dir) / symbol / tf
                                    symbol_dir.mkdir(parents=True, exist_ok=True)
                                    data.to_csv(symbol_dir / "data.csv")
                                else:
                                    results[symbol][tf] = pd.DataFrame()  # Empty DataFrame on error
                                    success = False
                    
                    ib_success = True
                finally:
                    # Always disconnect when done
                    logger.info("Disconnecting from Interactive Brokers")
                    ib.disconnect()
            else:
                logger.error("Failed to connect to Interactive Brokers")
        except Exception as e:
            logger.error(f"Error using Interactive Brokers: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # If IB failed and we want to use synthetic data, generate it
        if not ib_success and use_synthetic:
            logger.info("Using synthetic data generation as fallback")
            
            for symbol in symbols:
                for tf in timeframes:
                    try:
                        logger.info(f"Generating synthetic data for {symbol} {tf}")
                        data = generate_synthetic_data(symbol, tf, start_date, end_date)
                        results[symbol][tf] = data
                        
                        # Save synthetic data to CSV
                        symbol_dir = Path(data_dir) / symbol / tf
                        symbol_dir.mkdir(parents=True, exist_ok=True)
                        data.to_csv(symbol_dir / "data.csv")
                        
                        logger.info(f"Saved synthetic data for {symbol} {tf} with {len(data)} rows")
                    except Exception as e:
                        logger.error(f"Error generating synthetic data for {symbol} {tf}: {str(e)}")
                        results[symbol][tf] = pd.DataFrame()
                        success = False
        
        if success:
            logger.info("Data download/generation completed successfully")
        else:
            logger.warning("Data download/generation completed with some errors")
            
        return success
        
    except Exception as e:
        logger.error(f"Error in data download process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function to handle data download process"""
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--port", type=int, help="IB TWS/Gateway port (overrides config)")
    parser.add_argument("--days", type=int, help="Number of days to download (overrides config)")
    parser.add_argument("--symbols", help="Comma-separated list of symbols (overrides config)")
    parser.add_argument("--client-id", type=int, help="Specify a custom client ID base")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic data")
    parser.add_argument("--no-synthetic", action="store_true", help="Disable synthetic data fallback")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.port:
        config["ib_port"] = args.port
        logger.info(f"Using port {args.port} from command line")
        
    if args.days:
        config["days_to_download"] = args.days
        logger.info(f"Using {args.days} days from command line")
        
    if args.symbols:
        config["symbols"] = args.symbols.split(",")
        logger.info(f"Using symbols {config['symbols']} from command line")
        
    if args.client_id:
        config["client_id_base"] = args.client_id
        logger.info(f"Using client ID base {args.client_id} from command line")
    
    if args.synthetic:
        config["use_synthetic_data"] = False
        logger.info("Forcing use of synthetic data")
    
    if args.no_synthetic:
        config["use_synthetic_data"] = False
        logger.info("Disabling synthetic data fallback")
    
    # Create data directory if it doesn't exist
    data_dir = config.get("data_dir", os.path.join(project_root, "data", "historical"))
    os.makedirs(data_dir, exist_ok=True)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Download the data
    start_time = time.time()
    success = download_data(config)
    elapsed_time = time.time() - start_time
    
    # Log the results
    if success:
        logger.info(f"Data process completed successfully in {elapsed_time:.2f} seconds")
        return 0
    else:
        logger.error(f"Data process completed with errors after {elapsed_time:.2f} seconds")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 