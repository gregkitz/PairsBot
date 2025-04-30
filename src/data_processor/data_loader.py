import pandas as pd
import numpy as np
import requests
import os
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Tuple

# Import IBConnector
from src.connectors.ib.ib_connector import IBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and downloading financial data from various sources including Interactive Brokers
    """
    def __init__(self, data_dir: str = "data/historical", api_key: Optional[str] = None,
                 ib_host: str = "127.0.0.1", ib_port: int = 7497, ib_client_id: int = 1):
        """
        Initialize the DataLoader
        
        Parameters:
        -----------
        data_dir : str
            Directory to store historical data
        api_key : str, optional
            API key for data provider (if needed)
        ib_host : str
            Interactive Brokers TWS/Gateway hostname or IP
        ib_port : int
            Interactive Brokers TWS/Gateway port
        ib_client_id : int
            Interactive Brokers client ID
        """
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # IB connection parameters
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self.ib_connector = None
        
    def _get_ib_connector(self) -> IBConnector:
        """
        Get an initialized IBConnector instance (creating one if needed)
        
        Returns:
        --------
        IBConnector
            Connected Interactive Brokers connector
        """
        if self.ib_connector is None or not self.ib_connector.is_connected():
            # Create and connect to IB
            self.ib_connector = IBConnector(
                host=self.ib_host,
                port=self.ib_port,
                client_id=self.ib_client_id,
                read_only=True  # Read-only for data collection
            )
            
            # Try to connect to IB
            connected = self.ib_connector.connect()
            if not connected:
                logger.error("Failed to connect to Interactive Brokers")
            else:
                logger.info(f"Connected to Interactive Brokers at {self.ib_host}:{self.ib_port}")
                
        return self.ib_connector
        
    def load_data(self, symbol: str, timeframe: str = "1day", start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from local storage or download if not available
        
        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        timeframe : str
            Timeframe of the data ('1min', '5min', '1hour', '1day')
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with the requested data
        """
        # Create path for the symbol and timeframe
        symbol_dir = self.data_dir / symbol / timeframe
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = symbol_dir / "data.csv"
        
        # Check if data file exists
        if data_file.exists():
            logger.info(f"Loading {symbol} {timeframe} data from {data_file}")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            
            # Filter by date range if provided
            if start_date or end_date:
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                    
            return df
        else:
            logger.info(f"Data file for {symbol} {timeframe} not found, downloading...")
            df = self.download_data(symbol, timeframe, start_date, end_date)
            
            if df is not None and not df.empty:
                # Save to CSV
                df.to_csv(data_file)
                return df
            else:
                logger.error(f"Failed to download data for {symbol} {timeframe}")
                return pd.DataFrame()
    
    def download_data(self, symbol: str, timeframe: str = "1day", 
                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download data from Interactive Brokers
        
        Parameters:
        -----------
        symbol : str
            Symbol to download data for
        timeframe : str
            Timeframe of the data ('1min', '5min', '1hour', '1day')
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with the downloaded data
        """
        logger.info(f"Downloading {symbol} {timeframe} data from Interactive Brokers...")
        
        try:
            # Get IB connector
            ib = self._get_ib_connector()
            
            if not ib.is_connected():
                logger.error("Not connected to Interactive Brokers")
                # Fall back to synthetic data for testing
                return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
            # Convert start_date and end_date strings to datetime objects
            start_dt = None
            end_dt = None
            
            if start_date:
                start_dt = pd.Timestamp(start_date).to_pydatetime()
            
            if end_date:
                end_dt = pd.Timestamp(end_date).to_pydatetime()
            else:
                end_dt = datetime.now()
            
            # Map timeframe to IB bar size
            bar_size_map = {
                "1min": "1 min",
                "5min": "5 mins",
                "1hour": "1 hour",
                "1day": "1 day"
            }
            
            bar_size = bar_size_map.get(timeframe, "1 day")
            
            # Request historical data from IB
            df = ib.get_historical_data(
                symbol=symbol,
                start=start_dt,
                end=end_dt,
                bar_size=bar_size,
                what_to_show='TRADES',
                use_rth=True
            )
            
            # if df.empty:
            #     logger.warning(f"No data returned from IB for {symbol} {timeframe}, falling back to synthetic data")
            #     return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
            
            logger.info(f"Successfully downloaded {len(df)} rows for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol} {timeframe} from IB: {str(e)}")
            logger.info("Falling back to synthetic data")
            return self._generate_synthetic_data(symbol, timeframe, start_date, end_date)
    
    # def _generate_synthetic_data(self, symbol: str, timeframe: str = "1day",
    #                            start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    #     """
    #     Generate synthetic data for testing purposes
    #
    #     Parameters:
    #     -----------
    #     symbol : str
    #         Symbol to generate data for
    #     timeframe : str
    #         Timeframe of the data ('1min', '5min', '1hour', '1day')
    #     start_date : str, optional
    #         Start date in YYYY-MM-DD format
    #     end_date : str, optional
    #         End date in YYYY-MM-DD format
    #
    #     Returns:
    #     --------
    #     pd.DataFrame
    #         DataFrame with synthetic data
    #     """
    #     logger.info(f"Generating synthetic data for {symbol} {timeframe}")
    #
    #     # Create mock data for testing
    #     if not start_date:
    #         start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    #     if not end_date:
    #         end_date = datetime.now().strftime("%Y-%m-%d")
    #
    #     start = pd.Timestamp(start_date)
    #     end = pd.Timestamp(end_date)
    #
    #     # Generate different time indices based on timeframe
    #     if timeframe == "1min":
    #         idx = pd.date_range(start=start, end=end, freq="1min")
    #     elif timeframe == "5min":
    #         idx = pd.date_range(start=start, end=end, freq="5min")
    #     elif timeframe == "1hour":
    #         idx = pd.date_range(start=start, end=end, freq="1H")
    #     else:  # Default to daily
    #         idx = pd.date_range(start=start, end=end, freq="1D")
    #
    #     # Filter for trading hours (9:30 AM to 4:00 PM Eastern, weekdays)
    #     if timeframe in ["1min", "5min", "1hour"]:
    #         idx = idx[
    #             (idx.hour >= 9) &
    #             (idx.hour < 16) &
    #             (idx.weekday < 5)
    #         ]
    #
    #         # Adjust for market open at 9:30
    #         idx = idx[~((idx.hour == 9) & (idx.minute < 30))]
    #     else:
    #         # For daily data, just keep weekdays
    #         idx = idx[idx.weekday < 5]
    #
    #     # Create mock price data
    #     np.random.seed(sum(ord(c) for c in symbol))  # Seed based on symbol for reproducibility
    #
    #     # Start with a base price that's different for each symbol
    #     base_price = 100 + (sum(ord(c) for c in symbol) % 900)
    #
    #     # Create price series with random walk
    #     prices = np.cumprod(
    #         1 + np.random.normal(0.0002, 0.001, size=len(idx))
    #     ) * base_price
    #
    #     # Create OHLCV data
    #     data = pd.DataFrame(index=idx)
    #     data['open'] = prices
    #     data['high'] = prices * (1 + np.random.uniform(0, 0.005, size=len(idx)))
    #     data['low'] = prices * (1 - np.random.uniform(0, 0.005, size=len(idx)))
    #     data['close'] = prices * (1 + np.random.normal(0, 0.001, size=len(idx)))
    #     data['volume'] = np.random.randint(1000, 1000000, size=len(idx))
    #
    #     # Ensure high/low are actually highest/lowest
    #     data['high'] = np.maximum(
    #         np.maximum(data['high'], data['open']),
    #         data['close']
    #     )
    #     data['low'] = np.minimum(
    #         np.minimum(data['low'], data['open']),
    #         data['close']
    #     )
    #
    #     logger.info(f"Successfully generated {len(data)} rows of synthetic data for {symbol} {timeframe}")
    #     return data
    
    def __del__(self):
        """
        Clean up IB connection when the object is destroyed
        """
        if hasattr(self, 'ib_connector') and self.ib_connector is not None:
            if self.ib_connector.is_connected():
                self.ib_connector.disconnect()


def download_historical_data(symbols: Union[str, List[str]], 
                            timeframes: Union[str, List[str]] = "1day",
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            data_dir: str = "data/historical",
                            api_key: Optional[str] = None,
                            parallel: bool = True,
                            ib_host: str = "127.0.0.1",
                            ib_port: int = 7497,
                            client_id_base: int = 1000,
                            client_id_spread: int = 1) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Download historical data for multiple symbols and timeframes
    
    Parameters:
    -----------
    symbols : str or List[str]
        Symbol(s) to download data for
    timeframes : str or List[str]
        Timeframe(s) of the data ('1min', '5min', '1hour', '1day')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    data_dir : str
        Directory to store historical data
    api_key : str, optional
        API key for data provider (if needed)
    parallel : bool
        Whether to download data in parallel (using multiple threads)
    ib_host : str
        Interactive Brokers TWS/Gateway hostname or IP
    ib_port : int
        Interactive Brokers TWS/Gateway port
    client_id_base : int
        Base client ID for IB connections (will increment for parallel connections)
    client_id_spread : int
        Spread between client IDs to avoid conflicts in parallel processing
        
    Returns:
    --------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary of dictionaries with downloaded data by symbol and timeframe
    """
    # Convert to lists if not already
    if isinstance(symbols, str):
        symbols = [symbols]
    if isinstance(timeframes, str):
        timeframes = [timeframes]
    
    # Create a DataLoader for each symbol/timeframe combination
    loaders = []
    client_id = client_id_base
    
    for symbol in symbols:
        for tf in timeframes:
            # Create a new DataLoader with incremented client_id
            loader = DataLoader(
                data_dir=data_dir,
                api_key=api_key,
                ib_host=ib_host,
                ib_port=ib_port,
                ib_client_id=client_id
            )
            
            loaders.append((symbol, tf, loader))
            # Increment client_id by spread to avoid conflicts
            client_id += client_id_spread
    
    logger.info(f"Downloading data for {len(symbols)} symbols and {len(timeframes)} timeframes {'in parallel' if parallel else 'sequentially'}")
    
    # Dictionary to hold results
    results = {symbol: {} for symbol in symbols}
    
    if parallel and len(loaders) > 1:
        # Download data in parallel using concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(loaders), 10)) as executor:
            # Submit download tasks
            future_to_loader = {
                executor.submit(
                    loader.download_data, 
                    symbol, 
                    tf, 
                    start_date, 
                    end_date
                ): (symbol, tf) for symbol, tf, loader in loaders
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_loader):
                symbol, tf = future_to_loader[future]
                try:
                    data = future.result()
                    results[symbol][tf] = data
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {tf} data: {str(e)}")
                    results[symbol][tf] = pd.DataFrame()  # Empty DataFrame on error
    else:
        # Download data sequentially
        for symbol, tf, loader in loaders:
            try:
                data = loader.download_data(symbol, tf, start_date, end_date)
                results[symbol][tf] = data
            except Exception as e:
                logger.error(f"Error downloading {symbol} {tf} data: {str(e)}")
                results[symbol][tf] = pd.DataFrame()  # Empty DataFrame on error
    
    return results


def load_multiple_timeframes(symbol: str, 
                            timeframes: List[str],
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None,
                            data_dir: str = "data/historical") -> Dict[str, pd.DataFrame]:
    """
    Load data for a symbol across multiple timeframes
    
    Parameters:
    -----------
    symbol : str
        Symbol to load data for
    timeframes : List[str]
        List of timeframes to load ('1min', '5min', '1hour', '1day')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    data_dir : str
        Directory to load historical data from
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframe as key and DataFrame as value
    """
    loader = DataLoader(data_dir=data_dir)
    results = {}
    
    for tf in timeframes:
        data = loader.load_data(symbol, tf, start_date, end_date)
        if data is not None and not data.empty:
            results[tf] = data
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument("--symbols", required=True, help="Comma-separated list of symbols")
    parser.add_argument("--timeframes", default="1day", help="Comma-separated list of timeframes (1min,5min,1hour,1day)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default="data/historical", help="Data directory")
    parser.add_argument("--api-key", help="API key for data provider")
    parser.add_argument("--sequential", action="store_true", help="Download sequentially instead of in parallel")
    parser.add_argument("--ib-host", default="127.0.0.1", help="Interactive Brokers TWS/Gateway hostname or IP")
    parser.add_argument("--ib-port", type=int, default=7497, help="Interactive Brokers TWS/Gateway port")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    timeframes = args.timeframes.split(",")
    
    download_historical_data(
        symbols=symbols,
        timeframes=timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        api_key=args.api_key,
        parallel=not args.sequential,
        ib_host=args.ib_host,
        ib_port=args.ib_port
    ) 