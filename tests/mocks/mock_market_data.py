"""
Mock Market Data Provider for testing purposes.

This module provides a mock implementation of a market data provider
for unit and integration testing without requiring real market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

class MockMarketDataProvider:
    """
    Mock implementation of a market data provider for testing purposes.
    
    This class simulates a market data provider that can generate synthetic
    price data or load test data from files.
    """
    
    def __init__(self):
        """Initialize the mock market data provider."""
        self.data = {}  # Cache for loaded or generated data
    
    def load_test_data(self, symbol: str, test_data_path: str) -> pd.DataFrame:
        """
        Load test data for a symbol from a file.
        
        Parameters:
        -----------
        symbol : str
            Symbol to load data for
        test_data_path : str
            Path to the test data file
        
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        # Load data from file
        df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
        
        # Store in cache
        self.data[symbol] = df
        
        return df
    
    def get_data(self, symbol: str, start_date: Union[str, datetime],
                end_date: Union[str, datetime], frequency: str = '1d') -> pd.DataFrame:
        """
        Get data for a symbol within a date range.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get data for
        start_date : Union[str, datetime]
            Start date
        end_date : Union[str, datetime]
            End date
        frequency : str
            Data frequency (e.g., '1d', '1h', '5m')
        
        Returns:
        --------
        pd.DataFrame
            Price data
        """
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check if we have data for this symbol
        if symbol in self.data:
            # Filter by date range
            return self.data[symbol].loc[start_date:end_date]
        
        # Generate synthetic data
        return self._generate_synthetic_data(symbol, start_date, end_date, frequency)
    
    def _generate_synthetic_data(self, symbol: str, start_date: datetime,
                               end_date: datetime, frequency: str = '1d') -> pd.DataFrame:
        """
        Generate synthetic price data.
        
        Parameters:
        -----------
        symbol : str
            Symbol to generate data for
        start_date : datetime
            Start date
        end_date : datetime
            End date
        frequency : str
            Data frequency (e.g., '1d', '1h', '5m')
        
        Returns:
        --------
        pd.DataFrame
            Synthetic price data
        """
        # Parse frequency to pandas frequency
        if frequency == '1d':
            freq = 'D'
        elif frequency == '1h':
            freq = 'H'
        elif frequency == '5m':
            freq = '5min'
        else:
            freq = 'D'  # Default to daily
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate synthetic price series
        np.random.seed(hash(symbol) % 10000)  # Deterministic but varies by symbol
        
        # Base parameters vary by symbol to create different behaviors
        symbol_hash = hash(symbol)
        base_price = 50 + (symbol_hash % 500)  # Base price between 50 and 550
        volatility = 0.01 + (symbol_hash % 100) / 1000  # Volatility between 0.01 and 0.11
        
        # Generate random walk with drift
        returns = np.random.normal(0.0002, volatility, len(dates))  # Small positive drift
        prices = base_price * np.cumprod(1 + returns)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # Ensure High ≥ Open ≥ Low and High ≥ Close ≥ Low
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
        
        # Store in cache
        self.data[symbol] = df
        
        return df
    
    def get_cointegrated_series(self, base_symbol: str, hedge_ratio: float,
                              noise_level: float = 0.01, length: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic cointegrated series for testing.
        
        Parameters:
        -----------
        base_symbol : str
            Symbol for the base series
        hedge_ratio : float
            Hedge ratio between the series
        noise_level : float
            Noise level for the cointegrated series
        length : int
            Number of data points to generate
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Base and cointegrated price data
        """
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=length)
        
        # Get or generate base series
        if base_symbol in self.data:
            base_data = self.data[base_symbol].loc[start_date:end_date]
            if len(base_data) < length:
                # Not enough data, generate new
                base_data = self._generate_synthetic_data(base_symbol, start_date, end_date)
        else:
            base_data = self._generate_synthetic_data(base_symbol, start_date, end_date)
        
        # Create cointegrated series name
        coint_symbol = f"{base_symbol}_COINT"
        
        # Generate cointegrated series
        base_close = base_data['Close']
        
        # Create cointegrated series = hedge_ratio * base + constant + noise
        constant = 50  # Arbitrary constant
        noise = np.random.normal(0, noise_level * base_close.mean(), len(base_close))
        coint_close = hedge_ratio * base_close + constant + noise
        
        # Create full OHLCV data for cointegrated series
        coint_data = pd.DataFrame({
            'Open': coint_close * (1 + np.random.normal(0, 0.005, len(coint_close))),
            'High': coint_close * (1 + np.abs(np.random.normal(0, 0.01, len(coint_close)))),
            'Low': coint_close * (1 - np.abs(np.random.normal(0, 0.01, len(coint_close)))),
            'Close': coint_close,
            'Volume': np.random.randint(1000, 10000, len(coint_close))
        }, index=base_close.index)
        
        # Ensure High ≥ Open ≥ Low and High ≥ Close ≥ Low
        coint_data['High'] = coint_data[['High', 'Open', 'Close']].max(axis=1)
        coint_data['Low'] = coint_data[['Low', 'Open', 'Close']].min(axis=1)
        
        # Store in cache
        self.data[coint_symbol] = coint_data
        
        return base_data, coint_data
    
    def get_universe(self, symbols: List[str], start_date: Union[str, datetime],
                   end_date: Union[str, datetime], frequency: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols within a date range.
        
        Parameters:
        -----------
        symbols : List[str]
            List of symbols to get data for
        start_date : Union[str, datetime]
            Start date
        end_date : Union[str, datetime]
            End date
        frequency : str
            Data frequency (e.g., '1d', '1h', '5m')
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of price data by symbol
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = self.get_data(symbol, start_date, end_date, frequency)
        
        return result
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to get price for
        
        Returns:
        --------
        float
            Latest price
        """
        # Check if we have data for this symbol
        if symbol in self.data:
            return self.data[symbol]['Close'].iloc[-1]
        
        # Generate a random price
        return 100.0 + np.random.normal(0, 10) 