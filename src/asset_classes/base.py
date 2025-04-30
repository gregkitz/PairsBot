"""
Base Asset Classes for the Intraday Statistical Arbitrage System.

This module defines the base Asset and AssetClass interfaces for all
asset classes supported by the system.
"""

import abc
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


class Asset(abc.ABC):
    """
    Base class for all asset types.
    
    This abstract base class defines the interface that all assets must implement.
    """
    
    def __init__(self, symbol: str, name: Optional[str] = None, **kwargs):
        """
        Initialize an asset.
        
        Parameters:
        -----------
        symbol : str
            The asset's ticker symbol or identifier
        name : str, optional
            Human-readable name for the asset
        **kwargs : dict
            Additional asset-specific parameters
        """
        self.symbol = symbol
        self.name = name if name is not None else symbol
        self.metadata = kwargs
    
    @abc.abstractmethod
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the asset.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for data retrieval
        end_date : str or datetime
            End date for data retrieval
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing OHLCV data
        """
        pass
    
    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the asset.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing asset metadata
        """
        pass
    
    @abc.abstractmethod
    def get_current_price(self) -> float:
        """
        Get the current price of the asset.
        
        Returns:
        --------
        float
            Current price
        """
        pass
    
    @abc.abstractmethod
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get the trading hours for the asset.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing trading hours information
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.symbol})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(symbol='{self.symbol}', name='{self.name}')"


class AssetClass(abc.ABC):
    """
    Base class for all asset classes.
    
    This abstract base class defines the interface that all asset classes must implement.
    """
    
    def __init__(self, name: str):
        """
        Initialize an asset class.
        
        Parameters:
        -----------
        name : str
            Name of the asset class
        """
        self.name = name
        self._assets = {}
    
    @abc.abstractmethod
    def create_asset(self, symbol: str, **kwargs) -> Asset:
        """
        Create an asset of this class.
        
        Parameters:
        -----------
        symbol : str
            The asset's ticker symbol or identifier
        **kwargs : dict
            Additional asset-specific parameters
            
        Returns:
        --------
        Asset
            The created asset
        """
        pass
    
    @abc.abstractmethod
    def get_all_assets(self) -> List[Asset]:
        """
        Get all available assets in this class.
        
        Returns:
        --------
        List[Asset]
            List of all available assets
        """
        pass
    
    @abc.abstractmethod
    def get_trading_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the trading calendar for this asset class.
        
        Parameters:
        -----------
        year : int
            Year to get the calendar for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing trading days and hours
        """
        pass
    
    @abc.abstractmethod
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for this asset class.
        
        Parameters:
        -----------
        year : int
            Year to get holidays for
            
        Returns:
        --------
        List[datetime]
            List of holiday dates
        """
        pass
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """
        Get an asset by symbol.
        
        Parameters:
        -----------
        symbol : str
            The asset's ticker symbol or identifier
            
        Returns:
        --------
        Asset or None
            The asset if found, None otherwise
        """
        return self._assets.get(symbol)
    
    def add_asset(self, asset: Asset) -> None:
        """
        Add an asset to this asset class.
        
        Parameters:
        -----------
        asset : Asset
            The asset to add
        """
        self._assets[asset.symbol] = asset
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')" 