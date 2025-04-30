"""
Futures Asset Implementation for the Intraday Statistical Arbitrage System.

This module implements the futures asset and asset class for trading
futures contracts.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.asset_classes.base import Asset, AssetClass
from src.connectors.ib import IBConnector

# Configure logging
logger = logging.getLogger(__name__)

class FuturesAsset(Asset):
    """
    Futures asset implementation for trading futures contracts.
    """
    
    def __init__(self, symbol: str, name: Optional[str] = None, 
                exchange: Optional[str] = None, 
                contract_size: Optional[float] = None,
                tick_size: Optional[float] = None,
                connector: Optional[Any] = None,
                **kwargs):
        """
        Initialize a futures asset.
        
        Parameters:
        -----------
        symbol : str
            The futures contract symbol
        name : str, optional
            Human-readable name for the contract
        exchange : str, optional
            Exchange where the futures contract is traded
        contract_size : float, optional
            Size of one contract in base units
        tick_size : float, optional
            Minimum price movement
        connector : Any, optional
            Data connector to use for retrieving price data
        **kwargs : dict
            Additional parameters
        """
        super().__init__(symbol, name, **kwargs)
        self.exchange = exchange
        self.contract_size = contract_size
        self.tick_size = tick_size
        self.connector = connector or IBConnector()
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the futures contract.
        
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
        logger.info(f"Getting data for {self.symbol} from {start_date} to {end_date}")
        try:
            return self.connector.get_historical_data(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the futures contract.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing futures contract metadata
        """
        try:
            contract_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange
            )
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "contract_size": self.contract_size or contract_info.get("contract_size"),
                "tick_size": self.tick_size or contract_info.get("tick_size"),
                "expiration": contract_info.get("expiration"),
                "currency": contract_info.get("currency"),
                "min_tick": contract_info.get("min_tick"),
                "trading_hours": contract_info.get("trading_hours"),
                "liquid_hours": contract_info.get("liquid_hours")
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {self.symbol}: {str(e)}")
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "contract_size": self.contract_size,
                "tick_size": self.tick_size
            }
    
    def get_current_price(self) -> float:
        """
        Get the current price of the futures contract.
        
        Returns:
        --------
        float
            Current price
        """
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.exchange
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get the trading hours for the futures contract.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing trading hours information
        """
        try:
            contract_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange
            )
            return {
                "trading_hours": contract_info.get("trading_hours"),
                "liquid_hours": contract_info.get("liquid_hours"),
                "timezone": contract_info.get("timezone")
            }
        except Exception as e:
            logger.error(f"Error getting trading hours for {self.symbol}: {str(e)}")
            return {}
    
    def get_margin_requirements(self) -> Dict[str, float]:
        """
        Get margin requirements for the futures contract.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing margin requirements
        """
        try:
            return self.connector.get_margin_requirements(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting margin requirements for {self.symbol}: {str(e)}")
            return {}
    
    def get_contract_months(self) -> List[str]:
        """
        Get available contract months for the futures.
        
        Returns:
        --------
        List[str]
            List of available contract months
        """
        try:
            return self.connector.get_contract_months(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting contract months for {self.symbol}: {str(e)}")
            return []
    
    def get_continuous_contract(self, start_date: Union[str, datetime], 
                               end_date: Union[str, datetime],
                               roll_method: str = 'volume') -> pd.DataFrame:
        """
        Get continuous contract data for the futures.
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date for data retrieval
        end_date : str or datetime
            End date for data retrieval
        roll_method : str, optional
            Method to use for rolling contracts ('volume', 'open_interest', 'date')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing continuous contract data
        """
        try:
            return self.connector.get_continuous_contract(
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                exchange=self.exchange,
                roll_method=roll_method
            )
        except Exception as e:
            logger.error(f"Error getting continuous contract for {self.symbol}: {str(e)}")
            return pd.DataFrame()


class FuturesAssetClass(AssetClass):
    """
    Futures asset class implementation.
    """
    
    def __init__(self, name: str = "Futures", connector: Optional[Any] = None):
        """
        Initialize the futures asset class.
        
        Parameters:
        -----------
        name : str, optional
            Name of the asset class
        connector : Any, optional
            Data connector to use for retrieving data
        """
        super().__init__(name)
        self.connector = connector or IBConnector()
    
    def create_asset(self, symbol: str, **kwargs) -> FuturesAsset:
        """
        Create a futures asset.
        
        Parameters:
        -----------
        symbol : str
            The futures contract symbol
        **kwargs : dict
            Additional parameters for the futures asset
            
        Returns:
        --------
        FuturesAsset
            The created futures asset
        """
        # Use the connector if not provided
        kwargs.setdefault('connector', self.connector)
        
        # Create the asset
        asset = FuturesAsset(symbol, **kwargs)
        
        # Add to the registry
        self.add_asset(asset)
        
        return asset
    
    def get_all_assets(self) -> List[FuturesAsset]:
        """
        Get all available futures assets.
        
        Returns:
        --------
        List[FuturesAsset]
            List of all available futures assets
        """
        try:
            # Get all futures contracts from the connector
            contracts = self.connector.get_all_futures_contracts()
            
            # Create assets for each contract
            assets = []
            for contract in contracts:
                symbol = contract.get('symbol')
                exchange = contract.get('exchange')
                name = contract.get('name')
                
                # Create the asset if it doesn't exist
                if symbol not in self._assets:
                    asset = self.create_asset(
                        symbol=symbol,
                        name=name,
                        exchange=exchange,
                        contract_size=contract.get('contract_size'),
                        tick_size=contract.get('tick_size')
                    )
                    assets.append(asset)
                else:
                    assets.append(self._assets[symbol])
            
            return assets
        except Exception as e:
            logger.error(f"Error getting all futures assets: {str(e)}")
            return list(self._assets.values())
    
    def get_trading_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the trading calendar for futures markets.
        
        Parameters:
        -----------
        year : int
            Year to get the calendar for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing trading days and hours
        """
        try:
            return self.connector.get_trading_calendar(year)
        except Exception as e:
            logger.error(f"Error getting futures trading calendar: {str(e)}")
            return pd.DataFrame()
    
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for futures markets.
        
        Parameters:
        -----------
        year : int
            Year to get holidays for
            
        Returns:
        --------
        List[datetime]
            List of holiday dates
        """
        try:
            return self.connector.get_market_holidays(year)
        except Exception as e:
            logger.error(f"Error getting futures market holidays: {str(e)}")
            return []
    
    def get_futures_curve(self, base_symbol: str, 
                         date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get the futures curve for a base symbol.
        
        Parameters:
        -----------
        base_symbol : str
            Base symbol for the futures (e.g., 'ES' for E-mini S&P 500)
        date : str or datetime, optional
            Date for which to get the futures curve, defaults to current date
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the futures curve
        """
        try:
            return self.connector.get_futures_curve(base_symbol, date)
        except Exception as e:
            logger.error(f"Error getting futures curve for {base_symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_popular_spreads(self) -> List[Dict[str, Any]]:
        """
        Get popular futures spreads.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of popular spreads with their details
        """
        try:
            return self.connector.get_popular_spreads()
        except Exception as e:
            logger.error(f"Error getting popular futures spreads: {str(e)}")
            return [] 