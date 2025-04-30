"""
Equity Asset Implementation for the Intraday Statistical Arbitrage System.

This module implements the equity asset and asset class for trading
equity securities.
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

class EquityAsset(Asset):
    """
    Equity asset implementation for trading stocks and ETFs.
    """
    
    def __init__(self, symbol: str, name: Optional[str] = None, 
                exchange: Optional[str] = None,
                primary_exchange: Optional[str] = None,
                sector: Optional[str] = None,
                industry: Optional[str] = None,
                connector: Optional[Any] = None,
                **kwargs):
        """
        Initialize an equity asset.
        
        Parameters:
        -----------
        symbol : str
            The equity ticker symbol
        name : str, optional
            Human-readable name for the equity
        exchange : str, optional
            Exchange where the equity is traded
        primary_exchange : str, optional
            Primary exchange where the equity is listed
        sector : str, optional
            The business sector the equity belongs to
        industry : str, optional
            The specific industry within the sector
        connector : Any, optional
            Data connector to use for retrieving price data
        **kwargs : dict
            Additional parameters
        """
        super().__init__(symbol, name, **kwargs)
        self.exchange = exchange
        self.primary_exchange = primary_exchange
        self.sector = sector
        self.industry = industry
        self.connector = connector or IBConnector()
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the equity.
        
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
                exchange=self.exchange,
                security_type='STK'
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the equity.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing equity metadata
        """
        try:
            stock_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='STK'
            )
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "primary_exchange": self.primary_exchange or stock_info.get("primary_exchange"),
                "sector": self.sector or stock_info.get("sector"),
                "industry": self.industry or stock_info.get("industry"),
                "market_cap": stock_info.get("market_cap"),
                "currency": stock_info.get("currency"),
                "average_volume": stock_info.get("average_volume")
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {self.symbol}: {str(e)}")
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "primary_exchange": self.primary_exchange,
                "sector": self.sector,
                "industry": self.industry
            }
    
    def get_current_price(self) -> float:
        """
        Get the current price of the equity.
        
        Returns:
        --------
        float
            Current price
        """
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='STK'
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get the trading hours for the equity.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing trading hours information
        """
        try:
            stock_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='STK'
            )
            return {
                "trading_hours": stock_info.get("trading_hours"),
                "liquid_hours": stock_info.get("liquid_hours"),
                "timezone": stock_info.get("timezone")
            }
        except Exception as e:
            logger.error(f"Error getting trading hours for {self.symbol}: {str(e)}")
            return {}
    
    def get_fundamental_data(self) -> Dict[str, Any]:
        """
        Get fundamental data for the equity.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing fundamental data
        """
        try:
            return self.connector.get_fundamental_data(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting fundamental data for {self.symbol}: {str(e)}")
            return {}
    
    def get_short_interest(self) -> Dict[str, Any]:
        """
        Get short interest data for the equity.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing short interest data
        """
        try:
            return self.connector.get_short_interest(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting short interest for {self.symbol}: {str(e)}")
            return {}
    
    def get_option_chain(self) -> pd.DataFrame:
        """
        Get option chain data for the equity.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing option chain data
        """
        try:
            return self.connector.get_option_chain(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting option chain for {self.symbol}: {str(e)}")
            return pd.DataFrame()


class EquityAssetClass(AssetClass):
    """
    Equity asset class implementation.
    """
    
    def __init__(self, name: str = "Equities", connector: Optional[Any] = None):
        """
        Initialize the equity asset class.
        
        Parameters:
        -----------
        name : str, optional
            Name of the asset class
        connector : Any, optional
            Data connector to use for retrieving data
        """
        super().__init__(name)
        self.connector = connector or IBConnector()
    
    def create_asset(self, symbol: str, **kwargs) -> EquityAsset:
        """
        Create an equity asset.
        
        Parameters:
        -----------
        symbol : str
            The equity ticker symbol
        **kwargs : dict
            Additional equity-specific parameters
            
        Returns:
        --------
        EquityAsset
            The created equity asset
        """
        logger.info(f"Creating equity asset for {symbol}")
        asset = EquityAsset(symbol=symbol, connector=self.connector, **kwargs)
        self.add_asset(asset)
        return asset
    
    def get_all_assets(self) -> List[EquityAsset]:
        """
        Get all available equities in this class.
        
        Returns:
        --------
        List[EquityAsset]
            List of all available equity assets
        """
        try:
            # Get a list of available equities from the connector
            symbols = self.connector.get_available_symbols(security_type='STK')
            
            # Create assets for each symbol if not already in our list
            for symbol_info in symbols:
                symbol = symbol_info.get('symbol')
                if symbol and not self.get_asset(symbol):
                    self.create_asset(
                        symbol=symbol,
                        name=symbol_info.get('name'),
                        exchange=symbol_info.get('exchange'),
                        primary_exchange=symbol_info.get('primary_exchange'),
                        sector=symbol_info.get('sector'),
                        industry=symbol_info.get('industry')
                    )
            
            return list(self._assets.values())
        except Exception as e:
            logger.error(f"Error getting all equity assets: {str(e)}")
            return list(self._assets.values())
    
    def get_trading_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the trading calendar for equities.
        
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
            return self.connector.get_trading_calendar(
                year=year,
                security_type='STK'
            )
        except Exception as e:
            logger.error(f"Error getting equity trading calendar for {year}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for equities.
        
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
            return self.connector.get_market_holidays(
                year=year,
                security_type='STK'
            )
        except Exception as e:
            logger.error(f"Error getting equity market holidays for {year}: {str(e)}")
            return []
    
    def get_sectors(self) -> List[str]:
        """
        Get list of available sectors.
        
        Returns:
        --------
        List[str]
            List of available sectors
        """
        try:
            return self.connector.get_sectors()
        except Exception as e:
            logger.error(f"Error getting equity sectors: {str(e)}")
            return []
    
    def get_equities_by_sector(self, sector: str) -> List[EquityAsset]:
        """
        Get equities filtered by sector.
        
        Parameters:
        -----------
        sector : str
            Sector to filter by
            
        Returns:
        --------
        List[EquityAsset]
            List of equity assets in the specified sector
        """
        return [asset for asset in self._assets.values() 
                if isinstance(asset, EquityAsset) and asset.sector == sector]
    
    def get_etfs(self) -> List[EquityAsset]:
        """
        Get all ETFs.
        
        Returns:
        --------
        List[EquityAsset]
            List of ETF assets
        """
        try:
            etf_symbols = self.connector.get_available_symbols(security_type='ETF')
            
            for symbol_info in etf_symbols:
                symbol = symbol_info.get('symbol')
                if symbol and not self.get_asset(symbol):
                    self.create_asset(
                        symbol=symbol,
                        name=symbol_info.get('name'),
                        exchange=symbol_info.get('exchange'),
                        primary_exchange=symbol_info.get('primary_exchange')
                    )
            
            return [asset for asset in self._assets.values() 
                    if isinstance(asset, EquityAsset) and 
                    asset.metadata.get('security_type') == 'ETF']
        except Exception as e:
            logger.error(f"Error getting ETFs: {str(e)}")
            return [] 