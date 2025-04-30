"""
Cryptocurrency Asset Implementation for the Intraday Statistical Arbitrage System.

This module implements the cryptocurrency asset and asset class for trading
digital currencies.
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

class CryptoAsset(Asset):
    """
    Cryptocurrency asset implementation for trading digital currencies.
    """
    
    def __init__(self, symbol: str, name: Optional[str] = None,
                exchange: Optional[str] = None,
                market_type: Optional[str] = None,
                is_stablecoin: Optional[bool] = False,
                connector: Optional[Any] = None,
                **kwargs):
        """
        Initialize a cryptocurrency asset.
        
        Parameters:
        -----------
        symbol : str
            The cryptocurrency symbol (e.g., BTC, ETH)
        name : str, optional
            Human-readable name for the cryptocurrency
        exchange : str, optional
            Exchange where the cryptocurrency is traded
        market_type : str, optional
            Market type (spot, futures, perpetual)
        is_stablecoin : bool, optional
            Whether the asset is a stablecoin
        connector : Any, optional
            Data connector to use for retrieving price data
        **kwargs : dict
            Additional parameters
        """
        super().__init__(symbol, name, **kwargs)
        self.exchange = exchange
        self.market_type = market_type
        self.is_stablecoin = is_stablecoin
        self.connector = connector or IBConnector()
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the cryptocurrency.
        
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
                security_type='CRYPTO'
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the cryptocurrency.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing cryptocurrency metadata
        """
        try:
            crypto_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='CRYPTO'
            )
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "market_type": self.market_type or crypto_info.get("market_type"),
                "is_stablecoin": self.is_stablecoin,
                "market_cap": crypto_info.get("market_cap"),
                "circulating_supply": crypto_info.get("circulating_supply"),
                "max_supply": crypto_info.get("max_supply")
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {self.symbol}: {str(e)}")
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "market_type": self.market_type,
                "is_stablecoin": self.is_stablecoin
            }
    
    def get_current_price(self) -> float:
        """
        Get the current price of the cryptocurrency.
        
        Returns:
        --------
        float
            Current price
        """
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='CRYPTO'
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get the trading hours for the cryptocurrency.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing trading hours information
        """
        # Most cryptocurrencies trade 24/7
        return {
            "trading_hours": "24/7",
            "liquid_hours": "24/7",
            "timezone": "UTC"
        }
    
    def get_orderbook(self, depth: int = 10) -> Dict[str, Any]:
        """
        Get the current order book for the cryptocurrency.
        
        Parameters:
        -----------
        depth : int, optional
            Depth of the order book to retrieve
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing orderbook data
        """
        try:
            return self.connector.get_orderbook(
                symbol=self.symbol,
                exchange=self.exchange,
                depth=depth
            )
        except Exception as e:
            logger.error(f"Error getting orderbook for {self.symbol}: {str(e)}")
            return {"bids": [], "asks": []}
    
    def get_funding_rate(self) -> float:
        """
        Get the current funding rate (for perpetual contracts).
        
        Returns:
        --------
        float
            Current funding rate as a percentage
        """
        if self.market_type != 'perpetual':
            return 0.0
            
        try:
            return self.connector.get_funding_rate(
                symbol=self.symbol,
                exchange=self.exchange
            )
        except Exception as e:
            logger.error(f"Error getting funding rate for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_liquidation_price(self, position_size: float, leverage: float) -> float:
        """
        Calculate the liquidation price for a leveraged position.
        
        Parameters:
        -----------
        position_size : float
            Size of the position
        leverage : float
            Leverage used
            
        Returns:
        --------
        float
            Liquidation price
        """
        try:
            current_price = self.get_current_price()
            if current_price == 0:
                return 0
                
            # Simple liquidation price calculation
            if position_size > 0:  # Long position
                return current_price * (1 - 1/leverage)
            else:  # Short position
                return current_price * (1 + 1/leverage)
        except Exception as e:
            logger.error(f"Error calculating liquidation price for {self.symbol}: {str(e)}")
            return 0.0


class CryptoAssetClass(AssetClass):
    """
    Cryptocurrency asset class implementation.
    """
    
    def __init__(self, name: str = "Cryptocurrencies", connector: Optional[Any] = None):
        """
        Initialize the cryptocurrency asset class.
        
        Parameters:
        -----------
        name : str, optional
            Name of the asset class
        connector : Any, optional
            Data connector to use for retrieving data
        """
        super().__init__(name)
        self.connector = connector or IBConnector()
    
    def create_asset(self, symbol: str, **kwargs) -> CryptoAsset:
        """
        Create a cryptocurrency asset.
        
        Parameters:
        -----------
        symbol : str
            The cryptocurrency symbol
        **kwargs : dict
            Additional cryptocurrency-specific parameters
            
        Returns:
        --------
        CryptoAsset
            The created cryptocurrency asset
        """
        logger.info(f"Creating cryptocurrency asset for {symbol}")
        asset = CryptoAsset(symbol=symbol, connector=self.connector, **kwargs)
        self.add_asset(asset)
        return asset
    
    def get_all_assets(self) -> List[CryptoAsset]:
        """
        Get all available cryptocurrencies in this class.
        
        Returns:
        --------
        List[CryptoAsset]
            List of all available cryptocurrency assets
        """
        try:
            # Get a list of available cryptocurrencies from the connector
            symbols = self.connector.get_available_symbols(security_type='CRYPTO')
            
            # Create assets for each symbol if not already in our list
            for symbol_info in symbols:
                symbol = symbol_info.get('symbol')
                if symbol and not self.get_asset(symbol):
                    self.create_asset(
                        symbol=symbol,
                        name=symbol_info.get('name'),
                        exchange=symbol_info.get('exchange'),
                        market_type=symbol_info.get('market_type'),
                        is_stablecoin=symbol_info.get('is_stablecoin', False)
                    )
            
            return list(self._assets.values())
        except Exception as e:
            logger.error(f"Error getting all cryptocurrency assets: {str(e)}")
            return list(self._assets.values())
    
    def get_trading_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the trading calendar for cryptocurrencies.
        
        Parameters:
        -----------
        year : int
            Year to get the calendar for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing trading days and hours
        """
        # Cryptocurrencies trade 24/7, so create a full calendar
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date = pd.Timestamp(f"{year}-12-31")
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        return pd.DataFrame({
            'date': date_range,
            'is_trading_day': True,
            'trading_hours': '24/7',
            'liquid_hours': '24/7'
        })
    
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for cryptocurrencies.
        
        Parameters:
        -----------
        year : int
            Year to get holidays for
            
        Returns:
        --------
        List[datetime]
            List of holiday dates (empty for 24/7 markets)
        """
        # Cryptocurrency markets don't have holidays as they're 24/7
        return []
    
    def get_top_assets_by_market_cap(self, limit: int = 20) -> List[CryptoAsset]:
        """
        Get top cryptocurrencies by market capitalization.
        
        Parameters:
        -----------
        limit : int, optional
            Maximum number of assets to return
            
        Returns:
        --------
        List[CryptoAsset]
            List of top cryptocurrency assets by market cap
        """
        try:
            # Ensure assets are loaded
            self.get_all_assets()
            
            # Get market cap data for each asset
            market_caps = []
            for asset in self._assets.values():
                if isinstance(asset, CryptoAsset):
                    metadata = asset.get_metadata()
                    market_cap = metadata.get("market_cap", 0)
                    market_caps.append((asset, market_cap))
            
            # Sort by market cap and return top assets
            market_caps.sort(key=lambda x: x[1], reverse=True)
            return [asset for asset, _ in market_caps[:limit]]
        except Exception as e:
            logger.error(f"Error getting top assets by market cap: {str(e)}")
            return []
    
    def get_stablecoins(self) -> List[CryptoAsset]:
        """
        Get all stablecoins.
        
        Returns:
        --------
        List[CryptoAsset]
            List of stablecoin assets
        """
        return [asset for asset in self._assets.values() 
                if isinstance(asset, CryptoAsset) and asset.is_stablecoin] 