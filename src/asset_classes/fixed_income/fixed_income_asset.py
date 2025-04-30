"""
Fixed Income Asset Implementation for the Intraday Statistical Arbitrage System.

This module implements the fixed income asset and asset class for trading
bonds and other fixed income securities.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging

from src.asset_classes.base import Asset, AssetClass
from src.connectors.ib import IBConnector

# Configure logging
logger = logging.getLogger(__name__)

class FixedIncomeAsset(Asset):
    """
    Fixed Income asset implementation for trading bonds and other debt instruments.
    """
    
    def __init__(self, symbol: str, name: Optional[str] = None,
                exchange: Optional[str] = None,
                maturity_date: Optional[Union[str, date]] = None,
                coupon: Optional[float] = None,
                bond_type: Optional[str] = None,
                issuer: Optional[str] = None,
                connector: Optional[Any] = None,
                **kwargs):
        """
        Initialize a fixed income asset.
        
        Parameters:
        -----------
        symbol : str
            The bond CUSIP or identifier
        name : str, optional
            Human-readable name for the bond
        exchange : str, optional
            Exchange where the bond is traded
        maturity_date : str or date, optional
            Maturity date of the bond
        coupon : float, optional
            Coupon rate as a percentage
        bond_type : str, optional
            Type of bond (government, corporate, municipal)
        issuer : str, optional
            Name of the issuing entity
        connector : Any, optional
            Data connector to use for retrieving price data
        **kwargs : dict
            Additional parameters
        """
        super().__init__(symbol, name, **kwargs)
        self.exchange = exchange
        self.maturity_date = maturity_date
        self.coupon = coupon
        self.bond_type = bond_type
        self.issuer = issuer
        self.connector = connector or IBConnector()
    
    def get_data(self, start_date: Union[str, datetime], 
                end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical price data for the bond.
        
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
                security_type='BOND'
            )
        except Exception as e:
            logger.error(f"Error getting data for {self.symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the bond.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing bond metadata
        """
        try:
            bond_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='BOND'
            )
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "maturity_date": self.maturity_date or bond_info.get("maturity_date"),
                "coupon": self.coupon or bond_info.get("coupon"),
                "bond_type": self.bond_type or bond_info.get("bond_type"),
                "issuer": self.issuer or bond_info.get("issuer"),
                "rating": bond_info.get("rating"),
                "face_value": bond_info.get("face_value"),
                "currency": bond_info.get("currency")
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {self.symbol}: {str(e)}")
            return {
                "symbol": self.symbol,
                "name": self.name,
                "exchange": self.exchange,
                "maturity_date": self.maturity_date,
                "coupon": self.coupon,
                "bond_type": self.bond_type,
                "issuer": self.issuer
            }
    
    def get_current_price(self) -> float:
        """
        Get the current price of the bond.
        
        Returns:
        --------
        float
            Current price as a percentage of face value
        """
        try:
            return self.connector.get_market_data(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='BOND'
            ).get("last_price", 0.0)
        except Exception as e:
            logger.error(f"Error getting current price for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_trading_hours(self) -> Dict[str, Any]:
        """
        Get the trading hours for the bond.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing trading hours information
        """
        try:
            bond_info = self.connector.get_contract_details(
                symbol=self.symbol,
                exchange=self.exchange,
                security_type='BOND'
            )
            return {
                "trading_hours": bond_info.get("trading_hours"),
                "liquid_hours": bond_info.get("liquid_hours"),
                "timezone": bond_info.get("timezone")
            }
        except Exception as e:
            logger.error(f"Error getting trading hours for {self.symbol}: {str(e)}")
            return {}
    
    def get_yield_to_maturity(self) -> float:
        """
        Calculate the yield to maturity for the bond.
        
        Returns:
        --------
        float
            Yield to maturity as a percentage
        """
        try:
            current_price = self.get_current_price()
            if current_price == 0:
                return 0
                
            metadata = self.get_metadata()
            coupon = metadata.get("coupon", 0)
            face_value = metadata.get("face_value", 100)
            
            if isinstance(self.maturity_date, str):
                maturity_date = datetime.strptime(self.maturity_date, "%Y-%m-%d").date()
            else:
                maturity_date = self.maturity_date
            
            if not maturity_date:
                return 0
                
            years_to_maturity = (maturity_date - date.today()).days / 365.25
            
            if years_to_maturity <= 0:
                return 0
                
            # Simple YTM calculation (approximate)
            annual_payment = coupon * face_value / 100
            ytm = (annual_payment + (face_value - current_price) / years_to_maturity) / ((face_value + current_price) / 2)
            
            return ytm * 100  # Convert to percentage
        except Exception as e:
            logger.error(f"Error calculating YTM for {self.symbol}: {str(e)}")
            return 0.0
    
    def get_duration(self) -> float:
        """
        Calculate the Macaulay duration for the bond.
        
        Returns:
        --------
        float
            Duration in years
        """
        try:
            ytm = self.get_yield_to_maturity() / 100  # Convert from percentage
            if ytm == 0:
                return 0
                
            metadata = self.get_metadata()
            coupon = metadata.get("coupon", 0) / 100  # Convert from percentage
            face_value = metadata.get("face_value", 100)
            
            if isinstance(self.maturity_date, str):
                maturity_date = datetime.strptime(self.maturity_date, "%Y-%m-%d").date()
            else:
                maturity_date = self.maturity_date
            
            if not maturity_date:
                return 0
                
            years_to_maturity = (maturity_date - date.today()).days / 365.25
            
            if years_to_maturity <= 0:
                return 0
            
            # Assume semi-annual coupon payments
            periods = int(years_to_maturity * 2)
            coupon_payment = coupon * face_value / 2
            
            # Calculate PV of each cash flow
            pv_cashflows = []
            time_periods = []
            
            for i in range(1, periods + 1):
                t = i / 2  # Time in years
                pv = coupon_payment / ((1 + ytm/2) ** i)
                pv_cashflows.append(pv)
                time_periods.append(t)
            
            # Add final principal repayment
            pv_cashflows.append(face_value / ((1 + ytm/2) ** periods))
            time_periods.append(years_to_maturity)
            
            # Calculate price
            price = sum(pv_cashflows)
            
            # Calculate duration
            weighted_time = sum(t * pv for t, pv in zip(time_periods, pv_cashflows))
            duration = weighted_time / price
            
            return duration
        except Exception as e:
            logger.error(f"Error calculating duration for {self.symbol}: {str(e)}")
            return 0.0


class FixedIncomeAssetClass(AssetClass):
    """
    Fixed Income asset class implementation.
    """
    
    def __init__(self, name: str = "Fixed Income", connector: Optional[Any] = None):
        """
        Initialize the fixed income asset class.
        
        Parameters:
        -----------
        name : str, optional
            Name of the asset class
        connector : Any, optional
            Data connector to use for retrieving data
        """
        super().__init__(name)
        self.connector = connector or IBConnector()
    
    def create_asset(self, symbol: str, **kwargs) -> FixedIncomeAsset:
        """
        Create a fixed income asset.
        
        Parameters:
        -----------
        symbol : str
            The bond CUSIP or identifier
        **kwargs : dict
            Additional bond-specific parameters
            
        Returns:
        --------
        FixedIncomeAsset
            The created fixed income asset
        """
        logger.info(f"Creating fixed income asset for {symbol}")
        asset = FixedIncomeAsset(symbol=symbol, connector=self.connector, **kwargs)
        self.add_asset(asset)
        return asset
    
    def get_all_assets(self) -> List[FixedIncomeAsset]:
        """
        Get all available fixed income assets in this class.
        
        Returns:
        --------
        List[FixedIncomeAsset]
            List of all available fixed income assets
        """
        try:
            # Get a list of available bonds from the connector
            symbols = self.connector.get_available_symbols(security_type='BOND')
            
            # Create assets for each symbol if not already in our list
            for symbol_info in symbols:
                symbol = symbol_info.get('symbol')
                if symbol and not self.get_asset(symbol):
                    self.create_asset(
                        symbol=symbol,
                        name=symbol_info.get('name'),
                        exchange=symbol_info.get('exchange'),
                        maturity_date=symbol_info.get('maturity_date'),
                        coupon=symbol_info.get('coupon'),
                        bond_type=symbol_info.get('bond_type'),
                        issuer=symbol_info.get('issuer')
                    )
            
            return list(self._assets.values())
        except Exception as e:
            logger.error(f"Error getting all fixed income assets: {str(e)}")
            return list(self._assets.values())
    
    def get_trading_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the trading calendar for fixed income.
        
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
                security_type='BOND'
            )
        except Exception as e:
            logger.error(f"Error getting fixed income trading calendar for {year}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_holidays(self, year: int) -> List[datetime]:
        """
        Get market holidays for fixed income.
        
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
                security_type='BOND'
            )
        except Exception as e:
            logger.error(f"Error getting fixed income market holidays for {year}: {str(e)}")
            return []
    
    def get_yield_curve(self, date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get the yield curve for treasury securities.
        
        Parameters:
        -----------
        date : str or datetime, optional
            Date to get the yield curve for (default: current date)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing yield curve data
        """
        try:
            return self.connector.get_yield_curve(date=date)
        except Exception as e:
            logger.error(f"Error getting yield curve: {str(e)}")
            return pd.DataFrame()
    
    def get_treasuries(self) -> List[FixedIncomeAsset]:
        """
        Get all treasury securities.
        
        Returns:
        --------
        List[FixedIncomeAsset]
            List of treasury assets
        """
        return [asset for asset in self._assets.values() 
                if isinstance(asset, FixedIncomeAsset) and asset.bond_type == 'government']
    
    def get_corporate_bonds(self) -> List[FixedIncomeAsset]:
        """
        Get all corporate bonds.
        
        Returns:
        --------
        List[FixedIncomeAsset]
            List of corporate bond assets
        """
        return [asset for asset in self._assets.values() 
                if isinstance(asset, FixedIncomeAsset) and asset.bond_type == 'corporate'] 