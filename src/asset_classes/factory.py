"""
Factory Module for Asset Classes.

This module provides factory functions for creating assets and getting
asset classes based on their type.
"""

from typing import Dict, Optional, Any
import logging

from .base import Asset, AssetClass

# Configure logging
logger = logging.getLogger(__name__)

# Registry of asset classes
_asset_classes = {}

def register_asset_class(asset_class_type: str, asset_class_cls: type) -> None:
    """
    Register an asset class type.
    
    Parameters:
    -----------
    asset_class_type : str
        Type identifier for the asset class
    asset_class_cls : type
        Class to instantiate for this asset class type
    """
    logger.debug(f"Registering asset class type: {asset_class_type}")
    _asset_classes[asset_class_type] = asset_class_cls

def get_asset_class(asset_class_type: str, **kwargs) -> Optional[AssetClass]:
    """
    Get an asset class by type.
    
    Parameters:
    -----------
    asset_class_type : str
        Type identifier for the asset class
    **kwargs : dict
        Arguments to pass to the asset class constructor
    
    Returns:
    --------
    AssetClass or None
        The asset class instance if type is registered, None otherwise
    """
    asset_class_cls = _asset_classes.get(asset_class_type)
    
    if asset_class_cls is None:
        logger.warning(f"Unknown asset class type: {asset_class_type}")
        return None
    
    try:
        return asset_class_cls(**kwargs)
    except Exception as e:
        logger.error(f"Error creating asset class {asset_class_type}: {str(e)}")
        return None

def create_asset(asset_class_type: str, symbol: str, **kwargs) -> Optional[Asset]:
    """
    Create an asset of the specified class.
    
    Parameters:
    -----------
    asset_class_type : str
        Type identifier for the asset class
    symbol : str
        Symbol of the asset to create
    **kwargs : dict
        Additional arguments for asset creation
    
    Returns:
    --------
    Asset or None
        The created asset if successful, None otherwise
    """
    asset_class = get_asset_class(asset_class_type)
    
    if asset_class is None:
        return None
    
    try:
        return asset_class.create_asset(symbol, **kwargs)
    except Exception as e:
        logger.error(f"Error creating asset {symbol} of class {asset_class_type}: {str(e)}")
        return None

# Register built-in asset classes
from .equities import EquityAssetClass
from .cryptocurrencies import CryptoAssetClass
from .fixed_income import FixedIncomeAssetClass
from .futures import FuturesAssetClass

register_asset_class('equities', EquityAssetClass)
register_asset_class('stocks', EquityAssetClass)  # Alias
register_asset_class('equity', EquityAssetClass)  # Alias
register_asset_class('cryptocurrencies', CryptoAssetClass)
register_asset_class('crypto', CryptoAssetClass)  # Alias
register_asset_class('fixed_income', FixedIncomeAssetClass)
register_asset_class('bonds', FixedIncomeAssetClass)  # Alias
register_asset_class('futures', FuturesAssetClass) 