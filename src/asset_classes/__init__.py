"""
Asset Classes Module for the Intraday Statistical Arbitrage System.

This module provides various asset classes for different types of financial instruments.
"""

# Base asset classes
from .base import Asset, AssetClass

# Import asset class implementations
from .futures import FuturesAsset, FuturesAssetClass
from .equities import EquityAsset, EquityAssetClass
from .cryptocurrencies import CryptoAsset, CryptoAssetClass
from .fixed_income import FixedIncomeAsset, FixedIncomeAssetClass

# Factory functions
from .factory import create_asset, get_asset_class, register_asset_class

__all__ = [
    'Asset', 'AssetClass',
    'FuturesAsset', 'FuturesAssetClass',
    'EquityAsset', 'EquityAssetClass',
    'CryptoAsset', 'CryptoAssetClass',
    'FixedIncomeAsset', 'FixedIncomeAssetClass',
    'create_asset', 'get_asset_class', 'register_asset_class'
] 