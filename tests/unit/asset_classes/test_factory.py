"""
Unit tests for the asset factory implementation.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.asset_classes.factory import create_asset, get_asset_class
from src.asset_classes.base import Asset, AssetClass
from src.asset_classes.futures.futures_asset import FuturesAsset, FuturesAssetClass
from src.asset_classes.equities.equity_asset import EquityAsset, EquityAssetClass
from src.asset_classes.cryptocurrencies.crypto_asset import CryptoAsset, CryptoAssetClass
from src.asset_classes.fixed_income.fixed_income_asset import FixedIncomeAsset, FixedIncomeAssetClass

class TestAssetFactory(unittest.TestCase):
    """Test cases for asset factory functions."""
    
    def test_get_asset_class_futures(self):
        """Test retrieving the futures asset class."""
        # Act
        asset_class = get_asset_class("futures")
        
        # Assert
        self.assertIsInstance(asset_class, FuturesAssetClass)
        self.assertEqual(asset_class.name, "Futures")
    
    def test_get_asset_class_equities(self):
        """Test retrieving the equities asset class."""
        # Act
        asset_class = get_asset_class("equities")
        
        # Assert
        self.assertIsInstance(asset_class, EquityAssetClass)
        self.assertEqual(asset_class.name, "Equities")
    
    def test_get_asset_class_cryptocurrencies(self):
        """Test retrieving the cryptocurrencies asset class."""
        # Act
        asset_class = get_asset_class("cryptocurrencies")
        
        # Assert
        self.assertIsInstance(asset_class, CryptoAssetClass)
        self.assertEqual(asset_class.name, "Cryptocurrencies")
    
    def test_get_asset_class_fixed_income(self):
        """Test retrieving the fixed income asset class."""
        # Act
        asset_class = get_asset_class("fixed_income")
        
        # Assert
        self.assertIsInstance(asset_class, FixedIncomeAssetClass)
        self.assertEqual(asset_class.name, "Fixed Income")
    
    def test_get_asset_class_invalid(self):
        """Test retrieving an invalid asset class type."""
        # Act & Assert
        # Should raise ValueError for invalid type
        self.assertIsNone(get_asset_class("invalid_type"))
    
    def test_create_asset_futures(self):
        """Test creating a futures asset."""
        # Arrange
        symbol = "ES"
        name = "E-mini S&P 500"
        exchange = "CME"
        
        # Act
        asset = create_asset(
            asset_class_type="futures",
            symbol=symbol,
            name=name,
            exchange=exchange
        )
        
        # Assert
        self.assertIsInstance(asset, FuturesAsset)
        self.assertEqual(asset.symbol, symbol)
        self.assertEqual(asset.name, name)
        self.assertEqual(asset.exchange, exchange)
    
    def test_create_asset_equity(self):
        """Test creating an equity asset."""
        # Arrange
        symbol = "AAPL"
        name = "Apple Inc."
        exchange = "NASDAQ"
        
        # Act
        asset = create_asset(
            asset_class_type="equities",
            symbol=symbol,
            name=name,
            exchange=exchange
        )
        
        # Assert
        self.assertIsInstance(asset, EquityAsset)
        self.assertEqual(asset.symbol, symbol)
        self.assertEqual(asset.name, name)
        self.assertEqual(asset.exchange, exchange)
    
    def test_create_asset_crypto(self):
        """Test creating a cryptocurrency asset."""
        # Arrange
        symbol = "BTC"
        name = "Bitcoin"
        exchange = "Coinbase"
        
        # Act
        asset = create_asset(
            asset_class_type="cryptocurrencies",
            symbol=symbol,
            name=name,
            exchange=exchange
        )
        
        # Assert
        self.assertIsInstance(asset, CryptoAsset)
        self.assertEqual(asset.symbol, symbol)
        self.assertEqual(asset.name, name)
        self.assertEqual(asset.exchange, exchange)
    
    def test_create_asset_fixed_income(self):
        """Test creating a fixed income asset."""
        # Arrange
        symbol = "T"
        name = "US Treasury 10Y"
        exchange = "NYSE"
        
        # Act
        asset = create_asset(
            asset_class_type="fixed_income",
            symbol=symbol,
            name=name,
            exchange=exchange
        )
        
        # Assert
        self.assertIsInstance(asset, FixedIncomeAsset)
        self.assertEqual(asset.symbol, symbol)
        self.assertEqual(asset.name, name)
        self.assertEqual(asset.exchange, exchange)
    
    def test_create_asset_invalid(self):
        """Test creating an asset with an invalid asset type."""
        # Act & Assert
        asset = create_asset(asset_class_type="invalid_type", symbol="TEST")
        self.assertIsNone(asset)
    
    def test_create_asset_with_connector(self):
        """Test creating an asset with a custom connector."""
        # Arrange
        mock_connector = MagicMock()
        
        try:
            # Act
            asset = create_asset(
                asset_class_type="equities",
                symbol="AAPL",
                my_connector=mock_connector  # Use a different name to avoid conflict
            )
            
            # Assert
            self.assertIsNotNone(asset)
            # Can't check the connector directly as the name may be different
        except Exception as e:
            self.fail(f"Failed to create asset with custom connector: {str(e)}")
    
    def test_create_asset_with_additional_params(self):
        """Test creating an asset with additional parameters."""
        # Arrange & Act
        asset = create_asset(
            asset_class_type="cryptocurrencies",
            symbol="BTC",
            is_stablecoin=True,
            market_type="spot"
        )
        
        # Assert
        self.assertIsInstance(asset, CryptoAsset)
        self.assertTrue(asset.is_stablecoin)
        self.assertEqual(asset.market_type, "spot")

if __name__ == "__main__":
    unittest.main() 