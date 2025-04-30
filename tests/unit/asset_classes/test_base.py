"""
Unit tests for the base asset and asset class implementations.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from src.asset_classes.base import Asset, AssetClass

class ConcreteAsset(Asset):
    """Concrete implementation of Asset for testing."""
    
    def get_data(self, start_date, end_date):
        """Mock implementation of get_data."""
        return pd.DataFrame()
    
    def get_metadata(self):
        """Mock implementation of get_metadata."""
        return {"symbol": self.symbol}
    
    def get_current_price(self):
        """Mock implementation of get_current_price."""
        return 100.0
    
    def get_trading_hours(self):
        """Mock implementation of get_trading_hours."""
        return {"trading_hours": "9:30-16:00"}

class ConcreteAssetClass(AssetClass):
    """Concrete implementation of AssetClass for testing."""
    
    def create_asset(self, symbol, **kwargs):
        """Create a concrete asset."""
        asset = ConcreteAsset(symbol=symbol, **kwargs)
        self.add_asset(asset)
        return asset
    
    def get_all_assets(self):
        """Get all assets in this class."""
        return list(self._assets.values())
    
    def get_trading_calendar(self, year):
        """Get trading calendar for the specified year."""
        return pd.DataFrame()
    
    def get_market_holidays(self, year):
        """Get market holidays for the specified year."""
        return []

class TestAsset(unittest.TestCase):
    """Test case for the Asset base class."""
    
    def test_initialization(self):
        """Test that the Asset base class initializes correctly."""
        # Arrange
        symbol = "TEST"
        name = "Test Asset"
        metadata = {"key": "value"}
        
        # Act
        asset = ConcreteAsset(symbol=symbol, name=name, key=metadata["key"])
        
        # Assert
        self.assertEqual(asset.symbol, symbol)
        self.assertEqual(asset.name, name)
        self.assertEqual(asset.metadata["key"], metadata["key"])
    
    def test_default_name(self):
        """Test that name defaults to symbol if not provided."""
        # Arrange & Act
        asset = ConcreteAsset(symbol="TEST")
        
        # Assert
        self.assertEqual(asset.name, "TEST")
    
    def test_string_representation(self):
        """Test string and representation methods."""
        # Arrange
        asset = ConcreteAsset(symbol="TEST", name="Test Asset")
        
        # Act & Assert
        self.assertIn("TEST", str(asset))
        self.assertIn("ConcreteAsset", str(asset))
        self.assertIn("TEST", repr(asset))
        self.assertIn("Test Asset", repr(asset))

class TestAssetClass(unittest.TestCase):
    """Test case for the AssetClass base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.asset_class = ConcreteAssetClass(name="Test Assets")
    
    def test_initialization(self):
        """Test that the AssetClass base class initializes correctly."""
        # Assert
        self.assertEqual(self.asset_class.name, "Test Assets")
        self.assertEqual(len(self.asset_class._assets), 0)
    
    def test_add_and_get_asset(self):
        """Test adding and retrieving assets."""
        # Arrange
        asset = ConcreteAsset(symbol="TEST", name="Test Asset")
        
        # Act
        self.asset_class.add_asset(asset)
        retrieved_asset = self.asset_class.get_asset("TEST")
        
        # Assert
        self.assertEqual(len(self.asset_class._assets), 1)
        self.assertEqual(retrieved_asset, asset)
        self.assertIsNone(self.asset_class.get_asset("NONEXISTENT"))
    
    def test_create_asset(self):
        """Test creating assets through the asset class."""
        # Act
        asset = self.asset_class.create_asset(symbol="TEST", name="Test Asset")
        
        # Assert
        self.assertEqual(len(self.asset_class._assets), 1)
        self.assertEqual(asset.symbol, "TEST")
        self.assertEqual(asset.name, "Test Asset")
        self.assertIn("TEST", self.asset_class._assets)
    
    def test_get_all_assets(self):
        """Test retrieving all assets."""
        # Arrange
        self.asset_class.create_asset(symbol="TEST1")
        self.asset_class.create_asset(symbol="TEST2")
        self.asset_class.create_asset(symbol="TEST3")
        
        # Act
        all_assets = self.asset_class.get_all_assets()
        
        # Assert
        self.assertEqual(len(all_assets), 3)
        symbols = [asset.symbol for asset in all_assets]
        self.assertIn("TEST1", symbols)
        self.assertIn("TEST2", symbols)
        self.assertIn("TEST3", symbols)
    
    def test_string_representation(self):
        """Test string and representation methods."""
        # Act & Assert
        self.assertIn("ConcreteAssetClass", str(self.asset_class))
        self.assertIn("Test Assets", str(self.asset_class))
        self.assertIn("ConcreteAssetClass", repr(self.asset_class))
        self.assertIn("Test Assets", repr(self.asset_class))

if __name__ == "__main__":
    unittest.main() 