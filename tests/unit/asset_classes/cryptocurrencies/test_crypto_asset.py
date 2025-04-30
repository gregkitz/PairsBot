"""
Unit tests for the cryptocurrency asset and asset class implementations.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from src.asset_classes.cryptocurrencies.crypto_asset import CryptoAsset, CryptoAssetClass
from src.connectors.ib import IBConnector

class TestCryptoAsset(unittest.TestCase):
    """Test cases for the CryptoAsset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector with all required methods
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Add required methods that might be missing from the spec
        self.mock_connector.get_orderbook = MagicMock()
        self.mock_connector.get_funding_rate = MagicMock()
        self.mock_connector.get_available_symbols = MagicMock()
        
        # Set up mock return values
        self.mock_connector.get_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Open': [50000] * 10,
            'High': [51000] * 10,
            'Low': [49000] * 10,
            'Close': [50500] * 10,
            'Volume': [1000] * 10
        })
        
        self.mock_connector.get_market_data.return_value = {
            'last_price': 50500.5,
            'bid': 50500.0,
            'ask': 50501.0,
            'volume': 5000
        }
        
        self.mock_connector.get_contract_details.return_value = {
            'symbol': 'BTC',
            'name': 'Bitcoin',
            'market_type': 'spot',
            'market_cap': 1000000000000,
            'circulating_supply': 19000000,
            'max_supply': 21000000
        }
        
        self.mock_connector.get_orderbook.return_value = {
            'bids': [(50500.0, 1.5), (50499.5, 2.0), (50499.0, 3.0)],
            'asks': [(50501.0, 1.0), (50501.5, 2.5), (50502.0, 3.5)]
        }
        
        self.mock_connector.get_funding_rate.return_value = 0.01  # 1%
        
        # Create the crypto asset
        self.crypto = CryptoAsset(
            symbol="BTC",
            name="Bitcoin",
            exchange="Coinbase",
            market_type="spot",
            is_stablecoin=False,
            connector=self.mock_connector
        )
    
    def test_initialization(self):
        """Test that the CryptoAsset initializes correctly."""
        # Assert
        self.assertEqual(self.crypto.symbol, "BTC")
        self.assertEqual(self.crypto.name, "Bitcoin")
        self.assertEqual(self.crypto.exchange, "Coinbase")
        self.assertEqual(self.crypto.market_type, "spot")
        self.assertFalse(self.crypto.is_stablecoin)
        self.assertEqual(self.crypto.connector, self.mock_connector)
    
    def test_get_data(self):
        """Test the get_data method."""
        # Arrange
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Act
        data = self.crypto.get_data(start_date, end_date)
        
        # Assert
        self.mock_connector.get_historical_data.assert_called_once_with(
            symbol="BTC",
            start_date=start_date,
            end_date=end_date,
            exchange="Coinbase",
            security_type='CRYPTO'
        )
        self.assertFalse(data.empty)
    
    def test_get_data_exception(self):
        """Test the get_data method handles exceptions."""
        # Arrange
        self.mock_connector.get_historical_data.side_effect = Exception("API Error")
        
        # Act
        data = self.crypto.get_data("2023-01-01", "2023-01-10")
        
        # Assert
        self.assertTrue(data.empty)
    
    def test_get_metadata(self):
        """Test the get_metadata method."""
        # Act
        metadata = self.crypto.get_metadata()
        
        # Assert
        self.mock_connector.get_contract_details.assert_called_once_with(
            symbol="BTC",
            exchange="Coinbase",
            security_type='CRYPTO'
        )
        self.assertEqual(metadata["symbol"], "BTC")
        self.assertEqual(metadata["name"], "Bitcoin")
        self.assertEqual(metadata["exchange"], "Coinbase")
        self.assertEqual(metadata["market_type"], "spot")
        self.assertFalse(metadata["is_stablecoin"])
        self.assertEqual(metadata["market_cap"], 1000000000000)
        self.assertEqual(metadata["circulating_supply"], 19000000)
        self.assertEqual(metadata["max_supply"], 21000000)
    
    def test_get_metadata_exception(self):
        """Test the get_metadata method handles exceptions."""
        # Arrange
        self.mock_connector.get_contract_details.side_effect = Exception("API Error")
        
        # Act
        metadata = self.crypto.get_metadata()
        
        # Assert
        self.assertEqual(metadata["symbol"], "BTC")
        self.assertEqual(metadata["name"], "Bitcoin")
        self.assertEqual(metadata["exchange"], "Coinbase")
        self.assertEqual(metadata["market_type"], "spot")
        self.assertFalse(metadata["is_stablecoin"])
    
    def test_get_current_price(self):
        """Test the get_current_price method."""
        # Act
        price = self.crypto.get_current_price()
        
        # Assert
        self.mock_connector.get_market_data.assert_called_once_with(
            symbol="BTC",
            exchange="Coinbase",
            security_type='CRYPTO'
        )
        self.assertEqual(price, 50500.5)
    
    def test_get_current_price_exception(self):
        """Test the get_current_price method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_data.side_effect = Exception("API Error")
        
        # Act
        price = self.crypto.get_current_price()
        
        # Assert
        self.assertEqual(price, 0.0)
    
    def test_get_trading_hours(self):
        """Test the get_trading_hours method."""
        # Act
        hours = self.crypto.get_trading_hours()
        
        # Assert
        # No connector call needed as crypto markets are 24/7
        self.assertEqual(hours["trading_hours"], "24/7")
        self.assertEqual(hours["liquid_hours"], "24/7")
        self.assertEqual(hours["timezone"], "UTC")
    
    def test_get_orderbook(self):
        """Test the get_orderbook method."""
        # Act
        orderbook = self.crypto.get_orderbook(depth=5)
        
        # Assert
        self.mock_connector.get_orderbook.assert_called_once_with(
            symbol="BTC",
            exchange="Coinbase",
            depth=5
        )
        self.assertIn("bids", orderbook)
        self.assertIn("asks", orderbook)
        self.assertEqual(len(orderbook["bids"]), 3)
        self.assertEqual(len(orderbook["asks"]), 3)
    
    def test_get_orderbook_exception(self):
        """Test the get_orderbook method handles exceptions."""
        # Arrange
        self.mock_connector.get_orderbook.side_effect = Exception("API Error")
        
        # Act
        orderbook = self.crypto.get_orderbook()
        
        # Assert
        self.assertEqual(orderbook, {"bids": [], "asks": []})
    
    def test_get_funding_rate_not_perpetual(self):
        """Test the get_funding_rate method for non-perpetual contracts."""
        # Act
        rate = self.crypto.get_funding_rate()
        
        # Assert
        self.assertEqual(rate, 0.0)  # Should be 0 for spot markets
    
    def test_get_funding_rate_perpetual(self):
        """Test the get_funding_rate method for perpetual contracts."""
        # Arrange
        self.crypto.market_type = "perpetual"
        
        # Act
        rate = self.crypto.get_funding_rate()
        
        # Assert
        self.mock_connector.get_funding_rate.assert_called_once_with(
            symbol="BTC",
            exchange="Coinbase"
        )
        self.assertEqual(rate, 0.01)  # 1%
    
    def test_get_funding_rate_exception(self):
        """Test the get_funding_rate method handles exceptions."""
        # Arrange
        self.crypto.market_type = "perpetual"
        self.mock_connector.get_funding_rate.side_effect = Exception("API Error")
        
        # Act
        rate = self.crypto.get_funding_rate()
        
        # Assert
        self.assertEqual(rate, 0.0)
    
    def test_get_liquidation_price_long(self):
        """Test the get_liquidation_price method for long positions."""
        # Arrange
        # Mock current price
        self.mock_connector.get_market_data.return_value = {'last_price': 50000.0}
        position_size = 1.0  # 1 BTC long
        leverage = 5.0  # 5x leverage
        
        # Act
        liquidation_price = self.crypto.get_liquidation_price(position_size, leverage)
        
        # Assert
        # For long positions: liquidation_price = entry_price * (1 - 1/leverage)
        # 50000 * (1 - 1/5) = 50000 * 0.8 = 40000
        self.assertEqual(liquidation_price, 40000.0)
    
    def test_get_liquidation_price_short(self):
        """Test the get_liquidation_price method for short positions."""
        # Arrange
        # Mock current price
        self.mock_connector.get_market_data.return_value = {'last_price': 50000.0}
        position_size = -1.0  # 1 BTC short
        leverage = 5.0  # 5x leverage
        
        # Act
        liquidation_price = self.crypto.get_liquidation_price(position_size, leverage)
        
        # Assert
        # For short positions: liquidation_price = entry_price * (1 + 1/leverage)
        # 50000 * (1 + 1/5) = 50000 * 1.2 = 60000
        self.assertEqual(liquidation_price, 60000.0)
    
    def test_get_liquidation_price_exception(self):
        """Test the get_liquidation_price method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_data.side_effect = Exception("API Error")
        
        # Act
        liquidation_price = self.crypto.get_liquidation_price(1.0, 5.0)
        
        # Assert
        self.assertEqual(liquidation_price, 0.0)

class TestCryptoAssetClass(unittest.TestCase):
    """Test cases for the CryptoAssetClass class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector with all required methods
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Add required methods that might be missing from the spec
        self.mock_connector.get_orderbook = MagicMock()
        self.mock_connector.get_funding_rate = MagicMock()
        self.mock_connector.get_available_symbols = MagicMock()
        
        # Set up mock return values
        self.mock_connector.get_available_symbols.return_value = [
            {
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'exchange': 'Coinbase',
                'market_type': 'spot',
                'is_stablecoin': False
            },
            {
                'symbol': 'ETH',
                'name': 'Ethereum',
                'exchange': 'Coinbase',
                'market_type': 'spot',
                'is_stablecoin': False
            },
            {
                'symbol': 'USDT',
                'name': 'Tether',
                'exchange': 'Binance',
                'market_type': 'spot',
                'is_stablecoin': True
            }
        ]
        
        # Create the crypto asset class
        self.crypto_class = CryptoAssetClass(connector=self.mock_connector)
    
    def test_initialization(self):
        """Test that the CryptoAssetClass initializes correctly."""
        # Assert
        self.assertEqual(self.crypto_class.name, "Cryptocurrencies")
        self.assertEqual(self.crypto_class.connector, self.mock_connector)
        self.assertEqual(len(self.crypto_class._assets), 0)
    
    def test_create_asset(self):
        """Test the create_asset method."""
        # Act
        asset = self.crypto_class.create_asset(
            symbol="SOL",
            name="Solana",
            exchange="Binance",
            market_type="spot"
        )
        
        # Assert
        self.assertIsInstance(asset, CryptoAsset)
        self.assertEqual(asset.symbol, "SOL")
        self.assertEqual(asset.name, "Solana")
        self.assertEqual(asset.exchange, "Binance")
        self.assertEqual(asset.market_type, "spot")
        self.assertEqual(asset.connector, self.mock_connector)
        self.assertEqual(len(self.crypto_class._assets), 1)
        self.assertIn("SOL", self.crypto_class._assets)
    
    def test_get_all_assets(self):
        """Test the get_all_assets method."""
        # Act
        assets = self.crypto_class.get_all_assets()
        
        # Assert
        self.mock_connector.get_available_symbols.assert_called_once_with(security_type='CRYPTO')
        self.assertEqual(len(assets), 3)
        symbols = [asset.symbol for asset in assets]
        self.assertIn("BTC", symbols)
        self.assertIn("ETH", symbols)
        self.assertIn("USDT", symbols)
    
    def test_get_all_assets_exception(self):
        """Test the get_all_assets method handles exceptions."""
        # Arrange
        self.mock_connector.get_available_symbols.side_effect = Exception("API Error")
        
        # Act
        assets = self.crypto_class.get_all_assets()
        
        # Assert
        self.assertEqual(len(assets), 0)
    
    def test_get_trading_calendar(self):
        """Test the get_trading_calendar method."""
        # Act
        calendar = self.crypto_class.get_trading_calendar(2023)
        
        # Assert
        # No connector call as crypto markets are 24/7
        self.assertFalse(calendar.empty)
        self.assertEqual(len(calendar), 365)  # Full year of trading
        self.assertTrue(all(calendar['is_trading_day']))  # All days are trading days
        self.assertTrue(all(calendar['trading_hours'] == '24/7'))  # 24/7 trading
    
    def test_get_market_holidays(self):
        """Test the get_market_holidays method."""
        # Act
        holidays = self.crypto_class.get_market_holidays(2023)
        
        # Assert
        # No connector call as crypto markets don't have holidays
        self.assertEqual(len(holidays), 0)  # No holidays for 24/7 markets
    
    def test_get_top_assets_by_market_cap(self):
        """Test the get_top_assets_by_market_cap method."""
        # Arrange
        # Get assets loaded
        self.crypto_class.get_all_assets()
        
        # Mock get_metadata to return market caps
        with patch.object(CryptoAsset, 'get_metadata') as mock_get_metadata:
            mock_get_metadata.side_effect = [
                {'market_cap': 1000000000000},  # BTC
                {'market_cap': 500000000000},   # ETH
                {'market_cap': 100000000000}    # USDT
            ]
            
            # Act
            top_assets = self.crypto_class.get_top_assets_by_market_cap(limit=2)
            
            # Assert
            self.assertEqual(len(top_assets), 2)
            self.assertEqual(top_assets[0].symbol, "BTC")  # Highest market cap
            self.assertEqual(top_assets[1].symbol, "ETH")  # Second highest
    
    def test_get_top_assets_by_market_cap_exception(self):
        """Test the get_top_assets_by_market_cap method handles exceptions."""
        # Arrange
        self.mock_connector.get_available_symbols.side_effect = Exception("API Error")
        
        # Act
        top_assets = self.crypto_class.get_top_assets_by_market_cap()
        
        # Assert
        self.assertEqual(len(top_assets), 0)
    
    def test_get_stablecoins(self):
        """Test the get_stablecoins method."""
        # Arrange
        # Get assets loaded
        self.crypto_class.get_all_assets()
        
        # Act
        stablecoins = self.crypto_class.get_stablecoins()
        
        # Assert
        self.assertEqual(len(stablecoins), 1)
        self.assertEqual(stablecoins[0].symbol, "USDT")
        self.assertTrue(stablecoins[0].is_stablecoin)

if __name__ == "__main__":
    unittest.main() 