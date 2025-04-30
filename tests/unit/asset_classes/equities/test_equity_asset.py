"""
Unit tests for the equity asset and asset class implementations.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from src.asset_classes.equities.equity_asset import EquityAsset, EquityAssetClass
from src.connectors.ib import IBConnector

class TestEquityAsset(unittest.TestCase):
    """Test cases for the EquityAsset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector with all required methods
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Add required methods that might be missing from the spec
        self.mock_connector.get_available_symbols = MagicMock()
        self.mock_connector.get_fundamental_data = MagicMock()
        self.mock_connector.get_short_interest = MagicMock()
        self.mock_connector.get_option_chain = MagicMock()
        self.mock_connector.get_sectors = MagicMock()
        
        # Set up mock return values
        self.mock_connector.get_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
            'Volume': [1000] * 10
        })
        
        self.mock_connector.get_market_data.return_value = {
            'last_price': 102.5,
            'bid': 102.0,
            'ask': 103.0,
            'volume': 5000
        }
        
        self.mock_connector.get_contract_details.return_value = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'primary_exchange': 'NASDAQ',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 2500000000000,
            'currency': 'USD',
            'average_volume': 50000000,
            'trading_hours': '9:30-16:00',
            'liquid_hours': '9:30-16:00',
            'timezone': 'America/New_York'
        }
        
        # Create the equity asset
        self.equity = EquityAsset(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            primary_exchange="NASDAQ",
            sector="Technology",
            industry="Consumer Electronics",
            connector=self.mock_connector
        )
    
    def test_initialization(self):
        """Test that the EquityAsset initializes correctly."""
        # Assert
        self.assertEqual(self.equity.symbol, "AAPL")
        self.assertEqual(self.equity.name, "Apple Inc.")
        self.assertEqual(self.equity.exchange, "NASDAQ")
        self.assertEqual(self.equity.primary_exchange, "NASDAQ")
        self.assertEqual(self.equity.sector, "Technology")
        self.assertEqual(self.equity.industry, "Consumer Electronics")
        self.assertEqual(self.equity.connector, self.mock_connector)
    
    def test_get_data(self):
        """Test the get_data method."""
        # Arrange
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Act
        data = self.equity.get_data(start_date, end_date)
        
        # Assert
        self.mock_connector.get_historical_data.assert_called_once_with(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            exchange="NASDAQ",
            security_type='STK'
        )
        self.assertFalse(data.empty)
    
    def test_get_data_exception(self):
        """Test the get_data method handles exceptions."""
        # Arrange
        self.mock_connector.get_historical_data.side_effect = Exception("API Error")
        
        # Act
        data = self.equity.get_data("2023-01-01", "2023-01-10")
        
        # Assert
        self.assertTrue(data.empty)
    
    def test_get_metadata(self):
        """Test the get_metadata method."""
        # Act
        metadata = self.equity.get_metadata()
        
        # Assert
        self.mock_connector.get_contract_details.assert_called_once_with(
            symbol="AAPL",
            exchange="NASDAQ",
            security_type='STK'
        )
        self.assertEqual(metadata["symbol"], "AAPL")
        self.assertEqual(metadata["name"], "Apple Inc.")
        self.assertEqual(metadata["exchange"], "NASDAQ")
        self.assertEqual(metadata["primary_exchange"], "NASDAQ")
        self.assertEqual(metadata["sector"], "Technology")
        self.assertEqual(metadata["industry"], "Consumer Electronics")
        self.assertEqual(metadata["market_cap"], 2500000000000)
    
    def test_get_metadata_exception(self):
        """Test the get_metadata method handles exceptions."""
        # Arrange
        self.mock_connector.get_contract_details.side_effect = Exception("API Error")
        
        # Act
        metadata = self.equity.get_metadata()
        
        # Assert
        self.assertEqual(metadata["symbol"], "AAPL")
        self.assertEqual(metadata["name"], "Apple Inc.")
        self.assertEqual(metadata["exchange"], "NASDAQ")
        self.assertEqual(metadata["primary_exchange"], "NASDAQ")
        self.assertEqual(metadata["sector"], "Technology")
        self.assertEqual(metadata["industry"], "Consumer Electronics")
    
    def test_get_current_price(self):
        """Test the get_current_price method."""
        # Act
        price = self.equity.get_current_price()
        
        # Assert
        self.mock_connector.get_market_data.assert_called_once_with(
            symbol="AAPL",
            exchange="NASDAQ",
            security_type='STK'
        )
        self.assertEqual(price, 102.5)
    
    def test_get_current_price_exception(self):
        """Test the get_current_price method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_data.side_effect = Exception("API Error")
        
        # Act
        price = self.equity.get_current_price()
        
        # Assert
        self.assertEqual(price, 0.0)
    
    def test_get_trading_hours(self):
        """Test the get_trading_hours method."""
        # Act
        hours = self.equity.get_trading_hours()
        
        # Assert
        self.mock_connector.get_contract_details.assert_called_with(
            symbol="AAPL",
            exchange="NASDAQ",
            security_type='STK'
        )
        self.assertEqual(hours["trading_hours"], "9:30-16:00")
        self.assertEqual(hours["liquid_hours"], "9:30-16:00")
        self.assertEqual(hours["timezone"], "America/New_York")
    
    def test_get_trading_hours_exception(self):
        """Test the get_trading_hours method handles exceptions."""
        # Arrange
        self.mock_connector.get_contract_details.side_effect = Exception("API Error")
        
        # Act
        hours = self.equity.get_trading_hours()
        
        # Assert
        self.assertEqual(hours, {})

class TestEquityAssetClass(unittest.TestCase):
    """Test cases for the EquityAssetClass class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector with all required methods
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Add required methods that might be missing from the spec
        self.mock_connector.get_available_symbols = MagicMock()
        self.mock_connector.get_trading_calendar = MagicMock()
        self.mock_connector.get_market_holidays = MagicMock()
        self.mock_connector.get_sectors = MagicMock()
        
        # Set up mock return values
        self.mock_connector.get_available_symbols.return_value = [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'exchange': 'NASDAQ',
                'primary_exchange': 'NASDAQ',
                'sector': 'Technology',
                'industry': 'Consumer Electronics'
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'exchange': 'NASDAQ',
                'primary_exchange': 'NASDAQ',
                'sector': 'Technology',
                'industry': 'Software'
            },
            {
                'symbol': 'SPY',
                'name': 'SPDR S&P 500 ETF',
                'exchange': 'NYSE',
                'primary_exchange': 'NYSE',
                'security_type': 'ETF'
            }
        ]
        
        self.mock_connector.get_trading_calendar.return_value = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=252, freq='B'),
            'is_trading_day': [True] * 252,
            'trading_hours': ['9:30-16:00'] * 252
        })
        
        self.mock_connector.get_market_holidays.return_value = [
            datetime(2023, 1, 2),   # New Year's Day (observed)
            datetime(2023, 1, 16),  # MLK Day
            datetime(2023, 2, 20),  # Presidents' Day
            datetime(2023, 4, 7),   # Good Friday
            datetime(2023, 5, 29),  # Memorial Day
            datetime(2023, 6, 19),  # Juneteenth
            datetime(2023, 7, 4)    # Independence Day
        ]
        
        self.mock_connector.get_sectors.return_value = [
            'Technology',
            'Healthcare',
            'Financials',
            'Consumer Discretionary',
            'Industrials'
        ]
        
        # Create the equity asset class
        self.equity_class = EquityAssetClass(connector=self.mock_connector)
    
    def test_initialization(self):
        """Test that the EquityAssetClass initializes correctly."""
        # Assert
        self.assertEqual(self.equity_class.name, "Equities")
        self.assertEqual(self.equity_class.connector, self.mock_connector)
        self.assertEqual(len(self.equity_class._assets), 0)
    
    def test_create_asset(self):
        """Test the create_asset method."""
        # Act
        asset = self.equity_class.create_asset(
            symbol="GOOG",
            name="Alphabet Inc.",
            exchange="NASDAQ",
            sector="Technology"
        )
        
        # Assert
        self.assertIsInstance(asset, EquityAsset)
        self.assertEqual(asset.symbol, "GOOG")
        self.assertEqual(asset.name, "Alphabet Inc.")
        self.assertEqual(asset.exchange, "NASDAQ")
        self.assertEqual(asset.sector, "Technology")
        self.assertEqual(asset.connector, self.mock_connector)
        self.assertEqual(len(self.equity_class._assets), 1)
        self.assertIn("GOOG", self.equity_class._assets)
    
    def test_get_all_assets(self):
        """Test the get_all_assets method."""
        # Act
        assets = self.equity_class.get_all_assets()
        
        # Assert
        self.mock_connector.get_available_symbols.assert_called_once_with(security_type='STK')
        self.assertEqual(len(assets), 3)
        symbols = [asset.symbol for asset in assets]
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
        self.assertIn("SPY", symbols)
    
    def test_get_all_assets_exception(self):
        """Test the get_all_assets method handles exceptions."""
        # Arrange
        self.mock_connector.get_available_symbols.side_effect = Exception("API Error")
        
        # Act
        assets = self.equity_class.get_all_assets()
        
        # Assert
        self.assertEqual(len(assets), 0)
    
    def test_get_trading_calendar(self):
        """Test the get_trading_calendar method."""
        # Act
        calendar = self.equity_class.get_trading_calendar(2023)
        
        # Assert
        self.mock_connector.get_trading_calendar.assert_called_once_with(
            year=2023,
            security_type='STK'
        )
        self.assertFalse(calendar.empty)
        self.assertEqual(len(calendar), 252)  # Typical number of trading days in a year
    
    def test_get_trading_calendar_exception(self):
        """Test the get_trading_calendar method handles exceptions."""
        # Arrange
        self.mock_connector.get_trading_calendar.side_effect = Exception("API Error")
        
        # Act
        calendar = self.equity_class.get_trading_calendar(2023)
        
        # Assert
        self.assertTrue(calendar.empty)
    
    def test_get_market_holidays(self):
        """Test the get_market_holidays method."""
        # Act
        holidays = self.equity_class.get_market_holidays(2023)
        
        # Assert
        self.mock_connector.get_market_holidays.assert_called_once_with(
            year=2023,
            security_type='STK'
        )
        self.assertEqual(len(holidays), 7)  # Typical number of market holidays in a year
    
    def test_get_market_holidays_exception(self):
        """Test the get_market_holidays method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_holidays.side_effect = Exception("API Error")
        
        # Act
        holidays = self.equity_class.get_market_holidays(2023)
        
        # Assert
        self.assertEqual(len(holidays), 0)
    
    def test_get_sectors(self):
        """Test the get_sectors method."""
        # Act
        sectors = self.equity_class.get_sectors()
        
        # Assert
        self.mock_connector.get_sectors.assert_called_once()
        self.assertEqual(len(sectors), 5)
        self.assertIn("Technology", sectors)
    
    def test_get_sectors_exception(self):
        """Test the get_sectors method handles exceptions."""
        # Arrange
        self.mock_connector.get_sectors.side_effect = Exception("API Error")
        
        # Act
        sectors = self.equity_class.get_sectors()
        
        # Assert
        self.assertEqual(len(sectors), 0)
    
    def test_get_equities_by_sector(self):
        """Test the get_equities_by_sector method."""
        # Arrange
        self.equity_class.get_all_assets()  # Populate with mock assets
        
        # Act
        tech_assets = self.equity_class.get_equities_by_sector("Technology")
        
        # Assert
        self.assertEqual(len(tech_assets), 2)  # AAPL and MSFT
        symbols = [asset.symbol for asset in tech_assets]
        self.assertIn("AAPL", symbols)
        self.assertIn("MSFT", symbols)
    
    def test_get_etfs(self):
        """Test the get_etfs method."""
        # Arrange
        self.equity_class.get_all_assets()  # Populate with mock assets
        
        # Force SPY to have ETF security_type in the mock
        for asset in self.equity_class._assets.values():
            if asset.symbol == "SPY":
                asset.metadata = {"security_type": "ETF"}
        
        # Act
        etfs = self.equity_class.get_etfs()
        
        # Assert
        self.assertEqual(len(etfs), 1)  # SPY
        self.assertEqual(etfs[0].symbol, "SPY")
    
    def test_get_etfs_exception(self):
        """Test the get_etfs method handles exceptions."""
        # Arrange
        self.mock_connector.get_available_symbols.side_effect = Exception("API Error")
        
        # Act
        etfs = self.equity_class.get_etfs()
        
        # Assert
        self.assertEqual(len(etfs), 0)

if __name__ == "__main__":
    unittest.main() 