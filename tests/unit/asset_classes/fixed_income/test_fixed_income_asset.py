"""
Unit tests for the fixed income asset and asset class implementations.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime, date

from src.asset_classes.fixed_income.fixed_income_asset import FixedIncomeAsset, FixedIncomeAssetClass
from src.connectors.ib import IBConnector

class TestFixedIncomeAsset(unittest.TestCase):
    """Test cases for the FixedIncomeAsset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Set up mock return values
        self.mock_connector.get_historical_data.return_value = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=10),
            'Open': [98.5] * 10,
            'High': [99.0] * 10,
            'Low': [98.0] * 10,
            'Close': [98.75] * 10,
            'Volume': [1000] * 10
        })
        
        self.mock_connector.get_market_data.return_value = {
            'last_price': 98.75,
            'bid': 98.70,
            'ask': 98.80,
            'volume': 500
        }
        
        self.mock_connector.get_contract_details.return_value = {
            'symbol': 'T',
            'name': 'US Treasury 10Y',
            'maturity_date': '2033-01-15',
            'coupon': 4.5,
            'bond_type': 'government',
            'issuer': 'US Treasury',
            'rating': 'AAA',
            'face_value': 100,
            'currency': 'USD',
            'trading_hours': '8:00-17:00',
            'liquid_hours': '8:00-17:00',
            'timezone': 'America/New_York'
        }
        
        # Create the fixed income asset
        self.fixed_income = FixedIncomeAsset(
            symbol="T",
            name="US Treasury 10Y",
            exchange="NYSE",
            maturity_date="2033-01-15",
            coupon=4.5,
            bond_type="government",
            issuer="US Treasury",
            connector=self.mock_connector
        )
    
    def test_initialization(self):
        """Test that the FixedIncomeAsset initializes correctly."""
        # Assert
        self.assertEqual(self.fixed_income.symbol, "T")
        self.assertEqual(self.fixed_income.name, "US Treasury 10Y")
        self.assertEqual(self.fixed_income.exchange, "NYSE")
        self.assertEqual(self.fixed_income.maturity_date, "2033-01-15")
        self.assertEqual(self.fixed_income.coupon, 4.5)
        self.assertEqual(self.fixed_income.bond_type, "government")
        self.assertEqual(self.fixed_income.issuer, "US Treasury")
        self.assertEqual(self.fixed_income.connector, self.mock_connector)
    
    def test_get_data(self):
        """Test the get_data method."""
        # Arrange
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Act
        data = self.fixed_income.get_data(start_date, end_date)
        
        # Assert
        self.mock_connector.get_historical_data.assert_called_once_with(
            symbol="T",
            start_date=start_date,
            end_date=end_date,
            exchange="NYSE",
            security_type='BOND'
        )
        self.assertFalse(data.empty)
    
    def test_get_data_exception(self):
        """Test the get_data method handles exceptions."""
        # Arrange
        self.mock_connector.get_historical_data.side_effect = Exception("API Error")
        
        # Act
        data = self.fixed_income.get_data("2023-01-01", "2023-01-10")
        
        # Assert
        self.assertTrue(data.empty)
    
    def test_get_metadata(self):
        """Test the get_metadata method."""
        # Act
        metadata = self.fixed_income.get_metadata()
        
        # Assert
        self.mock_connector.get_contract_details.assert_called_once_with(
            symbol="T",
            exchange="NYSE",
            security_type='BOND'
        )
        self.assertEqual(metadata["symbol"], "T")
        self.assertEqual(metadata["name"], "US Treasury 10Y")
        self.assertEqual(metadata["exchange"], "NYSE")
        self.assertEqual(metadata["maturity_date"], "2033-01-15")
        self.assertEqual(metadata["coupon"], 4.5)
        self.assertEqual(metadata["bond_type"], "government")
        self.assertEqual(metadata["issuer"], "US Treasury")
        self.assertEqual(metadata["rating"], "AAA")
        self.assertEqual(metadata["face_value"], 100)
    
    def test_get_metadata_exception(self):
        """Test the get_metadata method handles exceptions."""
        # Arrange
        self.mock_connector.get_contract_details.side_effect = Exception("API Error")
        
        # Act
        metadata = self.fixed_income.get_metadata()
        
        # Assert
        self.assertEqual(metadata["symbol"], "T")
        self.assertEqual(metadata["name"], "US Treasury 10Y")
        self.assertEqual(metadata["exchange"], "NYSE")
        self.assertEqual(metadata["maturity_date"], "2033-01-15")
        self.assertEqual(metadata["coupon"], 4.5)
        self.assertEqual(metadata["bond_type"], "government")
        self.assertEqual(metadata["issuer"], "US Treasury")
    
    def test_get_current_price(self):
        """Test the get_current_price method."""
        # Act
        price = self.fixed_income.get_current_price()
        
        # Assert
        self.mock_connector.get_market_data.assert_called_once_with(
            symbol="T",
            exchange="NYSE",
            security_type='BOND'
        )
        self.assertEqual(price, 98.75)
    
    def test_get_current_price_exception(self):
        """Test the get_current_price method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_data.side_effect = Exception("API Error")
        
        # Act
        price = self.fixed_income.get_current_price()
        
        # Assert
        self.assertEqual(price, 0.0)
    
    def test_get_trading_hours(self):
        """Test the get_trading_hours method."""
        # Act
        hours = self.fixed_income.get_trading_hours()
        
        # Assert
        self.mock_connector.get_contract_details.assert_called_with(
            symbol="T",
            exchange="NYSE",
            security_type='BOND'
        )
        self.assertEqual(hours["trading_hours"], "8:00-17:00")
        self.assertEqual(hours["liquid_hours"], "8:00-17:00")
        self.assertEqual(hours["timezone"], "America/New_York")
    
    def test_get_trading_hours_exception(self):
        """Test the get_trading_hours method handles exceptions."""
        # Arrange
        self.mock_connector.get_contract_details.side_effect = Exception("API Error")
        
        # Act
        hours = self.fixed_income.get_trading_hours()
        
        # Assert
        self.assertEqual(hours, {})
    
    @patch('src.asset_classes.fixed_income.fixed_income_asset.date')
    def test_get_yield_to_maturity(self, mock_date):
        """Test the get_yield_to_maturity method."""
        # Arrange
        # Set a fixed date for testing
        mock_date.today.return_value = date(2023, 1, 15)
        
        # Mock the get_current_price method
        self.fixed_income.get_current_price = MagicMock(return_value=98.75)
        
        # Set a fixed maturity date for 10 years from today
        self.fixed_income.maturity_date = "2033-01-15"
        
        # Mock the get_metadata method
        self.fixed_income.get_metadata = MagicMock(return_value={
            "coupon": 4.5,
            "face_value": 100
        })
        
        # Act
        ytm = self.fixed_income.get_yield_to_maturity()
        
        # Assert
        # Check that YTM is calculated and is a reasonable value (approximate check)
        self.assertGreater(ytm, 0)
        self.assertLess(ytm, 10)  # Should be less than 10% for a typical treasury
    
    def test_get_yield_to_maturity_price_zero(self):
        """Test the get_yield_to_maturity method when price is zero."""
        # Arrange
        self.fixed_income.get_current_price = MagicMock(return_value=0)
        
        # Act
        ytm = self.fixed_income.get_yield_to_maturity()
        
        # Assert
        self.assertEqual(ytm, 0)
    
    def test_get_yield_to_maturity_no_maturity(self):
        """Test the get_yield_to_maturity method when maturity date is missing."""
        # Arrange
        self.fixed_income.maturity_date = None
        
        # Act
        ytm = self.fixed_income.get_yield_to_maturity()
        
        # Assert
        self.assertEqual(ytm, 0)
    
    @patch('src.asset_classes.fixed_income.fixed_income_asset.date')
    def test_get_duration(self, mock_date):
        """Test the get_duration method."""
        # Arrange
        # Set a fixed date for testing
        mock_date.today.return_value = date(2023, 1, 15)
        
        # Mock the get_yield_to_maturity method
        self.fixed_income.get_yield_to_maturity = MagicMock(return_value=4.75)
        
        # Set a fixed maturity date for 10 years from today
        self.fixed_income.maturity_date = "2033-01-15"
        
        # Mock the get_metadata method
        self.fixed_income.get_metadata = MagicMock(return_value={
            "coupon": 4.5,
            "face_value": 100
        })
        
        # Act
        duration = self.fixed_income.get_duration()
        
        # Assert
        # Modified duration for a 10-year bond with 4.5% coupon should be around 7-8 years
        self.assertGreater(duration, 6)
        self.assertLess(duration, 9)
    
    def test_get_duration_ytm_zero(self):
        """Test the get_duration method when YTM is zero."""
        # Arrange
        self.fixed_income.get_yield_to_maturity = MagicMock(return_value=0)
        
        # Act
        duration = self.fixed_income.get_duration()
        
        # Assert
        self.assertEqual(duration, 0)
    
    def test_get_duration_no_maturity(self):
        """Test the get_duration method when maturity date is missing."""
        # Arrange
        self.fixed_income.maturity_date = None
        
        # Act
        duration = self.fixed_income.get_duration()
        
        # Assert
        self.assertEqual(duration, 0)

class TestFixedIncomeAssetClass(unittest.TestCase):
    """Test cases for the FixedIncomeAssetClass class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock connector with all required methods
        self.mock_connector = MagicMock(spec=IBConnector)
        
        # Add required methods that might be missing from the spec
        self.mock_connector.get_available_symbols = MagicMock()
        self.mock_connector.get_trading_calendar = MagicMock()
        self.mock_connector.get_market_holidays = MagicMock()
        self.mock_connector.get_yield_curve = MagicMock()
        
        # Set up mock return values
        self.mock_connector.get_available_symbols.return_value = [
            {
                'symbol': 'T',
                'name': 'US Treasury 10Y',
                'exchange': 'NYSE',
                'maturity_date': '2033-01-15',
                'coupon': 4.5,
                'bond_type': 'government',
                'issuer': 'US Treasury'
            },
            {
                'symbol': 'C',
                'name': 'Citigroup Bond',
                'exchange': 'NYSE',
                'maturity_date': '2030-06-30',
                'coupon': 5.0,
                'bond_type': 'corporate',
                'issuer': 'Citigroup'
            },
            {
                'symbol': 'MUB',
                'name': 'iShares National Muni Bond ETF',
                'exchange': 'NYSE',
                'maturity_date': None,
                'coupon': None,
                'bond_type': 'municipal',
                'issuer': 'Various'
            }
        ]
        
        self.mock_connector.get_trading_calendar.return_value = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=252, freq='B'),
            'is_trading_day': [True] * 252,
            'trading_hours': ['8:00-17:00'] * 252
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
        
        self.mock_connector.get_yield_curve.return_value = pd.DataFrame({
            'maturity': ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y'],
            'yield': [5.20, 5.25, 5.30, 5.15, 4.80, 4.60, 4.50, 4.40]
        })
        
        # Create the fixed income asset class
        self.fixed_income_class = FixedIncomeAssetClass(connector=self.mock_connector)
    
    def test_initialization(self):
        """Test that the FixedIncomeAssetClass initializes correctly."""
        # Assert
        self.assertEqual(self.fixed_income_class.name, "Fixed Income")
        self.assertEqual(self.fixed_income_class.connector, self.mock_connector)
        self.assertEqual(len(self.fixed_income_class._assets), 0)
    
    def test_create_asset(self):
        """Test the create_asset method."""
        # Act
        asset = self.fixed_income_class.create_asset(
            symbol="TLT",
            name="iShares 20+ Year Treasury Bond ETF",
            exchange="NASDAQ",
            bond_type="government"
        )
        
        # Assert
        self.assertIsInstance(asset, FixedIncomeAsset)
        self.assertEqual(asset.symbol, "TLT")
        self.assertEqual(asset.name, "iShares 20+ Year Treasury Bond ETF")
        self.assertEqual(asset.exchange, "NASDAQ")
        self.assertEqual(asset.bond_type, "government")
        self.assertEqual(asset.connector, self.mock_connector)
        self.assertEqual(len(self.fixed_income_class._assets), 1)
        self.assertIn("TLT", self.fixed_income_class._assets)
    
    def test_get_all_assets(self):
        """Test the get_all_assets method."""
        # Act
        assets = self.fixed_income_class.get_all_assets()
        
        # Assert
        self.mock_connector.get_available_symbols.assert_called_once_with(security_type='BOND')
        self.assertEqual(len(assets), 3)
        symbols = [asset.symbol for asset in assets]
        self.assertIn("T", symbols)
        self.assertIn("C", symbols)
        self.assertIn("MUB", symbols)
    
    def test_get_all_assets_exception(self):
        """Test the get_all_assets method handles exceptions."""
        # Arrange
        self.mock_connector.get_available_symbols.side_effect = Exception("API Error")
        
        # Act
        assets = self.fixed_income_class.get_all_assets()
        
        # Assert
        self.assertEqual(len(assets), 0)
    
    def test_get_trading_calendar(self):
        """Test the get_trading_calendar method."""
        # Act
        calendar = self.fixed_income_class.get_trading_calendar(2023)
        
        # Assert
        self.mock_connector.get_trading_calendar.assert_called_once_with(
            year=2023,
            security_type='BOND'
        )
        self.assertFalse(calendar.empty)
        self.assertEqual(len(calendar), 252)  # Typical number of trading days in a year
    
    def test_get_trading_calendar_exception(self):
        """Test the get_trading_calendar method handles exceptions."""
        # Arrange
        self.mock_connector.get_trading_calendar.side_effect = Exception("API Error")
        
        # Act
        calendar = self.fixed_income_class.get_trading_calendar(2023)
        
        # Assert
        self.assertTrue(calendar.empty)
    
    def test_get_market_holidays(self):
        """Test the get_market_holidays method."""
        # Act
        holidays = self.fixed_income_class.get_market_holidays(2023)
        
        # Assert
        self.mock_connector.get_market_holidays.assert_called_once_with(
            year=2023,
            security_type='BOND'
        )
        self.assertEqual(len(holidays), 7)  # Typical number of market holidays in a year
    
    def test_get_market_holidays_exception(self):
        """Test the get_market_holidays method handles exceptions."""
        # Arrange
        self.mock_connector.get_market_holidays.side_effect = Exception("API Error")
        
        # Act
        holidays = self.fixed_income_class.get_market_holidays(2023)
        
        # Assert
        self.assertEqual(len(holidays), 0)
    
    def test_get_yield_curve(self):
        """Test the get_yield_curve method."""
        # Act
        curve = self.fixed_income_class.get_yield_curve()
        
        # Assert
        self.mock_connector.get_yield_curve.assert_called_once_with(date=None)
        self.assertFalse(curve.empty)
        self.assertEqual(len(curve), 8)  # 8 points on the yield curve
        self.assertIn('yield', curve.columns)
        self.assertIn('maturity', curve.columns)
    
    def test_get_yield_curve_with_date(self):
        """Test the get_yield_curve method with a specific date."""
        # Arrange
        date_param = "2023-01-15"
        
        # Act
        curve = self.fixed_income_class.get_yield_curve(date=date_param)
        
        # Assert
        self.mock_connector.get_yield_curve.assert_called_once_with(date=date_param)
    
    def test_get_yield_curve_exception(self):
        """Test the get_yield_curve method handles exceptions."""
        # Arrange
        self.mock_connector.get_yield_curve.side_effect = Exception("API Error")
        
        # Act
        curve = self.fixed_income_class.get_yield_curve()
        
        # Assert
        self.assertTrue(curve.empty)
    
    def test_get_treasuries(self):
        """Test the get_treasuries method."""
        # Arrange
        # Get assets loaded
        self.fixed_income_class.get_all_assets()
        
        # Act
        treasuries = self.fixed_income_class.get_treasuries()
        
        # Assert
        self.assertEqual(len(treasuries), 1)
        self.assertEqual(treasuries[0].symbol, "T")
        self.assertEqual(treasuries[0].bond_type, "government")
    
    def test_get_corporate_bonds(self):
        """Test the get_corporate_bonds method."""
        # Arrange
        # Get assets loaded
        self.fixed_income_class.get_all_assets()
        
        # Act
        corporates = self.fixed_income_class.get_corporate_bonds()
        
        # Assert
        self.assertEqual(len(corporates), 1)
        self.assertEqual(corporates[0].symbol, "C")
        self.assertEqual(corporates[0].bond_type, "corporate")

if __name__ == "__main__":
    unittest.main() 