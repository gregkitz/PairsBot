"""
Unit tests for the FuturesAsset class.

This module contains tests for the FuturesAsset class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.asset_classes.futures.futures_asset import FuturesAsset
from tests.mocks.mock_ib_connector import MockIBConnector

@pytest.fixture
def mock_ib_connector():
    """Fixture providing a mock IB connector."""
    connector = MockIBConnector()
    connector.connect()
    return connector

@pytest.fixture
def futures_asset(mock_ib_connector):
    """Fixture providing a simple futures asset instance."""
    asset = FuturesAsset(
        symbol="ESM23",
        name="E-mini S&P 500 June 2023",
        exchange="CME",
        contract_size=50.0,
        tick_size=0.25,
        connector=mock_ib_connector
    )
    return asset

class TestFuturesAsset:
    """Test class for FuturesAsset."""
    
    def test_initialization(self):
        """Test that the asset is initialized correctly."""
        asset = FuturesAsset(
            symbol="ESM23",
            name="E-mini S&P 500 June 2023",
            exchange="CME",
            contract_size=50.0,
            tick_size=0.25
        )
        
        assert asset.symbol == "ESM23"
        assert asset.name == "E-mini S&P 500 June 2023"
        assert asset.exchange == "CME"
        assert asset.contract_size == 50.0
        assert asset.tick_size == 0.25
        assert asset.asset_type == "futures"
    
    def test_get_data(self, futures_asset, mock_ib_connector):
        """Test retrieving historical data."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        data = futures_asset.get_data(start_date, end_date)
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
    
    def test_get_metadata(self, futures_asset):
        """Test retrieving metadata."""
        metadata = futures_asset.get_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata['symbol'] == "ESM23"
        assert metadata['name'] == "E-mini S&P 500 June 2023"
        assert metadata['exchange'] == "CME"
        assert metadata['contract_size'] == 50.0
        assert metadata['tick_size'] == 0.25
        assert metadata['asset_type'] == "futures"
    
    def test_get_current_price(self, futures_asset):
        """Test retrieving current price."""
        price = futures_asset.get_current_price()
        
        assert isinstance(price, float)
        assert price > 0
    
    def test_get_contract_months(self, futures_asset):
        """Test retrieving contract months."""
        months = futures_asset.get_contract_months()
        
        assert isinstance(months, list)
        assert len(months) > 0
        # Make sure month codes are in the list (standard CME codes)
        assert all(month in ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"] 
                 for month in months)
    
    def test_get_continuous_contract(self, futures_asset):
        """Test retrieving continuous contract data."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        data = futures_asset.get_continuous_contract(start_date, end_date)
        
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Close' in data.columns
        assert 'Volume' in data.columns
        # Continuous contract should include roll information
        assert 'Roll' in data.columns
    
    def test_get_margin_requirements(self, futures_asset):
        """Test retrieving margin requirements."""
        margin_req = futures_asset.get_margin_requirements()
        
        assert isinstance(margin_req, dict)
        assert 'initial' in margin_req
        assert 'maintenance' in margin_req
        assert margin_req['initial'] > 0
        assert margin_req['maintenance'] > 0
        assert margin_req['initial'] >= margin_req['maintenance']
    
    def test_get_trading_hours(self, futures_asset):
        """Test retrieving trading hours."""
        hours = futures_asset.get_trading_hours()
        
        assert isinstance(hours, dict)
        assert 'regular' in hours
        assert 'open' in hours['regular']
        assert 'close' in hours['regular'] 