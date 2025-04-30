"""
Unit tests for the DataProcessor class.

This module contains tests for the data processing functionality including
loading, cleaning, and preparing data for analysis.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.data_processor.data_processor import DataProcessor
from tests.mocks.mock_ib_connector import MockIBConnector

@pytest.fixture
def mock_ib_connector():
    """Fixture providing a mock IB connector."""
    connector = MockIBConnector()
    connector.connect()
    return connector

@pytest.fixture
def sample_data():
    """Fixture providing sample price data for testing."""
    # Create synthetic data for two futures contracts
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # ES data
    es_df = pd.DataFrame({
        'Open': np.random.normal(4200, 20, len(dates)),
        'High': np.random.normal(4250, 20, len(dates)),
        'Low': np.random.normal(4150, 20, len(dates)),
        'Close': np.random.normal(4200, 20, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # NQ data
    nq_df = pd.DataFrame({
        'Open': np.random.normal(14000, 50, len(dates)),
        'High': np.random.normal(14100, 50, len(dates)),
        'Low': np.random.normal(13900, 50, len(dates)),
        'Close': np.random.normal(14000, 50, len(dates)),
        'Volume': np.random.randint(500, 5000, len(dates))
    }, index=dates)
    
    # Create a copy with missing values for testing
    es_missing = es_df.copy()
    es_missing.iloc[5:10, 0:2] = np.nan
    
    # Return dict of data
    return {
        'ESM23': es_df,
        'NQM23': nq_df,
        'ESM23_missing': es_missing
    }

@pytest.fixture
def data_processor(mock_ib_connector):
    """Fixture providing a DataProcessor instance."""
    processor = DataProcessor()
    # Set the IB connector for data access
    processor.connector = mock_ib_connector
    return processor

class TestDataProcessor:
    """Test class for DataProcessor."""
    
    def test_initialization(self):
        """Test that the data processor is initialized correctly."""
        processor = DataProcessor()
        assert processor is not None
        assert hasattr(processor, 'load_data')
        assert hasattr(processor, 'clean_data')
        assert hasattr(processor, 'resample_data')
    
    def test_load_data(self, data_processor, mock_ib_connector):
        """Test loading data from connector."""
        # Define test parameters
        symbols = ['ESM23', 'NQM23']
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Load data
        data = data_processor.load_data(symbols, start_date, end_date)
        
        # Check results
        assert isinstance(data, dict)
        assert all(symbol in data for symbol in symbols)
        assert all(isinstance(data[symbol], pd.DataFrame) for symbol in symbols)
        assert all(not data[symbol].empty for symbol in symbols)
        
        # Check data structure
        for symbol in symbols:
            df = data[symbol]
            assert 'Open' in df.columns
            assert 'High' in df.columns
            assert 'Low' in df.columns
            assert 'Close' in df.columns
            assert 'Volume' in df.columns
    
    def test_clean_data(self, data_processor, sample_data):
        """Test cleaning data with missing values."""
        # Get sample data with missing values
        data = {'ESM23': sample_data['ESM23_missing']}
        
        # Clean data
        cleaned_data = data_processor.clean_data(data)
        
        # Check results
        assert isinstance(cleaned_data, dict)
        assert 'ESM23' in cleaned_data
        assert isinstance(cleaned_data['ESM23'], pd.DataFrame)
        assert not cleaned_data['ESM23'].isna().any().any()  # No NaN values
    
    def test_resample_data(self, data_processor, sample_data):
        """Test resampling data to different frequencies."""
        # Get sample data
        data = {'ESM23': sample_data['ESM23']}
        
        # Original frequency is daily, resample to weekly
        resampled_data = data_processor.resample_data(data, freq='W')
        
        # Check results
        assert isinstance(resampled_data, dict)
        assert 'ESM23' in resampled_data
        assert isinstance(resampled_data['ESM23'], pd.DataFrame)
        
        # Check resampling results
        assert len(resampled_data['ESM23']) < len(data['ESM23'])
        assert all(col in resampled_data['ESM23'].columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_align_data(self, data_processor, sample_data):
        """Test aligning data from different sources to same dates."""
        # Get sample data
        data = {
            'ESM23': sample_data['ESM23'],
            'NQM23': sample_data['NQM23'].iloc[5:]  # Cut off first 5 days
        }
        
        # Align data
        aligned_data = data_processor.align_data(data)
        
        # Check results
        assert isinstance(aligned_data, dict)
        assert 'ESM23' in aligned_data
        assert 'NQM23' in aligned_data
        
        # Check alignment
        assert len(aligned_data['ESM23']) == len(aligned_data['NQM23'])
        assert aligned_data['ESM23'].index.equals(aligned_data['NQM23'].index)
    
    def test_calculate_returns(self, data_processor, sample_data):
        """Test calculating returns from price data."""
        # Get sample data
        data = {'ESM23': sample_data['ESM23']}
        
        # Calculate returns
        returns = data_processor.calculate_returns(data)
        
        # Check results
        assert isinstance(returns, dict)
        assert 'ESM23' in returns
        assert isinstance(returns['ESM23'], pd.DataFrame)
        assert 'Returns' in returns['ESM23'].columns
        
        # Check calculation
        assert len(returns['ESM23']) == len(data['ESM23']) - 1  # One less due to diff
        assert not returns['ESM23'].isnull().values.any()
    
    def test_apply_filter(self, data_processor, sample_data):
        """Test applying filters to data."""
        # Get sample data
        data = {'ESM23': sample_data['ESM23']}
        
        # Define a simple filter function
        def volume_filter(df):
            return df[df['Volume'] > df['Volume'].median()]
        
        # Apply filter
        filtered_data = data_processor.apply_filter(data, volume_filter)
        
        # Check results
        assert isinstance(filtered_data, dict)
        assert 'ESM23' in filtered_data
        assert isinstance(filtered_data['ESM23'], pd.DataFrame)
        
        # Check filtering results
        assert len(filtered_data['ESM23']) < len(data['ESM23'])
        assert all(filtered_data['ESM23']['Volume'] > sample_data['ESM23']['Volume'].median())
    
    def test_merge_data(self, data_processor, sample_data):
        """Test merging data from multiple sources."""
        # Get sample data
        data = {
            'ESM23': sample_data['ESM23'][['Close']],
            'NQM23': sample_data['NQM23'][['Close']]
        }
        
        # Merge data
        merged_data = data_processor.merge_data(data)
        
        # Check results
        assert isinstance(merged_data, pd.DataFrame)
        assert 'ESM23_Close' in merged_data.columns
        assert 'NQM23_Close' in merged_data.columns
        
        # Check merging results
        assert len(merged_data) <= len(sample_data['ESM23'])  # May be less due to alignment
        assert not merged_data.isnull().values.any()
    
    def test_get_data(self, data_processor, mock_ib_connector):
        """Test the main get_data method."""
        # Define test parameters
        symbols = ['ESM23', 'NQM23']
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Get data
        data = data_processor.get_data(symbols, start_date, end_date)
        
        # Check results
        assert isinstance(data, dict)
        assert all(symbol in data for symbol in symbols)
        assert all(isinstance(data[symbol], pd.DataFrame) for symbol in symbols)
        assert all(not data[symbol].empty for symbol in symbols)
        
        # Check data structure and integrity
        for symbol in symbols:
            df = data[symbol]
            assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
            assert not df.isnull().values.any()  # No missing values
            assert df.index.is_monotonic_increasing  # Index is sorted 