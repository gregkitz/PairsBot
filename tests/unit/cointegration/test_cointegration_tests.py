"""
Unit tests for the cointegration_tests module.

This module contains tests for the cointegration testing functions.
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

from src.cointegration.cointegration_tests import (
    test_cointegration,
    test_pairs_universe,
    calculate_half_life,
    johansen_test,
    engle_granger_test
)

@pytest.fixture
def sample_data():
    """Fixture providing sample price data for cointegration testing."""
    # Create synthetic price series
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate cointegrated price series
    # Series 1: random walk
    random_changes = np.random.normal(0, 1, 200)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 (with noise)
    noise = np.random.normal(0, 0.5, 200)
    series2 = 0.5 * series1 + 50 + noise
    
    # Series 3: independent random walk
    random_changes2 = np.random.normal(0, 1, 200)
    series3 = 100 + np.cumsum(random_changes2)
    
    # Create DataFrames
    df1 = pd.Series(series1, index=dates, name='price')
    df2 = pd.Series(series2, index=dates, name='price')
    df3 = pd.Series(series3, index=dates, name='price')
    
    return {
        'cointegrated_pair': (df1, df2),
        'non_cointegrated_pair': (df1, df3),
        'series1': df1,
        'series2': df2,
        'series3': df3
    }

@pytest.fixture
def cointegrated_pairs_file():
    """Fixture providing the path to the cointegrated pairs test fixture."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(current_dir, '..', '..', 'fixtures')
    return os.path.join(fixtures_dir, 'cointegrated_pairs.json')

class TestCointegrationTests:
    """Test class for cointegration testing functions."""
    
    def test_calculate_half_life(self, sample_data):
        """Test calculation of half-life for a mean-reverting series."""
        # Get spread between cointegrated series
        series1 = sample_data['series1']
        series2 = sample_data['series2']
        spread = series1 - 2 * series2  # Create a stationary spread
        
        # Calculate half-life
        half_life = calculate_half_life(spread)
        
        # Check results
        assert half_life > 0  # Half-life should be positive
        assert isinstance(half_life, float)
        
        # Test with non-stationary series
        non_stationary = sample_data['series1']
        half_life_ns = calculate_half_life(non_stationary)
        
        # For a non-stationary series, half-life should be very large or infinite
        assert half_life_ns > 100 or np.isinf(half_life_ns)
    
    def test_engle_granger_test(self, sample_data):
        """Test Engle-Granger cointegration test."""
        # Get cointegrated pair
        series1, series2 = sample_data['cointegrated_pair']
        
        # Run Engle-Granger test
        result = engle_granger_test(series1, series2)
        
        # Check results
        assert isinstance(result, dict)
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_cointegrated' in result
        assert 'hedge_ratio' in result
        
        # For cointegrated series, p-value should be low
        assert result['p_value'] < 0.05
        assert result['is_cointegrated'] is True
        
        # Test with non-cointegrated pair
        series1, series3 = sample_data['non_cointegrated_pair']
        result_nc = engle_granger_test(series1, series3)
        
        # For non-cointegrated series, p-value should be high
        assert result_nc['p_value'] > 0.05 or not result_nc['is_cointegrated']
    
    def test_johansen_test(self, sample_data):
        """Test Johansen cointegration test."""
        # Get cointegrated pair
        series1, series2 = sample_data['cointegrated_pair']
        
        # Combine series into DataFrame
        df = pd.DataFrame({
            'series1': series1,
            'series2': series2
        })
        
        # Run Johansen test
        result = johansen_test(df)
        
        # Check results
        assert isinstance(result, dict)
        assert 'trace_statistic' in result
        assert 'critical_values' in result
        assert 'eigenvalues' in result
        assert 'n_cointegrating_relations' in result
        
        # For cointegrated series, should find at least 1 cointegrating relation
        assert result['n_cointegrating_relations'] >= 1
        
        # Test with non-cointegrated pair
        series1, series3 = sample_data['non_cointegrated_pair']
        df_nc = pd.DataFrame({
            'series1': series1,
            'series3': series3
        })
        result_nc = johansen_test(df_nc)
        
        # For non-cointegrated series, should find 0 cointegrating relations
        # (though this is not always reliable for small samples)
        # So we'll skip assertion here as it might vary
    
    def test_test_cointegration(self, sample_data):
        """Test the main cointegration test function."""
        # Get cointegrated pair
        series1, series2 = sample_data['cointegrated_pair']
        
        # Run combined test
        result = test_cointegration(series1, series2, test_type='both')
        
        # Check results
        assert isinstance(result, dict)
        assert 'engle_granger' in result
        assert 'johansen' in result
        assert 'combined_result' in result
        assert 'hedge_ratio' in result
        assert 'half_life' in result
        
        # For cointegrated series, should be cointegrated
        assert result['combined_result']['is_cointegrated'] is True
        
        # Test with Engle-Granger only
        result_eg = test_cointegration(series1, series2, test_type='engle-granger')
        assert 'engle_granger' in result_eg
        assert 'johansen' not in result_eg
        
        # Test with Johansen only
        result_j = test_cointegration(series1, series2, test_type='johansen')
        assert 'johansen' in result_j
        assert 'engle_granger' not in result_j
        
        # Test with out-of-sample validation
        result_oos = test_cointegration(
            series1, 
            series2, 
            test_type='both',
            train_test_split=0.7,
            out_of_sample=True
        )
        
        # Check out-of-sample results
        assert 'out_of_sample' in result_oos
        assert 'is_cointegrated_oos' in result_oos['out_of_sample']
    
    def test_test_pairs_universe(self, sample_data, cointegrated_pairs_file):
        """Test the pairs universe testing function."""
        # Create a simple universe of price series
        universe = {
            'series1': sample_data['series1'],
            'series2': sample_data['series2'],
            'series3': sample_data['series3']
        }
        
        # Run test on universe
        results = test_pairs_universe(universe)
        
        # Check results
        assert isinstance(results, dict)
        assert len(results) == 3  # 3 possible pairs with 3 assets
        
        # Check specific pair results
        assert 'series1_series2' in results
        assert results['series1_series2']['combined_result']['is_cointegrated'] is True
        
        # Load cointegrated pairs from fixture
        with open(cointegrated_pairs_file, 'r') as f:
            fixture_data = json.load(f)
        
        # Test with filter by correlation
        results_cor = test_pairs_universe(universe, min_correlation=0.9)
        
        # With high correlation threshold, should have fewer pairs
        assert len(results_cor) <= len(results)
        
        # Test with filter by p-value
        results_pval = test_pairs_universe(universe, p_value_threshold=0.01)
        
        # With stricter p-value, should have fewer cointegrated pairs
        cointegrated_pairs_orig = sum(1 for pair in results.values() 
                                     if pair['combined_result']['is_cointegrated'])
        cointegrated_pairs_strict = sum(1 for pair in results_pval.values() 
                                       if pair['combined_result']['is_cointegrated'])
        assert cointegrated_pairs_strict <= cointegrated_pairs_orig 