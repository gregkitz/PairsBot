"""
Integration tests for the data flow between components.

This module contains tests for the data flow between data processor,
pair finder, spread analyzer, and signal generator components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from src.data_processor.data_processor import DataProcessor
from src.cointegration.pair_finder import PairFinder
from src.spread_analytics.spread_analyzer import SpreadAnalyzer
from src.signal_generation.signal_generator import SignalGenerator
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
    # Create synthetic data for four assets
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducibility
    
    # Base price series with different starting points
    price1 = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    price2 = 500 + np.cumsum(np.random.normal(0, 3, len(dates)))
    
    # Create cointegrated pair by making price3 related to price1
    price3 = 0.5 * price1 + 50 + np.random.normal(0, 0.5, len(dates))
    
    # Independent price series
    price4 = 75 + np.cumsum(np.random.normal(0, 0.8, len(dates)))
    
    # Create DataFrames
    es_df = pd.DataFrame({
        'Open': price1 * 0.99,
        'High': price1 * 1.01,
        'Low': price1 * 0.98,
        'Close': price1,
        'Volume': np.random.randint(10000, 50000, len(dates))
    }, index=dates)
    
    nq_df = pd.DataFrame({
        'Open': price2 * 0.99,
        'High': price2 * 1.01,
        'Low': price2 * 0.98,
        'Close': price2,
        'Volume': np.random.randint(5000, 30000, len(dates))
    }, index=dates)
    
    gc_df = pd.DataFrame({
        'Open': price3 * 0.99,
        'High': price3 * 1.01,
        'Low': price3 * 0.98,
        'Close': price3,
        'Volume': np.random.randint(3000, 15000, len(dates))
    }, index=dates)
    
    zn_df = pd.DataFrame({
        'Open': price4 * 0.99,
        'High': price4 * 1.01,
        'Low': price4 * 0.98,
        'Close': price4,
        'Volume': np.random.randint(2000, 10000, len(dates))
    }, index=dates)
    
    return {
        'ESM23': es_df,  # E-mini S&P 500
        'NQM23': nq_df,  # E-mini NASDAQ 100
        'GCM23': gc_df,  # Gold
        'ZNM23': zn_df   # 10Y Treasury Notes
    }

@pytest.fixture
def test_config_file():
    """Fixture providing the path to the test configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(current_dir, '..', 'fixtures')
    return os.path.join(fixtures_dir, 'test_config.json')

@pytest.fixture
def test_config(test_config_file):
    """Fixture providing the test configuration."""
    with open(test_config_file, 'r') as f:
        config = json.load(f)
    return config

@pytest.fixture
def data_processor(mock_ib_connector):
    """Fixture providing a DataProcessor instance."""
    processor = DataProcessor()
    processor.connector = mock_ib_connector
    return processor

@pytest.fixture
def pair_finder(test_config):
    """Fixture providing a PairFinder instance."""
    finder = PairFinder(test_config.get('pair_finder', {}))
    return finder

@pytest.fixture
def spread_analyzer(test_config):
    """Fixture providing a SpreadAnalyzer instance."""
    analyzer = SpreadAnalyzer(test_config.get('spread_analyzer', {}))
    return analyzer

@pytest.fixture
def signal_generator(test_config):
    """Fixture providing a SignalGenerator instance."""
    generator = SignalGenerator(test_config.get('signal_generator', {}))
    return generator

@pytest.mark.integration
class TestDataFlow:
    """Integration tests for the data flow between components."""
    
    def test_end_to_end_data_flow(self, sample_data, data_processor, pair_finder, 
                                 spread_analyzer, signal_generator):
        """Test the full data flow from data processing to signal generation."""
        # Inject sample data into the data processor
        data_processor._data = sample_data
        
        # Find cointegrated pairs
        universe = sample_data
        pairs = pair_finder.find_pairs(universe)
        
        # Check that pairs were found
        assert isinstance(pairs, dict)
        assert len(pairs) > 0
        
        # For each pair, run the full data flow
        results = {}
        
        for pair_id, pair_info in pairs.items():
            # Extract legs
            leg1, leg2 = pair_id.split('_')
            
            # Generate spread analysis
            spread_analysis = spread_analyzer.analyze_pair(
                pair_id=pair_id,
                data=universe,
                leg1=leg1,
                leg2=leg2,
                hedge_ratio_method='ols'
            )
            
            # Generate signals
            signals = signal_generator.generate_signals(
                spread_analysis=spread_analysis,
                entry_threshold=2.0,
                exit_threshold=0.5,
                stop_loss_threshold=3.0
            )
            
            # Store results
            results[pair_id] = {
                'pair_info': pair_info,
                'spread_analysis': spread_analysis,
                'signals': signals
            }
        
        # Verify end-to-end data flow
        for pair_id, result in results.items():
            # Check that each stage produced the expected output format
            assert 'is_cointegrated' in result['pair_info']
            assert 'spread_series' in result['spread_analysis']
            assert 'z_score_series' in result['spread_analysis']
            assert 'signal' in result['signals']
            
            # Check that data lengths match across the flow
            leg1, leg2 = pair_id.split('_')
            original_length = len(sample_data[leg1])
            assert len(result['spread_analysis']['spread_series']) == original_length
            assert len(result['spread_analysis']['z_score_series']) == original_length
            assert len(result['signals']['signal']) == original_length
    
    def test_data_alignment_preserved(self, sample_data, spread_analyzer, signal_generator):
        """Test that data alignment is preserved throughout the flow."""
        # Pick a specific pair
        pair_id = 'ESM23_GCM23'  # S&P 500 and Gold (should be cointegrated in our synthetic data)
        leg1 = 'ESM23'
        leg2 = 'GCM23'
        
        # Get original data
        df1 = sample_data[leg1]
        df2 = sample_data[leg2]
        
        # Cut off 10 days from the beginning of df2 to simulate misaligned data
        df2_cut = df2.iloc[10:].copy()
        data = {leg1: df1, leg2: df2_cut}
        
        # Run spread analysis
        spread_analysis = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=data,
            leg1=leg1,
            leg2=leg2
        )
        
        # Generate signals
        signals = signal_generator.generate_signals(
            spread_analysis=spread_analysis
        )
        
        # Check that data is aligned
        assert len(spread_analysis['spread_series']) == min(len(df1), len(df2_cut))
        assert spread_analysis['spread_series'].index.equals(spread_analysis['z_score_series'].index)
        assert spread_analysis['z_score_series'].index.equals(signals['signal'].index)
        
        # Check alignment quality
        date_diff = abs((df1.index.min() - df2_cut.index.min()).days)
        assert date_diff >= 10  # Original misalignment
    
    def test_multiple_hedge_ratio_methods(self, sample_data, spread_analyzer):
        """Test that different hedge ratio methods produce consistent output formats."""
        # Pick a specific pair
        pair_id = 'ESM23_GCM23'
        leg1 = 'ESM23'
        leg2 = 'GCM23'
        
        # Run analysis with OLS
        spread_ols = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=sample_data,
            leg1=leg1,
            leg2=leg2,
            hedge_ratio_method='ols'
        )
        
        # Run analysis with Kalman filter
        spread_kalman = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=sample_data,
            leg1=leg1,
            leg2=leg2,
            hedge_ratio_method='kalman'
        )
        
        # Check both methods produce required outputs
        assert 'spread_series' in spread_ols
        assert 'z_score_series' in spread_ols
        assert 'hedge_ratio' in spread_ols
        
        assert 'spread_series' in spread_kalman
        assert 'z_score_series' in spread_kalman
        assert 'hedge_ratio' in spread_kalman
        assert 'hedge_ratio_series' in spread_kalman  # Kalman should have this
        
        # Spreads should be different but have same length
        assert len(spread_ols['spread_series']) == len(spread_kalman['spread_series'])
        assert not spread_ols['spread_series'].equals(spread_kalman['spread_series'])
    
    def test_parameter_propagation(self, sample_data, spread_analyzer, signal_generator):
        """Test that parameter changes propagate through the flow correctly."""
        # Pick a specific pair
        pair_id = 'ESM23_GCM23'
        leg1 = 'ESM23'
        leg2 = 'GCM23'
        
        # Run with default parameters
        spread_default = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=sample_data,
            leg1=leg1,
            leg2=leg2,
            z_score_window=20  # Default
        )
        
        # Run with different window
        spread_large_window = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=sample_data,
            leg1=leg1,
            leg2=leg2,
            z_score_window=50  # Larger window
        )
        
        # Z-scores should be different due to different windows
        assert not spread_default['z_score_series'].equals(spread_large_window['z_score_series'])
        
        # Generate signals with different thresholds
        signals_default = signal_generator.generate_signals(
            spread_analysis=spread_default,
            entry_threshold=2.0,  # Default
            exit_threshold=0.5    # Default
        )
        
        signals_tight = signal_generator.generate_signals(
            spread_analysis=spread_default,
            entry_threshold=1.5,  # Lower threshold = more entries
            exit_threshold=0.3    # Lower threshold = quicker exits
        )
        
        # Signals should be different due to different thresholds
        assert not signals_default['signal'].equals(signals_tight['signal'])
        
        # Lower entry threshold should generate more trades
        default_trades = signals_default['signal'].diff().abs().sum()
        tight_trades = signals_tight['signal'].diff().abs().sum()
        assert tight_trades >= default_trades
    
    def test_partial_data_handling(self, sample_data, spread_analyzer, signal_generator):
        """Test handling of partial or missing data."""
        # Pick a specific pair
        pair_id = 'ESM23_GCM23'
        leg1 = 'ESM23'
        leg2 = 'GCM23'
        
        # Create a copy of the data and introduce some NaN values
        data_with_nans = sample_data.copy()
        data_with_nans[leg1].iloc[10:20, 0:2] = np.nan  # Set some Open and High to NaN
        
        # Run analysis with missing data
        spread_analysis = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=data_with_nans,
            leg1=leg1,
            leg2=leg2
        )
        
        # Generate signals
        signals = signal_generator.generate_signals(
            spread_analysis=spread_analysis
        )
        
        # Check that spread and z-score don't have NaNs (should be handled)
        assert not spread_analysis['spread_series'].isna().any()
        assert not spread_analysis['z_score_series'].isna().any()
        
        # Signals should not have NaNs
        assert not signals['signal'].isna().any() 