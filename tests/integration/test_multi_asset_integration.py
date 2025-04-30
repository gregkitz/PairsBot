"""
Integration tests for multi-asset class support.

This module contains tests that verify the system can handle multiple asset classes
and properly integrate them with the core strategy components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.asset_classes.base import Asset, AssetClass
from src.asset_classes.futures.futures_asset import FuturesAsset, FuturesAssetClass
from src.asset_classes.factory import AssetFactory
from src.data_processor.data_processor import DataProcessor
from src.spread_analytics.spread_analyzer import SpreadAnalyzer
from src.signal_generation.signal_generator import SignalGenerator
from src.pairs_trading_strategy import PairsTradingStrategy
from tests.mocks.mock_ib_connector import MockIBConnector

@pytest.fixture
def mock_ib_connector():
    """Fixture providing a mock IB connector."""
    connector = MockIBConnector()
    connector.connect()
    return connector

@pytest.fixture
def asset_factory(mock_ib_connector):
    """Fixture providing an asset factory with futures asset class registered."""
    factory = AssetFactory()
    futures_asset_class = FuturesAssetClass(connector=mock_ib_connector)
    factory.register_asset_class("futures", futures_asset_class)
    return factory

@pytest.fixture
def sample_data():
    """Fixture providing sample price data for multiple asset classes."""
    # Create synthetic data for two futures contracts
    start_date = datetime.now() - timedelta(days=60)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # ES price data with some randomness
    np.random.seed(42)  # For reproducibility
    es_base = 4200
    es_prices = es_base + np.cumsum(np.random.normal(0, 10, len(dates)))
    
    # NQ price data - correlated with ES but with occasional divergence
    nq_base = 14000
    nq_prices = nq_base + np.cumsum(np.random.normal(0, 30, len(dates)))
    # Add correlation with ES
    nq_prices = nq_prices + 3 * (es_prices - es_base)
    
    # Gold price data - partially correlated with ES
    gc_base = 1800
    gc_prices = gc_base + np.cumsum(np.random.normal(0, 5, len(dates)))
    # Add some correlation with ES but less than NQ
    gc_prices = gc_prices + 0.5 * (es_prices - es_base)
    
    # Create DataFrames
    es_df = pd.DataFrame({
        'Open': es_prices * 0.99,
        'High': es_prices * 1.01,
        'Low': es_prices * 0.98,
        'Close': es_prices,
        'Volume': np.random.randint(10000, 50000, len(dates))
    }, index=dates)
    
    nq_df = pd.DataFrame({
        'Open': nq_prices * 0.99,
        'High': nq_prices * 1.01,
        'Low': nq_prices * 0.98,
        'Close': nq_prices,
        'Volume': np.random.randint(5000, 30000, len(dates))
    }, index=dates)
    
    gc_df = pd.DataFrame({
        'Open': gc_prices * 0.99,
        'High': gc_prices * 1.01,
        'Low': gc_prices * 0.98,
        'Close': gc_prices,
        'Volume': np.random.randint(5000, 20000, len(dates))
    }, index=dates)
    
    return {
        'ESM23': es_df,  # E-mini S&P 500
        'NQM23': nq_df,  # E-mini NASDAQ 100
        'GCM23': gc_df   # Gold
    }

@pytest.fixture
def strategy_config():
    """Fixture providing a multi-asset strategy configuration."""
    return {
        'pairs': {
            'ESM23_NQM23': {
                'leg1': 'ESM23',
                'leg2': 'NQM23',
                'hedge_ratio_method': 'kalman',
                'z_score_window': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.0,
                'max_holding_period': 10,
                'asset_class': 'futures'
            },
            'ESM23_GCM23': {
                'leg1': 'ESM23',
                'leg2': 'GCM23',
                'hedge_ratio_method': 'kalman',
                'z_score_window': 20,
                'entry_threshold': 2.5,  # Higher threshold for less correlated pair
                'exit_threshold': 0.5,
                'stop_loss_threshold': 3.5,
                'max_holding_period': 10,
                'asset_class': 'futures'
            }
        },
        'risk_management': {
            'max_position_size': 10,
            'max_pair_exposure': 0.2,
            'max_total_exposure': 0.5
        }
    }

@pytest.mark.integration
class TestMultiAssetIntegration:
    """Integration tests for multi-asset support."""
    
    def test_asset_creation_and_data_retrieval(self, asset_factory, mock_ib_connector):
        """Test creating assets of different classes and retrieving data."""
        # Create futures assets
        es_asset = asset_factory.create_asset("ESM23", asset_class="futures")
        nq_asset = asset_factory.create_asset("NQM23", asset_class="futures")
        gc_asset = asset_factory.create_asset("GCM23", asset_class="futures")
        
        # Verify assets were created with correct type
        assert isinstance(es_asset, FuturesAsset)
        assert isinstance(nq_asset, FuturesAsset)
        assert isinstance(gc_asset, FuturesAsset)
        
        # Check that assets have correct properties
        assert es_asset.symbol == "ESM23"
        assert nq_asset.symbol == "NQM23"
        assert gc_asset.symbol == "GCM23"
        
        # Retrieve data for each asset
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        es_data = es_asset.get_data(start_date, end_date)
        nq_data = nq_asset.get_data(start_date, end_date)
        gc_data = gc_asset.get_data(start_date, end_date)
        
        # Verify data
        assert isinstance(es_data, pd.DataFrame)
        assert isinstance(nq_data, pd.DataFrame)
        assert isinstance(gc_data, pd.DataFrame)
        
        assert not es_data.empty
        assert not nq_data.empty
        assert not gc_data.empty
        
        # Check data structure
        for df in [es_data, nq_data, gc_data]:
            assert 'Open' in df.columns
            assert 'High' in df.columns
            assert 'Low' in df.columns
            assert 'Close' in df.columns
            assert 'Volume' in df.columns
    
    def test_multi_pair_strategy(self, sample_data, strategy_config, mock_ib_connector):
        """Test running the strategy with multiple pairs across asset classes."""
        # Create components
        data_processor = DataProcessor()
        spread_analyzer = SpreadAnalyzer()
        signal_generator = SignalGenerator()
        strategy = PairsTradingStrategy(strategy_config)
        
        # Inject sample data
        data_processor._data = sample_data
        
        # Process all pairs
        results = {}
        for pair_id, pair_config in strategy_config['pairs'].items():
            # Analyze spread
            spread_analysis = spread_analyzer.analyze_pair(
                pair_id=pair_id,
                data=sample_data,
                leg1=pair_config['leg1'],
                leg2=pair_config['leg2'],
                hedge_ratio_method=pair_config['hedge_ratio_method'],
                z_score_window=pair_config['z_score_window']
            )
            
            # Generate signals
            signals = signal_generator.generate_signals(
                spread_analysis=spread_analysis,
                entry_threshold=pair_config['entry_threshold'],
                exit_threshold=pair_config['exit_threshold'],
                stop_loss_threshold=pair_config['stop_loss_threshold']
            )
            
            # Update strategy
            strategy_result = strategy.update(pair_id, spread_analysis, signals)
            
            results[pair_id] = {
                'spread_analysis': spread_analysis,
                'signals': signals,
                'strategy_result': strategy_result
            }
        
        # Verify results for each pair
        for pair_id, result in results.items():
            # Check spread analysis
            assert 'spread_series' in result['spread_analysis']
            assert 'z_score_series' in result['spread_analysis']
            assert 'hedge_ratio' in result['spread_analysis']
            
            # Check signals
            assert 'signal' in result['signals']
            
            # Check strategy result
            assert 'position' in result['strategy_result']
            assert 'trades' in result['strategy_result']
            assert 'pnl' in result['strategy_result']
        
        # Verify that strategy maintains pair-specific state
        assert 'ESM23_NQM23' in strategy.pairs
        assert 'ESM23_GCM23' in strategy.pairs
        
        # Verify that position sizing accounts for different asset classes
        if any(results[pair_id]['signals']['signal'] != 0 for pair_id in results):
            assert 'max_position_size' in strategy.risk_params
            assert strategy.risk_params['max_position_size'] > 0
    
    def test_cross_asset_correlation(self, sample_data, asset_factory):
        """Test correlation analysis between different asset classes."""
        # Create assets
        es_asset = asset_factory.create_asset("ESM23", asset_class="futures")
        nq_asset = asset_factory.create_asset("NQM23", asset_class="futures")
        gc_asset = asset_factory.create_asset("GCM23", asset_class="futures")
        
        # Get close prices
        es_prices = sample_data['ESM23']['Close']
        nq_prices = sample_data['NQM23']['Close']
        gc_prices = sample_data['GCM23']['Close']
        
        # Calculate correlations
        es_nq_corr = es_prices.corr(nq_prices)
        es_gc_corr = es_prices.corr(gc_prices)
        nq_gc_corr = nq_prices.corr(gc_prices)
        
        # ES and NQ should be highly correlated
        assert es_nq_corr > 0.7
        
        # ES and GC should have lower correlation
        assert es_gc_corr < es_nq_corr
        
        # Check that correlations are within reasonable bounds
        assert -1 <= es_nq_corr <= 1
        assert -1 <= es_gc_corr <= 1
        assert -1 <= nq_gc_corr <= 1
    
    def test_data_alignment_across_assets(self, sample_data):
        """Test that data from different assets is properly aligned for analysis."""
        # Get data for different assets
        es_data = sample_data['ESM23']
        nq_data = sample_data['NQM23']
        gc_data = sample_data['GCM23']
        
        # Check that indices match
        assert es_data.index.equals(nq_data.index)
        assert es_data.index.equals(gc_data.index)
        
        # Create a spread analyzer
        spread_analyzer = SpreadAnalyzer()
        
        # Calculate spread for ES-NQ pair
        es_nq_analysis = spread_analyzer.analyze_pair(
            pair_id='ESM23_NQM23',
            data=sample_data,
            leg1='ESM23',
            leg2='NQM23',
            hedge_ratio_method='ols',
            z_score_window=20
        )
        
        # Calculate spread for ES-GC pair
        es_gc_analysis = spread_analyzer.analyze_pair(
            pair_id='ESM23_GCM23',
            data=sample_data,
            leg1='ESM23',
            leg2='GCM23',
            hedge_ratio_method='ols',
            z_score_window=20
        )
        
        # Check spread calculation
        assert len(es_nq_analysis['spread_series']) == len(es_data)
        assert len(es_gc_analysis['spread_series']) == len(es_data)
        
        # Verify z-score calculation (should have same length but NaNs at beginning)
        assert len(es_nq_analysis['z_score_series']) == len(es_data)
        assert len(es_gc_analysis['z_score_series']) == len(es_data)
        
        # Check that hedge ratios make sense relative to price levels
        es_avg_price = es_data['Close'].mean()
        nq_avg_price = nq_data['Close'].mean()
        gc_avg_price = gc_data['Close'].mean()
        
        assert es_nq_analysis['hedge_ratio'] * es_avg_price < nq_avg_price * 2
        assert es_gc_analysis['hedge_ratio'] * es_avg_price < gc_avg_price * 2 