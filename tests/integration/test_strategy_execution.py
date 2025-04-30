"""
Integration tests for the strategy execution flow.

This module contains tests for the integration between data processing,
spread analysis, signal generation, and strategy components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

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
def sample_data():
    """Fixture providing sample price data for testing."""
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
    # Base correlation
    nq_prices = nq_base + np.cumsum(np.random.normal(0, 30, len(dates)))
    # Add correlation with ES
    nq_prices = nq_prices + 3 * (es_prices - es_base)
    
    # Create divergence in the middle (to test signal generation)
    divergence_start = len(dates) // 2
    divergence_end = divergence_start + 10
    nq_prices[divergence_start:divergence_end] += np.linspace(0, 200, divergence_end - divergence_start)
    nq_prices[divergence_end:divergence_end+10] -= np.linspace(200, 0, 10)
    
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
    
    return {'ESM23': es_df, 'NQM23': nq_df}

@pytest.fixture
def strategy_config():
    """Fixture providing a sample strategy configuration."""
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
            }
        },
        'risk_management': {
            'max_position_size': 10,
            'max_pair_exposure': 0.2,
            'max_total_exposure': 0.5
        }
    }

@pytest.mark.integration
class TestStrategyExecution:
    """Integration tests for the strategy execution flow."""
    
    def test_end_to_end_execution(self, sample_data, strategy_config, mock_ib_connector):
        """Test the full strategy execution flow from data to signals."""
        # Create components
        data_processor = DataProcessor()
        spread_analyzer = SpreadAnalyzer()
        signal_generator = SignalGenerator()
        strategy = PairsTradingStrategy(strategy_config)
        
        # Inject sample data
        data_processor._data = sample_data  # Not ideal, but works for testing
        
        # Execute strategy pipeline
        pair_id = 'ESM23_NQM23'
        pair_config = strategy_config['pairs'][pair_id]
        
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
        
        # Execute strategy update
        strategy_result = strategy.update(pair_id, spread_analysis, signals)
        
        # Assertions
        assert 'spread_series' in spread_analysis
        assert 'z_score_series' in spread_analysis
        assert 'hedge_ratio' in spread_analysis
        assert len(spread_analysis['spread_series']) == len(sample_data['ESM23'])
        
        # Check that z-score is calculated correctly
        assert not spread_analysis['z_score_series'].isna().all()
        assert spread_analysis['z_score_series'].mean() < 0.1  # Should be close to zero
        
        # Check that signals are generated
        assert 'signal' in signals
        # Due to the divergence we created in the sample data, we should have some non-zero signals
        assert (signals['signal'] != 0).any()
        
        # Check strategy output
        assert 'position' in strategy_result
        assert 'trades' in strategy_result
        assert 'pnl' in strategy_result
    
    def test_data_flow_integrity(self, sample_data, strategy_config):
        """Test that data flows correctly through the system, preserving time alignment."""
        # Create components
        data_processor = DataProcessor()
        spread_analyzer = SpreadAnalyzer()
        
        # Inject sample data
        data_processor._data = sample_data
        
        # Get pair configuration
        pair_id = 'ESM23_NQM23'
        pair_config = strategy_config['pairs'][pair_id]
        
        # Analyze spread
        spread_analysis = spread_analyzer.analyze_pair(
            pair_id=pair_id,
            data=sample_data,
            leg1=pair_config['leg1'],
            leg2=pair_config['leg2'],
            hedge_ratio_method=pair_config['hedge_ratio_method'],
            z_score_window=pair_config['z_score_window']
        )
        
        # Check that indices are preserved
        leg1_data = sample_data[pair_config['leg1']]
        leg2_data = sample_data[pair_config['leg2']]
        spread_series = spread_analysis['spread_series']
        z_score_series = spread_analysis['z_score_series']
        
        assert leg1_data.index.equals(leg2_data.index)
        assert leg1_data.index.equals(spread_series.index)
        # Z-score should have same length but possibly NaNs at beginning due to rolling window
        assert len(z_score_series) == len(leg1_data)
        
        # Check that data types are preserved
        assert isinstance(spread_series, pd.Series)
        assert isinstance(z_score_series, pd.Series)
        
        # Check that hedge ratio makes sense (should be related to price ratio)
        hedge_ratio = spread_analysis['hedge_ratio']
        avg_leg1_price = leg1_data['Close'].mean()
        avg_leg2_price = leg2_data['Close'].mean()
        price_ratio = avg_leg2_price / avg_leg1_price
        
        # Hedge ratio should be in the ballpark of price ratio
        # (may differ due to Kalman filter or other methods)
        assert 0.1 * price_ratio < hedge_ratio < 10 * price_ratio 