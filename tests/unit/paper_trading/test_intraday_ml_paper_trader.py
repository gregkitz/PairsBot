"""
Unit tests for the IntradayMLPaperTrader class.

This module contains tests to verify the functionality of the IntradayMLPaperTrader class,
providing a foundation for ensuring behavior correctness during refactoring.
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.paper_trading.intraday_ml_paper_trader import IntradayMLPaperTrader
from src.ml_enhancements.intraday_integration import IntradayMLSystem
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from tests.mocks.mock_ib_connector import MockIBConnector

# Constants for testing
TEST_PAIR = 'GC_SI'
TEST_TIMEFRAME = '5min'


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for paper trading data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_ml_config():
    """Fixture providing a mock ML configuration."""
    return {
        "feature_lookback": 20,
        "prediction_threshold": 0.6,
        "use_rsi_filter": True,
        "use_volume_filter": True,
        "use_volatility_filter": True,
        "enable_ml_filtering": True,
        "enable_ml_timing": True,
        "enable_ml_adaptation": True,
        "pairs": [TEST_PAIR],
        "timeframe": TEST_TIMEFRAME
    }


@pytest.fixture
def mock_ml_system():
    """Fixture providing a mock ML system."""
    ml_system = Mock(spec=IntradayMLSystem)
    ml_system.process_data.return_value = pd.DataFrame({
        'zscore': np.random.randn(100),
        'signal': np.random.choice([-1, 0, 1], size=100),
        'quality': np.random.random(100)
    })
    ml_system.get_current_regime.return_value = "normal"
    ml_system.enhance_signals.return_value = (
        pd.Series(np.random.choice([-1, 0, 1], size=100)),
        pd.DataFrame({'signal_quality': np.random.random(100)})
    )
    return ml_system


@pytest.fixture
def mock_regime_classifier():
    """Fixture providing a mock regime classifier."""
    classifier = Mock(spec=MarketRegimeClassifier)
    classifier.predict_regime.return_value = "normal"
    classifier.get_regime_parameters.return_value = {
        "z_entry": 2.0,
        "z_exit": 0.5,
        "max_holding_period": 120
    }
    return classifier


@pytest.fixture
def intraday_ml_paper_trader(temp_dir, mock_ml_config, monkeypatch):
    """Fixture providing an IntradayMLPaperTrader instance with mocked dependencies."""
    # Mock the IBConnector and ML system
    monkeypatch.setattr("src.paper_trading.paper_trader.IBConnector", MockIBConnector)
    
    # Create trader with test mode enabled to avoid actual IB connection
    trader = IntradayMLPaperTrader(
        initial_capital=100000.0,
        ib_host='127.0.0.1',
        ib_port=7497,
        ib_client_id=1,
        ml_config=mock_ml_config,
        output_dir=os.path.join(temp_dir, "output"),
        models_dir=os.path.join(temp_dir, "models"),
        test_mode=True
    )
    
    # Replace the ML system with our mock
    trader.ml_system = Mock(spec=IntradayMLSystem)
    trader.regime_classifier = Mock(spec=MarketRegimeClassifier)
    trader.regime_classifier.predict_regime.return_value = "normal"
    trader.regime_classifier.get_regime_parameters.return_value = {
        "z_entry": 2.0,
        "z_exit": 0.5,
        "max_holding_period": 120
    }
    
    return trader


class TestIntradayMLPaperTrader:
    """Test class for IntradayMLPaperTrader."""
    
    def test_initialization(self, intraday_ml_paper_trader, temp_dir, mock_ml_config):
        """Test that the intraday ML paper trader is initialized correctly."""
        # Check basic attributes
        assert intraday_ml_paper_trader.ml_config == mock_ml_config
        assert intraday_ml_paper_trader.output_dir == os.path.join(temp_dir, "output")
        assert intraday_ml_paper_trader.models_dir == os.path.abspath(os.path.join(temp_dir, "models"))
        assert intraday_ml_paper_trader.test_mode is True
        
        # Check that output directories were created
        assert os.path.exists(os.path.join(temp_dir, "output", "logs"))
        assert os.path.exists(os.path.join(temp_dir, "output", "dashboard"))
        assert os.path.exists(os.path.join(temp_dir, "output", "signals"))
    
    def test_start_stop(self, intraday_ml_paper_trader):
        """Test starting and stopping the intraday ML paper trader."""
        # Start trader
        assert not intraday_ml_paper_trader.is_running
        intraday_ml_paper_trader.start()
        assert intraday_ml_paper_trader.is_running
        
        # Stop trader
        intraday_ml_paper_trader.stop()
        assert not intraday_ml_paper_trader.is_running
    
    def test_add_pair(self, intraday_ml_paper_trader):
        """Test adding a trading pair."""
        # Start trader
        intraday_ml_paper_trader.start()
        
        # Add a pair
        pair_config = {
            "pair_id": TEST_PAIR,
            "leg1": "GC",
            "leg2": "SI",
            "timeframe": TEST_TIMEFRAME,
            "z_entry": 2.0,
            "z_exit": 0.5
        }
        result = intraday_ml_paper_trader.add_pair(pair_config)
        assert result is True
        assert TEST_PAIR in intraday_ml_paper_trader.pairs
        assert "GC" in intraday_ml_paper_trader.symbols
        assert "SI" in intraday_ml_paper_trader.symbols
    
    @patch("src.paper_trading.intraday_ml_paper_trader.pd.read_csv")
    def test_process_market_data(self, mock_read_csv, intraday_ml_paper_trader):
        """Test processing market data."""
        # Setup mock data
        mock_prices = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min'),
            'GC': np.random.rand(100) * 1000 + 1800,
            'SI': np.random.rand(100) * 10 + 20
        }).set_index('timestamp')
        mock_read_csv.return_value = mock_prices
        
        # Setup mock for ML system process_data method
        mock_processed_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min'),
            'zscore': np.random.randn(100),
            'spread': np.random.randn(100),
            'hedge_ratio': [0.05] * 100
        }).set_index('timestamp')
        intraday_ml_paper_trader.ml_system.process_data.return_value = mock_processed_data
        
        # Start trader and add a pair
        intraday_ml_paper_trader.start()
        pair_config = {
            "pair_id": TEST_PAIR,
            "leg1": "GC",
            "leg2": "SI",
            "timeframe": TEST_TIMEFRAME,
            "z_entry": 2.0,
            "z_exit": 0.5
        }
        intraday_ml_paper_trader.add_pair(pair_config)
        
        # Process market data
        intraday_ml_paper_trader._process_market_data()
        
        # Check that ML system was called
        intraday_ml_paper_trader.ml_system.process_data.assert_called_once()
    
    @patch("src.paper_trading.intraday_ml_paper_trader.pd.read_csv")
    def test_generate_trading_signals(self, mock_read_csv, intraday_ml_paper_trader):
        """Test generating trading signals."""
        # Setup mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        mock_prices = pd.DataFrame({
            'timestamp': dates,
            'GC': np.random.rand(100) * 1000 + 1800,
            'SI': np.random.rand(100) * 10 + 20
        }).set_index('timestamp')
        mock_read_csv.return_value = mock_prices
        
        # Setup mock processed data
        mock_processed_data = pd.DataFrame({
            'timestamp': dates,
            'zscore': np.random.randn(100),
            'spread': np.random.randn(100),
            'hedge_ratio': [0.05] * 100
        }).set_index('timestamp')
        intraday_ml_paper_trader.ml_system.process_data.return_value = mock_processed_data
        
        # Setup mock enhanced signals
        mock_signals = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates)
        mock_metrics = pd.DataFrame({'signal_quality': np.random.random(100)}, index=dates)
        intraday_ml_paper_trader.ml_system.enhance_signals.return_value = (mock_signals, mock_metrics)
        
        # Start trader and add a pair
        intraday_ml_paper_trader.start()
        pair_config = {
            "pair_id": TEST_PAIR,
            "leg1": "GC",
            "leg2": "SI",
            "timeframe": TEST_TIMEFRAME,
            "z_entry": 2.0,
            "z_exit": 0.5
        }
        intraday_ml_paper_trader.add_pair(pair_config)
        
        # Generate trading signals
        signals = intraday_ml_paper_trader._generate_trading_signals(TEST_PAIR, mock_processed_data)
        
        # Check that ML system's enhance_signals was called
        intraday_ml_paper_trader.ml_system.enhance_signals.assert_called_once()
        
        # Check that we got signals back
        assert isinstance(signals, pd.Series)
        assert len(signals) > 0
    
    def test_detect_regime(self, intraday_ml_paper_trader):
        """Test market regime detection."""
        # Setup mock prices
        prices = pd.DataFrame({
            'GC': np.random.rand(100) * 1000 + 1800,
            'SI': np.random.rand(100) * 10 + 20
        })
        
        # Setup mock regime detection
        intraday_ml_paper_trader.regime_classifier.predict_regime.return_value = "volatile"
        
        # Detect regime
        regime = intraday_ml_paper_trader._detect_regime(prices)
        
        # Check result
        assert regime == "volatile"
        intraday_ml_paper_trader.regime_classifier.predict_regime.assert_called_once()
    
    def test_update_dashboard(self, intraday_ml_paper_trader):
        """Test dashboard update."""
        # Start trader
        intraday_ml_paper_trader.start()
        
        # Setup test data
        pair_data = {
            TEST_PAIR: {
                "prices": pd.DataFrame({
                    'GC': np.random.rand(100) * 1000 + 1800,
                    'SI': np.random.rand(100) * 10 + 20
                }),
                "signals": pd.DataFrame({
                    'zscore': np.random.randn(100),
                    'signal': np.random.choice([-1, 0, 1], size=100),
                    'hedge_ratio': [0.05] * 100
                }),
                "performance": {
                    "pnl": 1000.0,
                    "win_rate": 0.6,
                    "sharpe_ratio": 1.5
                }
            }
        }
        intraday_ml_paper_trader.pair_data = pair_data
        
        # Update dashboard
        with patch("src.paper_trading.intraday_ml_paper_trader.plt") as mock_plt:
            intraday_ml_paper_trader._update_dashboard()
            
            # Check that plots were created
            assert mock_plt.figure.call_count > 0
            assert mock_plt.savefig.call_count > 0
    
    def test_performance_metrics(self, intraday_ml_paper_trader):
        """Test performance metrics calculation."""
        # Setup test data
        trades = [
            {"pair_id": TEST_PAIR, "entry_time": datetime(2023, 1, 1, 10, 0), 
             "exit_time": datetime(2023, 1, 1, 11, 0), "pnl": 100.0, "exit_reason": "target"},
            {"pair_id": TEST_PAIR, "entry_time": datetime(2023, 1, 1, 12, 0), 
             "exit_time": datetime(2023, 1, 1, 13, 0), "pnl": -50.0, "exit_reason": "stop"},
            {"pair_id": TEST_PAIR, "entry_time": datetime(2023, 1, 1, 14, 0), 
             "exit_time": datetime(2023, 1, 1, 15, 0), "pnl": 200.0, "exit_reason": "target"}
        ]
        
        # Calculate metrics
        metrics = intraday_ml_paper_trader._create_performance_metrics(trades)
        
        # Check metrics
        assert metrics["total_pnl"] == 250.0
        assert metrics["win_rate"] == 2/3
        assert metrics["avg_winning_trade"] == 150.0
        assert metrics["avg_losing_trade"] == -50.0
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
    
    def test_execute_signals(self, intraday_ml_paper_trader):
        """Test signal execution."""
        # Start trader
        intraday_ml_paper_trader.start()
        
        # Setup test data
        pair_config = {
            "pair_id": TEST_PAIR,
            "leg1": "GC",
            "leg2": "SI",
            "timeframe": TEST_TIMEFRAME,
            "z_entry": 2.0,
            "z_exit": 0.5,
            "leg1_multiplier": 1.0,
            "leg2_multiplier": 1.0
        }
        intraday_ml_paper_trader.add_pair(pair_config)
        
        # Create test signals
        dates = pd.date_range(start='2023-01-01', periods=3, freq='5min')
        signals = pd.Series([1, 0, -1], index=dates)  # Buy, hold, sell
        
        # Mock the paper trader's place_order method to return success
        intraday_ml_paper_trader.paper_trader.place_order = Mock(return_value="test_order_id")
        intraday_ml_paper_trader.paper_trader.get_position = Mock(return_value=None)  # No current position
        
        # Execute signals
        intraday_ml_paper_trader._execute_signals(TEST_PAIR, signals)
        
        # Check that orders were placed
        assert intraday_ml_paper_trader.paper_trader.place_order.call_count > 0
    
    def test_save_signals(self, intraday_ml_paper_trader, temp_dir):
        """Test saving signals to file."""
        # Start trader
        intraday_ml_paper_trader.start()
        
        # Create test signals
        dates = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        signals = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates)
        
        # Save signals
        filename = intraday_ml_paper_trader._save_signals(TEST_PAIR, signals)
        
        # Check that file was created
        assert os.path.exists(filename)
        
        # Check that signals match
        saved_signals = pd.read_csv(filename, index_col=0, parse_dates=True)
        assert len(saved_signals) == len(signals)


# Additional test fixtures and classes for component testing during refactoring

@pytest.fixture
def mock_signal_enhancer():
    """Mock signal enhancer component for refactoring tests."""
    enhancer = Mock()
    enhancer.enhance_signals.return_value = pd.Series(np.random.choice([-1, 0, 1], size=100))
    return enhancer

@pytest.fixture
def mock_position_manager():
    """Mock position manager component for refactoring tests."""
    manager = Mock()
    manager.update_positions.return_value = True
    manager.get_current_positions.return_value = {}
    return manager

@pytest.fixture
def mock_performance_tracker():
    """Mock performance tracker component for refactoring tests."""
    tracker = Mock()
    tracker.update_metrics.return_value = {
        "total_pnl": 1000.0,
        "win_rate": 0.6,
        "sharpe_ratio": 1.5
    }
    return tracker

@pytest.fixture
def mock_dashboard_generator():
    """Mock dashboard generator component for refactoring tests."""
    generator = Mock()
    generator.generate_dashboard.return_value = "dashboard.html"
    return generator

@pytest.fixture
def mock_alert_system():
    """Mock alert system component for refactoring tests."""
    alert_system = Mock()
    alert_system.check_conditions.return_value = []
    alert_system.send_alert.return_value = True
    return alert_system


class TestComponentIntegration:
    """
    Tests for validating the refactored component-based architecture.
    These tests will be used during the refactoring process to ensure
    that the new component-based implementation behaves the same as
    the original monolithic implementation.
    """
    
    def test_signal_enhancer_integration(self, intraday_ml_paper_trader, mock_signal_enhancer):
        """Test integration with the signal enhancer component."""
        # This test will be implemented during refactoring
        pass
    
    def test_position_manager_integration(self, intraday_ml_paper_trader, mock_position_manager):
        """Test integration with the position manager component."""
        # This test will be implemented during refactoring
        pass
    
    def test_performance_tracker_integration(self, intraday_ml_paper_trader, mock_performance_tracker):
        """Test integration with the performance tracker component."""
        # This test will be implemented during refactoring
        pass
    
    def test_dashboard_generator_integration(self, intraday_ml_paper_trader, mock_dashboard_generator):
        """Test integration with the dashboard generator component."""
        # This test will be implemented during refactoring
        pass
    
    def test_alert_system_integration(self, intraday_ml_paper_trader, mock_alert_system):
        """Test integration with the alert system component."""
        # This test will be implemented during refactoring
        pass
    
    def test_end_to_end_behavior_comparison(self, intraday_ml_paper_trader):
        """
        Test that the refactored implementation produces the same results as the original.
        This is a critical test for validating the refactoring.
        """
        # This test will be implemented during refactoring
        pass 