"""
Unit tests for the RiskManager class.

This module contains tests for the risk management functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

from src.risk_management.risk_manager import RiskManager

@pytest.fixture
def sample_portfolio_data():
    """Fixture providing sample portfolio data for risk management."""
    # Create synthetic portfolio data
    np.random.seed(42)  # For reproducibility
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate positions for two pairs
    positions = {
        'ESM23_NQM23': pd.DataFrame({
            'quantity1': np.random.randint(-5, 6, len(dates)),  # Leg 1 quantity
            'quantity2': np.random.randint(-5, 6, len(dates)),  # Leg 2 quantity
            'notional1': np.random.uniform(5000, 50000, len(dates)),  # Leg 1 notional
            'notional2': np.random.uniform(5000, 50000, len(dates)),  # Leg 2 notional
            'pnl': np.random.normal(0, 1000, len(dates)),  # P&L
            'unrealized_pnl': np.random.normal(0, 500, len(dates))  # Unrealized P&L
        }, index=dates),
        
        'GCM23_SIM23': pd.DataFrame({
            'quantity1': np.random.randint(-3, 4, len(dates)),  # Leg 1 quantity
            'quantity2': np.random.randint(-3, 4, len(dates)),  # Leg 2 quantity
            'notional1': np.random.uniform(8000, 30000, len(dates)),  # Leg 1 notional
            'notional2': np.random.uniform(8000, 30000, len(dates)),  # Leg 2 notional
            'pnl': np.random.normal(0, 800, len(dates)),  # P&L
            'unrealized_pnl': np.random.normal(0, 400, len(dates))  # Unrealized P&L
        }, index=dates)
    }
    
    # Generate pair metadata
    pair_metadata = {
        'ESM23_NQM23': {
            'leg1': 'ESM23',
            'leg2': 'NQM23',
            'asset_class': 'futures',
            'sector': 'equity_index',
            'hedge_ratio': 0.21,
            'volatility': 0.015
        },
        'GCM23_SIM23': {
            'leg1': 'GCM23',
            'leg2': 'SIM23',
            'asset_class': 'futures',
            'sector': 'metals',
            'hedge_ratio': 0.16,
            'volatility': 0.022
        }
    }
    
    # Generate asset metadata
    asset_metadata = {
        'ESM23': {
            'margin_requirement': 13200,
            'contract_size': 50,
            'tick_size': 0.25,
            'sector': 'equity_index',
            'volatility': 0.012
        },
        'NQM23': {
            'margin_requirement': 18700,
            'contract_size': 20,
            'tick_size': 0.25,
            'sector': 'equity_index',
            'volatility': 0.018
        },
        'GCM23': {
            'margin_requirement': 11275,
            'contract_size': 100,
            'tick_size': 0.10,
            'sector': 'metals',
            'volatility': 0.015
        },
        'SIM23': {
            'margin_requirement': 8800,
            'contract_size': 5000,
            'tick_size': 0.005,
            'sector': 'metals',
            'volatility': 0.025
        }
    }
    
    # Generate account data
    account = {
        'cash': 100000.0,
        'equity': 100000.0 + np.cumsum(np.random.normal(0, 200, len(dates))),
        'margin_used': np.random.uniform(20000, 40000, len(dates)),
        'margin_available': np.random.uniform(60000, 80000, len(dates)),
        'realized_pnl': np.cumsum(np.random.normal(100, 500, len(dates))),
        'unrealized_pnl': np.random.normal(0, 1000, len(dates))
    }
    account = pd.DataFrame(account, index=dates)
    
    return {
        'positions': positions,
        'pair_metadata': pair_metadata,
        'asset_metadata': asset_metadata,
        'account': account
    }

@pytest.fixture
def test_config_file():
    """Fixture providing the path to the test configuration file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(current_dir, '..', '..', 'fixtures')
    return os.path.join(fixtures_dir, 'test_config.json')

@pytest.fixture
def risk_manager(test_config_file):
    """Fixture providing a RiskManager instance."""
    # Load config
    with open(test_config_file, 'r') as f:
        config = json.load(f)
    
    # Create risk manager with config
    risk_manager = RiskManager(config.get('risk_manager', {}))
    return risk_manager

class TestRiskManager:
    """Test class for RiskManager."""
    
    def test_initialization(self, test_config_file):
        """Test that the risk manager is initialized correctly."""
        # Load config
        with open(test_config_file, 'r') as f:
            config = json.load(f)
            
        # Create risk manager
        risk_manager = RiskManager(config.get('risk_manager', {}))
        
        # Check config loaded correctly
        assert risk_manager.max_position_size == config['risk_manager']['max_position_size']
        assert risk_manager.max_pair_exposure == config['risk_manager']['max_pair_exposure']
        assert risk_manager.max_total_exposure == config['risk_manager']['max_total_exposure']
    
    def test_calculate_position_size_volatility(self, risk_manager, sample_portfolio_data):
        """Test calculating position size based on volatility."""
        # Get pair data
        pair_id = 'ESM23_NQM23'
        pair_metadata = sample_portfolio_data['pair_metadata'][pair_id]
        asset_metadata = sample_portfolio_data['asset_metadata']
        account = sample_portfolio_data['account'].iloc[-1]
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            pair_id=pair_id,
            pair_metadata=pair_metadata,
            asset_metadata=asset_metadata,
            account=account,
            method='volatility'
        )
        
        # Check results
        assert isinstance(position_size, dict)
        assert 'leg1_size' in position_size
        assert 'leg2_size' in position_size
        assert isinstance(position_size['leg1_size'], (int, float))
        assert isinstance(position_size['leg2_size'], (int, float))
        
        # Position size should respect max size constraint
        assert position_size['leg1_size'] <= risk_manager.max_position_size
        assert position_size['leg2_size'] <= risk_manager.max_position_size
        
        # Test another pair with different volatility
        pair_id2 = 'GCM23_SIM23'
        pair_metadata2 = sample_portfolio_data['pair_metadata'][pair_id2]
        
        position_size2 = risk_manager.calculate_position_size(
            pair_id=pair_id2,
            pair_metadata=pair_metadata2,
            asset_metadata=asset_metadata,
            account=account,
            method='volatility'
        )
        
        # Higher volatility pair should have smaller position size
        if pair_metadata2['volatility'] > pair_metadata['volatility']:
            assert position_size2['leg1_size'] < position_size['leg1_size']
    
    def test_calculate_position_size_equal(self, risk_manager, sample_portfolio_data):
        """Test calculating position size with equal risk allocation."""
        # Get pair data
        pair_id = 'ESM23_NQM23'
        pair_metadata = sample_portfolio_data['pair_metadata'][pair_id]
        asset_metadata = sample_portfolio_data['asset_metadata']
        account = sample_portfolio_data['account'].iloc[-1]
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            pair_id=pair_id,
            pair_metadata=pair_metadata,
            asset_metadata=asset_metadata,
            account=account,
            method='equal'
        )
        
        # Check results
        assert isinstance(position_size, dict)
        assert 'leg1_size' in position_size
        assert 'leg2_size' in position_size
        assert isinstance(position_size['leg1_size'], (int, float))
        assert isinstance(position_size['leg2_size'], (int, float))
        
        # Position size should respect max size constraint
        assert position_size['leg1_size'] <= risk_manager.max_position_size
        assert position_size['leg2_size'] <= risk_manager.max_position_size
    
    def test_calculate_exposure(self, risk_manager, sample_portfolio_data):
        """Test calculating portfolio exposure."""
        # Get portfolio data
        positions = sample_portfolio_data['positions']
        asset_metadata = sample_portfolio_data['asset_metadata']
        
        # Calculate exposure
        exposure = risk_manager.calculate_exposure(
            positions=positions,
            asset_metadata=asset_metadata
        )
        
        # Check results
        assert isinstance(exposure, dict)
        assert 'total' in exposure
        assert 'by_pair' in exposure
        assert 'by_asset' in exposure
        assert 'by_sector' in exposure
        
        # Check types and values
        assert isinstance(exposure['total'], float)
        assert isinstance(exposure['by_pair'], dict)
        assert isinstance(exposure['by_asset'], dict)
        assert exposure['total'] >= 0
        
        # Total exposure should equal sum of pair exposures
        assert np.isclose(exposure['total'], sum(exposure['by_pair'].values()))
    
    def test_check_risk_limits(self, risk_manager, sample_portfolio_data):
        """Test checking risk limits."""
        # Get portfolio data
        positions = sample_portfolio_data['positions']
        asset_metadata = sample_portfolio_data['asset_metadata']
        account = sample_portfolio_data['account'].iloc[-1]
        pair_metadata = sample_portfolio_data['pair_metadata']
        
        # Calculate exposure first
        exposure = risk_manager.calculate_exposure(
            positions=positions,
            asset_metadata=asset_metadata
        )
        
        # Check risk limits
        risk_check = risk_manager.check_risk_limits(
            exposure=exposure,
            account=account,
            pair_metadata=pair_metadata
        )
        
        # Check results
        assert isinstance(risk_check, dict)
        assert 'within_limits' in risk_check
        assert 'violations' in risk_check
        assert isinstance(risk_check['within_limits'], bool)
        assert isinstance(risk_check['violations'], list)
    
    def test_calculate_drawdown(self, risk_manager, sample_portfolio_data):
        """Test calculating drawdown."""
        # Get account data
        equity = sample_portfolio_data['account']['equity']
        
        # Calculate drawdown
        drawdown = risk_manager.calculate_drawdown(equity)
        
        # Check results
        assert isinstance(drawdown, pd.Series)
        assert len(drawdown) == len(equity)
        assert all(drawdown <= 0)  # Drawdown should be non-positive
        
        # Check maximum drawdown
        max_dd = drawdown.min()
        assert max_dd <= 0
    
    def test_calculate_var(self, risk_manager, sample_portfolio_data):
        """Test calculating value at risk."""
        # Get returns data - generate some synthetic returns
        returns = pd.Series(np.random.normal(0.001, 0.01, 250))
        
        # Calculate VaR
        var_95 = risk_manager.calculate_var(returns, confidence=0.95)
        var_99 = risk_manager.calculate_var(returns, confidence=0.99)
        
        # Check results
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
        
        # 99% VaR should be more conservative than (greater than) 95% VaR
        assert var_99 > var_95
        
        # VaR should be negative (representing losses)
        assert var_95 < 0
        assert var_99 < 0
    
    def test_adjust_for_correlation(self, risk_manager, sample_portfolio_data):
        """Test adjusting position sizes for correlation."""
        # Define two highly correlated pairs
        correlated_pairs = {
            'ESM23_NQM23': {'sector': 'equity_index', 'position_size': 2.0},
            'ESM23_MNQM23': {'sector': 'equity_index', 'position_size': 3.0}
        }
        
        # Adjust position sizes
        adjusted_sizes = risk_manager.adjust_for_correlation(correlated_pairs)
        
        # Check results
        assert isinstance(adjusted_sizes, dict)
        assert 'ESM23_NQM23' in adjusted_sizes
        assert 'ESM23_MNQM23' in adjusted_sizes
        
        # Position sizes should be reduced for correlated pairs
        assert adjusted_sizes['ESM23_NQM23'] < correlated_pairs['ESM23_NQM23']['position_size']
        assert adjusted_sizes['ESM23_MNQM23'] < correlated_pairs['ESM23_MNQM23']['position_size']
    
    def test_get_stress_test_scenarios(self, risk_manager):
        """Test generating stress test scenarios."""
        # Generate scenarios
        scenarios = risk_manager.get_stress_test_scenarios()
        
        # Check results
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        
        # Each scenario should be a dictionary with specific fields
        for scenario in scenarios:
            assert isinstance(scenario, dict)
            assert 'name' in scenario
            assert 'shocks' in scenario
            assert isinstance(scenario['shocks'], dict)
    
    def test_run_stress_test(self, risk_manager, sample_portfolio_data):
        """Test running a stress test."""
        # Get portfolio data
        positions = sample_portfolio_data['positions']
        asset_metadata = sample_portfolio_data['asset_metadata']
        pair_metadata = sample_portfolio_data['pair_metadata']
        account = sample_portfolio_data['account'].iloc[-1]
        
        # Define a simple stress scenario
        scenario = {
            'name': 'Market Crash',
            'shocks': {
                'equity_index': -0.10,  # 10% drop in equity indices
                'metals': -0.05         # 5% drop in metals
            }
        }
        
        # Run stress test
        results = risk_manager.run_stress_test(
            scenario=scenario,
            positions=positions,
            asset_metadata=asset_metadata,
            pair_metadata=pair_metadata,
            account=account
        )
        
        # Check results
        assert isinstance(results, dict)
        assert 'scenario' in results
        assert 'impact' in results
        assert 'post_stress_account' in results
        
        # Impact should be negative for a market crash scenario
        assert results['impact']['equity_change'] <= 0
    
    def test_get_position_recommendations(self, risk_manager, sample_portfolio_data):
        """Test getting position size recommendations."""
        # Get data
        pair_metadata = sample_portfolio_data['pair_metadata']
        asset_metadata = sample_portfolio_data['asset_metadata']
        account = sample_portfolio_data['account'].iloc[-1]
        
        # Get recommendations for multiple pairs
        recommendations = risk_manager.get_position_recommendations(
            pair_metadata=pair_metadata,
            asset_metadata=asset_metadata,
            account=account
        )
        
        # Check results
        assert isinstance(recommendations, dict)
        assert all(pair_id in recommendations for pair_id in pair_metadata)
        
        # Each recommendation should include leg sizes
        for pair_id, rec in recommendations.items():
            assert 'leg1_size' in rec
            assert 'leg2_size' in rec
            assert isinstance(rec['leg1_size'], (int, float))
            assert isinstance(rec['leg2_size'], (int, float))
            
            # Sizes should be reasonable
            assert 0 <= rec['leg1_size'] <= risk_manager.max_position_size
            assert 0 <= rec['leg2_size'] <= risk_manager.max_position_size 