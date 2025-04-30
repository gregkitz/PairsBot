"""
Integration Tests for Statistical Methods

This module contains tests that evaluate the integration of multiple statistical methods
working together to analyze cointegration in realistic market scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime, timedelta

from src.cointegration.statistical_methods import (
    johansen_test,
    engle_granger_test,
    phillips_ouliaris_test,
    detect_structural_breaks,
    analyze_residuals,
    calculate_half_life,
    calculate_hurst_exponent
)

@pytest.fixture
def market_scenario_data():
    """
    Create realistic market scenario data with regime changes.
    
    This includes:
    1. Normal market period with stable cointegration
    2. Market stress period with weakened cointegration
    3. Recovery period with restored cointegration at a different level
    """
    np.random.seed(42)
    
    # Generate 2 years of daily data with three distinct regimes
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='B')
    n_days = len(dates)
    
    # Split into three regimes
    normal_end = int(n_days * 0.4)      # 40% of data: normal period
    stress_end = int(n_days * 0.7)      # 30% of data: stress period
    recovery_end = n_days               # 30% of data: recovery period
    
    # Base random walk
    returns = np.random.normal(0.0002, 0.01, n_days)  # Slight upward drift
    
    # Add volatility regimes
    # Higher volatility during stress period
    returns[normal_end:stress_end] = np.random.normal(0.0001, 0.02, stress_end - normal_end)
    returns[stress_end:] = np.random.normal(0.0004, 0.012, recovery_end - stress_end)
    
    # Convert to price series
    base_price = 100 * np.exp(np.cumsum(returns))
    
    # Create cointegrated asset with regime-specific properties
    related_price = np.zeros(n_days)
    
    # Normal regime: Strong cointegration
    hedge_ratio_normal = 0.7
    noise_normal = np.random.normal(0, 0.5, normal_end)
    related_price[:normal_end] = hedge_ratio_normal * base_price[:normal_end] + 20 + noise_normal
    
    # Stress regime: Weaker/breaking cointegration
    hedge_ratio_stress = 0.5  # Different relationship
    noise_stress = np.random.normal(0, 1.5, stress_end - normal_end)  # More noise
    # Add a jump at the regime change
    jump = 5.0
    related_price[normal_end:stress_end] = (
        hedge_ratio_stress * base_price[normal_end:stress_end] + 
        15 + noise_stress + jump
    )
    
    # Recovery regime: Restored cointegration but at different level
    hedge_ratio_recovery = 0.8
    noise_recovery = np.random.normal(0, 0.7, recovery_end - stress_end)
    # Ensure smooth transition from stress to recovery
    level_adjustment = related_price[stress_end-1] - (hedge_ratio_recovery * base_price[stress_end-1] + 25)
    related_price[stress_end:] = hedge_ratio_recovery * base_price[stress_end:] + 25 + noise_recovery
    
    # Create Series objects
    base_series = pd.Series(base_price, index=dates, name='base_asset')
    related_series = pd.Series(related_price, index=dates, name='related_asset')
    
    # Calculate spread series for each regime
    spread_normal = related_series[:normal_end] - hedge_ratio_normal * base_series[:normal_end]
    spread_stress = related_series[normal_end:stress_end] - hedge_ratio_stress * base_series[normal_end:stress_end]
    spread_recovery = related_series[stress_end:] - hedge_ratio_recovery * base_series[stress_end:]
    
    # Full spread series (using most recent hedge ratio - not accurate over regimes)
    spread_full = related_series - 0.7 * base_series
    
    # Create DataFrame
    df = pd.DataFrame({
        'base_asset': base_series,
        'related_asset': related_series
    })
    
    return {
        'dataframe': df,
        'base_asset': base_series,
        'related_asset': related_series,
        'spread_normal': spread_normal,
        'spread_stress': spread_stress,
        'spread_recovery': spread_recovery,
        'spread_full': spread_full,
        'regime_breaks': [normal_end, stress_end],
        'dates': dates,
        'hedge_ratios': [hedge_ratio_normal, hedge_ratio_stress, hedge_ratio_recovery]
    }


class TestIntegratedAnalysis:
    """Tests for integrated cointegration analysis of realistic market scenarios."""
    
    def test_regime_detection_and_analysis(self, market_scenario_data):
        """Test the ability to detect and analyze different market regimes."""
        data = market_scenario_data
        base_asset = data['base_asset']
        related_asset = data['related_asset']
        
        # Step 1: Detect structural breaks in the relationship
        breaks_result = detect_structural_breaks(
            related_asset, 
            base_asset,
            test_method='recursive_cusum',
            min_segment_size=60  # Approximately 3 months of trading days
        )
        
        # Should detect structural breaks close to the actual regime changes
        assert len(breaks_result['break_points']) >= 1
        
        # At least one detected break should be close to an actual regime break
        actual_breaks = data['regime_breaks']
        detected_breaks = breaks_result['break_points']
        
        break_detection_success = False
        for actual_break in actual_breaks:
            for detected_break in detected_breaks:
                if abs(detected_break - actual_break) < 30:  # Within 30 days
                    break_detection_success = True
                    break
        
        assert break_detection_success, "Failed to detect any of the actual regime changes"
        
        # Step 2: Analyze each regime separately
        # If we detected at least one break, use it to split the data
        if len(detected_breaks) > 0:
            # Sort detected breaks
            detected_breaks = sorted(detected_breaks)
            
            # Split into segments
            segments = []
            start_idx = 0
            
            # Add each segment defined by the breaks
            for break_point in detected_breaks:
                segments.append((start_idx, break_point))
                start_idx = break_point
            
            # Add the final segment
            segments.append((start_idx, len(base_asset)))
            
            # Analyze cointegration in each segment
            segment_results = []
            
            for i, (start, end) in enumerate(segments):
                segment_base = base_asset.iloc[start:end]
                segment_related = related_asset.iloc[start:end]
                
                # Skip very short segments
                if len(segment_base) < 60:
                    continue
                
                # Run Engle-Granger test on the segment
                eg_result = engle_granger_test(segment_related, segment_base)
                
                # Calculate half-life of the spread in this segment
                segment_spread = segment_related - eg_result['hedge_ratio'] * segment_base
                hl_result = calculate_half_life(segment_spread)
                
                # Store results for this segment
                segment_results.append({
                    'segment': (start, end),
                    'engle_granger': eg_result,
                    'half_life': hl_result,
                    'is_cointegrated': eg_result['is_cointegrated'],
                    'hedge_ratio': eg_result['hedge_ratio'],
                    'n_observations': len(segment_base)
                })
            
            # Check if at least one segment is cointegrated
            assert any(result['is_cointegrated'] for result in segment_results)
            
            # Check for differing hedge ratios between segments
            if len(segment_results) > 1:
                hedge_ratios = [result['hedge_ratio'] for result in segment_results]
                # At least one pair of hedge ratios should differ significantly
                ratio_differences = [abs(a - b) for a in hedge_ratios for b in hedge_ratios if a != b]
                assert any(diff > 0.1 for diff in ratio_differences)
    
    def test_multiple_cointegration_tests_comparison(self, market_scenario_data):
        """Compare results from different cointegration tests on the same data."""
        data = market_scenario_data
        
        # Focus on the normal regime for this test
        normal_end = data['regime_breaks'][0]
        base_asset = data['base_asset'][:normal_end]
        related_asset = data['related_asset'][:normal_end]
        
        # Run multiple cointegration tests
        # 1. Engle-Granger test
        eg_result = engle_granger_test(related_asset, base_asset)
        
        # 2. Phillips-Ouliaris test
        po_result = phillips_ouliaris_test(related_asset, base_asset)
        
        # 3. Johansen test
        jo_result = johansen_test(pd.DataFrame({'base': base_asset, 'related': related_asset}))
        
        # All tests should agree on cointegration in the normal regime
        assert eg_result['is_cointegrated'] is True
        assert po_result['is_cointegrated'] is True
        assert jo_result['n_cointegrating_relations_trace'] >= 1
        
        # Hedge ratios should be similar
        eg_hedge_ratio = eg_result['hedge_ratio']
        po_hedge_ratio = po_result['hedge_ratio']
        
        assert abs(eg_hedge_ratio - po_hedge_ratio) < 0.1
        assert abs(eg_hedge_ratio - data['hedge_ratios'][0]) < 0.1
    
    def test_stress_period_cointegration_breakdown(self, market_scenario_data):
        """Test cointegration breakdown during the stress period."""
        data = market_scenario_data
        
        # Extract stress period data
        stress_start = data['regime_breaks'][0]
        stress_end = data['regime_breaks'][1]
        
        base_stress = data['base_asset'][stress_start:stress_end]
        related_stress = data['related_asset'][stress_start:stress_end]
        
        # Run combined analysis
        # 1. Cointegration test
        eg_result = engle_granger_test(related_stress, base_stress)
        
        # 2. Spread analysis
        spread = related_stress - eg_result['hedge_ratio'] * base_stress
        half_life = calculate_half_life(spread)
        residual_analysis = analyze_residuals(spread)
        hurst_result = calculate_hurst_exponent(spread)
        
        # The stress period should show:
        # - Weaker or no cointegration
        # - Longer half-life (slower mean reversion)
        # - Less stationary residuals
        # - Higher Hurst exponent (closer to random walk)
        
        # Print detailed results for analysis
        print("\nStress period analysis:")
        print(f"Cointegrated: {eg_result['is_cointegrated']}")
        print(f"p-value: {eg_result['p_value']}")
        print(f"Half-life: {half_life}")
        print(f"Residuals stationary: {residual_analysis['is_stationary']}")
        print(f"Hurst exponent: {hurst_result['hurst_exponent']}")
        
        # In stress periods, statistical properties often deteriorate
        if not eg_result['is_cointegrated']:
            print("Cointegration breaks down in stress period as expected")
        else:
            # If still cointegrated, other metrics should show deterioration
            assert half_life > 10  # Slower mean reversion
        
        # Hurst exponent should be higher in stress period (closer to random walk)
        assert hurst_result['hurst_exponent'] > 0.4
    
    def test_recovery_period_new_equilibrium(self, market_scenario_data):
        """Test cointegration restoration in recovery period but at a new level."""
        data = market_scenario_data
        
        # Extract recovery period data
        recovery_start = data['regime_breaks'][1]
        
        base_recovery = data['base_asset'][recovery_start:]
        related_recovery = data['related_asset'][recovery_start:]
        
        # Run cointegration tests
        eg_result = engle_granger_test(related_recovery, base_recovery)
        
        # The recovery period should show:
        # - Restored cointegration
        # - New hedge ratio different from normal period
        
        print("\nRecovery period analysis:")
        print(f"Cointegrated: {eg_result['is_cointegrated']}")
        print(f"Hedge ratio: {eg_result['hedge_ratio']}")
        print(f"Expected hedge ratio: {data['hedge_ratios'][2]}")
        
        # Should be cointegrated in recovery period
        assert eg_result['is_cointegrated'] is True
        
        # Hedge ratio should be close to the recovery period hedge ratio (not the normal period)
        assert abs(eg_result['hedge_ratio'] - data['hedge_ratios'][2]) < 0.1
        
        # Should be different from normal period hedge ratio
        assert abs(eg_result['hedge_ratio'] - data['hedge_ratios'][0]) > 0.05
    
    def test_full_period_analysis_limitations(self, market_scenario_data):
        """Test limitations of using a single model for the full period with regime changes."""
        data = market_scenario_data
        
        # Get full period data
        base_full = data['base_asset']
        related_full = data['related_asset']
        
        # Run cointegration test on full period
        eg_result_full = engle_granger_test(related_full, base_full)
        
        # Run separate tests on each regime
        normal_end, stress_end = data['regime_breaks']
        
        eg_result_normal = engle_granger_test(
            related_full[:normal_end], 
            base_full[:normal_end]
        )
        
        eg_result_stress = engle_granger_test(
            related_full[normal_end:stress_end], 
            base_full[normal_end:stress_end]
        )
        
        eg_result_recovery = engle_granger_test(
            related_full[stress_end:], 
            base_full[stress_end:]
        )
        
        # Compare hedge ratios across regimes
        print("\nHedge ratios by regime:")
        print(f"Full period: {eg_result_full['hedge_ratio']}")
        print(f"Normal regime: {eg_result_normal['hedge_ratio']}")
        print(f"Stress regime: {eg_result_stress['hedge_ratio']}")
        print(f"Recovery regime: {eg_result_recovery['hedge_ratio']}")
        
        # Full period analysis fails to capture regime-specific relationships
        normal_error = abs(eg_result_full['hedge_ratio'] - eg_result_normal['hedge_ratio'])
        stress_error = abs(eg_result_full['hedge_ratio'] - eg_result_stress['hedge_ratio'])
        recovery_error = abs(eg_result_full['hedge_ratio'] - eg_result_recovery['hedge_ratio'])
        
        print("\nHedge ratio estimation errors relative to full period:")
        print(f"Normal regime error: {normal_error}")
        print(f"Stress regime error: {stress_error}")
        print(f"Recovery regime error: {recovery_error}")
        
        # At least one regime should show significant difference from full-period estimate
        assert normal_error > 0.05 or stress_error > 0.05 or recovery_error > 0.05
    
    def test_integrated_cointegration_workflow(self, market_scenario_data):
        """Test a complete integrated workflow for cointegration analysis."""
        data = market_scenario_data
        base_asset = data['base_asset']
        related_asset = data['related_asset']
        
        # Step 1: Check for structural breaks
        breaks_result = detect_structural_breaks(related_asset, base_asset)
        
        # Step 2: Use break information to segment the data
        segments = []
        if len(breaks_result['break_points']) > 0:
            # If breaks detected, use them to segment the data
            break_points = sorted([0] + breaks_result['break_points'] + [len(base_asset)])
            segments = [(break_points[i], break_points[i+1]) for i in range(len(break_points)-1)]
        else:
            # If no breaks detected, use the full period
            segments = [(0, len(base_asset))]
        
        # Step 3: Analyze each segment
        segment_analyses = []
        
        for i, (start, end) in enumerate(segments):
            if end - start < 60:  # Skip very short segments
                continue
                
            segment_base = base_asset.iloc[start:end]
            segment_related = related_asset.iloc[start:end]
            
            # Cointegration tests
            eg_result = engle_granger_test(segment_related, segment_base)
            
            # If cointegrated, analyze the spread properties
            segment_analysis = {
                'segment': (start, end),
                'is_cointegrated': eg_result['is_cointegrated'],
                'hedge_ratio': eg_result['hedge_ratio'],
                'p_value': eg_result['p_value']
            }
            
            if eg_result['is_cointegrated']:
                # Calculate spread and analyze its properties
                spread = segment_related - eg_result['hedge_ratio'] * segment_base
                
                half_life_result = calculate_half_life(spread)
                hurst_result = calculate_hurst_exponent(spread)
                residual_result = analyze_residuals(spread)
                
                segment_analysis.update({
                    'half_life': half_life_result,
                    'hurst_exponent': hurst_result['hurst_exponent'],
                    'is_spread_stationary': residual_result['is_stationary'],
                    'has_serial_correlation': residual_result['has_serial_correlation']
                })
            
            segment_analyses.append(segment_analysis)
        
        # Step 4: Validate the results
        # At least one segment should be cointegrated
        assert any(analysis['is_cointegrated'] for analysis in segment_analyses)
        
        # Segments should show different characteristics
        if len(segment_analyses) > 1:
            # At least one pair of segments should have different hedge ratios
            hedge_ratios = [analysis['hedge_ratio'] for analysis in segment_analyses]
            hedge_ratio_diffs = [abs(a - b) for i, a in enumerate(hedge_ratios) 
                               for b in hedge_ratios[i+1:]]
            
            assert any(diff > 0.05 for diff in hedge_ratio_diffs)
            
        # Print summary of findings
        print("\nSegmented Analysis Results:")
        for i, analysis in enumerate(segment_analyses):
            print(f"\nSegment {i+1} ({analysis['segment'][0]}-{analysis['segment'][1]}):")
            print(f"Cointegrated: {analysis['is_cointegrated']}")
            print(f"Hedge ratio: {analysis['hedge_ratio']:.4f}")
            print(f"p-value: {analysis['p_value']:.4f}")
            
            if analysis['is_cointegrated']:
                print(f"Half-life: {analysis['half_life']}")
                print(f"Hurst exponent: {analysis['hurst_exponent']:.4f}")
                print(f"Spread stationary: {analysis['is_spread_stationary']}")
                print(f"Serial correlation: {analysis['has_serial_correlation']}")
                
        # The test is considered successful if we could complete the entire workflow 