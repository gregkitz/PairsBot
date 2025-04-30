"""
Advanced Unit Tests for Statistical Methods

This module contains enhanced tests for advanced statistical methods used in cointegration analysis,
focusing on edge cases, numerical stability, and complex real-world scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.cointegration.statistical_methods import (
    phillips_ouliaris_test,
    detect_structural_breaks,
    analyze_residuals,
    calculate_hurst_exponent,
    johansen_test,
    engle_granger_test
)

@pytest.fixture
def complex_price_data():
    """
    Fixture providing complex price data with specific statistical properties.
    
    This includes:
    1. Cointegrated series with structural breaks
    2. Series with time-varying volatility
    3. Series with different persistence levels
    4. Series with seasonal patterns
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate 500 days of data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    # 1. Base random walk series
    random_changes = np.random.normal(0, 1, 500)
    series1 = 100 + np.cumsum(random_changes)
    
    # 2. Cointegrated series with a structural break at t=250
    hedge_ratio_1 = 0.7
    hedge_ratio_2 = 0.3
    noise_1 = np.random.normal(0, 0.5, 250)
    noise_2 = np.random.normal(0, 0.8, 250)  # More noise after break
    
    series2 = np.zeros(500)
    series2[:250] = hedge_ratio_1 * series1[:250] + 20 + noise_1
    series2[250:] = hedge_ratio_2 * series1[250:] + 50 + noise_2
    
    # 3. Series with time-varying volatility (GARCH-like behavior)
    vol_series = np.zeros(500)
    volatility = np.ones(500)
    for i in range(1, 500):
        # Update volatility (simple GARCH-like process)
        volatility[i] = 0.1 + 0.8 * volatility[i-1] + 0.1 * vol_series[i-1]**2
        vol_series[i] = np.random.normal(0, np.sqrt(volatility[i]))
    
    # Make it cointegrated with series1 but with variable volatility
    series3 = 0.5 * series1 + 30 + vol_series
    
    # 4. Series with seasonal pattern
    seasonal_component = 10 * np.sin(2 * np.pi * np.arange(500) / 365.25)  # Annual cycle
    series4 = 0.6 * series1 + 40 + seasonal_component + np.random.normal(0, 0.7, 500)
    
    # 5. Nearly non-stationary series (highly persistent)
    persistent_series = np.zeros(500)
    persistent_series[0] = 0
    for i in range(1, 500):
        persistent_series[i] = 0.99 * persistent_series[i-1] + np.random.normal(0, 0.5)
    
    # 6. Mean-reverting series with different half-lives
    mean_rev_1 = np.zeros(500)  # Fast mean-reversion
    mean_rev_2 = np.zeros(500)  # Slow mean-reversion
    
    for i in range(1, 500):
        mean_rev_1[i] = 0.7 * mean_rev_1[i-1] + np.random.normal(0, 1)  # Half-life ~2
        mean_rev_2[i] = 0.95 * mean_rev_2[i-1] + np.random.normal(0, 1)  # Half-life ~14
    
    # Package all series into pandas Series objects
    s1 = pd.Series(series1, index=dates, name='base_random_walk')
    s2 = pd.Series(series2, index=dates, name='structural_break')
    s3 = pd.Series(series3, index=dates, name='time_varying_vol')
    s4 = pd.Series(series4, index=dates, name='seasonal')
    s5 = pd.Series(persistent_series, index=dates, name='persistent')
    s6 = pd.Series(mean_rev_1, index=dates, name='fast_mean_rev')
    s7 = pd.Series(mean_rev_2, index=dates, name='slow_mean_rev')
    
    # Create DataFrame
    df = pd.DataFrame({
        'base_random_walk': s1,
        'structural_break': s2,
        'time_varying_vol': s3,
        'seasonal': s4,
        'persistent': s5,
        'fast_mean_rev': s6,
        'slow_mean_rev': s7
    })
    
    return {
        'dataframe': df,
        'base_random_walk': s1,
        'structural_break': s2,
        'time_varying_vol': s3,
        'seasonal': s4,
        'persistent': s5,
        'fast_mean_rev': s6,
        'slow_mean_rev': s7,
        'hedge_ratios': [hedge_ratio_1, hedge_ratio_2],
        'break_point': 250,
        'dates': dates
    }


class TestPhillipsOuliarisAdvanced:
    """Enhanced tests for Phillips-Ouliaris cointegration test."""
    
    def test_structural_break_impact(self, complex_price_data):
        """Test Phillips-Ouliaris test with a series containing a structural break."""
        y = complex_price_data['structural_break']
        x = complex_price_data['base_random_walk']
        
        # Test on the entire series (with structural break)
        full_result = phillips_ouliaris_test(y, x)
        
        # Test on the pre-break period
        break_point = complex_price_data['break_point']
        pre_break_result = phillips_ouliaris_test(y[:break_point], x[:break_point])
        
        # Test on the post-break period
        post_break_result = phillips_ouliaris_test(y[break_point:], x[break_point:])
        
        # Pre and post break periods should be cointegrated when tested separately
        assert pre_break_result['is_cointegrated'] is True
        assert post_break_result['is_cointegrated'] is True
        
        # The full series might not be identified as cointegrated due to the structural break
        # Or might be identified with a different hedge ratio
        if full_result['is_cointegrated']:
            # Check if the full-series hedge ratio is between the two segment hedge ratios
            assert abs(full_result['hedge_ratio'] - pre_break_result['hedge_ratio']) > 0.1
            assert abs(full_result['hedge_ratio'] - post_break_result['hedge_ratio']) > 0.1
    
    def test_with_time_varying_volatility(self, complex_price_data):
        """Test Phillips-Ouliaris test with time-varying volatility in the series."""
        y = complex_price_data['time_varying_vol']
        x = complex_price_data['base_random_walk']
        
        # Test with different kernel options to handle heteroskedasticity
        bartlett_result = phillips_ouliaris_test(y, x, kernel='bartlett')
        parzen_result = phillips_ouliaris_test(y, x, kernel='parzen')
        quadratic_result = phillips_ouliaris_test(y, x, kernel='quadratic')
        
        # All should detect cointegration despite heteroskedasticity
        assert bartlett_result['is_cointegrated'] is True
        assert parzen_result['is_cointegrated'] is True
        assert quadratic_result['is_cointegrated'] is True
        
        # Compare statistics - they should be different due to different kernel weighting
        assert bartlett_result['test_statistic'] != parzen_result['test_statistic']
        assert bartlett_result['test_statistic'] != quadratic_result['test_statistic']
    
    def test_with_seasonal_patterns(self, complex_price_data):
        """Test Phillips-Ouliaris test with seasonal patterns in the series."""
        y = complex_price_data['seasonal']
        x = complex_price_data['base_random_walk']
        
        # Test with different regression methods
        ols_result = phillips_ouliaris_test(y, x, regression_method='ols')
        gls_result = phillips_ouliaris_test(y, x, regression_method='gls')
        
        # Both should detect the underlying cointegration relationship
        assert ols_result['is_cointegrated'] is True
        assert gls_result['is_cointegrated'] is True
        
        # Compare the hedge ratios - should be around 0.6 (from data generation)
        assert 0.5 < ols_result['hedge_ratio'] < 0.7
        assert 0.5 < gls_result['hedge_ratio'] < 0.7
    
    def test_test_types_comparison(self, complex_price_data):
        """Test different test types (Zt vs Za) in Phillips-Ouliaris test."""
        y = complex_price_data['structural_break']
        x = complex_price_data['base_random_walk']
        
        # Test with both test types
        zt_result = phillips_ouliaris_test(y, x, test_type='Zt')
        za_result = phillips_ouliaris_test(y, x, test_type='Za')
        
        # Both tests should exist even though they might give different results
        assert 'test_statistic' in zt_result
        assert 'test_statistic' in za_result
        assert 'critical_values' in zt_result
        assert 'critical_values' in za_result
        
        # The test statistics should be different
        assert zt_result['test_statistic'] != za_result['test_statistic']
        
        # Za test is often more conservative - might detect cointegration less often
        if zt_result['is_cointegrated'] != za_result['is_cointegrated']:
            print("Different cointegration results between Zt and Za tests.")
            print(f"Zt test: {zt_result['is_cointegrated']}")
            print(f"Za test: {za_result['is_cointegrated']}")
            print(f"Zt test statistic: {zt_result['test_statistic']}")
            print(f"Za test statistic: {za_result['test_statistic']}")


class TestStructuralBreakDetection:
    """Enhanced tests for structural break detection."""
    
    def test_detect_known_break(self, complex_price_data):
        """Test detection of a known structural break."""
        y = complex_price_data['structural_break']
        x = complex_price_data['base_random_walk']
        known_break = complex_price_data['break_point']
        
        # Test with different detection methods
        recursive_result = detect_structural_breaks(y, x, test_method='recursive_cusum')
        ols_cusum_result = detect_structural_breaks(y, x, test_method='ols_cusum')
        chow_result = detect_structural_breaks(y, x, test_method='chow_test')
        
        # All methods should detect at least one break
        assert len(recursive_result['break_points']) > 0
        assert len(ols_cusum_result['break_points']) > 0
        assert len(chow_result['break_points']) > 0
        
        # At least one method should detect a break close to the known break point
        break_detected = False
        for result in [recursive_result, ols_cusum_result, chow_result]:
            for break_point in result['break_points']:
                if abs(break_point - known_break) < 30:  # Within 30 data points of the actual break
                    break_detected = True
                    break
        
        assert break_detected, "No method detected the known structural break within tolerance"
    
    def test_multiple_breaks(self):
        """Test detection of multiple structural breaks."""
        # Create series with multiple known breaks
        np.random.seed(42)
        n_points = 600
        dates = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
        
        # Base series
        base_series = 100 + np.cumsum(np.random.normal(0, 1, n_points))
        
        # Series with 3 breaks (at t=150, t=300, t=450)
        multi_break_series = np.zeros(n_points)
        
        # Segment 1: 0-150
        multi_break_series[:150] = 0.8 * base_series[:150] + 20 + np.random.normal(0, 0.5, 150)
        
        # Segment 2: 151-300
        multi_break_series[150:300] = 0.4 * base_series[150:300] + 40 + np.random.normal(0, 0.5, 150)
        
        # Segment 3: 301-450
        multi_break_series[300:450] = 0.6 * base_series[300:450] + 10 + np.random.normal(0, 0.5, 150)
        
        # Segment 4: 451-600
        multi_break_series[450:] = 0.9 * base_series[450:] + 5 + np.random.normal(0, 0.5, 150)
        
        # Convert to Series
        y = pd.Series(multi_break_series, index=dates)
        x = pd.Series(base_series, index=dates)
        
        # Known break points
        known_breaks = [150, 300, 450]
        
        # Detect breaks
        result = detect_structural_breaks(y, x, test_method='recursive_cusum', min_segment_size=50)
        
        # Should detect at least 2 of the 3 breaks
        assert len(result['break_points']) >= 2
        
        # Check if detected breaks are close to known breaks
        detected_count = 0
        for known_break in known_breaks:
            for detected_break in result['break_points']:
                if abs(detected_break - known_break) < 30:  # Within 30 data points
                    detected_count += 1
                    break
        
        assert detected_count >= 2, f"Only detected {detected_count} of 3 known breaks"
    
    def test_no_breaks(self, complex_price_data):
        """Test with series that don't have structural breaks."""
        # Use pre-break data only
        break_point = complex_price_data['break_point']
        y = complex_price_data['structural_break'][:break_point]
        x = complex_price_data['base_random_walk'][:break_point]
        
        # Detect breaks
        result = detect_structural_breaks(y, x, test_method='recursive_cusum')
        
        # Should not detect significant breaks
        assert len(result['break_points']) == 0 or result['break_significance'][0] > 0.01
    
    def test_break_with_different_regression_methods(self, complex_price_data):
        """Test break detection with different regression methods."""
        y = complex_price_data['structural_break']
        x = complex_price_data['base_random_walk']
        
        # Test with different regression methods
        ols_result = detect_structural_breaks(y, x, regression_method='ols')
        gls_result = detect_structural_breaks(y, x, regression_method='gls')
        robust_result = detect_structural_breaks(y, x, regression_method='robust')
        
        # All methods should produce results with the expected structure
        for result in [ols_result, gls_result, robust_result]:
            assert 'break_points' in result
            assert 'break_significance' in result
            assert 'regression_coefficients' in result


class TestHurstExponentAdvanced:
    """Enhanced tests for Hurst exponent calculation."""
    
    def test_known_mean_reversion_strengths(self, complex_price_data):
        """Test Hurst exponent with series of known mean-reversion strengths."""
        # Fast mean-reverting series should have lower Hurst exponent
        fast_mr = complex_price_data['fast_mean_rev']
        slow_mr = complex_price_data['slow_mean_rev']
        
        fast_result = calculate_hurst_exponent(fast_mr)
        slow_result = calculate_hurst_exponent(slow_mr)
        
        # Mean-reverting series should have H < 0.5
        assert fast_result['hurst_exponent'] < 0.5
        assert slow_result['hurst_exponent'] < 0.5
        
        # Faster mean-reversion should have lower H
        assert fast_result['hurst_exponent'] < slow_result['hurst_exponent']
    
    def test_with_seasonal_data(self, complex_price_data):
        """Test Hurst exponent with seasonal data."""
        seasonal = complex_price_data['seasonal']
        
        # Test with different max lag values to capture seasonality
        short_lag_result = calculate_hurst_exponent(seasonal, max_lag=50)  # Won't capture annual pattern
        long_lag_result = calculate_hurst_exponent(seasonal, max_lag=200)  # Should capture annual pattern
        
        # Results should differ when using different lag ranges
        assert abs(short_lag_result['hurst_exponent'] - long_lag_result['hurst_exponent']) > 0.05
    
    def test_persistence_level(self, complex_price_data):
        """Test Hurst exponent with different persistence levels."""
        # Highly persistent (nearly non-stationary) series
        persistent = complex_price_data['persistent']
        # Random walk (should have H ≈ 0.5)
        random_walk = complex_price_data['base_random_walk']
        
        persistent_result = calculate_hurst_exponent(persistent)
        random_walk_result = calculate_hurst_exponent(random_walk)
        
        # Persistent series should have H > 0.5 (trending)
        assert persistent_result['hurst_exponent'] > 0.5
        
        # Random walk should have H ≈ 0.5
        assert 0.45 < random_walk_result['hurst_exponent'] < 0.55
    
    def test_with_log_returns(self, complex_price_data):
        """Test Hurst exponent when applied to log returns instead of price levels."""
        # Calculate log returns
        prices = complex_price_data['base_random_walk']
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        # Calculate Hurst exponent on log returns
        result = calculate_hurst_exponent(log_returns)
        
        # Log returns of a random walk should be close to white noise (H ≈ 0.5)
        assert 0.4 < result['hurst_exponent'] < 0.6
        assert result['mean_reversion_strength'] == 'weak or none'


class TestResidualAnalysisAdvanced:
    """Enhanced tests for residual analysis."""
    
    def test_residuals_with_structural_break(self, complex_price_data):
        """Test residual analysis when the underlying relationship has a structural break."""
        y = complex_price_data['structural_break']
        x = complex_price_data['base_random_walk']
        
        # Run regression on the full sample
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        # Analyze the residuals
        result = analyze_residuals(residuals)
        
        # Residuals from a regression with a structural break should have:
        # - Failed stationarity test (due to mean shift)
        # - Significant autocorrelation
        # - Possible heteroskedasticity
        assert result['is_stationary'] is False
        assert result['has_serial_correlation'] is True
    
    def test_residuals_with_heteroskedasticity(self, complex_price_data):
        """Test residual analysis with heteroskedastic errors."""
        y = complex_price_data['time_varying_vol']
        x = complex_price_data['base_random_walk']
        
        # Run regression
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        # Analyze the residuals
        result = analyze_residuals(residuals)
        
        # Residuals should show heteroskedasticity
        assert result['has_heteroskedasticity'] is True
        
        # Print test statistics for manual verification
        print(f"Heteroskedasticity test p-value: {result['heteroskedasticity_test_p_value']}")
    
    def test_residuals_with_seasonality(self, complex_price_data):
        """Test residual analysis with seasonal patterns in the residuals."""
        y = complex_price_data['seasonal']
        x = complex_price_data['base_random_walk']
        
        # Run regression
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        # Analyze the residuals
        result = analyze_residuals(residuals)
        
        # Residuals should have:
        # - Serial correlation (due to unmodeled seasonality)
        assert result['has_serial_correlation'] is True
        
        # Potentially still stationary if seasonality is not too strong
        if result['is_stationary']:
            print("Residuals are stationary despite seasonal pattern in the data")
        else:
            print("Residuals are non-stationary due to seasonal pattern")
    
    def test_comprehensive_diagnostics(self, complex_price_data):
        """Test all diagnostic statistics from residual analysis."""
        # Generate some residuals with known properties
        np.random.seed(42)
        n = 500
        
        # Create autocorrelated residuals (AR(1) process)
        ar_residuals = np.zeros(n)
        ar_residuals[0] = np.random.normal(0, 1)
        for i in range(1, n):
            ar_residuals[i] = 0.7 * ar_residuals[i-1] + np.random.normal(0, 1)
        
        # Analyze the residuals
        result = analyze_residuals(ar_residuals)
        
        # Should detect autocorrelation
        assert result['has_serial_correlation'] is True
        
        # Check all returned diagnostics
        assert 'is_stationary' in result
        assert 'adf_p_value' in result
        assert 'has_serial_correlation' in result
        assert 'ljung_box_p_value' in result
        assert 'has_heteroskedasticity' in result
        assert 'heteroskedasticity_test_p_value' in result
        assert 'is_normally_distributed' in result
        assert 'jarque_bera_p_value' in result
        
        # Print diagnostics for verification
        for key, value in result.items():
            print(f"{key}: {value}")


class TestJohansenAdvanced:
    """Enhanced tests for Johansen cointegration test."""
    
    def test_with_structural_breaks(self, complex_price_data):
        """Test Johansen test when the relationship has structural breaks."""
        # Create dataframe with the random walk and structural break series
        df = pd.DataFrame({
            'random_walk': complex_price_data['base_random_walk'],
            'structural_break': complex_price_data['structural_break']
        })
        
        # Full sample test
        full_result = johansen_test(df)
        
        # Split sample tests (before and after break)
        break_point = complex_price_data['break_point']
        pre_break_df = df.iloc[:break_point]
        post_break_df = df.iloc[break_point:]
        
        pre_result = johansen_test(pre_break_df)
        post_result = johansen_test(post_break_df)
        
        # Check if the sub-samples show stronger evidence of cointegration
        if full_result['n_cointegrating_relations_trace'] == 0:
            # If full sample shows no cointegration, at least one sub-sample should show it
            assert pre_result['n_cointegrating_relations_trace'] > 0 or post_result['n_cointegrating_relations_trace'] > 0
        
        # Print detailed results
        print(f"Full sample cointegration relations: {full_result['n_cointegrating_relations_trace']}")
        print(f"Pre-break cointegration relations: {pre_result['n_cointegrating_relations_trace']}")
        print(f"Post-break cointegration relations: {post_result['n_cointegrating_relations_trace']}")
    
    def test_with_multiple_cointegrating_relations(self, complex_price_data):
        """Test Johansen test with multiple potential cointegration relationships."""
        # Create dataframe with several related series
        df = pd.DataFrame({
            'random_walk': complex_price_data['base_random_walk'],
            'cointegrated1': 0.5 * complex_price_data['base_random_walk'] + 30 + np.random.normal(0, 0.7, 500),
            'cointegrated2': 0.7 * complex_price_data['base_random_walk'] + 20 + np.random.normal(0, 0.7, 500),
            'non_cointegrated': complex_price_data['persistent']
        })
        
        # Run Johansen test
        result = johansen_test(df)
        
        # Should find multiple cointegrating relations
        assert result['n_cointegrating_relations_trace'] >= 2
        
        # Check eigenvectors
        if result['eigenvectors'] is not None:
            assert len(result['eigenvectors']) == df.shape[1]
    
    def test_with_different_lag_orders(self, complex_price_data):
        """Test Johansen test with different lag orders."""
        # Create dataframe with related series
        df = pd.DataFrame({
            'random_walk': complex_price_data['base_random_walk'],
            'structural_break': complex_price_data['structural_break']
        })
        
        # Test with different lag orders
        result_lag1 = johansen_test(df, k_ar_diff=1)
        result_lag2 = johansen_test(df, k_ar_diff=2)
        result_lag5 = johansen_test(df, k_ar_diff=5)
        
        # Print results for different lag orders
        print(f"Lag 1: {result_lag1['n_cointegrating_relations_trace']} cointegrating relations")
        print(f"Lag 2: {result_lag2['n_cointegrating_relations_trace']} cointegrating relations")
        print(f"Lag 5: {result_lag5['n_cointegrating_relations_trace']} cointegrating relations")
        
        # Different lag orders may produce different results
        # We're testing that the function handles different lag orders properly
        for result in [result_lag1, result_lag2, result_lag5]:
            assert 'trace_statistic' in result
            assert 'eigenvalues' in result
            assert 'n_cointegrating_relations_trace' in result 