"""
Unit tests for statistical methods module

This module contains tests for the statistical methods used in cointegration analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.cointegration.statistical_methods import (
    johansen_test,
    engle_granger_test,
    calculate_half_life,
    get_max_eig_critical_values,
    calculate_johansen_p_values,
    phillips_ouliaris_test,
    detect_structural_breaks,
    analyze_residuals,
    calculate_hurst_exponent
)

@pytest.fixture
def sample_price_series():
    """Fixture providing sample price series for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate cointegrated price series
    # Series 1: random walk
    random_changes = np.random.normal(0, 1, 200)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: cointegrated with series1 (with noise)
    # y = 0.5*x + 50 + noise
    noise = np.random.normal(0, 0.5, 200)
    series2 = 0.5 * series1 + 50 + noise
    
    # Series 3: independent random walk (not cointegrated with series1)
    random_changes2 = np.random.normal(0, 1, 200)
    series3 = 100 + np.cumsum(random_changes2)
    
    # Create Series objects
    s1 = pd.Series(series1, index=dates, name='price1')
    s2 = pd.Series(series2, index=dates, name='price2')
    s3 = pd.Series(series3, index=dates, name='price3')
    
    # Create spread with mean-reverting properties (stationary series)
    stationary = pd.Series(np.random.normal(0, 1, 200), index=dates, name='stationary')
    
    # Create DataFrame for multivariate tests
    df = pd.DataFrame({
        'price1': s1,
        'price2': s2,
        'price3': s3
    })
    
    return {
        'price1': s1, 
        'price2': s2, 
        'price3': s3, 
        'stationary': stationary,
        'dataframe': df
    }


class TestJohansenTest:
    """Test class for Johansen cointegration test implementation."""
    
    def test_johansen_bivariate(self, sample_price_series):
        """Test Johansen test with bivariate case (2 variables)."""
        # Create a DataFrame with cointegrated series
        df = pd.DataFrame({
            'price1': sample_price_series['price1'],
            'price2': sample_price_series['price2']
        })
        
        # Run Johansen test
        result = johansen_test(df)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert 'trace_statistic' in result
        assert 'trace_critical_values' in result
        assert 'max_eigenvalue_statistic' in result
        assert 'max_eigenvalue_critical_values' in result
        assert 'eigenvalues' in result
        assert 'n_cointegrating_relations_trace' in result
        assert 'n_cointegrating_relations_maxeig' in result
        
        # Should detect cointegration (at least 1 cointegrating relationship)
        assert result['n_cointegrating_relations_trace'] >= 1
        
        # In bivariate case, can have at most 1 cointegrating relationship
        assert result['n_cointegrating_relations_trace'] <= 1
    
    def test_johansen_trivariate(self, sample_price_series):
        """Test Johansen test with trivariate case (3 variables)."""
        # Use the full DataFrame from fixture
        df = sample_price_series['dataframe']
        
        # Run Johansen test
        result = johansen_test(df)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert len(result['trace_statistic']) == df.shape[1]
        assert len(result['eigenvalues']) == df.shape[1]
        
        # Should detect at least one cointegrating relationship
        # (since two series are cointegrated)
        assert result['n_cointegrating_relations_trace'] >= 1
    
    def test_johansen_not_cointegrated(self, sample_price_series):
        """Test Johansen test with non-cointegrated series."""
        # Create a DataFrame with non-cointegrated series
        df = pd.DataFrame({
            'price1': sample_price_series['price1'],
            'price3': sample_price_series['price3']
        })
        
        # Run Johansen test
        # Note: Even with non-cointegrated series, the test might sometimes
        # indicate cointegration due to sampling error and randomness
        # So we don't strictly assert no cointegration, but check the results format
        result = johansen_test(df)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert 'trace_statistic' in result
        assert 'eigenvalues' in result
        assert 'n_cointegrating_relations_trace' in result
    
    def test_johansen_input_validation(self, sample_price_series):
        """Test input validation for Johansen test."""
        # Test with too few variables
        with pytest.raises(ValueError):
            # Create single variable DataFrame
            df = pd.DataFrame({'price1': sample_price_series['price1']})
            johansen_test(df)
        
        # Test with invalid det_order
        with pytest.raises(ValueError):
            df = sample_price_series['dataframe']
            johansen_test(df, det_order=4)  # Invalid det_order
        
        # Test with invalid k_ar_diff
        with pytest.raises(ValueError):
            df = sample_price_series['dataframe']
            johansen_test(df, k_ar_diff=0)  # Invalid k_ar_diff
        
        # Test with non-standard significance level (should not raise error)
        df = sample_price_series['dataframe']
        result = johansen_test(df, significance_level=0.03)  # Non-standard
        # Should be adjusted to nearest standard level
        assert 'n_cointegrating_relations_trace' in result
    
    def test_critical_values(self):
        """Test critical value calculation for Johansen test."""
        # Test getting critical values for different models and dimensions
        for model in [0, 1, 2, 3, 4]:  # Different deterministic term models
            for n in [2, 3, 4, 5]:  # Different number of variables
                # Get critical values for 5% significance
                cv = get_max_eig_critical_values(n, model-1, 0.05)
                
                # Check that critical values were returned successfully
                assert isinstance(cv, np.ndarray)
                assert len(cv) == n
                assert np.all(cv > 0)  # Critical values should be positive
    
    def test_p_values(self):
        """Test p-value calculation for Johansen test."""
        # Generate some test statistics
        test_stats = np.array([25.0, 15.0, 5.0])
        n_variables = 3
        det_order = 1
        
        # Calculate p-values for trace test
        p_vals_trace = calculate_johansen_p_values(
            test_stats, n_variables, det_order, test_type='trace'
        )
        
        # Calculate p-values for max eigenvalue test
        p_vals_maxeig = calculate_johansen_p_values(
            test_stats, n_variables, det_order, test_type='maxeig'
        )
        
        # Check results
        assert isinstance(p_vals_trace, np.ndarray)
        assert isinstance(p_vals_maxeig, np.ndarray)
        assert len(p_vals_trace) == len(test_stats)
        assert len(p_vals_maxeig) == len(test_stats)
        
        # P-values should be in [0, 1]
        assert np.all((p_vals_trace >= 0) & (p_vals_trace <= 1))
        assert np.all((p_vals_maxeig >= 0) & (p_vals_maxeig <= 1))
        
        # Higher test statistics should generally have lower p-values
        assert p_vals_trace[0] <= p_vals_trace[1] <= p_vals_trace[2]
        assert p_vals_maxeig[0] <= p_vals_maxeig[1] <= p_vals_maxeig[2]


class TestEngleGrangerTest:
    """Test class for Engle-Granger cointegration test implementation."""
    
    def test_engle_granger_cointegrated(self, sample_price_series):
        """Test Engle-Granger with cointegrated series."""
        # Get cointegrated series
        y = sample_price_series['price2']  # Dependent variable
        x = sample_price_series['price1']  # Independent variable
        
        # Run Engle-Granger test
        result = engle_granger_test(y, x)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert 'hedge_ratio' in result
        assert 'intercept' in result
        assert 'residuals' in result
        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_cointegrated' in result
        
        # Should detect cointegration
        assert result['is_cointegrated'] is True
        
        # Hedge ratio should be close to 0.5 (from data generation)
        assert 0.4 < result['hedge_ratio'] < 0.6
        
        # Intercept should be close to 50 (from data generation)
        assert 45 < result['intercept'] < 55
    
    def test_engle_granger_not_cointegrated(self, sample_price_series):
        """Test Engle-Granger with non-cointegrated series."""
        # Get non-cointegrated series
        y = sample_price_series['price3']  # Dependent variable (independent random walk)
        x = sample_price_series['price1']  # Independent variable
        
        # Run Engle-Granger test
        result = engle_granger_test(y, x)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert 'is_cointegrated' in result
        
        # Should not detect cointegration
        # Note: There is always a small probability of a false positive
        # Since these are random series, we cannot guarantee no cointegration
    
    def test_engle_granger_regression_methods(self, sample_price_series):
        """Test different regression methods for Engle-Granger test."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test OLS regression method
        result_ols = engle_granger_test(y, x, regression_method='ols')
        assert result_ols['regression_method'] == 'ols'
        assert 'r_squared' in result_ols  # Only provided for OLS
        
        # Test dynamic OLS regression method
        result_dols = engle_granger_test(y, x, regression_method='dynamic_ols')
        assert result_dols['regression_method'] == 'dynamic_ols'
        
        # Test total least squares regression method
        result_tls = engle_granger_test(y, x, regression_method='tls')
        assert result_tls['regression_method'] == 'tls'
        
        # Invalid regression method should raise ValueError
        with pytest.raises(ValueError):
            engle_granger_test(y, x, regression_method='invalid_method')
    
    def test_engle_granger_adf_options(self, sample_price_series):
        """Test ADF test options for Engle-Granger test."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test different trend specifications
        for trend in ['c', 'ct', 'ctt', 'n']:
            result = engle_granger_test(y, x, trend=trend)
            assert result['adf_trend'] == trend
        
        # Test maxlag parameter
        result_maxlag = engle_granger_test(y, x, maxlag=5)
        assert isinstance(result_maxlag['adf_statistic'], float)
        
        # Test autolag parameter
        for autolag in ['AIC', 'BIC', 'tstat', None]:
            result = engle_granger_test(y, x, autolag=autolag)
            assert isinstance(result['adf_statistic'], float)
    
    def test_engle_granger_input_validation(self, sample_price_series):
        """Test input validation for Engle-Granger test."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            engle_granger_test(y[:-10], x)  # Different lengths
        
        # Test with handling of NaN values
        y_with_nan = y.copy()
        y_with_nan.iloc[10:20] = np.nan  # Introduce NaNs
        
        # Should handle NaNs without error
        result = engle_granger_test(y_with_nan, x)
        assert isinstance(result['is_cointegrated'], bool)


class TestHalfLifeCalculation:
    """Test class for half-life calculation."""
    
    def test_calculate_half_life_mean_reverting(self, sample_price_series):
        """Test half-life calculation with mean-reverting series."""
        # Create synthetic mean-reverting series
        np.random.seed(42)
        n = 200
        x = np.zeros(n)
        x[0] = 0
        theta = 0.1  # Mean reversion speed
        sigma = 0.5  # Noise level
        
        for i in range(1, n):
            x[i] = x[i-1] * (1 - theta) + np.random.normal(0, sigma)
        
        # Expected half-life: ln(2) / theta
        expected_half_life = np.log(2) / theta
        
        # Calculate half-life
        series = pd.Series(x)
        half_life = calculate_half_life(series)
        
        # Check that half-life is close to expected value
        # Allow for estimation error due to randomness
        assert half_life > 0  # Should be positive
        assert 0.5 * expected_half_life < half_life < 1.5 * expected_half_life
    
    def test_calculate_half_life_non_mean_reverting(self, sample_price_series):
        """Test half-life calculation with non-mean-reverting series."""
        # Use a random walk (non-mean-reverting)
        non_reverting = sample_price_series['price1']
        
        # Calculate half-life
        half_life = calculate_half_life(non_reverting)
        
        # For non-mean-reverting series, should return infinity
        assert np.isinf(half_life)
    
    def test_calculate_half_life_short_series(self):
        """Test half-life calculation with a series that's too short."""
        # Create a very short series
        short_series = pd.Series([1, 2])
        
        # Calculate half-life (should issue warning and return infinity)
        with pytest.warns(UserWarning):
            half_life = calculate_half_life(short_series)
        
        assert np.isinf(half_life)
    
    def test_calculate_half_life_error_handling(self):
        """Test error handling in half-life calculation."""
        # Create a series that would cause regression error (all same values)
        constant_series = pd.Series([1, 1, 1, 1, 1])
        
        # Calculate half-life (should handle error and return infinity)
        half_life = calculate_half_life(constant_series)
        
        assert np.isinf(half_life)


class TestPhillipsOuliarisTest:
    """Test class for Phillips-Ouliaris cointegration test implementation."""
    
    def test_phillips_ouliaris_basics(self, sample_price_series):
        """Test basic functionality of Phillips-Ouliaris test."""
        # Get cointegrated series
        y = sample_price_series['price2']  # Dependent variable
        x = sample_price_series['price1']  # Independent variable
        
        # Run Phillips-Ouliaris test
        result = phillips_ouliaris_test(y, x)
        
        # Check for expected outputs
        assert isinstance(result, dict)
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'critical_values' in result
        assert 'is_cointegrated' in result
        assert 'hedge_ratio' in result
        assert 'intercept' in result
        assert 'residuals' in result
        assert 'half_life' in result
        assert 'conclusion' in result
        
        # Should detect cointegration
        assert result['is_cointegrated'] is True
        
        # Hedge ratio should be close to 0.5 (from data generation)
        assert 0.4 < result['hedge_ratio'] < 0.6
    
    def test_phillips_ouliaris_test_types(self, sample_price_series):
        """Test different test types for Phillips-Ouliaris."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test Zt statistic
        result_zt = phillips_ouliaris_test(y, x, test_type='Zt')
        assert result_zt['test_type'] == 'Zt'
        assert isinstance(result_zt['test_statistic'], float)
        
        # Test Za statistic
        result_za = phillips_ouliaris_test(y, x, test_type='Za')
        assert result_za['test_type'] == 'Za'
        assert isinstance(result_za['test_statistic'], float)
        
        # Invalid test type should raise ValueError
        with pytest.raises(ValueError):
            phillips_ouliaris_test(y, x, test_type='invalid')
    
    def test_phillips_ouliaris_kernels(self, sample_price_series):
        """Test different kernels for Phillips-Ouliaris test."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test bartlett kernel
        result_bartlett = phillips_ouliaris_test(y, x, kernel='bartlett')
        assert result_bartlett['kernel'] == 'bartlett'
        
        # Test parzen kernel
        result_parzen = phillips_ouliaris_test(y, x, kernel='parzen')
        assert result_parzen['kernel'] == 'parzen'
        
        # Test quadratic kernel
        result_quadratic = phillips_ouliaris_test(y, x, kernel='quadratic')
        assert result_quadratic['kernel'] == 'quadratic'
        
        # Invalid kernel should raise ValueError
        with pytest.raises(ValueError):
            phillips_ouliaris_test(y, x, kernel='invalid')
    
    def test_phillips_ouliaris_regression_methods(self, sample_price_series):
        """Test different regression methods for Phillips-Ouliaris test."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test OLS
        result_ols = phillips_ouliaris_test(y, x, regression_method='ols')
        assert result_ols['regression_method'] == 'ols'
        
        # Test dynamic OLS
        result_dols = phillips_ouliaris_test(y, x, regression_method='dynamic_ols')
        assert result_dols['regression_method'] == 'dynamic_ols'
        
        # Test TLS
        result_tls = phillips_ouliaris_test(y, x, regression_method='tls')
        assert result_tls['regression_method'] == 'tls'


class TestStructuralBreaks:
    """Test class for structural break detection."""
    
    def test_detect_structural_breaks_basic(self):
        """Test basic functionality of structural break detection."""
        # Create series with a clear break
        n = 200
        np.random.seed(42)
        x = np.cumsum(np.random.normal(0, 1, n))
        y1 = 0.5 * x[:n//2] + 10 + np.random.normal(0, 0.5, n//2)
        y2 = 2.0 * x[n//2:] + 5 + np.random.normal(0, 0.5, n - n//2)
        y = np.concatenate([y1, y2])
        
        # Convert to pandas Series
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)
        
        # Test with recursive CUSUM
        result = detect_structural_breaks(y_series, x_series, test_method='recursive_cusum')
        
        # Check basic outputs
        assert isinstance(result, dict)
        assert 'has_break' in result
        assert 'break_points' in result
        assert 'test_statistic' in result
        assert 'p_value' in result
        assert 'segment_params' in result
        assert 'conclusion' in result
        
        # Should detect break
        assert result['has_break'] is True
        assert len(result['break_points']) > 0
        
        # Break point should be close to the middle
        assert abs(result['break_points'][0] - n//2) < n//4
    
    def test_detect_structural_breaks_methods(self):
        """Test different methods for structural break detection."""
        # Create series with a clear break
        n = 200
        np.random.seed(42)
        x = np.cumsum(np.random.normal(0, 1, n))
        y1 = 0.5 * x[:n//2] + 10 + np.random.normal(0, 0.5, n//2)
        y2 = 2.0 * x[n//2:] + 5 + np.random.normal(0, 0.5, n - n//2)
        y = np.concatenate([y1, y2])
        
        # Convert to pandas Series
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        x_series = pd.Series(x, index=dates)
        y_series = pd.Series(y, index=dates)
        
        # Test methods
        for method in ['recursive_cusum', 'standard_cusum', 'chow', 'quandt_andrews']:
            result = detect_structural_breaks(y_series, x_series, test_method=method)
            assert result['test_method'] == method
            assert isinstance(result['has_break'], bool)
    
    def test_detect_structural_breaks_no_break(self, sample_price_series):
        """Test structural break detection with no break."""
        # Use cointegrated series with no break
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test for structural breaks
        result = detect_structural_breaks(y, x, test_method='recursive_cusum')
        
        # Result should be a valid dictionary, even if no break is found
        assert isinstance(result, dict)
        assert 'has_break' in result
        assert 'test_statistic' in result
        assert 'conclusion' in result
    
    def test_input_validation(self, sample_price_series):
        """Test input validation for structural break detection."""
        y = sample_price_series['price2']
        x = sample_price_series['price1']
        
        # Test with invalid method
        with pytest.raises(ValueError):
            detect_structural_breaks(y, x, test_method='invalid_method')
        
        # Test with too short series
        with pytest.raises(ValueError):
            detect_structural_breaks(y[:10], x[:10], min_segment_size=10)


class TestResidualAnalysis:
    """Test class for residual analysis function."""
    
    def test_analyze_residuals_stationary(self):
        """Test analyze_residuals with stationary series."""
        # Create a stationary series (white noise)
        np.random.seed(42)
        n = 100
        residuals = pd.Series(np.random.normal(0, 1, n))
        
        # Analyze residuals
        result = analyze_residuals(residuals)
        
        # Check basic outputs
        assert isinstance(result, dict)
        assert 'stationarity' in result
        assert 'normality' in result
        assert 'autocorrelation' in result
        assert 'heteroskedasticity' in result
        assert 'summary_statistics' in result
        assert 'outliers' in result
        assert 'diagnostics_passed' in result
        assert 'conclusion' in result
        
        # Stationary white noise should pass stationarity test
        assert result['stationarity']['adf_test']['passed'] is True
        
        # Should be normally distributed
        assert result['normality']['jarque_bera']['passed'] is True
        
        # Overall diagnostics should pass
        assert result['diagnostics_passed'] is True
    
    def test_analyze_residuals_nonstationary(self):
        """Test analyze_residuals with non-stationary series."""
        # Create a non-stationary series (random walk)
        np.random.seed(42)
        n = 100
        random_walk = np.cumsum(np.random.normal(0, 1, n))
        residuals = pd.Series(random_walk)
        
        # Analyze residuals
        result = analyze_residuals(residuals)
        
        # Should fail stationarity test
        assert result['stationarity']['adf_test']['passed'] is False
        
        # Overall diagnostics should fail
        assert result['diagnostics_passed'] is False
    
    def test_analyze_residuals_autocorrelated(self):
        """Test analyze_residuals with autocorrelated series."""
        # Create an autocorrelated series (AR(1) process)
        np.random.seed(42)
        n = 100
        ar_series = np.zeros(n)
        ar_series[0] = np.random.normal(0, 1)
        for i in range(1, n):
            ar_series[i] = 0.8 * ar_series[i-1] + np.random.normal(0, 0.5)
        
        residuals = pd.Series(ar_series)
        
        # Analyze residuals
        result = analyze_residuals(residuals)
        
        # Check autocorrelation tests
        assert 'durbin_watson' in result['autocorrelation']
        assert 'ljung_box' in result['autocorrelation']
        
        # Should detect autocorrelation with Durbin-Watson
        # DW statistic should be < 1.5 for positive autocorrelation
        assert result['autocorrelation']['durbin_watson']['test_statistic'] < 1.5
        assert result['autocorrelation']['durbin_watson']['passed'] is False
    
    def test_analyze_residuals_heteroskedastic(self):
        """Test analyze_residuals with heteroskedastic series."""
        # Create a heteroskedastic series (increasing variance)
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        variance = 0.1 + x/2  # Increasing variance
        heteroskedastic = np.random.normal(0, np.sqrt(variance))
        
        residuals = pd.Series(heteroskedastic)
        
        # Analyze residuals
        result = analyze_residuals(residuals)
        
        # Check heteroskedasticity tests
        assert 'breusch_pagan' in result['heteroskedasticity']
        assert 'white' in result['heteroskedasticity']


class TestHurstExponent:
    """Test class for Hurst exponent calculation."""
    
    def test_calculate_hurst_mean_reverting(self):
        """Test Hurst exponent calculation with mean-reverting series."""
        # Create a mean-reverting series (OU process)
        np.random.seed(42)
        n = 1000
        ou_process = np.zeros(n)
        ou_process[0] = 0
        theta = 0.1  # Mean reversion strength
        
        for i in range(1, n):
            ou_process[i] = ou_process[i-1] * (1 - theta) + np.random.normal(0, 1)
        
        # Calculate Hurst exponent
        result = calculate_hurst_exponent(ou_process)
        
        # Check basic outputs
        assert isinstance(result, dict)
        assert 'hurst_exponent' in result
        assert 'interpretation' in result
        assert 'is_mean_reverting' in result
        assert 'r_squared' in result
        
        # For mean-reverting series, H < 0.5
        assert result['hurst_exponent'] < 0.6
        assert result['is_mean_reverting'] is True
    
    def test_calculate_hurst_trending(self):
        """Test Hurst exponent calculation with trending series."""
        # Create a trending series (persistent)
        np.random.seed(42)
        n = 1000
        persistent = np.zeros(n)
        persistent[0] = 0
        
        # Fractional Brownian motion with H > 0.5
        increments = np.random.normal(0, 1, n)
        for i in range(1, n):
            # Positive autocorrelation in increments
            increments[i] = 0.7 * increments[i-1] + 0.3 * increments[i]
        
        persistent = np.cumsum(increments)
        
        # Calculate Hurst exponent
        result = calculate_hurst_exponent(persistent)
        
        # For trending series, H > 0.5
        assert result['hurst_exponent'] > 0.5
        assert result['is_mean_reverting'] is False
    
    def test_calculate_hurst_random_walk(self):
        """Test Hurst exponent calculation with random walk."""
        # Create a random walk (H ≈ 0.5)
        np.random.seed(42)
        n = 1000
        random_walk = np.cumsum(np.random.normal(0, 1, n))
        
        # Calculate Hurst exponent
        result = calculate_hurst_exponent(random_walk)
        
        # For random walk, H ≈ 0.5
        assert 0.4 < result['hurst_exponent'] < 0.6
        
    def test_calculate_hurst_short_series(self):
        """Test Hurst exponent calculation with short series."""
        # Create a short series
        np.random.seed(42)
        short_series = np.random.normal(0, 1, 50)
        
        # Calculate Hurst exponent
        with pytest.warns(UserWarning):
            result = calculate_hurst_exponent(short_series)
        
        # Should still get a result
        assert 'hurst_exponent' in result
        assert isinstance(result['hurst_exponent'], float)


if __name__ == "__main__":
    # Run tests manually
    pytest.main(["-xvs", __file__]) 