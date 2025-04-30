"""
Unit tests for Kalman filter module

This module contains tests for the Kalman filter implementations used for time-varying
cointegration analysis.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import tempfile
import os

from src.cointegration.kalman_filter import (
    LinearKalmanFilter,
    ExtendedKalmanFilter,
    estimate_timevarying_hedge_ratio,
    estimate_nonlinear_timevarying_hedge_ratio,
    compare_kalman_models,
    calculate_spread_metrics,
    optimize_kalman_parameters,
    plot_timevarying_hedge_ratio,
    plot_nonlinear_hedge_ratio,
    plot_model_comparison
)


@pytest.fixture
def synthetic_cointegrated_data():
    """Fixture providing synthetic time-varying cointegrated data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate price1 as a random walk
    random_changes = np.random.normal(0, 1, 200)
    price1_values = 100 + np.cumsum(random_changes)
    
    # Create a time-varying hedge ratio
    time_factor = np.linspace(0, 1, 200)
    hedge_ratio = 0.5 + 0.3 * np.sin(time_factor * 5)
    
    # Generate price2 with time-varying relationship to price1
    intercept = 50
    noise = np.random.normal(0, 2, 200)
    price2_values = intercept + hedge_ratio * price1_values + noise
    
    # Create Series objects
    price1 = pd.Series(price1_values, index=dates, name='price1')
    price2 = pd.Series(price2_values, index=dates, name='price2')
    true_hedge_ratio = pd.Series(hedge_ratio, index=dates, name='hedge_ratio')
    
    return {
        'price1': price1,
        'price2': price2,
        'true_hedge_ratio': true_hedge_ratio,
        'intercept': intercept
    }


@pytest.fixture
def threshold_cointegrated_data():
    """Fixture providing synthetic data with threshold cointegration relationship."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 days of price data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Generate price1 as a random walk
    random_changes = np.random.normal(0, 1, 200)
    price1_values = 100 + np.cumsum(random_changes)
    
    # Set threshold
    threshold = 100
    
    # Create regime-dependent hedge ratio
    beta_low = 0.3
    beta_high = 0.7
    hedge_ratio = np.where(price1_values < threshold, beta_low, beta_high)
    
    # Generate price2 with threshold relationship to price1
    intercept = 50
    noise = np.random.normal(0, 2, 200)
    price2_values = intercept + hedge_ratio * price1_values + noise
    
    # Create Series objects
    price1 = pd.Series(price1_values, index=dates, name='price1')
    price2 = pd.Series(price2_values, index=dates, name='price2')
    true_hedge_ratio = pd.Series(hedge_ratio, index=dates, name='hedge_ratio')
    
    return {
        'price1': price1,
        'price2': price2,
        'true_hedge_ratio': true_hedge_ratio,
        'intercept': intercept,
        'threshold': threshold,
        'beta_low': beta_low,
        'beta_high': beta_high
    }


@pytest.fixture
def edge_case_data():
    """Fixture providing synthetic data with edge cases like outliers and NaN values."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 days of price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate base price1 as a random walk
    random_changes = np.random.normal(0, 1, 100)
    price1_values = 100 + np.cumsum(random_changes)
    
    # Generate base price2 with relationship to price1
    hedge_ratio = 0.7
    intercept = 50
    noise = np.random.normal(0, 2, 100)
    price2_values = intercept + hedge_ratio * price1_values + noise
    
    # Add outliers
    price1_values[25] = price1_values[25] * 1.5  # Add outlier in price1
    price2_values[75] = price2_values[75] * 0.7  # Add outlier in price2
    
    # Add some NaN values
    price1_values[40] = np.nan
    price2_values[60] = np.nan
    
    # Create Series objects
    price1 = pd.Series(price1_values, index=dates, name='price1')
    price2 = pd.Series(price2_values, index=dates, name='price2')
    
    return {
        'price1': price1,
        'price2': price2,
        'hedge_ratio': hedge_ratio,
        'intercept': intercept,
        'outlier_indices': [25, 75],
        'nan_indices': [40, 60]
    }


class TestLinearKalmanFilter:
    """Test class for LinearKalmanFilter implementation."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        kf = LinearKalmanFilter()
        assert kf.transition_covariance == 1e-4
        assert kf.observation_covariance == 1e-2
        assert kf.initial_state_mean is None
        assert kf.initial_state_covariance is None
        assert kf.adapt_observation_noise is False
        assert kf.em_iterations == 0
        assert kf.is_fitted is False
    
    def test_fit_predict_simple_data(self):
        """Test fitting and prediction with simple synthetic data."""
        # Create simple data with constant coefficients
        n = 100
        X = np.column_stack([np.ones(n), np.arange(n)])
        beta = np.array([2.0, 0.5])
        y = np.dot(X, beta) + np.random.normal(0, 0.1, n).reshape(-1, 1)
        
        # Fit Kalman filter
        kf = LinearKalmanFilter(
            transition_covariance=1e-6,  # Small noise for stable coefficients
            observation_covariance=1e-2
        )
        kf.fit(y, X)
        
        # Check if fitted
        assert kf.is_fitted
        assert kf.states is not None
        assert kf.state_covariances is not None
        assert kf.log_likelihood is not None
        
        # Check dimensions
        assert kf.states.shape == (n, 2)
        assert kf.state_covariances.shape == (n, 2, 2)
        
        # Check final state estimates
        final_state = kf.states[-1]
        assert np.allclose(final_state, beta, rtol=0.1, atol=0.1)
        
        # Test prediction
        predictions = kf.predict(X)
        assert predictions.shape == (n, 1)
        assert np.mean(np.abs(predictions - y)) < 0.2  # Mean absolute error should be small
    
    def test_estimate_timevarying_hedge_ratio(self, synthetic_cointegrated_data):
        """Test the estimate_timevarying_hedge_ratio function."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        true_hedge_ratio = synthetic_cointegrated_data['true_hedge_ratio']
        
        # Estimate time-varying hedge ratio
        results = estimate_timevarying_hedge_ratio(price1, price2)
        
        # Check result format
        assert isinstance(results, pd.DataFrame)
        assert 'hedge_ratio' in results.columns
        assert 'intercept' in results.columns
        assert 'spread' in results.columns
        assert len(results) == len(price1)
        
        # Verify that estimated hedge ratio approximates the true hedge ratio
        estimated_hr = results['hedge_ratio']
        correlation = np.corrcoef(estimated_hr, true_hedge_ratio)[0, 1]
        assert correlation > 0.7  # Strong correlation with true hedge ratio
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((estimated_hr - true_hedge_ratio) ** 2))
        assert rmse < 0.2  # RMSE should be reasonably small
        
        # Check spread characteristics
        spread = results['spread']
        assert np.abs(spread.mean()) < 3.0  # Mean should be close to zero
        assert np.std(spread) < 5.0  # Std should be reasonable

    def test_robustness_to_outliers(self, edge_case_data):
        """Test Kalman filter's robustness to outliers."""
        price1 = edge_case_data['price1']
        price2 = edge_case_data['price2']
        outlier_indices = edge_case_data['outlier_indices']
        true_hedge_ratio = edge_case_data['hedge_ratio']
        
        # Fill NaN values to focus on outlier testing
        price1 = price1.fillna(method='ffill')
        price2 = price2.fillna(method='ffill')
        
        # Estimate with standard settings
        results = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=1e-4,
            observation_covariance=1e-2
        )
        
        # Estimate with robust settings (higher observation_covariance)
        results_robust = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=1e-4,
            observation_covariance=1.0  # Higher value to be more robust to outliers
        )
        
        # Check that hedge ratio doesn't change dramatically at outlier points
        for idx in outlier_indices:
            # Calculate rate of change around outlier points
            if idx > 0 and idx < len(results) - 1:
                # Standard settings might have larger changes
                std_change = abs(results['hedge_ratio'][idx] - results['hedge_ratio'][idx-1])
                # Robust settings should have smaller changes
                robust_change = abs(results_robust['hedge_ratio'][idx] - results_robust['hedge_ratio'][idx-1])
                
                # Robust should be less reactive to outliers
                assert robust_change <= std_change * 1.2, f"Robust estimation should be less sensitive to outliers at index {idx}"
    
    def test_handling_of_nan_values(self, edge_case_data):
        """Test Kalman filter's handling of NaN values."""
        price1 = edge_case_data['price1']
        price2 = edge_case_data['price2']
        nan_indices = edge_case_data['nan_indices']
        
        # Fill NaN values to focus on outlier testing
        price1 = price1.fillna(method='ffill')
        price2 = price2.fillna(method='ffill')
        
        # Estimate with standard settings
        results = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=1e-4,
            observation_covariance=1e-2
        )
        
        # Check that hedge ratio doesn't change dramatically at outlier points
        for idx in nan_indices:
            # Calculate rate of change around outlier points
            if idx > 0 and idx < len(results) - 1:
                # Standard settings might have larger changes
                pass
                
        # Test with missing data
        price1_missing = price1.copy()
        price2_missing = price2.copy()
        
        # Add more NaN values
        price1_missing.iloc[10:15] = np.nan
        price2_missing.iloc[50:55] = np.nan
        
        # Estimate with missing data (should not crash)
        results_missing = estimate_timevarying_hedge_ratio(
            price1_missing, price2_missing,
            transition_covariance=1e-4,
            observation_covariance=0.1
        )
        
        # Should have values even at NaN positions (fill method)
        assert not results_missing['hedge_ratio'].isna().any()
        assert not results_missing['intercept'].isna().any()
    
    def test_dimension_mismatches(self):
        """Test Kalman filter's robustness to dimension mismatches and edge cases."""
        # Create test data with different dimensions
        np.random.seed(42)
        
        # Short series (just enough data)
        dates_short = pd.date_range(start='2023-01-01', periods=10, freq='D')
        price1_short = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 10)), index=dates_short)
        price2_short = pd.Series(50 + 0.5 * price1_short + np.random.normal(0, 1, 10), index=dates_short)
        
        # Should work with minimal data
        results_short = estimate_timevarying_hedge_ratio(price1_short, price2_short)
        assert len(results_short) == len(price1_short)
        
        # Unequal length series
        dates_long = pd.date_range(start='2023-01-01', periods=15, freq='D')
        price1_long = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 15)), index=dates_long)
        
        # Should raise a ValueError for unequal lengths
        with pytest.raises(ValueError):
            estimate_timevarying_hedge_ratio(price1_long, price2_short)
        
        # Test with constant price series (should handle near-zero variance)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        price1_const = pd.Series(100.0, index=dates)
        price2_const = pd.Series(50.0, index=dates)
        
        # Should not raise an error but might have unstable estimates
        results_const = estimate_timevarying_hedge_ratio(
            price1_const, price2_const,
            observation_covariance=1.0  # Increase to handle the constant series
        )
        
        # At minimum, it shouldn't crash
        assert len(results_const) == len(price1_const)

    def test_optimization_edge_cases(self):
        """Test Kalman filter parameter optimization with edge cases."""
        np.random.seed(42)
        
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        price1 = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 200)), index=dates)
        price2 = pd.Series(50 + 0.7 * price1 + np.random.normal(0, 2, 200), index=dates)
        
        # Add structural break in the middle
        price2.iloc[100:] = 40 + 0.9 * price1.iloc[100:] + np.random.normal(0, 2, 100)
        
        # Try parameter optimization
        from src.cointegration.kalman_filter import optimize_kalman_parameters
        
        # Basic parameter optimization
        params = optimize_kalman_parameters(price1, price2, model_type='linear')
        
        # Should return a dictionary with optimized parameters
        assert isinstance(params, dict)
        assert 'transition_covariance' in params
        assert 'observation_covariance' in params
        
        # Parameters should be positive
        assert params['transition_covariance'] > 0
        assert params['observation_covariance'] > 0
        
        # Test with small sample
        small_price1 = price1.iloc[:20]
        small_price2 = price2.iloc[:20]
        
        # Should still work with small sample but might return different results
        small_params = optimize_kalman_parameters(small_price1, small_price2, model_type='linear')
        assert isinstance(small_params, dict)
        
        # Test with custom parameter grid
        custom_grid = {
            'transition_covariance': [1e-5, 1e-4, 1e-3],
            'observation_covariance': [1e-3, 1e-2, 1e-1]
        }
        
        custom_params = optimize_kalman_parameters(
            price1, price2, 
            model_type='linear',
            param_grid=custom_grid
        )
        
        # Should return one of the grid values
        assert custom_params['transition_covariance'] in custom_grid['transition_covariance']
        assert custom_params['observation_covariance'] in custom_grid['observation_covariance']


class TestExtendedKalmanFilter:
    """Test class for ExtendedKalmanFilter implementation."""
    
    def test_threshold_model(self, threshold_cointegrated_data):
        """Test threshold model performance with simulated data."""
        price1 = threshold_cointegrated_data['price1']
        price2 = threshold_cointegrated_data['price2']
        threshold = threshold_cointegrated_data['threshold']
        beta_low = threshold_cointegrated_data['beta_low']
        beta_high = threshold_cointegrated_data['beta_high']
        
        # Estimate parameters using nonlinear model
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type='threshold', threshold=threshold
        )
        
        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert 'beta_low' in results.columns
        assert 'beta_high' in results.columns
        assert 'effective_hedge_ratio' in results.columns
        assert 'spread' in results.columns
        
        # Check parameter recovery (approximate)
        assert abs(results['beta_low'].mean() - beta_low) < 0.15
        assert abs(results['beta_high'].mean() - beta_high) < 0.15
    
    def test_regime_switch_model(self, synthetic_cointegrated_data):
        """Test regime switching model with simulated data."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Estimate parameters using regime switching model
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type='regime_switch'
        )
        
        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert 'regime_probability' in results.columns
        assert 'hedge_ratio_regime1' in results.columns
        assert 'hedge_ratio_regime2' in results.columns
        assert 'effective_hedge_ratio' in results.columns
        
        # Basic checks for plausible behavior
        assert 0 <= results['regime_probability'].min() <= 1
        assert 0 <= results['regime_probability'].max() <= 1
    
    def test_log_price_model(self, synthetic_cointegrated_data):
        """Test log price model with simulated data."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Ensure prices are positive for log transform
        price1 = price1.apply(lambda x: max(x, 0.01))
        price2 = price2.apply(lambda x: max(x, 0.01))
        
        # Estimate parameters using log price model
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type='log_price'
        )
        
        # Check results structure
        assert isinstance(results, pd.DataFrame)
        assert 'hedge_ratio' in results.columns
        assert 'intercept' in results.columns
        assert 'spread' in results.columns
        
        # Check that spread exhibits mean-reverting properties
        from statsmodels.tsa.stattools import adfuller
        
        # Spread should be more stationary than raw price differences
        raw_diff = price2 - price1
        spread = results['spread']
        
        # ADF test on both series
        raw_adf = adfuller(raw_diff.dropna(), maxlag=1)
        spread_adf = adfuller(spread.dropna(), maxlag=1)
        
        # Spread should have more negative ADF statistic (more stationary)
        assert spread_adf[0] <= raw_adf[0]
    
    def test_edge_case_handling(self):
        """Test edge case handling in nonlinear Kalman filter models."""
        np.random.seed(42)
        
        # Create minimal dataset
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        price1 = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 30)), index=dates)
        price2 = pd.Series(50 + 0.7 * price1 + np.random.normal(0, 1, 30), index=dates)
        
        # Test with minimal data
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type='threshold'
        )
        assert len(results) == len(price1)
        
        # Test with extreme values
        extreme_price1 = price1 * 1000
        extreme_price2 = price2 * 1000
        
        # Should handle extreme values without numerical issues
        results_extreme = estimate_nonlinear_timevarying_hedge_ratio(
            extreme_price1, extreme_price2, model_type='threshold'
        )
        assert len(results_extreme) == len(extreme_price1)
        
        # Test with near-constant series
        const_price1 = pd.Series(100 + np.random.normal(0, 0.001, 30), index=dates)
        const_price2 = pd.Series(50 + 0.7 * const_price1 + np.random.normal(0, 0.001, 30), index=dates)
        
        # Should handle near-constant series without issues
        results_const = estimate_nonlinear_timevarying_hedge_ratio(
            const_price1, const_price2, model_type='threshold',
            observation_covariance=0.1  # Increase to handle near-constant series
        )
        assert len(results_const) == len(const_price1)


class TestModelComparison:
    """Test class for model comparison functionality."""
    
    def test_compare_kalman_models(self, synthetic_cointegrated_data):
        """Test comparison of different Kalman filter models."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Compare models
        comparison_results = compare_kalman_models(
            price1, price2,
            models=['linear', 'threshold', 'regime_switch']
        )
        
        # Check result structure
        assert isinstance(comparison_results, pd.DataFrame)
        assert len(comparison_results) >= 3  # At least the three models we specified
        
        # Check metrics are included
        assert 'log_likelihood' in comparison_results.columns
        assert 'aic' in comparison_results.columns
        assert 'bic' in comparison_results.columns
        assert 'rmse' in comparison_results.columns
        
    def test_model_selection_criteria(self, threshold_cointegrated_data, synthetic_cointegrated_data):
        """Test that model selection criteria choose appropriate models."""
        # For threshold data, the threshold model should perform better
        price1_threshold = threshold_cointegrated_data['price1']
        price2_threshold = threshold_cointegrated_data['price2']
        
        threshold_comparison = compare_kalman_models(
            price1_threshold, price2_threshold,
            models=['linear', 'threshold']
        )
        
        # For this simulated threshold data, threshold model should be better by BIC
        assert threshold_comparison.loc['threshold', 'bic'] <= threshold_comparison.loc['linear', 'bic']
        
        # For time-varying sinusoidal data, linear model should be competitive
        price1_tv = synthetic_cointegrated_data['price1']
        price2_tv = synthetic_cointegrated_data['price2']
        
        tv_comparison = compare_kalman_models(
            price1_tv, price2_tv,
            models=['linear', 'threshold', 'regime_switch']
        )
        
        # Linear model should be among the top performers (not necessarily the best)
        linear_bic = tv_comparison.loc['linear', 'bic']
        min_bic = tv_comparison['bic'].min()
        
        # Linear should be close to the best model (within 10%)
        assert linear_bic <= min_bic * 1.1
    
    def test_calculate_spread_metrics(self):
        """Test spread metric calculation for model evaluation."""
        # Create a synthetic spread with known properties
        np.random.seed(42)
        n = 1000
        ar_coef = 0.8  # Strong mean reversion
        
        # Create AR(1) spread
        spread = np.zeros(n)
        spread[0] = np.random.normal(0, 1)
        for i in range(1, n):
            spread[i] = ar_coef * spread[i-1] + np.random.normal(0, 1)
        
        # Calculate metrics
        metrics = calculate_spread_metrics(spread)
        
        # Check metrics existence
        assert 'mean' in metrics
        assert 'std' in metrics
        assert 'half_life' in metrics
        assert 'adf_stat' in metrics
        assert 'adf_pvalue' in metrics
        assert 'hurst' in metrics
        
        # Check specific properties
        assert abs(metrics['mean']) < 0.5  # Mean should be close to zero
        assert 0.9 < metrics['std'] < 1.5  # Standard deviation should be close to 1
        
        # Check half-life (should be close to -ln(2)/ln(ar_coef))
        expected_half_life = -np.log(2) / np.log(ar_coef)
        assert abs(metrics['half_life'] - expected_half_life) < 2.0
        
        # ADF p-value should be small (stationary series)
        assert metrics['adf_pvalue'] < 0.05
        
        # Hurst exponent should be < 0.5 for mean-reverting series
        assert metrics['hurst'] < 0.55
    
    def test_visualization_functions(self, synthetic_cointegrated_data):
        """Test that visualization functions run without errors."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Calculate results for different models
        linear_results = estimate_timevarying_hedge_ratio(price1, price2)
        threshold_results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type='threshold'
        )
        
        # Test plot_timevarying_hedge_ratio without crashing
        fig1 = plot_timevarying_hedge_ratio(linear_results, price1, price2)
        assert fig1 is not None
        
        # Test plot_nonlinear_hedge_ratio without crashing
        fig2 = plot_nonlinear_hedge_ratio(threshold_results, price1, price2)
        assert fig2 is not None
        
        # Test plot_model_comparison without crashing
        fig3 = plot_model_comparison(price1, price2)
        assert fig3 is not None


class TestParameterOptimization:
    """Test class for parameter optimization functions."""
    
    def test_optimize_kalman_parameters(self, synthetic_cointegrated_data):
        """Test parameter optimization for Kalman filter."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Optimize parameters
        params = optimize_kalman_parameters(price1, price2)
        
        # Check result format
        assert isinstance(params, dict)
        assert 'transition_covariance' in params
        assert 'observation_covariance' in params
        
        # Check parameter values are positive
        assert params['transition_covariance'] > 0
        assert params['observation_covariance'] > 0
        
        # Check performance with optimized parameters
        results = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=params['transition_covariance'],
            observation_covariance=params['observation_covariance']
        )
        
        # Calculate spread metrics
        metrics = calculate_spread_metrics(results['spread'].values)
        
        # Basic checks for reasonable values
        assert metrics['half_life'] > 0
        assert metrics['adf_pvalue'] < 0.05  # Should be stationary
    
    def test_optimization_with_custom_grid(self, synthetic_cointegrated_data):
        """Test parameter optimization with custom parameter grid."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Define custom parameter grid
        param_grid = {
            'transition_covariance': [1e-5, 1e-4, 1e-3],
            'observation_covariance': [1e-3, 1e-2, 1e-1]
        }
        
        # Optimize parameters with custom grid
        params = optimize_kalman_parameters(
            price1, price2,
            param_grid=param_grid
        )
        
        # Check result format
        assert isinstance(params, dict)
        
        # Parameters should be from the custom grid
        assert params['transition_covariance'] in param_grid['transition_covariance']
        assert params['observation_covariance'] in param_grid['observation_covariance']
    
    def test_optimization_for_different_models(self, synthetic_cointegrated_data, threshold_cointegrated_data):
        """Test parameter optimization for different Kalman filter models."""
        # For linear model
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        linear_params = optimize_kalman_parameters(
            price1, price2,
            model_type='linear'
        )
        
        # For threshold model
        price1_threshold = threshold_cointegrated_data['price1']
        price2_threshold = threshold_cointegrated_data['price2']
        
        threshold_params = optimize_kalman_parameters(
            price1_threshold, price2_threshold,
            model_type='threshold'
        )
        
        # Both should return valid parameters
        assert 'transition_covariance' in linear_params
        assert 'observation_covariance' in linear_params
        
        assert 'transition_covariance' in threshold_params
        assert 'observation_covariance' in threshold_params
        
        # Optimized parameters may differ between models
        # We don't assert equality/inequality as optimal values depend on the data
    
    def test_optimization_time_performance(self, synthetic_cointegrated_data):
        """Test that parameter optimization completes in a reasonable time."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Define a smaller grid for faster testing
        param_grid = {
            'transition_covariance': [1e-4, 1e-3],
            'observation_covariance': [1e-2, 1e-1]
        }
        
        # Measure optimization time
        start_time = time.time()
        params = optimize_kalman_parameters(
            price1, price2,
            param_grid=param_grid
        )
        elapsed_time = time.time() - start_time
        
        # Even with a very small grid, optimization should complete in reasonable time
        assert elapsed_time < 30  # Should complete in under 30 seconds
        
        # Smaller dataset should be faster
        small_price1 = price1.iloc[:50]
        small_price2 = price2.iloc[:50]
        
        start_time = time.time()
        small_params = optimize_kalman_parameters(
            small_price1, small_price2,
            param_grid=param_grid
        )
        small_elapsed_time = time.time() - start_time
        
        # Should be faster with smaller dataset
        assert small_elapsed_time < elapsed_time


class TestVisualizationFunctions:
    """Test class for visualization functions."""
    
    def test_plot_timevarying_hedge_ratio(self, synthetic_cointegrated_data):
        """Test plotting of time-varying hedge ratio."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Estimate time-varying hedge ratio
        results = estimate_timevarying_hedge_ratio(price1, price2)
        
        # Test basic plotting
        fig = plot_timevarying_hedge_ratio(results, price1, price2)
        assert isinstance(fig, plt.Figure)
        
        # Test with confidence intervals
        fig_with_ci = plot_timevarying_hedge_ratio(
            results, price1, price2,
            confidence_intervals=True,
            state_covariances=np.ones((len(results), 2, 2))  # Mock covariances
        )
        assert isinstance(fig_with_ci, plt.Figure)
        
        # Test with save_path (should not raise an error)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_plot.png')
            plot_timevarying_hedge_ratio(
                results, price1, price2,
                save_path=save_path
            )
            assert os.path.exists(save_path)
    
    def test_plot_nonlinear_hedge_ratio(self, threshold_cointegrated_data):
        """Test plotting of nonlinear hedge ratio."""
        price1 = threshold_cointegrated_data['price1']
        price2 = threshold_cointegrated_data['price2']
        
        # Test different model types
        model_types = ['threshold', 'regime_switch', 'log_price']
        
        for model_type in model_types:
            # Estimate with nonlinear model
            results = estimate_nonlinear_timevarying_hedge_ratio(
                price1, price2, model_type=model_type
            )
            
            # Plot results
            fig = plot_nonlinear_hedge_ratio(
                results, price1, price2, model_type=model_type
            )
            assert isinstance(fig, plt.Figure)
    
    def test_plot_model_comparison(self, synthetic_cointegrated_data):
        """Test plotting of model comparison."""
        price1 = synthetic_cointegrated_data['price1']
        price2 = synthetic_cointegrated_data['price2']
        
        # Test with different model combinations
        model_sets = [
            ['linear', 'threshold'],
            ['linear', 'regime_switch', 'log_price'],
            ['linear', 'threshold', 'regime_switch', 'log_price']
        ]
        
        for models in model_sets:
            fig = plot_model_comparison(
                price1, price2,
                models=models
            )
            assert isinstance(fig, plt.Figure)
            
        # Test with custom parameters
        fig = plot_model_comparison(
            price1, price2,
            models=['linear', 'threshold'],
            transition_covariance=1e-3,
            observation_covariance=1e-1
        )
        assert isinstance(fig, plt.Figure)
        
        # Test with save_path (should not raise an error)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'comparison_plot.png')
            plot_model_comparison(
                price1, price2,
                save_path=save_path
            )
            assert os.path.exists(save_path)
    
    def test_visualization_edge_cases(self):
        """Test visualization functions with edge cases."""
        # Create minimal dataset
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        price1 = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 10)), index=dates)
        price2 = pd.Series(50 + 0.7 * price1 + np.random.normal(0, 1, 10), index=dates)
        
        # Short series should still work with linear model
        results_short = estimate_timevarying_hedge_ratio(price1, price2)
        fig = plot_timevarying_hedge_ratio(results_short, price1, price2)
        assert isinstance(fig, plt.Figure)
        
        # Test with constant series
        const_dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        price1_const = pd.Series(100, index=const_dates)
        price2_const = pd.Series(50, index=const_dates)
        
        # Should still produce a plot without errors
        results_const = estimate_timevarying_hedge_ratio(
            price1_const, price2_const,
            observation_covariance=0.1  # Higher value for stability
        )
        fig_const = plot_timevarying_hedge_ratio(results_const, price1_const, price2_const)
        assert isinstance(fig_const, plt.Figure)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 