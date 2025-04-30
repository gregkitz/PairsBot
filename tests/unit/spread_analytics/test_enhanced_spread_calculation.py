"""
Unit tests for enhanced spread calculation methods.

This module tests the advanced spread calculation, normalization techniques,
and integration with adaptive methods like Kalman filter.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from src.spread_analytics.spread_analyzer import SpreadAnalyzer

@pytest.fixture
def regime_changing_data():
    """Fixture providing sample price data with regime changes for testing."""
    # Create synthetic price series with regime changes
    np.random.seed(42)  # For reproducibility
    
    # Generate 400 days of price data
    dates = pd.date_range(start='2022-01-01', periods=400, freq='D')
    
    # Create a price series with different regimes
    # First 200 days: hedge ratio of 0.5
    # Next 200 days: hedge ratio of 0.8
    random_changes = np.random.normal(0, 1, 400)
    series1 = 100 + np.cumsum(random_changes)
    
    # Series 2: regime-dependent relationship with series1
    noise = np.random.normal(0, 0.5, 400)
    series2 = np.zeros(400)
    
    # First regime
    series2[:200] = 0.5 * series1[:200] + 50 + noise[:200]
    
    # Second regime
    series2[200:] = 0.8 * series1[200:] + 30 + noise[200:]
    
    # Create Series objects
    price1 = pd.Series(series1, index=dates)
    price2 = pd.Series(series2, index=dates)
    
    return {
        'price1': price1,
        'price2': price2,
        'regime_change_idx': 200,
        'true_hedge_ratios': [0.5, 0.8]
    }

@pytest.fixture
def seasonal_data():
    """Fixture providing sample price data with seasonal patterns."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 365 days of price data (one year)
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    
    # Base random walk for series1
    random_changes = np.random.normal(0, 1, 365)
    series1_base = 100 + np.cumsum(random_changes)
    
    # Add seasonal component (higher in summer, lower in winter)
    seasonal_component = 10 * np.sin(np.linspace(0, 2*np.pi, 365))
    series1 = series1_base + seasonal_component
    
    # Series 2: cointegrated with series1 but with different seasonal sensitivity
    noise = np.random.normal(0, 0.5, 365)
    seasonal_component2 = 5 * np.sin(np.linspace(0, 2*np.pi, 365))
    series2 = 0.6 * series1_base + 40 + seasonal_component2 + noise
    
    # Create Series objects
    price1 = pd.Series(series1, index=dates)
    price2 = pd.Series(series2, index=dates)
    
    return {
        'price1': price1,
        'price2': price2
    }

@pytest.fixture
def volatility_changing_data():
    """Fixture providing sample price data with changing volatility."""
    np.random.seed(42)  # For reproducibility
    
    # Generate 300 days of price data
    dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
    
    # Create series1 with changing volatility
    series1 = np.zeros(300)
    
    # Low volatility regime
    low_vol_changes = np.random.normal(0, 0.5, 100)
    series1[:100] = 100 + np.cumsum(low_vol_changes)
    
    # High volatility regime
    high_vol_changes = np.random.normal(0, 2.0, 100)
    series1[100:200] = series1[99] + np.cumsum(high_vol_changes)
    
    # Medium volatility regime
    med_vol_changes = np.random.normal(0, 1.0, 100)
    series1[200:] = series1[199] + np.cumsum(med_vol_changes)
    
    # Series 2: cointegrated with series1
    noise = np.random.normal(0, 0.5, 300)
    series2 = 0.7 * series1 + 45 + noise
    
    # Create Series objects
    price1 = pd.Series(series1, index=dates)
    price2 = pd.Series(series2, index=dates)
    
    return {
        'price1': price1,
        'price2': price2,
        'volatility_regimes': [
            (0, 100, 'low'),
            (100, 200, 'high'),
            (200, 300, 'medium')
        ]
    }


class TestEnhancedSpreadCalculation:
    """Test enhanced spread calculation methods."""
    
    def test_volatility_adjusted_spread(self, volatility_changing_data):
        """Test volatility-adjusted spread calculation."""
        price1 = volatility_changing_data['price1']
        price2 = volatility_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Calculate standard spread
        standard_spread = analyzer.calculate_spread(price2, price1)
        
        # Calculate volatility-adjusted spread
        vol_adjusted_spread = analyzer.calculate_volatility_adjusted_spread(
            price2, price1, vol_window=20, vol_method='rolling'
        )
        
        # Verify the volatility-adjusted spread is normalized compared to standard spread
        std_std = standard_spread.std()
        vol_adj_std = vol_adjusted_spread.std()
        
        # The volatility-adjusted spread should have more uniform volatility across regimes
        for start_idx, end_idx, regime in volatility_changing_data['volatility_regimes']:
            std_regime_vol = standard_spread.iloc[start_idx:end_idx].std()
            vol_adj_regime_vol = vol_adjusted_spread.iloc[start_idx:end_idx].std()
            
            # Calculate ratio of regime volatility to overall volatility
            std_ratio = std_regime_vol / std_std
            vol_adj_ratio = vol_adj_regime_vol / vol_adj_std
            
            # For volatility-adjusted spread, the ratio should be closer to 1.0 
            # (more uniform across regimes)
            assert abs(vol_adj_ratio - 1.0) <= abs(std_ratio - 1.0)
    
    def test_dynamic_hedge_ratio(self, regime_changing_data):
        """Test dynamic hedge ratio calculation with Kalman filter."""
        price1 = regime_changing_data['price1']
        price2 = regime_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Calculate spread with fixed hedge ratio (using all data)
        # This would be suboptimal as it averages across regimes
        fixed_hedge_ratio = price2.corr(price1)
        fixed_spread = analyzer.calculate_spread(price2, price1, hedge_ratio=fixed_hedge_ratio)
        
        # Calculate spread with Kalman filter dynamic hedge ratio
        kalman_results = analyzer.kalman_filter_hedge_ratio(price2, price1)
        dynamic_hedge_ratio = kalman_results['hedge_ratio']
        dynamic_spread = kalman_results['spread']
        
        # Verify that Kalman filter adapts to regime change
        regime_change_idx = regime_changing_data['regime_change_idx']
        
        # Calculate average hedge ratio before and after regime change
        pre_change_hr = dynamic_hedge_ratio.iloc[:regime_change_idx].mean()
        post_change_hr = dynamic_hedge_ratio.iloc[regime_change_idx:].mean()
        
        # Verify the hedge ratios are close to the true values
        assert abs(pre_change_hr - regime_changing_data['true_hedge_ratios'][0]) < 0.1
        assert abs(post_change_hr - regime_changing_data['true_hedge_ratios'][1]) < 0.1
        
        # The dynamic spread should be more stationary than the fixed spread
        from statsmodels.tsa.stattools import adfuller
        
        # ADF test on both spreads
        fixed_adf = adfuller(fixed_spread.dropna())
        dynamic_adf = adfuller(dynamic_spread.dropna())
        
        # More negative ADF statistic indicates more stationarity
        assert dynamic_adf[0] < fixed_adf[0]
    
    def test_multitimeframe_spread(self, seasonal_data):
        """Test multi-timeframe spread calculation."""
        price1 = seasonal_data['price1']
        price2 = seasonal_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Calculate multi-timeframe spread
        timeframes = [20, 60, 120]
        multi_tf_results = analyzer.calculate_multitimeframe_spread(
            price2, price1, timeframes=timeframes
        )
        
        # Verify results structure
        assert isinstance(multi_tf_results, dict)
        for tf in timeframes:
            assert f'zscore_{tf}' in multi_tf_results
            assert f'spread_{tf}' in multi_tf_results
        
        # Verify signal consistency calculation
        consistency = analyzer.calculate_signal_consistency(multi_tf_results)
        
        # Consistency should be in the range [0, 1]
        assert 0 <= consistency <= 1
        
        # Verify that different timeframes capture different aspects of the data
        # Short timeframe should be more volatile than long timeframe
        short_zscore = multi_tf_results[f'zscore_{timeframes[0]}']
        long_zscore = multi_tf_results[f'zscore_{timeframes[-1]}']
        
        assert short_zscore.std() > long_zscore.std()
    
    def test_normalization_techniques(self, volatility_changing_data):
        """Test different spread normalization techniques."""
        price1 = volatility_changing_data['price1']
        price2 = volatility_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Test different normalization techniques
        normalization_methods = ['zscore', 'percent', 'absolute']
        
        for method in normalization_methods:
            # Calculate normalized spread
            normalized_spread = analyzer.calculate_normalized_spread(
                price2, price1, normalization=method, window=20
            )
            
            # Verify the spread is normalized according to the method
            assert not normalized_spread.isnull().any()
            
            if method == 'zscore':
                # Z-score normalized spread should have approximate mean 0 and std 1
                assert abs(normalized_spread.mean()) < 0.3
                assert 0.7 < normalized_spread.std() < 1.3
            
            elif method == 'percent':
                # Percent normalized spread should be in a reasonable range
                assert normalized_spread.min() > -100
            
            elif method == 'absolute':
                # Absolute normalized spread should be in a reasonable range
                # specific to the data, harder to make general assertions
                assert normalized_spread.dtype == np.float64
    
    def test_dynamic_thresholds(self, volatility_changing_data):
        """Test dynamic threshold calculation for spread signals."""
        price1 = volatility_changing_data['price1']
        price2 = volatility_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Calculate spread
        spread = analyzer.calculate_spread(price2, price1)
        
        # Calculate z-score with rolling window
        zscore = analyzer.calculate_zscore(spread, window=20, method='rolling')
        
        # Calculate dynamic thresholds
        dynamic_thresholds = analyzer.dynamic_threshold_calculation(
            spread, window=60, target_percentile=0.05
        )
        
        # Calculate regime-based thresholds
        regime_thresholds = analyzer.regime_based_thresholds(
            spread, vol_window=30
        )
        
        # Verify the structure of the results
        assert isinstance(dynamic_thresholds, dict)
        assert 'upper_threshold' in dynamic_thresholds
        assert 'lower_threshold' in dynamic_thresholds
        
        assert isinstance(regime_thresholds, dict)
        assert 'upper_threshold' in regime_thresholds
        assert 'lower_threshold' in regime_thresholds
        
        # Verify the thresholds adapt to volatility changes
        # In high volatility regimes, thresholds should be wider
        vol_regimes = volatility_changing_data['volatility_regimes']
        
        # Calculate average threshold width for each regime
        for start_idx, end_idx, regime in vol_regimes:
            upper = dynamic_thresholds['upper_threshold'].iloc[start_idx:end_idx]
            lower = dynamic_thresholds['lower_threshold'].iloc[start_idx:end_idx]
            avg_width = (upper - lower).mean()
            
            # Verify that high volatility has wider thresholds than low volatility
            if regime == 'high':
                high_vol_width = avg_width
            elif regime == 'low':
                low_vol_width = avg_width
        
        # High volatility regime should have wider thresholds
        assert high_vol_width > low_vol_width
    
    def test_garch_volatility_calculation(self, volatility_changing_data):
        """Test GARCH volatility calculation for spread modeling."""
        price1 = volatility_changing_data['price1']
        price2 = volatility_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        # Calculate spread
        spread = analyzer.calculate_spread(price2, price1)
        
        # Calculate GARCH volatility
        garch_vol = analyzer.calculate_garch_volatility(spread, window=30)
        
        # Calculate GARCH z-score
        garch_zscore = analyzer.calculate_garch_zscore(spread, window=30)
        
        # Verify the results
        assert isinstance(garch_vol, pd.Series)
        assert isinstance(garch_zscore, pd.Series)
        
        # GARCH volatility should adapt to the volatility regimes
        vol_regimes = volatility_changing_data['volatility_regimes']
        
        # Calculate average GARCH volatility for each regime
        for start_idx, end_idx, regime in vol_regimes:
            avg_vol = garch_vol.iloc[start_idx:end_idx].mean()
            
            # Save the average volatility for comparison
            if regime == 'high':
                high_vol = avg_vol
            elif regime == 'low':
                low_vol = avg_vol
        
        # High volatility regime should have higher GARCH volatility
        assert high_vol > low_vol
    
    def test_performance_benchmark(self, regime_changing_data):
        """Benchmark the performance of different spread calculation methods."""
        price1 = regime_changing_data['price1']
        price2 = regime_changing_data['price2']
        
        # Create spread analyzer
        analyzer = SpreadAnalyzer()
        
        methods = {
            'standard': lambda: analyzer.calculate_spread(price2, price1),
            'kalman': lambda: analyzer.kalman_filter_hedge_ratio(price2, price1)['spread'],
            'rolling_hr': lambda: analyzer.calculate_spread(
                price2, price1, hedge_ratio=analyzer.rolling_hedge_ratio(price2, price1, window=20)
            ),
            'vol_adjusted': lambda: analyzer.calculate_volatility_adjusted_spread(
                price2, price1, vol_window=20
            ),
            'multitimeframe': lambda: analyzer.calculate_multitimeframe_spread(
                price2, price1, timeframes=[20, 60, 120]
            )['spread_60'],
        }
        
        # Measure execution time for each method
        timing_results = {}
        
        for name, method in methods.items():
            start_time = time.time()
            _ = method()
            end_time = time.time()
            timing_results[name] = end_time - start_time
        
        # Output timing results
        print("Spread calculation method timing results:")
        for name, duration in timing_results.items():
            print(f"{name}: {duration:.4f} seconds")
        
        # Verify that standard method is faster than advanced methods
        assert timing_results['standard'] < timing_results['kalman']
        
        # Test stationarity of different spread calculation methods
        from statsmodels.tsa.stattools import adfuller
        
        stationarity_results = {}
        for name, method in methods.items():
            if name != 'multitimeframe':  # Skip multitimeframe for simplicity
                spread = method()
                adf_result = adfuller(spread.dropna())
                stationarity_results[name] = adf_result[0]  # ADF statistic
        
        # Output stationarity results
        print("\nSpread calculation method stationarity results (ADF statistic):")
        for name, adf_stat in stationarity_results.items():
            print(f"{name}: {adf_stat:.4f}")
        
        # Verify that Kalman filter produces more stationary spread
        # (more negative ADF statistic indicates more stationarity)
        assert stationarity_results['kalman'] < stationarity_results['standard']


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 