import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.technical_indicators import (
    rsi, macd, bollinger_bands, moving_averages, exponential_moving_averages,
    atr, z_score, stochastic_oscillator, rate_of_change, add_all_indicators
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    # Create a DataFrame with sample data
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 data points of a random walk with drift
    n = 100
    prices = np.zeros(n)
    prices[0] = 100
    for i in range(1, n):
        prices[i] = prices[i-1] + np.random.normal(0.1, 1)
    
    # Create a Series
    series = pd.Series(prices)
    series.index = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    # Create a DataFrame with high, low, close
    df = pd.DataFrame({
        'high': series * (1 + np.random.uniform(0, 0.02, n)),
        'low': series * (1 - np.random.uniform(0, 0.02, n)),
        'close': series,
        'spread': series
    })
    
    return df


def test_rsi(sample_data):
    """Test RSI calculation."""
    # Calculate RSI with default window
    result = rsi(sample_data['spread'])
    
    # Check shape and type
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    
    # Check values are within expected range (0-100)
    assert result.min() >= 0
    assert result.max() <= 100
    
    # Check for NaN values
    assert result.isna().sum() <= 14  # Should have NaNs only within the window
    
    # Test with custom window
    result_small = rsi(sample_data['spread'], window=5)
    assert result_small.isna().sum() <= 5
    
    # Make sure the indicators have different means
    common_index = result.dropna().index.intersection(result_small.dropna().index)
    assert len(common_index) > 0
    assert abs(result.loc[common_index].mean() - result_small.loc[common_index].mean()) > 1e-3


def test_macd(sample_data):
    """Test MACD calculation."""
    # Calculate MACD with default parameters
    macd_line, signal_line, histogram = macd(sample_data['spread'])
    
    # Check types and shapes
    assert isinstance(macd_line, pd.Series)
    assert isinstance(signal_line, pd.Series)
    assert isinstance(histogram, pd.Series)
    assert len(macd_line) == len(sample_data)
    assert len(signal_line) == len(sample_data)
    assert len(histogram) == len(sample_data)
    
    # Check that histogram equals macd_line - signal_line
    pd.testing.assert_series_equal(
        histogram,
        macd_line - signal_line,
        check_names=False
    )
    
    # Check NaN handling
    assert macd_line.isna().sum() <= 26  # NaNs at the beginning
    assert signal_line.isna().sum() <= 26 + 9  # Signal requires additional periods
    
    # Test with custom parameters
    macd_short, signal_short, hist_short = macd(
        sample_data['spread'], fast_period=5, slow_period=10, signal_period=3
    )
    assert macd_short.isna().sum() <= 10
    assert signal_short.isna().sum() <= 10 + 3
    
    # Make sure the indicators are different
    common_index = macd_line.dropna().index.intersection(macd_short.dropna().index)
    assert len(common_index) > 0
    assert abs(macd_line.loc[common_index].mean() - macd_short.loc[common_index].mean()) > 1e-5


def test_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    # Calculate Bollinger Bands with default parameters
    middle, upper, lower = bollinger_bands(sample_data['spread'])
    
    # Check types and shapes
    assert isinstance(middle, pd.Series)
    assert isinstance(upper, pd.Series)
    assert isinstance(lower, pd.Series)
    assert len(middle) == len(sample_data)
    assert len(upper) == len(sample_data)
    assert len(lower) == len(sample_data)
    
    # Check that upper > middle > lower
    valid_indices = ~middle.isna()
    assert (upper[valid_indices] >= middle[valid_indices]).all()
    assert (middle[valid_indices] >= lower[valid_indices]).all()
    
    # Check NaN handling
    assert middle.isna().sum() <= 20  # NaNs within window
    
    # Test with custom parameters
    middle_narrow, upper_narrow, lower_narrow = bollinger_bands(
        sample_data['spread'], window=10, num_std=1.0
    )
    
    # With smaller std, bands should be narrower
    valid_indices = ~middle.isna() & ~middle_narrow.isna()
    assert (upper[valid_indices] - lower[valid_indices] > 
            upper_narrow[valid_indices] - lower_narrow[valid_indices]).all()


def test_moving_averages(sample_data):
    """Test moving averages calculation."""
    # Calculate moving averages with default windows
    sma_dict = moving_averages(sample_data['spread'])
    
    # Check result structure
    assert isinstance(sma_dict, dict)
    assert len(sma_dict) == 4  # Default 4 windows
    assert 'sma_5' in sma_dict
    assert 'sma_10' in sma_dict
    assert 'sma_20' in sma_dict
    assert 'sma_50' in sma_dict
    
    # Check types
    for key, value in sma_dict.items():
        assert isinstance(value, pd.Series)
        assert len(value) == len(sample_data)
        assert value.isna().sum() <= int(key.split('_')[1])
    
    # Test with custom windows
    custom_sma = moving_averages(sample_data['spread'], windows=[3, 15])
    assert len(custom_sma) == 2
    assert 'sma_3' in custom_sma
    assert 'sma_15' in custom_sma
    
    # Make sure the indicators are different
    common_index = sma_dict['sma_5'].dropna().index.intersection(sma_dict['sma_10'].dropna().index)
    assert len(common_index) > 0
    assert abs(sma_dict['sma_5'].loc[common_index].mean() - sma_dict['sma_10'].loc[common_index].mean()) > 1e-5


def test_exponential_moving_averages(sample_data):
    """Test exponential moving averages calculation."""
    # Calculate EMAs with default spans
    ema_dict = exponential_moving_averages(sample_data['spread'])
    
    # Check result structure
    assert isinstance(ema_dict, dict)
    assert len(ema_dict) == 4  # Default 4 spans
    assert 'ema_5' in ema_dict
    assert 'ema_10' in ema_dict
    assert 'ema_20' in ema_dict
    assert 'ema_50' in ema_dict
    
    # Check types
    for key, value in ema_dict.items():
        assert isinstance(value, pd.Series)
        assert len(value) == len(sample_data)
    
    # EMAs respond faster to recent data than SMAs
    sma_dict = moving_averages(sample_data['spread'])
    
    # Compare EMA vs SMA reaction to changes (for periods where both are not NaN)
    # This is a simple check that might not always hold, but generally EMAs react faster
    ema10 = ema_dict['ema_10']
    sma10 = sma_dict['sma_10']
    
    # Calculate correlation with recent data
    recent_data = sample_data['spread'].iloc[-20:]
    
    ema_corr = recent_data.corr(ema10.iloc[-20:])
    sma_corr = recent_data.corr(sma10.iloc[-20:])
    
    # EMAs typically have higher correlation with recent data
    assert ema_corr >= sma_corr * 0.9  # Allow some flexibility


def test_atr(sample_data):
    """Test ATR calculation."""
    # Calculate ATR with default window
    result = atr(sample_data['high'], sample_data['low'], sample_data['close'])
    
    # Check type and shape
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    
    # ATR should be positive
    assert (result.dropna() > 0).all()
    
    # Check NaN handling
    assert result.isna().sum() <= 15  # NaNs at the beginning (window + 1)
    
    # Test with custom window
    result_small = atr(
        sample_data['high'], sample_data['low'], sample_data['close'], window=5
    )
    assert result_small.isna().sum() <= 6


def test_z_score(sample_data):
    """Test Z-score calculation."""
    # Calculate Z-score with default window
    result = z_score(sample_data['spread'])
    
    # Check type and shape
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    
    # Z-scores should be approximately normally distributed
    valid_values = result.dropna()
    
    # Most values should be within +/- 3 standard deviations
    assert (abs(valid_values) <= 3).mean() > 0.95
    
    # Check NaN handling
    assert result.isna().sum() <= 20  # NaNs at the beginning (window)
    
    # Test with custom window
    result_small = z_score(sample_data['spread'], window=10)
    assert result_small.isna().sum() <= 10


def test_stochastic_oscillator(sample_data):
    """Test stochastic oscillator calculation."""
    # Calculate stochastic oscillator with default parameters
    k, d = stochastic_oscillator(
        sample_data['high'], sample_data['low'], sample_data['close']
    )
    
    # Check types and shapes
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
    assert len(k) == len(sample_data)
    assert len(d) == len(sample_data)
    
    # Values should be between 0 and 100
    assert k.dropna().between(0, 100).all()
    assert d.dropna().between(0, 100).all()
    
    # Check NaN handling
    assert k.isna().sum() <= 14  # k_window is 14 by default
    assert d.isna().sum() <= 14 + 3  # d requires k plus d_window
    
    # Test with custom parameters
    k_short, d_short = stochastic_oscillator(
        sample_data['high'], sample_data['low'], sample_data['close'],
        k_window=5, d_window=2
    )
    assert k_short.isna().sum() <= 5
    assert d_short.isna().sum() <= 5 + 2


def test_rate_of_change(sample_data):
    """Test rate of change calculation."""
    # Calculate ROC with default window
    result = rate_of_change(sample_data['spread'])
    
    # Check type and shape
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    
    # Check NaN handling
    assert result.isna().sum() <= 10  # NaNs at the beginning (window)
    
    # Test with custom window
    result_small = rate_of_change(sample_data['spread'], window=5)
    assert result_small.isna().sum() <= 5
    
    # Make sure the indicators are different
    common_index = result.dropna().index.intersection(result_small.dropna().index)
    assert len(common_index) > 0
    assert abs(result.loc[common_index].std() - result_small.loc[common_index].std()) > 1e-5
    
    # ROC should reflect actual percentage changes
    # Calculate direct percentage change for a specific point
    i = 20  # Some index with enough history
    actual_change = (sample_data['spread'].iloc[i] / 
                    sample_data['spread'].iloc[i-10] - 1) * 100
    assert abs(result.iloc[i] - actual_change) < 1e-10


def test_add_all_indicators(sample_data):
    """Test the add_all_indicators function."""
    # Add all indicators
    result = add_all_indicators(sample_data)
    
    # Check that it's a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_data)
    
    # Check that it has all the expected columns
    expected_columns = [
        'spread', 'high', 'low', 'close',
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'bb_middle', 'bb_upper', 'bb_lower',
        'zscore_20',
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        'roc_10'
    ]
    for col in expected_columns:
        assert col in result.columns
    
    # Test with different price column
    result_close = add_all_indicators(sample_data, price_col='close')
    assert 'rsi_14' in result_close.columns
    
    # Test with high-low indicators
    result_full = add_all_indicators(sample_data, include_high_low=True)
    assert 'atr_14' in result_full.columns
    assert 'stoch_k' in result_full.columns
    assert 'stoch_d' in result_full.columns 