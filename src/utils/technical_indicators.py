import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    window : int, default=14
        The lookback window for RSI calculation
        
    Returns:
    --------
    pd.Series
        Series containing RSI values
    """
    # Calculate price differences
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    return rsi_series


def macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    fast_period : int, default=12
        The lookback period for the fast EMA
    slow_period : int, default=26
        The lookback period for the slow EMA
    signal_period : int, default=9
        The lookback period for the signal line
        
    Returns:
    --------
    tuple
        (macd_line, signal_line, histogram)
    """
    # Calculate EMAs
    ema_fast = series.ewm(span=fast_period).mean()
    ema_slow = series.ewm(span=slow_period).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    window : int, default=20
        The lookback window for calculating the moving average and standard deviation
    num_std : float, default=2.0
        Number of standard deviations for the bands
        
    Returns:
    --------
    tuple
        (middle_band, upper_band, lower_band)
    """
    # Calculate the middle band (simple moving average)
    middle_band = series.rolling(window=window).mean()
    
    # Calculate the standard deviation
    std_dev = series.rolling(window=window).std()
    
    # Calculate the upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return middle_band, upper_band, lower_band


def moving_averages(series: pd.Series, windows: list = [5, 10, 20, 50]) -> dict:
    """
    Calculate multiple simple moving averages for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    windows : list, default=[5, 10, 20, 50]
        List of window sizes for the moving averages
        
    Returns:
    --------
    dict
        Dictionary of moving averages with keys as f'sma_{window}'
    """
    result = {}
    for window in windows:
        result[f'sma_{window}'] = series.rolling(window=window).mean()
    return result


def exponential_moving_averages(series: pd.Series, spans: list = [5, 10, 20, 50]) -> dict:
    """
    Calculate multiple exponential moving averages for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    spans : list, default=[5, 10, 20, 50]
        List of span values for the EMAs
        
    Returns:
    --------
    dict
        Dictionary of exponential moving averages with keys as f'ema_{span}'
    """
    result = {}
    for span in spans:
        result[f'ema_{span}'] = series.ewm(span=span).mean()
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Average True Range (ATR) for a price series.
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of closing prices
    window : int, default=14
        The lookback window for the ATR calculation
        
    Returns:
    --------
    pd.Series
        Series containing ATR values
    """
    # Previous close values (shifted by 1)
    prev_close = close.shift(1)
    
    # Calculate the three differences
    tr1 = high - low  # Current high - current low
    tr2 = (high - prev_close).abs()  # Current high - previous close
    tr3 = (low - prev_close).abs()  # Current low - previous close
    
    # True range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR as the moving average of true range
    atr_series = tr.rolling(window=window).mean()
    
    return atr_series


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate the Z-score for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    window : int, default=20
        The lookback window for calculating the mean and standard deviation
        
    Returns:
    --------
    pd.Series
        Series containing Z-score values
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Calculate Z-score
    z_score_series = (series - rolling_mean) / rolling_std
    
    return z_score_series


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate the Stochastic Oscillator (%K and %D) for a price series.
    
    Parameters:
    -----------
    high : pd.Series
        Series of high prices
    low : pd.Series
        Series of low prices
    close : pd.Series
        Series of closing prices
    k_window : int, default=14
        The lookback window for %K calculation
    d_window : int, default=3
        The window for the %D moving average
        
    Returns:
    --------
    tuple
        (%K, %D)
    """
    # Calculate the lowest low and highest high over the lookback period
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    # Calculate %K
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (3-day SMA of %K)
    d = k.rolling(window=d_window).mean()
    
    return k, d


def rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculate the Rate of Change (ROC) for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        The input time series (typically price or spread data)
    window : int, default=10
        The lookback window for the ROC calculation
        
    Returns:
    --------
    pd.Series
        Series containing ROC values (percentage change)
    """
    # Calculate ROC
    roc_series = 100 * (series / series.shift(window) - 1)
    
    return roc_series


def add_all_indicators(df: pd.DataFrame, price_col: str = 'spread', 
                      include_high_low: bool = False) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str, default='spread'
        Column name of the price or spread data
    include_high_low : bool, default=False
        Whether to include indicators requiring high/low prices
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional indicator columns
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Add RSI
    result['rsi_14'] = rsi(result[price_col])
    
    # Add MACD
    macd_line, signal_line, histogram = macd(result[price_col])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    # Add Bollinger Bands
    mid, upper, lower = bollinger_bands(result[price_col])
    result['bb_middle'] = mid
    result['bb_upper'] = upper
    result['bb_lower'] = lower
    
    # Add Z-scores
    result['zscore_20'] = z_score(result[price_col], 20)
    
    # Add moving averages
    sma_dict = moving_averages(result[price_col], [5, 10, 20, 50])
    for key, value in sma_dict.items():
        result[key] = value
        
    # Add exponential moving averages
    ema_dict = exponential_moving_averages(result[price_col], [5, 10, 20, 50])
    for key, value in ema_dict.items():
        result[key] = value
    
    # Add rate of change
    result['roc_10'] = rate_of_change(result[price_col])
    
    # Add indicators that require high/low if available
    if include_high_low and all(col in df.columns for col in ['high', 'low']):
        # Add ATR
        result['atr_14'] = atr(df['high'], df['low'], df[price_col])
        
        # Add Stochastic Oscillator
        k, d = stochastic_oscillator(df['high'], df['low'], df[price_col])
        result['stoch_k'] = k
        result['stoch_d'] = d
    
    return result 