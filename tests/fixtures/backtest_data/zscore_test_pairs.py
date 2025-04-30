"""
Realistic test data for z-score strategy backtesting.

This module provides functions to generate realistic price series with known
cointegration properties for testing the z-score trading strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mean_reverting_pair(length=500, seed=42):
    """
    Generate a pair of price series with a mean-reverting relationship.
    
    Parameters:
    -----------
    length : int
        Number of periods to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing:
        - price1: First price series
        - price2: Second price series
        - hedge_ratio: True hedge ratio used to generate the data
        - spread: The true spread series
        - dates: Date index for the series
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # Create a mean-reverting spread process
    ou_process = np.zeros(length)
    # Ornstein-Uhlenbeck parameters
    theta = 0.2  # Mean reversion strength
    mu = 0.0     # Long-term mean
    sigma = 0.1  # Volatility
    
    # Generate OU process
    for t in range(1, length):
        ou_process[t] = ou_process[t-1] + theta * (mu - ou_process[t-1]) + sigma * np.random.normal()
    
    # Create a random walk for the first price
    random_walk1 = np.cumsum(np.random.normal(0.001, 0.01, length))
    price1 = 100 * np.exp(random_walk1)
    
    # Set hedge ratio
    hedge_ratio = 0.75
    
    # Create second price using hedge ratio and adding the OU process
    price2 = (price1 / hedge_ratio) + ou_process
    
    # Calculate the true spread
    spread = price1 - hedge_ratio * price2
    
    return {
        'price1': pd.Series(price1, index=dates),
        'price2': pd.Series(price2, index=dates),
        'hedge_ratio': hedge_ratio,
        'spread': pd.Series(spread, index=dates),
        'dates': dates
    }

def generate_trending_pair_with_correlation_breakdown(length=500, breakdown_start=350, seed=43):
    """
    Generate a pair of initially cointegrated price series that experience a 
    correlation breakdown after a certain point.
    
    Parameters:
    -----------
    length : int
        Number of periods to generate
    breakdown_start : int
        Index at which the correlation should start to break down
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing price series and metadata
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # Initialize price series
    price1 = np.zeros(length)
    price2 = np.zeros(length)
    
    # Set initial prices
    price1[0] = 100
    price2[0] = 150
    
    # Set parameters
    hedge_ratio = 0.8
    common_drift = 0.0002
    common_vol = 0.008
    specific_vol = 0.004
    
    # Generate prices before breakdown
    for t in range(1, breakdown_start):
        # Common component
        common_shock = common_drift + common_vol * np.random.normal()
        
        # Specific components
        specific1 = specific_vol * np.random.normal()
        specific2 = specific_vol * np.random.normal()
        
        # Update prices
        price1[t] = price1[t-1] * (1 + common_shock + specific1)
        price2[t] = price2[t-1] * (1 + common_shock + specific2)
    
    # Generate prices after breakdown with decreasing correlation
    for t in range(breakdown_start, length):
        # Decay factor reduces correlation over time
        decay_factor = 1 - ((t - breakdown_start) / (length - breakdown_start))
        
        # Common component with decreasing influence
        common_shock = common_drift + decay_factor * common_vol * np.random.normal()
        
        # Specific components with increasing influence
        specific1 = specific_vol * (2 - decay_factor) * np.random.normal()
        specific2 = specific_vol * (2 - decay_factor) * np.random.normal()
        
        # Different drifts after breakdown
        drift1 = 0.001  # Stronger upward trend
        drift2 = 0.0005  # Weaker upward trend
        
        # Update prices
        price1[t] = price1[t-1] * (1 + drift1 + common_shock + specific1)
        price2[t] = price2[t-1] * (1 + drift2 + common_shock + specific2)
    
    # Convert to pandas Series
    price1_series = pd.Series(price1, index=dates)
    price2_series = pd.Series(price2, index=dates)
    
    return {
        'price1': price1_series,
        'price2': price2_series,
        'hedge_ratio': hedge_ratio,
        'breakdown_index': breakdown_start,
        'breakdown_date': dates[breakdown_start],
        'dates': dates
    }

def generate_extreme_volatility_pair(length=500, volatility_spike_indices=[200, 350], seed=44):
    """
    Generate a pair with periods of extreme volatility to test strategy robustness.
    
    Parameters:
    -----------
    length : int
        Number of periods to generate
    volatility_spike_indices : list of int
        Indices at which volatility spikes occur
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing price series and metadata
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # Baseline volatility
    normal_vol = 0.01
    spike_vol = 0.05
    
    # Create volatility series
    volatility = np.ones(length) * normal_vol
    
    # Add volatility spikes
    for spike_idx in volatility_spike_indices:
        # Create 10-day volatility spike
        for i in range(10):
            if spike_idx + i < length:
                # Gradual increase and decrease in volatility
                if i < 5:
                    volatility[spike_idx + i] = normal_vol + (i+1)/5 * (spike_vol - normal_vol)
                else:
                    volatility[spike_idx + i] = spike_vol - (i-4)/5 * (spike_vol - normal_vol)
    
    # Generate cointegrated prices with time-varying volatility
    price1 = np.zeros(length)
    price2 = np.zeros(length)
    
    # Initial prices
    price1[0] = 100
    price2[0] = 120
    
    # Hedge ratio
    hedge_ratio = 0.85
    
    # Mean-reverting component
    ou_process = np.zeros(length)
    theta = 0.15  # Mean reversion speed
    
    # Generate prices
    for t in range(1, length):
        # Update mean-reverting process
        ou_process[t] = ou_process[t-1] * (1 - theta) + volatility[t] * np.random.normal()
        
        # Update prices
        price1[t] = price1[t-1] * (1 + 0.0002 + volatility[t] * np.random.normal())
        # Second price follows the first with the hedge ratio relationship plus the OU process
        price2[t] = (price1[t] / hedge_ratio) + ou_process[t]
    
    return {
        'price1': pd.Series(price1, index=dates),
        'price2': pd.Series(price2, index=dates),
        'hedge_ratio': hedge_ratio,
        'volatility': pd.Series(volatility, index=dates),
        'volatility_spike_dates': [dates[idx] for idx in volatility_spike_indices],
        'dates': dates
    }

def generate_dataset_with_gaps(length=500, gap_indices=[100, 300], gap_lengths=[5, 7], seed=45):
    """
    Generate a dataset with missing data periods to test how the strategy
    handles data gaps.
    
    Parameters:
    -----------
    length : int
        Number of periods to generate
    gap_indices : list of int
        Starting indices for data gaps
    gap_lengths : list of int
        Lengths of data gaps
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing price series with gaps and metadata
    """
    np.random.seed(seed)
    
    # Create full date range
    full_dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # Generate basic cointegrated pair
    base_data = generate_mean_reverting_pair(length=length, seed=seed)
    price1 = base_data['price1']
    price2 = base_data['price2']
    
    # Create masks for gaps
    mask1 = np.ones(length, dtype=bool)
    mask2 = np.ones(length, dtype=bool)
    
    # Apply gaps to both series (some common, some series-specific)
    for i, (gap_idx, gap_len) in enumerate(zip(gap_indices, gap_lengths)):
        if gap_idx + gap_len < length:
            # For even-indexed gaps, apply to both series
            if i % 2 == 0:
                mask1[gap_idx:gap_idx+gap_len] = False
                mask2[gap_idx:gap_idx+gap_len] = False
            # For odd-indexed gaps, apply to only one series
            else:
                # Alternate which series gets the gap
                if i % 4 == 1:
                    mask1[gap_idx:gap_idx+gap_len] = False
                else:
                    mask2[gap_idx:gap_idx+gap_len] = False
    
    # Apply masks to create series with gaps
    price1_with_gaps = price1.copy()
    price2_with_gaps = price2.copy()
    
    price1_with_gaps[~mask1] = np.nan
    price2_with_gaps[~mask2] = np.nan
    
    return {
        'price1': price1_with_gaps,
        'price2': price2_with_gaps,
        'price1_full': price1,
        'price2_full': price2,
        'hedge_ratio': base_data['hedge_ratio'],
        'gap_indices': gap_indices,
        'gap_lengths': gap_lengths,
        'dates': full_dates
    }

def generate_all_test_datasets():
    """
    Generate all test datasets and return them in a dictionary.
    
    Returns:
    --------
    dict
        Dictionary containing all test datasets
    """
    return {
        'mean_reverting': generate_mean_reverting_pair(),
        'correlation_breakdown': generate_trending_pair_with_correlation_breakdown(),
        'volatility_spikes': generate_extreme_volatility_pair(),
        'data_gaps': generate_dataset_with_gaps()
    }

if __name__ == "__main__":
    # Generate sample data and print statistics
    data = generate_all_test_datasets()
    
    for name, dataset in data.items():
        print(f"Dataset: {name}")
        print(f"Length: {len(dataset['price1'])}")
        print(f"Price1 mean: {dataset['price1'].mean():.2f}")
        print(f"Price2 mean: {dataset['price2'].mean():.2f}")
        if 'hedge_ratio' in dataset:
            print(f"Hedge ratio: {dataset['hedge_ratio']:.4f}")
        print("-" * 40) 