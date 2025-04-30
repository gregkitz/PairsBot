#!/usr/bin/env python3
"""
Test data generation script for the Intraday Statistical Arbitrage System.

This script generates synthetic data for testing purposes, creating cointegrated
pairs with controlled properties to help users test the system without real data.
It can generate both daily and intraday data with specific cointegration characteristics.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
from scipy import stats

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import create_directory

def generate_ornstein_uhlenbeck_process(n_steps, theta=0.1, mu=0, sigma=0.1, dt=1.0):
    """
    Generate an Ornstein-Uhlenbeck process.
    
    Parameters:
    -----------
    n_steps : int
        Number of steps to generate
    theta : float
        Mean reversion strength
    mu : float
        Long-term mean
    sigma : float
        Volatility
    dt : float
        Time increment
        
    Returns:
    --------
    numpy.ndarray
        Array of simulated process values
    """
    x = np.zeros(n_steps)
    x[0] = mu
    
    for i in range(1, n_steps):
        dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
        x[i] = x[i-1] + dx
        
    return x

def generate_cointegrated_pair(n_points=1000, hedge_ratio=0.7, noise_ratio=0.3, mean_rev_strength=0.1, 
                               drift=0.0001, volatility=0.01, regime_shifts=False, breakpoints=None,
                               change_hedge_ratio=False, seasonal_pattern=False):
    """
    Generate a pair of cointegrated time series.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    hedge_ratio : float
        Relationship ratio between the two series
    noise_ratio : float
        Amount of noise to add to the cointegrating relationship
    mean_rev_strength : float
        Strength of mean reversion in the spread
    drift : float
        Average drift in the first series
    volatility : float
        Volatility of the random walk component
    regime_shifts : bool
        Whether to include sudden regime shifts
    breakpoints : list
        List of points where cointegration relationship breaks
    change_hedge_ratio : bool
        Whether to change the hedge ratio at breakpoints
    seasonal_pattern : bool
        Whether to add seasonal patterns to the series
        
    Returns:
    --------
    tuple
        (series1, series2, spread) where series1, series2 are the prices and spread is their relationship
    """
    # Generate the first random walk with possible drift
    random_steps = np.random.normal(drift, volatility, n_points)
    series1 = 100 + np.cumsum(random_steps)
    
    # Add seasonal component if requested
    if seasonal_pattern:
        # Annual seasonality (365 periods)
        seasonal_period = min(365, n_points // 2)
        seasonal_component = 5 * np.sin(2 * np.pi * np.arange(n_points) / seasonal_period)
        series1 += seasonal_component
    
    # Generate a cointegrated series
    spread = generate_ornstein_uhlenbeck_process(
        n_points, 
        theta=mean_rev_strength, 
        mu=0, 
        sigma=noise_ratio
    )
    
    # Create the second series based on the hedge ratio
    series2 = hedge_ratio * series1 + spread
    
    # Handle regime shifts and breakpoints
    if regime_shifts or breakpoints:
        # Default breakpoints if not specified
        if breakpoints is None:
            # Add random breakpoints
            n_breaks = 2 if regime_shifts else 0
            breakpoints = sorted(np.random.choice(range(100, n_points-100), size=n_breaks, replace=False))
        
        # Process each breakpoint
        current_hedge_ratio = hedge_ratio
        for point in breakpoints:
            # Determine type of break
            if change_hedge_ratio:
                # Change hedge ratio (alters cointegration relationship)
                new_hedge_ratio = current_hedge_ratio * np.random.uniform(0.8, 1.2)
                shift_size = (new_hedge_ratio - current_hedge_ratio) * series1[point:]
                series2[point:] += shift_size
                current_hedge_ratio = new_hedge_ratio
            else:
                # Add level shift to spread (maintains cointegration)
                shift_size = np.random.uniform(-2, 2)  # Random jump size
                series2[point:] += shift_size
    
    return series1, series2, spread

def generate_non_cointegrated_pair(n_points=1000, initial_correlation=0.6, volatility=0.01):
    """
    Generate a pair of non-cointegrated time series with initial correlation.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    initial_correlation : float
        Target correlation coefficient between the series
    volatility : float
        Volatility of the random walk components
        
    Returns:
    --------
    tuple
        (series1, series2) - two non-cointegrated price series
    """
    # Generate correlated random walks
    # Create a covariance matrix with desired correlation
    cov_matrix = np.array([[1, initial_correlation], [initial_correlation, 1]])
    
    # Generate correlated random steps
    random_steps = np.random.multivariate_normal(
        mean=[0.0001, 0.0002],  # Slightly different drifts
        cov=volatility**2 * cov_matrix,
        size=n_points
    )
    
    # Create the random walks
    series1 = 100 + np.cumsum(random_steps[:, 0])
    series2 = 100 + np.cumsum(random_steps[:, 1])
    
    return series1, series2

def generate_spurious_cointegration(n_points=1000, trending=True, volatility=0.01):
    """
    Generate a pair with spurious cointegration (both series have unit roots).
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    trending : bool
        Whether to add deterministic trends
    volatility : float
        Volatility of the series
        
    Returns:
    --------
    tuple
        (series1, series2) - two series with spurious correlation
    """
    # Generate two independent random walks
    rw1 = np.cumsum(np.random.normal(0.0001, volatility, n_points))
    rw2 = np.cumsum(np.random.normal(0.0002, volatility, n_points))
    
    # Add deterministic trends if requested
    if trending:
        trend = np.linspace(0, 10, n_points)
        rw1 += trend
        rw2 += trend * 0.8  # Different trend slope
    
    return 100 + rw1, 100 + rw2

def generate_dates(n_points, frequency='1d', start_date=None):
    """
    Generate a DatetimeIndex with the specified frequency.
    
    Parameters:
    -----------
    n_points : int
        Number of points to generate
    frequency : str
        Pandas frequency string ('1d' for daily, '1h' for hourly, etc.)
    start_date : datetime, optional
        Starting date, defaults to n_points ago from now
        
    Returns:
    --------
    pandas.DatetimeIndex
        Generated dates
    """
    if start_date is None:
        if frequency == '1d':
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=n_points)
        else:
            # For intraday, use trading hours
            end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
            if frequency == '1h':
                start_date = end_date - timedelta(hours=n_points)
            elif frequency == '5min':
                start_date = end_date - timedelta(minutes=5*n_points)
            elif frequency == '1min':
                start_date = end_date - timedelta(minutes=n_points)
    
    return pd.date_range(start=start_date, periods=n_points, freq=frequency)

def save_test_data(ticker1, ticker2, data1, data2, dates, frequency='1d', output_dir='data/historical'):
    """
    Save generated test data to the data directory.
    
    Parameters:
    -----------
    ticker1, ticker2 : str
        Ticker symbols for the pair
    data1, data2 : numpy.ndarray
        Price data for each ticker
    dates : pandas.DatetimeIndex
        Dates for the data
    frequency : str
        Data frequency ('1d', '1h', '5min', etc.)
    output_dir : str
        Directory to save the data
    """
    # Create directories
    freq_dir = {
        '1d': '1day',
        '1h': '1hour',
        '5min': '5min',
        '1min': '1min'
    }.get(frequency, '1day')
    
    for ticker in [ticker1, ticker2]:
        dir_path = f'{output_dir}/{ticker}/{freq_dir}'
        create_directory(dir_path)
    
    # Create DataFrames with standard OHLCV structure
    for ticker, data in [(ticker1, data1), (ticker2, data2)]:
        # Add some realistic variation for high/low based on volatility
        volatility_factor = 0.005 if frequency == '1d' else 0.002
        df = pd.DataFrame({
            'open': data,
            'high': data * (1 + np.random.uniform(0, volatility_factor, len(data))),
            'low': data * (1 - np.random.uniform(0, volatility_factor, len(data))),
            'close': data,
            'volume': np.random.randint(1000, 10000, len(data))
        }, index=dates)
        
        # Save to CSV
        df.to_csv(f'{output_dir}/{ticker}/{freq_dir}/data.csv')
        print(f"Saved {freq_dir} test data for {ticker}")

def create_pairs_file(pairs, output_file='data/pairs.csv'):
    """
    Create a CSV file with pair information for testing.
    
    Parameters:
    -----------
    pairs : list
        List of tuples with (ticker1, ticker2, hedge_ratio, mean_rev_strength)
    output_file : str
        Path to save the pairs file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create DataFrame with pair information
    pairs_data = []
    for ticker1, ticker2, hedge_ratio, _, mean_rev in pairs:
        pairs_data.append({
            'asset1': ticker1,
            'asset2': ticker2,
            'hedge_ratio': hedge_ratio,
            'half_life': int(1.0/mean_rev) if mean_rev > 0 else 0,
            'zscore_entry': 2.0,
            'zscore_exit': 0.5,
            'lookback_period': 20
        })
    
    pd.DataFrame(pairs_data).to_csv(output_file, index=False)
    print(f"Saved pairs file to {output_file}")

def plot_test_pair(ticker1, ticker2, data1, data2, spread, dates, output_dir='plots'):
    """
    Plot the generated pair data for visualization.
    
    Parameters:
    -----------
    ticker1, ticker2 : str
        Ticker symbols for the pair
    data1, data2 : numpy.ndarray
        Price data for each ticker
    spread : numpy.ndarray
        The spread between the two series
    dates : pandas.DatetimeIndex
        Dates for the data
    output_dir : str
        Directory to save the plots
    """
    # Create directory if it doesn't exist
    create_directory(output_dir)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot the price series
    ax1.plot(dates, data1, label=ticker1)
    ax1.plot(dates, data2, label=ticker2)
    ax1.set_title(f'Synthetic Price Series: {ticker1}-{ticker2}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the spread
    ax2.plot(dates, spread)
    ax2.axhline(y=0, color='r', linestyle='-')
    ax2.axhline(y=2, color='g', linestyle='--')
    ax2.axhline(y=-2, color='g', linestyle='--')
    ax2.set_title('Spread Z-Score')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{ticker1}_{ticker2}_test_data.png')
    plt.close()

def generate_academic_test_dataset(output_dir='tests/fixtures/cointegration_test_data'):
    """
    Generate a comprehensive set of test data for academic validation of cointegration tests.
    
    This function creates various types of pairs with known properties:
    1. Strongly cointegrated pairs
    2. Weakly cointegrated pairs
    3. Non-cointegrated pairs
    4. Pairs with regime shifts
    5. Pairs with structural breaks
    6. Pairs with spurious cointegration
    7. Pairs with seasonal cointegration patterns
    8. Pairs for testing Johansen procedure (multivariate)
    9. Edge cases with borderline cointegration
    10. Cases with different deterministic terms
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the test data
    
    Returns:
    --------
    dict
        Metadata about the generated test sets
    """
    create_directory(output_dir)
    
    # Seed for reproducibility
    np.random.seed(42)
    
    # Parameters for data generation
    n_points = 500
    dates = generate_dates(n_points, '1d')
    
    # Container for all test sets
    test_sets = {}
    metadata = {}
    
    # 1. Strongly cointegrated pair
    print("Generating strongly cointegrated pair...")
    strong_coint_s1, strong_coint_s2, strong_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.7,
        noise_ratio=0.1,  # Low noise
        mean_rev_strength=0.2  # Strong mean reversion
    )
    test_sets['strong_cointegration'] = {
        'series1': strong_coint_s1,
        'series2': strong_coint_s2,
        'spread': strong_spread,
        'dates': dates
    }
    metadata['strong_cointegration'] = {
        'hedge_ratio': 0.7,
        'noise_ratio': 0.1,
        'mean_rev_strength': 0.2,
        'expected_half_life': 5,  # Approximate expected half-life
        'expected_p_value': 0.01,  # Expected to be strongly significant
        'description': 'Strong cointegration with fast mean reversion'
    }
    
    # 2. Weakly cointegrated pair
    print("Generating weakly cointegrated pair...")
    weak_coint_s1, weak_coint_s2, weak_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.8,
        noise_ratio=0.3,  # More noise
        mean_rev_strength=0.05  # Weaker mean reversion
    )
    test_sets['weak_cointegration'] = {
        'series1': weak_coint_s1,
        'series2': weak_coint_s2,
        'spread': weak_spread,
        'dates': dates
    }
    metadata['weak_cointegration'] = {
        'hedge_ratio': 0.8,
        'noise_ratio': 0.3,
        'mean_rev_strength': 0.05,
        'expected_half_life': 20,
        'expected_p_value': 0.04,  # Expected to be just significant
        'description': 'Weak cointegration with slow mean reversion'
    }
    
    # 3. Non-cointegrated pair
    print("Generating non-cointegrated pair...")
    non_coint_s1, non_coint_s2 = generate_non_cointegrated_pair(
        n_points=n_points,
        initial_correlation=0.4
    )
    spread_nc = non_coint_s1 - 0.8 * non_coint_s2  # Arbitrary spread calculation
    test_sets['non_cointegrated'] = {
        'series1': non_coint_s1,
        'series2': non_coint_s2,
        'spread': spread_nc,
        'dates': dates
    }
    metadata['non_cointegrated'] = {
        'initial_correlation': 0.4,
        'expected_p_value': 0.3,  # Expected to be not significant
        'description': 'Correlated but non-cointegrated random walks'
    }
    
    # 4. Pair with regime shifts
    print("Generating pair with regime shifts...")
    regime_s1, regime_s2, regime_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.75,
        noise_ratio=0.2,
        mean_rev_strength=0.15,
        regime_shifts=True,
        breakpoints=[150, 300]  # Fixed breakpoints for reproducibility
    )
    test_sets['regime_shifts'] = {
        'series1': regime_s1,
        'series2': regime_s2,
        'spread': regime_spread,
        'dates': dates,
        'breakpoints': [150, 300]
    }
    metadata['regime_shifts'] = {
        'hedge_ratio': 0.75,
        'noise_ratio': 0.2,
        'mean_rev_strength': 0.15,
        'breakpoints': [150, 300],
        'description': 'Cointegrated pair with regime shifts in the mean'
    }
    
    # 5. Pair with structural break in the cointegration relationship
    print("Generating pair with structural break...")
    struct_s1, struct_s2, struct_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.65,
        noise_ratio=0.2,
        mean_rev_strength=0.1,
        breakpoints=[250],  # One major break
        change_hedge_ratio=True  # Change the cointegration relationship
    )
    test_sets['structural_break'] = {
        'series1': struct_s1,
        'series2': struct_s2,
        'spread': struct_spread,
        'dates': dates,
        'breakpoints': [250]
    }
    metadata['structural_break'] = {
        'initial_hedge_ratio': 0.65,
        'noise_ratio': 0.2,
        'mean_rev_strength': 0.1,
        'breakpoints': [250],
        'description': 'Pair with structural break in cointegration relationship'
    }
    
    # 6. Spurious cointegration
    print("Generating pair with spurious cointegration...")
    spurious_s1, spurious_s2 = generate_spurious_cointegration(
        n_points=n_points,
        trending=True
    )
    spread_spurious = spurious_s1 - 0.8 * spurious_s2
    test_sets['spurious_cointegration'] = {
        'series1': spurious_s1,
        'series2': spurious_s2,
        'spread': spread_spurious,
        'dates': dates
    }
    metadata['spurious_cointegration'] = {
        'trending': True,
        'description': 'Pair with spurious cointegration due to common trends'
    }
    
    # 7. Cointegrated with seasonality
    print("Generating cointegrated pair with seasonality...")
    seasonal_s1, seasonal_s2, seasonal_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.6,
        noise_ratio=0.15,
        mean_rev_strength=0.15,
        seasonal_pattern=True
    )
    test_sets['seasonal_cointegration'] = {
        'series1': seasonal_s1,
        'series2': seasonal_s2,
        'spread': seasonal_spread,
        'dates': dates
    }
    metadata['seasonal_cointegration'] = {
        'hedge_ratio': 0.6,
        'noise_ratio': 0.15,
        'mean_rev_strength': 0.15,
        'seasonal_pattern': True,
        'description': 'Cointegrated pair with seasonal patterns'
    }
    
    # 8. Multivariate cointegration for Johansen testing (3 variables)
    print("Generating multivariate cointegration dataset...")
    # First generate a basic cointegrated pair
    multi_base_s1, multi_base_s2, _ = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.65,
        noise_ratio=0.2,
        mean_rev_strength=0.12
    )
    # Generate a third series cointegrated with the first
    _, multi_s3, _ = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.9,
        noise_ratio=0.25,
        mean_rev_strength=0.1,
        # Use the first series from previous generation
        # This ensures s1, s2, and s3 are cointegrated
    )
    # Create a system with 2 cointegrating relationships
    multi_s1 = multi_base_s1
    multi_s2 = multi_base_s2
    multi_s3 = multi_base_s1 * 0.9 + np.random.normal(0, 1, n_points)
    
    # Calculate residuals/spreads
    spread_m1 = multi_s1 - (0.65 * multi_s2)
    spread_m2 = multi_s1 - (0.9 * multi_s3)
    
    test_sets['multivariate_cointegration'] = {
        'series1': multi_s1,
        'series2': multi_s2,
        'series3': multi_s3,
        'spread1': spread_m1,
        'spread2': spread_m2,
        'dates': dates
    }
    metadata['multivariate_cointegration'] = {
        'hedge_ratios': [0.65, 0.9],
        'expected_coint_relations': 2,
        'description': 'System with multiple cointegrating relationships for Johansen testing'
    }
    
    # 9. Edge case: Borderline cointegration
    print("Generating borderline cointegration case...")
    borderline_s1, borderline_s2, borderline_spread = generate_cointegrated_pair(
        n_points=n_points,
        hedge_ratio=0.75,
        noise_ratio=0.35,  # Higher noise
        mean_rev_strength=0.03  # Very weak mean reversion
    )
    test_sets['borderline_cointegration'] = {
        'series1': borderline_s1,
        'series2': borderline_s2,
        'spread': borderline_spread,
        'dates': dates
    }
    metadata['borderline_cointegration'] = {
        'hedge_ratio': 0.75,
        'noise_ratio': 0.35,
        'mean_rev_strength': 0.03,
        'expected_half_life': 25,
        'expected_p_value': 0.049,  # Just barely significant
        'description': 'Borderline case that tests robustness of cointegration methods'
    }
    
    # 10. Different deterministic terms cases for testing Johansen with varying models
    print("Generating cases with different deterministic terms...")
    
    # 10.1 No trend (drift only in data)
    drift_s1 = np.cumsum(np.random.normal(0.01, 0.1, n_points))  # With drift
    drift_s2 = 0.8 * drift_s1 + np.random.normal(0, 0.5, n_points)  # Cointegrated with drift
    drift_spread = drift_s1 - 0.8 * drift_s2
    
    test_sets['drift_term'] = {
        'series1': drift_s1,
        'series2': drift_s2,
        'spread': drift_spread,
        'dates': dates
    }
    metadata['drift_term'] = {
        'hedge_ratio': 0.8,
        'drift': 0.01,
        'description': 'Cointegrated with drift term (model 1 in Johansen)'
    }
    
    # 10.2 With linear trend
    trend = np.linspace(0, 5, n_points)
    trend_s1 = np.cumsum(np.random.normal(0, 0.1, n_points)) + trend
    trend_s2 = 0.7 * trend_s1 + np.random.normal(0, 0.5, n_points)
    trend_spread = trend_s1 - 0.7 * trend_s2
    
    test_sets['linear_trend'] = {
        'series1': trend_s1,
        'series2': trend_s2,
        'spread': trend_spread,
        'dates': dates
    }
    metadata['linear_trend'] = {
        'hedge_ratio': 0.7,
        'has_trend': True,
        'description': 'Cointegrated with linear trend (model 3-4 in Johansen)'
    }
    
    # Save all datasets
    print("Saving datasets...")
    for test_type, data in test_sets.items():
        # Convert to DataFrame
        series_data = {}
        for key, series in data.items():
            if key != 'dates' and key != 'breakpoints':
                if isinstance(series, np.ndarray):
                    series = pd.Series(series, index=data['dates'])
                series_data[f'{test_type}_{key}'] = series
        
        df = pd.DataFrame(series_data)
        
        # Save to pickle
        pickle_file = os.path.join(output_dir, f"{test_type}.pkl")
        df.to_pickle(pickle_file)
        
        # Also save as CSV for easier inspection
        csv_file = os.path.join(output_dir, f"{test_type}.csv")
        df.to_csv(csv_file)
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Academic test datasets saved to {output_dir}")
    return metadata

def main():
    """
    Main function to run the script.
    """
    parser = argparse.ArgumentParser(description='Generate test data for cointegration testing')
    parser.add_argument('--tickers', nargs='+', help='Specific ticker pairs to generate')
    parser.add_argument('--output-dir', default='data/historical', help='Output directory for data')
    parser.add_argument('--frequency', default='1d', help='Data frequency (1d, 1h, 5min)')
    parser.add_argument('--n-points', type=int, default=1000, help='Number of data points to generate')
    parser.add_argument('--plot', action='store_true', help='Generate plots of the data')
    parser.add_argument('--academic-data', action='store_true', help='Generate academic test dataset')
    
    args = parser.parse_args()
    
    # Generate academic test dataset
    print("Generating academic test dataset...")
    generate_academic_test_dataset()
    
    # Default tickers if none provided
    default_pairs = [('CL', 'HO'), ('GC', 'SI'), ('ZC', 'ZW')]
    
    if args.tickers:
        # Parse ticker pairs from command line
        pairs = []
        for ticker_arg in args.tickers:
            if '-' in ticker_arg:
                ticker1, ticker2 = ticker_arg.split('-')
                pairs.append((ticker1, ticker2))
            else:
                print(f"Warning: Ignoring ticker {ticker_arg}, expected format TICKER1-TICKER2")
    else:
        pairs = default_pairs
    
    # Create output directory if it doesn't exist
    create_directory(args.output_dir)
    
    # Generate data for each ticker pair
    for ticker1, ticker2 in pairs:
        print(f"Generating cointegrated pair: {ticker1}-{ticker2}")
        
        # Generate cointegrated price series
        price1, price2, spread = generate_cointegrated_pair(
            n_points=args.n_points,
            hedge_ratio=0.7,
            noise_ratio=0.2,
            mean_rev_strength=0.15
        )
        
        # Generate dates based on frequency
        dates = generate_dates(args.n_points, args.frequency)
        
        # Save the data
        save_test_data(ticker1, ticker2, price1, price2, dates, 
                      frequency=args.frequency, output_dir=args.output_dir)
        
        # Create plot if requested
        if args.plot:
            plot_test_pair(ticker1, ticker2, price1, price2, spread, dates)
    
    # Print summary of pairs (skip creating pairs.csv file for now)
    ticker_pairs = [f"{t1}-{t2}" for t1, t2 in pairs]
    print(f"You can now run the system on these test pairs:")
    print(f"./run.py --mode backtest --pairs {' '.join(ticker_pairs)}")
    
if __name__ == "__main__":
    main() 