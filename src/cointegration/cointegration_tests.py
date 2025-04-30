import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings

def calculate_half_life(spread, max_half_life=252):
    """
    Calculate half-life of mean reversion for a price spread series
    using Ornstein-Uhlenbeck process with improved robustness for edge cases.
    
    Parameters:
    -----------
    spread : pandas.Series
        The spread series to analyze
    max_half_life : int
        Maximum allowable half-life to return, to handle near unit-root processes
        
    Returns:
    --------
    dict
        Dictionary containing half-life and additional validation metrics:
        - half_life: Float, half-life of mean reversion in same frequency as input data
        - r_squared: Float, R-squared of the regression
        - valid_model: Boolean, whether the model is statistically valid
        - residual_normality: Boolean, whether the regression residuals are normally distributed
        - hurst_exponent: Float, Hurst exponent of the series (values < 0.5 indicate mean reversion)
    """
    # Validate input and handle empty series
    if not isinstance(spread, pd.Series):
        spread = pd.Series(spread)
    
    spread = spread.dropna()
    
    if len(spread) < 10:
        warnings.warn("Series too short for reliable half-life calculation")
        return {
            'half_life': np.inf,
            'r_squared': 0,
            'valid_model': False,
            'residual_normality': False,
            'hurst_exponent': None
        }
    
    # Calculate returns (price differences)
    lag_spread = spread.shift(1)
    delta_spread = spread - lag_spread
    
    # Remove NaN values
    spread_lag = lag_spread.dropna()
    delta_spread = delta_spread.dropna()
    
    # Make sure vectors are aligned
    spread_lag = spread_lag.iloc[delta_spread.index.get_indexer(spread_lag.index)]
    
    # Regression to estimate mean reversion
    spread_lag = sm.add_constant(spread_lag)
    
    try:
        model = sm.OLS(delta_spread, spread_lag).fit()
        
        # Extract coefficient and statistical measures
        beta = model.params[1]
        r_squared = model.rsquared
        p_value = model.pvalues[1]
        
        # Calculate residuals and test for normality
        residuals = model.resid
        
        # Check if model is valid
        valid_model = (p_value < 0.05) and (beta < 0)
        
        # Test residuals for normality if we have enough data points
        if len(residuals) >= 8:  # Minimum size for Shapiro-Wilk test
            from scipy import stats
            shapiro_test = stats.shapiro(residuals)
            residual_normality = shapiro_test[1] > 0.05  # p-value > 0.05 suggests normality
        else:
            residual_normality = None
        
        # Calculate Hurst exponent to test for mean reversion
        hurst_exponent = calculate_hurst_exponent(spread)
        
        # Calculate half-life with improved robustness
        if beta >= 0:
            # Not mean-reverting
            half_life = np.inf
        else:
            # Standard half-life calculation
            half_life = -np.log(2) / beta
            
            # Cap half-life at a reasonable value
            if half_life > max_half_life or not np.isfinite(half_life):
                half_life = max_half_life
        
        # Return complete results dictionary
        return {
            'half_life': float(half_life),
            'r_squared': float(r_squared),
            'valid_model': valid_model,
            'residual_normality': residual_normality,
            'hurst_exponent': hurst_exponent
        }
        
    except Exception as e:
        warnings.warn(f"Error in half-life calculation: {str(e)}")
        return {
            'half_life': np.inf,
            'r_squared': 0,
            'valid_model': False,
            'residual_normality': False,
            'hurst_exponent': None,
            'error': str(e)
        }

def calculate_hurst_exponent(time_series, max_lag=100):
    """
    Calculate the Hurst exponent of a time series.
    
    The Hurst exponent measures the long-term memory of a time series.
    H < 0.5 indicates mean reversion
    H = 0.5 indicates a random walk
    H > 0.5 indicates trend reinforcement
    
    Parameters:
    -----------
    time_series : pandas.Series or array-like
        Time series to analyze
    max_lag : int
        Maximum lag to use in calculation
        
    Returns:
    --------
    float
        Hurst exponent
    """
    if not isinstance(time_series, pd.Series):
        time_series = pd.Series(time_series)
    
    time_series = time_series.dropna()
    
    # If series is too short, return None
    if len(time_series) < 20:
        return None
    
    # Limit max_lag based on series length
    max_lag = min(max_lag, len(time_series) // 4)
    
    # Set minimum lag to 2
    lags = range(2, max_lag)
    
    # Calculate variance of differenced series for each lag
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:].values, time_series[:-lag].values))) for lag in lags]
    
    # Avoid division by zero or log(0)
    tau = [x for x in tau if x > 0]
    lags = lags[:len(tau)]
    
    if not tau:
        return None
    
    # Calculate Hurst as slope of log-log regression
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Hurst exponent is the slope
    hurst = m[0]
    
    return float(hurst)

def test_cointegration(price_series1, price_series2, window=60, test_type='both', 
                       train_test_split=0.7, use_log_prices=True, out_of_sample=True):
    """
    Test for cointegration between two price series using both Engle-Granger and Johansen methods
    with enhanced out-of-sample validation.
    
    Parameters:
    -----------
    price_series1 : pandas.Series
        Series of prices for first instrument
    price_series2 : pandas.Series
        Series of prices for second instrument
    window : int
        Rolling window size in days
    test_type : str
        'engle-granger', 'johansen', or 'both'
    train_test_split : float
        Proportion of data to use for training vs validation
    use_log_prices : bool
        Whether to use log prices for better stationarity properties
    out_of_sample : bool
        Whether to perform out-of-sample validation (default: True)
    
    Returns:
    --------
    dict
        Dictionary with cointegration test results, hedge ratio, validation results,
        and stability metrics
    """
    result = {}
    
    # Ensure series have same length by taking the intersection
    common_index = price_series1.index.intersection(price_series2.index)
    price_series1 = price_series1.loc[common_index]
    price_series2 = price_series2.loc[common_index]
    
    # Check if we have enough data
    if len(price_series1) < window:
        warnings.warn(f"Not enough data for window size {window}. Required: {window}, Available: {len(price_series1)}")
        return None
    
    # Split data into training and validation sets
    split_idx = int(len(price_series1) * train_test_split)
    train_price1 = price_series1.iloc[:split_idx]
    train_price2 = price_series2.iloc[:split_idx]
    valid_price1 = price_series1.iloc[split_idx:]
    valid_price2 = price_series2.iloc[split_idx:]
    
    # Skip if split results in too little data
    if len(train_price1) < window or (out_of_sample and len(valid_price1) < 20):
        warnings.warn(f"Insufficient data after split. Training: {len(train_price1)}, Validation: {len(valid_price1)}")
        return None
    
    # Perform Engle-Granger test if requested
    if test_type in ['engle-granger', 'both']:
        eg_result = engle_granger_test(train_price1, train_price2, use_log_prices=use_log_prices)
        result['engle_granger'] = eg_result
        
        # Store hedge ratio and half-life from Engle-Granger test
        result['hedge_ratio'] = eg_result['hedge_ratio']
        result['half_life'] = eg_result['half_life']
        result['half_life_metrics'] = eg_result['half_life_metrics']
    
    # Perform Johansen test if requested
    if test_type in ['johansen', 'both']:
        # Prepare data for Johansen test
        if use_log_prices:
            train_data = pd.DataFrame({
                'series1': np.log(train_price1),
                'series2': np.log(train_price2)
            })
        else:
            train_data = pd.DataFrame({
                'series1': train_price1,
                'series2': train_price2
            })
        
        j_result = johansen_test(train_data)
        result['johansen'] = j_result
    
    # Create combined result
    if test_type == 'both':
        # A pair is considered cointegrated if either test indicates cointegration
        is_cointegrated = (
            result['engle_granger']['is_cointegrated'] or
            result['johansen']['n_cointegrating_relations'] > 0
        )
    elif test_type == 'engle-granger':
        is_cointegrated = result['engle_granger']['is_cointegrated']
    else:  # johansen
        is_cointegrated = result['johansen']['n_cointegrating_relations'] > 0
    
    result['combined_result'] = {
        'is_cointegrated': is_cointegrated,
        'test_type': test_type
    }
    
    # Enhanced out-of-sample validation
    if out_of_sample and len(valid_price1) > 0:
        oos_results = {}
        
        # Use the hedge ratio from training for out-of-sample testing
        if test_type in ['engle-granger', 'both']:
            beta = result['hedge_ratio']
            alpha = result['engle_granger']['constant']
            
            # Calculate spread in validation set
            if use_log_prices:
                valid_log1 = np.log(valid_price1)
                valid_log2 = np.log(valid_price2)
                residuals = valid_log1 - (alpha + beta * valid_log2)
            else:
                residuals = valid_price1 - (alpha + beta * valid_price2)
            
            # Full ADF test on residuals
            adf_result = adfuller(residuals, regression='c')
            
            # Calculate half-life for the validation period
            half_life_result = calculate_half_life(residuals)
            
            # Calculate additional statistics for validation
            oos_mean = residuals.mean()
            oos_std = residuals.std()
            oos_zscore = (residuals - oos_mean) / oos_std
            
            # Check for stability of spread
            mean_deviation = abs(oos_mean) / oos_std  # Normalized mean
            
            # Test normality of spread 
            if len(residuals) >= 8:  # Minimum sample size for meaningful normality test
                from scipy import stats
                normality_test = stats.shapiro(residuals)
                is_normal = normality_test[1] > 0.05  # p-value > 0.05 suggests normality
            else:
                normality_test = (None, None)
                is_normal = None
            
            # Calculate stability metrics
            if len(residuals) >= 30:  # Minimum for meaningful stability metrics
                # Split validation period in half
                mid_idx = len(residuals) // 2
                first_half = residuals[:mid_idx]
                second_half = residuals[mid_idx:]
                
                # Calculate statistics for each half
                first_mean = first_half.mean()
                second_mean = second_half.mean()
                first_std = first_half.std() 
                second_std = second_half.std()
                
                # Calculate ratios for stability comparison
                mean_stability = 1 - min(1, abs(first_mean - second_mean) / oos_std)  # 1 = perfectly stable
                std_stability = 1 - min(1, abs(first_std - second_std) / ((first_std + second_std) / 2))  # 1 = perfectly stable
                
                # Calculate stationarity tests for each half
                adf_first = adfuller(first_half, regression='c')
                adf_second = adfuller(second_half, regression='c')
                
                # Check consistency of stationarity
                stationary_first = adf_first[1] < 0.05
                stationary_second = adf_second[1] < 0.05
                consistent_stationarity = stationary_first == stationary_second
            else:
                mean_stability = None
                std_stability = None
                consistent_stationarity = None
            
            # Extreme deviation check
            max_abs_zscore = abs(oos_zscore).max()
            has_extreme_deviation = max_abs_zscore > 4.0  # z-score > 4 is suspicious for a mean-reverting series
            
            # Create enhanced out-of-sample results
            oos_results = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': {
                    '1%': adf_result[4]['1%'],
                    '5%': adf_result[4]['5%'],
                    '10%': adf_result[4]['10%']
                },
                'is_cointegrated_oos': adf_result[1] < 0.05,  # p-value < 0.05
                'half_life_oos': half_life_result['half_life'],
                'half_life_metrics': {
                    'r_squared': half_life_result['r_squared'],
                    'valid_model': half_life_result['valid_model'],
                    'residual_normality': half_life_result['residual_normality'],
                    'hurst_exponent': half_life_result['hurst_exponent']
                },
                'mean': float(oos_mean),
                'std': float(oos_std),
                'mean_deviation_normalized': float(mean_deviation),
                'max_abs_zscore': float(max_abs_zscore),
                'has_extreme_deviation': has_extreme_deviation,
                'normality_test': {
                    'statistic': float(normality_test[0]) if normality_test[0] is not None else None,
                    'p_value': float(normality_test[1]) if normality_test[1] is not None else None,
                    'is_normal': is_normal
                }
            }
            
            # Add stability metrics if available
            if mean_stability is not None:
                oos_results.update({
                    'stability': {
                        'mean_stability': float(mean_stability),
                        'std_stability': float(std_stability),
                        'consistent_stationarity': consistent_stationarity
                    }
                })
            
            # Combine training and validation results
            result['out_of_sample'] = oos_results
            
            # Add training-validation comparison metrics
            result['comparison'] = {
                'consistent_cointegration': result['combined_result']['is_cointegrated'] == oos_results['is_cointegrated_oos'],
                'half_life_ratio': float(half_life_result['half_life'] / result['half_life']) if result['half_life'] > 0 else float('inf'),
                'r_squared_ratio': float(half_life_result['r_squared'] / result['half_life_metrics']['r_squared']) if result['half_life_metrics']['r_squared'] > 0 else float('inf'),
                'overall_stability': compute_overall_stability(result, oos_results)
            }
    
    return result

def compute_overall_stability(training_results, validation_results):
    """
    Compute an overall stability score that considers multiple factors of stability 
    between training and validation periods.
    
    Parameters:
    -----------
    training_results : dict
        Results from the training period
    validation_results : dict
        Results from the validation period
        
    Returns:
    --------
    float
        Stability score between 0 and 1, where 1 is perfectly stable
    """
    stability_score = 0.0
    factors = 0
    
    # Factor 1: Consistent cointegration finding (highest weight)
    if training_results['combined_result']['is_cointegrated'] == validation_results['is_cointegrated_oos']:
        stability_score += 0.5
    factors += 0.5
    
    # Factor 2: Half-life stability
    # Ideal: validation half-life is between 0.5x and 2x the training half-life
    if training_results['half_life'] > 0 and validation_results['half_life_oos'] > 0:
        half_life_ratio = validation_results['half_life_oos'] / training_results['half_life']
        if 0.5 <= half_life_ratio <= 2.0:
            stability_score += 0.2
        elif 0.25 <= half_life_ratio <= 4.0:
            stability_score += 0.1
        factors += 0.2
    
    # Factor 3: No extreme deviations in validation
    if not validation_results.get('has_extreme_deviation', False):
        stability_score += 0.15
    factors += 0.15
    
    # Factor 4: Validation period stability metrics if available
    if 'stability' in validation_results:
        stability = validation_results['stability']
        
        # Mean stability
        if stability.get('mean_stability', 0) > 0.7:
            stability_score += 0.075
        
        # Std stability
        if stability.get('std_stability', 0) > 0.7:
            stability_score += 0.075
            
        factors += 0.15
    
    # Normalize the score based on available factors
    if factors > 0:
        stability_score = stability_score / factors
    else:
        stability_score = 0.0
    
    return float(stability_score)

def test_pairs_universe(price_data, pairs_to_test=None, min_correlation=0.7, 
                        max_half_life=20, min_half_life=1, use_log_prices=True):
    """
    Test multiple pairs for cointegration and filter based on correlation and half-life.
    
    Parameters:
    -----------
    price_data : dict or pd.DataFrame
        Dictionary of price series (ticker -> Series) or DataFrame with tickers as columns
    pairs_to_test : list of tuples, optional
        List of ticker pairs to test, if None, test all combinations
    min_correlation : float
        Minimum correlation threshold for considering a pair
    max_half_life : float
        Maximum half-life for mean reversion (in days)
    min_half_life : float
        Minimum half-life for mean reversion (in days)
    use_log_prices : bool
        Whether to use log prices for correlation check
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cointegration test results for filtered pairs
    """
    # Convert DataFrame to dict of Series if needed
    if isinstance(price_data, pd.DataFrame):
        price_dict = {col: price_data[col] for col in price_data.columns}
    else:
        price_dict = price_data
    
    tickers = list(price_dict.keys())
    
    # Generate all pairs if not provided
    if pairs_to_test is None:
        pairs_to_test = [(tickers[i], tickers[j]) 
                         for i in range(len(tickers)) 
                         for j in range(i+1, len(tickers))]
    
    results = []
    
    for ticker1, ticker2 in pairs_to_test:
        if ticker1 not in price_dict or ticker2 not in price_dict:
            continue
        
        series1 = price_dict[ticker1]
        series2 = price_dict[ticker2]
        
        # Ensure series have same length
        common_index = series1.index.intersection(series2.index)
        if len(common_index) < 60:  # Minimum data requirement
            continue
            
        s1 = series1.loc[common_index]
        s2 = series2.loc[common_index]
        
        # Calculate correlation
        if use_log_prices:
            correlation = np.log(s1).corr(np.log(s2))
        else:
            correlation = s1.corr(s2)
        
        # Filter on correlation first
        if correlation < min_correlation:
            continue
        
        # Run cointegration test
        coint_result = test_cointegration(s1, s2, use_log_prices=use_log_prices)
        
        if coint_result is None or coint_result['overall'] is None:
            continue
        
        # Filter on half-life
        hl_train = coint_result['overall']['half_life_training']
        hl_valid = coint_result['overall']['half_life_validation']
        
        if not (min_half_life <= hl_train <= max_half_life and 
                min_half_life <= hl_valid <= max_half_life):
            continue
            
        # Check if cointegrated in both training and validation
        if not (coint_result['overall']['is_cointegrated_training'] and 
                coint_result['overall']['is_cointegrated_validation']):
            continue
        
        # If passed all filters, add to results
        result_item = {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'correlation': correlation,
            'hedge_ratio': coint_result['training']['hedge_ratio'].iloc[-1] if 'hedge_ratio' in coint_result['training'] else None,
            'half_life_train': hl_train,
            'half_life_valid': hl_valid,
            'p_value_train': coint_result['training']['p_value'].iloc[-1] if 'p_value' in coint_result['training'] else None,
            'p_value_valid': coint_result['validation']['valid_p_value']
        }
        
        results.append(result_item)
    
    return pd.DataFrame(results)


def rolling_cointegration(price_series1, price_series2, window=60, step=1, use_log_prices=True, 
                        window_sizes=None, significance_level=0.05):
    """
    Perform rolling window cointegration tests to analyze relationship stability over time.
    Enhanced with validation and statistical significance testing across multiple window sizes.
    
    Parameters:
    -----------
    price_series1 : pandas.Series
        Series of prices for first instrument
    price_series2 : pandas.Series
        Series of prices for second instrument
    window : int
        Rolling window size (default window if multiple are tested)
    step : int
        Number of periods to move forward in each step
    use_log_prices : bool
        Whether to use log prices
    window_sizes : list, optional
        List of window sizes to test for stability assessment
    significance_level : float
        P-value threshold for statistical significance (default: 0.05)
        
    Returns:
    --------
    dict
        Dictionary with rolling cointegration results and stability metrics
    """
    # Ensure series have same length by taking the intersection
    common_index = price_series1.index.intersection(price_series2.index)
    price_series1 = price_series1.loc[common_index]
    price_series2 = price_series2.loc[common_index]
    
    # Check if we have enough data
    if len(price_series1) < window:
        raise ValueError(f"Not enough data for window size {window}. Required: {window}, Available: {len(price_series1)}")
    
    # Convert to log prices if requested
    if use_log_prices:
        series1 = np.log(price_series1)
        series2 = np.log(price_series2)
    else:
        series1 = price_series1.copy()
        series2 = price_series2.copy()
    
    # Process primary window size
    results = []
    
    for i in range(0, len(series1) - window, step):
        end_idx = i + window
        s1_window = series1.iloc[i:end_idx]
        s2_window = series2.iloc[i:end_idx]
        
        # Use the Engle-Granger test function for consistency
        eg_test = engle_granger_test(
            pd.Series(s1_window.values, index=s1_window.index),
            pd.Series(s2_window.values, index=s2_window.index),
            use_log_prices=False  # Already converted if needed
        )
        
        # Record results with enhanced metrics
        result = {
            'end_date': series1.index[end_idx-1],
            'hedge_ratio': eg_test['hedge_ratio'],
            'half_life': eg_test['half_life'],
            'adf_statistic': eg_test['adf_statistic'],
            'p_value': eg_test['p_value'],
            'is_cointegrated': eg_test['is_cointegrated'],
            'critical_value_5pct': eg_test['critical_values']['5%'] if '5%' in eg_test['critical_values'] else None,
            'hurst_exponent': eg_test['half_life_metrics']['hurst_exponent'],
            'r_squared': eg_test['half_life_metrics']['r_squared'],
            'start_date': series1.index[i]
        }
        
        results.append(result)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results).set_index('end_date')
    
    # If we need to test multiple window sizes
    window_results = {}
    
    if window_sizes is not None:
        window_results[window] = results_df  # Store primary results
        
        for w in window_sizes:
            if w == window:
                continue  # Already computed
                
            if len(series1) < w:
                warnings.warn(f"Not enough data for window size {w}. Skipping.")
                continue
                
            # Process additional window size
            w_results = []
            
            for i in range(0, len(series1) - w, step):
                end_idx = i + w
                s1_window = series1.iloc[i:end_idx]
                s2_window = series2.iloc[i:end_idx]
                
                # Use the Engle-Granger test function
                eg_test = engle_granger_test(
                    pd.Series(s1_window.values, index=s1_window.index),
                    pd.Series(s2_window.values, index=s2_window.index),
                    use_log_prices=False  # Already converted if needed
                )
                
                # Record results
                w_result = {
                    'end_date': series1.index[end_idx-1],
                    'hedge_ratio': eg_test['hedge_ratio'],
                    'half_life': eg_test['half_life'],
                    'adf_statistic': eg_test['adf_statistic'],
                    'p_value': eg_test['p_value'],
                    'is_cointegrated': eg_test['is_cointegrated'],
                    'hurst_exponent': eg_test['half_life_metrics']['hurst_exponent'],
                    'r_squared': eg_test['half_life_metrics']['r_squared'],
                    'start_date': series1.index[i]
                }
                
                w_results.append(w_result)
            
            # Store results for this window size
            window_results[w] = pd.DataFrame(w_results).set_index('end_date')
        
        # Calculate stability metrics across window sizes
        stability_metrics = {
            'window_consistency': _calculate_window_consistency(window_results, significance_level),
            'hedge_ratio_stability': _calculate_hedge_ratio_stability(window_results),
            'cointegration_frequency': _calculate_cointegration_frequency(results_df),
            'mean_half_life': results_df['half_life'].mean(),
            'mean_hurst_exponent': results_df['hurst_exponent'].dropna().mean() if 'hurst_exponent' in results_df else None,
            'p_value_volatility': results_df['p_value'].std(),
            'primary_window_results': results_df
        }
        
        if window_sizes:
            stability_metrics['window_results'] = window_results
        
        return stability_metrics
    
    # Simple single window result
    stability_metrics = {
        'cointegration_frequency': _calculate_cointegration_frequency(results_df),
        'mean_half_life': results_df['half_life'].mean(),
        'mean_p_value': results_df['p_value'].mean(),
        'mean_hurst_exponent': results_df['hurst_exponent'].dropna().mean() if 'hurst_exponent' in results_df else None,
        'mean_r_squared': results_df['r_squared'].mean() if 'r_squared' in results_df else None,
        'p_value_volatility': results_df['p_value'].std(),
        'hedge_ratio_volatility': results_df['hedge_ratio'].std() / results_df['hedge_ratio'].mean(),
        'primary_window_results': results_df
    }
    
    return stability_metrics

def _calculate_window_consistency(window_results, significance_level=0.05):
    """
    Calculate the consistency of cointegration results across different window sizes.
    
    Parameters:
    -----------
    window_results : dict
        Dictionary of DataFrames with results for each window size
    significance_level : float
        P-value threshold for statistical significance
        
    Returns:
    --------
    float
        Consistency score (0-1) where 1 is perfectly consistent
    """
    if len(window_results) <= 1:
        return 1.0  # Only one window size, so perfectly consistent
    
    # Extract cointegration results for each window
    is_cointegrated = {}
    
    for window, df in window_results.items():
        is_cointegrated[window] = df['is_cointegrated']
    
    # Create a DataFrame with all results
    consistency_df = pd.DataFrame(is_cointegrated)
    
    # Calculate agreement percentage (how often all windows agree)
    if consistency_df.empty:
        return 0.0
        
    # Count rows where all values are the same (all True or all False)
    agreement_count = (consistency_df.std(axis=1) == 0).sum()
    
    # Calculate consistency score
    consistency = agreement_count / len(consistency_df)
    
    return consistency

def _calculate_hedge_ratio_stability(window_results):
    """
    Calculate the stability of hedge ratios across different window sizes.
    
    Parameters:
    -----------
    window_results : dict
        Dictionary of DataFrames with results for each window size
        
    Returns:
    --------
    float
        Stability score (0-1) where 1 is perfectly stable
    """
    if len(window_results) <= 1:
        return 1.0  # Only one window size, so perfectly stable
    
    # Extract mean hedge ratios for each window
    mean_hedge_ratios = [df['hedge_ratio'].mean() for df in window_results.values()]
    
    # Calculate coefficient of variation (lower is better)
    mean = np.mean(mean_hedge_ratios)
    std = np.std(mean_hedge_ratios)
    
    if mean == 0:
        return 0.0  # Avoid division by zero
        
    cv = std / abs(mean)
    
    # Convert to stability score (1 - CV, bounded at 0)
    stability = max(0, 1 - cv)
    
    return stability

def _calculate_cointegration_frequency(results_df):
    """
    Calculate the frequency of cointegration in the rolling windows.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with rolling window results
        
    Returns:
    --------
    float
        Frequency of cointegration (0-1)
    """
    if results_df.empty:
        return 0.0
        
    return results_df['is_cointegrated'].mean()

def engle_granger_test(price_series1, price_series2, use_log_prices=True):
    """
    Perform Engle-Granger two-step cointegration test on a pair of price series.
    
    Parameters:
    -----------
    price_series1 : pandas.Series
        Series of prices for first instrument
    price_series2 : pandas.Series
        Series of prices for second instrument
    use_log_prices : bool
        Whether to use log prices for better stationarity properties
        
    Returns:
    --------
    dict
        Dictionary with ADF statistic, p-value, critical values, cointegration flag, and hedge ratio
    """
    # Ensure series have same length by taking the intersection
    common_index = price_series1.index.intersection(price_series2.index)
    series1 = price_series1.loc[common_index]
    series2 = price_series2.loc[common_index]
    
    # Convert to log prices if requested
    if use_log_prices:
        series1 = np.log(series1)
        series2 = np.log(series2)
    
    # Step 1: Run linear regression to get beta (hedge ratio)
    X = sm.add_constant(series2)
    model = sm.OLS(series1, X).fit()
    beta = model.params[1]  # Hedge ratio
    alpha = model.params[0]  # Constant term
    
    # Step 2: Calculate residuals (spread)
    residuals = series1 - (alpha + beta * series2)
    
    # Step 3: Test for stationarity of residuals using ADF test
    adf_result = adfuller(residuals, regression='c')
    
    # Extract key statistics
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    is_cointegrated = p_value < 0.05
    
    # Calculate half-life (now returns a dictionary)
    half_life_result = calculate_half_life(residuals)
    
    # Return results as dictionary
    result = {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_cointegrated': is_cointegrated,
        'hedge_ratio': beta,
        'constant': alpha,
        'half_life': half_life_result['half_life'],
        'half_life_metrics': {
            'r_squared': half_life_result['r_squared'],
            'valid_model': half_life_result['valid_model'],
            'residual_normality': half_life_result['residual_normality'],
            'hurst_exponent': half_life_result['hurst_exponent']
        }
    }
    
    return result

def johansen_test(data, deterministic=0, k_ar_diff=1):
    """
    Perform Johansen cointegration test on a multivariate time series.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with columns as different time series to test for cointegration
    deterministic : int
        Deterministic term inclusion:
        0: no deterministic terms (default)
        1: constant term
        2: constant and linear trend
    k_ar_diff : int
        Number of lagged differences in the VECM
        
    Returns:
    --------
    dict
        Dictionary with trace statistics, critical values, eigenvalues, and number of cointegrating relations
    """
    # Validate input
    if not isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            raise ValueError("At least two series are required for Johansen test")
        else:
            data = pd.DataFrame(data)
    
    # Check data is sufficient
    if data.shape[1] < 2:
        raise ValueError("At least two series are required for Johansen test")
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Run Johansen test
    try:
        johansen_results = coint_johansen(data, det_order=deterministic, k_ar_diff=k_ar_diff)
        
        # Extract trace statistics, critical values and eigenvalues
        trace_statistics = johansen_results.lr1
        critical_values = johansen_results.cvt
        eigenvalues = johansen_results.eig
        
        # Determine number of cointegrating relations
        n_cointegrating_relations = sum(trace_statistics > critical_values[:, 1])
        
        # Create result dictionary
        result = {
            'trace_statistic': trace_statistics.tolist(),
            'critical_values': {
                '90%': critical_values[:, 0].tolist(),
                '95%': critical_values[:, 1].tolist(),
                '99%': critical_values[:, 2].tolist()
            },
            'eigenvalues': eigenvalues.tolist(),
            'n_cointegrating_relations': int(n_cointegrating_relations)
        }
        
        return result
    except Exception as e:
        # Handle potential errors with Johansen test
        warnings.warn(f"Error in Johansen test: {str(e)}")
        return {
            'trace_statistic': None,
            'critical_values': None,
            'eigenvalues': None,
            'n_cointegrating_relations': 0,
            'error': str(e)
        } 