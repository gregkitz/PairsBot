"""
Validation utilities for cointegration statistical methods.

This module contains functions to validate statistical methods against 
known datasets, benchmarks, and theoretical properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from scipy import stats
import os
import json

# Import our implementations
from src.cointegration.statistical_methods import (
    johansen_test,
    engle_granger_test,
    calculate_half_life
)


def generate_cointegrated_series(
    n_samples: int = 1000,
    hedge_ratio: float = 0.5,
    noise_level: float = 0.2,
    seed: Optional[int] = None,
    mean_reversion_strength: float = 0.05,
    include_drift: bool = False
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Generate synthetic cointegrated price series for testing.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of data points to generate
    
    hedge_ratio : float, default=0.5
        The true hedge ratio in the cointegrating relationship
    
    noise_level : float, default=0.2
        Standard deviation of the noise in the cointegrating relationship
    
    seed : Optional[int], default=None
        Random seed for reproducibility
    
    mean_reversion_strength : float, default=0.05
        Strength of mean reversion in the spread (higher = faster reversion)
    
    include_drift : bool, default=False
        Whether to include drift in the random walks
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, Dict]
        - Series 1 (random walk)
        - Series 2 (cointegrated with series 1)
        - Dictionary with the true parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate first random walk
    drift = 0.001 if include_drift else 0
    returns = np.random.normal(drift, 1, n_samples)
    series1 = 100 + np.cumsum(returns)
    
    # Set true parameters
    intercept = 50
    
    # Generate cointegrated series
    # Create spread with mean-reverting properties
    spread = np.zeros(n_samples)
    spread[0] = np.random.normal(0, noise_level)
    
    for i in range(1, n_samples):
        # Ornstein-Uhlenbeck process
        spread[i] = spread[i-1] * (1 - mean_reversion_strength) + np.random.normal(0, noise_level)
    
    # Calculate second series based on cointegrating relationship
    series2 = hedge_ratio * series1 + intercept + spread
    
    # Create pandas Series
    s1 = pd.Series(series1, index=dates, name='price1')
    s2 = pd.Series(series2, index=dates, name='price2')
    
    # True parameters
    true_params = {
        'hedge_ratio': hedge_ratio,
        'intercept': intercept,
        'noise_level': noise_level,
        'mean_reversion_strength': mean_reversion_strength,
        'theoretical_half_life': np.log(2) / mean_reversion_strength if mean_reversion_strength > 0 else np.inf,
        'is_cointegrated': True
    }
    
    return s1, s2, true_params


def generate_non_cointegrated_series(
    n_samples: int = 1000,
    correlation: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Generate synthetic non-cointegrated random walk price series.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of data points to generate
    
    correlation : float, default=0.5
        Correlation between the innovations of the two series
    
    seed : Optional[int], default=None
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, Dict]
        - Series 1 (random walk)
        - Series 2 (random walk, potentially correlated with series 1)
        - Dictionary with the true parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate correlated random walks
    # Create correlated innovations (if correlation is not 0)
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    
    innovations = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Create random walks
    series1 = 100 + np.cumsum(innovations[:, 0])
    series2 = 100 + np.cumsum(innovations[:, 1])
    
    # Create pandas Series
    s1 = pd.Series(series1, index=dates, name='price1')
    s2 = pd.Series(series2, index=dates, name='price2')
    
    # True parameters
    true_params = {
        'correlation': correlation,
        'is_cointegrated': False
    }
    
    return s1, s2, true_params


def validate_engle_granger(
    n_trials: int = 50,
    sample_sizes: List[int] = [100, 500, 1000],
    hedge_ratios: List[float] = [0.5, 1.0, 2.0],
    noise_levels: List[float] = [0.1, 0.5, 1.0],
    mean_reversion_strengths: List[float] = [0.01, 0.05, 0.1],
    save_results: bool = True,
    output_dir: str = 'validation_results'
) -> Dict:
    """
    Validate the Engle-Granger test implementation against synthetic data
    with known cointegration properties.
    
    Parameters
    ----------
    n_trials : int, default=50
        Number of trials for each parameter combination
    
    sample_sizes : List[int], default=[100, 500, 1000]
        Sample sizes to test
    
    hedge_ratios : List[float], default=[0.5, 1.0, 2.0]
        Hedge ratios to test
    
    noise_levels : List[float], default=[0.1, 0.5, 1.0]
        Noise levels to test
    
    mean_reversion_strengths : List[float], default=[0.01, 0.05, 0.1]
        Mean reversion strengths to test
    
    save_results : bool, default=True
        Whether to save the validation results to disk
    
    output_dir : str, default='validation_results'
        Directory to save validation results
        
    Returns
    -------
    Dict
        Dictionary containing validation metrics and results
    """
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        'cointegrated': {
            'n_trials': n_trials,
            'detection_rate': {},
            'hedge_ratio_error': {},
            'half_life_error': {}
        },
        'non_cointegrated': {
            'n_trials': n_trials,
            'false_positive_rate': {}
        }
    }
    
    # Test with cointegrated series
    for n in sample_sizes:
        for hr in hedge_ratios:
            for nl in noise_levels:
                for mrs in mean_reversion_strengths:
                    key = f"n={n}_hr={hr}_nl={nl}_mrs={mrs}"
                    
                    detection_count = 0
                    hedge_ratio_errors = []
                    half_life_errors = []
                    
                    for trial in range(n_trials):
                        # Generate cointegrated series
                        s1, s2, true_params = generate_cointegrated_series(
                            n_samples=n,
                            hedge_ratio=hr,
                            noise_level=nl,
                            seed=trial,  # Different seed for each trial
                            mean_reversion_strength=mrs
                        )
                        
                        # Test cointegration
                        test_result = engle_granger_test(s2, s1)
                        
                        # Record results
                        if test_result['is_cointegrated']:
                            detection_count += 1
                            hedge_ratio_errors.append(
                                abs(test_result['hedge_ratio'] - true_params['hedge_ratio']) / true_params['hedge_ratio']
                            )
                            
                            if test_result['half_life'] is not None and np.isfinite(test_result['half_life']):
                                if np.isfinite(true_params['theoretical_half_life']):
                                    half_life_errors.append(
                                        abs(test_result['half_life'] - true_params['theoretical_half_life']) / 
                                        true_params['theoretical_half_life']
                                    )
                    
                    # Calculate metrics
                    detection_rate = detection_count / n_trials
                    avg_hedge_ratio_error = np.mean(hedge_ratio_errors) if hedge_ratio_errors else np.nan
                    avg_half_life_error = np.mean(half_life_errors) if half_life_errors else np.nan
                    
                    # Record results
                    results['cointegrated']['detection_rate'][key] = detection_rate
                    results['cointegrated']['hedge_ratio_error'][key] = avg_hedge_ratio_error
                    results['cointegrated']['half_life_error'][key] = avg_half_life_error
    
    # Test with non-cointegrated series
    for n in sample_sizes:
        for corr in [0.0, 0.5, 0.9]:
            key = f"n={n}_corr={corr}"
            
            false_positive_count = 0
            
            for trial in range(n_trials):
                # Generate non-cointegrated series
                s1, s2, _ = generate_non_cointegrated_series(
                    n_samples=n,
                    correlation=corr,
                    seed=trial
                )
                
                # Test cointegration
                test_result = engle_granger_test(s2, s1)
                
                # Record results
                if test_result['is_cointegrated']:
                    false_positive_count += 1
            
            # Calculate metrics
            false_positive_rate = false_positive_count / n_trials
            
            # Record results
            results['non_cointegrated']['false_positive_rate'][key] = false_positive_rate
    
    # Save results
    if save_results:
        with open(os.path.join(output_dir, 'engle_granger_validation.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def validate_johansen(
    n_trials: int = 50,
    sample_sizes: List[int] = [100, 500, 1000],
    n_variables: List[int] = [2, 3, 4],
    n_cointegrating_relations: List[int] = [1, 2],
    save_results: bool = True,
    output_dir: str = 'validation_results'
) -> Dict:
    """
    Validate the Johansen test implementation against synthetic data
    with known cointegration properties.
    
    Parameters
    ----------
    n_trials : int, default=50
        Number of trials for each parameter combination
    
    sample_sizes : List[int], default=[100, 500, 1000]
        Sample sizes to test
    
    n_variables : List[int], default=[2, 3, 4]
        Number of variables in the system
    
    n_cointegrating_relations : List[int], default=[1, 2]
        Number of cointegrating relationships to test
    
    save_results : bool, default=True
        Whether to save the validation results to disk
    
    output_dir : str, default='validation_results'
        Directory to save validation results
        
    Returns
    -------
    Dict
        Dictionary containing validation metrics and results
    """
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        'n_trials': n_trials,
        'detection_rate': {},
        'relation_count_accuracy': {},
    }
    
    for n in sample_sizes:
        for nvar in n_variables:
            # Can't have more cointegrating relations than variables-1
            max_rels = nvar - 1
            for ncr in [r for r in n_cointegrating_relations if r <= max_rels]:
                key = f"n={n}_nvar={nvar}_ncr={ncr}"
                
                detection_count = 0
                exact_relation_count = 0
                
                for trial in range(n_trials):
                    # Generate multivariate system with known cointegration properties
                    data = generate_cointegrated_system(
                        n_samples=n,
                        n_variables=nvar,
                        n_cointegrating_relations=ncr,
                        seed=trial
                    )
                    
                    # Test cointegration
                    test_result = johansen_test(data)
                    
                    # Record results
                    if test_result['n_cointegrating_relations_trace'] > 0:
                        detection_count += 1
                    
                    if test_result['n_cointegrating_relations_trace'] == ncr:
                        exact_relation_count += 1
                
                # Calculate metrics
                detection_rate = detection_count / n_trials
                relation_accuracy = exact_relation_count / n_trials
                
                # Record results
                results['detection_rate'][key] = detection_rate
                results['relation_count_accuracy'][key] = relation_accuracy
    
    # Save results
    if save_results:
        with open(os.path.join(output_dir, 'johansen_validation.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def generate_cointegrated_system(
    n_samples: int = 1000,
    n_variables: int = 3,
    n_cointegrating_relations: int = 1,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic multivariate system with known cointegration properties.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of data points to generate
    
    n_variables : int, default=3
        Number of variables in the system
    
    n_cointegrating_relations : int, default=1
        Number of cointegrating relationships
    
    seed : Optional[int], default=None
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the multivariate system
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_cointegrating_relations >= n_variables:
        raise ValueError("Number of cointegrating relations must be less than number of variables")
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate random walks for the common trends
    n_trends = n_variables - n_cointegrating_relations
    trends = np.zeros((n_samples, n_trends))
    
    for i in range(n_trends):
        trends[:, i] = np.cumsum(np.random.normal(0, 1, n_samples))
    
    # Generate loading matrix (how trends impact each variable)
    # This matrix defines how much each trend affects each variable
    A = np.random.normal(0, 1, (n_variables, n_trends))
    
    # Generate data matrix
    X = np.zeros((n_samples, n_variables))
    for i in range(n_variables):
        # Each variable is a linear combination of trends plus noise
        X[:, i] = np.sum(A[i, :].reshape(1, -1) * trends, axis=1) + np.random.normal(0, 0.5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(X, index=dates, columns=[f'var{i+1}' for i in range(n_variables)])
    
    return df


def compare_with_external_library(
    n_samples: int = 1000,
    n_trials: int = 10,
    external_library: str = 'statsmodels'
) -> Dict:
    """
    Compare our implementation with an external statistical library.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of data points to generate
    
    n_trials : int, default=10
        Number of trials
    
    external_library : str, default='statsmodels'
        External library to compare with ('statsmodels' or 'arch')
        
    Returns
    -------
    Dict
        Dictionary containing comparison metrics
    """
    results = {
        'engle_granger': {
            'agreement_rate': 0,
            'hedge_ratio_difference': [],
            'p_value_difference': []
        },
        'johansen': {
            'agreement_rate': 0,
            'n_relations_difference': []
        }
    }
    
    eg_agreement = 0
    jo_agreement = 0
    
    for trial in range(n_trials):
        # Generate cointegrated data
        s1, s2, _ = generate_cointegrated_series(n_samples=n_samples, seed=trial)
        
        # Convert to DataFrame for Johansen test
        df = pd.DataFrame({'s1': s1, 's2': s2})
        
        # Engle-Granger comparison
        # Our implementation
        our_eg = engle_granger_test(s2, s1)
        
        # External implementation
        if external_library == 'statsmodels':
            from statsmodels.tsa.stattools import coint
            _, pvalue, _ = coint(s2, s1)
            ext_is_cointegrated = pvalue < 0.05
            
            # Get hedge ratio
            X = sm.add_constant(s1)
            model = sm.OLS(s2, X).fit()
            ext_beta = model.params[1]
        else:
            # Add support for other libraries if needed
            raise ValueError(f"Unsupported external library: {external_library}")
        
        # Compare results
        if our_eg['is_cointegrated'] == ext_is_cointegrated:
            eg_agreement += 1
        
        results['engle_granger']['hedge_ratio_difference'].append(
            abs(our_eg['hedge_ratio'] - ext_beta) / ext_beta
        )
        
        results['engle_granger']['p_value_difference'].append(
            abs(our_eg['p_value'] - pvalue)
        )
        
        # Johansen comparison
        # Our implementation
        our_jo = johansen_test(df)
        
        # External implementation
        if external_library == 'statsmodels':
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            jo_result = coint_johansen(df.values, 0, 1)
            
            # Determine number of cointegrating relations
            trace_stat = jo_result.lr1
            crit_vals = jo_result.cvt
            ext_n_relations = sum(trace_stat > crit_vals[:, 1])
        else:
            # Add support for other libraries if needed
            raise ValueError(f"Unsupported external library: {external_library}")
        
        # Compare results
        if our_jo['n_cointegrating_relations_trace'] == ext_n_relations:
            jo_agreement += 1
        
        results['johansen']['n_relations_difference'].append(
            abs(our_jo['n_cointegrating_relations_trace'] - ext_n_relations)
        )
    
    # Calculate agreement rates
    results['engle_granger']['agreement_rate'] = eg_agreement / n_trials
    results['johansen']['agreement_rate'] = jo_agreement / n_trials
    
    # Calculate average differences
    results['engle_granger']['avg_hedge_ratio_difference'] = np.mean(
        results['engle_granger']['hedge_ratio_difference']
    )
    results['engle_granger']['avg_p_value_difference'] = np.mean(
        results['engle_granger']['p_value_difference']
    )
    results['johansen']['avg_n_relations_difference'] = np.mean(
        results['johansen']['n_relations_difference']
    )
    
    return results


def plot_validation_results(
    results: Dict,
    test_type: str = 'engle_granger',
    output_dir: str = 'validation_results'
) -> None:
    """
    Plot validation results for visual analysis.
    
    Parameters
    ----------
    results : Dict
        Dictionary containing validation results
    
    test_type : str, default='engle_granger'
        Type of test to plot ('engle_granger' or 'johansen')
    
    output_dir : str, default='validation_results'
        Directory to save plots
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if test_type == 'engle_granger':
        # Plot detection rate for cointegrated series
        plt.figure(figsize=(12, 8))
        
        # Extract parameters from keys
        params = []
        rates = []
        for key, rate in results['cointegrated']['detection_rate'].items():
            # Parse key like "n=1000_hr=0.5_nl=0.1_mrs=0.05"
            parts = key.split('_')
            n = int(parts[0].split('=')[1])
            hr = float(parts[1].split('=')[1])
            nl = float(parts[2].split('=')[1])
            mrs = float(parts[3].split('=')[1])
            
            params.append((n, hr, nl, mrs))
            rates.append(rate)
        
        # Convert to numpy arrays for easier manipulation
        params = np.array(params)
        rates = np.array(rates)
        
        # Create plots
        # 1. Detection rate vs. sample size
        n_values = sorted(list(set(params[:, 0])))
        for nl in sorted(list(set(params[:, 2]))):
            mask = params[:, 2] == nl
            plt.plot(params[mask, 0], rates[mask], label=f'Noise={nl}')
        
        plt.xlabel('Sample Size')
        plt.ylabel('Detection Rate')
        plt.title('Cointegration Detection Rate vs. Sample Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'eg_detection_vs_sample_size.png'))
        
        # 2. Detection rate vs. noise level
        plt.figure(figsize=(12, 8))
        nl_values = sorted(list(set(params[:, 2])))
        for n in sorted(list(set(params[:, 0]))):
            mask = params[:, 0] == n
            plt.plot(params[mask, 2], rates[mask], label=f'Sample Size={n}')
        
        plt.xlabel('Noise Level')
        plt.ylabel('Detection Rate')
        plt.title('Cointegration Detection Rate vs. Noise Level')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'eg_detection_vs_noise.png'))
        
        # 3. False positive rate
        plt.figure(figsize=(12, 8))
        
        # Extract parameters from keys
        nc_params = []
        nc_rates = []
        for key, rate in results['non_cointegrated']['false_positive_rate'].items():
            # Parse key like "n=1000_corr=0.5"
            parts = key.split('_')
            n = int(parts[0].split('=')[1])
            corr = float(parts[1].split('=')[1])
            
            nc_params.append((n, corr))
            nc_rates.append(rate)
        
        nc_params = np.array(nc_params)
        nc_rates = np.array(nc_rates)
        
        # Plot false positive rate vs. correlation
        corr_values = sorted(list(set(nc_params[:, 1])))
        for n in sorted(list(set(nc_params[:, 0]))):
            mask = nc_params[:, 0] == n
            plt.plot(nc_params[mask, 1], nc_rates[mask], label=f'Sample Size={n}')
        
        plt.xlabel('Correlation')
        plt.ylabel('False Positive Rate')
        plt.title('False Positive Rate vs. Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'eg_false_positive_vs_correlation.png'))
    
    elif test_type == 'johansen':
        # Plot detection rate and relation count accuracy
        plt.figure(figsize=(12, 8))
        
        # Extract parameters from keys
        params = []
        det_rates = []
        rel_acc = []
        
        for key in results['detection_rate'].keys():
            # Parse key like "n=1000_nvar=3_ncr=1"
            parts = key.split('_')
            n = int(parts[0].split('=')[1])
            nvar = int(parts[1].split('=')[1])
            ncr = int(parts[2].split('=')[1])
            
            params.append((n, nvar, ncr))
            det_rates.append(results['detection_rate'][key])
            rel_acc.append(results['relation_count_accuracy'][key])
        
        params = np.array(params)
        det_rates = np.array(det_rates)
        rel_acc = np.array(rel_acc)
        
        # 1. Detection rate vs. sample size
        n_values = sorted(list(set(params[:, 0])))
        for nvar in sorted(list(set(params[:, 1]))):
            for ncr in sorted(list(set(params[:, 2]))):
                mask = (params[:, 1] == nvar) & (params[:, 2] == ncr)
                if np.any(mask):
                    plt.plot(params[mask, 0], det_rates[mask], 
                             label=f'Vars={nvar}, Relations={ncr}')
        
        plt.xlabel('Sample Size')
        plt.ylabel('Detection Rate')
        plt.title('Cointegration Detection Rate vs. Sample Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'jo_detection_vs_sample_size.png'))
        
        # 2. Relation count accuracy vs. sample size
        plt.figure(figsize=(12, 8))
        for nvar in sorted(list(set(params[:, 1]))):
            for ncr in sorted(list(set(params[:, 2]))):
                mask = (params[:, 1] == nvar) & (params[:, 2] == ncr)
                if np.any(mask):
                    plt.plot(params[mask, 0], rel_acc[mask], 
                             label=f'Vars={nvar}, Relations={ncr}')
        
        plt.xlabel('Sample Size')
        plt.ylabel('Relation Count Accuracy')
        plt.title('Relation Count Accuracy vs. Sample Size')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'jo_accuracy_vs_sample_size.png'))
    
    else:
        raise ValueError(f"Unsupported test type: {test_type}")


if __name__ == "__main__":
    # Usage example:
    
    # Validate Engle-Granger test
    print("Validating Engle-Granger test...")
    eg_results = validate_engle_granger(n_trials=10)  # Reduced trials for quicker execution
    
    # Validate Johansen test
    print("Validating Johansen test...")
    jo_results = validate_johansen(n_trials=10)  # Reduced trials for quicker execution
    
    # Compare with external library
    print("Comparing with statsmodels...")
    comparison = compare_with_external_library(n_trials=5)
    
    # Plot results
    print("Plotting validation results...")
    plot_validation_results(eg_results, test_type='engle_granger')
    plot_validation_results(jo_results, test_type='johansen')
    
    print("Validation complete. Results saved to 'validation_results' directory.") 