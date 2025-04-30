"""
Kalman Filter Implementation for Time-Varying Cointegration

This module provides implementations of Kalman filter algorithms for estimating
time-varying parameters in cointegration relationships, particularly hedge ratios
that adapt to changing market conditions.

References:
    - Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." 
      Journal of Basic Engineering, 82(1), 35-45.
    - Harvey, A. C. (1990). "Forecasting, Structural Time Series Models and the Kalman Filter." 
      Cambridge University Press.
    - Chan, N. H., et al. (2017). "Time-varying cointegration models for longitudinal data." 
      Journal of Business & Economic Statistics, 35(3), 349-362.

Example:
    ```python
    import pandas as pd
    from src.cointegration.kalman_filter import (
        estimate_timevarying_hedge_ratio, 
        plot_timevarying_hedge_ratio
    )
    
    # Load price data
    price1 = pd.read_csv('data/price1.csv', index_col='date', parse_dates=True)['close']
    price2 = pd.read_csv('data/price2.csv', index_col='date', parse_dates=True)['close']
    
    # Estimate time-varying hedge ratio using Kalman filter
    results = estimate_timevarying_hedge_ratio(price1, price2)
    
    # Plot the results
    plot_timevarying_hedge_ratio(results, price1, price2)
    ```
    
    For non-linear models:
    ```python
    from src.cointegration.kalman_filter import (
        estimate_nonlinear_timevarying_hedge_ratio,
        plot_nonlinear_hedge_ratio,
        compare_kalman_models
    )
    
    # Compare different models
    model_comparison = compare_kalman_models(price1, price2)
    print(model_comparison)
    
    # Use recommended model
    recommended_model = model_comparison.index[0]
    if recommended_model == 'linear':
        results = estimate_timevarying_hedge_ratio(price1, price2)
    else:
        results = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2, model_type=recommended_model
        )
    ```
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
import scipy.stats as stats  # Add this import for the stats module


class KalmanFilterBase:
    """
    Base class for Kalman filter implementations.
    
    This class provides the core functionality shared by all Kalman filter variants.
    """
    
    def __init__(self):
        """Initialize the base Kalman filter."""
        self.is_fitted = False
        self.states = None
        self.state_covariances = None
        self.measurements = None
        self.log_likelihood = None
        self.innovations = None
        self.innovation_covariances = None
    
    def fit(self, endog: np.ndarray, exog: np.ndarray, **kwargs):
        """
        Fit the Kalman filter to the data.
        
        Parameters
        ----------
        endog : np.ndarray
            Endogenous variable (dependent variable, y)
        exog : np.ndarray
            Exogenous variables (independent variables, X)
        **kwargs : dict
            Additional parameters for specific filter implementations
            
        Returns
        -------
        self : KalmanFilterBase
            Fitted model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, exog: np.ndarray = None) -> np.ndarray:
        """
        Predict using the fitted Kalman filter.
        
        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction. If None, uses the training data.
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_states(self) -> np.ndarray:
        """
        Get the estimated state vectors (e.g., time-varying coefficients).
        
        Returns
        -------
        np.ndarray
            State vectors for each time point
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing states")
        return self.states
    
    def plot_states(self, 
                   state_names: Optional[List[str]] = None,
                   confidence_interval: Optional[float] = 0.95,
                   figsize: Tuple[int, int] = (12, 8),
                   title: Optional[str] = None) -> plt.Figure:
        """
        Plot the estimated states over time with confidence intervals.
        
        Parameters
        ----------
        state_names : List[str], optional
            Names of the state variables for labeling
        confidence_interval : float, optional
            Confidence interval width (0-1)
        figsize : tuple, default=(12, 8)
            Figure size
        title : str, optional
            Plot title
            
        Returns
        -------
        plt.Figure
            The figure object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting states")
            
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get time index
        time_idx = np.arange(len(self.states))
        
        # Set state names if not provided
        if state_names is None:
            state_names = [f'State {i+1}' for i in range(self.states.shape[1])]
        
        # Plot each state
        for i in range(self.states.shape[1]):
            ax.plot(time_idx, self.states[:, i], label=state_names[i])
            
            # Add confidence intervals if state covariances are available
            if confidence_interval and hasattr(self, 'state_covariances'):
                z_score = stats.norm.ppf((1 + confidence_interval) / 2)
                std = np.sqrt(self.state_covariances[:, i, i])
                ax.fill_between(time_idx, 
                               self.states[:, i] - z_score * std,
                               self.states[:, i] + z_score * std,
                               alpha=0.2)
        
        # Set title and legend
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Time-Varying State Estimates')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('State Value')
        
        plt.tight_layout()
        return fig
    
    def get_innovations(self) -> np.ndarray:
        """
        Get the innovations (prediction errors) from the Kalman filter.
        
        The innovations are the differences between predicted and actual measurements.
        
        Returns
        -------
        np.ndarray
            Innovations for each time point
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing innovations")
        if self.innovations is None:
            raise ValueError("Innovations were not stored during filtering")
        return self.innovations
    
    def get_innovation_statistics(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute statistics of innovations for diagnostics.
        
        Returns
        -------
        Dict
            Dictionary with innovation statistics including:
            - mean: Mean of innovations
            - std: Standard deviation of innovations
            - autocorrelation: Autocorrelation of innovations
            - normality_test: p-value of Shapiro-Wilk normality test
            - white_noise_test: p-value of Ljung-Box test for white noise
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing innovation statistics")
        if self.innovations is None:
            raise ValueError("Innovations were not stored during filtering")
            
        innovations = self.innovations
        
        # Basic statistics
        mean = np.mean(innovations, axis=0)
        std = np.std(innovations, axis=0)
        
        # Autocorrelation (lag-1)
        n = innovations.shape[0]
        if n > 1:
            autocorr = np.array([
                np.corrcoef(innovations[1:, i], innovations[:-1, i])[0, 1] 
                for i in range(innovations.shape[1])
            ])
        else:
            autocorr = np.zeros(innovations.shape[1])
        
        # Normality test
        if n >= 3:  # Minimum required for Shapiro-Wilk test
            from scipy import stats
            normality_pvals = np.array([
                stats.shapiro(innovations[:, i])[1] if len(np.unique(innovations[:, i])) > 1 else np.nan
                for i in range(innovations.shape[1])
            ])
        else:
            normality_pvals = np.zeros(innovations.shape[1]) * np.nan
        
        # White noise test (Ljung-Box)
        if n >= 10:  # Need enough data for test to be meaningful
            from statsmodels.stats.diagnostic import acorr_ljungbox
            try:
                white_noise_pvals = np.array([
                    acorr_ljungbox(innovations[:, i], lags=[10])[1][0] 
                    if len(np.unique(innovations[:, i])) > 1 else np.nan
                    for i in range(innovations.shape[1])
                ])
            except:
                white_noise_pvals = np.zeros(innovations.shape[1]) * np.nan
        else:
            white_noise_pvals = np.zeros(innovations.shape[1]) * np.nan
            
        return {
            'mean': mean,
            'std': std,
            'autocorrelation': autocorr,
            'normality_test': normality_pvals,
            'white_noise_test': white_noise_pvals
        }
    
    def plot_innovation_diagnostics(self, 
                                   measurement_names: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (15, 10),
                                   lags: int = 20) -> plt.Figure:
        """
        Plot innovation diagnostics for Kalman filter performance evaluation.
        
        This includes:
        1. Innovation time series
        2. Innovation histogram with normal fit
        3. Autocorrelation function (ACF) of innovations
        4. Standardized innovations (if innovation_covariances are available)
        
        Parameters
        ----------
        measurement_names : List[str], optional
            Names of measurement variables for labeling
        figsize : tuple, default=(15, 10)
            Figure size
        lags : int, default=20
            Number of lags for autocorrelation plot
            
        Returns
        -------
        plt.Figure
            The figure with diagnostic plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting diagnostics")
        if self.innovations is None:
            raise ValueError("Innovations were not stored during filtering")
            
        innovations = self.innovations
        n_series = innovations.shape[1] if len(innovations.shape) > 1 else 1
        
        # Set measurement names if not provided
        if measurement_names is None:
            measurement_names = [f'Measurement {i+1}' for i in range(n_series)]
        
        # Create figure
        fig, axes = plt.subplots(n_series, 4, figsize=figsize)
        if n_series == 1:
            axes = np.array([axes])
        
        from scipy import stats
        import statsmodels.api as sm
        
        for i in range(n_series):
            series = innovations[:, i] if n_series > 1 else innovations
            
            # 1. Innovation time series
            ax = axes[i, 0]
            ax.plot(series)
            ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            ax.set_title(f'{measurement_names[i]}: Innovations')
            ax.grid(True)
            
            # 2. Innovation histogram
            ax = axes[i, 1]
            ax.hist(series, bins=min(30, len(series)//5), density=True, alpha=0.6)
            
            # Fit normal distribution
            if len(np.unique(series)) > 1:  # Only if there's variation
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, np.mean(series), np.std(series))
                ax.plot(x, p, 'k', linewidth=2)
                
                # Add normality test result
                normality_pval = stats.shapiro(series)[1] if len(series) >= 3 else np.nan
                if np.isfinite(normality_pval):
                    normality_text = f"Normality test p-value: {normality_pval:.4f}"
                    ax.text(0.05, 0.95, normality_text, transform=ax.transAxes,
                           verticalalignment='top', fontsize=9)
            
            ax.set_title(f'{measurement_names[i]}: Histogram')
            
            # 3. Autocorrelation function
            ax = axes[i, 2]
            if len(series) > lags:
                # Compute autocorrelation function
                acf = sm.tsa.acf(series, nlags=lags, fft=True)
                lags_array = np.arange(len(acf))
                ax.stem(lags_array, acf, markerfmt='ro', linefmt='r-', basefmt='k-')
                
                # Add confidence bands (95%)
                confidence_interval = 1.96 / np.sqrt(len(series))
                ax.axhspan(-confidence_interval, confidence_interval, alpha=0.2, color='blue')
                
                # Add white noise test result
                try:
                    white_noise_pval = sm.stats.diagnostic.acorr_ljungbox(series, lags=[lags])[1][0]
                    if np.isfinite(white_noise_pval):
                        white_noise_text = f"White noise test p-value: {white_noise_pval:.4f}"
                        ax.text(0.05, 0.95, white_noise_text, transform=ax.transAxes,
                               verticalalignment='top', fontsize=9)
                except:
                    pass
            
            ax.set_title(f'{measurement_names[i]}: Autocorrelation')
            
            # 4. Standardized innovations (if innovation_covariances are available)
            ax = axes[i, 3]
            
            if self.innovation_covariances is not None:
                # Standardize innovations by dividing by their standard deviations
                std_innovations = np.zeros_like(series)
                for t in range(len(series)):
                    # Get standard deviation from innovation covariance
                    std = np.sqrt(self.innovation_covariances[t, i, i]) if n_series > 1 else np.sqrt(self.innovation_covariances[t])
                    if std > 0:
                        std_innovations[t] = series[t] / std
                
                # Plot standardized innovations
                ax.plot(std_innovations)
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.axhline(y=1.96, color='g', linestyle='--', alpha=0.3)
                ax.axhline(y=-1.96, color='g', linestyle='--', alpha=0.3)
                
                # Add text showing percentage of points outside 95% bands
                outside_bands = np.sum(np.abs(std_innovations) > 1.96) / len(std_innovations) * 100
                outside_text = f"Outside 95% bands: {outside_bands:.1f}%"
                ax.text(0.05, 0.95, outside_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=9)
                
                ax.set_title(f'{measurement_names[i]}: Standardized Innovations')
            else:
                ax.text(0.5, 0.5, 'Innovation covariances not available',
                       horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{measurement_names[i]}: Standardized Innovations')
            
            ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def calculate_filter_performance_metrics(self, true_states: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate performance metrics for the Kalman filter.
        
        If true_states are provided, computes metrics comparing estimated to true states.
        If not, computes metrics based on innovations.
        
        Parameters
        ----------
        true_states : np.ndarray, optional
            True state values for comparison, shape (n_samples, n_states)
            
        Returns
        -------
        Dict
            Dictionary with performance metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating performance metrics")
            
        metrics = {}
        
        # Metrics based on true states (if provided)
        if true_states is not None:
            if true_states.shape != self.states.shape:
                raise ValueError(f"Shape mismatch: true_states shape {true_states.shape} != states shape {self.states.shape}")
                
            # Calculate mean squared error
            mse = np.mean((true_states - self.states) ** 2, axis=0)
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            
            # Calculate mean absolute error
            mae = np.mean(np.abs(true_states - self.states), axis=0)
            metrics['mae'] = mae
            
            # Calculate correlation between estimated and true states
            correlations = np.array([
                np.corrcoef(self.states[:, i], true_states[:, i])[0, 1]
                for i in range(self.states.shape[1])
            ])
            metrics['correlations'] = correlations
        
        # Metrics based on innovations
        if self.innovations is not None:
            innovation_stats = self.get_innovation_statistics()
            metrics.update(innovation_stats)
            
            # Standardized innovation sum of squares
            if self.innovation_covariances is not None:
                n_samples = self.innovations.shape[0]
                n_innovations = self.innovations.shape[1] if len(self.innovations.shape) > 1 else 1
                
                # Initialize sum
                niss = 0
                
                # Calculate normalized innovation squared statistic
                for t in range(n_samples):
                    innovation = self.innovations[t]
                    innovation_cov = self.innovation_covariances[t]
                    
                    # Handle various shapes appropriately
                    if n_innovations == 1:
                        niss += (innovation ** 2) / innovation_cov
                    else:
                        try:
                            niss += innovation @ np.linalg.solve(innovation_cov, innovation)
                        except:
                            # Skip if there's a numerical issue
                            pass
                            
                metrics['niss'] = niss
                metrics['niss_per_sample'] = niss / n_samples
                
                # NISS should be close to n_innovations * n_samples for optimal filter
                expected_niss = n_innovations * n_samples
                metrics['niss_ratio'] = niss / expected_niss
        
        # Add log-likelihood if available
        if self.log_likelihood is not None:
            metrics['log_likelihood'] = self.log_likelihood
            metrics['aic'] = -2 * self.log_likelihood + 2 * self.states.shape[1]
            
        return metrics


class LinearKalmanFilter(KalmanFilterBase):
    """
    Linear Kalman Filter for time-varying regression.
    
    This implements the standard Kalman filter for estimating time-varying
    coefficients in a linear regression model: y[t] = X[t] * beta[t] + v[t],
    where beta[t] follows a random walk.
    
    Parameters
    ----------
    transition_covariance : float or np.ndarray, default=1e-4
        Covariance matrix of the transition noise (process noise)
    observation_covariance : float or np.ndarray, default=1e-2
        Covariance matrix of the observation noise (measurement noise)
    initial_state_mean : np.ndarray, optional
        Initial state mean. If None, it will be estimated from the data.
    initial_state_covariance : np.ndarray, optional
        Initial state covariance. If None, it will be set to a diagonal matrix.
    adapt_observation_noise : bool, default=False
        Whether to adapt the observation noise variance over time
    em_iterations : int, default=0
        Number of EM iterations for parameter estimation
        
    Attributes
    ----------
    states : np.ndarray
        Estimated states (time-varying coefficients) for each time point
    state_covariances : np.ndarray
        Estimated state covariances for each time point
    transition_matrices : np.ndarray
        Transition matrices for the state equation
    observation_matrices : np.ndarray
        Observation matrices for the measurement equation
    transition_covariance : np.ndarray
        Transition covariance matrix (Q)
    observation_covariance : np.ndarray
        Observation covariance matrix (R)
    initial_state_mean : np.ndarray
        Initial state mean
    initial_state_covariance : np.ndarray
        Initial state covariance
    log_likelihood : float
        Log-likelihood of the fitted model
    """
    
    def __init__(self, 
                transition_covariance: Union[float, np.ndarray] = 1e-4,
                observation_covariance: Union[float, np.ndarray] = 1e-2,
                initial_state_mean: Optional[np.ndarray] = None,
                initial_state_covariance: Optional[np.ndarray] = None,
                adapt_observation_noise: bool = False,
                em_iterations: int = 0):
        """Initialize the linear Kalman filter."""
        super().__init__()
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.adapt_observation_noise = adapt_observation_noise
        self.em_iterations = em_iterations
    
    def fit(self, endog: np.ndarray, exog: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> 'LinearKalmanFilter':
        """
        Fit the linear Kalman filter to the data.
        
        Parameters
        ----------
        endog : np.ndarray
            Endogenous variable (dependent variable, y)
        exog : np.ndarray
            Exogenous variables (independent variables, X)
            Should include a column of ones for the intercept if needed
        dates : pd.DatetimeIndex, optional
            Dates for the time series data
            
        Returns
        -------
        self : LinearKalmanFilter
            Fitted model
        """
        # Ensure arrays are numpy arrays
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        
        # Check dimensions
        if endog.ndim == 1:
            endog = endog.reshape(-1, 1)
        
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        
        n_obs, n_states = exog.shape
        
        # Set up transition matrices (assuming random walk model for states)
        transition_matrices = np.eye(n_states)
        
        # Set up observation matrices (X in the regression model)
        observation_matrices = exog
        
        # Convert scalar covariances to matrices if needed
        if np.isscalar(self.transition_covariance):
            transition_covariance = np.eye(n_states) * self.transition_covariance
        else:
            transition_covariance = self.transition_covariance
            
        if np.isscalar(self.observation_covariance):
            observation_covariance = np.eye(1) * self.observation_covariance
        else:
            observation_covariance = self.observation_covariance
        
        # Set up initial state
        if self.initial_state_mean is None:
            # Estimate using OLS on first few observations
            init_obs = min(50, n_obs // 10 + 5)  # Use 10% of data or 50 obs, whichever is smaller
            initial_state_mean = np.linalg.lstsq(exog[:init_obs], endog[:init_obs], rcond=None)[0].T
        else:
            initial_state_mean = self.initial_state_mean
            
        if self.initial_state_covariance is None:
            initial_state_covariance = np.eye(n_states) * 1.0
        else:
            initial_state_covariance = self.initial_state_covariance
        
        # Run Kalman filter
        # We'll use the pykalman library if available, otherwise use our own implementation
        try:
            from pykalman import KalmanFilter
            
            kf = KalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance,
                em_vars=['transition_covariance', 'observation_covariance'] if self.em_iterations > 0 else []
            )
            
            # Run EM algorithm if requested
            if self.em_iterations > 0:
                kf = kf.em(endog, n_iter=self.em_iterations)
            
            # Smooth to get state estimates
            self.states, self.state_covariances = kf.smooth(endog)
            
            # Store loglikelihood
            self.log_likelihood = kf.loglikelihood(endog)
            
        except ImportError:
            # Use our own implementation
            self.states, self.state_covariances, self.log_likelihood = self._kalman_filter(
                endog=endog,
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                initial_state_mean=initial_state_mean,
                initial_state_covariance=initial_state_covariance
            )
        
        # Store data and parameters
        self.endog = endog
        self.exog = exog
        self.dates = dates
        self.n_obs = n_obs
        self.n_states = n_states
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the fitted Kalman filter.
        
        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction. If None, uses the training data.
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use training exog if not provided
        if exog is None:
            exog = self.exog
        else:
            exog = np.asarray(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
        
        # Predict using the time-varying coefficients
        if exog.shape[0] == self.n_obs:
            # If same number of observations, use the estimated state for each time point
            predictions = np.sum(exog * self.states, axis=1, keepdims=True)
        else:
            # Otherwise, use the last estimated state for all predictions
            predictions = np.dot(exog, self.states[-1])
            
        return predictions
    
    def _kalman_filter(self, 
                      endog: np.ndarray,
                      transition_matrices: np.ndarray,
                      observation_matrices: np.ndarray,
                      transition_covariance: np.ndarray,
                      observation_covariance: np.ndarray,
                      initial_state_mean: np.ndarray,
                      initial_state_covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the Kalman filter algorithm.
        
        Parameters
        ----------
        endog : np.ndarray
            Endogenous variable (dependent variable, y)
        transition_matrices : np.ndarray
            Transition matrices for the state equation
        observation_matrices : np.ndarray
            Observation matrices for the measurement equation
        transition_covariance : np.ndarray
            Transition covariance matrix (Q)
        observation_covariance : np.ndarray
            Observation covariance matrix (R)
        initial_state_mean : np.ndarray
            Initial state mean
        initial_state_covariance : np.ndarray
            Initial state covariance
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states, state covariances, and log likelihood
        """
        n_obs = len(endog)
        n_states = len(initial_state_mean)
        
        # Reshape input for consistency
        if len(endog.shape) == 1:
            endog = endog.reshape(-1, 1)
        
        # Initialize storage for filtered states, covariances, and innovations
        filtered_states = np.zeros((n_obs, n_states))
        filtered_covs = np.zeros((n_obs, n_states, n_states))
        self.innovations = np.zeros((n_obs, endog.shape[1]))
        self.innovation_covariances = np.zeros((n_obs, endog.shape[1], endog.shape[1]))
        log_likelihood = 0.0
        
        # Initialize state and covariance
        state = initial_state_mean.copy()
        state_cov = initial_state_covariance.copy()
        
        for t in range(n_obs):
            # Make transition_matrices time-varying if it's a 3D array
            if len(transition_matrices.shape) == 3:
                current_transition = transition_matrices[t]
            else:
                current_transition = transition_matrices
            
            # Make observation_matrices time-varying if it's a 3D array
            if len(observation_matrices.shape) == 3:
                current_observation = observation_matrices[t]
            else:
                current_observation = observation_matrices
            
            # Make observation_covariance time-varying if it's a 3D array
            if len(observation_covariance.shape) == 3:
                current_obs_cov = observation_covariance[t]
            else:
                current_obs_cov = observation_covariance
            
            # Prediction step (time update)
            state = current_transition @ state
            state_cov = current_transition @ state_cov @ current_transition.T + transition_covariance
            
            # Ensure state covariance is symmetric and positive definite
            state_cov = (state_cov + state_cov.T) / 2
            
            # Correction step (measurement update)
            predicted_obs = current_observation @ state
            
            # Calculate innovation and its covariance
            innovation = endog[t] - predicted_obs
            innovation_cov = current_observation @ state_cov @ current_observation.T + current_obs_cov
            
            # Store innovation and its covariance
            self.innovations[t] = innovation
            self.innovation_covariances[t] = innovation_cov
            
            # Ensure innovation covariance is symmetric
            innovation_cov = (innovation_cov + innovation_cov.T) / 2
            
            # In case of numerical issues, add a small value to the diagonal
            min_eig = np.min(np.real(np.linalg.eigvals(innovation_cov)))
            if min_eig < 1e-10:
                innovation_cov += np.eye(len(innovation_cov)) * (1e-10 - min_eig)
            
            # Calculate Kalman gain
            try:
                kalman_gain = state_cov @ current_observation.T @ np.linalg.inv(innovation_cov)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                kalman_gain = state_cov @ current_observation.T @ np.linalg.pinv(innovation_cov)
            
            # Update state and covariance
            state = state + kalman_gain @ innovation
            state_cov = state_cov - kalman_gain @ innovation_cov @ kalman_gain.T
            
            # Ensure updated state covariance is symmetric
            state_cov = (state_cov + state_cov.T) / 2
            
            # Store filtered state and covariance
            filtered_states[t] = state
            filtered_covs[t] = state_cov
            
            # Update log likelihood if needed
            try:
                sign, logdet = np.linalg.slogdet(innovation_cov)
                if sign > 0:
                    log_det = logdet
                else:
                    # If determinant is negative due to numerical issues, use pseudo-determinant
                    eigvals = np.linalg.eigvalsh(innovation_cov)
                    log_det = np.sum(np.log(np.maximum(eigvals, 1e-10)))
                
                log_lik = -0.5 * (log_det + innovation @ np.linalg.solve(innovation_cov, innovation) + 
                                 len(innovation) * np.log(2 * np.pi))
                log_likelihood += log_lik[0] if len(log_lik.shape) > 0 else log_lik
            except:
                # Skip likelihood contribution if there's a numerical issue
                pass
        
        return filtered_states, filtered_covs, log_likelihood


def estimate_timevarying_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    add_intercept: bool = True,
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    em_iterations: int = 0,
    return_model: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, LinearKalmanFilter]]:
    """
    Estimate time-varying hedge ratio between two price series using Kalman filter.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series (independent variable, X)
    price2 : pd.Series
        Second price series (dependent variable, Y)
    add_intercept : bool, default=True
        Whether to add an intercept term to the model
    transition_covariance : float, default=1e-4
        Transition noise covariance
    observation_covariance : float, default=1e-2
        Observation noise covariance
    em_iterations : int, default=0
        Number of EM iterations for parameter estimation
    return_model : bool, default=False
        Whether to return the fitted model object along with results
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, LinearKalmanFilter]
        DataFrame with time-varying hedge ratio and spread, and optionally the fitted model
    """
    # Align the price series
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Prepare data for Kalman filter
    y = s2.values.reshape(-1, 1)  # Dependent variable
    
    if add_intercept:
        # Add column of ones for intercept
        X = np.column_stack([np.ones(len(s1)), s1.values])
    else:
        X = s1.values.reshape(-1, 1)
    
    # Create and fit the Kalman filter
    kf = LinearKalmanFilter(
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        adapt_observation_noise=False,
        em_iterations=em_iterations
    )
    
    kf.fit(y, X, dates=common_index)
    
    # Extract the time-varying coefficients
    if add_intercept:
        intercept = kf.states[:, 0]
        hedge_ratio = kf.states[:, 1]
    else:
        intercept = np.zeros(len(y))
        hedge_ratio = kf.states[:, 0]
    
    # Calculate the spread
    spread = y.flatten() - (intercept + hedge_ratio * s1.values)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'intercept': intercept,
        'hedge_ratio': hedge_ratio,
        'spread': spread
    }, index=common_index)
    
    if return_model:
        return results, kf
    else:
        return results


def plot_timevarying_hedge_ratio(
    results: pd.DataFrame,
    price1: Optional[pd.Series] = None,
    price2: Optional[pd.Series] = None,
    confidence_intervals: bool = False,
    state_covariances: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the time-varying hedge ratio and related statistics.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame with time-varying coefficients and spread
    price1 : pd.Series, optional
        First price series
    price2 : pd.Series, optional
        Second price series
    confidence_intervals : bool, default=False
        Whether to plot confidence intervals
    state_covariances : np.ndarray, optional
        State covariances for calculating confidence intervals
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Determine number of subplots
    n_plots = 2  # Hedge ratio and spread are always shown
    if price1 is not None and price2 is not None:
        n_plots += 1  # Add prices plot
    
    gs = plt.GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
    
    plot_idx = 0
    
    # Plot original price series if provided
    if price1 is not None and price2 is not None:
        ax1 = fig.add_subplot(gs[plot_idx])
        ax1.plot(price1, label=price1.name if hasattr(price1, 'name') else 'Price 1')
        ax1.plot(price2, label=price2.name if hasattr(price2, 'name') else 'Price 2')
        ax1.set_title('Price Series')
        ax1.legend()
        plot_idx += 1
    
    # Plot time-varying hedge ratio
    ax2 = fig.add_subplot(gs[plot_idx])
    ax2.plot(results.index, results['hedge_ratio'], 'b-')
    
    # Add confidence intervals if requested
    if confidence_intervals and state_covariances is not None:
        z_score = stats.norm.ppf(0.975)  # 95% confidence interval
        std = np.sqrt(state_covariances[:, 1, 1] if 'intercept' in results else state_covariances[:, 0, 0])
        ax2.fill_between(results.index, 
                        results['hedge_ratio'] - z_score * std,
                        results['hedge_ratio'] + z_score * std,
                        alpha=0.2, color='blue')
    
    ax2.set_title('Time-Varying Hedge Ratio')
    plot_idx += 1
    
    # Plot spread
    ax3 = fig.add_subplot(gs[plot_idx])
    ax3.plot(results.index, results['spread'], 'g-')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax3.set_title('Spread')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 


class ExtendedKalmanFilter(KalmanFilterBase):
    """
    Extended Kalman Filter for nonlinear state space models.
    
    This implements the extended Kalman filter for estimating time-varying
    coefficients in non-linear state-space models by linearizing the model
    around the current state estimate.
    
    Parameters
    ----------
    state_transition_function : Callable
        Function that computes the state transition: f(x_{t-1}) -> x_t
    observation_function : Callable
        Function that computes the observation given state: h(x_t) -> y_t
    state_transition_jacobian : Callable
        Function that computes the Jacobian of the state transition function
    observation_jacobian : Callable
        Function that computes the Jacobian of the observation function
    transition_covariance : float or np.ndarray, default=1e-4
        Covariance matrix of the transition noise (process noise)
    observation_covariance : float or np.ndarray, default=1e-2
        Covariance matrix of the observation noise (measurement noise)
    initial_state_mean : np.ndarray, optional
        Initial state mean. If None, it will be set to zeros.
    initial_state_covariance : np.ndarray, optional
        Initial state covariance. If None, it will be set to a diagonal matrix.
        
    Attributes
    ----------
    states : np.ndarray
        Estimated states (time-varying coefficients) for each time point
    state_covariances : np.ndarray
        Estimated state covariances for each time point
    """
    
    def __init__(self,
                state_transition_function: Callable,
                observation_function: Callable,
                state_transition_jacobian: Callable,
                observation_jacobian: Callable,
                transition_covariance: Union[float, np.ndarray] = 1e-4,
                observation_covariance: Union[float, np.ndarray] = 1e-2,
                initial_state_mean: Optional[np.ndarray] = None,
                initial_state_covariance: Optional[np.ndarray] = None):
        """Initialize the extended Kalman filter."""
        super().__init__()
        self.state_transition_function = state_transition_function
        self.observation_function = observation_function
        self.state_transition_jacobian = state_transition_jacobian
        self.observation_jacobian = observation_jacobian
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
    
    def fit(self, 
           observations: np.ndarray, 
           n_states: int,
           exog: Optional[np.ndarray] = None,
           dates: Optional[pd.DatetimeIndex] = None) -> 'ExtendedKalmanFilter':
        """
        Fit the extended Kalman filter to the data.
        
        Parameters
        ----------
        observations : np.ndarray
            Observation data, shape (n_samples, n_observations)
        n_states : int
            Number of state variables
        exog : np.ndarray, optional
            Exogenous variables, shape (n_samples, n_exog_variables)
        dates : pd.DatetimeIndex, optional
            Dates for the observations
            
        Returns
        -------
        ExtendedKalmanFilter
            Fitted model
        """
        # Force observations to be 2D for consistent processing
        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1)
        
        n_samples, n_obs = observations.shape
        
        # Initialize state mean and covariance
        if self.initial_state_mean is None:
            # Default initial state: zeros
            self.initial_state_mean = np.zeros(n_states)
        
        if self.initial_state_covariance is None:
            # Default initial covariance: identity
            self.initial_state_covariance = np.eye(n_states)
        
        # Initialize transition and observation covariances
        if isinstance(self.transition_covariance, (int, float)):
            self.transition_covariance = self.transition_covariance * np.eye(n_states)
        
        if isinstance(self.observation_covariance, (int, float)):
            self.observation_covariance = self.observation_covariance * np.eye(n_obs)
        
        # Prepare storage for states and covariances
        self.states = np.zeros((n_samples, n_states))
        self.state_covariances = np.zeros((n_samples, n_states, n_states))
        self.innovations = np.zeros((n_samples, n_obs))
        self.innovation_covariances = np.zeros((n_samples, n_obs, n_obs))
        
        # Initialize state
        state = self.initial_state_mean.copy()
        state_cov = self.initial_state_covariance.copy()
        
        # Run filter
        log_likelihood = 0.0
        
        for t in range(n_samples):
            # Get current exogenous variables
            current_exog = None if exog is None else exog[t]
            
            # Prediction step
            # Get state transition Jacobian for current state
            F = self.state_transition_jacobian(state, current_exog)
            
            # Predict state using nonlinear function
            predicted_state = self.state_transition_function(state, current_exog)
            
            # Predict covariance using linearized model
            predicted_cov = F @ state_cov @ F.T + self.transition_covariance
            
            # Ensure predicted_cov is symmetric
            predicted_cov = (predicted_cov + predicted_cov.T) / 2
            
            # Correction step
            # Get observation Jacobian for predicted state
            H = self.observation_jacobian(predicted_state, current_exog)
            
            # Predict observation using nonlinear function
            predicted_obs = self.observation_function(predicted_state, current_exog)
            if not isinstance(predicted_obs, np.ndarray):
                predicted_obs = np.array([predicted_obs])
            
            # Calculate innovation and its covariance
            innovation = observations[t] - predicted_obs
            innovation_cov = H @ predicted_cov @ H.T + self.observation_covariance
            
            # Store innovation and its covariance
            self.innovations[t] = innovation
            self.innovation_covariances[t] = innovation_cov
            
            # Ensure innovation_cov is symmetric
            innovation_cov = (innovation_cov + innovation_cov.T) / 2
            
            # In case of numerical issues, add a small value to the diagonal
            min_eig = np.min(np.real(np.linalg.eigvals(innovation_cov)))
            if min_eig < 1e-10:
                innovation_cov += np.eye(len(innovation_cov)) * (1e-10 - min_eig)
            
            # Calculate Kalman gain
            try:
                kalman_gain = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                kalman_gain = predicted_cov @ H.T @ np.linalg.pinv(innovation_cov)
            
            # Update state and covariance
            state = predicted_state + kalman_gain @ innovation
            state_cov = predicted_cov - kalman_gain @ innovation_cov @ kalman_gain.T
            
            # Ensure state_cov is symmetric
            state_cov = (state_cov + state_cov.T) / 2
            
            # Store filtered state and covariance
            self.states[t] = state
            self.state_covariances[t] = state_cov
            
            # Update log likelihood if possible
            try:
                sign, logdet = np.linalg.slogdet(innovation_cov)
                if sign > 0:
                    log_det = logdet
                else:
                    # If determinant is negative due to numerical issues, use pseudo-determinant
                    eigvals = np.linalg.eigvalsh(innovation_cov)
                    log_det = np.sum(np.log(np.maximum(eigvals, 1e-10)))
                
                log_lik = -0.5 * (log_det + innovation @ np.linalg.solve(innovation_cov, innovation) + 
                                 len(innovation) * np.log(2 * np.pi))
                log_likelihood += log_lik
            except:
                # Skip likelihood contribution if there's a numerical issue
                pass
        
        self.log_likelihood = log_likelihood
        self.is_fitted = True
        
        return self
    
    def predict(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict observations using the fitted extended Kalman filter.
        
        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction. If None, uses the training data.
            
        Returns
        -------
        np.ndarray
            Predicted observations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Use training exog if not provided
        if exog is None and hasattr(self, 'exog'):
            exog = self.exog
        
        # Generate predictions
        n_pred = len(exog) if exog is not None else self.n_obs
        predictions = np.zeros((n_pred, self.observations.shape[1]))
        
        for t in range(n_pred):
            # Use smoothed state for prediction
            idx = min(t, len(self.states) - 1)
            state = self.states[idx]
            
            # Apply observation function to state
            predictions[t] = self.observation_function(
                state, 
                exog[t] if exog is not None else None
            )
        
        return predictions 


def estimate_nonlinear_timevarying_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    model_type: str = 'threshold',
    threshold: Optional[float] = None,
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    return_model: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ExtendedKalmanFilter]]:
    """
    Estimate time-varying hedge ratio with non-linear dynamics using Extended Kalman Filter.
    
    This function supports different types of non-linear models for the hedge ratio dynamics.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series (independent variable, X)
    price2 : pd.Series
        Second price series (dependent variable, Y)
    model_type : str, default='threshold'
        Type of non-linear model:
        - 'threshold': Threshold model where hedge ratio changes based on price level
        - 'regime_switch': Regime switching model where hedge ratio depends on volatility
        - 'log_price': Log-transformed price relationship
    threshold : float, optional
        Threshold value for threshold models. If None, it's estimated from the data.
    transition_covariance : float, default=1e-4
        Process noise covariance
    observation_covariance : float, default=1e-2
        Measurement noise covariance
    return_model : bool, default=False
        Whether to return the fitted model object along with results
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, ExtendedKalmanFilter]
        DataFrame with time-varying parameters and spread, and optionally the fitted model
    """
    # Align the price series
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Prepare data
    y = s2.values.reshape(-1, 1)  # Observations
    x = s1.values.reshape(-1, 1)  # Exogenous variables
    
    # Set up non-linear model based on model_type
    if model_type == 'threshold':
        # Estimate threshold if not provided
        if threshold is None:
            threshold = np.median(x)
        
        # State variables: [intercept, beta_low, beta_high]
        n_states = 3
        
        # Define state transition (random walk for parameters)
        def state_transition(state, exog=None):
            return state
        
        # State transition Jacobian (identity for random walk)
        def state_transition_jacobian(state, exog=None):
            return np.eye(n_states)
        
        # Observation function (threshold model)
        def observation_function(state, exog):
            intercept = state[0]
            beta_low = state[1]
            beta_high = state[2]
            
            # Apply different beta based on threshold
            if exog < threshold:
                return intercept + beta_low * exog
            else:
                return intercept + beta_high * exog
        
        # Observation Jacobian
        def observation_jacobian(state, exog):
            H = np.zeros((1, n_states))
            H[0, 0] = 1.0  # derivative w.r.t. intercept
            
            # Derivatives for betas depend on threshold
            if exog < threshold:
                H[0, 1] = exog  # derivative w.r.t. beta_low
                H[0, 2] = 0.0   # derivative w.r.t. beta_high
            else:
                H[0, 1] = 0.0   # derivative w.r.t. beta_low
                H[0, 2] = exog  # derivative w.r.t. beta_high
                
            return H
        
        # Initial state: [intercept, beta_low, beta_high]
        # Use OLS to initialize
        X_init = np.column_stack([np.ones_like(x[:50]), x[:50]])
        coeffs_init = np.linalg.lstsq(X_init, y[:50], rcond=None)[0]
        initial_state_mean = np.array([coeffs_init[0, 0], coeffs_init[1, 0], coeffs_init[1, 0]])
        
    elif model_type == 'regime_switch':
        # State variables: [intercept, beta, volatility_state]
        n_states = 3
        
        # Define state transition (random walk with volatility dynamics)
        def state_transition(state, exog=None):
            # Volatility state has mean-reversion
            state[2] = 0.8 * state[2] + 0.2 * np.log(np.std(y[-20:]) if len(y) > 20 else 0.1)
            return state
        
        # State transition Jacobian
        def state_transition_jacobian(state, exog=None):
            F = np.eye(n_states)
            F[2, 2] = 0.8  # Mean-reversion for volatility state
            return F
        
        # Observation function (volatility affects hedge ratio)
        def observation_function(state, exog):
            intercept = state[0]
            beta_base = state[1]
            vol_state = state[2]
            
            # Adjust beta based on volatility state
            vol_factor = np.exp(vol_state) / (1 + np.exp(vol_state))
            adjusted_beta = beta_base * (1 + 0.5 * (vol_factor - 0.5))
            
            return intercept + adjusted_beta * exog
        
        # Observation Jacobian
        def observation_jacobian(state, exog):
            H = np.zeros((1, n_states))
            H[0, 0] = 1.0  # derivative w.r.t. intercept
            
            # Compute volatility factor and derivatives
            vol_state = state[2]
            vol_factor = np.exp(vol_state) / (1 + np.exp(vol_state))
            
            H[0, 1] = exog * (1 + 0.5 * (vol_factor - 0.5))  # derivative w.r.t. beta_base
            
            # Derivative of beta adjustment w.r.t. volatility state
            vol_deriv = 0.5 * state[1] * exog * (np.exp(vol_state) / ((1 + np.exp(vol_state)) ** 2))
            H[0, 2] = vol_deriv
            
            return H
        
        # Initial state: [intercept, beta, volatility_state]
        X_init = np.column_stack([np.ones_like(x[:50]), x[:50]])
        coeffs_init = np.linalg.lstsq(X_init, y[:50], rcond=None)[0]
        initial_state_mean = np.array([
            coeffs_init[0, 0],  # intercept
            coeffs_init[1, 0],  # beta
            np.log(np.std(y[:50]))  # initial volatility state
        ])
        
    elif model_type == 'log_price':
        # State variables: [intercept, beta, log_factor]
        n_states = 3
        
        # Define state transition (random walk for parameters)
        def state_transition(state, exog=None):
            return state
        
        # State transition Jacobian (identity for random walk)
        def state_transition_jacobian(state, exog=None):
            return np.eye(n_states)
        
        # Observation function (log-price transformation)
        def observation_function(state, exog):
            intercept = state[0]
            beta = state[1]
            log_factor = state[2]
            
            # Apply log transformation based on log_factor
            if log_factor > 0:
                transformed_exog = beta * (exog + log_factor * np.log(exog + 1))
            else:
                transformed_exog = beta * exog
                
            return intercept + transformed_exog
        
        # Observation Jacobian
        def observation_jacobian(state, exog):
            H = np.zeros((1, n_states))
            H[0, 0] = 1.0  # derivative w.r.t. intercept
            
            # Compute derivatives for beta and log_factor
            log_factor = state[2]
            
            if log_factor > 0:
                H[0, 1] = exog + log_factor * np.log(exog + 1)  # derivative w.r.t. beta
                H[0, 2] = state[1] * np.log(exog + 1)  # derivative w.r.t. log_factor
            else:
                H[0, 1] = exog  # derivative w.r.t. beta
                H[0, 2] = 0.0   # derivative w.r.t. log_factor
                
            return H
        
        # Initial state: [intercept, beta, log_factor]
        X_init = np.column_stack([np.ones_like(x[:50]), x[:50]])
        coeffs_init = np.linalg.lstsq(X_init, y[:50], rcond=None)[0]
        initial_state_mean = np.array([coeffs_init[0, 0], coeffs_init[1, 0], 0.0])
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create and fit Extended Kalman Filter
    ekf = ExtendedKalmanFilter(
        state_transition_function=state_transition,
        observation_function=observation_function,
        state_transition_jacobian=state_transition_jacobian,
        observation_jacobian=observation_jacobian,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean
    )
    
    ekf.fit(y, n_states=n_states, exog=x)
    
    # Extract estimated states
    states = ekf.states
    
    # Create results DataFrame based on model type
    if model_type == 'threshold':
        results = pd.DataFrame({
            'intercept': states[:, 0],
            'beta_low': states[:, 1],
            'beta_high': states[:, 2],
            'threshold': threshold
        }, index=common_index)
        
        # Calculate effective hedge ratio for each point
        effective_beta = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] < threshold:
                effective_beta[i] = states[i, 1]  # beta_low
            else:
                effective_beta[i] = states[i, 2]  # beta_high
                
        results['effective_hedge_ratio'] = effective_beta
        
    elif model_type == 'regime_switch':
        # Calculate volatility factor
        vol_state = states[:, 2]
        vol_factor = np.exp(vol_state) / (1 + np.exp(vol_state))
        
        # Calculate effective hedge ratio
        beta_base = states[:, 1]
        effective_beta = beta_base * (1 + 0.5 * (vol_factor - 0.5))
        
        results = pd.DataFrame({
            'intercept': states[:, 0],
            'beta_base': beta_base,
            'volatility_state': vol_state,
            'volatility_factor': vol_factor,
            'effective_hedge_ratio': effective_beta
        }, index=common_index)
        
    elif model_type == 'log_price':
        # Calculate effective hedge ratio
        log_factor = states[:, 2]
        beta = states[:, 1]
        
        effective_beta = np.zeros(len(x))
        for i in range(len(x)):
            if log_factor[i] > 0:
                effective_beta[i] = beta[i] * (1 + log_factor[i] * np.log(x[i] + 1) / x[i])
            else:
                effective_beta[i] = beta[i]
                
        results = pd.DataFrame({
            'intercept': states[:, 0],
            'beta': beta,
            'log_factor': log_factor,
            'effective_hedge_ratio': effective_beta
        }, index=common_index)
    
    # Calculate spread
    predicted = ekf.predict(exog=x)
    spread = y.flatten() - predicted.flatten()
    results['spread'] = spread
    
    if return_model:
        return results, ekf
    else:
        return results


def plot_nonlinear_hedge_ratio(
    results: pd.DataFrame,
    price1: Optional[pd.Series] = None,
    price2: Optional[pd.Series] = None,
    model_type: str = 'threshold',
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot non-linear time-varying hedge ratio and related statistics.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results DataFrame from estimate_nonlinear_timevarying_hedge_ratio
    price1 : pd.Series, optional
        First price series
    price2 : pd.Series, optional
        Second price series
    model_type : str, default='threshold'
        Type of non-linear model ('threshold', 'regime_switch', or 'log_price')
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Determine number of subplots based on model type and available data
    n_plots = 3  # Minimum: effective hedge ratio, model-specific plot, spread
    if price1 is not None and price2 is not None:
        n_plots += 1  # Add prices plot
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
    
    plot_idx = 0
    
    # Plot original price series if provided
    if price1 is not None and price2 is not None:
        ax1 = fig.add_subplot(gs[plot_idx])
        ax1.plot(price1, label=price1.name if hasattr(price1, "name") else 'Price 1')
        ax1.plot(price2, label=price2.name if hasattr(price2, "name") else 'Price 2')
        ax1.set_title('Price Series')
        ax1.legend()
        plot_idx += 1
    
    # Plot model-specific parameters
    ax2 = fig.add_subplot(gs[plot_idx])
    
    if model_type == 'threshold':
        ax2.plot(results.index, results['beta_low'], 'b-', label='Beta (Low Regime)')
        ax2.plot(results.index, results['beta_high'], 'r-', label='Beta (High Regime)')
        ax2.axhline(y=results['threshold'].iloc[0], color='k', linestyle='--', label=f'Threshold = {results["threshold"].iloc[0]:.4f}')
        ax2.set_title('Threshold Model Parameters')
        
    elif model_type == 'regime_switch':
        ax2.plot(results.index, results['beta_base'], 'b-', label='Base Beta')
        ax2.plot(results.index, results['volatility_factor'], 'r-', label='Volatility Factor')
        ax2.set_title('Regime-Switching Model Parameters')
        
    elif model_type == 'log_price':
        ax2.plot(results.index, results['beta'], 'b-', label='Base Beta')
        ax2.plot(results.index, results['log_factor'], 'r-', label='Log Factor')
        ax2.set_title('Log-Price Model Parameters')
        
    ax2.legend()
    plot_idx += 1
    
    # Plot effective hedge ratio
    ax3 = fig.add_subplot(gs[plot_idx])
    ax3.plot(results.index, results['effective_hedge_ratio'], 'g-')
    ax3.set_title('Effective Time-Varying Hedge Ratio')
    plot_idx += 1
    
    # Plot spread
    ax4 = fig.add_subplot(gs[plot_idx])
    ax4.plot(results.index, results['spread'], 'b-')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax4.set_title('Spread')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 


def compare_kalman_models(
    price1: pd.Series,
    price2: pd.Series,
    models: List[str] = ['linear', 'threshold', 'regime_switch', 'log_price'],
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2
) -> pd.DataFrame:
    """
    Compare different Kalman filter models for time-varying hedge ratio estimation.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series (independent variable, X)
    price2 : pd.Series
        Second price series (dependent variable, Y)
    models : List[str], default=['linear', 'threshold', 'regime_switch', 'log_price']
        List of models to compare
    transition_covariance : float, default=1e-4
        Process noise covariance
    observation_covariance : float, default=1e-2
        Measurement noise covariance
    
    Returns
    -------
    pd.DataFrame
        Comparison of models based on log-likelihood, AIC, BIC, and spread statistics
    """
    # Align the price series
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Dictionary to store comparison metrics
    model_metrics = {}
    
    # Train and evaluate each model
    for model_name in models:
        if model_name == 'linear':
            # Fit linear model (with and without returning model)
            results, model = estimate_timevarying_hedge_ratio(
                s1, s2, 
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                return_model=True
            )
            
            # Calculate metrics
            n_params = 2  # intercept and hedge_ratio
            spread = results['spread'].values
            
        else:
            # Fit non-linear model
            results, model = estimate_nonlinear_timevarying_hedge_ratio(
                s1, s2,
                model_type=model_name,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                return_model=True
            )
            
            # Calculate number of parameters based on model type
            if model_name == 'threshold':
                n_params = 3  # intercept, beta_low, beta_high
            elif model_name in ['regime_switch', 'log_price']:
                n_params = 3  # Three state variables
                
            spread = results['spread'].values
        
        # Calculate likelihood-based metrics
        log_likelihood = model.log_likelihood
        n = len(spread)
        
        # Information criteria
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n)
        
        # Spread statistics
        mean_abs_spread = np.mean(np.abs(spread))
        std_spread = np.std(spread)
        
        # Calculate half-life of spread
        try:
            # Use our existing half-life calculation
            from src.cointegration.cointegration_tests import calculate_half_life
            half_life_result = calculate_half_life(pd.Series(spread))
            half_life = half_life_result['half_life'] if isinstance(half_life_result, dict) else half_life_result
        except:
            # Simple AR(1) estimation if the import fails
            X = np.vstack([np.ones(len(spread)-1), spread[:-1]]).T
            y = spread[1:]
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
                half_life = -np.log(2) / np.log(beta) if 0 < beta < 1 else np.nan
            except:
                half_life = np.nan
        
        # Store metrics
        model_metrics[model_name] = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'mean_abs_spread': mean_abs_spread,
            'std_spread': std_spread,
            'half_life': half_life,
            'n_params': n_params
        }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics).T
    
    # Add rank columns
    metrics_df['aic_rank'] = metrics_df['aic'].rank()
    metrics_df['bic_rank'] = metrics_df['bic'].rank()
    metrics_df['spread_rank'] = metrics_df['mean_abs_spread'].rank()
    
    # Calculate overall score (lower is better)
    metrics_df['overall_score'] = (
        metrics_df['aic_rank'] + 
        metrics_df['bic_rank'] + 
        metrics_df['spread_rank']
    )
    
    # Add recommendation
    best_model = metrics_df['overall_score'].idxmin()
    metrics_df.loc[best_model, 'recommended'] = '*'
    
    # Sort by overall score
    metrics_df = metrics_df.sort_values('overall_score')
    
    return metrics_df


def plot_model_comparison(
    price1: pd.Series,
    price2: pd.Series,
    models: List[str] = ['linear', 'threshold', 'regime_switch'],
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create visual comparison of different Kalman filter models for time-varying hedge ratios.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series (independent variable, X)
    price2 : pd.Series
        Second price series (dependent variable, Y)
    models : List[str], default=['linear', 'threshold', 'regime_switch']
        List of models to compare
    transition_covariance : float, default=1e-4
        Process noise covariance
    observation_covariance : float, default=1e-2
        Measurement noise covariance
    figsize : tuple, default=(15, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Align the price series
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Fit models and extract results
    results = {}
    
    for model_name in models:
        if model_name == 'linear':
            results[model_name] = estimate_timevarying_hedge_ratio(
                s1, s2, 
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance
            )
        else:
            results[model_name] = estimate_nonlinear_timevarying_hedge_ratio(
                s1, s2,
                model_type=model_name,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance
            )
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # Plot 1: Price Series
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(s1.index, s1, label=f'{s1.name if hasattr(s1, "name") else "Price 1"}')
    ax1.plot(s2.index, s2, label=f'{s2.name if hasattr(s2, "name") else "Price 2"}')
    ax1.set_title('Price Series')
    ax1.legend()
    
    # Plot 2: Hedge Ratios
    ax2 = fig.add_subplot(gs[1])
    
    for model_name, model_results in results.items():
        if model_name == 'linear':
            ax2.plot(model_results.index, model_results['hedge_ratio'], 
                     label=f'Linear Model')
        else:
            ax2.plot(model_results.index, model_results['effective_hedge_ratio'], 
                     label=f'{model_name.replace("_", " ").title()} Model')
    
    ax2.set_title('Comparison of Time-Varying Hedge Ratios')
    ax2.legend()
    
    # Plot 3: Spreads
    ax3 = fig.add_subplot(gs[2])
    
    for model_name, model_results in results.items():
        ax3.plot(model_results.index, model_results['spread'], 
                 label=f'{model_name.replace("_", " ").title()} Spread')
    
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_title('Comparison of Spreads')
    ax3.legend()
    
    # Calculate and display model statistics in a table
    comparison_df = compare_kalman_models(
        s1, s2, models, 
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance
    )
    
    # Display model comparison as a table
    from matplotlib.table import Table
    
    table_data = []
    columns = ['Model', 'AIC', 'BIC', 'Mean |Spread|', 'Half-Life']
    
    for model_name in comparison_df.index:
        row = [
            model_name.replace('_', ' ').title(),
            f"{comparison_df.loc[model_name, 'aic']:.2f}",
            f"{comparison_df.loc[model_name, 'bic']:.2f}",
            f"{comparison_df.loc[model_name, 'mean_abs_spread']:.4f}",
            f"{comparison_df.loc[model_name, 'half_life']:.2f}"
        ]
        table_data.append(row)
    
    # Add table to figure
    ax3.table(
        cellText=table_data,
        colLabels=columns,
        loc='bottom',
        cellLoc='center',
        bbox=[0, -0.5, 1, 0.3]
    )
    
    # Adjust layout to make room for table
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 


# Additional utility functions

def calculate_spread_metrics(spread: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for spread analysis.
    
    Parameters
    ----------
    spread : np.ndarray
        Spread time series
        
    Returns
    -------
    Dict[str, float]
        Dictionary of spread metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean'] = np.mean(spread)
    metrics['std'] = np.std(spread)
    metrics['min'] = np.min(spread)
    metrics['max'] = np.max(spread)
    metrics['median'] = np.median(spread)
    metrics['mad'] = np.mean(np.abs(spread - np.mean(spread)))  # Mean absolute deviation
    
    # Stationarity test (ADF test p-value)
    try:
        from statsmodels.tsa.stattools import adfuller
        metrics['adf_pvalue'] = adfuller(spread)[1]
        metrics['is_stationary'] = metrics['adf_pvalue'] < 0.05
    except:
        metrics['adf_pvalue'] = np.nan
        metrics['is_stationary'] = False
    
    # Calculate crossing statistics
    zero_crossings = np.where(np.diff(np.signbit(spread)))[0]
    metrics['zero_crossings'] = len(zero_crossings)
    metrics['zero_crossings_per_year'] = len(zero_crossings) * 252 / len(spread)
    
    # Calculate mean time between crossings
    if len(zero_crossings) > 1:
        metrics['mean_days_between_crossings'] = np.mean(np.diff(zero_crossings))
    else:
        metrics['mean_days_between_crossings'] = np.nan
    
    return metrics


def optimize_kalman_parameters(
    price1: pd.Series,
    price2: pd.Series,
    model_type: str = 'linear',
    param_grid: Optional[Dict] = None
) -> Dict:
    """
    Optimize Kalman filter parameters using grid search.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    model_type : str, default='linear'
        Type of Kalman filter model
    param_grid : Dict, optional
        Parameter grid to search over. If None, default grid is used.
        
    Returns
    -------
    Dict
        Best parameters and results
    """
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'transition_covariance': [1e-5, 1e-4, 1e-3, 1e-2],
            'observation_covariance': [1e-3, 1e-2, 1e-1]
        }
    
    # Align price series
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Prepare grid search
    best_ll = -np.inf
    best_params = None
    best_results = None
    best_model = None
    
    # Perform grid search
    for tc in param_grid['transition_covariance']:
        for oc in param_grid['observation_covariance']:
            # Fit model
            try:
                if model_type == 'linear':
                    results, model = estimate_timevarying_hedge_ratio(
                        s1, s2, 
                        transition_covariance=tc,
                        observation_covariance=oc,
                        return_model=True
                    )
                else:
                    results, model = estimate_nonlinear_timevarying_hedge_ratio(
                        s1, s2,
                        model_type=model_type,
                        transition_covariance=tc,
                        observation_covariance=oc,
                        return_model=True
                    )
                
                # Check if this is the best model so far
                if model.log_likelihood > best_ll:
                    best_ll = model.log_likelihood
                    best_params = {'transition_covariance': tc, 'observation_covariance': oc}
                    best_results = results
                    best_model = model
            except:
                # Skip failed parameter combinations
                continue
    
    # Return best parameters and results
    return {
        'best_params': best_params,
        'log_likelihood': best_ll,
        'results': best_results,
        'model': best_model
    }


class UnscentedKalmanFilter(KalmanFilterBase):
    """
    Unscented Kalman Filter for highly non-linear and non-Gaussian models.
    
    This implements the Unscented Kalman Filter (UKF), which uses the unscented transform
    to handle non-linearities more effectively than the Extended Kalman Filter.
    The unscented transform uses a deterministic sampling approach to capture the mean and
    covariance of the state distribution through carefully selected sigma points.
    
    Parameters
    ----------
    state_transition_function : Callable
        Function that implements the state transition model: x_{t+1} = f(x_t)
    observation_function : Callable
        Function that implements the observation model: y_t = h(x_t)
    process_noise_covariance : Union[float, np.ndarray], default=1e-4
        Covariance matrix of the process noise
    observation_noise_covariance : Union[float, np.ndarray], default=1e-2
        Covariance matrix of the observation noise
    initial_state_mean : np.ndarray, optional
        Initial state mean. If None, it will be estimated from the data.
    initial_state_covariance : np.ndarray, optional
        Initial state covariance. If None, it will be set to a diagonal matrix.
    alpha : float, default=1e-3
        Spread of sigma points around mean; usually a small positive value (1e-3 to 1)
    beta : float, default=2.0
        Prior knowledge of distribution; 2 is optimal for Gaussian distributions
    kappa : float, default=0.0
        Secondary scaling parameter, usually set to 0 or 3-n for n dimensions
        
    Attributes
    ----------
    states : np.ndarray
        Estimated states for each time point
    state_covariances : np.ndarray
        Estimated state covariances for each time point
    log_likelihood : float
        Log-likelihood of the fitted model
    """
    
    def __init__(self, 
                state_transition_function: Callable,
                observation_function: Callable,
                process_noise_covariance: Union[float, np.ndarray] = 1e-4,
                observation_noise_covariance: Union[float, np.ndarray] = 1e-2,
                initial_state_mean: Optional[np.ndarray] = None,
                initial_state_covariance: Optional[np.ndarray] = None,
                alpha: float = 1e-3,
                beta: float = 2.0,
                kappa: float = 0.0):
        """Initialize the unscented Kalman filter."""
        super().__init__()
        self.state_transition_function = state_transition_function
        self.observation_function = observation_function
        self.process_noise_covariance = process_noise_covariance
        self.observation_noise_covariance = observation_noise_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Will be set during fit
        self.n_states = None
        self.weights_mean = None
        self.weights_covariance = None
    
    def _compute_sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Compute sigma points for the unscented transform.
        
        Parameters
        ----------
        mean : np.ndarray
            Mean of the distribution
        covariance : np.ndarray
            Covariance of the distribution
            
        Returns
        -------
        np.ndarray
            Sigma points, shape (2n+1, n) where n is the state dimension
        """
        n = len(mean)
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Calculate square root of covariance matrix using Cholesky decomposition
        # Add a small diagonal term to ensure positive definiteness
        try:
            L = np.linalg.cholesky((n + lambda_) * (covariance + 1e-8 * np.eye(n)))
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition as fallback
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-8)  # Ensure positive eigenvalues
            L = eigvecs @ np.diag(np.sqrt(eigvals * (n + lambda_))) @ eigvecs.T
        
        # Create 2n+1 sigma points
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = mean
        
        for i in range(n):
            sigma_points[i+1] = mean + L[i]
            sigma_points[i+1+n] = mean - L[i]
            
        return sigma_points
    
    def _compute_weights(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weights for the mean and covariance in unscented transform.
        
        Parameters
        ----------
        n_states : int
            Number of state variables
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Weights for mean and covariance calculations
        """
        n = n_states
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # Weights for the mean
        weights_mean = np.zeros(2*n + 1)
        weights_mean[0] = lambda_ / (n + lambda_)
        weights_mean[1:] = 1 / (2 * (n + lambda_))
        
        # Weights for the covariance
        weights_cov = np.copy(weights_mean)
        weights_cov[0] = weights_mean[0] + (1 - self.alpha**2 + self.beta)
        
        return weights_mean, weights_cov
    
    def fit(self, 
           observations: np.ndarray, 
           n_states: int,
           exog: Optional[np.ndarray] = None,
           dates: Optional[pd.DatetimeIndex] = None) -> 'UnscentedKalmanFilter':
        """
        Fit the unscented Kalman filter to the data.
        
        Parameters
        ----------
        observations : np.ndarray
            Observation data, shape (n_samples, n_observations)
        n_states : int
            Number of state variables
        exog : np.ndarray, optional
            Exogenous variables, shape (n_samples, n_exog_variables)
        dates : pd.DatetimeIndex, optional
            Dates for the observations
            
        Returns
        -------
        UnscentedKalmanFilter
            Fitted model
        """
        n_samples = observations.shape[0]
        self.n_states = n_states
        
        # Initialize weights for unscented transform
        self.weights_mean, self.weights_covariance = self._compute_weights(n_states)
        
        # Initialize state mean and covariance
        if self.initial_state_mean is None:
            # Default initial state: zeros
            self.initial_state_mean = np.zeros(n_states)
        
        if self.initial_state_covariance is None:
            # Default initial covariance: identity
            self.initial_state_covariance = np.eye(n_states)
        
        # Initialize process and observation noise covariances
        if isinstance(self.process_noise_covariance, (int, float)):
            self.process_noise_covariance = self.process_noise_covariance * np.eye(n_states)
        
        if isinstance(self.observation_noise_covariance, (int, float)):
            n_obs = observations.shape[1] if len(observations.shape) > 1 else 1
            self.observation_noise_covariance = self.observation_noise_covariance * np.eye(n_obs)
        
        # Prepare storage for states and covariances
        self.states = np.zeros((n_samples, n_states))
        self.state_covariances = np.zeros((n_samples, n_states, n_states))
        
        # Run filter
        filtered_states, filtered_covs, self.log_likelihood = self._unscented_kalman_filter(
            observations=observations,
            exog=exog
        )
        
        self.states = filtered_states
        self.state_covariances = filtered_covs
        self.is_fitted = True
        
        return self
    
    def _unscented_kalman_filter(self, 
                                observations: np.ndarray,
                                exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run the unscented Kalman filter algorithm.
        
        Parameters
        ----------
        observations : np.ndarray
            Observation data
        exog : np.ndarray, optional
            Exogenous variables
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Filtered states, state covariances, and log likelihood
        """
        n_samples = observations.shape[0]
        n_obs = observations.shape[1] if len(observations.shape) > 1 else 1
        
        # Reshape singleton observation dimensions for consistent processing
        if len(observations.shape) == 1:
            observations = observations.reshape(-1, 1)
        
        # Initialize storage
        filtered_states = np.zeros((n_samples, self.n_states))
        filtered_covs = np.zeros((n_samples, self.n_states, self.n_states))
        self.innovations = np.zeros((n_samples, n_obs))
        self.innovation_covariances = np.zeros((n_samples, n_obs, n_obs))
        log_likelihood = 0.0
        
        # Initialize state and covariance
        state = self.initial_state_mean
        state_cov = self.initial_state_covariance
        
        for t in range(n_samples):
            # Get current exogenous variables if available
            current_exog = None if exog is None else exog[t]
            
            # Prediction step (time update)
            # 1. Generate sigma points from current state and covariance
            sigma_points = self._compute_sigma_points(state, state_cov)
            
            # 2. Propagate sigma points through state transition function
            transformed_sigma_points = np.array([
                self.state_transition_function(sigma_point, current_exog) 
                for sigma_point in sigma_points
            ])
            
            # 3. Calculate predicted state and covariance
            predicted_state = np.sum(self.weights_mean[:, np.newaxis] * transformed_sigma_points, axis=0)
            predicted_cov = self.process_noise_covariance.copy()  # Start with process noise
            
            # Add weighted deviation from each sigma point
            for i in range(len(sigma_points)):
                dev = transformed_sigma_points[i] - predicted_state
                predicted_cov += self.weights_covariance[i] * np.outer(dev, dev)
            
            # Correction step (measurement update)
            # 1. Generate new sigma points from predicted state and covariance
            sigma_points = self._compute_sigma_points(predicted_state, predicted_cov)
            
            # 2. Transform sigma points through observation function
            predicted_observations = np.array([
                self.observation_function(sigma_point, current_exog) 
                for sigma_point in sigma_points
            ])
            if len(predicted_observations.shape) == 1:
                predicted_observations = predicted_observations.reshape(-1, 1)
            
            # 3. Calculate predicted observation and innovation covariance
            predicted_obs = np.sum(self.weights_mean[:, np.newaxis] * predicted_observations, axis=0)
            innovation_cov = self.observation_noise_covariance.copy()  # Start with observation noise
            
            cross_cov = np.zeros((self.n_states, n_obs))
            
            for i in range(len(sigma_points)):
                dev_obs = predicted_observations[i] - predicted_obs
                dev_state = sigma_points[i] - predicted_state
                
                innovation_cov += self.weights_covariance[i] * np.outer(dev_obs, dev_obs)
                cross_cov += self.weights_covariance[i] * np.outer(dev_state, dev_obs)
            
            # Calculate innovation (measurement residual)
            innovation = observations[t] - predicted_obs
            
            # Store innovation and innovation covariance
            self.innovations[t] = innovation
            self.innovation_covariances[t] = innovation_cov
            
            # 4. Calculate Kalman gain
            try:
                kalman_gain = cross_cov @ np.linalg.inv(innovation_cov)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if regular inverse fails
                kalman_gain = cross_cov @ np.linalg.pinv(innovation_cov)
            
            # 5. Update state and covariance
            state = predicted_state + kalman_gain @ innovation
            state_cov = predicted_cov - kalman_gain @ innovation_cov @ kalman_gain.T
            
            # Ensure state covariance stays positive definite
            state_cov = (state_cov + state_cov.T) / 2  # Make symmetric
            eigvals = np.linalg.eigvalsh(state_cov)
            if np.min(eigvals) < 1e-8:
                # Add small diagonal term if needed
                state_cov += 1e-8 * np.eye(self.n_states)
            
            # Store results
            filtered_states[t] = state
            filtered_covs[t] = state_cov
            
            # Update log likelihood
            try:
                # Multivariate normal log-likelihood
                log_det = np.log(np.linalg.det(innovation_cov))
                if not np.isfinite(log_det) or log_det < -30:
                    # If determinant is too small, use pseudo-determinant
                    eigvals = np.linalg.eigvalsh(innovation_cov)
                    positive_eigvals = eigvals[eigvals > 1e-8]
                    log_det = np.sum(np.log(positive_eigvals))
                
                mahalanobis = innovation @ np.linalg.solve(innovation_cov, innovation)
                curr_ll = -0.5 * (log_det + mahalanobis + n_obs * np.log(2 * np.pi))
                log_likelihood += curr_ll
            except:
                # Skip likelihood calculation if there's a numerical issue
                pass
        
        return filtered_states, filtered_covs, log_likelihood
    
    def predict(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict using the fitted unscented Kalman filter.
        
        Parameters
        ----------
        exog : np.ndarray, optional
            Exogenous variables for prediction
            
        Returns
        -------
        np.ndarray
            Predicted observations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        n_samples = len(self.states)
        predictions = np.zeros((n_samples, 1))  # Initialize with single dimension output
        
        for t in range(n_samples):
            current_exog = None if exog is None else exog[t]
            predictions[t] = self.observation_function(self.states[t], current_exog)
            
        return predictions


def estimate_ukf_timevarying_hedge_ratio(
    price1: pd.Series,
    price2: pd.Series,
    model_type: str = 'stochastic_volatility',
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    return_model: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, UnscentedKalmanFilter]]:
    """
    Estimate time-varying hedge ratio using Unscented Kalman Filter for non-Gaussian models.
    
    This function implements several specialized non-Gaussian models for the hedge ratio,
    including stochastic volatility, Student-t innovations, and jump diffusion models.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    model_type : str, default='stochastic_volatility'
        Type of non-Gaussian model to use. Options:
        - 'stochastic_volatility': Model with time-varying volatility
        - 'student_t': Model with Student-t distributed innovations
        - 'jump_diffusion': Model with occasional jumps in the hedge ratio
    transition_covariance : float, default=1e-4
        State transition noise covariance
    observation_covariance : float, default=1e-2
        Observation noise covariance
    return_model : bool, default=False
        Whether to return the fitted model along with results
        
    Returns
    -------
    pd.DataFrame or tuple
        If return_model=False, returns DataFrame with results
        If return_model=True, returns tuple (DataFrame, UnscentedKalmanFilter)
    """
    # Validate inputs
    if not isinstance(price1, pd.Series) or not isinstance(price2, pd.Series):
        raise TypeError("Price inputs must be pandas Series")
    
    # Align series on same index
    price1, price2 = price1.align(price2, join='inner')
    
    if len(price1) == 0:
        raise ValueError("No overlapping dates in price series")
    
    # Prepare data
    dates = price1.index
    observations = price2.values.reshape(-1, 1)  # Shape (n_samples, 1)
    exog = price1.values.reshape(-1, 1)  # Shape (n_samples, 1)
    
    # Define model-specific functions
    if model_type == 'stochastic_volatility':
        # State: [intercept, hedge_ratio, log_volatility]
        n_states = 3
        
        def state_transition(state, exog=None):
            """State transition with autoregressive volatility."""
            intercept, hedge_ratio, log_vol = state
            # AR(1) model for log volatility with mean reversion
            phi = 0.98  # Persistence parameter
            target_log_vol = -1.0  # Target log volatility
            new_log_vol = (1 - phi) * target_log_vol + phi * log_vol
            return np.array([intercept, hedge_ratio, new_log_vol])
        
        def observation_function(state, exog):
            """Observation model with stochastic volatility."""
            intercept, hedge_ratio, log_vol = state
            vol = np.exp(0.5 * log_vol)  # Convert log volatility to standard deviation
            # Ignore vol in observation for now, it affects the observation noise
            return intercept + hedge_ratio * exog.squeeze()
        
        # Initial state [intercept, hedge_ratio, log_volatility]
        init_intercept = 0.0
        init_hedge_ratio = 1.0
        init_log_vol = 0.0
        initial_state = np.array([init_intercept, init_hedge_ratio, init_log_vol])
        
        # Initial state covariance
        initial_cov = np.diag([0.1, 0.1, 1.0])
        
        # Process noise covariance: diagonal with different variances
        proc_noise = np.diag([0.01 * transition_covariance, 
                               transition_covariance, 
                               0.1])  # Larger variance for log volatility
        
    elif model_type == 'student_t':
        # State: [intercept, hedge_ratio, df_parameter]
        n_states = 3
        
        def state_transition(state, exog=None):
            """State transition with slowly varying degrees of freedom."""
            intercept, hedge_ratio, df_param = state
            # Constrain df_param to reasonable range (3 to 30)
            df_param = max(3.0, min(df_param, 30.0))
            return np.array([intercept, hedge_ratio, df_param])
        
        def observation_function(state, exog):
            """Observation model with Student-t innovations (approx)."""
            intercept, hedge_ratio, df_param = state
            return intercept + hedge_ratio * exog.squeeze()
        
        # Initial state [intercept, hedge_ratio, df_parameter]
        initial_state = np.array([0.0, 1.0, 5.0])
        
        # Initial state covariance
        initial_cov = np.diag([0.1, 0.1, 2.0])
        
        # Process noise covariance
        proc_noise = np.diag([0.01 * transition_covariance, 
                               transition_covariance, 
                               0.05])  # Small variance for df parameter
        
    elif model_type == 'jump_diffusion':
        # State: [intercept, hedge_ratio, jump_probability]
        n_states = 3
        
        def state_transition(state, exog=None):
            """State transition with jump probability."""
            intercept, hedge_ratio, jump_prob = state
            # Constrain jump probability to [0, 0.2] range
            jump_prob = max(0.0, min(jump_prob, 0.2))
            return np.array([intercept, hedge_ratio, jump_prob])
        
        def observation_function(state, exog):
            """Observation model (jump effect handled in noise)."""
            intercept, hedge_ratio, jump_prob = state
            return intercept + hedge_ratio * exog.squeeze()
        
        # Initial state [intercept, hedge_ratio, jump_probability]
        initial_state = np.array([0.0, 1.0, 0.01])
        
        # Initial state covariance
        initial_cov = np.diag([0.1, 0.1, 0.01])
        
        # Process noise covariance
        proc_noise = np.diag([0.01 * transition_covariance, 
                               transition_covariance, 
                               0.001])  # Small variance for jump probability
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create and fit the Unscented Kalman Filter
    ukf = UnscentedKalmanFilter(
        state_transition_function=state_transition,
        observation_function=observation_function,
        process_noise_covariance=proc_noise,
        observation_noise_covariance=observation_covariance,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_cov,
        alpha=1e-3,  # Default UKF parameters
        beta=2.0,
        kappa=0.0
    )
    
    ukf.fit(observations, n_states=n_states, exog=exog, dates=dates)
    
    # Extract results
    states = ukf.states
    
    # Create DataFrame with results
    results = pd.DataFrame(index=dates)
    results['intercept'] = states[:, 0]
    results['hedge_ratio'] = states[:, 1]
    
    if model_type == 'stochastic_volatility':
        results['log_volatility'] = states[:, 2]
        results['volatility'] = np.exp(0.5 * states[:, 2])
    elif model_type == 'student_t':
        results['degrees_of_freedom'] = states[:, 2]
    elif model_type == 'jump_diffusion':
        results['jump_probability'] = states[:, 2]
    
    # Calculate spread
    results['spread'] = price2 - (results['intercept'] + results['hedge_ratio'] * price1)
    
    if return_model:
        return results, ukf
    else:
        return results


def plot_ukf_hedge_ratio(
    results: pd.DataFrame,
    price1: Optional[pd.Series] = None,
    price2: Optional[pd.Series] = None,
    model_type: str = 'stochastic_volatility',
    figsize: Tuple[int, int] = (14, 16),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot results of Unscented Kalman Filter hedge ratio estimation.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from estimate_ukf_timevarying_hedge_ratio
    price1 : pd.Series, optional
        First price series, for plotting prices
    price2 : pd.Series, optional
        Second price series, for plotting prices
    model_type : str, default='stochastic_volatility'
        Type of non-Gaussian model used
    figsize : tuple, default=(14, 16)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Figure with plots
    """
    if model_type == 'stochastic_volatility':
        n_rows = 5 if price1 is not None and price2 is not None else 4
    else:
        n_rows = 4 if price1 is not None and price2 is not None else 3
    
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    ax_idx = 0
    
    # Plot prices if provided
    if price1 is not None and price2 is not None:
        ax = axes[ax_idx]
        ax_idx += 1
        ax.plot(price1.index, price1, label='Price 1')
        ax.plot(price2.index, price2, label='Price 2')
        ax.set_title('Price Series')
        ax.legend()
        ax.grid(True)
    
    # Plot hedge ratio
    ax = axes[ax_idx]
    ax_idx += 1
    ax.plot(results.index, results['hedge_ratio'])
    ax.set_title('Time-Varying Hedge Ratio')
    ax.grid(True)
    
    # Plot intercept
    ax = axes[ax_idx]
    ax_idx += 1
    ax.plot(results.index, results['intercept'])
    ax.set_title('Time-Varying Intercept')
    ax.grid(True)
    
    # Plot model-specific state variable
    ax = axes[ax_idx]
    ax_idx += 1
    if model_type == 'stochastic_volatility':
        ax.plot(results.index, results['volatility'])
        ax.set_title('Time-Varying Volatility')
    elif model_type == 'student_t':
        ax.plot(results.index, results['degrees_of_freedom'])
        ax.set_title('Time-Varying Degrees of Freedom')
    elif model_type == 'jump_diffusion':
        ax.plot(results.index, results['jump_probability'])
        ax.set_title('Jump Probability')
    ax.grid(True)
    
    # Plot spread
    if model_type == 'stochastic_volatility':
        ax = axes[ax_idx]
        ax_idx += 1
        
        # Plot spread with volatility-based bands
        spread = results['spread']
        vol = results['volatility']
        
        ax.plot(results.index, spread, label='Spread')
        ax.fill_between(
            results.index,
            -2 * vol, 2 * vol,
            color='gray', alpha=0.3,
            label='±2σ Band'
        )
        ax.set_title('Spread with Time-Varying Volatility Bands')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compare_kalman_filter_with_reference(
    price1: pd.Series,
    price2: pd.Series,
    reference_implementation: str = 'pykalman',
    model_type: str = 'linear',
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    plot_results: bool = True,
    figsize: Tuple[int, int] = (15, 12)
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Compare our Kalman filter implementation against a reference implementation.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    reference_implementation : str, default='pykalman'
        Name of reference implementation: 'pykalman' or 'statsmodels'
    model_type : str, default='linear'
        Type of Kalman filter model to compare: 'linear', 'extended', or 'unscented'
    transition_covariance : float, default=1e-4
        State transition noise covariance
    observation_covariance : float, default=1e-2
        Observation noise covariance
    plot_results : bool, default=True
        Whether to plot comparison results
    figsize : tuple, default=(15, 12)
        Figure size for plots
        
    Returns
    -------
    Dict
        Dictionary with comparison metrics and results
    """
    # Ensure prices are aligned
    price1, price2 = price1.align(price2, join='inner')
    dates = price1.index
    
    # Run our implementation
    if model_type == 'linear':
        our_results, our_model = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    elif model_type == 'extended':
        our_results, our_model = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2,
            model_type='threshold',
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    elif model_type == 'unscented':
        our_results, our_model = estimate_ukf_timevarying_hedge_ratio(
            price1, price2,
            model_type='stochastic_volatility',
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Run reference implementation
    if reference_implementation == 'pykalman':
        try:
            from pykalman import KalmanFilter as PyKalmanFilter
            
            # Prepare data for pykalman
            observations = price2.values
            design = price1.values.reshape(-1, 1)
            
            # For linear model, use standard KalmanFilter
            if model_type == 'linear':
                kf = PyKalmanFilter(
                    transition_matrices=np.eye(2),
                    observation_matrices=np.vstack([1, design]).T,
                    transition_covariance=transition_covariance * np.eye(2),
                    observation_covariance=observation_covariance,
                    initial_state_mean=np.zeros(2),
                    initial_state_covariance=np.eye(2),
                    em_vars=['transition_covariance', 'observation_covariance']
                )
                
                # Fit model
                ref_states, ref_covs = kf.filter(observations)
                
                # Extract results
                ref_intercept = ref_states[:, 0]
                ref_hedge_ratio = ref_states[:, 1]
                
            else:
                # For non-linear models, pykalman doesn't have direct equivalents
                # We'll use a simplified linear model for comparison
                kf = PyKalmanFilter(
                    transition_matrices=np.eye(2),
                    observation_matrices=np.vstack([1, design]).T,
                    transition_covariance=transition_covariance * np.eye(2),
                    observation_covariance=observation_covariance,
                    initial_state_mean=np.zeros(2),
                    initial_state_covariance=np.eye(2)
                )
                
                # Fit model
                ref_states, ref_covs = kf.filter(observations)
                
                # Extract results
                ref_intercept = ref_states[:, 0]
                ref_hedge_ratio = ref_states[:, 1]
                
            # Create reference results dataframe
            ref_results = pd.DataFrame({
                'intercept': ref_intercept,
                'hedge_ratio': ref_hedge_ratio
            }, index=dates)
            
            # Calculate spread
            ref_results['spread'] = price2 - (ref_results['intercept'] + ref_results['hedge_ratio'] * price1)
            
        except ImportError:
            print("pykalman not installed. Using statsmodels instead.")
            reference_implementation = 'statsmodels'
    
    if reference_implementation == 'statsmodels':
        try:
            import statsmodels.api as sm
            
            # Prepare data for statsmodels
            endog = price2.values
            exog = sm.add_constant(price1.values)
            
            # For linear model, use statespace KalmanFilter
            if model_type == 'linear':
                # Define state space model
                k_states = 2  # [intercept, hedge_ratio]
                
                # Initialize state space model
                mod = sm.tsa.statespace.MLEModel(
                    endog, exog, k_states=k_states
                )
                
                # Set up transition equation: state[t] = state[t-1] + state_noise
                mod['transition'] = np.eye(k_states)
                mod['state_cov'] = transition_covariance * np.eye(k_states)
                
                # Set up observation equation: y[t] = intercept + hedge_ratio * x[t] + obs_noise
                mod['design'] = exog
                mod['obs_cov'] = observation_covariance
                
                # Initialize state
                mod.initialize_approximate_diffuse()
                
                # Run filter
                ref_states = mod.filter().filtered_state
                
                # Extract results
                ref_intercept = ref_states[0, :]  # Rows are states, columns are time
                ref_hedge_ratio = ref_states[1, :]
                
            else:
                # For non-linear models, statsmodels doesn't have direct equivalents
                # We'll use local linear trend model as an approximation
                mod = sm.tsa.UnobservedComponents(
                    price2.values, exog=price1.values, level='local linear trend'
                )
                
                # Fit model
                res = mod.fit(disp=False)
                
                # Extract results (using level as intercept, slope as hedge ratio)
                ref_intercept = res.states.filtered['level']
                ref_hedge_ratio = res.states.filtered['slope']
            
            # Create reference results dataframe
            ref_results = pd.DataFrame({
                'intercept': ref_intercept,
                'hedge_ratio': ref_hedge_ratio
            }, index=dates)
            
            # Calculate spread
            ref_results['spread'] = price2 - (ref_results['intercept'] + ref_results['hedge_ratio'] * price1)
            
        except ImportError:
            print("statsmodels not installed. Cannot run reference implementation.")
            return {
                'error': 'Reference implementation not available',
                'our_results': our_results
            }
    
    # Calculate comparison metrics
    comparison = {}
    
    # Calculate RMSE between implementations
    rmse_intercept = np.sqrt(np.mean((our_results['intercept'] - ref_results['intercept']) ** 2))
    rmse_hedge_ratio = np.sqrt(np.mean((our_results['hedge_ratio'] - ref_results['hedge_ratio']) ** 2))
    rmse_spread = np.sqrt(np.mean((our_results['spread'] - ref_results['spread']) ** 2))
    
    # Calculate correlation between implementations
    corr_intercept = np.corrcoef(our_results['intercept'], ref_results['intercept'])[0, 1]
    corr_hedge_ratio = np.corrcoef(our_results['hedge_ratio'], ref_results['hedge_ratio'])[0, 1]
    corr_spread = np.corrcoef(our_results['spread'], ref_results['spread'])[0, 1]
    
    # Store metrics
    comparison['rmse_intercept'] = rmse_intercept
    comparison['rmse_hedge_ratio'] = rmse_hedge_ratio
    comparison['rmse_spread'] = rmse_spread
    comparison['corr_intercept'] = corr_intercept
    comparison['corr_hedge_ratio'] = corr_hedge_ratio
    comparison['corr_spread'] = corr_spread
    
    # Calculate log-likelihood in our model
    comparison['our_log_likelihood'] = our_model.log_likelihood
    
    # Calculate our model diagnostics
    comparison['our_diagnostics'] = our_model.get_innovation_statistics()
    
    # Store results for comparison
    comparison['our_results'] = our_results
    comparison['ref_results'] = ref_results
    
    # Plot results if requested
    if plot_results:
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot prices
        ax = axes[0]
        ax.plot(dates, price1, label='Price 1')
        ax.plot(dates, price2, label='Price 2')
        ax.set_title(f'Price Series - {model_type.capitalize()} Kalman Filter')
        ax.legend()
        ax.grid(True)
        
        # Plot hedge ratio comparison
        ax = axes[1]
        ax.plot(dates, our_results['hedge_ratio'], label='Our Implementation')
        ax.plot(dates, ref_results['hedge_ratio'], label=f'Reference ({reference_implementation})', linestyle='--')
        ax.set_title(f'Hedge Ratio Comparison (RMSE: {rmse_hedge_ratio:.4f}, Corr: {corr_hedge_ratio:.4f})')
        ax.legend()
        ax.grid(True)
        
        # Plot intercept comparison
        ax = axes[2]
        ax.plot(dates, our_results['intercept'], label='Our Implementation')
        ax.plot(dates, ref_results['intercept'], label=f'Reference ({reference_implementation})', linestyle='--')
        ax.set_title(f'Intercept Comparison (RMSE: {rmse_intercept:.4f}, Corr: {corr_intercept:.4f})')
        ax.legend()
        ax.grid(True)
        
        # Plot spread comparison
        ax = axes[3]
        ax.plot(dates, our_results['spread'], label='Our Implementation')
        ax.plot(dates, ref_results['spread'], label=f'Reference ({reference_implementation})', linestyle='--')
        ax.set_title(f'Spread Comparison (RMSE: {rmse_spread:.4f}, Corr: {corr_spread:.4f})')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        comparison['figure'] = fig
    
    return comparison

def optimize_adaptive_kalman_filter_parameters(
    price1: pd.Series,
    price2: pd.Series,
    validation_size: float = 0.3,
    model_type: str = 'linear',
    param_grid: Optional[Dict] = None,
    n_iter: int = 20,
    random_state: int = 42,
    metric: str = 'combined'
) -> Dict[str, Any]:
    """
    Optimize Kalman filter parameters using cross-validation to improve forecasting.
    
    This function implements adaptive parameter estimation by dividing the data into
    training and validation sets and optimizing parameters based on performance metrics.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    validation_size : float, default=0.3
        Fraction of data to use for validation
    model_type : str, default='linear'
        Type of Kalman filter to optimize: 'linear', 'extended', or 'unscented'
    param_grid : Dict, optional
        Grid of parameters to search. If None, uses default grid.
    n_iter : int, default=20
        Number of iterations for randomized search
    random_state : int, default=42
        Random seed for reproducibility
    metric : str, default='combined'
        Metric to optimize: 'mse', 'mae', 'likelihood', or 'combined'
        
    Returns
    -------
    Dict
        Dictionary with optimized parameters and results
    """
    # Align price series
    price1, price2 = price1.align(price2, join='inner')
    
    # Create training/validation split
    n_samples = len(price1)
    split_idx = int(n_samples * (1 - validation_size))
    
    train_price1 = price1.iloc[:split_idx]
    train_price2 = price2.iloc[:split_idx]
    
    val_price1 = price1.iloc[split_idx:]
    val_price2 = price2.iloc[split_idx:]
    
    # Define default parameter grid if not provided
    if param_grid is None:
        if model_type == 'linear':
            param_grid = {
                'transition_covariance': np.logspace(-6, -2, 20),
                'observation_covariance': np.logspace(-4, 0, 20),
                'em_iterations': [0, 1, 2, 5]
            }
        elif model_type == 'extended':
            param_grid = {
                'transition_covariance': np.logspace(-6, -2, 10),
                'observation_covariance': np.logspace(-4, 0, 10),
                'model_type': ['threshold', 'regime_switch']
            }
        elif model_type == 'unscented':
            param_grid = {
                'transition_covariance': np.logspace(-6, -2, 10),
                'observation_covariance': np.logspace(-4, 0, 10),
                'model_type': ['stochastic_volatility', 'student_t', 'jump_diffusion'],
                'alpha': [1e-4, 1e-3, 1e-2],
                'beta': [0.0, 2.0]
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Initialize results storage
    results = []
    
    # Generate parameter combinations
    param_combinations = []
    for _ in range(n_iter):
        params = {}
        for param_name, param_values in param_grid.items():
            params[param_name] = param_values[rng.randint(0, len(param_values))]
        param_combinations.append(params)
    
    # Evaluate parameter combinations
    for i, params in enumerate(param_combinations):
        try:
            print(f"Evaluating parameter set {i+1}/{len(param_combinations)}: {params}")
            
            # Train model on training set
            if model_type == 'linear':
                train_results, train_model = estimate_timevarying_hedge_ratio(
                    train_price1, train_price2,
                    transition_covariance=params['transition_covariance'],
                    observation_covariance=params['observation_covariance'],
                    em_iterations=params['em_iterations'] if 'em_iterations' in params else 0,
                    return_model=True
                )
            elif model_type == 'extended':
                train_results, train_model = estimate_nonlinear_timevarying_hedge_ratio(
                    train_price1, train_price2,
                    model_type=params['model_type'] if 'model_type' in params else 'threshold',
                    transition_covariance=params['transition_covariance'],
                    observation_covariance=params['observation_covariance'],
                    return_model=True
                )
            elif model_type == 'unscented':
                train_results, train_model = estimate_ukf_timevarying_hedge_ratio(
                    train_price1, train_price2,
                    model_type=params['model_type'] if 'model_type' in params else 'stochastic_volatility',
                    transition_covariance=params['transition_covariance'],
                    observation_covariance=params['observation_covariance'],
                    return_model=True
                )
            
            # Apply model to validation set
            # For validation, we need to get the last state from training and use it for prediction
            last_state = train_model.states[-1]
            last_state_cov = train_model.state_covariances[-1]
            
            # Create model for validation set with initial state from end of training
            if model_type == 'linear':
                val_model = LinearKalmanFilter(
                    transition_covariance=params['transition_covariance'],
                    observation_covariance=params['observation_covariance'],
                    initial_state_mean=last_state,
                    initial_state_covariance=last_state_cov
                )
                
                # Prepare data for validation
                val_X = np.column_stack([np.ones(len(val_price1)), val_price1.values])
                val_y = val_price2.values.reshape(-1, 1)
                
                # Run filter on validation set
                val_model.fit(val_y, val_X)
                
                # Calculate predictions
                val_predictions = val_model.predict(val_X)
                
                # Calculate validation metrics
                val_mse = np.mean((val_y - val_predictions) ** 2)
                val_mae = np.mean(np.abs(val_y - val_predictions))
                val_likelihood = val_model.log_likelihood
                
            elif model_type in ['extended', 'unscented']:
                # For more complex models, we'll generate predictions directly
                # This is a simplification - ideally we'd refit the model with validation data
                
                # Generate hedge ratio and intercept predictions
                val_hedge_ratio = last_state[1]  # Assuming state is [intercept, hedge_ratio, ...]
                val_intercept = last_state[0]
                
                # Calculate predictions
                val_predictions = val_intercept + val_hedge_ratio * val_price1.values
                
                # Calculate validation metrics
                val_mse = np.mean((val_price2.values - val_predictions) ** 2)
                val_mae = np.mean(np.abs(val_price2.values - val_predictions))
                val_likelihood = -val_mse  # Proxy, since we didn't run the filter
            
            # Combine metrics if needed
            if metric == 'mse':
                val_metric = -val_mse  # Negative because we want to maximize
            elif metric == 'mae':
                val_metric = -val_mae  # Negative because we want to maximize
            elif metric == 'likelihood':
                val_metric = val_likelihood
            elif metric == 'combined':
                # Normalized combination of metrics
                normalized_mse = -val_mse / np.var(val_price2.values)
                normalized_likelihood = val_likelihood / len(val_price2)
                val_metric = normalized_mse + normalized_likelihood
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            # Store results
            results.append({
                'params': params,
                'train_likelihood': train_model.log_likelihood,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_likelihood': val_likelihood,
                'val_metric': val_metric
            })
            
        except Exception as e:
            print(f"Error evaluating parameter set: {e}")
            continue
    
    # Find best parameters
    if results:
        # Sort by validation metric
        results.sort(key=lambda x: x['val_metric'], reverse=True)
        best_params = results[0]['params']
        
        # Fit model on entire dataset with best parameters
        if model_type == 'linear':
            full_results, full_model = estimate_timevarying_hedge_ratio(
                price1, price2,
                transition_covariance=best_params['transition_covariance'],
                observation_covariance=best_params['observation_covariance'],
                em_iterations=best_params['em_iterations'] if 'em_iterations' in best_params else 0,
                return_model=True
            )
        elif model_type == 'extended':
            full_results, full_model = estimate_nonlinear_timevarying_hedge_ratio(
                price1, price2,
                model_type=best_params['model_type'] if 'model_type' in best_params else 'threshold',
                transition_covariance=best_params['transition_covariance'],
                observation_covariance=best_params['observation_covariance'],
                return_model=True
            )
        elif model_type == 'unscented':
            full_results, full_model = estimate_ukf_timevarying_hedge_ratio(
                price1, price2,
                model_type=best_params['model_type'] if 'model_type' in best_params else 'stochastic_volatility',
                transition_covariance=best_params['transition_covariance'],
                observation_covariance=best_params['observation_covariance'],
                return_model=True
            )
        
        # Run diagnostics
        diagnostics = full_model.calculate_filter_performance_metrics()
        
        # Create summary
        return {
            'best_params': best_params,
            'all_results': results,
            'full_results': full_results,
            'full_model': full_model,
            'diagnostics': diagnostics
        }
    else:
        return {
            'error': 'No valid parameter combinations found'
        }


def create_kalman_spread_analyzer(
    price1: pd.Series,
    price2: pd.Series,
    model_type: str = 'linear',
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    window_size: int = 20,
    use_volatility_adjustment: bool = False
) -> Tuple[pd.DataFrame, Any]:
    """
    Create a spread analyzer that uses Kalman filter for dynamic hedge ratio estimation.
    
    This function integrates the Kalman filter with the SpreadAnalyzer component to provide
    a comprehensive framework for dynamic spread calculation and Z-score generation.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    model_type : str, default='linear'
        Type of Kalman filter to use: 'linear', 'extended', 'unscented'
    transition_covariance : float, default=1e-4
        Transition noise covariance parameter
    observation_covariance : float, default=1e-2
        Observation noise covariance parameter
    window_size : int, default=20
        Window size for Z-score calculation
    use_volatility_adjustment : bool, default=False
        Whether to use volatility adjustment for Z-score calculation
        
    Returns
    -------
    Tuple[pd.DataFrame, Any]
        DataFrame with analysis results and SpreadAnalyzer instance
    """
    try:
        from src.spread_analytics.spread_analyzer import SpreadAnalyzer
    except ImportError:
        # Create a simple SpreadAnalyzer if the module is not available
        class SpreadAnalyzer:
            def __init__(self):
                pass
            
            def calculate_zscore(self, spread, window=20, method='rolling'):
                """Calculate z-score for a spread series using rolling window"""
                if method == 'rolling':
                    mean = spread.rolling(window=window).mean()
                    std = spread.rolling(window=window).std()
                    return (spread - mean) / std
                else:
                    # Expanding window
                    mean = spread.expanding().mean()
                    std = spread.expanding().std()
                    return (spread - mean) / std
            
            def calculate_half_life(self, spread):
                """Estimate mean reversion half-life of a spread"""
                # Basic half-life calculation using OLS
                lagged_spread = spread.shift(1).dropna()
                delta_spread = (spread - spread.shift(1)).dropna()
                
                # Regression: delta_spread = alpha + beta * lagged_spread + error
                X = lagged_spread.values.reshape(-1, 1)
                y = delta_spread.values
                
                # Add constant
                X = np.column_stack([np.ones(X.shape[0]), X])
                
                # Fit OLS
                beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
                
                # Calculate half-life: -ln(2) / ln(1 + beta)
                half_life = -np.log(2) / np.log(1 + beta)
                
                return max(1.0, half_life)  # Min half-life of 1.0
    
    # Align price series
    price1, price2 = price1.align(price2, join='inner')
    
    # Create SpreadAnalyzer
    spread_analyzer = SpreadAnalyzer()
    
    # Estimate dynamic hedge ratio using appropriate Kalman filter
    if model_type == 'linear':
        results, model = estimate_timevarying_hedge_ratio(
            price1, price2,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    elif model_type == 'extended':
        results, model = estimate_nonlinear_timevarying_hedge_ratio(
            price1, price2,
            model_type='threshold',
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    elif model_type == 'unscented':
        results, model = estimate_ukf_timevarying_hedge_ratio(
            price1, price2,
            model_type='stochastic_volatility',
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            return_model=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Extract spread
    spread = results['spread']
    
    # Calculate Z-score
    z_score = spread_analyzer.calculate_zscore(pd.Series(spread, index=price1.index), window=window_size)
    results['zscore'] = z_score
    
    # Calculate half-life
    try:
        half_life = spread_analyzer.calculate_half_life(pd.Series(spread, index=price1.index))
        results['half_life'] = half_life
    except:
        # If half-life calculation fails, set to default value
        results['half_life'] = np.nan
    
    # Calculate volatility-adjusted Z-score if requested
    if use_volatility_adjustment and model_type == 'unscented' and 'volatility' in results.columns:
        # For UKF with stochastic volatility model
        vol_adjusted_zscore = spread / results['volatility']
        results['volatility_adjusted_zscore'] = vol_adjusted_zscore
    elif use_volatility_adjustment:
        # For other models, estimate GARCH volatility
        try:
            from arch import arch_model
            
            # Fit GARCH model to spread
            garch_model = arch_model(spread.dropna(), vol='GARCH', p=1, q=1)
            garch_result = garch_model.fit(disp='off')
            
            # Get conditional volatility
            vol = pd.Series(np.sqrt(garch_result.conditional_volatility), index=spread.dropna().index)
            
            # Reindex to original spread index
            vol = vol.reindex(spread.index)
            
            # Calculate volatility-adjusted Z-score
            vol_adjusted_zscore = spread / vol
            results['garch_volatility'] = vol
            results['volatility_adjusted_zscore'] = vol_adjusted_zscore
        except:
            # If GARCH estimation fails, fall back to rolling volatility
            rolling_vol = pd.Series(spread, index=price1.index).rolling(window=window_size).std()
            vol_adjusted_zscore = spread / rolling_vol
            results['rolling_volatility'] = rolling_vol
            results['volatility_adjusted_zscore'] = vol_adjusted_zscore
    
    # Add filter diagnostics
    try:
        innovation_stats = model.get_innovation_statistics()
        for key, value in innovation_stats.items():
            if isinstance(value, np.ndarray) and len(value) == 1:
                results[f'diagnostic_{key}'] = value[0]
    except:
        pass
    
    return results, spread_analyzer

def backtest_kalman_zscore_strategy(
    price1: pd.Series,
    price2: pd.Series,
    model_type: str = 'linear',
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    window_size: int = 20,
    use_volatility_adjustment: bool = False,
    include_costs: bool = True,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    max_holding_period: Optional[int] = None,
    use_trailing_stop: bool = False,
    trailing_stop_atr_multiple: float = 3.0,
    plot_results: bool = True,
    figsize: Tuple[int, int] = (15, 12)
) -> Dict[str, Any]:
    """
    Backtest a Z-score strategy using Kalman filter for dynamic hedge ratio estimation.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    model_type : str, default='linear'
        Type of Kalman filter to use: 'linear', 'extended', 'unscented'
    transition_covariance : float, default=1e-4
        Transition noise covariance parameter
    observation_covariance : float, default=1e-2
        Observation noise covariance parameter
    entry_threshold : float, default=2.0
        Z-score threshold for trade entry
    exit_threshold : float, default=0.5
        Z-score threshold for trade exit
    window_size : int, default=20
        Window size for Z-score calculation
    use_volatility_adjustment : bool, default=False
        Whether to use volatility adjustment for Z-score calculation
    include_costs : bool, default=True
        Whether to include transaction costs
    commission_rate : float, default=0.001
        Commission rate per trade (proportion of trade value)
    slippage_rate : float, default=0.0005
        Slippage rate per trade (proportion of trade value)
    max_holding_period : int, optional
        Maximum holding period for trades in days
    use_trailing_stop : bool, default=False
        Whether to use trailing stop for trade exits
    trailing_stop_atr_multiple : float, default=3.0
        Multiple of ATR for trailing stop calculation
    plot_results : bool, default=True
        Whether to plot backtest results
    figsize : tuple, default=(15, 12)
        Figure size for plots
        
    Returns
    -------
    Dict
        Dictionary with backtest results
    """
    # Align price series
    price1, price2 = price1.align(price2, join='inner')
    
    # Get dynamic spread analysis
    spread_results, _ = create_kalman_spread_analyzer(
        price1, price2,
        model_type=model_type,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        window_size=window_size,
        use_volatility_adjustment=use_volatility_adjustment
    )
    
    # Determine which Z-score to use
    if use_volatility_adjustment and 'volatility_adjusted_zscore' in spread_results.columns:
        z_score = spread_results['volatility_adjusted_zscore']
    else:
        z_score = spread_results['zscore']
    
    # Filter out NaN values at the beginning
    valid_indices = ~z_score.isna()
    price1 = price1[valid_indices]
    price2 = price2[valid_indices]
    spread_results = spread_results[valid_indices]
    z_score = z_score[valid_indices]
    
    # Extract hedge ratio
    hedge_ratio = spread_results['hedge_ratio']
    
    # Generate signals
    signals = pd.Series(0, index=z_score.index)
    
    # Long signals (when z-score is below negative threshold)
    signals[z_score < -entry_threshold] = 1
    
    # Short signals (when z-score is above positive threshold)
    signals[z_score > entry_threshold] = -1
    
    # Exit long positions when z-score crosses back above negative exit threshold
    exit_long = (z_score > -exit_threshold) & (z_score.shift(1) <= -exit_threshold)
    signals[exit_long] = 0
    
    # Exit short positions when z-score crosses back below positive exit threshold
    exit_short = (z_score < exit_threshold) & (z_score.shift(1) >= exit_threshold)
    signals[exit_short] = 0
    
    # Initialize positions and returns
    positions = pd.DataFrame(0, index=signals.index, columns=['asset1', 'asset2'])
    trade_returns = pd.Series(0.0, index=signals.index)
    cash_position = pd.Series(1.0, index=signals.index)  # Start with $1
    total_value = pd.Series(1.0, index=signals.index)    # Total portfolio value
    
    # For tracking transaction costs
    costs = pd.Series(0.0, index=signals.index)
    
    # For trailing stops
    if use_trailing_stop:
        # Calculate ATR for trailing stop
        high = pd.DataFrame({
            'price1_high': price1,
            'price2_high': price2
        }).max(axis=1)
        
        low = pd.DataFrame({
            'price1_low': price1,
            'price2_low': price2
        }).min(axis=1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - high.shift(1))
        tr3 = abs(low - low.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average True Range (ATR)
        atr = tr.rolling(window=window_size).mean()
        
        # Trailing stop levels
        long_stop_level = pd.Series(float('-inf'), index=signals.index)
        short_stop_level = pd.Series(float('inf'), index=signals.index)
        
        # For tracking stop activations
        stop_triggered = pd.Series(False, index=signals.index)
    
    # For maximum holding period
    if max_holding_period is not None:
        entry_date = pd.Series(pd.NaT, index=signals.index)
    
    # Backtest loop
    for i in range(1, len(signals)):
        # Get current signals and prices
        current_signal = signals.iloc[i]
        prev_signal = signals.iloc[i-1]
        
        # Update positions based on signals
        if current_signal != prev_signal:
            # Close existing position if any
            if prev_signal != 0:
                # Calculate costs for exit
                if include_costs:
                    exit_value = abs(positions.iloc[i-1, 0] * price1.iloc[i]) + abs(positions.iloc[i-1, 1] * price2.iloc[i])
                    exit_cost = exit_value * (commission_rate + slippage_rate)
                    costs.iloc[i] += exit_cost
                
                # Reset positions and trailing stops
                positions.iloc[i] = [0, 0]
                if use_trailing_stop:
                    long_stop_level.iloc[i] = float('-inf')
                    short_stop_level.iloc[i] = float('inf')
                
                # Reset entry date
                if max_holding_period is not None:
                    entry_date.iloc[i] = pd.NaT
            
            # Enter new position if signal is non-zero
            if current_signal != 0:
                # Calculate dynamic position sizes based on hedge ratio
                current_hedge_ratio = hedge_ratio.iloc[i]
                
                if current_signal == 1:  # Long spread
                    # Long asset2, short asset1 (scaled by hedge_ratio)
                    notional_value = 1.0  # Simplified: $1 exposure
                    asset2_qty = notional_value / price2.iloc[i]
                    asset1_qty = -asset2_qty * current_hedge_ratio
                    
                    positions.iloc[i] = [asset1_qty, asset2_qty]
                    
                    # Set trailing stop if enabled
                    if use_trailing_stop:
                        stop_size = atr.iloc[i] * trailing_stop_atr_multiple
                        long_stop_level.iloc[i] = spread_results['spread'].iloc[i] - stop_size
                    
                    # Record entry date
                    if max_holding_period is not None:
                        entry_date.iloc[i] = signals.index[i]
                        
                elif current_signal == -1:  # Short spread
                    # Short asset2, long asset1 (scaled by hedge_ratio)
                    notional_value = 1.0  # Simplified: $1 exposure
                    asset2_qty = -notional_value / price2.iloc[i]
                    asset1_qty = -asset2_qty * current_hedge_ratio
                    
                    positions.iloc[i] = [asset1_qty, asset2_qty]
                    
                    # Set trailing stop if enabled
                    if use_trailing_stop:
                        stop_size = atr.iloc[i] * trailing_stop_atr_multiple
                        short_stop_level.iloc[i] = spread_results['spread'].iloc[i] + stop_size
                        
                    # Record entry date
                    if max_holding_period is not None:
                        entry_date.iloc[i] = signals.index[i]
                
                # Calculate entry costs
                if include_costs:
                    entry_value = abs(positions.iloc[i, 0] * price1.iloc[i]) + abs(positions.iloc[i, 1] * price2.iloc[i])
                    entry_cost = entry_value * (commission_rate + slippage_rate)
                    costs.iloc[i] += entry_cost
        else:
            # Maintain positions if no signal change
            positions.iloc[i] = positions.iloc[i-1]
            
            # Update trailing stops if enabled
            if use_trailing_stop:
                if prev_signal == 1:  # Long position
                    # Update stop level to lock in profits
                    long_stop_level.iloc[i] = max(long_stop_level.iloc[i-1], spread_results['spread'].iloc[i] - atr.iloc[i] * trailing_stop_atr_multiple)
                    
                    # Check if stop is triggered
                    if spread_results['spread'].iloc[i] < long_stop_level.iloc[i]:
                        stop_triggered.iloc[i] = True
                        signals.iloc[i] = 0  # Exit position
                        
                        # Calculate costs for exit
                        if include_costs:
                            exit_value = abs(positions.iloc[i, 0] * price1.iloc[i]) + abs(positions.iloc[i, 1] * price2.iloc[i])
                            exit_cost = exit_value * (commission_rate + slippage_rate)
                            costs.iloc[i] += exit_cost
                        
                        # Reset positions
                        positions.iloc[i] = [0, 0]
                        
                elif prev_signal == -1:  # Short position
                    # Update stop level to lock in profits
                    short_stop_level.iloc[i] = min(short_stop_level.iloc[i-1], spread_results['spread'].iloc[i] + atr.iloc[i] * trailing_stop_atr_multiple)
                    
                    # Check if stop is triggered
                    if spread_results['spread'].iloc[i] > short_stop_level.iloc[i]:
                        stop_triggered.iloc[i] = True
                        signals.iloc[i] = 0  # Exit position
                        
                        # Calculate costs for exit
                        if include_costs:
                            exit_value = abs(positions.iloc[i, 0] * price1.iloc[i]) + abs(positions.iloc[i, 1] * price2.iloc[i])
                            exit_cost = exit_value * (commission_rate + slippage_rate)
                            costs.iloc[i] += exit_cost
                        
                        # Reset positions
                        positions.iloc[i] = [0, 0]
            
            # Check for maximum holding period
            if max_holding_period is not None and not pd.isna(entry_date.iloc[i-1]):
                # Copy entry date forward
                entry_date.iloc[i] = entry_date.iloc[i-1]
                
                # Check if max holding period is reached
                current_date = signals.index[i]
                holding_days = (current_date - entry_date.iloc[i]).days
                
                if holding_days >= max_holding_period:
                    signals.iloc[i] = 0  # Exit position
                    
                    # Calculate costs for exit
                    if include_costs:
                        exit_value = abs(positions.iloc[i, 0] * price1.iloc[i]) + abs(positions.iloc[i, 1] * price2.iloc[i])
                        exit_cost = exit_value * (commission_rate + slippage_rate)
                        costs.iloc[i] += exit_cost
                    
                    # Reset positions
                    positions.iloc[i] = [0, 0]
                    entry_date.iloc[i] = pd.NaT
        
        # Calculate returns
        if i > 0:
            # Returns from holding each asset
            asset1_return = (price1.iloc[i] / price1.iloc[i-1] - 1.0) * positions.iloc[i-1, 0]
            asset2_return = (price2.iloc[i] / price2.iloc[i-1] - 1.0) * positions.iloc[i-1, 1]
            
            # Total return from the spread position
            trade_return = asset1_return + asset2_return
            
            # Deduct costs
            if include_costs:
                trade_return -= costs.iloc[i]
            
            # Store return
            trade_returns.iloc[i] = trade_return
            
            # Update portfolio value
            total_value.iloc[i] = total_value.iloc[i-1] * (1.0 + trade_return)
    
    # Calculate performance metrics
    daily_returns = trade_returns
    
    # Annualized metrics
    n_years = (signals.index[-1] - signals.index[0]).days / 365.25
    
    # Performance metrics
    cum_returns = total_value / total_value.iloc[0] - 1.0
    annualized_return = (total_value.iloc[-1] / total_value.iloc[0]) ** (1.0 / n_years) - 1.0
    annualized_volatility = daily_returns.std() * np.sqrt(252) 
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    max_drawdown = (total_value / total_value.cummax() - 1.0).min()
    
    # Trade metrics
    trades = np.diff(signals.astype(float).fillna(0).values.astype(int) != 0)
    n_trades = np.sum(trades > 0)
    
    # Win/loss metrics
    trade_periods = []
    current_trade = None
    
    for i in range(len(signals)):
        if i > 0 and signals.iloc[i] != signals.iloc[i-1]:
            if current_trade is not None:
                # End of previous trade
                end_date = signals.index[i-1]
                trade_return = total_value.loc[end_date] / total_value.loc[current_trade[0]] - 1.0
                current_trade.append(end_date)
                current_trade.append(trade_return)
                trade_periods.append(current_trade)
                
            if signals.iloc[i] != 0:
                # Start of new trade
                current_trade = [signals.index[i], signals.iloc[i]]
            else:
                current_trade = None
    
    # Add last trade if still open
    if current_trade is not None:
        end_date = signals.index[-1]
        trade_return = total_value.iloc[-1] / total_value.loc[current_trade[0]] - 1.0
        current_trade.append(end_date)
        current_trade.append(trade_return)
        trade_periods.append(current_trade)
    
    # Calculate win rate and average return
    if trade_periods:
        trade_returns_list = [trade[3] for trade in trade_periods]
        winning_trades = sum(1 for r in trade_returns_list if r > 0)
        win_rate = winning_trades / len(trade_returns_list)
        avg_trade_return = np.mean(trade_returns_list)
        profit_factor = abs(sum(r for r in trade_returns_list if r > 0) / sum(r for r in trade_returns_list if r < 0)) if sum(r for r in trade_returns_list if r < 0) != 0 else np.inf
    else:
        win_rate = 0
        avg_trade_return = 0
        profit_factor = 0
    
    # Store results
    results = {
        'spread_results': spread_results,
        'signals': signals,
        'positions': positions,
        'trade_returns': trade_returns,
        'total_value': total_value,
        'costs': costs if include_costs else None,
        'trailing_stops': {'long_level': long_stop_level, 'short_level': short_stop_level, 'triggered': stop_triggered} if use_trailing_stop else None,
        'entry_dates': entry_date if max_holding_period is not None else None,
        'trade_periods': trade_periods,
        'metrics': {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_trade_return': avg_trade_return,
            'profit_factor': profit_factor,
            'total_return': cum_returns.iloc[-1]
        }
    }
    
    # Plot results if requested
    if plot_results:
        n_plots = 5
        if use_trailing_stop:
            n_plots += 1
        
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Plot prices
        ax = axes[0]
        ax.plot(price1.index, price1, label='Price 1')
        ax.plot(price2.index, price2, label='Price 2')
        ax.set_title(f'Price Series - {model_type.capitalize()} Kalman Filter')
        ax.legend()
        ax.grid(True)
        
        # Plot hedge ratio
        ax = axes[1]
        ax.plot(hedge_ratio.index, hedge_ratio)
        ax.set_title('Time-Varying Hedge Ratio')
        ax.grid(True)
        
        # Plot spread
        ax = axes[2]
        ax.plot(spread_results.index, spread_results['spread'])
        ax.set_title('Spread')
        ax.grid(True)
        
        # Plot Z-score with entry/exit thresholds
        ax = axes[3]
        ax.plot(z_score.index, z_score)
        ax.axhline(y=entry_threshold, color='r', linestyle='--')
        ax.axhline(y=-entry_threshold, color='r', linestyle='--')
        ax.axhline(y=exit_threshold, color='g', linestyle='--')
        ax.axhline(y=-exit_threshold, color='g', linestyle='--')
        ax.set_title('Z-Score with Trade Thresholds')
        ax.grid(True)
        
        # Plot trailing stops if enabled
        plot_idx = 4
        if use_trailing_stop:
            ax = axes[plot_idx]
            plot_idx += 1
        models=['linear', 'threshold', 'regime_switch', 'log_price']
    )
    print(model_comparison)
    
    print("\n4. Creating visualizations...")
    # Create plots
    fig1 = plot_timevarying_hedge_ratio(linear_results, price1, price2)
    fig2 = plot_nonlinear_hedge_ratio(threshold_results, price1, price2, model_type='threshold')
    fig3 = plot_model_comparison(price1, price2)
    
    print("\nExample complete. Visualizations created.")
    plt.show() 