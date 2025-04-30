import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

class SpreadCalculator:
    """
    Class for calculating and normalizing spreads between two assets.
    Includes methods for static and dynamic (Kalman filter) hedge ratio calculations.
    """
    
    def __init__(self, lookback_window=20, z_score_window=20):
        """
        Initialize the spread calculator.
        
        Parameters:
        -----------
        lookback_window : int
            Window size for rolling statistics calculations
        z_score_window : int
            Window size for z-score calculation
        """
        self.lookback_window = lookback_window
        self.z_score_window = z_score_window
    
    def calculate_static_spread(self, series_y, series_x):
        """
        Calculate spread using static hedge ratio from OLS regression.
        
        Parameters:
        -----------
        series_y : pandas.Series
            Price series for the dependent variable (y)
        series_x : pandas.Series
            Price series for the independent variable (x)
            
        Returns:
        --------
        tuple
            (spread_series, hedge_ratio, OLS_model)
        """
        # Add constant to independent variable
        X = sm.add_constant(series_x)
        
        # Fit OLS model to find hedge ratio
        model = sm.OLS(series_y, X).fit()
        
        # Extract hedge ratio and intercept
        beta = model.params[1]
        alpha = model.params[0]
        
        # Calculate the spread
        spread = series_y - (alpha + beta * series_x)
        
        return spread, beta, model
    
    def calculate_kalman_spread(self, series_y, series_x, transition_covariance=0.01, 
                               observation_covariance=1.0, initial_state_covariance=1.0):
        """
        Calculate spread using Kalman filter for dynamic hedge ratio estimation.
        
        Parameters:
        -----------
        series_y : pandas.Series
            Price series for the dependent variable (y)
        series_x : pandas.Series
            Price series for the independent variable (x)
        transition_covariance : float
            Covariance for state transition matrix (controls hedge ratio adaptation speed)
        observation_covariance : float
            Covariance for observation matrix
        initial_state_covariance : float
            Initial covariance for state
            
        Returns:
        --------
        tuple
            (spread_series, hedge_ratio_series, state_means, state_covariances)
        """
        # Prepare the data
        y = series_y.values
        x = series_x.values
        
        # Reshape for Kalman filter
        delta = 1e-5  # Small constant to avoid numerical issues
        observations = y
        observation_matrices = np.vstack([np.ones(len(x)), x]).T[:, np.newaxis]
        
        # Initialize Kalman filter
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=observation_matrices,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.eye(2) * initial_state_covariance,
            transition_covariance=np.eye(2) * transition_covariance,
            observation_covariance=observation_covariance
        )
        
        # Run the Kalman filter
        state_means, state_covariances = kf.filter(observations)
        
        # Extract the dynamic hedge ratio and intercept
        alphas = state_means[:, 0]  # Intercepts
        betas = state_means[:, 1]   # Hedge ratios
        
        # Calculate the spread
        spread = y - (alphas + betas * x)
        
        # Create Series for the results
        spread_series = pd.Series(spread, index=series_y.index)
        hedge_ratio_series = pd.Series(betas, index=series_y.index)
        
        return spread_series, hedge_ratio_series, state_means, state_covariances
    
    def calculate_zscore(self, spread, window=None):
        """
        Calculate the z-score of a spread using a rolling window.
        
        Parameters:
        -----------
        spread : pandas.Series
            The spread series to normalize
        window : int, optional
            Window size for rolling statistics, if None uses self.z_score_window
            
        Returns:
        --------
        pandas.Series
            Z-score series
        """
        if window is None:
            window = self.z_score_window
        
        # Calculate rolling mean and standard deviation
        roll_mean = spread.rolling(window=window).mean()
        roll_std = spread.rolling(window=window).std()
        
        # Avoid division by zero
        roll_std = roll_std.replace(0, np.nan)
        
        # Calculate z-score
        zscore = (spread - roll_mean) / roll_std
        
        return zscore
    
    def calculate_ewm_zscore(self, spread, span=None, adjust=True):
        """
        Calculate the z-score using exponentially weighted moving statistics.
        
        Parameters:
        -----------
        spread : pandas.Series
            The spread series to normalize
        span : int, optional
            Span for EWM, if None uses self.z_score_window
        adjust : bool
            Whether to adjust the statistics (for unbiased estimation)
            
        Returns:
        --------
        pandas.Series
            Z-score series
        """
        if span is None:
            span = self.z_score_window
        
        # Calculate EWM mean and standard deviation
        ewm_mean = spread.ewm(span=span, adjust=adjust).mean()
        ewm_std = spread.ewm(span=span, adjust=adjust).std()
        
        # Avoid division by zero
        ewm_std = ewm_std.replace(0, np.nan)
        
        # Calculate z-score
        zscore = (spread - ewm_mean) / ewm_std
        
        return zscore
    
    def calculate_half_life(self, spread):
        """
        Calculate half-life of mean reversion for a price spread series.
        
        Parameters:
        -----------
        spread : pandas.Series
            The spread series to analyze
            
        Returns:
        --------
        float
            Half-life of mean reversion in the same frequency as input data
        """
        # Calculate returns (price differences)
        lag_spread = spread.shift(1)
        delta_spread = spread - lag_spread
        
        # Remove NaN values
        lag_spread = lag_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        # Regression to estimate mean reversion
        X = sm.add_constant(lag_spread)
        model = sm.OLS(delta_spread, X).fit()
        
        # Extract coefficient
        beta = model.params[1]
        
        if beta >= 0:
            # Not mean-reverting
            return np.inf
        
        # Calculate half-life
        half_life = -np.log(2) / beta
        
        return half_life
    
    def identify_outliers(self, zscore, threshold=3.0):
        """
        Identify outliers in the z-score series.
        
        Parameters:
        -----------
        zscore : pandas.Series
            The z-score series to analyze
        threshold : float
            Threshold for identifying outliers
            
        Returns:
        --------
        pandas.Series
            Boolean series with True for outliers
        """
        return abs(zscore) > threshold
    
    def plot_spread_analysis(self, spread, zscore, hedge_ratio=None, title="Spread Analysis"):
        """
        Plot spread, z-score, and hedge ratio for analysis.
        
        Parameters:
        -----------
        spread : pandas.Series
            The spread series
        zscore : pandas.Series
            The z-score series
        hedge_ratio : pandas.Series, optional
            Dynamic hedge ratio series (if available)
        title : str
            Title for the plot
        """
        fig, axes = plt.subplots(2 if hedge_ratio is None else 3, 1, figsize=(12, 8 if hedge_ratio is None else 10), sharex=True)
        
        # Plot spread
        axes[0].plot(spread.index, spread.values)
        axes[0].set_title(f"{title} - Spread")
        axes[0].set_ylabel("Spread")
        axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Plot z-score
        axes[1].plot(zscore.index, zscore.values)
        axes[1].set_title("Z-Score")
        axes[1].set_ylabel("Z-Score")
        axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axes[1].axhline(y=2, color='g', linestyle='--', alpha=0.3)
        axes[1].axhline(y=-2, color='g', linestyle='--', alpha=0.3)
        
        # Plot hedge ratio if provided
        if hedge_ratio is not None:
            axes[2].plot(hedge_ratio.index, hedge_ratio.values)
            axes[2].set_title("Hedge Ratio")
            axes[2].set_ylabel("Beta")
        
        plt.tight_layout()
        return fig, axes
    
    def analyze_spread_volatility(self, spread, window=None):
        """
        Analyze the volatility regime of the spread.
        
        Parameters:
        -----------
        spread : pandas.Series
            The spread series to analyze
        window : int, optional
            Window size for volatility calculation
            
        Returns:
        --------
        tuple
            (volatility_series, regime_labels)
        """
        if window is None:
            window = self.lookback_window
        
        # Calculate rolling volatility
        volatility = spread.rolling(window=window).std()
        
        # Calculate long-term volatility over a longer window
        long_term_vol = spread.rolling(window=window*3).std()
        
        # Identify volatility regimes
        regimes = pd.Series(index=spread.index, dtype='object')
        regimes.iloc[:] = 'normal'
        
        # Mark high volatility periods
        high_vol_mask = volatility > (long_term_vol * 1.2)  # 20% higher than long-term vol
        regimes[high_vol_mask] = 'high'
        
        # Mark low volatility periods
        low_vol_mask = volatility < (long_term_vol * 0.8)  # 20% lower than long-term vol
        regimes[low_vol_mask] = 'low'
        
        return volatility, regimes 