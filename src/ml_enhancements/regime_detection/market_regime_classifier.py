"""
Market Regime Classifier Module

This module provides functionality to detect market regimes using various
statistical and machine learning methods to adapt trading parameters dynamically.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """
    Detect market regimes to adapt trading parameters.
    """
    
    def __init__(self, 
                 n_regimes=3, 
                 lookback_window=60, 
                 features=None,
                 method='kmeans'):
        """
        Initialize the MarketRegimeClassifier.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        lookback_window : int
            Number of days to use for feature calculation
        features : list, optional
            List of features to use for regime detection
        method : str
            Method for regime detection ('kmeans', 'hmm', 'thresh')
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.method = method
        
        # Default features if none provided
        self.features = features or [
            'volatility',
            'trend_strength',
            'correlation',
            'mean_reversion'
        ]
        
        # Initialized in fit
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.regime_stats = None
    
    def calculate_features(self, prices, volumes=None):
        """
        Calculate features for regime detection.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data for instruments
        volumes : pd.DataFrame, optional
            DataFrame with volume data for instruments
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated features
        """
        features_df = pd.DataFrame(index=prices.index[self.lookback_window:])
        
        # 1. Calculate volatility features
        if 'volatility' in self.features:
            # Rolling volatility for each instrument
            for col in prices.columns:
                returns = prices[col].pct_change()
                features_df[f'vol_{col}'] = returns.rolling(self.lookback_window).std()
            
            # Average volatility across instruments
            vol_cols = [col for col in features_df.columns if col.startswith('vol_')]
            features_df['volatility_avg'] = features_df[vol_cols].mean(axis=1)
            
            # Volatility of volatility
            features_df['vol_of_vol'] = features_df['volatility_avg'].rolling(self.lookback_window).std()
        
        # 2. Calculate trend features
        if 'trend_strength' in self.features:
            for col in prices.columns:
                # Linear regression slope
                returns = prices[col].pct_change()
                
                # Calculate rolling slopes
                slopes = []
                for i in range(self.lookback_window, len(returns)):
                    x = np.arange(self.lookback_window)
                    window_returns = returns.iloc[i-self.lookback_window:i].values
                    
                    if not np.isnan(window_returns).any():
                        slope, _ = np.polyfit(x, window_returns, 1)
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
                
                features_df[f'trend_{col}'] = slopes
            
            # Average trend strength
            trend_cols = [col for col in features_df.columns if col.startswith('trend_')]
            features_df['trend_strength_avg'] = features_df[trend_cols].mean(axis=1)
        
        # 3. Calculate correlation features
        if 'correlation' in self.features:
            # Calculate pairwise correlations between instruments
            corr_values = []
            
            for i in range(self.lookback_window, len(prices)):
                window = prices.iloc[i-self.lookback_window:i]
                corr_matrix = window.pct_change().corr()
                
                # Average of off-diagonal elements
                corr_avg = 0
                count = 0
                
                for j in range(len(corr_matrix)):
                    for k in range(j+1, len(corr_matrix)):
                        corr_avg += corr_matrix.iloc[j, k]
                        count += 1
                
                if count > 0:
                    corr_avg /= count
                
                corr_values.append(corr_avg)
            
            features_df['correlation_avg'] = corr_values
        
        # 4. Calculate mean reversion features
        if 'mean_reversion' in self.features:
            for col in prices.columns:
                # Calculate Hurst exponent
                hurst_values = []
                
                for i in range(self.lookback_window, len(prices)):
                    window = prices[col].iloc[i-self.lookback_window:i]
                    
                    if not window.isna().any():
                        hurst = self._calculate_hurst_exponent(window)
                        hurst_values.append(hurst)
                    else:
                        hurst_values.append(np.nan)
                
                features_df[f'hurst_{col}'] = hurst_values
            
            # Average Hurst exponent
            hurst_cols = [col for col in features_df.columns if col.startswith('hurst_')]
            features_df['mean_reversion_avg'] = features_df[hurst_cols].mean(axis=1)
        
        # 5. Include volume features if available
        if volumes is not None and 'volume' in self.features:
            for col in volumes.columns:
                # Volume rate of change
                vol_roc = volumes[col].pct_change()
                features_df[f'vol_roc_{col}'] = vol_roc.rolling(self.lookback_window).mean()
            
            # Average volume ROC
            vol_roc_cols = [col for col in features_df.columns if col.startswith('vol_roc_')]
            features_df['volume_roc_avg'] = features_df[vol_roc_cols].mean(axis=1)
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _calculate_hurst_exponent(self, ts, max_lag=20):
        """
        Calculate Hurst exponent for a time series.
        
        Parameters:
        -----------
        ts : pd.Series
            Time series data
        max_lag : int
            Maximum lag for R/S calculation
            
        Returns:
        --------
        float
            Hurst exponent
        """
        # Convert to numpy array
        ts = ts.values
        
        # Calculate R/S values for different lags
        lags = range(2, min(max_lag, len(ts)//4))
        
        # Return 0.5 if series is too short
        if len(lags) < 2:
            return 0.5
        
        # Calculate tau
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        
        # Linear regression
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        hurst = m[0]
        
        return hurst
    
    def fit(self, features_df):
        """
        Fit the regime detection model.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame with features for regime detection
            
        Returns:
        --------
        self
        """
        # Select the main feature columns
        main_features = []
        
        for feature in self.features:
            # Find columns related to this feature
            feature_cols = [col for col in features_df.columns if feature in col and '_avg' in col]
            
            if feature_cols:
                main_features.extend(feature_cols)
        
        # If no main features found, use all columns
        if not main_features:
            main_features = features_df.columns
        
        # Standardize features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features_df[main_features])
        
        # Fit model based on selected method
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self.model.fit(scaled_features)
            
            # Calculate cluster centers and statistics
            self.regime_stats = {}
            
            for i in range(self.n_regimes):
                # Get samples in this cluster
                cluster_samples = features_df[self.model.labels_ == i]
                
                # Calculate statistics
                self.regime_stats[i] = {
                    'count': len(cluster_samples),
                    'percentage': len(cluster_samples) / len(features_df) * 100,
                    'mean': cluster_samples.mean(),
                    'std': cluster_samples.std()
                }
        
        # Calculate feature importance (based on variance)
        feature_vars = features_df[main_features].var()
        total_var = feature_vars.sum()
        self.feature_importance = feature_vars / total_var
        
        return self
    
    def predict(self, features_df):
        """
        Predict market regimes for the given features.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame with features for regime detection
            
        Returns:
        --------
        pd.Series
            Series with regime labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Select the main feature columns
        main_features = []
        
        for feature in self.features:
            # Find columns related to this feature
            feature_cols = [col for col in features_df.columns if feature in col and '_avg' in col]
            
            if feature_cols:
                main_features.extend(feature_cols)
        
        # If no main features found, use all columns
        if not main_features:
            main_features = features_df.columns
        
        # Standardize features
        scaled_features = self.scaler.transform(features_df[main_features])
        
        # Predict regimes
        if self.method == 'kmeans':
            regimes = pd.Series(self.model.predict(scaled_features), index=features_df.index)
        
        return regimes
    
    def get_regime_parameters(self, regime):
        """
        Get recommended trading parameters for a given regime.
        
        Parameters:
        -----------
        regime : int
            Regime label
            
        Returns:
        --------
        dict
            Dictionary with recommended trading parameters
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if regime not in self.regime_stats:
            raise ValueError(f"Invalid regime: {regime}")
        
        # Get regime statistics
        stats = self.regime_stats[regime]
        
        # Define parameter mapping functions
        
        # 1. Entry Z-Score: Higher during high volatility
        # Scale between 1.5 and 3.0
        def map_entry_zscore(vol_stat):
            vol_percentile = (vol_stat - 0) / (1 - 0)  # Assuming vol is between 0 and 1 after scaling
            return 1.5 + vol_percentile * 1.5
        
        # 2. Exit Z-Score: Lower during high mean reversion
        # Scale between 0.0 and 1.0
        def map_exit_zscore(mr_stat):
            # Lower Hurst = higher mean reversion
            mr_percentile = 1 - (mr_stat - 0) / (1 - 0)  # Assuming Hurst is between 0 and 1
            return mr_percentile * 1.0
        
        # 3. Stop Loss: Wider during high volatility
        # Scale between 3.0 and 6.0 standard deviations
        def map_stop_loss(vol_stat):
            vol_percentile = (vol_stat - 0) / (1 - 0)
            return 3.0 + vol_percentile * 3.0
        
        # 4. Position Size: Smaller during high volatility
        # Scale between 0.5 and 1.0 of max position
        def map_position_size(vol_stat):
            vol_percentile = (vol_stat - 0) / (1 - 0)
            return 1.0 - vol_percentile * 0.5
        
        # Get relevant statistics
        vol_stat = stats['mean'].get('volatility_avg', 0.5)
        mr_stat = stats['mean'].get('mean_reversion_avg', 0.5)
        
        # Map to parameters
        params = {
            'entry_zscore': map_entry_zscore(vol_stat),
            'exit_zscore': map_exit_zscore(mr_stat),
            'stop_loss_std': map_stop_loss(vol_stat),
            'position_size_factor': map_position_size(vol_stat),
            'regime_description': self.get_regime_description(regime)
        }
        
        return params
    
    def get_regime_description(self, regime):
        """
        Get a descriptive name for a given regime.
        
        Parameters:
        -----------
        regime : int
            Regime label
            
        Returns:
        --------
        str
            Descriptive name for the regime
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if regime not in self.regime_stats:
            raise ValueError(f"Invalid regime: {regime}")
        
        # Get regime statistics
        stats = self.regime_stats[regime]
        
        # Extract key statistics
        vol = stats['mean'].get('volatility_avg', 0.5)
        trend = stats['mean'].get('trend_strength_avg', 0.5)
        corr = stats['mean'].get('correlation_avg', 0.5)
        mr = stats['mean'].get('mean_reversion_avg', 0.5)
        
        # Determine regime characteristics
        descriptors = []
        
        # Volatility
        if vol > 0.7:
            descriptors.append("High Volatility")
        elif vol < 0.3:
            descriptors.append("Low Volatility")
        
        # Trend
        if trend > 0.7:
            descriptors.append("Strong Trend")
        elif trend < 0.3:
            descriptors.append("Weak Trend")
        
        # Correlation
        if corr > 0.7:
            descriptors.append("High Correlation")
        elif corr < 0.3:
            descriptors.append("Low Correlation")
        
        # Mean Reversion
        if mr < 0.4:  # Hurst < 0.4 indicates mean reversion
            descriptors.append("Strong Mean Reversion")
        elif mr > 0.6:  # Hurst > 0.6 indicates trend following
            descriptors.append("Weak Mean Reversion")
        
        # If no clear characteristics, use generic description
        if not descriptors:
            return f"Regime {regime + 1}"
        else:
            return " / ".join(descriptors)
    
    def plot_regimes(self, prices, regimes, main_symbols=None, save_path=None):
        """
        Plot price data with regime overlays.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            DataFrame with price data
        regimes : pd.Series
            Series with regime labels
        main_symbols : list, optional
            List of main symbols to highlight
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        None
        """
        if main_symbols is None:
            main_symbols = prices.columns[:2]  # Default to first two symbols
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot prices
        for symbol in main_symbols:
            if symbol in prices.columns:
                # Normalize to start at 100
                normalized = prices[symbol] / prices[symbol].iloc[0] * 100
                ax1.plot(normalized, label=symbol)
        
        ax1.set_title('Price with Market Regimes')
        ax1.legend()
        ax1.grid(True)
        
        # Plot regimes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        prev_regime = None
        start_idx = None
        
        for idx, regime in regimes.items():
            if prev_regime != regime:
                if prev_regime is not None and start_idx is not None:
                    ax1.axvspan(start_idx, idx, alpha=0.3, color=colors[prev_regime % len(colors)])
                start_idx = idx
                prev_regime = regime
        
        # Final regime
        if prev_regime is not None and start_idx is not None:
            ax1.axvspan(start_idx, regimes.index[-1], alpha=0.3, color=colors[prev_regime % len(colors)])
        
        # Plot regime values
        ax2.plot(regimes, drawstyle='steps-post')
        ax2.set_yticks(range(self.n_regimes))
        ax2.set_yticklabels([self.get_regime_description(i) for i in range(self.n_regimes)])
        ax2.grid(True)
        ax2.set_title('Market Regimes')
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_feature_importance(self, save_path=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        None
        """
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance
        sorted_features = self.feature_importance.sort_values(ascending=False)
        
        # Plot feature importance
        plt.bar(sorted_features.index, sorted_features.values)
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 
    def detect_regime(self, data):
        """
        Detect the current market regime based on the provided data.
        
        This is a simple placeholder implementation. In practice, this would use
        more sophisticated methods like clustering, HMM, or other statistical techniques.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Historical price or returns data
            
        Returns:
        --------
        str
            Identified market regime ('normal', 'volatile', 'trending', 'mean_reverting')
        """
        # Basic implementation - in practice, would be more sophisticated
        if data is None or len(data) < 20:
            return 'unknown'
            
        # Convert to returns if prices were provided
        if isinstance(data, pd.Series):
            returns = data.pct_change().dropna()
        else:
            # Assuming the last column is the price
            returns = data.iloc[:, -1].pct_change().dropna()
            
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        if volatility > 0.25:
            return 'volatile'
        elif volatility < 0.10:
            return 'low_volatility'
        
        # Check for trend using simple moving average crossover
        if len(returns) >= 50:
            ma_short = returns.rolling(20).mean()
            ma_long = returns.rolling(50).mean()
            
            # Check last 5 days of crossover
            crossover_points = (ma_short.iloc[-5:] > ma_long.iloc[-5:]).sum()
            
            if crossover_points >= 4:
                return 'trending_up'
            elif crossover_points <= 1:
                return 'trending_down'
        
        # Check mean reversion using autocorrelation
        if len(returns) >= 20:
            autocorr = returns.autocorr(lag=1)
            if autocorr < -0.2:
                return 'mean_reverting'
                
        return 'normal'
        
    def describe_regime(self, regime):
        """
        Provide a description of the given market regime.
        
        Parameters:
        -----------
        regime : str
            Market regime identifier
            
        Returns:
        --------
        str
            Description of the market regime
        """
        descriptions = {
            'volatile': 'High volatility environment with large price swings',
            'low_volatility': 'Periods of low volatility with small price movements',
            'trending_up': 'Upward trending market with persistent price increases',
            'trending_down': 'Downward trending market with persistent price decreases',
            'mean_reverting': 'Mean-reverting market with frequent reversals',
            'normal': 'Normal market conditions with mixed characteristics',
            'unknown': 'Insufficient data to determine market regime'
        }
        
        return descriptions.get(regime, 'Unknown market regime')
