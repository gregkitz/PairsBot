"""
Regime Detector for Pairs Trading.

This module provides tools to detect breakdowns in spread relationships
and regime shifts in the spread behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema
import warnings
from src.utils.technical_indicators import rsi, macd, z_score


class RegimeDetector:
    """
    Regime Detector for Pairs Trading.
    
    This class provides methods to detect breakdowns in cointegration relationships
    and regime shifts in spread behavior to avoid trading during unfavorable conditions.
    """
    
    def __init__(self,
                window_size: int = 60,
                rolling_window: int = 20,
                p_value_threshold: float = 0.05,
                correlation_threshold: float = 0.5,
                spread_change_threshold: float = 3.0,
                num_clusters: int = 3,
                random_state: int = 42):
        """
        Initialize the regime detector.
        
        Parameters:
        -----------
        window_size : int
            Size of window for statistical tests (in days)
        rolling_window : int
            Size of rolling window for metrics calculation
        p_value_threshold : float
            Threshold for statistical significance
        correlation_threshold : float
            Minimum correlation to consider stable relationship
        spread_change_threshold : float
            Threshold for spread volatility (in standard deviations)
        num_clusters : int
            Number of regimes/clusters to detect
        random_state : int
            Random seed for reproducibility
        """
        self.window_size = window_size
        self.rolling_window = rolling_window
        self.p_value_threshold = p_value_threshold
        self.correlation_threshold = correlation_threshold
        self.spread_change_threshold = spread_change_threshold
        self.num_clusters = num_clusters
        self.random_state = random_state
        
        # Results
        self.results = None
        self.spread_features = None
        self.regime_labels = None
        self.current_regime = None
        self.regime_stats = None
    
    def detect_cointegration_breakdown(self, 
                                     price1: pd.Series, 
                                     price2: pd.Series,
                                     hedge_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect breakdown in cointegration relationship between two price series.
        
        Parameters:
        -----------
        price1 : pandas.Series
            Price series of the first asset
        price2 : pandas.Series
            Price series of the second asset
        hedge_ratio : float, optional
            Pre-computed hedge ratio. If None, it will be estimated.
            
        Returns:
        --------
        dict
            Dictionary with detection results
        """
        # Ensure both series have the same index
        common_idx = price1.index.intersection(price2.index)
        price1 = price1.loc[common_idx]
        price2 = price2.loc[common_idx]
        
        # Create rolling windows
        results = []
        for i in range(len(price1) - self.window_size + 1):
            window_price1 = price1.iloc[i:i+self.window_size]
            window_price2 = price2.iloc[i:i+self.window_size]
            
            # Calculate correlation
            correlation = window_price1.corr(window_price2)
            
            # Estimate hedge ratio if not provided
            if hedge_ratio is None:
                X = sm.add_constant(window_price2)
                model = sm.OLS(window_price1, X).fit()
                window_hedge_ratio = model.params[1]
            else:
                window_hedge_ratio = hedge_ratio
            
            # Calculate spread
            spread = window_price1 - window_hedge_ratio * window_price2
            
            # Test stationarity
            adf_result = adfuller(spread, maxlag=1)
            adf_pvalue = adf_result[1]
            
            # Test cointegration
            coint_result = coint(window_price1, window_price2)
            coint_pvalue = coint_result[1]
            
            # Calculate spread volatility
            spread_std = spread.std()
            spread_mean = spread.mean()
            
            # Calculate half-life of mean reversion
            try:
                spread_lag = spread.shift(1).iloc[1:]
                spread_diff = spread.diff().iloc[1:]
                model = sm.OLS(spread_diff, spread_lag).fit()
                half_life = -np.log(2) / model.params[0] if model.params[0] < 0 else np.nan
            except:
                half_life = np.nan
            
            # Store results
            results.append({
                'start_date': window_price1.index[0],
                'end_date': window_price1.index[-1],
                'correlation': correlation,
                'hedge_ratio': window_hedge_ratio,
                'adf_pvalue': adf_pvalue,
                'coint_pvalue': coint_pvalue,
                'spread_std': spread_std,
                'spread_mean': spread_mean,
                'half_life': half_life,
                'is_cointegrated': coint_pvalue < self.p_value_threshold,
                'is_stationary': adf_pvalue < self.p_value_threshold,
                'has_strong_correlation': correlation > self.correlation_threshold
            })
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results)
        
        # Detect breakdown
        recent_results = self.results.iloc[-self.rolling_window:]
        
        breakdown_metrics = {
            'percentage_cointegrated': recent_results['is_cointegrated'].mean() * 100,
            'percentage_stationary': recent_results['is_stationary'].mean() * 100,
            'percentage_correlated': recent_results['has_strong_correlation'].mean() * 100,
            'avg_correlation': recent_results['correlation'].mean(),
            'avg_half_life': recent_results['half_life'].mean(),
            'avg_adf_pvalue': recent_results['adf_pvalue'].mean(),
            'avg_coint_pvalue': recent_results['coint_pvalue'].mean(),
            'hedge_ratio_volatility': recent_results['hedge_ratio'].std() / recent_results['hedge_ratio'].mean(),
            'is_relationship_stable': (
                recent_results['is_cointegrated'].mean() > 0.7 and
                recent_results['has_strong_correlation'].mean() > 0.7
            )
        }
        
        return breakdown_metrics
    
    def detect_spread_regime(self, spread: pd.Series) -> Dict[str, Any]:
        """
        Detect market regime based on spread behavior.
        
        Parameters
        ----------
        spread : pd.Series
            Spread time series
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing regime detection results
        """
        if len(spread) < self.num_clusters:
            return {'error': 'Not enough data for regime detection'}
        
        # Create DataFrame with spread
        df = pd.DataFrame({'spread': spread})
        
        # Calculate features for clustering
        df['volatility'] = df['spread'].rolling(window=20).std()
        df['zscore_20'] = z_score(df['spread'], window=20)
        df['mean_reversion'] = df['spread'].rolling(window=20).mean() - df['spread']
        df['autocorr_1'] = df['spread'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else np.nan
        )
        df['volatility_ratio'] = df['spread'].rolling(window=10).std() / df['spread'].rolling(window=30).std()
        
        # RSI
        df['rsi_14'] = rsi(df['spread'])
        
        # MACD
        macd_line, macd_signal, macd_histogram = macd(df['spread'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Extract local maxima and minima
        df['is_max'] = df.index.isin(
            df.iloc[argrelextrema(df['spread'].values, np.greater_equal, order=5)[0]].index
        )
        df['is_min'] = df.index.isin(
            df.iloc[argrelextrema(df['spread'].values, np.less_equal, order=5)[0]].index
        )
        
        # Store spread features
        self.spread_features = df.dropna()
        
        if len(self.spread_features) < self.num_clusters:
            return {'error': 'Not enough data points for clustering'}
        
        # Select features for clustering
        cluster_features = [
            'volatility_ratio', 'zscore_20', 'autocorr_1', 'rsi_14', 'macd_histogram',
            'trend_ratio'
        ]
        
        # Keep only available features
        cluster_features = [f for f in cluster_features if f in self.spread_features.columns]
        
        if not cluster_features:
            return {'error': 'No valid features for clustering'}
        
        # Prepare data for clustering
        X = self.spread_features[cluster_features].dropna()
        
        if len(X) < self.num_clusters:
            return {'error': 'Not enough data points for clustering after removing NaN values'}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        self.regime_labels = kmeans.fit_predict(X_scaled)
        
        # Assign regimes to the spread data
        self.spread_features['regime'] = np.nan
        self.spread_features.loc[X.index, 'regime'] = self.regime_labels
        
        # Get the current regime (last known regime)
        self.current_regime = int(self.spread_features['regime'].iloc[-1])
        
        # Calculate statistics for each regime
        regime_stats = []
        for regime in range(self.num_clusters):
            regime_data = self.spread_features[self.spread_features['regime'] == regime]
            
            # Calculate performance metrics for this regime
            mean_reversion_success = self._calculate_mean_reversion_success(regime_data)
            
            regime_stats.append({
                'regime': regime,
                'count': len(regime_data),
                'percentage': len(regime_data) / len(self.spread_features) * 100,
                'avg_volatility': regime_data['std_20'].mean(),
                'avg_zscore': regime_data['zscore_20'].mean(),
                'avg_autocorr': regime_data['autocorr_1'].mean() if 'autocorr_1' in regime_data else np.nan,
                'avg_rsi': regime_data['rsi_14'].mean(),
                'regime_type': self._determine_regime_type(regime_data),
                'mean_reversion_success': mean_reversion_success,
                'suitable_for_trading': mean_reversion_success > 0.6  # If mean reversion works more than 60% of the time
            })
        
        self.regime_stats = pd.DataFrame(regime_stats)
        
        # Get current regime info
        current_regime_info = self.regime_stats[self.regime_stats['regime'] == self.current_regime].iloc[0].to_dict()
        
        return {
            'current_regime': self.current_regime,
            'current_regime_type': current_regime_info['regime_type'],
            'current_regime_suitable': current_regime_info['suitable_for_trading'],
            'current_mean_reversion_success': current_regime_info['mean_reversion_success'],
            'all_regimes': self.regime_stats.to_dict(orient='records')
        }
    
    def _determine_regime_type(self, regime_data: pd.DataFrame) -> str:
        """
        Determine the type of regime based on its characteristics.
        
        Parameters:
        -----------
        regime_data : pandas.DataFrame
            Data for a specific regime
            
        Returns:
        --------
        str
            Regime type
        """
        # Calculate metrics
        avg_volatility = regime_data['std_20'].mean() if 'std_20' in regime_data else np.nan
        avg_autocorr = regime_data['autocorr_1'].mean() if 'autocorr_1' in regime_data else np.nan
        avg_rsi = regime_data['rsi_14'].mean() if 'rsi_14' in regime_data else np.nan
        
        # Determine regime type
        if np.isnan(avg_volatility) or np.isnan(avg_autocorr):
            return "Unknown"
        
        if avg_volatility > 1.5 and avg_autocorr < 0.3:
            return "High Volatility / Unstable"
        elif avg_volatility < 0.5 and avg_autocorr > 0.7:
            return "Low Volatility / Stable"
        elif avg_rsi > 70:
            return "Overbought"
        elif avg_rsi < 30:
            return "Oversold"
        elif 0.5 <= avg_autocorr <= 0.7:
            return "Moderate Mean-Reversion"
        elif 0.3 <= avg_autocorr <= 0.5:
            return "Weak Mean-Reversion"
        else:
            return "Mixed"
    
    def _calculate_mean_reversion_success(self, regime_data: pd.DataFrame) -> float:
        """
        Calculate the success rate of mean reversion strategy in this regime.
        
        Parameters:
        -----------
        regime_data : pandas.DataFrame
            Data for a specific regime
            
        Returns:
        --------
        float
            Success rate (0.0-1.0)
        """
        if 'zscore_20' not in regime_data.columns or len(regime_data) < 5:
            return np.nan
        
        # Look for high z-score values and check if they revert
        high_zscore = regime_data[regime_data['zscore_20'] > 2.0]
        low_zscore = regime_data[regime_data['zscore_20'] < -2.0]
        
        success_count = 0
        total_count = 0
        
        # Check high z-scores
        for idx in high_zscore.index:
            if idx + 5 in regime_data.index:
                future_zscore = regime_data.loc[idx + 5, 'zscore_20']
                if future_zscore < high_zscore.loc[idx, 'zscore_20']:
                    success_count += 1
                total_count += 1
        
        # Check low z-scores
        for idx in low_zscore.index:
            if idx + 5 in regime_data.index:
                future_zscore = regime_data.loc[idx + 5, 'zscore_20']
                if future_zscore > low_zscore.loc[idx, 'zscore_20']:
                    success_count += 1
                total_count += 1
        
        # Calculate success rate
        if total_count > 0:
            return success_count / total_count
        else:
            return np.nan
    
    def is_tradable(self, 
                  correlation: float = None,
                  adf_pvalue: float = None,
                  coint_pvalue: float = None,
                  current_regime: int = None) -> Dict[str, Any]:
        """
        Determine if the current state is tradable based on multiple criteria.
        
        Parameters:
        -----------
        correlation : float, optional
            Current correlation between the two assets
        adf_pvalue : float, optional
            p-value from ADF test on the spread
        coint_pvalue : float, optional
            p-value from cointegration test
        current_regime : int, optional
            Current regime ID (if None, use the detected regime)
            
        Returns:
        --------
        dict
            Dictionary with tradability assessment
        """
        # Initialize result
        result = {
            'is_tradable': True,
            'reasons': []
        }
        
        # Check correlation
        if correlation is not None and correlation < self.correlation_threshold:
            result['is_tradable'] = False
            result['reasons'].append(f"Low correlation: {correlation:.2f} < {self.correlation_threshold}")
        
        # Check cointegration
        if coint_pvalue is not None and coint_pvalue > self.p_value_threshold:
            result['is_tradable'] = False
            result['reasons'].append(f"Not cointegrated: p-value = {coint_pvalue:.4f} > {self.p_value_threshold}")
        
        # Check stationarity
        if adf_pvalue is not None and adf_pvalue > self.p_value_threshold:
            result['is_tradable'] = False
            result['reasons'].append(f"Non-stationary spread: p-value = {adf_pvalue:.4f} > {self.p_value_threshold}")
        
        # Check regime
        if self.regime_stats is not None:
            regime = current_regime if current_regime is not None else self.current_regime
            regime_info = self.regime_stats[self.regime_stats['regime'] == regime]
            
            if not regime_info.empty and not regime_info.iloc[0]['suitable_for_trading']:
                result['is_tradable'] = False
                regime_type = regime_info.iloc[0]['regime_type']
                result['reasons'].append(f"Unfavorable regime: {regime_type}")
        
        # Add relationship breakdown check
        if self.results is not None:
            recent_cointegration = self.results.iloc[-self.rolling_window:]['is_cointegrated'].mean()
            if recent_cointegration < 0.7:
                result['is_tradable'] = False
                result['reasons'].append(f"Unstable cointegration: only {recent_cointegration*100:.1f}% windows cointegrated")
        
        # Return assessment
        result['reason_count'] = len(result['reasons'])
        return result
    
    def plot_regimes(self, spread: pd.Series, save_path=None):
        """
        Plot the spread with regimes highlighted.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread time series
        save_path : str, optional
            Path to save the plot
        """
        if self.spread_features is None or 'regime' not in self.spread_features.columns:
            print("No regime detection results available. Run detect_spread_regime first.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot the spread
        ax1 = plt.subplot(211)
        for regime in range(self.num_clusters):
            regime_data = self.spread_features[self.spread_features['regime'] == regime]
            label = f"Regime {regime}: {self.regime_stats[self.regime_stats['regime'] == regime].iloc[0]['regime_type']}"
            is_tradable = self.regime_stats[self.regime_stats['regime'] == regime].iloc[0]['suitable_for_trading']
            color = 'green' if is_tradable else 'red'
            ax1.scatter(regime_data.index, regime_data['spread'], label=label, color=color, alpha=0.7)
        
        # Plot full spread line
        ax1.plot(spread.index, spread, color='black', alpha=0.3)
        
        ax1.set_title('Spread with Regime Classification')
        ax1.set_ylabel('Spread Value')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot regime statistics
        ax2 = plt.subplot(212)
        
        if self.results is not None:
            ax2.plot(self.results['end_date'], self.results['correlation'], label='Correlation', color='blue')
            ax2.plot(self.results['end_date'], 1 - self.results['coint_pvalue'], label='Cointegration Confidence', color='green')
            
            # Mark regions where cointegration breaks down
            breakdown_mask = self.results['coint_pvalue'] > self.p_value_threshold
            ax2.fill_between(self.results['end_date'], 0, 1, where=breakdown_mask, color='red', alpha=0.3, label='Cointegration Breakdown')
            
            ax2.set_ylim(0, 1.1)
            ax2.set_title('Relationship Stability Metrics')
            ax2.set_ylabel('Metric Value')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot regime characteristics
        plt.figure(figsize=(12, 6))
        
        # Create a grouped bar chart of regime characteristics
        if self.regime_stats is not None:
            metrics = ['avg_volatility', 'avg_zscore', 'avg_autocorr', 'mean_reversion_success']
            available_metrics = [m for m in metrics if m in self.regime_stats.columns]
            
            if available_metrics:
                bar_width = 0.2
                r = np.arange(len(self.regime_stats))
                
                for i, metric in enumerate(available_metrics):
                    plt.bar(r + i * bar_width, self.regime_stats[metric], width=bar_width, label=metric)
                
                plt.xlabel('Regime')
                plt.ylabel('Value')
                plt.title('Regime Characteristics')
                plt.xticks(r + bar_width * (len(available_metrics)-1)/2, [f"Regime {i}" for i in self.regime_stats['regime']])
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path.replace('.png', '_characteristics.png'), dpi=300, bbox_inches='tight')
                
                plt.show()
    
    def get_tradability_report(self, spread: pd.Series, price1: pd.Series, price2: pd.Series) -> Dict[str, Any]:
        """
        Generate a comprehensive tradability report.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread time series
        price1 : pandas.Series
            Price series of the first asset
        price2 : pandas.Series
            Price series of the second asset
            
        Returns:
        --------
        dict
            Dictionary with tradability assessment
        """
        # Detect cointegration breakdown
        breakdown_metrics = self.detect_cointegration_breakdown(price1, price2)
        
        # Detect regimes
        regime_results = self.detect_spread_regime(spread)
        
        # Combine results
        report = {
            **breakdown_metrics,
            **regime_results
        }
        
        # Add overall tradability assessment
        tradability = self.is_tradable(
            correlation=breakdown_metrics.get('avg_correlation'),
            coint_pvalue=breakdown_metrics.get('avg_coint_pvalue'),
            adf_pvalue=breakdown_metrics.get('avg_adf_pvalue')
        )
        
        report['tradability'] = tradability
        
        # Add timestamp
        report['timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return report 