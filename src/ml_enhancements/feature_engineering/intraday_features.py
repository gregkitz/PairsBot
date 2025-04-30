"""
Intraday Feature Engineering Module

This module provides specialized feature engineering for intraday trading,
focusing on time-of-day patterns, volatility profiles, and liquidity metrics.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import os

from src.ml_enhancements.feature_engineering.feature_generator import FeatureGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayFeatureEngineering:
    """
    Generate and analyze features specifically for intraday trading strategies.
    """
    
    def __init__(self, 
                 base_lookback: int = 20, 
                 custom_windows: List[int] = None,
                 output_dir: str = "output/feature_analysis"):
        """
        Initialize the intraday feature engineering system.
        
        Parameters:
        -----------
        base_lookback : int
            Base lookback window for feature calculation
        custom_windows : List[int], optional
            Custom lookback windows for multiple timeframes
        output_dir : str
            Directory to save feature analysis outputs
        """
        self.base_lookback = base_lookback
        self.custom_windows = custom_windows if custom_windows else [5, 10, 20, 50]
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize base feature generator
        self.feature_generator = FeatureGenerator(
            lookback_window=base_lookback,
            custom_windows=custom_windows,
            scale_features=False  # We'll scale at the end after adding intraday features
        )
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
        # Store feature importance data
        self.feature_importance = {}
    
    def generate_intraday_features(self,
                                   prices_df: pd.DataFrame,
                                   spreads_df: pd.DataFrame,
                                   volumes_df: Optional[pd.DataFrame] = None,
                                   include_time_features: bool = True,
                                   include_liquidity_features: bool = True,
                                   include_microstructure: bool = False,
                                   scale_output: bool = True) -> pd.DataFrame:
        """
        Generate features for intraday trading, including time-of-day features.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for each instrument in the pair
        spreads_df : pd.DataFrame
            DataFrame with spread data including z-scores
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data for each instrument
        include_time_features : bool
            Whether to include time-of-day features
        include_liquidity_features : bool
            Whether to include liquidity-related features
        include_microstructure : bool
            Whether to include market microstructure features (more data-intensive)
        scale_output : bool
            Whether to scale the output features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with generated features
        """
        if len(prices_df.columns) < 1:
            logger.error("Prices DataFrame must have at least one column")
            return pd.DataFrame()
        
        if 'spread' not in spreads_df.columns or 'zscore' not in spreads_df.columns:
            logger.error("Spreads DataFrame must have 'spread' and 'zscore' columns")
            return pd.DataFrame()
        
        # Extract symbols
        symbols = list(prices_df.columns)
        
        # Initialize features with base spread features
        symbol1 = symbols[0] if len(symbols) > 0 else None
        symbol2 = symbols[1] if len(symbols) > 1 else None
        
        # Use the base feature generator for standard features
        base_features = self.feature_generator.generate_features(
            spread=spreads_df['spread'],
            price1=prices_df[symbol1] if symbol1 else None,
            price2=prices_df[symbol2] if symbol2 else None,
            volume1=volumes_df[symbol1] if volumes_df is not None and symbol1 in volumes_df else None,
            volume2=volumes_df[symbol2] if volumes_df is not None and symbol2 in volumes_df else None
        )
        
        # Get index from base features
        features = pd.DataFrame(index=base_features.index)
        
        # Add z-score and other spread metrics from spreads_df
        for col in ['zscore', 'hedge_ratio', 'spread', 'mean', 'std']:
            if col in spreads_df.columns:
                features[col] = spreads_df.loc[features.index, col]
        
        # Add base features (exclude duplicates)
        for col in base_features.columns:
            if col not in features.columns:
                features[col] = base_features[col]
        
        # Add time-of-day features
        if include_time_features and isinstance(features.index, pd.DatetimeIndex):
            features = self._add_time_features(features)
        
        # Add liquidity features
        if include_liquidity_features and volumes_df is not None:
            features = self._add_liquidity_features(features, volumes_df)
        
        # Add market microstructure features
        if include_microstructure and volumes_df is not None:
            features = self._add_microstructure_features(features, prices_df, volumes_df)
        
        # Add intraday spread reversion features
        features = self._add_intraday_reversion_features(features, spreads_df)
        
        # Add intraday regime features
        features = self._add_intraday_regime_features(features, prices_df)
        
        # Scale features if requested
        if scale_output:
            # Get numeric columns only
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-of-day and session-related features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time features added
        """
        # Extract time components
        features['hour'] = features.index.hour
        features['minute'] = features.index.minute
        features['day_of_week'] = features.index.dayofweek
        
        # Time of day as decimal (e.g., 9:30 = 9.5)
        features['time_of_day'] = features['hour'] + features['minute'] / 60.0
        
        # Market session (3 phases)
        # Market open: 9:30 AM - 11:30 AM
        # Mid-day: 11:30 AM - 2:00 PM
        # Market close: 2:00 PM - 4:00 PM
        features['session_open'] = ((features['time_of_day'] >= 9.5) & 
                                   (features['time_of_day'] < 11.5)).astype(int)
        features['session_mid'] = ((features['time_of_day'] >= 11.5) & 
                                   (features['time_of_day'] < 14.0)).astype(int)
        features['session_close'] = ((features['time_of_day'] >= 14.0) & 
                                     (features['time_of_day'] <= 16.0)).astype(int)
        
        # Normalized time within session (0-1)
        market_open = 9.5  # 9:30 AM
        market_close = 16.0  # 4:00 PM
        session_length = market_close - market_open
        features['session_progress'] = (features['time_of_day'] - market_open) / session_length
        features['session_progress'] = features['session_progress'].clip(0, 1)
        
        # First and last 30 minutes of the trading day
        features['first_30min'] = ((features['time_of_day'] >= 9.5) & 
                                  (features['time_of_day'] < 10.0)).astype(int)
        features['last_30min'] = ((features['time_of_day'] >= 15.5) & 
                                 (features['time_of_day'] <= 16.0)).astype(int)
        
        # Day of week (one-hot encoding)
        for day in range(5):  # Monday=0, Friday=4
            features[f'day_{day}'] = (features['day_of_week'] == day).astype(int)
        
        return features
    
    def _add_liquidity_features(self, features: pd.DataFrame, volumes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add liquidity-related features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        volumes_df : pd.DataFrame
            DataFrame with volume data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with liquidity features added
        """
        for col in volumes_df.columns:
            # Get volume for this symbol
            volume = volumes_df[col].reindex(features.index)
            
            # Volume relative to average
            features[f'{col}_vol_ratio'] = volume / volume.rolling(window=self.base_lookback).mean()
            
            # Volume momentum
            features[f'{col}_vol_mom'] = volume.pct_change(5)
            
            # Volume trend (regression slope over lookback)
            for window in [20, 50]:
                if len(volume) >= window:
                    features[f'{col}_vol_trend_{window}'] = calculate_slope(volume, window)
            
            # Intraday volume profile
            # Group by hour and calculate average volume for each hour
            if isinstance(features.index, pd.DatetimeIndex):
                hourly_avg = volume.groupby(volume.index.hour).mean()
                
                # Map each bar's hour to its average volume
                hour_to_avg = dict(zip(hourly_avg.index, hourly_avg.values))
                
                # Calculate relative volume (current / average for this hour)
                features[f'{col}_hour_vol_ratio'] = volume / features['hour'].map(hour_to_avg)
        
        # Pair volume features
        if len(volumes_df.columns) >= 2:
            # Volume ratio between assets
            symbol1, symbol2 = volumes_df.columns[0], volumes_df.columns[1]
            features['vol_imbalance'] = volumes_df[symbol1] / volumes_df[symbol2]
            
            # Normalized volume imbalance
            vol_imb_mean = features['vol_imbalance'].rolling(window=self.base_lookback).mean()
            vol_imb_std = features['vol_imbalance'].rolling(window=self.base_lookback).std()
            features['vol_imbalance_zscore'] = (features['vol_imbalance'] - vol_imb_mean) / vol_imb_std
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, prices_df: pd.DataFrame, volumes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market microstructure features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        prices_df : pd.DataFrame
            DataFrame with price data
        volumes_df : pd.DataFrame
            DataFrame with volume data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with microstructure features added
        """
        for col in prices_df.columns:
            price = prices_df[col].reindex(features.index)
            volume = volumes_df[col].reindex(features.index) if col in volumes_df else None
            
            if volume is not None:
                # VWAP (Volume Weighted Average Price)
                # Calculate cumulative (price * volume) and cumulative volume
                cum_pv = (price * volume).cumsum()
                cum_vol = volume.cumsum()
                
                # Calculate VWAP
                vwap = cum_pv / cum_vol
                
                # VWAP-based features
                features[f'{col}_vwap'] = vwap
                features[f'{col}_vwap_diff'] = (price - vwap) / vwap
                
                # Volume-weighted momentum
                features[f'{col}_vw_mom'] = price.diff() * volume
                
                # VWAP trend
                for window in [20, 50]:
                    if len(vwap) >= window:
                        features[f'{col}_vwap_trend_{window}'] = calculate_slope(vwap, window)
            
            # Price acceleration
            price_diff = price.diff()
            features[f'{col}_accel'] = price_diff.diff()
            
            # Intraday high-low range
            if isinstance(features.index, pd.DatetimeIndex):
                day_groups = price.groupby(price.index.date)
                
                # Calculate daily high and low
                daily_high = day_groups.transform('max')
                daily_low = day_groups.transform('min')
                
                # Calculate relative position within daily range
                daily_range = daily_high - daily_low
                rel_pos = (price - daily_low) / daily_range
                features[f'{col}_day_position'] = rel_pos
                
                # Distance from day high/low
                features[f'{col}_from_day_high'] = (daily_high - price) / price
                features[f'{col}_from_day_low'] = (price - daily_low) / price
        
        return features
    
    def _add_intraday_reversion_features(self, features: pd.DataFrame, spreads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to intraday mean reversion patterns.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with intraday reversion features added
        """
        if 'zscore' not in spreads_df.columns:
            return features
        
        zscore = spreads_df['zscore'].reindex(features.index)
        
        # Extreme z-score indicators
        features['zscore_extreme_pos'] = (zscore > 2.0).astype(int)
        features['zscore_extreme_neg'] = (zscore < -2.0).astype(int)
        
        # Z-score derivatives
        features['zscore_diff'] = zscore.diff()
        features['zscore_diff2'] = features['zscore_diff'].diff()  # Second derivative
        
        # Z-score direction change
        zscore_diff_sign = np.sign(features['zscore_diff'])
        features['zscore_direction_change'] = (zscore_diff_sign != zscore_diff_sign.shift(1)).astype(int)
        
        # Z-score extrema
        # Local maxima and minima within lookback window
        for window in [20, 50]:
            if len(zscore) >= window:
                # Use rolling max/min with centered window
                rolling_max = zscore.rolling(window=window, center=True).max()
                rolling_min = zscore.rolling(window=window, center=True).min()
                
                # Identify local extrema
                features[f'zscore_local_max_{window}'] = (zscore == rolling_max).astype(int)
                features[f'zscore_local_min_{window}'] = (zscore == rolling_min).astype(int)
        
        # Time since extreme z-score
        if 'session_progress' in features.columns:
            # Track consecutive periods with extreme z-scores
            extreme_pos_count = (zscore > 2.0).astype(int).groupby(features.index.date).cumsum()
            extreme_neg_count = (zscore < -2.0).astype(int).groupby(features.index.date).cumsum()
            
            features['zscore_extreme_pos_duration'] = extreme_pos_count
            features['zscore_extreme_neg_duration'] = extreme_neg_count
            
            # Reset counts when z-score returns to normal range
            normal_range = (zscore >= -0.5) & (zscore <= 0.5)
            extreme_pos_reset = (normal_range & (extreme_pos_count > 0)).astype(int)
            extreme_neg_reset = (normal_range & (extreme_neg_count > 0)).astype(int)
            
            features['zscore_pos_mean_reversion'] = extreme_pos_reset
            features['zscore_neg_mean_reversion'] = extreme_neg_reset
        
        # Intraday seasonality of z-score
        if isinstance(features.index, pd.DatetimeIndex):
            # Average z-score by hour
            hourly_zscore = zscore.groupby(zscore.index.hour).mean()
            
            # Map each bar's hour to its average z-score
            hour_to_zscore = dict(zip(hourly_zscore.index, hourly_zscore.values))
            
            # Historical z-score for this hour
            features['zscore_hour_avg'] = features['hour'].map(hour_to_zscore)
            
            # Z-score deviation from hourly average
            features['zscore_hour_deviation'] = zscore - features['zscore_hour_avg']
        
        return features
    
    def _add_intraday_regime_features(self, features: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to intraday market regimes.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        prices_df : pd.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with market regime features added
        """
        # Correlation features
        if len(prices_df.columns) >= 2:
            symbol1, symbol2 = prices_df.columns[0], prices_df.columns[1]
            
            # Extract returns
            returns1 = prices_df[symbol1].pct_change().reindex(features.index)
            returns2 = prices_df[symbol2].pct_change().reindex(features.index)
            
            # Rolling correlation
            for window in [20, 50]:
                if len(returns1) >= window and len(returns2) >= window:
                    features[f'rolling_corr_{window}'] = returns1.rolling(window).corr(returns2)
            
            # Intraday correlation stability
            if 'rolling_corr_20' in features.columns and 'rolling_corr_50' in features.columns:
                features['corr_stability'] = 1 - abs(features['rolling_corr_20'] - features['rolling_corr_50'])
        
        # Volatility regime features
        for col in prices_df.columns:
            returns = prices_df[col].pct_change().reindex(features.index)
            
            # Rolling volatility at different windows
            for window in [10, 20, 50]:
                if len(returns) >= window:
                    features[f'{col}_vol_{window}'] = returns.rolling(window).std()
            
            # Volatility of volatility
            if f'{col}_vol_20' in features.columns:
                features[f'{col}_vol_of_vol'] = features[f'{col}_vol_20'].rolling(20).std()
            
            # Volatility regime classification
            if all(f'{col}_vol_{w}' in features.columns for w in [10, 50]):
                # Ratio of short-term to long-term volatility
                features[f'{col}_vol_ratio'] = features[f'{col}_vol_10'] / features[f'{col}_vol_50']
                
                # Classify volatility regime (high/normal/low)
                vol_ratio = features[f'{col}_vol_ratio']
                features[f'{col}_high_vol_regime'] = (vol_ratio > 1.2).astype(int)
                features[f'{col}_low_vol_regime'] = (vol_ratio < 0.8).astype(int)
                features[f'{col}_normal_vol_regime'] = ((vol_ratio >= 0.8) & (vol_ratio <= 1.2)).astype(int)
        
        # Trend/mean-reversion regime detection
        if 'zscore' in features.columns:
            zscore = features['zscore']
            
            # Count zero crossings in z-score (more crossings = more mean-reverting)
            for window in [20, 50]:
                if len(zscore) >= window:
                    # Identify zero crossings (sign changes)
                    sign_changes = ((np.sign(zscore) != np.sign(zscore.shift(1))) & 
                                   (zscore.shift(1) != 0)).rolling(window).sum()
                    
                    features[f'zscore_crossings_{window}'] = sign_changes
                    
                    # Classify regime based on crossings
                    # More crossings = more mean-reverting, fewer = more trending
                    expected_crossings = window / 10  # Heuristic
                    features[f'mean_reverting_regime_{window}'] = (sign_changes > expected_crossings).astype(int)
                    features[f'trending_regime_{window}'] = (sign_changes < expected_crossings/2).astype(int)
        
        return features
    
    def analyze_feature_importance(self, features: pd.DataFrame, target: pd.Series, 
                                  method: str = 'mutual_info', output_file: str = None,
                                  n_features: int = 20) -> Dict[str, float]:
        """
        Analyze feature importance relative to a target variable.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        target : pd.Series
            Target variable for importance analysis
        method : str
            Method for importance calculation ('mutual_info', 'f_regression', 'correlation')
        output_file : str, optional
            Path to save visualization
        n_features : int
            Number of top features to select
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with feature importance scores
        """
        if len(features) != len(target):
            logger.error("Features and target must have the same length")
            return {}
        
        # Align indices
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Calculate feature importance
        importance_dict = {}
        
        if method == 'correlation':
            # Calculate correlation with target
            correlations = X.corrwith(y).abs()
            importance_dict = correlations.to_dict()
        
        elif method == 'mutual_info':
            # Use mutual information
            selector = SelectKBest(mutual_info_regression, k='all')
            selector.fit(X, y)
            importance_dict = dict(zip(X.columns, selector.scores_))
        
        elif method == 'f_regression':
            # Use F-regression
            selector = SelectKBest(f_regression, k='all')
            selector.fit(X, y)
            importance_dict = dict(zip(X.columns, selector.scores_))
        
        else:
            logger.error(f"Unknown method: {method}")
            return {}
        
        # Sort by importance
        importance_dict = {k: v for k, v in sorted(importance_dict.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)}
        
        # Store feature importance
        self.feature_importance = importance_dict
        
        # Visualize if needed
        if output_file is not None:
            self.visualize_feature_importance(n_features=n_features, output_file=output_file)
        
        return importance_dict
    
    def visualize_feature_importance(self, n_features: int = 20, output_file: str = None) -> str:
        """
        Visualize feature importance.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to show
        output_file : str, optional
            Path to save visualization
            
        Returns:
        --------
        str
            Path to saved visualization
        """
        if not self.feature_importance:
            logger.error("No feature importance data to visualize")
            return None
        
        # Get top N features
        top_features = list(self.feature_importance.items())[:n_features]
        
        # Sort by importance
        top_features.sort(key=lambda x: x[1])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot horizontal bar chart
        feature_names = [x[0] for x in top_features]
        importance_values = [x[1] for x in top_features]
        
        plt.barh(feature_names, importance_values)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Top {n_features} Feature Importance')
        plt.tight_layout()
        
        # Save or show
        if output_file is None:
            output_file = os.path.join(self.output_dir, "feature_importance.png")
        
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def select_important_features(self, 
                                 features: pd.DataFrame, 
                                 importance_threshold: float = 0.1,
                                 max_features: int = None) -> pd.DataFrame:
        """
        Select important features based on importance scores.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        importance_threshold : float
            Minimum importance score to include
        max_features : int, optional
            Maximum number of features to include
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features
        """
        if not self.feature_importance:
            logger.error("No feature importance data for selection")
            return features
        
        # Filter by threshold
        selected_features = [k for k, v in self.feature_importance.items() 
                            if v >= importance_threshold]
        
        # Limit to max_features if specified
        if max_features is not None and len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        # Ensure all selected features exist in the DataFrame
        selected_features = [f for f in selected_features if f in features.columns]
        
        logger.info(f"Selected {len(selected_features)} features based on importance")
        
        return features[selected_features]


def calculate_slope(series, window):
    """
    Calculate the slope of a time series over a window.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    window : int
        Window size for calculation
        
    Returns:
    --------
    pd.Series
        Series of slope values
    """
    slopes = []
    
    for i in range(window, len(series) + 1):
        # Get the window of data
        y = series.iloc[i-window:i].values
        x = np.arange(len(y))
        
        # Add constant column
        x = np.vstack([x, np.ones(len(x))]).T
        
        # Linear regression
        try:
            slope, _ = np.linalg.lstsq(x, y, rcond=None)[0]
            slopes.append(slope)
        except:
            slopes.append(np.nan)
    
    # Create Series with same index as input
    result = pd.Series(index=series.index[window-1:], data=slopes)
    
    return result 