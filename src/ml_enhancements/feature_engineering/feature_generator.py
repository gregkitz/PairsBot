"""
Feature Generator for Pairs Trading.

This module provides tools to create advanced features for pairs trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.preprocessing import StandardScaler
import talib
from scipy import stats
from scipy.signal import argrelextrema
import warnings


class FeatureGenerator:
    """
    Feature Generator for Pairs Trading.
    
    This class provides methods to create advanced features for improving signal
    generation in pairs trading strategies.
    """
    
    def __init__(self,
                lookback_window: int = 20,
                custom_windows: List[int] = None,
                scale_features: bool = True):
        """
        Initialize the feature generator.
        
        Parameters:
        -----------
        lookback_window : int
            Main lookback window for feature generation
        custom_windows : list of int, optional
            Additional lookback windows for feature generation
        scale_features : bool
            Whether to scale the generated features
        """
        self.lookback_window = lookback_window
        self.custom_windows = custom_windows if custom_windows is not None else [5, 10, 20, 50]
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        
        # Results
        self.features = None
        self.feature_importance = None
    
    def generate_features(self, 
                         spread: pd.Series, 
                         price1: Optional[pd.Series] = None,
                         price2: Optional[pd.Series] = None,
                         volume1: Optional[pd.Series] = None,
                         volume2: Optional[pd.Series] = None,
                         include_technical: bool = True,
                         include_statistical: bool = True,
                         include_seasonal: bool = True) -> pd.DataFrame:
        """
        Generate features for pairs trading.
        
        Parameters:
        -----------
        spread : pandas.Series
            Spread time series
        price1 : pandas.Series, optional
            Price series of the first asset
        price2 : pandas.Series, optional
            Price series of the second asset
        volume1 : pandas.Series, optional
            Volume series of the first asset
        volume2 : pandas.Series, optional
            Volume series of the second asset
        include_technical : bool
            Whether to include technical indicators
        include_statistical : bool
            Whether to include statistical features
        include_seasonal : bool
            Whether to include seasonal features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with generated features
        """
        # Initialize features DataFrame
        features = pd.DataFrame(index=spread.index)
        
        # Base features
        features['spread'] = spread
        
        # Add asset prices if provided
        if price1 is not None:
            features['price1'] = price1
        if price2 is not None:
            features['price2'] = price2
        
        # Add volume features if provided
        if volume1 is not None and volume2 is not None:
            features['volume1'] = volume1
            features['volume2'] = volume2
            features['volume_ratio'] = volume1 / volume2
            features['volume_imbalance'] = (volume1 - volume2) / (volume1 + volume2)
        
        # Generate rolling window features
        self._add_rolling_features(features, 'spread', self.custom_windows)
        
        # Generate lag features
        self._add_lag_features(features, 'spread', lags=[1, 2, 3, 5, 10])
        
        # Generate statistical features
        if include_statistical:
            self._add_statistical_features(features)
        
        # Generate technical indicators
        if include_technical:
            self._add_technical_features(features)
        
        # Generate seasonal features
        if include_seasonal:
            self._add_seasonal_features(features)
        
        # Remove rows with NaN values
        features = features.dropna()
        
        # Scale features if requested
        if self.scale_features and len(features) > 0:
            # Exclude columns that shouldn't be scaled
            cols_to_scale = [col for col in features.columns 
                            if col not in ['spread', 'price1', 'price2', 'volume1', 'volume2',
                                           'hour', 'day_of_week', 'month', 'is_morning',
                                           'is_afternoon', 'is_end_of_day']]
            
            if cols_to_scale:
                features[cols_to_scale] = self.scaler.fit_transform(features[cols_to_scale])
        
        self.features = features
        return features
    
    def _add_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int]):
        """
        Add rolling window features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add features to
        column : str
            Column to calculate features for
        windows : list of int
            List of rolling window sizes
        """
        for window in windows:
            if window <= len(df):
                # Basic statistics
                df[f'{column}_mean_{window}'] = df[column].rolling(window=window).mean()
                df[f'{column}_std_{window}'] = df[column].rolling(window=window).std()
                df[f'{column}_min_{window}'] = df[column].rolling(window=window).min()
                df[f'{column}_max_{window}'] = df[column].rolling(window=window).max()
                
                # Z-score
                df[f'{column}_zscore_{window}'] = (
                    df[column] - df[f'{column}_mean_{window}']
                ) / df[f'{column}_std_{window}']
                
                # Momentum
                df[f'{column}_momentum_{window}'] = df[column] - df[column].shift(window)
                
                # Rate of change
                df[f'{column}_roc_{window}'] = (
                    (df[column] - df[column].shift(window)) / df[column].shift(window)
                ) * 100
                
                # Bollinger Bands
                df[f'{column}_bb_upper_{window}'] = df[f'{column}_mean_{window}'] + 2 * df[f'{column}_std_{window}']
                df[f'{column}_bb_lower_{window}'] = df[f'{column}_mean_{window}'] - 2 * df[f'{column}_std_{window}']
                df[f'{column}_bb_width_{window}'] = (
                    df[f'{column}_bb_upper_{window}'] - df[f'{column}_bb_lower_{window}']
                ) / df[f'{column}_mean_{window}']
                df[f'{column}_bb_pct_{window}'] = (
                    df[column] - df[f'{column}_bb_lower_{window}']
                ) / (df[f'{column}_bb_upper_{window}'] - df[f'{column}_bb_lower_{window}'])
    
    def _add_lag_features(self, df: pd.DataFrame, column: str, lags: List[int]):
        """
        Add lagged features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add features to
        column : str
            Column to calculate features for
        lags : list of int
            List of lag values
        """
        for lag in lags:
            if lag < len(df):
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
                
                # Add difference features
                df[f'{column}_diff_{lag}'] = df[column] - df[f'{column}_lag_{lag}']
                
                # Add percentage change features
                df[f'{column}_pct_change_{lag}'] = df[column].pct_change(periods=lag)
                
                # Add acceleration features (change in momentum)
                if lag > 1:
                    df[f'{column}_accel_{lag}'] = df[f'{column}_diff_{lag}'] - df[f'{column}_diff_{lag}'].shift(1)
    
    def _add_statistical_features(self, df: pd.DataFrame):
        """
        Add statistical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add features to
        """
        # Calculate autocorrelation features
        for lag in [1, 5, 10]:
            if lag < len(df) - 20:  # Need some data for rolling window
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[f'autocorr_{lag}'] = df['spread'].rolling(window=20).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan, raw=False
                    )
        
        # Calculate moving average convergence/divergence
        if len(df) > 26:
            ema12 = df['spread'].ewm(span=12).mean()
            ema26 = df['spread'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            df['macd_cross'] = np.where(
                (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)),
                1,  # Bullish crossover
                np.where(
                    (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)),
                    -1,  # Bearish crossover
                    0  # No crossover
                )
            )
        
        # Calculate mean reversion features
        if len(df) > 20:
            # Mean reversion strength
            df['mean_reversion_strength'] = -1 * df['autocorr_1']  # Negative autocorrelation = stronger mean reversion
            
            # Mean reversion speed (half-life approximation)
            # Use a rolling window to estimate half-life
            window_size = min(60, len(df) // 2)
            if window_size > 10:
                half_lives = []
                for i in range(window_size, len(df)):
                    try:
                        spread_window = df['spread'].iloc[i-window_size:i]
                        spread_lag = spread_window.shift(1).iloc[1:]
                        spread_diff = spread_window.diff().iloc[1:]
                        model = stats.linregress(spread_lag, spread_diff)
                        half_life = -np.log(2) / model.slope if model.slope < 0 else np.nan
                        half_lives.append(half_life)
                    except:
                        half_lives.append(np.nan)
                
                # Pad with NaNs to match the DataFrame length
                half_lives = [np.nan] * window_size + half_lives
                df['half_life'] = half_lives
        
        # Volatility regime features
        if 'spread_std_5' in df.columns and 'spread_std_20' in df.columns:
            df['volatility_regime'] = df['spread_std_5'] / df['spread_std_20']
        
        # Historical extreme features
        if 'spread_zscore_20' in df.columns:
            df['is_extreme_high'] = df['spread_zscore_20'] > 2.0
            df['is_extreme_low'] = df['spread_zscore_20'] < -2.0
            
            # Distance to extreme values
            df['distance_to_mean'] = np.abs(df['spread_zscore_20'])
        
        # Local extrema features
        if len(df) > 10:
            # Find local maxima and minima using signal processing
            try:
                max_idx = argrelextrema(df['spread'].values, np.greater_equal, order=5)[0]
                min_idx = argrelextrema(df['spread'].values, np.less_equal, order=5)[0]
                
                df['is_local_max'] = False
                df['is_local_min'] = False
                
                df.iloc[max_idx, df.columns.get_loc('is_local_max')] = True
                df.iloc[min_idx, df.columns.get_loc('is_local_min')] = True
                
                # Distance to last extreme
                df['distance_to_last_max'] = np.nan
                df['distance_to_last_min'] = np.nan
                
                for i in range(len(df)):
                    if i > 0:
                        if df.iloc[i-1]['is_local_max']:
                            df.iloc[i, df.columns.get_loc('distance_to_last_max')] = 1
                        elif not np.isnan(df.iloc[i-1]['distance_to_last_max']):
                            df.iloc[i, df.columns.get_loc('distance_to_last_max')] = df.iloc[i-1]['distance_to_last_max'] + 1
                        
                        if df.iloc[i-1]['is_local_min']:
                            df.iloc[i, df.columns.get_loc('distance_to_last_min')] = 1
                        elif not np.isnan(df.iloc[i-1]['distance_to_last_min']):
                            df.iloc[i, df.columns.get_loc('distance_to_last_min')] = df.iloc[i-1]['distance_to_last_min'] + 1
            except:
                pass
    
    def _add_technical_features(self, df: pd.DataFrame):
        """
        Add technical indicators.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add features to
        """
        # Only compute if we have enough data
        if len(df) < 30:
            return
        
        try:
            # RSI
            df['rsi'] = talib.RSI(df['spread'].values, timeperiod=14)
            
            # Stochastic Oscillator
            df['slowk'], df['slowd'] = talib.STOCH(
                df['spread'].values, 
                df['spread'].values, 
                df['spread'].values,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            
            # CCI (Commodity Channel Index)
            df['cci'] = talib.CCI(
                df['spread'].values,
                df['spread'].values,
                df['spread'].values,
                timeperiod=20
            )
            
            # ADX (Average Directional Index)
            df['adx'] = talib.ADX(
                df['spread'].values,
                df['spread'].values,
                df['spread'].values,
                timeperiod=14
            )
            
            # OBV (On Balance Volume) - only if volume is available
            if 'volume1' in df.columns:
                df['obv'] = talib.OBV(df['spread'].values, df['volume1'].values)
            
            # ATR (Average True Range)
            df['atr'] = talib.ATR(
                df['spread'].values,
                df['spread'].values,
                df['spread'].values,
                timeperiod=14
            )
            
            # Williams %R
            df['willr'] = talib.WILLR(
                df['spread'].values,
                df['spread'].values,
                df['spread'].values,
                timeperiod=14
            )
            
            # Relative Volatility
            if 'spread_std_5' in df.columns and 'spread_std_20' in df.columns:
                df['rel_volatility'] = df['spread_std_5'] / df['spread_std_20']
        except:
            pass
    
    def _add_seasonal_features(self, df: pd.DataFrame):
        """
        Add seasonal features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add features to
        """
        # Only add if index is a datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Session features
        df['is_morning'] = (df['hour'] >= 9) & (df['hour'] < 12)
        df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 16)
        df['is_end_of_day'] = (df['hour'] >= 15) & (df['hour'] < 16)
        
        # Trading session volatility patterns
        if 'spread_std_5' in df.columns:
            # Group by hour and calculate average volatility
            hourly_vol = df.groupby('hour')['spread_std_5'].mean()
            
            # Map back to DataFrame
            df['hour_avg_volatility'] = df['hour'].map(hourly_vol)
            
            # Relative volatility compared to average for that hour
            df['rel_hour_volatility'] = df['spread_std_5'] / df['hour_avg_volatility']
    
    def select_important_features(self, target: pd.Series, method: str = 'correlation', threshold: float = 0.1, top_n: int = None):
        """
        Select important features based on correlation or other methods.
        
        Parameters:
        -----------
        target : pandas.Series
            Target variable (e.g., future returns)
        method : str
            Method for feature selection ('correlation', 'mutual_info', or 'random_forest')
        threshold : float
            Threshold for feature selection
        top_n : int, optional
            Number of top features to select
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with important features only
        """
        if self.features is None:
            raise ValueError("No features available. Generate features first.")
        
        # Ensure target and features have the same index
        common_idx = self.features.index.intersection(target.index)
        features = self.features.loc[common_idx]
        target = target.loc[common_idx]
        
        # Calculate feature importance
        importance = {}
        
        if method == 'correlation':
            # Calculate correlation with target
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    importance[col] = abs(features[col].corr(target))
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Select only numeric columns
            numeric_cols = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
            if not numeric_cols:
                return pd.DataFrame()
            
            # Calculate mutual information
            mi = mutual_info_regression(features[numeric_cols], target)
            for i, col in enumerate(numeric_cols):
                importance[col] = mi[i]
        
        elif method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            
            # Select only numeric columns
            numeric_cols = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
            if not numeric_cols:
                return pd.DataFrame()
            
            # Train a random forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(features[numeric_cols], target)
            
            # Get feature importance
            for i, col in enumerate(numeric_cols):
                importance[col] = model.feature_importances_[i]
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'correlation', 'mutual_info', or 'random_forest'.")
        
        # Convert to Series and sort
        self.feature_importance = pd.Series(importance).sort_values(ascending=False)
        
        # Select features
        if top_n is not None:
            important_features = self.feature_importance.head(top_n).index.tolist()
        else:
            important_features = self.feature_importance[self.feature_importance >= threshold].index.tolist()
        
        # Return only important features
        return self.features[important_features]
    
    def plot_feature_importance(self, top_n: int = 20, save_path=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Run select_important_features first.")
        
        plt.figure(figsize=(12, 10))
        
        # Plot top N features
        top_features = self.feature_importance.head(top_n)
        top_features.sort_values().plot(kind='barh')
        
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_prediction_features(self, df: pd.DataFrame, lookback: int = None) -> pd.DataFrame:
        """
        Create a feature set specifically for prediction.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with historical data including 'spread' column
        lookback : int, optional
            Number of historical periods to include. If None, uses self.lookback_window.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with features for prediction
        """
        if lookback is None:
            lookback = self.lookback_window
        
        # Get the most recent data points
        if len(df) > lookback:
            recent_data = df.iloc[-lookback:]
        else:
            recent_data = df.copy()
        
        # Generate features for the recent data
        features = self.generate_features(
            spread=recent_data['spread'],
            price1=recent_data.get('price1'),
            price2=recent_data.get('price2'),
            volume1=recent_data.get('volume1'),
            volume2=recent_data.get('volume2')
        )
        
        # Return the most recent feature set for prediction
        if not features.empty:
            return features.iloc[[-1]]
        else:
            return pd.DataFrame() 