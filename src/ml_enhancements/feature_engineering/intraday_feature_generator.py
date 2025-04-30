"""
Intraday Feature Generation Module

This module provides feature generation functionality for intraday ML signal enhancement,
focusing on creating features from price, spread, and volume data for use in 
machine learning models.
"""

import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayFeatureGenerator:
    """
    Generates features for intraday trading signal enhancement.
    
    This class calculates a wide range of features from price, spread, and volume data,
    which can be used as inputs to machine learning models for signal enhancement,
    entry/exit timing optimization, and regime detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize the intraday feature generator.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for feature generation
        """
        self.config = config or {
            "feature_lookback": 20,         # Lookback window for feature calculation
        }
        
        logger.info("IntradayFeatureGenerator initialized with lookback %d", 
                   self.config["feature_lookback"])

    def calculate_features(self, prices_df, spreads_df, volumes_df=None):
        """
        Calculate features for ML signal enhancement.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for each instrument
        spreads_df : pd.DataFrame
            DataFrame with spread data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data for each instrument
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=spreads_df.index)
        lookback = self.config["feature_lookback"]
        
        # 1. Spread features
        features = self._calculate_spread_features(features, spreads_df)
        
        # 2. Time features for intraday patterns
        features = self._calculate_time_features(features, spreads_df)
        
        # 3. Instrument-specific features
        features = self._calculate_instrument_features(features, prices_df, volumes_df)
        
        # 4. Correlation features
        features = self._calculate_correlation_features(features, prices_df, lookback)
        
        # 5. Mean reversion strength
        features = self._calculate_mean_reversion_features(features, spreads_df)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        logger.info("Generated %d features for %d time periods", 
                   len(features.columns), len(features))
        
        return features 

    def _calculate_spread_features(self, features, spreads_df):
        """
        Calculate features based on spread data.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread features added
        """
        if 'zscore' in spreads_df.columns:
            features['zscore'] = spreads_df['zscore']
            features['zscore_abs'] = np.abs(spreads_df['zscore'])
            
            # Z-score derivatives
            features['zscore_change'] = spreads_df['zscore'].diff()
            features['zscore_acceleration'] = features['zscore_change'].diff()
            
            # Z-score momentum
            features['zscore_mom_5'] = spreads_df['zscore'].diff(5)
            features['zscore_mom_10'] = spreads_df['zscore'].diff(10)
            
            # Z-score smoothing
            features['zscore_ma5'] = spreads_df['zscore'].rolling(5).mean()
            features['zscore_ma10'] = spreads_df['zscore'].rolling(10).mean()
            
            # Z-score volatility
            features['zscore_vol_5'] = spreads_df['zscore'].rolling(5).std()
            features['zscore_vol_10'] = spreads_df['zscore'].rolling(10).std()
            
            # Z-score extremes
            features['zscore_max_5'] = spreads_df['zscore'].rolling(5).max()
            features['zscore_min_5'] = spreads_df['zscore'].rolling(5).min()
            
        return features

    def _calculate_time_features(self, features, spreads_df):
        """
        Calculate time-based features from datetime index.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time features added
        """
        if isinstance(spreads_df.index, pd.DatetimeIndex):
            # Hour of day
            features['hour'] = spreads_df.index.hour
            
            # Minute of hour
            features['minute'] = spreads_df.index.minute
            
            # Time of day (decimal hours)
            features['tod'] = features['hour'] + features['minute'] / 60.0
            
            # Session progress (0-1 range)
            market_open = 9.5  # 9:30 AM
            market_close = 16.0  # 4:00 PM
            session_length = market_close - market_open
            features['session_progress'] = (features['tod'] - market_open) / session_length
            features['session_progress'] = features['session_progress'].clip(0, 1)
            
            # Session phase (early, mid, late)
            features['early_session'] = (features['session_progress'] < 0.33).astype(int)
            features['mid_session'] = ((features['session_progress'] >= 0.33) & 
                                      (features['session_progress'] < 0.67)).astype(int)
            features['late_session'] = (features['session_progress'] >= 0.67).astype(int)
            
            # Day of week
            features['day_of_week'] = spreads_df.index.dayofweek
            
        return features

    def _calculate_instrument_features(self, features, prices_df, volumes_df=None):
        """
        Calculate features specific to trading instruments.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        prices_df : pd.DataFrame
            DataFrame with price data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with instrument-specific features added
        """
        for col in prices_df.columns:
            prices = prices_df[col]
            
            # Price momentum
            features[f'{col}_mom_5'] = prices.pct_change(5)
            features[f'{col}_mom_10'] = prices.pct_change(10)
            
            # RSI
            # Use the internal implementation instead of TA-Lib
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility
            features[f'{col}_vol_5'] = prices.pct_change().rolling(5).std()
            features[f'{col}_vol_10'] = prices.pct_change().rolling(10).std()
            features[f'{col}_vol_20'] = prices.pct_change().rolling(20).std()
            
            # Volume features if available
            if volumes_df is not None and col in volumes_df.columns:
                vol = volumes_df[col]
                features[f'{col}_vol_ratio'] = vol / vol.rolling(20).mean()
                features[f'{col}_vol_change'] = vol.pct_change(5)
                
                # Volume momentum
                features[f'{col}_vol_mom_5'] = vol.pct_change(5)
                features[f'{col}_vol_mom_10'] = vol.pct_change(10)
                
                # Volume trend
                features[f'{col}_vol_trend'] = vol.rolling(5).mean() / vol.rolling(20).mean()
                
                # Price-volume correlation
                features[f'{col}_price_vol_corr'] = (
                    prices.pct_change().rolling(10)
                    .corr(vol.pct_change())
                    .fillna(0)
                )
                
        return features

    def _calculate_correlation_features(self, features, prices_df, lookback):
        """
        Calculate correlation-based features between instruments.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        prices_df : pd.DataFrame
            DataFrame with price data
        lookback : int
            Lookback window for calculations
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with correlation features added
        """
        price_returns = prices_df.pct_change()
        
        if len(prices_df.columns) >= 2:
            # Rolling correlation between pairs
            for i, col1 in enumerate(prices_df.columns):
                for j, col2 in enumerate(prices_df.columns):
                    if i < j:  # Only calculate each pair once
                        # Standard correlation
                        corr = price_returns[col1].rolling(lookback).corr(price_returns[col2])
                        features[f'corr_{col1}_{col2}'] = corr
                        
                        # Shorter-term correlation
                        corr_short = price_returns[col1].rolling(lookback // 2).corr(price_returns[col2])
                        features[f'corr_short_{col1}_{col2}'] = corr_short
                        
                        # Correlation change
                        features[f'corr_change_{col1}_{col2}'] = corr - corr.shift(5)
                        
        return features

    def _calculate_mean_reversion_features(self, features, spreads_df):
        """
        Calculate mean reversion strength features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with mean reversion features added
        """
        if 'zscore' in spreads_df.columns:
            # Half-life calculation (simplified for feature)
            zscore_lag = spreads_df['zscore'].shift(1)
            zscore_diff = spreads_df['zscore'] - zscore_lag
            
            # Regression of diff on lag to get mean reversion speed
            # We'll calculate this over rolling windows
            for window in [20, 50]:
                if len(zscore_diff) > window:
                    mr_speed = []
                    
                    for i in range(window, len(zscore_diff)):
                        y = zscore_diff.iloc[i-window:i].values
                        x = zscore_lag.iloc[i-window:i].values
                        x = np.reshape(x, (len(x), 1))
                        
                        # Add constant
                        x_with_const = np.column_stack([np.ones(len(x)), x])
                        
                        try:
                            # OLS regression
                            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                            mr_speed.append(beta[1])
                        except:
                            mr_speed.append(np.nan)
                    
                    # Pad with NaNs for the initial periods
                    mr_speed = [np.nan] * window + mr_speed
                    features[f'mr_speed_{window}'] = mr_speed
                    
                    # Convert to half-life
                    # Half-life = log(2) / log(1 + abs(beta))
                    features[f'half_life_{window}'] = np.log(2) / np.log(1 + np.abs(features[f'mr_speed_{window}']))
                    features[f'half_life_{window}'] = features[f'half_life_{window}'].replace([np.inf, -np.inf], np.nan)
                    
        return features 