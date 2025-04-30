"""
Advanced Feature Engineering Module

This module extends the base intraday feature engineering with more sophisticated 
features including:
- Advanced technical indicators
- Cross-asset interaction features
- Nonlinear transformations
- Temporal pattern detection
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, time, timedelta
import logging
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import find_peaks

from src.ml_enhancements.feature_engineering.intraday_features import IntradayFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for intraday trading.
    
    This class extends the base IntradayFeatureEngineering class with more sophisticated
    features for improved model performance.
    """
    
    def __init__(self, 
                 base_lookback: int = 20, 
                 custom_windows: List[int] = None,
                 output_dir: str = "output/feature_analysis/advanced"):
        """
        Initialize the advanced feature engineering system.
        
        Parameters:
        -----------
        base_lookback : int
            Base lookback window for feature calculation
        custom_windows : List[int], optional
            Custom lookback windows for multiple timeframes
        output_dir : str
            Directory to save feature analysis outputs
        """
        # Initialize base feature engineering
        self.base_features = IntradayFeatureEngineering(
            base_lookback=base_lookback, 
            custom_windows=custom_windows,
            output_dir=output_dir
        )
        
        self.base_lookback = base_lookback
        self.custom_windows = custom_windows if custom_windows else [5, 10, 20, 50]
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scalers and transformers
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Store feature importance data
        self.feature_importance = {}
    
    def generate_advanced_features(self,
                                  prices_df: pd.DataFrame,
                                  spreads_df: pd.DataFrame,
                                  volumes_df: Optional[pd.DataFrame] = None,
                                  include_base_features: bool = True,
                                  include_nonlinear: bool = True,
                                  include_temporal: bool = True,
                                  include_interaction: bool = True,
                                  scale_output: bool = True) -> pd.DataFrame:
        """
        Generate advanced features for intraday trading.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for multiple instruments
        spreads_df : pd.DataFrame
            DataFrame with spread data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
        include_base_features : bool
            Whether to include base features
        include_nonlinear : bool
            Whether to include nonlinear transformations
        include_temporal : bool
            Whether to include temporal pattern features
        include_interaction : bool
            Whether to include interaction features
        scale_output : bool
            Whether to scale output features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with advanced features
        """
        # Generate base features first
        if include_base_features:
            features = self.base_features.generate_intraday_features(
                prices_df=prices_df,
                spreads_df=spreads_df,
                volumes_df=volumes_df,
                scale_output=False
            )
        else:
            symbols = prices_df.columns.tolist()
            features = pd.DataFrame(index=prices_df.index)
            
            # Add basic spread info
            for col in ['zscore', 'hedge_ratio', 'spread', 'mean', 'std']:
                if col in spreads_df.columns:
                    features[col] = spreads_df.loc[features.index, col]
        
        # Add advanced technical indicators
        features = self._add_advanced_indicators(features, prices_df, spreads_df)
        
        # Add nonlinear transformations
        if include_nonlinear:
            features = self._add_nonlinear_features(features, spreads_df)
        
        # Add temporal pattern features
        if include_temporal:
            features = self._add_temporal_pattern_features(features, spreads_df)
        
        # Add interaction features
        if include_interaction:
            features = self._add_interaction_features(features, prices_df, volumes_df)
        
        # Scale features if requested
        if scale_output:
            # Get numeric columns only
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        
        return features
    
    def _add_advanced_indicators(self, 
                                features: pd.DataFrame, 
                                prices_df: pd.DataFrame,
                                spreads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        prices_df : pd.DataFrame
            DataFrame with price data
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with advanced indicators added
        """
        # Add indicators for each instrument
        for col in prices_df.columns:
            price = prices_df[col].values
            
            if len(price) > 50:  # Ensure enough data
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(price, timeperiod=20, nbdevup=2, nbdevdn=2)
                features[f'{col}_bb_upper'] = upper
                features[f'{col}_bb_middle'] = middle
                features[f'{col}_bb_lower'] = lower
                features[f'{col}_bb_width'] = (upper - lower) / middle
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(price, fastperiod=12, slowperiod=26, signalperiod=9)
                features[f'{col}_macd'] = macd
                features[f'{col}_macd_signal'] = macd_signal
                features[f'{col}_macd_hist'] = macd_hist
                
                # Commodity Channel Index
                cci = talib.CCI(price, price, price, timeperiod=14)  # Using same price for high/low/close
                features[f'{col}_cci'] = cci
                
                # Rate of Change
                for period in [5, 10, 20]:
                    roc = talib.ROC(price, timeperiod=period)
                    features[f'{col}_roc_{period}'] = roc
                
                # Ichimoku Cloud - simplified version
                high = price  # Using close as proxy for high/low
                low = price
                tenkan_period = 9
                kijun_period = 26
                
                tenkan_sen = (pd.Series(high).rolling(tenkan_period).max() + 
                             pd.Series(low).rolling(tenkan_period).min()) / 2
                kijun_sen = (pd.Series(high).rolling(kijun_period).max() + 
                            pd.Series(low).rolling(kijun_period).min()) / 2
                
                features[f'{col}_tenkan_sen'] = tenkan_sen.values
                features[f'{col}_kijun_sen'] = kijun_sen.values
                features[f'{col}_tenkan_kijun_diff'] = (tenkan_sen - kijun_sen).values
        
        # Add indicators for spread
        if 'spread' in spreads_df.columns:
            spread = spreads_df['spread'].values
            
            if len(spread) > 50:
                # Bollinger Bands on spread
                upper, middle, lower = talib.BBANDS(spread, timeperiod=20, nbdevup=2, nbdevdn=2)
                features['spread_bb_upper'] = upper
                features['spread_bb_lower'] = lower
                features['spread_bb_width'] = (upper - lower) / middle
                
                # RSI for spread
                for period in [7, 14, 21]:
                    rsi = talib.RSI(spread, timeperiod=period)
                    features[f'spread_rsi_{period}'] = rsi
                
                # Stochastic oscillator for spread
                high = pd.Series(spread).rolling(14).max()
                low = pd.Series(spread).rolling(14).min()
                k = 100 * ((pd.Series(spread) - low) / (high - low))
                features['spread_stoch_k'] = k.values
                features['spread_stoch_d'] = k.rolling(3).mean().values
        
        # Z-score extremes
        if 'zscore' in spreads_df.columns:
            zscore = spreads_df['zscore']
            features['zscore_abs'] = np.abs(zscore)
            features['zscore_squared'] = zscore ** 2
            
            # Extreme Z-score indicators
            features['zscore_extreme_positive'] = (zscore > 2).astype(int)
            features['zscore_extreme_negative'] = (zscore < -2).astype(int)
            features['zscore_neutral'] = ((zscore >= -0.5) & (zscore <= 0.5)).astype(int)
        
        return features
    
    def _add_nonlinear_features(self, 
                               features: pd.DataFrame, 
                               spreads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add nonlinear transformations of features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with nonlinear features added
        """
        # Get columns to transform (z-score and spread-related)
        transform_cols = [col for col in features.columns if any(
            keyword in col for keyword in ['zscore', 'spread', 'correlation', 'vol']
        )]
        
        for col in transform_cols:
            if col in features.columns:
                # Skip columns with NaN or zero values
                if features[col].isna().sum() > 0 or (features[col] == 0).sum() > 0:
                    continue
                
                # Log transformation for positive values
                if features[col].min() > 0:
                    features[f'{col}_log'] = np.log(features[col])
                
                # Square root for positive values
                if features[col].min() >= 0:
                    features[f'{col}_sqrt'] = np.sqrt(features[col])
                
                # Square and cube
                features[f'{col}_squared'] = features[col] ** 2
                features[f'{col}_cubed'] = features[col] ** 3
                
                # Sigmoid transformation
                features[f'{col}_sigmoid'] = 1 / (1 + np.exp(-features[col]))
                
                # Hyperbolic tangent
                features[f'{col}_tanh'] = np.tanh(features[col])
        
        # Add threshold crossing features for z-score
        if 'zscore' in spreads_df.columns:
            zscore = spreads_df['zscore']
            
            for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
                # Positive threshold crossing
                features[f'zscore_cross_pos_{threshold:.1f}'] = (
                    (zscore > threshold) & (zscore.shift(1) <= threshold)
                ).astype(int)
                
                # Negative threshold crossing
                features[f'zscore_cross_neg_{threshold:.1f}'] = (
                    (zscore < -threshold) & (zscore.shift(1) >= -threshold)
                ).astype(int)
        
        return features
    
    def _add_temporal_pattern_features(self, 
                                      features: pd.DataFrame, 
                                      spreads_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal pattern detection features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame to add features to
        spreads_df : pd.DataFrame
            DataFrame with spread data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with temporal pattern features added
        """
        if not isinstance(features.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex. Skipping temporal pattern features.")
            return features
        
        # Time-based seasonality
        features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        
        # Add features for special market times
        market_open = (features.index.hour == 9) & (features.index.minute >= 30)
        market_close = (features.index.hour == 15) & (features.index.minute >= 45)
        
        features['is_market_open'] = market_open.astype(int)
        features['is_market_close'] = market_close.astype(int)
        features['is_lunch_hour'] = ((features.index.hour == 12) | 
                                    ((features.index.hour == 13) & 
                                     (features.index.minute < 30))).astype(int)
        
        # Pattern detection in z-score
        if 'zscore' in spreads_df.columns:
            zscore = spreads_df['zscore']
            
            # Check for mean reversion pattern - crossing back toward mean
            features['zscore_mean_reversion'] = (
                ((zscore > 0) & (zscore.diff() < 0)) | 
                ((zscore < 0) & (zscore.diff() > 0))
            ).astype(int)
            
            # Momentum continuation
            features['zscore_momentum'] = (
                ((zscore > 0) & (zscore.diff() > 0)) | 
                ((zscore < 0) & (zscore.diff() < 0))
            ).astype(int)
            
            # Z-score acceleration (change in the change)
            features['zscore_acceleration'] = zscore.diff().diff()
            
            # Z-score oscillation detection
            if len(zscore) >= 30:
                try:
                    # Calculate auto-correlation for oscillation detection
                    autocorr = acf(zscore.dropna(), nlags=10)
                    
                    # Find peaks in autocorrelation (indicates oscillation)
                    peaks, _ = find_peaks(autocorr)
                    
                    if len(peaks) > 0:
                        # Estimate oscillation period from first peak
                        osc_period = peaks[0] if peaks[0] > 0 else 0
                        
                        # Add oscillation strength feature
                        features['zscore_osc_strength'] = autocorr[osc_period] if osc_period > 0 else 0
                        
                        # Add oscillation period feature
                        features['zscore_osc_period'] = osc_period
                except Exception as e:
                    logger.warning(f"Error calculating oscillation features: {e}")
        
        return features
    
    def _add_interaction_features(self, 
                                 features: pd.DataFrame, 
                                 prices_df: pd.DataFrame,
                                 volumes_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Add interaction features between different instruments.
        
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
            DataFrame with interaction features added
        """
        symbols = prices_df.columns.tolist()
        
        if len(symbols) >= 2:
            # Pairwise relative strength
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates
                        # Relative price performance
                        rel_perf = prices_df[symbol1] / prices_df[symbol2]
                        features[f'{symbol1}_{symbol2}_rel_perf'] = rel_perf
                        
                        # Normalized relative performance
                        rel_perf_norm = rel_perf / rel_perf.rolling(20).mean()
                        features[f'{symbol1}_{symbol2}_rel_perf_norm'] = rel_perf_norm
                        
                        # Calculate relative momentum
                        rel_mom_5 = rel_perf / rel_perf.shift(5) - 1
                        rel_mom_20 = rel_perf / rel_perf.shift(20) - 1
                        features[f'{symbol1}_{symbol2}_rel_mom_5'] = rel_mom_5
                        features[f'{symbol1}_{symbol2}_rel_mom_20'] = rel_mom_20
                        
                        # Relative strength divergence
                        price_corr = prices_df[symbol1].pct_change().rolling(20).corr(
                            prices_df[symbol2].pct_change()
                        )
                        features[f'{symbol1}_{symbol2}_price_corr'] = price_corr
        
        # Add volume-price interaction features
        if volumes_df is not None:
            for symbol in symbols:
                if symbol in volumes_df.columns and symbol in prices_df.columns:
                    # Price-volume correlation
                    pv_corr = prices_df[symbol].pct_change().rolling(20).corr(
                        volumes_df[symbol].pct_change()
                    )
                    features[f'{symbol}_price_vol_corr'] = pv_corr
                    
                    # Volume-weighted price
                    vwap = (prices_df[symbol] * volumes_df[symbol]).rolling(20).sum() / volumes_df[symbol].rolling(20).sum()
                    features[f'{symbol}_vwap'] = vwap
                    
                    # Price deviation from VWAP
                    features[f'{symbol}_vwap_dev'] = (prices_df[symbol] - vwap) / vwap
                    
                    # Money flow index components
                    if len(prices_df) > 14:
                        # Typical price approximation (using close price)
                        typical_price = prices_df[symbol]
                        
                        # Money flow
                        money_flow = typical_price * volumes_df[symbol]
                        
                        # Positive and negative money flow
                        pos_flow = money_flow.copy()
                        neg_flow = money_flow.copy()
                        
                        # Set values where price decreased to 0 for positive flow
                        pos_flow[typical_price.diff() <= 0] = 0
                        
                        # Set values where price increased to 0 for negative flow
                        neg_flow[typical_price.diff() >= 0] = 0
                        
                        # Calculate 14-period sums
                        pos_flow_sum = pos_flow.rolling(14).sum()
                        neg_flow_sum = neg_flow.rolling(14).sum()
                        
                        # Money Ratio
                        money_ratio = pos_flow_sum / (neg_flow_sum + 1e-10)  # Avoid division by zero
                        
                        # Money Flow Index
                        mfi = 100 - (100 / (1 + money_ratio))
                        features[f'{symbol}_mfi'] = mfi
        
        return features
    
    def reduce_dimensionality(self, features: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """
        Reduce dimensionality of features using PCA.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        n_components : float
            Number of components to keep (can be fraction of variance)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with reduced features
        """
        # Get numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for PCA.")
            return features
        
        # Scale data first
        X_scaled = self.scaler.fit_transform(features[numeric_cols])
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create new DataFrame with PCA components
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        pca_df = pd.DataFrame(X_pca, index=features.index, columns=pca_cols)
        
        # Add non-numeric columns back
        non_numeric_cols = features.columns.difference(numeric_cols)
        if len(non_numeric_cols) > 0:
            pca_df = pd.concat([pca_df, features[non_numeric_cols]], axis=1)
        
        logger.info(f"Reduced features from {len(numeric_cols)} to {len(pca_cols)} components")
        
        return pca_df
    
    def select_top_features(self, 
                           features: pd.DataFrame, 
                           target: pd.Series, 
                           n_features: int = 20,
                           method: str = 'mutual_info') -> pd.DataFrame:
        """
        Select top features based on importance to target.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        target : pd.Series
            Target variable
        n_features : int
            Number of features to select
        method : str
            Method for feature selection ('mutual_info', 'f_regression', 'correlation')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with selected features
        """
        # Get numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for feature selection.")
            return features
        
        # Ensure no missing values
        X = features[numeric_cols].fillna(0)
        y = target.fillna(0)
        
        # Select features based on method
        if method == 'correlation':
            # Calculate correlation with target
            corr = pd.DataFrame(index=numeric_cols)
            corr['correlation'] = [abs(X[col].corr(y)) for col in numeric_cols]
            corr = corr.sort_values('correlation', ascending=False)
            
            # Select top features
            selected_features = corr.index[:n_features].tolist()
            feature_importance = corr['correlation'].to_dict()
            
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi = mutual_info_regression(X, y)
            mi_df = pd.DataFrame({'feature': numeric_cols, 'importance': mi})
            mi_df = mi_df.sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = mi_df['feature'][:n_features].tolist()
            feature_importance = dict(zip(mi_df['feature'], mi_df['importance']))
            
        elif method == 'f_regression':
            from sklearn.feature_selection import f_regression
            
            # Calculate F-statistic
            f_val, _ = f_regression(X, y)
            f_df = pd.DataFrame({'feature': numeric_cols, 'importance': f_val})
            f_df = f_df.sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = f_df['feature'][:n_features].tolist()
            feature_importance = dict(zip(f_df['feature'], f_df['importance']))
            
        else:
            logger.warning(f"Unknown method: {method}. Using correlation instead.")
            return self.select_top_features(features, target, n_features, 'correlation')
        
        # Store feature importance
        self.feature_importance = feature_importance
        
        # Return selected features
        selected_df = features[selected_features]
        
        # Add non-numeric columns if requested
        non_numeric_cols = features.columns.difference(numeric_cols)
        if len(non_numeric_cols) > 0:
            selected_df = pd.concat([selected_df, features[non_numeric_cols]], axis=1)
        
        logger.info(f"Selected {len(selected_features)} top features using {method} method")
        
        return selected_df
    
    def visualize_feature_importance(self, output_file: Optional[str] = None) -> None:
        """
        Visualize feature importance.
        
        Parameters:
        -----------
        output_file : str, optional
            Path to save the visualization
        """
        if not self.feature_importance:
            logger.warning("No feature importance data available. Run select_top_features first.")
            return
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top 20 features only for better visualization
        top_features = sorted_features[:20]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot feature importance
        features = [item[0] for item in top_features]
        importances = [item[1] for item in top_features]
        
        bars = plt.barh(features, importances)
        
        # Add values to bars
        for bar, val in zip(bars, importances):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center')
        
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save plot
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Saved feature importance visualization to {output_file}")
        else:
            plt.show()


def create_advanced_feature_set(prices_df: pd.DataFrame, 
                              spreads_df: pd.DataFrame,
                              volumes_df: Optional[pd.DataFrame] = None,
                              target: Optional[pd.Series] = None,
                              n_features: int = 20) -> pd.DataFrame:
    """
    Create an advanced feature set for intraday trading.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data
    spreads_df : pd.DataFrame
        DataFrame with spread data
    volumes_df : pd.DataFrame, optional
        DataFrame with volume data
    target : pd.Series, optional
        Target variable for feature selection
    n_features : int
        Number of features to select if target is provided
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with advanced features
    """
    # Initialize advanced feature engineering
    feature_eng = AdvancedFeatureEngineering()
    
    # Generate advanced features
    features = feature_eng.generate_advanced_features(
        prices_df=prices_df,
        spreads_df=spreads_df,
        volumes_df=volumes_df,
        scale_output=True
    )
    
    # Select top features if target is provided
    if target is not None:
        features = feature_eng.select_top_features(
            features=features,
            target=target,
            n_features=n_features,
            method='mutual_info'
        )
        
        # Save feature importance visualization
        feature_eng.visualize_feature_importance(
            output_file="output/feature_analysis/advanced/feature_importance.png"
        )
    
    return features 