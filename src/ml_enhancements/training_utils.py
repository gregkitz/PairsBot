"""
ML Training Utilities Module

This module provides utilities for training machine learning models for intraday trading,
including walk-forward validation and label generation functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.model_selection import TimeSeriesSplit
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Implements walk-forward validation for time series data.
    
    Walk-forward validation is a technique where the model is trained on a rolling window 
    of data and then tested on subsequent data points, simulating real-life trading.
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 train_size: int = None, 
                 test_size: int = None,
                 gap: int = 0,
                 purge_gap: int = 0):
        """
        Initialize the walk-forward validator.
        
        Parameters:
        -----------
        n_splits : int
            Number of train/test splits
        train_size : int
            Size of the training window
        test_size : int
            Size of the testing window
        gap : int
            Gap between training and testing sets
        purge_gap : int
            Number of samples to purge around train/test boundary
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.purge_gap = purge_gap
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # If train_size is not specified, calculate it based on n_splits
        if self.train_size is None:
            # Use a reasonable default: total size / (n_splits + 1) to ensure we have enough data
            self.train_size = n_samples // (self.n_splits + 1)
        
        # If test_size is not specified, use a reasonable default
        if self.test_size is None:
            self.test_size = min(n_samples // (self.n_splits * 2), 30)  # Default to 30 samples or less
        
        test_starts = range(self.train_size, n_samples, self.test_size)
        test_starts = list(test_starts)[:self.n_splits]  # Limit to n_splits
        
        splits = []
        for test_start in test_starts:
            # Determine train start (for expanding window, this would be 0)
            train_start = max(0, test_start - self.train_size)
            
            # Determine test end
            test_end = min(test_start + self.test_size, n_samples)
            
            # Apply gap between train and test if specified
            if self.gap > 0:
                effective_train_end = max(train_start, test_start - self.gap)
            else:
                effective_train_end = test_start
            
            # Apply purge_gap around train/test boundary if specified
            if self.purge_gap > 0:
                purge_start = max(train_start, test_start - self.purge_gap)
                purge_end = min(test_end, test_start + self.purge_gap)
                
                # Create train and test indices, excluding the purge range
                train_indices = indices[train_start:purge_start]
                test_indices = indices[test_start:test_end]
            else:
                train_indices = indices[train_start:effective_train_end]
                test_indices = indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def get_fold_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about each fold in the walk-forward validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with fold information
        """
        splits = self.split(X)
        
        fold_info = []
        for i, (train_idx, test_idx) in enumerate(splits):
            train_start = train_idx[0]
            train_end = train_idx[-1]
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            # Get start and end dates if index is DatetimeIndex
            if isinstance(X.index, pd.DatetimeIndex):
                train_start_date = X.index[train_start]
                train_end_date = X.index[train_end]
                test_start_date = X.index[test_start]
                test_end_date = X.index[test_end]
            else:
                train_start_date = train_start
                train_end_date = train_end
                test_start_date = test_start
                test_end_date = test_end
            
            fold_info.append({
                'fold': i,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start': train_start_date,
                'train_end': train_end_date,
                'test_start': test_start_date,
                'test_end': test_end_date
            })
        
        return pd.DataFrame(fold_info)
    
    def plot_folds(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                  output_file: Optional[str] = None) -> None:
        """
        Plot the walk-forward validation folds.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        output_file : str, optional
            Path to save the plot
        """
        splits = self.split(X)
        fold_info = self.get_fold_info(X)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # If we have a target variable and it's plottable, use it for visualization
        if y is not None and isinstance(y, (pd.Series, np.ndarray)):
            y_values = y if isinstance(y, np.ndarray) else y.values
            plt.plot(range(len(y_values)), y_values, '-', color='gray', alpha=0.5, label='Target')
        
        # Plot each fold
        colors = plt.cm.tab10.colors
        
        for i, (train_idx, test_idx) in enumerate(splits):
            color = colors[i % len(colors)]
            
            # Plot train indices
            plt.scatter(train_idx, [0.05] * len(train_idx), marker='|', color=color, alpha=0.5)
            
            # Plot test indices
            plt.scatter(test_idx, [0.1] * len(test_idx), marker='|', color=color, alpha=1.0)
            
            # Add fold label
            if isinstance(X.index, pd.DatetimeIndex):
                train_start = X.index[train_idx[0]].strftime('%Y-%m-%d')
                test_end = X.index[test_idx[-1]].strftime('%Y-%m-%d')
                label = f"Fold {i+1}: {train_start} to {test_end}"
            else:
                label = f"Fold {i+1}"
            
            plt.text(train_idx[0], 0.15, label, color=color, fontsize=10, ha='left')
        
        plt.yticks([])
        if isinstance(X.index, pd.DatetimeIndex):
            plt.xticks(range(0, len(X), len(X) // 8), 
                      [X.index[i].strftime('%Y-%m-%d') for i in range(0, len(X), len(X) // 8)],
                      rotation=45)
        
        plt.title('Walk-Forward Validation Folds')
        plt.xlabel('Time')
        plt.tight_layout()
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()


def generate_labels(prices_df: pd.DataFrame, 
                   spreads_df: pd.DataFrame,
                   lookforward: int = 5,
                   threshold: float = 0.001,
                   label_type: str = 'direction'
                  ) -> pd.Series:
    """
    Generate labels for supervised learning from price and spread data.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame with price data for each instrument in the pair
    spreads_df : pd.DataFrame
        DataFrame with spread data including z-scores
    lookforward : int
        Number of periods to look forward for label generation
    threshold : float
        Threshold for determining significant price/spread changes
    label_type : str
        Type of label to generate ('direction', 'return', 'zscore_reversal', 'price_reversal')
        
    Returns:
    --------
    pd.Series
        Series with generated labels
    """
    if label_type == 'direction':
        # Generate direction labels (1 for up, -1 for down, 0 for flat)
        if 'spread' in spreads_df.columns:
            # Use spread for direction
            future_change = spreads_df['spread'].shift(-lookforward) - spreads_df['spread']
            # Normalize by current spread
            future_change_pct = future_change / spreads_df['spread'].abs()
            
            # Generate labels based on threshold
            labels = pd.Series(0, index=spreads_df.index)
            labels[future_change_pct > threshold] = 1  # Up
            labels[future_change_pct < -threshold] = -1  # Down
        
        else:
            # No spread data, use first instrument price
            symbol = prices_df.columns[0]
            future_change_pct = prices_df[symbol].pct_change(lookforward).shift(-lookforward)
            
            # Generate labels based on threshold
            labels = pd.Series(0, index=prices_df.index)
            labels[future_change_pct > threshold] = 1  # Up
            labels[future_change_pct < -threshold] = -1  # Down
    
    elif label_type == 'return':
        # Generate continuous return labels
        if 'spread' in spreads_df.columns:
            # Use spread for return calculation
            future_change = spreads_df['spread'].shift(-lookforward) - spreads_df['spread']
            # Normalize by current spread
            labels = future_change / spreads_df['spread'].abs()
        else:
            # No spread data, use first instrument price
            symbol = prices_df.columns[0]
            labels = prices_df[symbol].pct_change(lookforward).shift(-lookforward)
    
    elif label_type == 'zscore_reversal':
        # Generate labels for z-score mean reversion
        if 'zscore' in spreads_df.columns:
            # Current z-score
            current_zscore = spreads_df['zscore']
            
            # Future z-score
            future_zscore = spreads_df['zscore'].shift(-lookforward)
            
            # Calculate whether z-score moved toward 0 (mean reversion)
            # Returns 1 if z-score reverted to mean, 0 otherwise
            labels = pd.Series(0, index=spreads_df.index)
            
            # If current z-score is positive, mean reversion is when future z-score is closer to 0
            labels[(current_zscore > 0) & (future_zscore < current_zscore)] = 1
            
            # If current z-score is negative, mean reversion is when future z-score is closer to 0
            labels[(current_zscore < 0) & (future_zscore > current_zscore)] = 1
        else:
            # Can't calculate without z-score
            logger.error("Need z-score data for zscore_reversal label type")
            return pd.Series()
    
    elif label_type == 'price_reversal':
        # Generate labels for price reversals
        if len(prices_df.columns) < 2:
            logger.error("Need at least two instruments for price_reversal label type")
            return pd.Series()
        
        # Get the two instruments
        symbol1, symbol2 = prices_df.columns[0], prices_df.columns[1]
        
        # Calculate future returns
        future_ret1 = prices_df[symbol1].pct_change(lookforward).shift(-lookforward)
        future_ret2 = prices_df[symbol2].pct_change(lookforward).shift(-lookforward)
        
        # Calculate current momentum
        current_mom1 = prices_df[symbol1].pct_change(lookforward)
        current_mom2 = prices_df[symbol2].pct_change(lookforward)
        
        # Identify reversals (when future return is opposite to current momentum)
        reversal1 = current_mom1 * future_ret1 < 0
        reversal2 = current_mom2 * future_ret2 < 0
        
        # Combined reversal - when either instrument shows reversal
        labels = pd.Series(0, index=prices_df.index)
        labels[reversal1 | reversal2] = 1
    
    else:
        logger.error(f"Unknown label type: {label_type}")
        return pd.Series()
    
    return labels


def create_train_test_labels(features: pd.DataFrame, 
                            prices_df: pd.DataFrame, 
                            spreads_df: pd.DataFrame,
                            lookforward_periods: List[int] = [5, 10, 20],
                            label_types: List[str] = ['direction', 'zscore_reversal'],
                            thresholds: Dict[str, float] = None) -> Dict[str, pd.Series]:
    """
    Create multiple label sets for different prediction horizons and label types.
    
    Parameters:
    -----------
    features : pd.DataFrame
        DataFrame with features
    prices_df : pd.DataFrame
        DataFrame with price data
    spreads_df : pd.DataFrame
        DataFrame with spread data
    lookforward_periods : List[int]
        List of lookforward periods to generate labels for
    label_types : List[str]
        List of label types to generate
    thresholds : Dict[str, float]
        Dictionary mapping label types to thresholds
        
    Returns:
    --------
    Dict[str, pd.Series]
        Dictionary mapping label names to label Series
    """
    if thresholds is None:
        thresholds = {
            'direction': 0.001,
            'return': 0.0,
            'zscore_reversal': 0.0,
            'price_reversal': 0.0
        }
    
    labels_dict = {}
    
    for period in lookforward_periods:
        for label_type in label_types:
            if label_type in thresholds:
                threshold = thresholds[label_type]
            else:
                threshold = 0.0
            
            # Generate labels
            labels = generate_labels(
                prices_df=prices_df,
                spreads_df=spreads_df,
                lookforward=period,
                threshold=threshold,
                label_type=label_type
            )
            
            # Skip if empty
            if labels.empty:
                continue
            
            # Store with descriptive name
            label_name = f"{label_type}_{period}"
            labels_dict[label_name] = labels
            
            # Align with features
            if not features.empty:
                common_index = features.index.intersection(labels.index)
                labels_dict[label_name] = labels.loc[common_index]
    
    return labels_dict


def create_sample_weights(labels: pd.Series, class_weights: Dict[int, float] = None) -> np.ndarray:
    """
    Create sample weights based on class frequency or custom weights.
    
    Parameters:
    -----------
    labels : pd.Series
        Series with class labels
    class_weights : Dict[int, float], optional
        Custom class weights
        
    Returns:
    --------
    np.ndarray
        Array with sample weights
    """
    if class_weights is None:
        # Calculate class weights inversely proportional to class frequency
        class_counts = labels.value_counts()
        total_samples = len(labels)
        
        class_weights = {}
        for class_val, count in class_counts.items():
            class_weights[class_val] = total_samples / (len(class_counts) * count)
    
    # Apply weights to samples
    sample_weights = np.ones(len(labels))
    
    for class_val, weight in class_weights.items():
        sample_weights[labels == class_val] = weight
    
    return sample_weights 