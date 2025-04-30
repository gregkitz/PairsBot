"""
Time Series Data Augmentation Module

This module provides data augmentation techniques for time series financial data
to improve model generalization and robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging
from scipy import signal
from scipy.interpolate import CubicSpline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesAugmentation:
    """
    Time series data augmentation techniques for financial data.
    """
    
    def __init__(self, 
                 preserve_features: List[str] = None, 
                 random_seed: int = 42):
        """
        Initialize the time series augmentation.
        
        Parameters:
        -----------
        preserve_features : List[str], optional
            List of feature names that should not be augmented (e.g., categorical features)
        random_seed : int
            Random seed for reproducibility
        """
        self.preserve_features = preserve_features or []
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def jitter(self, 
              X: pd.DataFrame, 
              y: Optional[pd.Series] = None, 
              std: float = 0.01) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Add random noise to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        std : float
            Standard deviation of the noise relative to the feature's std
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = X.copy()
        
        # Only augment numeric features that are not in preserve_features
        for col in X.columns:
            if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                noise = np.random.normal(0, std * X[col].std(), size=len(X))
                X_aug[col] = X[col] + noise
        
        return X_aug, y
    
    def scaling(self, 
               X: pd.DataFrame, 
               y: Optional[pd.Series] = None, 
               scaling_factor: float = 1.1) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Scale the data by a random factor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        scaling_factor : float
            Maximum scaling factor
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = X.copy()
        
        # Generate random scaling factor between 1/factor and factor
        factor = np.random.uniform(1/scaling_factor, scaling_factor)
        
        # Only scale numeric features that are not in preserve_features
        for col in X.columns:
            if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                X_aug[col] = X[col] * factor
        
        return X_aug, y
    
    def magnitude_warp(self, 
                      X: pd.DataFrame, 
                      y: Optional[pd.Series] = None, 
                      sigma: float = 0.2, 
                      knot_count: int = 4) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply magnitude warping to the data (smooth, random changes in magnitude).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        sigma : float
            Standard deviation of the random warping
        knot_count : int
            Number of knots for the spline (more knots = more complex warping)
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = X.copy()
        
        # Only warp numeric features that are not in preserve_features
        for col in X.columns:
            if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                # Create random knots for cubic spline
                knots = np.linspace(0, len(X) - 1, knot_count)
                warps = np.random.normal(1.0, sigma, size=knot_count)
                
                # Interpolate warps to the data length
                spline = CubicSpline(knots, warps)
                warp_factors = spline(np.arange(len(X)))
                
                # Apply warping
                X_aug[col] = X[col] * warp_factors
        
        return X_aug, y
    
    def time_warp(self, 
                 X: pd.DataFrame, 
                 y: Optional[pd.Series] = None, 
                 sigma: float = 0.2, 
                 knot_count: int = 4) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply time warping to the data (stretching and compressing time).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        sigma : float
            Standard deviation of the random warping
        knot_count : int
            Number of knots for the spline (more knots = more complex warping)
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = pd.DataFrame(index=X.index)
        
        # Create random knots for cubic spline
        knots = np.linspace(0, len(X) - 1, knot_count)
        warps = np.random.normal(0.0, sigma, size=knot_count)
        
        # Ensure endpoint has no warp
        warps[0] = 0.0
        warps[-1] = 0.0
        
        # Interpolate warps to the data length
        spline = CubicSpline(knots, warps)
        warp_steps = spline(np.arange(len(X)))
        
        # Calculate new indices with warping
        time_indices = np.arange(len(X))
        new_indices = time_indices + warp_steps
        
        # Only warp numeric features that are not in preserve_features
        for col in X.columns:
            if col in self.preserve_features or not np.issubdtype(X[col].dtype, np.number):
                # Don't warp preserved features
                X_aug[col] = X[col].values
            else:
                # Interpolate original values onto warped indices
                interpolated = np.interp(
                    new_indices, 
                    time_indices, 
                    X[col].values,
                    left=X[col].values[0],
                    right=X[col].values[-1]
                )
                X_aug[col] = interpolated
        
        return X_aug, y
    
    def window_slice(self, 
                    X: pd.DataFrame, 
                    y: Optional[pd.Series] = None, 
                    window_ratio: float = 0.9) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Extract a random slice of the time series.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        window_ratio : float
            Ratio of the window size to the original size
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        window_size = int(len(X) * window_ratio)
        
        # Ensure window_size is at least 10 samples
        window_size = max(window_size, min(10, len(X)))
        
        # Choose random start point
        start_idx = np.random.randint(0, len(X) - window_size + 1)
        end_idx = start_idx + window_size
        
        # Extract slice
        X_aug = X.iloc[start_idx:end_idx].copy()
        
        # If target is provided, extract corresponding slice
        if y is not None:
            y_aug = y.iloc[start_idx:end_idx].copy()
        else:
            y_aug = None
        
        return X_aug, y_aug
    
    def window_warp(self, 
                   X: pd.DataFrame, 
                   y: Optional[pd.Series] = None, 
                   window_ratio: float = 0.1,
                   scales: List[float] = [0.5, 2.0]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Stretch or compress a random window of the time series.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        window_ratio : float
            Ratio of the window size to the original size
        scales : List[float]
            List of scaling factors to apply randomly
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = X.copy()
        
        window_size = int(len(X) * window_ratio)
        
        # Ensure window_size is at least 10 samples
        window_size = max(window_size, min(10, len(X)))
        
        # Choose random start point
        start_idx = np.random.randint(0, len(X) - window_size + 1)
        end_idx = start_idx + window_size
        
        # Choose random scaling factor
        scale = np.random.choice(scales)
        
        # Only warp numeric features that are not in preserve_features
        for col in X.columns:
            if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                # Extract window
                window = X[col].iloc[start_idx:end_idx].values
                
                # Apply scaling
                if scale != 1.0:
                    # Resample window
                    original_indices = np.arange(len(window))
                    new_size = int(len(window) * scale)
                    
                    if new_size <= 1:
                        continue
                    
                    # Interpolate to new size
                    new_indices = np.linspace(0, len(window) - 1, new_size)
                    warped_window = np.interp(new_indices, original_indices, window)
                    
                    # Replace original window
                    # If warped window is larger, truncate to original size
                    # If smaller, pad with boundary values
                    if len(warped_window) >= len(window):
                        X_aug[col].iloc[start_idx:end_idx] = warped_window[:len(window)]
                    else:
                        X_aug[col].iloc[start_idx:start_idx + len(warped_window)] = warped_window
                        X_aug[col].iloc[start_idx + len(warped_window):end_idx] = warped_window[-1]
        
        return X_aug, y
    
    def augment_dataset(self, 
                       X: pd.DataFrame, 
                       y: Optional[pd.Series] = None, 
                       methods: List[str] = None,
                       n_augmentations: int = 1) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply multiple augmentation techniques to create an augmented dataset.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        methods : List[str], optional
            List of augmentation methods to apply. If None, applies all methods.
            Available methods: ['jitter', 'scaling', 'magnitude_warp', 'time_warp', 'window_slice', 'window_warp']
        n_augmentations : int
            Number of augmented copies to create
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        available_methods = {
            'jitter': self.jitter,
            'scaling': self.scaling,
            'magnitude_warp': self.magnitude_warp,
            'time_warp': self.time_warp,
            'window_slice': self.window_slice,
            'window_warp': self.window_warp
        }
        
        # If methods not specified, use all available methods
        if methods is None:
            methods = list(available_methods.keys())
        
        # Validate methods
        for method in methods:
            if method not in available_methods:
                logger.warning(f"Unknown augmentation method: {method}. Skipping.")
                methods.remove(method)
        
        if not methods:
            logger.error("No valid augmentation methods specified.")
            return X, y
        
        # Create augmented datasets
        X_augmented_list = [X]
        y_augmented_list = [y] if y is not None else [None]
        
        for i in range(n_augmentations):
            # Choose random method for this augmentation
            method = np.random.choice(methods)
            
            # Apply augmentation
            X_aug, y_aug = available_methods[method](X, y)
            
            # Add to list
            X_augmented_list.append(X_aug)
            y_augmented_list.append(y_aug)
        
        # Combine augmented datasets
        X_combined = pd.concat(X_augmented_list)
        
        if y is not None:
            y_combined = pd.concat(y_augmented_list)
        else:
            y_combined = None
        
        return X_combined, y_combined
    
    def financial_specific_augmentation(self, 
                                       X: pd.DataFrame, 
                                       y: Optional[pd.Series] = None,
                                       vol_scaling: bool = True,
                                       trend_reset: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply domain-specific augmentations for financial time series.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series, optional
            Target Series
        vol_scaling : bool
            Whether to apply volatility-based scaling
        trend_reset : bool
            Whether to apply trend resetting (centering)
            
        Returns:
        --------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            Augmented features and target
        """
        X_aug = X.copy()
        
        # Features to consider volatility-specific
        vol_features = [col for col in X.columns if 'vol' in col.lower() or 'std' in col.lower()]
        
        # Features to consider price or level-specific
        level_features = [col for col in X.columns if 
                         ('price' in col.lower() or 
                          'spread' in col.lower() or 
                          'zscore' in col.lower()) and
                         col not in vol_features]
        
        # Apply volatility scaling
        if vol_scaling and vol_features:
            # Scale volatility features by a random factor
            vol_scale = np.random.uniform(0.5, 2.0)
            
            for col in vol_features:
                if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                    X_aug[col] = X[col] * vol_scale
        
        # Apply trend resetting (centering around mean)
        if trend_reset and level_features:
            for col in level_features:
                if col not in self.preserve_features and np.issubdtype(X[col].dtype, np.number):
                    # For z-scores, center around 0
                    if 'zscore' in col.lower():
                        X_aug[col] = X[col] - X[col].mean()
                    # For prices, keep the mean but scale the amplitude
                    elif 'price' in col.lower():
                        mean = X[col].mean()
                        X_aug[col] = mean + (X[col] - mean) * np.random.uniform(0.8, 1.2)
        
        return X_aug, y
    
    def balanced_augmentation(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             target_class_ratio: Dict[Any, float] = None,
                             methods: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate augmented samples to balance class distribution.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series
            Target Series with class labels
        target_class_ratio : Dict[Any, float], optional
            Target ratio for each class. If None, balances all classes equally.
        methods : List[str], optional
            List of augmentation methods to apply.
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Balanced features and target
        """
        if y is None:
            logger.error("Target required for balanced augmentation.")
            return X, y
        
        # Get class distribution
        class_counts = y.value_counts()
        
        # If target_class_ratio not provided, balance equally
        if target_class_ratio is None:
            max_count = class_counts.max()
            target_class_ratio = {cls: max_count for cls in class_counts.index}
        
        # Calculate how many samples to generate for each class
        augment_counts = {}
        for cls, count in class_counts.items():
            target_count = target_class_ratio.get(cls, count)
            if target_count > count:
                augment_counts[cls] = target_count - count
        
        if not augment_counts:
            logger.info("No augmentation needed for balancing.")
            return X, y
        
        # Generate augmented samples for each class
        X_augmented_list = [X]
        y_augmented_list = [y]
        
        for cls, count in augment_counts.items():
            # Get samples of this class
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            
            # Skip if no samples
            if len(X_cls) == 0:
                continue
            
            # Calculate required augmentation factor
            aug_factor = int(np.ceil(count / len(X_cls)))
            
            # Generate augmented samples
            for _ in range(aug_factor):
                X_aug, y_aug = self.augment_dataset(
                    X_cls, y_cls, 
                    methods=methods,
                    n_augmentations=1
                )
                
                # Add to list
                X_augmented_list.append(X_aug)
                y_augmented_list.append(y_aug)
        
        # Combine all samples
        X_combined = pd.concat(X_augmented_list)
        y_combined = pd.concat(y_augmented_list)
        
        # Re-sample to get the target class distribution
        final_X = []
        final_y = []
        
        for cls, target_count in target_class_ratio.items():
            # Get all samples of this class
            cls_mask = y_combined == cls
            X_cls = X_combined[cls_mask]
            y_cls = y_combined[cls_mask]
            
            if len(X_cls) <= target_count:
                # If we still don't have enough, use all samples
                final_X.append(X_cls)
                final_y.append(y_cls)
            else:
                # Randomly sample to get the target count
                indices = np.random.choice(len(X_cls), size=target_count, replace=False)
                final_X.append(X_cls.iloc[indices])
                final_y.append(y_cls.iloc[indices])
        
        # Combine final samples
        X_balanced = pd.concat(final_X)
        y_balanced = pd.concat(final_y)
        
        return X_balanced, y_balanced 