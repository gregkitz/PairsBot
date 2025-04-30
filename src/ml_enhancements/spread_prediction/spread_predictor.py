"""
Spread Predictor for Pairs Trading.

This module provides a SpreadPredictor class for predicting future spread movements
using various machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import joblib
import os
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from src.utils.technical_indicators import rsi, macd, z_score, bollinger_bands, rate_of_change
from src.utils.error_handling import (
    BaseError, ModelError, ModelPredictionError, DataValidationError, log_exception
)

from .model_factory import create_model, available_models


class SpreadPredictor:
    """
    Spread Predictor for Pairs Trading.
    
    This class provides methods for training machine learning models to predict
    future spread movements and generating trading signals based on those predictions.
    """
    
    def __init__(self, 
                model_type: str = 'gradient_boosting',
                forecast_horizon: int = 5,
                feature_lookback: int = 20,
                test_size: float = 0.2,
                scale_features: bool = True,
                model_params: Optional[Dict[str, Any]] = None,
                random_state: int = 42):
        """
        Initialize the spread predictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use for prediction
        forecast_horizon : int
            Number of periods ahead to forecast
        feature_lookback : int
            Number of periods to look back for feature generation
        test_size : float
            Proportion of data to use for testing
        scale_features : bool
            Whether to scale features
        model_params : dict, optional
            Additional parameters for the model
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.feature_lookback = feature_lookback
        self.test_size = test_size
        self.scale_features = scale_features
        self.model_params = model_params if model_params is not None else {}
        self.random_state = random_state
        
        # Create model
        self.model = create_model(
            model_type=model_type,
            scale_features=scale_features,
            random_state=random_state,
            **self.model_params
        )
        
        # Store results
        self.train_metrics = {}
        self.test_metrics = {}
        self.feature_importance = None
        self.prediction_history = None
    
    def _create_features(self, spread_data: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features for the spread prediction model.
        
        Parameters:
        -----------
        spread_data : pd.Series
            Spread time series
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            X: Feature DataFrame, y: Target Series
        """
        # Create target variable (future spread)
        df = pd.DataFrame({'spread': spread_data})
        df['target'] = df['spread'].shift(-self.forecast_horizon)
        
        # Create lagged features
        for lag in range(1, self.feature_lookback + 1):
            df[f'lag_{lag}'] = df['spread'].shift(lag)
        
        # Calculate rolling statistics
        df['rolling_mean_5'] = df['spread'].rolling(window=5).mean()
        df['rolling_mean_10'] = df['spread'].rolling(window=10).mean()
        df['rolling_std_5'] = df['spread'].rolling(window=5).std()
        df['rolling_std_10'] = df['spread'].rolling(window=10).std()
        
        # Add z-score
        df['zscore_20'] = z_score(df['spread'], window=20)
        
        # Calculate return features
        df['return_1'] = df['spread'].pct_change(1)
        df['return_5'] = df['spread'].pct_change(5)
        
        # Add rate of change
        df['roc_5'] = rate_of_change(df['spread'], window=5)
        df['roc_10'] = rate_of_change(df['spread'], window=10)
        
        # Bollinger Bands
        mid, upper, lower = bollinger_bands(df['spread'])
        df['bb_middle'] = mid
        df['bb_upper'] = upper
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / mid
        
        # RSI
        df['rsi_14'] = rsi(df['spread'])
        
        # MACD
        macd_line, macd_signal, macd_hist = macd(df['spread'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(['spread', 'target'], axis=1)
        y = df['target']
        
        return X, y
    
    def train(self, spread_data: pd.Series, use_time_series_cv: bool = True, n_splits: int = 5) -> Dict[str, float]:
        """
        Train the model on the spread data.
        
        Parameters:
        -----------
        spread_data : pandas.Series
            Spread data time series
        use_time_series_cv : bool
            Whether to use time series cross-validation
        n_splits : int
            Number of splits for time series cross-validation
            
        Returns:
        --------
        dict
            Dictionary with training and test metrics
        """
        # Create features and target
        X, y = self._create_features(spread_data)
        
        if use_time_series_cv:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            train_errors = []
            test_errors = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = self.model.predict(X_train)
                test_pred = self.model.predict(X_test)
                
                train_errors.append(mean_squared_error(y_train, train_pred))
                test_errors.append(mean_squared_error(y_test, test_pred))
            
            # Final training on all data
            self.model.fit(X, y)
            
            # Store metrics
            self.train_metrics = {
                'mse': np.mean(train_errors),
                'rmse': np.sqrt(np.mean(train_errors)),
                'mae': mean_absolute_error(y, self.model.predict(X))
            }
            
            self.test_metrics = {
                'mse': np.mean(test_errors),
                'rmse': np.sqrt(np.mean(test_errors)),
                'mae': mean_absolute_error(y.iloc[test_idx], test_pred)
            }
        else:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False, random_state=self.random_state
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            # Store metrics
            self.train_metrics = {
                'mse': mean_squared_error(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'mae': mean_absolute_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred)
            }
            
            self.test_metrics = {
                'mse': mean_squared_error(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }
            
            # Store predictions for visualization
            pred_index = pd.concat([X_train.index, X_test.index])
            self.prediction_history = pd.DataFrame({
                'actual': pd.concat([y_train, y_test]),
                'predicted': np.concatenate([train_pred, test_pred]),
                'is_train': [True] * len(y_train) + [False] * len(y_test)
            }, index=pred_index)
        
        # Try to get feature importance if available
        try:
            feature_importance = None
            # For tree-based models
            if hasattr(self.model['model'], 'feature_importances_'):
                feature_importance = pd.Series(
                    self.model['model'].feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            # For linear models
            elif hasattr(self.model['model'], 'coef_'):
                if len(self.model['model'].coef_.shape) == 1:
                    feature_importance = pd.Series(
                        np.abs(self.model['model'].coef_),
                        index=X.columns
                    ).sort_values(ascending=False)
            
            self.feature_importance = feature_importance
        except:
            self.feature_importance = None
        
        # Return the combined metrics
        return {**self.train_metrics, **{f'test_{k}': v for k, v in self.test_metrics.items()}}
    
    def predict(self, spread_data: pd.Series) -> float:
        """
        Make a prediction of the future spread value.
        
        Parameters:
        -----------
        spread_data : pandas.Series
            Recent spread data (must be at least feature_lookback periods)
            
        Returns:
        --------
        float
            Predicted future spread value
            
        Raises:
        -------
        DataValidationError
            If the spread data is insufficient or invalid
        ModelPredictionError
            If the model fails to make a prediction
        """
        try:
            # Validate input data
            if spread_data is None or len(spread_data) == 0:
                raise DataValidationError(
                    message="Spread data cannot be empty",
                    error_code="EMPTY_DATA",
                    source="SpreadPredictor.predict"
                )
                
            if len(spread_data) < self.feature_lookback:
                raise DataValidationError(
                    message=f"Spread data must contain at least {self.feature_lookback} periods",
                    error_code="INSUFFICIENT_DATA",
                    source="SpreadPredictor.predict",
                    details={
                        "required_length": self.feature_lookback,
                        "actual_length": len(spread_data)
                    }
                )
            
            # Check if model is trained
            if self.model is None:
                raise ModelPredictionError(
                    message="Model is not trained",
                    error_code="MODEL_NOT_TRAINED",
                    source="SpreadPredictor.predict"
                )
            
            # Create features from the recent data
            X, _ = self._create_features(spread_data)
            
            # Use the last available row of features
            X_last = X.iloc[[-1]]
            
            # Make prediction
            try:
                prediction = self.model.predict(X_last)[0]
                return prediction
            except Exception as e:
                raise ModelPredictionError(
                    message="Model prediction failed",
                    error_code="PREDICTION_FAILED",
                    source="SpreadPredictor.predict",
                    cause=e,
                    details={"features": X_last.to_dict(orient='records')[0]}
                )
                
        except (DataValidationError, ModelPredictionError) as e:
            # These are our custom errors, just log and re-raise
            log_exception(e, logger)
            raise
        except Exception as e:
            # Wrap any other exceptions
            error = ModelPredictionError(
                message=f"Unexpected error in prediction: {str(e)}",
                error_code="PREDICTION_ERROR",
                source="SpreadPredictor.predict",
                cause=e
            )
            log_exception(error, logger)
            raise error
    
    def generate_signal(self, 
                      current_spread: float, 
                      predicted_spread: float,
                      z_score: float,
                      entry_threshold: float = 2.0,
                      exit_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate a trading signal based on the predicted spread movement.
        
        Parameters:
        -----------
        current_spread : float
            Current spread value
        predicted_spread : float
            Predicted future spread value
        z_score : float
            Current z-score of the spread
        entry_threshold : float
            Z-score threshold for entry signals
        exit_threshold : float
            Z-score threshold for exit signals
            
        Returns:
        --------
        dict
            Dictionary with signal information
        """
        # Calculate predicted direction and magnitude
        predicted_direction = 1 if predicted_spread > current_spread else -1
        predicted_change = predicted_spread - current_spread
        predicted_change_pct = predicted_change / current_spread if current_spread != 0 else 0
        
        # Determine signal based on traditional z-score and prediction
        signal = 0  # No position
        confidence = abs(predicted_change_pct) * 100  # Confidence as percentage change
        
        # Case 1: Z-score indicates mean reversion opportunity and prediction agrees
        if z_score > entry_threshold and predicted_direction < 0:
            # Short signal (sell spread): high z-score and predicted to decrease
            signal = -1
        elif z_score < -entry_threshold and predicted_direction > 0:
            # Long signal (buy spread): low z-score and predicted to increase
            signal = 1
        
        # Case 2: Exit signals
        elif abs(z_score) < exit_threshold:
            # Exit signal (close position)
            signal = 0
        
        # Case 3: ML predicts strong move against traditional signal
        # This is where ML adds value by potentially avoiding false reversal signals
        elif abs(predicted_change_pct) > 0.02:  # If predicted change > 2%
            if z_score > entry_threshold and predicted_direction > 0:
                # Hold off on short, prediction says spread will continue up
                signal = 0
            elif z_score < -entry_threshold and predicted_direction < 0:
                # Hold off on long, prediction says spread will continue down
                signal = 0
        
        return {
            'signal': signal,  # 1: long, -1: short, 0: no position
            'confidence': confidence,
            'current_spread': current_spread,
            'predicted_spread': predicted_spread,
            'predicted_change': predicted_change,
            'predicted_change_pct': predicted_change_pct,
            'z_score': z_score
        }
    
    def plot_predictions(self, save_path=None):
        """
        Plot the actual vs. predicted spread values.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.prediction_history is None:
            print("No prediction history available. Train the model first.")
            return
        
        plt.figure(figsize=(12, 6))
        
        train_data = self.prediction_history[self.prediction_history['is_train']]
        test_data = self.prediction_history[~self.prediction_history['is_train']]
        
        plt.plot(train_data.index, train_data['actual'], 'b-', label='Actual (Train)', alpha=0.6)
        plt.plot(train_data.index, train_data['predicted'], 'r--', label='Predicted (Train)', alpha=0.6)
        
        plt.plot(test_data.index, test_data['actual'], 'g-', label='Actual (Test)', linewidth=2)
        plt.plot(test_data.index, test_data['predicted'], 'm--', label='Predicted (Test)', linewidth=2)
        
        plt.axvline(x=test_data.index[0], color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
        
        plt.title('Spread Prediction: Actual vs. Predicted')
        plt.ylabel('Spread Value')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the plot
        """
        if self.feature_importance is None:
            print("Feature importance not available for this model.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot top N features
        self.feature_importance.head(top_n).plot(kind='barh')
        
        plt.title(f'Top {top_n} Important Features')
        plt.ylabel('Feature')
        plt.xlabel('Importance')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, file_path: str):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the model
        """
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon,
            'feature_lookback': self.feature_lookback,
            'scale_features': self.scale_features,
            'model_params': self.model_params,
            'random_state': self.random_state,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'SpreadPredictor':
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved model
            
        Returns:
        --------
        SpreadPredictor
            Loaded spread predictor
        """
        model_data = joblib.load(file_path)
        
        predictor = cls(
            model_type=model_data['model_type'],
            forecast_horizon=model_data['forecast_horizon'],
            feature_lookback=model_data['feature_lookback'],
            scale_features=model_data['scale_features'],
            model_params=model_data['model_params'],
            random_state=model_data['random_state']
        )
        
        predictor.model = model_data['model']
        predictor.train_metrics = model_data['train_metrics']
        predictor.test_metrics = model_data['test_metrics']
        predictor.feature_importance = model_data['feature_importance']
        
        return predictor 