"""
Intraday Model Trainer Module

This module provides functionality for training and managing machine learning models
specifically for intraday trading signals.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml_enhancements.training_utils import WalkForwardValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradayModelTrainer:
    """
    Trainer for machine learning models focused on intraday trading signals.
    
    This class supports training various model types with appropriate validation
    for time series financial data.
    """
    
    def __init__(self, 
                 models_dir: str = "models/intraday",
                 random_seed: int = 42):
        """
        Initialize the intraday model trainer.
        
        Parameters:
        -----------
        models_dir : str
            Directory to save trained models
        random_seed : int
            Random seed for reproducibility
        """
        self.models_dir = models_dir
        self.random_seed = random_seed
        
        # Create directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize model containers
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        
        # Default configurations for different model types
        self.model_configs = {
            'signal_filter': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': random_seed
                }
            },
            'entry_timing': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': random_seed
                }
            },
            'exit_timing': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': random_seed
                }
            },
            'regime_detector': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': random_seed
                }
            }
        }
    
    def train_model(self, 
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   model_name: str,
                   model_type: str = 'signal_filter',
                   custom_model: Optional[Any] = None,
                   custom_params: Optional[Dict] = None,
                   use_walk_forward: bool = True,
                   n_splits: int = 5,
                   scale_features: bool = True) -> Tuple[Any, Dict]:
        """
        Train a machine learning model with appropriate validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series
            Target Series
        model_name : str
            Name to identify this model
        model_type : str
            Type of model to train ('signal_filter', 'entry_timing', 'exit_timing', 'regime_detector')
        custom_model : Any, optional
            Custom model class to use instead of defaults
        custom_params : Dict, optional
            Custom parameters for the model
        use_walk_forward : bool
            Whether to use walk-forward validation
        n_splits : int
            Number of splits for validation
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        Tuple[Any, Dict]
            Trained model and performance metrics
        """
        logger.info(f"Training {model_type} model: {model_name}")
        
        # Check for data issues
        if len(X) == 0 or len(y) == 0:
            logger.error("Empty training data")
            return None, {}
        
        if len(X) != len(y):
            logger.error(f"X and y have different lengths: {len(X)} vs {len(y)}")
            return None, {}
        
        # Create a pipeline with feature scaling if requested
        if scale_features:
            pipeline_steps = [
                ('scaler', StandardScaler()),
            ]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[model_name] = scaler
        else:
            X_scaled = X.values
        
        # Determine model and parameters
        if custom_model is not None:
            model_class = custom_model
            params = custom_params or {}
        else:
            model_config = self.model_configs.get(model_type)
            if model_config is None:
                logger.error(f"Unknown model type: {model_type}")
                return None, {}
            
            model_class = model_config['class']
            params = model_config['params']
        
        # Create model instance
        model = model_class(**params)
        
        # Validate using walk-forward if requested
        if use_walk_forward:
            validator = WalkForwardValidator(n_splits=n_splits)
            metrics = self._validate_with_walk_forward(model, X_scaled, y, validator)
        else:
            # Train on all data
            model.fit(X_scaled, y)
            
            # Calculate in-sample metrics
            y_pred = model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)
            metrics['validation_method'] = 'in_sample'
        
        # Retrain on all data for final model
        model.fit(X_scaled, y)
        
        # Save model and metrics
        self.models[model_name] = model
        self.metrics[model_name] = metrics
        
        # Save to disk
        self._save_model(model, model_name)
        
        logger.info(f"Model {model_name} trained successfully. Metrics: {metrics}")
        
        return model, metrics
    
    def _validate_with_walk_forward(self, 
                                  model: Any, 
                                  X: np.ndarray, 
                                  y: pd.Series,
                                  validator: WalkForwardValidator) -> Dict:
        """
        Validate a model using walk-forward validation.
        
        Parameters:
        -----------
        model : Any
            Model to validate
        X : np.ndarray
            Features array
        y : pd.Series
            Target Series
        validator : WalkForwardValidator
            Walk-forward validation instance
            
        Returns:
        --------
        Dict
            Validation metrics
        """
        # Get walk-forward splits
        splits = validator.split(pd.DataFrame(X))
        
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        
        # Validate on each fold
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model on this fold
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            fold_metric = self._calculate_metrics(y_test, y_pred)
            fold_metric['fold'] = fold
            
            fold_metrics.append(fold_metric)
            
            # Collect for overall metrics
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(all_y_true, all_y_pred)
        overall_metrics['fold_metrics'] = fold_metrics
        overall_metrics['validation_method'] = 'walk_forward'
        
        return overall_metrics
    
    def _calculate_metrics(self, y_true: Union[pd.Series, List], y_pred: Union[pd.Series, List]) -> Dict:
        """
        Calculate performance metrics for classification.
        
        Parameters:
        -----------
        y_true : Union[pd.Series, List]
            True labels
        y_pred : Union[pd.Series, List]
            Predicted labels
            
        Returns:
        --------
        Dict
            Performance metrics
        """
        metrics = {}
        
        # Convert to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Calculate classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        except Exception as e:
            logger.warning(f"Error calculating precision/recall metrics: {e}")
        
        # Calculate class distribution
        class_counts = pd.Series(y_true).value_counts().to_dict()
        metrics['class_distribution'] = class_counts
        
        # Calculate prediction distribution
        pred_counts = pd.Series(y_pred).value_counts().to_dict()
        metrics['prediction_distribution'] = pred_counts
        
        # Calculate timestamp
        metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return metrics
    
    def _save_model(self, model: Any, model_name: str) -> None:
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model : Any
            Trained model to save
        model_name : str
            Model name for the file
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        try:
            joblib.dump(model, model_path)
            
            # Save scaler if exists
            if model_name in self.scalers:
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
                joblib.dump(self.scalers[model_name], scaler_path)
            
            # Save metrics
            if model_name in self.metrics:
                metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.joblib")
                joblib.dump(self.metrics[model_name], metrics_path)
            
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
            
        Returns:
        --------
        Any
            Loaded model, or None if not found
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            
            # Try to load scaler
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
            
            # Try to load metrics
            metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.joblib")
            if os.path.exists(metrics_path):
                self.metrics[model_name] = joblib.load(metrics_path)
            
            logger.info(f"Model {model_name} loaded")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a trained model, loading it from disk if needed.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to get
            
        Returns:
        --------
        Any
            The requested model, or None if not found
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            return self.load_model(model_name)
    
    def predict(self, 
               X: pd.DataFrame, 
               model_name: str,
               return_proba: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using a trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
        model_name : str
            Name of the model to use
        return_proba : bool
            Whether to return class probabilities
            
        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Predictions, or (predictions, probabilities) if return_proba=True
        """
        model = self.get_model(model_name)
        
        if model is None:
            logger.error(f"Model {model_name} not found")
            return None
        
        # Apply scaler if available
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        if return_proba and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_scaled)
            return y_pred, y_proba
        else:
            return y_pred 