"""
Ensemble Models Module

This module provides ensemble learning techniques to improve model performance
through combining multiple models using stacking, boosting, and bagging methods.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier, 
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from src.ml_enhancements.training_utils import WalkForwardValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModelFactory:
    """
    Factory class for creating and managing ensemble models.
    
    This class provides methods to create various types of ensemble models
    for classification and regression tasks in intraday trading.
    """
    
    def __init__(self, 
                 models_dir: str = "models/intraday/ensemble",
                 random_seed: int = 42):
        """
        Initialize the ensemble model factory.
        
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
        
        # Initialize containers
        self.ensemble_models = {}
        self.base_models = {}
        self.scalers = {}
        self.metrics = {}
    
    def create_voting_ensemble(self, 
                              model_name: str,
                              base_models: List[Tuple[str, Any]],
                              voting: str = 'soft') -> VotingClassifier:
        """
        Create a voting ensemble classifier.
        
        Parameters:
        -----------
        model_name : str
            Name for the ensemble model
        base_models : List[Tuple[str, Any]]
            List of (name, model) tuples for the base models
        voting : str
            Voting strategy ('hard' or 'soft')
            
        Returns:
        --------
        VotingClassifier
            Voting ensemble classifier
        """
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=base_models,
            voting=voting
        )
        
        # Store model
        self.ensemble_models[model_name] = ensemble
        
        return ensemble
    
    def create_stacking_ensemble(self, 
                                model_name: str,
                                base_models: List[Tuple[str, Any]],
                                final_estimator: Any = None,
                                cv: int = 5) -> StackingClassifier:
        """
        Create a stacking ensemble classifier.
        
        Parameters:
        -----------
        model_name : str
            Name for the ensemble model
        base_models : List[Tuple[str, Any]]
            List of (name, model) tuples for the base models
        final_estimator : Any, optional
            Final estimator for stacking
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        StackingClassifier
            Stacking ensemble classifier
        """
        # Default final estimator is LogisticRegression
        if final_estimator is None:
            final_estimator = LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_seed
            )
        
        # Create stacking classifier
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=cv,
            stack_method='predict_proba'
        )
        
        # Store model
        self.ensemble_models[model_name] = ensemble
        
        return ensemble
    
    def create_boosting_ensemble(self, 
                                model_name: str,
                                base_estimator: Any = None,
                                n_estimators: int = 100,
                                learning_rate: float = 0.1) -> AdaBoostClassifier:
        """
        Create a boosting ensemble classifier.
        
        Parameters:
        -----------
        model_name : str
            Name for the ensemble model
        base_estimator : Any, optional
            Base estimator for boosting
        n_estimators : int
            Number of estimators
        learning_rate : float
            Learning rate for boosting
            
        Returns:
        --------
        AdaBoostClassifier
            Boosting ensemble classifier
        """
        # Create boosting classifier
        ensemble = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=self.random_seed
        )
        
        # Store model
        self.ensemble_models[model_name] = ensemble
        
        return ensemble
    
    def create_bagging_ensemble(self, 
                               model_name: str,
                               base_estimator: Any = None,
                               n_estimators: int = 100,
                               max_samples: float = 0.8,
                               max_features: float = 0.8) -> BaggingClassifier:
        """
        Create a bagging ensemble classifier.
        
        Parameters:
        -----------
        model_name : str
            Name for the ensemble model
        base_estimator : Any, optional
            Base estimator for bagging
        n_estimators : int
            Number of estimators
        max_samples : float
            Maximum fraction of samples for each estimator
        max_features : float
            Maximum fraction of features for each estimator
            
        Returns:
        --------
        BaggingClassifier
            Bagging ensemble classifier
        """
        # Default base estimator
        if base_estimator is None:
            base_estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                random_state=self.random_seed
            )
        
        # Create bagging classifier
        ensemble = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=self.random_seed
        )
        
        # Store model
        self.ensemble_models[model_name] = ensemble
        
        return ensemble
    
    def create_diversified_ensemble(self, model_name: str) -> VotingClassifier:
        """
        Create a diversified ensemble with multiple algorithm types.
        
        Parameters:
        -----------
        model_name : str
            Name for the ensemble model
            
        Returns:
        --------
        VotingClassifier
            Voting ensemble with diversified base models
        """
        # Create diverse base models
        base_models = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_seed
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_seed
            )),
            ('svm', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=self.random_seed
            )),
            ('lr', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_seed
            ))
        ]
        
        # Create voting ensemble
        ensemble = self.create_voting_ensemble(
            model_name=model_name,
            base_models=base_models,
            voting='soft'
        )
        
        return ensemble
    
    def train_ensemble(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      model_name: str,
                      use_walk_forward: bool = True,
                      n_splits: int = 5,
                      scale_features: bool = True) -> Tuple[Any, Dict]:
        """
        Train an ensemble model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features DataFrame
        y : pd.Series
            Target Series
        model_name : str
            Name of the ensemble model
        use_walk_forward : bool
            Whether to use walk-forward validation
        n_splits : int
            Number of walk-forward splits
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        Tuple[Any, Dict]
            Trained model and metrics
        """
        if model_name not in self.ensemble_models:
            logger.error(f"Ensemble model '{model_name}' not found")
            return None, {}
        
        ensemble = self.ensemble_models[model_name]
        
        # Create a copy to avoid modifying the original
        X_train = X.copy()
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            self.scalers[model_name] = scaler
        else:
            X_scaled = X_train.values
        
        # Validate using walk-forward if requested
        if use_walk_forward:
            validator = WalkForwardValidator(n_splits=n_splits)
            metrics = self._validate_with_walk_forward(ensemble, X_scaled, y, validator)
        else:
            # Train on all data
            ensemble.fit(X_scaled, y)
            
            # Calculate in-sample metrics
            y_pred = ensemble.predict(X_scaled)
            
            # Calculate metrics based on problem type
            if len(np.unique(y)) <= 5:  # Classification
                metrics = self._calculate_classification_metrics(y, y_pred)
            else:  # Regression
                metrics = self._calculate_regression_metrics(y, y_pred)
            
            metrics['validation_method'] = 'in_sample'
        
        # Retrain on all data for final model
        ensemble.fit(X_scaled, y)
        
        # Store metrics
        self.metrics[model_name] = metrics
        
        # Save model
        self._save_model(ensemble, model_name, scaler=self.scalers.get(model_name))
        
        logger.info(f"Ensemble model {model_name} trained successfully. Metrics: {metrics}")
        
        return ensemble, metrics
    
    def _validate_with_walk_forward(self, 
                                   model: Any, 
                                   X: np.ndarray, 
                                   y: pd.Series,
                                   validator: WalkForwardValidator) -> Dict:
        """
        Validate model using walk-forward validation.
        
        Parameters:
        -----------
        model : Any
            Model to validate
        X : np.ndarray
            Features array
        y : pd.Series
            Target Series
        validator : WalkForwardValidator
            Walk-forward validator
            
        Returns:
        --------
        Dict
            Validation metrics
        """
        # Get splits from validator
        splits = validator.split(X)
        
        # Track predictions and metrics for each fold
        all_y_true = []
        all_y_pred = []
        fold_metrics = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store predictions
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            # Calculate fold metrics
            if len(np.unique(y)) <= 5:  # Classification
                fold_metric = self._calculate_classification_metrics(y_test, y_pred)
            else:  # Regression
                fold_metric = self._calculate_regression_metrics(y_test, y_pred)
            
            fold_metric['fold'] = i
            fold_metrics.append(fold_metric)
        
        # Calculate overall metrics
        if len(np.unique(y)) <= 5:  # Classification
            overall_metrics = self._calculate_classification_metrics(all_y_true, all_y_pred)
        else:  # Regression
            overall_metrics = self._calculate_regression_metrics(all_y_true, all_y_pred)
        
        # Add fold metrics
        overall_metrics['folds'] = fold_metrics
        overall_metrics['validation_method'] = 'walk_forward'
        
        return overall_metrics
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: List) -> Dict:
        """
        Calculate classification metrics.
        
        Parameters:
        -----------
        y_true : pd.Series
            True labels
        y_pred : List
            Predicted labels
            
        Returns:
        --------
        Dict
            Classification metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if binary classification
        if len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Couldn't calculate AUC: {e}")
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: List) -> Dict:
        """
        Calculate regression metrics.
        
        Parameters:
        -----------
        y_true : pd.Series
            True values
        y_pred : List
            Predicted values
            
        Returns:
        --------
        Dict
            Regression metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
        
        return metrics
    
    def _save_model(self, model: Any, model_name: str, scaler: Any = None) -> None:
        """
        Save model and scaler to disk.
        
        Parameters:
        -----------
        model : Any
            Model to save
        model_name : str
            Name of the model
        scaler : Any, optional
            Feature scaler
        """
        # Create paths
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        Any or None
            Loaded model or None if not found
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return None
        
        try:
            model = joblib.load(model_path)
            
            # Load scaler if exists
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
            
            # Store model
            self.ensemble_models[model_name] = model
            
            logger.info(f"Loaded model {model_name}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def visualize_model_comparison(self, 
                                   metrics_list: List[Dict],
                                   model_names: List[str],
                                   output_file: Optional[str] = None) -> None:
        """
        Visualize model comparison with metrics.
        
        Parameters:
        -----------
        metrics_list : List[Dict]
            List of metrics dictionaries
        model_names : List[str]
            List of model names
        output_file : str, optional
            Path to save the visualization
        """
        # Extract common metrics across all models
        common_metrics = set.intersection(*[set(m.keys()) for m in metrics_list])
        common_metrics = [m for m in common_metrics if isinstance(metrics_list[0][m], (int, float))]
        
        if not common_metrics:
            logger.error("No common metrics found for visualization")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(common_metrics), 1, figsize=(10, 3 * len(common_metrics)))
        
        if len(common_metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(common_metrics):
            values = [m[metric] for m in metrics_list]
            
            ax = axes[i]
            bars = ax.bar(model_names, values)
            
            # Add values on top of bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                      f'{val:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.upper()}')
            ax.set_ylim(0, max(values) * 1.1)  # Add 10% padding
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save plot
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Saved model comparison to {output_file}")
        else:
            plt.show()


def create_intraday_ensemble(features_df: pd.DataFrame, 
                           labels: pd.Series,
                           model_type: str = 'signal_filter',
                           ensemble_type: str = 'voting',
                           models_dir: str = "models/intraday/ensemble") -> Any:
    """
    Create and train an ensemble model for intraday trading.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Features DataFrame
    labels : pd.Series
        Target labels
    model_type : str
        Type of model ('signal_filter', 'entry_timing', etc.)
    ensemble_type : str
        Type of ensemble ('voting', 'stacking', 'boosting', 'bagging', 'diversified')
    models_dir : str
        Directory to save models
        
    Returns:
    --------
    Any
        Trained ensemble model
    """
    factory = EnsembleModelFactory(models_dir=models_dir)
    model_name = f"{model_type}_{ensemble_type}_ensemble"
    
    # Create ensemble based on type
    if ensemble_type == 'voting':
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('lr', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
        ]
        ensemble = factory.create_voting_ensemble(model_name, base_models)
    
    elif ensemble_type == 'stacking':
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        ensemble = factory.create_stacking_ensemble(model_name, base_models)
    
    elif ensemble_type == 'boosting':
        ensemble = factory.create_boosting_ensemble(model_name)
    
    elif ensemble_type == 'bagging':
        ensemble = factory.create_bagging_ensemble(model_name)
    
    elif ensemble_type == 'diversified':
        ensemble = factory.create_diversified_ensemble(model_name)
    
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    # Train ensemble
    trained_model, metrics = factory.train_ensemble(
        X=features_df,
        y=labels,
        model_name=model_name,
        use_walk_forward=True
    )
    
    logger.info(f"Trained {ensemble_type} ensemble for {model_type}")
    logger.info(f"Metrics: {metrics}")
    
    return trained_model 