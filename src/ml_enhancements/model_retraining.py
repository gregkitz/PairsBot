"""
Automated Model Retraining Module

This module provides functionality for automated retraining of machine learning models
for intraday trading when new data becomes available or model performance degrades.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import joblib
import time
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainingManager:
    """
    Manages automated retraining of machine learning models for intraday trading.
    
    This class monitors model performance, detects when retraining is needed,
    and orchestrates the retraining process.
    """
    
    def __init__(self, 
                models_dir: str = "models/intraday",
                output_dir: str = "output/retraining",
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model retraining manager.
        
        Parameters:
        -----------
        models_dir : str
            Directory where models are stored
        output_dir : str
            Directory for retraining output and logs
        config : dict, optional
            Configuration parameters for retraining
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
        
        # Default configuration
        default_config = {
            "performance_threshold": 0.7,      # Performance threshold for retraining
            "performance_metric": "f1",        # Default metric for classification
            "data_drift_threshold": 0.05,      # Distribution drift threshold
            "min_samples_for_retraining": 500, # Minimum samples needed for retraining
            "max_models_to_keep": 5,           # Maximum number of model versions to keep
            "retraining_frequency": "weekly",  # How often to check for retraining
            "last_retraining_date": None,      # Last retraining date
            "retrain_on_regime_change": True,  # Whether to retrain on regime change
            "preserve_history": True,          # Keep model history and metrics
            "enable_monitoring": True          # Enable performance monitoring
        }
        
        # Update with user config if provided
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize model tracking
        self.model_registry = {}      # Store model metadata
        self.performance_history = {} # Track performance over time
        self.data_drift_metrics = {}  # Track data drift metrics
        
        # Load existing model registry if available
        self._load_model_registry()
        
    def _load_model_registry(self):
        """Load existing model registry from disk."""
        registry_path = os.path.join(self.output_dir, "model_registry.json")
        
        if os.path.exists(registry_path):
            try:
                self.model_registry = pd.read_json(registry_path).to_dict(orient="index")
                logger.info(f"Loaded model registry with {len(self.model_registry)} models")
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
                self.model_registry = {}
        
    def _save_model_registry(self):
        """Save model registry to disk."""
        registry_path = os.path.join(self.output_dir, "model_registry.json")
        
        try:
            pd.DataFrame.from_dict(self.model_registry, orient="index").to_json(registry_path)
            logger.info(f"Saved model registry with {len(self.model_registry)} models")
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    def register_model(self, 
                      model_name: str, 
                      model_type: str, 
                      model_path: str, 
                      initial_metrics: Dict[str, float],
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Register a new model for monitoring and automated retraining.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model_type : str
            Type of model (e.g., 'classification', 'regression')
        model_path : str
            Path to the saved model file
        initial_metrics : dict
            Initial performance metrics
        metadata : dict, optional
            Additional metadata about the model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_name}_{timestamp}"
        
        # Create model registry entry
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "model_path": model_path,
            "created_at": timestamp,
            "last_updated": timestamp,
            "retraining_count": 0,
            "current_metrics": initial_metrics,
            "best_metrics": initial_metrics.copy(),
            "active": True  # Flag to indicate if model is active
        }
        
        # Add metadata if provided
        if metadata:
            model_info.update(metadata)
        
        # Add to registry
        self.model_registry[model_id] = model_info
        
        # Initialize performance history
        self.performance_history[model_id] = [{
            "timestamp": timestamp,
            "metrics": initial_metrics.copy(),
            "event": "initial_registration"
        }]
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Registered model {model_name} with ID {model_id}")
        
        return model_id
    
    def update_model_metrics(self, 
                            model_id: str, 
                            new_metrics: Dict[str, float], 
                            event: str = "performance_update"):
        """
        Update metrics for a tracked model.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to update
        new_metrics : dict
            New performance metrics
        event : str
            Event type for the update (e.g., 'performance_update', 'retraining')
        """
        if model_id not in self.model_registry:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get model info
        model_info = self.model_registry[model_id]
        
        # Update metrics
        model_info["current_metrics"] = new_metrics
        model_info["last_updated"] = timestamp
        
        # Check if new metrics are better than best metrics
        metric_improved = False
        primary_metric = self.config["performance_metric"]
        
        if primary_metric in new_metrics and primary_metric in model_info["best_metrics"]:
            # For classification metrics, higher is better
            if model_info["model_type"] == "classification":
                if new_metrics[primary_metric] > model_info["best_metrics"][primary_metric]:
                    model_info["best_metrics"] = new_metrics.copy()
                    metric_improved = True
            # For regression metrics (like MSE, RMSE), lower is better
            elif model_info["model_type"] == "regression":
                if new_metrics[primary_metric] < model_info["best_metrics"][primary_metric]:
                    model_info["best_metrics"] = new_metrics.copy()
                    metric_improved = True
        
        # Add to performance history
        if model_id in self.performance_history:
            self.performance_history[model_id].append({
                "timestamp": timestamp,
                "metrics": new_metrics.copy(),
                "event": event,
                "improved": metric_improved
            })
        else:
            self.performance_history[model_id] = [{
                "timestamp": timestamp,
                "metrics": new_metrics.copy(),
                "event": event,
                "improved": metric_improved
            }]
        
        # Save registry
        self._save_model_registry()
        
        # Log update
        if metric_improved:
            logger.info(f"Model {model_id} metrics improved: {new_metrics}")
        else:
            logger.info(f"Model {model_id} metrics updated: {new_metrics}")
        
        return True
    
    def check_retraining_needed(self, 
                              model_id: str, 
                              current_metrics: Optional[Dict[str, float]] = None,
                              force_check: bool = False) -> bool:
        """
        Check if model retraining is needed based on performance metrics.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to check
        current_metrics : dict, optional
            Current performance metrics (if not provided, uses last recorded metrics)
        force_check : bool
            Force retraining check regardless of time since last check
            
        Returns:
        --------
        bool
            True if retraining is needed, False otherwise
        """
        if model_id not in self.model_registry:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        model_info = self.model_registry[model_id]
        primary_metric = self.config["performance_metric"]
        performance_threshold = self.config["performance_threshold"]
        
        # Use provided metrics or current metrics from registry
        metrics = current_metrics if current_metrics is not None else model_info["current_metrics"]
        
        # Check retraining frequency
        if not force_check:
            # Check if enough time has passed since last retraining
            last_retrained = model_info.get("last_retrained")
            frequency = self.config["retraining_frequency"]
            
            if last_retrained:
                last_date = datetime.strptime(last_retrained, "%Y%m%d_%H%M%S")
                current_date = datetime.now()
                
                # Check based on frequency
                if frequency == "daily" and (current_date - last_date).days < 1:
                    return False
                elif frequency == "weekly" and (current_date - last_date).days < 7:
                    return False
                elif frequency == "monthly" and (current_date - last_date).days < 30:
                    return False
        
        # Check performance metrics
        if primary_metric in metrics:
            current_value = metrics[primary_metric]
            best_value = model_info["best_metrics"].get(primary_metric, current_value)
            
            # For classification metrics (higher is better)
            if model_info["model_type"] == "classification":
                # Retrain if below threshold or significant degradation from best
                return current_value < performance_threshold or current_value < best_value * 0.9
            
            # For regression metrics (lower is better)
            elif model_info["model_type"] == "regression":
                # Retrain if above threshold or significant degradation from best
                return current_value > performance_threshold or current_value > best_value * 1.1
        
        return False
    
    def detect_data_drift(self, 
                         model_id: str, 
                         new_data: pd.DataFrame, 
                         reference_data: Optional[pd.DataFrame] = None) -> Tuple[bool, Dict[str, float]]:
        """
        Detect drift in input data distribution.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to check
        new_data : pd.DataFrame
            New incoming data
        reference_data : pd.DataFrame, optional
            Reference data distribution (if not provided, uses stored reference)
            
        Returns:
        --------
        tuple
            (drift_detected, drift_metrics)
        """
        if model_id not in self.model_registry:
            logger.error(f"Model ID {model_id} not found in registry")
            return False, {}
        
        drift_threshold = self.config["data_drift_threshold"]
        drift_metrics = {}
        
        # Get reference data if not provided
        if reference_data is None:
            reference_path = os.path.join(self.output_dir, "references", f"{model_id}_reference.pkl")
            if os.path.exists(reference_path):
                try:
                    reference_data = joblib.load(reference_path)
                except Exception as e:
                    logger.error(f"Error loading reference data: {e}")
                    return False, {}
            else:
                logger.warning(f"No reference data found for model {model_id}")
                return False, {}
        
        # Check column match
        if set(new_data.columns) != set(reference_data.columns):
            logger.error("Column mismatch between new and reference data")
            return False, {}
        
        # Calculate drift metrics for each numeric column
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Calculate distribution similarity using Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            ks_stat, p_value = ks_2samp(reference_data[col].values, new_data[col].values)
            
            drift_metrics[f"{col}_ks_stat"] = ks_stat
            drift_metrics[f"{col}_p_value"] = p_value
            
            # Calculate mean and std deviation drift
            mean_drift = abs(reference_data[col].mean() - new_data[col].mean()) / max(1e-6, reference_data[col].std())
            std_drift = abs(reference_data[col].std() - new_data[col].std()) / max(1e-6, reference_data[col].std())
            
            drift_metrics[f"{col}_mean_drift"] = mean_drift
            drift_metrics[f"{col}_std_drift"] = std_drift
        
        # Calculate overall drift score (average KS statistic)
        ks_stats = [v for k, v in drift_metrics.items() if k.endswith('_ks_stat')]
        overall_drift = np.mean(ks_stats) if ks_stats else 0
        drift_metrics["overall_drift"] = overall_drift
        
        # Store drift metrics
        if model_id in self.data_drift_metrics:
            self.data_drift_metrics[model_id].append({
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "metrics": drift_metrics
            })
        else:
            self.data_drift_metrics[model_id] = [{
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "metrics": drift_metrics
            }]
        
        # Check if drift is significant
        drift_detected = overall_drift > drift_threshold
        
        if drift_detected:
            logger.info(f"Data drift detected for model {model_id}: {overall_drift:.4f} > {drift_threshold}")
        
        return drift_detected, drift_metrics
    
    def retrain_model(self, 
                     model_id: str, 
                     training_func: Callable, 
                     X_train: pd.DataFrame, 
                     y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None) -> Tuple[bool, str, Dict[str, float]]:
        """
        Retrain a model and update its registry entry.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to retrain
        training_func : callable
            Function that takes X_train, y_train and returns a trained model
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
            
        Returns:
        --------
        tuple
            (success, new_model_id, metrics)
        """
        if model_id not in self.model_registry:
            logger.error(f"Model ID {model_id} not found in registry")
            return False, "", {}
        
        # Get model info
        model_info = self.model_registry[model_id]
        model_name = model_info["model_name"]
        model_type = model_info["model_type"]
        
        # Check if enough samples for training
        min_samples = self.config["min_samples_for_retraining"]
        if len(X_train) < min_samples:
            logger.warning(f"Not enough samples for retraining: {len(X_train)} < {min_samples}")
            return False, "", {}
        
        # Generate timestamp for new model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_id = f"{model_name}_{timestamp}"
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Start timing
            start_time = time.time()
            
            # Train model
            logger.info(f"Retraining model {model_name}...")
            new_model = training_func(X_train, y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Save model
            model_path = os.path.join(model_dir, f"{new_model_id}.joblib")
            joblib.dump(new_model, model_path)
            
            # Save reference data for drift detection
            reference_dir = os.path.join(self.output_dir, "references")
            os.makedirs(reference_dir, exist_ok=True)
            joblib.dump(X_train, os.path.join(reference_dir, f"{new_model_id}_reference.pkl"))
            
            # Calculate metrics
            metrics = {}
            
            # If validation data is provided, calculate metrics
            if X_val is not None and y_val is not None:
                # Make predictions
                if model_type == "classification":
                    y_pred = new_model.predict(X_val)
                    y_prob = new_model.predict_proba(X_val)[:, 1] if hasattr(new_model, "predict_proba") else None
                    
                    # Calculate classification metrics
                    metrics["accuracy"] = accuracy_score(y_val, y_pred)
                    metrics["precision"] = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                    metrics["recall"] = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                    metrics["f1"] = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                    
                    # Calculate AUC if binary classification and probabilities available
                    if y_prob is not None and len(np.unique(y_val)) == 2:
                        from sklearn.metrics import roc_auc_score
                        metrics["auc"] = roc_auc_score(y_val, y_prob)
                
                elif model_type == "regression":
                    y_pred = new_model.predict(X_val)
                    
                    # Calculate regression metrics
                    metrics["mse"] = mean_squared_error(y_val, y_pred)
                    metrics["rmse"] = np.sqrt(metrics["mse"])
                    metrics["mae"] = mean_absolute_error(y_val, y_pred)
                    metrics["r2"] = r2_score(y_val, y_pred)
            
            # Add training info to metrics
            metrics["training_time"] = training_time
            metrics["training_samples"] = len(X_train)
            metrics["validation_samples"] = len(X_val) if X_val is not None else 0
            
            # Create model registry entry
            new_model_info = {
                "model_id": new_model_id,
                "model_name": model_name,
                "model_type": model_type,
                "model_path": model_path,
                "created_at": timestamp,
                "last_updated": timestamp,
                "retraining_count": model_info["retraining_count"] + 1,
                "parent_model_id": model_id,
                "current_metrics": metrics,
                "best_metrics": metrics.copy(),
                "active": True
            }
            
            # Add to registry
            self.model_registry[new_model_id] = new_model_info
            
            # Mark previous model as inactive if configured to do so
            if not self.config["preserve_history"]:
                model_info["active"] = False
            
            # Initialize performance history
            self.performance_history[new_model_id] = [{
                "timestamp": timestamp,
                "metrics": metrics.copy(),
                "event": "retraining"
            }]
            
            # Save registry
            self._save_model_registry()
            
            # Clean up old model versions if needed
            self._cleanup_old_models(model_name)
            
            logger.info(f"Model {model_name} retrained successfully as {new_model_id}")
            logger.info(f"Metrics: {metrics}")
            
            return True, new_model_id, metrics
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {e}")
            return False, "", {}
    
    def _cleanup_old_models(self, model_name: str):
        """
        Remove old versions of a model to conserve space.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to clean up
        """
        max_models = self.config["max_models_to_keep"]
        
        # Get all models with this name
        models = [(k, v) for k, v in self.model_registry.items() 
                if v.get("model_name") == model_name and v.get("active", True)]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
        
        # If we have more than max_models, remove the oldest ones
        if len(models) > max_models:
            for model_id, _ in models[max_models:]:
                # Mark as inactive
                self.model_registry[model_id]["active"] = False
                
                # Only delete file if configured to not preserve history
                if not self.config["preserve_history"]:
                    try:
                        # Get model path
                        model_path = self.model_registry[model_id].get("model_path")
                        if model_path and os.path.exists(model_path):
                            os.remove(model_path)
                            logger.info(f"Removed old model file: {model_path}")
                    except Exception as e:
                        logger.error(f"Error removing old model file: {e}")
            
            logger.info(f"Cleaned up {len(models) - max_models} old versions of model {model_name}")
    
    def get_active_models(self, model_name: Optional[str] = None) -> List[str]:
        """
        Get list of active model IDs, optionally filtered by name.
        
        Parameters:
        -----------
        model_name : str, optional
            If provided, only return models with this name
            
        Returns:
        --------
        list
            List of active model IDs
        """
        active_models = []
        
        for model_id, model_info in self.model_registry.items():
            # Check if active
            if model_info.get("active", True):
                # If model_name provided, check for match
                if model_name is None or model_info.get("model_name") == model_name:
                    active_models.append(model_id)
        
        return active_models
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """
        Load a model from disk by ID.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to load
            
        Returns:
        --------
        object or None
            Loaded model or None if not found
        """
        if model_id not in self.model_registry:
            logger.error(f"Model ID {model_id} not found in registry")
            return None
        
        model_path = self.model_registry[model_id].get("model_path")
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            # Load model
            model = joblib.load(model_path)
            logger.info(f"Loaded model {model_id} from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def visualize_performance_history(self, 
                                     model_id: str, 
                                     metric: Optional[str] = None,
                                     output_file: Optional[str] = None):
        """
        Visualize performance history of a model.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to visualize
        metric : str, optional
            Specific metric to visualize (if None, uses primary metric)
        output_file : str, optional
            Path to save visualization (if None, displays plot)
        """
        if model_id not in self.model_registry or model_id not in self.performance_history:
            logger.error(f"Model ID {model_id} not found in registry or has no history")
            return
        
        # Get model info
        model_info = self.model_registry[model_id]
        model_name = model_info.get("model_name", "Unknown")
        
        # Get performance history
        history = self.performance_history[model_id]
        
        # Use specified metric or primary metric
        if metric is None:
            metric = self.config["performance_metric"]
        
        # Extract timestamps and metric values
        timestamps = []
        values = []
        events = []
        
        for entry in history:
            if metric in entry.get("metrics", {}):
                timestamps.append(datetime.strptime(entry["timestamp"], "%Y%m%d_%H%M%S"))
                values.append(entry["metrics"][metric])
                events.append(entry.get("event", "unknown"))
        
        if not timestamps:
            logger.warning(f"No {metric} data found in performance history for model {model_id}")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot metric values
        plt.plot(timestamps, values, 'b-', marker='o')
        
        # Highlight different events with different colors
        event_types = set(events)
        colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
        
        for i, event_type in enumerate(event_types):
            event_indices = [j for j, e in enumerate(events) if e == event_type]
            plt.scatter([timestamps[j] for j in event_indices], 
                      [values[j] for j in event_indices],
                      label=event_type, color=colors[i], s=100)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel(metric.capitalize())
        plt.title(f'{model_name} {metric.capitalize()} History')
        plt.grid(True)
        plt.legend()
        
        # Format x-axis as dates
        plt.gcf().autofmt_xdate()
        
        # Adjust y-axis limits
        buffer = (max(values) - min(values)) * 0.1
        plt.ylim(min(values) - buffer, max(values) + buffer)
        
        # Save or display plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Saved performance history visualization to {output_file}")
        else:
            plt.tight_layout()
            plt.show()
    
    def visualize_drift_metrics(self, 
                              model_id: str,
                              output_file: Optional[str] = None):
        """
        Visualize data drift metrics over time.
        
        Parameters:
        -----------
        model_id : str
            ID of the model to visualize
        output_file : str, optional
            Path to save visualization (if None, displays plot)
        """
        if model_id not in self.data_drift_metrics:
            logger.error(f"No drift metrics found for model {model_id}")
            return
        
        # Get model info
        model_name = self.model_registry.get(model_id, {}).get("model_name", "Unknown")
        
        # Get drift metrics
        drift_history = self.data_drift_metrics[model_id]
        
        # Extract timestamps and overall drift
        timestamps = []
        drift_values = []
        
        for entry in drift_history:
            if "metrics" in entry and "overall_drift" in entry["metrics"]:
                timestamps.append(datetime.strptime(entry["timestamp"], "%Y%m%d_%H%M%S"))
                drift_values.append(entry["metrics"]["overall_drift"])
        
        if not timestamps:
            logger.warning(f"No overall drift metrics found for model {model_id}")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot drift values
        plt.plot(timestamps, drift_values, 'r-', marker='o')
        
        # Add threshold line
        threshold = self.config["data_drift_threshold"]
        plt.axhline(y=threshold, color='k', linestyle='--', 
                  label=f'Drift Threshold ({threshold})')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Data Drift Score')
        plt.title(f'{model_name} Data Drift History')
        plt.grid(True)
        plt.legend()
        
        # Format x-axis as dates
        plt.gcf().autofmt_xdate()
        
        # Save or display plot
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Saved drift metrics visualization to {output_file}")
        else:
            plt.tight_layout()
            plt.show()


# Helper functions for working with the ModelRetrainingManager

def create_retraining_manager(config_path: Optional[str] = None) -> ModelRetrainingManager:
    """
    Create and initialize a model retraining manager.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file
        
    Returns:
    --------
    ModelRetrainingManager
        Initialized retraining manager
    """
    # Load configuration if provided
    config = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Create manager
    return ModelRetrainingManager(config=config)


def auto_retrain_if_needed(model_id: str, 
                         manager: ModelRetrainingManager, 
                         training_func: Callable,
                         X_new: pd.DataFrame,
                         y_new: pd.Series,
                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                         reference_data: Optional[pd.DataFrame] = None,
                         force_retrain: bool = False) -> Tuple[bool, str]:
    """
    Check if retraining is needed and perform retraining if necessary.
    
    Parameters:
    -----------
    model_id : str
        ID of the model to check
    manager : ModelRetrainingManager
        Retraining manager instance
    training_func : callable
        Function that takes X_train, y_train and returns a trained model
    X_new : pd.DataFrame
        New features data
    y_new : pd.Series
        New target data
    validation_data : tuple, optional
        (X_val, y_val) tuple for model validation
    reference_data : pd.DataFrame, optional
        Reference data for drift detection
    force_retrain : bool
        Force retraining regardless of checks
        
    Returns:
    --------
    tuple
        (retrained, new_model_id)
    """
    # Extract validation data if provided
    X_val, y_val = validation_data if validation_data else (None, None)
    
    # Check for data drift if reference data provided
    drift_detected = False
    if reference_data is not None:
        drift_detected, _ = manager.detect_data_drift(model_id, X_new, reference_data)
    
    # Check if retraining is needed based on performance or drift
    retrain_needed = force_retrain or drift_detected
    
    if not retrain_needed:
        retrain_needed = manager.check_retraining_needed(model_id)
    
    # Perform retraining if needed
    if retrain_needed:
        logger.info(f"Retraining model {model_id}...")
        success, new_id, _ = manager.retrain_model(
            model_id=model_id,
            training_func=training_func,
            X_train=X_new,
            y_train=y_new,
            X_val=X_val,
            y_val=y_val
        )
        
        if success:
            return True, new_id
        else:
            logger.warning(f"Retraining failed for model {model_id}")
            return False, model_id
    
    logger.info(f"No retraining needed for model {model_id}")
    return False, model_id 