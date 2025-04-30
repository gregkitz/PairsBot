#!/usr/bin/env python3
"""
Performance tests for model training components.

This module measures the performance of model training, hyperparameter tuning,
and feature selection operations.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import tracemalloc
from functools import wraps
from contextlib import contextmanager
import importlib

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.utils import create_directory
except ImportError:
    def create_directory(path):
        """Create directory if it doesn't exist."""
        Path(path).mkdir(parents=True, exist_ok=True)

# Try importing ML components
try:
    from src.ml_enhancements.model_retraining import ModelRetraining
    from src.ml_enhancements.feature_engineering.feature_generator import FeatureGenerator
    from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
    ML_COMPONENTS_AVAILABLE = True
except ImportError:
    ML_COMPONENTS_AVAILABLE = False

# Try importing scikit-learn for fallback implementation
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import xgboost as xgb
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False


def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Get stats
        stats = snapshot.statistics('lineno')
        total = sum(stat.size for stat in stats)
        
        # Convert to MB
        total_mb = total / (1024 * 1024)
        
        return result, total_mb
    
    return wrapper


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


def get_data_size():
    """Get the test data size from environment variables."""
    data_size = os.environ.get("PERFORMANCE_DATA_SIZE", "medium")
    
    # Define data sizes
    sizes = {
        "small": {"samples": 1000, "features": 20},
        "medium": {"samples": 5000, "features": 50},
        "large": {"samples": 20000, "features": 100},
        "production": {"samples": 100000, "features": 200}
    }
    
    return sizes.get(data_size, sizes["medium"])


def create_synthetic_training_data(data_size, random_state=42):
    """
    Create synthetic data for model training performance tests.
    
    Parameters:
    -----------
    data_size : dict
        Dictionary with data size parameters
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    samples = data_size["samples"]
    features = data_size["features"]
    
    # Generate random features
    X = np.random.randn(samples, features)
    
    # Generate synthetic labels (binary classification for signal direction)
    # Use a non-linear function of some features to create patterns
    y = np.zeros(samples)
    
    # Feature 0 and 1 interaction
    y += 0.5 * X[:, 0] * X[:, 1]
    
    # Feature 2 squared
    y += 0.3 * X[:, 2] ** 2
    
    # Feature 3 and 4 difference
    y += 0.4 * (X[:, 3] - X[:, 4])
    
    # Add some noise
    y += 0.2 * np.random.randn(samples)
    
    # Convert to binary classification
    y_binary = (y > 0).astype(int)
    
    # Create time-series like features (lagged variables)
    for i in range(1, 5):
        if i < X.shape[0]:
            X[i:, features-i] = np.roll(X[:, 0], i)[i:]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def test_model_training_performance(X_train, X_test, y_train, y_test, output_dir):
    """
    Test performance of different model training algorithms.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    if not ML_LIBRARIES_AVAILABLE:
        return {"status": "skipped", "reason": "ML libraries not available"}
    
    # Models to benchmark
    model_classes = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "xgboost": xgb.XGBClassifier
    }
    
    # Default parameters
    model_params = {
        "logistic_regression": {"max_iter": 1000, "random_state": 42},
        "random_forest": {"n_estimators": 100, "random_state": 42},
        "gradient_boosting": {"n_estimators": 100, "random_state": 42},
        "xgboost": {"n_estimators": 100, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"}
    }
    
    results = {}
    
    # Create a scaler for preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test each model
    for model_name, model_class in model_classes.items():
        model_results = {}
        
        # Create and train model
        @profile_memory
        def train_model():
            model = model_class(**model_params[model_name])
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            return model, metrics
        
        # Measure training performance
        with measure_time() as training_time:
            model_tuple, memory_usage = train_model()
            model, metrics = model_tuple
        
        # Measure prediction speed
        prediction_times = []
        for _ in range(10):  # Repeat for stability
            start_time = time.time()
            model.predict(X_test_scaled)
            prediction_times.append(time.time() - start_time)
        
        avg_prediction_time = sum(prediction_times) / len(prediction_times)
        
        # Record results
        model_results = {
            "training_time": training_time,
            "prediction_time": avg_prediction_time,
            "memory_usage": memory_usage,
            "metrics": metrics
        }
        
        results[model_name] = model_results
    
    # Create comparison chart for training time
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    training_times = [results[name]["training_time"] for name in model_names]
    
    plt.bar(model_names, training_times)
    plt.title("Model Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.savefig(output_dir / "model_training_time_comparison.png")
    
    # Create comparison chart for prediction time
    plt.figure(figsize=(10, 6))
    prediction_times = [results[name]["prediction_time"] for name in model_names]
    
    plt.bar(model_names, prediction_times)
    plt.title("Model Prediction Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Prediction Time (seconds)")
    plt.savefig(output_dir / "model_prediction_time_comparison.png")
    
    # Create comparison chart for memory usage
    plt.figure(figsize=(10, 6))
    memory_usages = [results[name]["memory_usage"] for name in model_names]
    
    plt.bar(model_names, memory_usages)
    plt.title("Model Memory Usage Comparison")
    plt.xlabel("Model")
    plt.ylabel("Memory Usage (MB)")
    plt.savefig(output_dir / "model_memory_usage_comparison.png")
    
    return results


def test_hyperparameter_tuning_performance(X_train, X_test, y_train, y_test, output_dir):
    """
    Test performance of hyperparameter tuning methods.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    if not ML_LIBRARIES_AVAILABLE:
        return {"status": "skipped", "reason": "ML libraries not available"}
    
    results = {}
    
    # Create a scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    }
    
    # Define base model
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Test grid search
    @profile_memory
    def run_grid_search():
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Make predictions with best model
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        
        return best_model, best_params, best_score, metrics
    
    # Measure grid search performance
    with measure_time() as grid_search_time:
        grid_search_results, memory_usage = run_grid_search()
        best_model, best_params, best_score, metrics = grid_search_results
    
    # Record results
    results["grid_search"] = {
        "execution_time": grid_search_time,
        "memory_usage": memory_usage,
        "best_params": best_params,
        "best_score": best_score,
        "metrics": metrics
    }
    
    # Test random search if scikit-learn version allows
    if hasattr(sklearn.model_selection, 'RandomizedSearchCV'):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import uniform, randint
        
        # Parameter distribution for random search
        param_dist = {
            'C': uniform(0.1, 10),
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        
        @profile_memory
        def run_random_search():
            random_search = RandomizedSearchCV(
                base_model, param_dist, n_iter=10, cv=5, 
                scoring='accuracy', n_jobs=-1, random_state=42
            )
            random_search.fit(X_train_scaled, y_train)
            
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            # Make predictions with best model
            y_pred = best_model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            return best_model, best_params, best_score, metrics
        
        # Measure random search performance
        with measure_time() as random_search_time:
            random_search_results, random_memory_usage = run_random_search()
            random_best_model, random_best_params, random_best_score, random_metrics = random_search_results
        
        # Record results
        results["random_search"] = {
            "execution_time": random_search_time,
            "memory_usage": random_memory_usage,
            "best_params": random_best_params,
            "best_score": random_best_score,
            "metrics": random_metrics
        }
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    times = [results[method]["execution_time"] for method in methods]
    
    plt.bar(methods, times)
    plt.title("Hyperparameter Tuning Method Comparison")
    plt.xlabel("Method")
    plt.ylabel("Execution Time (seconds)")
    plt.savefig(output_dir / "hyperparameter_tuning_comparison.png")
    
    return results


def test_feature_selection_performance(X_train, X_test, y_train, y_test, output_dir):
    """
    Test performance of feature selection methods.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test data
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    if not ML_LIBRARIES_AVAILABLE:
        return {"status": "skipped", "reason": "ML libraries not available"}
    
    results = {}
    
    # Try importing feature selection methods
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Number of features to select (half of total)
        k = X_train.shape[1] // 2
        
        # Test SelectKBest
        @profile_memory
        def run_select_kbest():
            selector = SelectKBest(f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train a model on selected features
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            # Get selected feature indices
            selected_indices = np.where(selector.get_support())[0]
            
            return model, metrics, selected_indices
        
        # Measure SelectKBest performance
        with measure_time() as kbest_time:
            kbest_results, kbest_memory = run_select_kbest()
            kbest_model, kbest_metrics, kbest_indices = kbest_results
        
        # Record results
        results["select_kbest"] = {
            "execution_time": kbest_time,
            "memory_usage": kbest_memory,
            "num_features": len(kbest_indices),
            "metrics": kbest_metrics
        }
        
        # Test RFE
        @profile_memory
        def run_rfe():
            model = LogisticRegression(random_state=42, max_iter=1000)
            selector = RFE(model, n_features_to_select=k, step=1)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train a model on selected features
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            # Get selected feature indices
            selected_indices = np.where(selector.support_)[0]
            
            return model, metrics, selected_indices
        
        # Measure RFE performance
        with measure_time() as rfe_time:
            rfe_results, rfe_memory = run_rfe()
            rfe_model, rfe_metrics, rfe_indices = rfe_results
        
        # Record results
        results["rfe"] = {
            "execution_time": rfe_time,
            "memory_usage": rfe_memory,
            "num_features": len(rfe_indices),
            "metrics": rfe_metrics
        }
        
        # Test SelectFromModel
        @profile_memory
        def run_select_from_model():
            # Use random forest to determine feature importance
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(model, threshold="median")
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train a model on selected features
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }
            
            # Get selected feature indices
            selected_indices = np.where(selector.get_support())[0]
            
            return model, metrics, selected_indices
        
        # Measure SelectFromModel performance
        with measure_time() as sfm_time:
            sfm_results, sfm_memory = run_select_from_model()
            sfm_model, sfm_metrics, sfm_indices = sfm_results
        
        # Record results
        results["select_from_model"] = {
            "execution_time": sfm_time,
            "memory_usage": sfm_memory,
            "num_features": len(sfm_indices),
            "metrics": sfm_metrics
        }
        
        # Create comparison chart
        plt.figure(figsize=(10, 6))
        methods = list(results.keys())
        times = [results[method]["execution_time"] for method in methods]
        
        plt.bar(methods, times)
        plt.title("Feature Selection Method Comparison")
        plt.xlabel("Method")
        plt.ylabel("Execution Time (seconds)")
        plt.savefig(output_dir / "feature_selection_comparison.png")
        
    except ImportError:
        results = {"status": "skipped", "reason": "Feature selection libraries not available"}
    
    return results


def test_model_scaling_performance(output_dir):
    """
    Test how model training performance scales with data size.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    if not ML_LIBRARIES_AVAILABLE:
        return {"status": "skipped", "reason": "ML libraries not available"}
    
    results = {}
    
    # Define data sizes to test
    data_sizes = [
        {"samples": 1000, "features": 20},
        {"samples": 5000, "features": 20},
        {"samples": 10000, "features": 20},
        {"samples": 20000, "features": 20}
    ]
    
    # Select a single model for scaling tests
    model_class = RandomForestClassifier
    model_params = {"n_estimators": 100, "random_state": 42}
    
    # Test each data size
    scaling_results = []
    
    for size in data_sizes:
        # Create test data
        X_train, X_test, y_train, y_test = create_synthetic_training_data(size)
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        @profile_memory
        def train_model():
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            return model
        
        # Measure training performance
        with measure_time() as training_time:
            model, memory_usage = train_model()
        
        # Record results
        scaling_results.append({
            "samples": size["samples"],
            "features": size["features"],
            "training_time": training_time,
            "memory_usage": memory_usage
        })
    
    results["scaling_tests"] = scaling_results
    
    # Create scaling chart - training time vs data size
    plt.figure(figsize=(12, 6))
    samples = [result["samples"] for result in scaling_results]
    times = [result["training_time"] for result in scaling_results]
    
    plt.plot(samples, times, 'o-')
    plt.title(f"Training Time Scaling for {model_class.__name__}")
    plt.xlabel("Number of Samples")
    plt.ylabel("Training Time (seconds)")
    plt.grid(True)
    plt.savefig(output_dir / "training_time_scaling.png")
    
    # Create scaling chart - memory usage vs data size
    plt.figure(figsize=(12, 6))
    memory = [result["memory_usage"] for result in scaling_results]
    
    plt.plot(samples, memory, 'o-')
    plt.title(f"Memory Usage Scaling for {model_class.__name__}")
    plt.xlabel("Number of Samples")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(True)
    plt.savefig(output_dir / "memory_usage_scaling.png")
    
    return results


def test_integrated_ml_components():
    """
    Test the performance of the actual ML components used in the system.
    
    Returns:
    --------
    dict
        Test results
    """
    if not ML_COMPONENTS_AVAILABLE:
        return {"status": "skipped", "reason": "ML components not available"}
    
    # Implementation will depend on the actual ML components in the system
    results = {
        "status": "not_implemented",
        "reason": "System-specific ML component testing needs customization based on actual implementation"
    }
    
    return results


def run_tests(output_dir):
    """
    Run all model training performance tests.
    
    Parameters:
    -----------
    output_dir : Path
        Directory to save test results
        
    Returns:
    --------
    dict
        Test results
    """
    # Create plots directory
    plots_dir = output_dir / "plots"
    create_directory(plots_dir)
    
    # Report ML component availability
    print(f"ML Components Available: {ML_COMPONENTS_AVAILABLE}")
    print(f"ML Libraries Available: {ML_LIBRARIES_AVAILABLE}")
    
    # Results dictionary
    results = {
        "status": "success" if ML_LIBRARIES_AVAILABLE else "limited",
        "metrics": {
            "ml_components_available": ML_COMPONENTS_AVAILABLE,
            "ml_libraries_available": ML_LIBRARIES_AVAILABLE
        },
        "test_results": {}
    }
    
    if not ML_LIBRARIES_AVAILABLE:
        results["status"] = "skipped"
        results["reason"] = "ML libraries not available"
        return results
    
    # Get data size and create test data
    data_size = get_data_size()
    print(f"Running tests with data size: {data_size}")
    
    print("Creating test data...")
    X_train, X_test, y_train, y_test = create_synthetic_training_data(data_size)
    
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Run tests
    print("Testing model training performance...")
    results["test_results"]["model_training"] = test_model_training_performance(
        X_train, X_test, y_train, y_test, plots_dir
    )
    
    print("Testing hyperparameter tuning performance...")
    results["test_results"]["hyperparameter_tuning"] = test_hyperparameter_tuning_performance(
        X_train, X_test, y_train, y_test, plots_dir
    )
    
    print("Testing feature selection performance...")
    results["test_results"]["feature_selection"] = test_feature_selection_performance(
        X_train, X_test, y_train, y_test, plots_dir
    )
    
    print("Testing model scaling performance...")
    results["test_results"]["scaling"] = test_model_scaling_performance(plots_dir)
    
    if ML_COMPONENTS_AVAILABLE:
        print("Testing integrated ML components...")
        results["test_results"]["integrated_ml"] = test_integrated_ml_components()
    
    # Save detailed results
    with open(output_dir / "model_training_performance.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == "__main__":
    # Run tests with default settings
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"performance_results/model_training_{timestamp}")
    create_directory(output_dir)
    
    results = run_tests(output_dir)
    
    print(f"Tests completed. Results saved to {output_dir}") 