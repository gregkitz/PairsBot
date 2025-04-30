#!/usr/bin/env python3
"""
Test Script for Model Refinement Components

This script demonstrates the use of the model refinement components:
1. Ensemble models for improved accuracy
2. Advanced feature engineering
3. Automated model retraining
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

from src.ml_enhancements.ensemble_models import EnsembleModelFactory, create_intraday_ensemble
from src.ml_enhancements.feature_engineering.advanced_features import AdvancedFeatureEngineering, create_advanced_feature_set
from src.ml_enhancements.model_retraining import ModelRetrainingManager, auto_retrain_if_needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(start_date="2022-01-01", end_date="2022-12-31", n_samples=1000):
    """
    Load or generate test data for demonstration.
    
    Parameters:
    -----------
    start_date : str
        Start date for data
    end_date : str
        End date for data
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    tuple
        (prices_df, spreads_df, volumes_df, labels)
    """
    # Try to load real data if available
    data_dir = "data/processed"
    symbols = ["GC", "SI", "ZB", "ZN"]
    
    if os.path.exists(data_dir):
        prices = {}
        volumes = {}
        
        for symbol in symbols:
            file_path = os.path.join(data_dir, f"{symbol}.parquet")
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    
                    # Filter by date
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df[(df.index >= start_date) & (df.index <= end_date)]
                    else:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        df.set_index('date', inplace=True)
                    
                    if 'close' in df.columns:
                        prices[symbol] = df['close']
                    
                    if 'volume' in df.columns:
                        volumes[symbol] = df['volume']
                        
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
        
        if prices:
            # Create dataframes
            prices_df = pd.DataFrame(prices)
            volumes_df = pd.DataFrame(volumes) if volumes else None
            
            # Create a simple spread
            if len(prices) >= 2:
                pair = (list(prices.keys())[0], list(prices.keys())[1])
                hedge_ratio = 0.5  # Simple hedge ratio
                
                # Calculate spread
                spread = prices_df[pair[0]] - hedge_ratio * prices_df[pair[1]]
                
                # Calculate z-score
                mean = spread.rolling(window=20).mean()
                std = spread.rolling(window=20).std()
                zscore = (spread - mean) / std
                
                # Create spreads dataframe
                spreads_df = pd.DataFrame({
                    'spread': spread,
                    'mean': mean,
                    'std': std,
                    'zscore': zscore,
                    'hedge_ratio': hedge_ratio
                })
                
                # Create labels (entry signals: 1 for entry, 0 for no entry)
                labels = ((zscore > 2) | (zscore < -2)).astype(int)
                
                return prices_df, spreads_df, volumes_df, labels
    
    # Generate synthetic data if real data not available
    logger.info("Generating synthetic data for testing")
    
    # Generate dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='1H')[:n_samples]
    
    # Generate price data
    prices = {}
    for symbol in symbols:
        # Random walk with drift
        price = 100 + np.random.normal(0, 1, size=len(dates)).cumsum()
        prices[symbol] = pd.Series(price, index=dates)
    
    prices_df = pd.DataFrame(prices)
    
    # Generate volume data
    volumes = {}
    for symbol in symbols:
        # Random volumes with time-of-day pattern
        hour_factor = np.array([0.8 if 12 <= h <= 13 else 1.2 if h < 10 or h > 14 else 1.0 
                              for h in dates.hour])
        volume = np.abs(np.random.normal(1000, 200, size=len(dates)) * hour_factor)
        volumes[symbol] = pd.Series(volume, index=dates)
    
    volumes_df = pd.DataFrame(volumes)
    
    # Generate spread data
    pair = (symbols[0], symbols[1])
    hedge_ratio = 0.5
    
    # Calculate spread
    spread = prices_df[pair[0]] - hedge_ratio * prices_df[pair[1]]
    
    # Add mean-reverting pattern
    noise = np.random.normal(0, 1, size=len(dates))
    ar_factor = 0.8  # AR(1) coefficient
    
    # Make more mean-reverting
    for i in range(1, len(noise)):
        noise[i] = ar_factor * noise[i-1] + (1 - ar_factor) * noise[i]
    
    spread += noise * 5
    
    # Calculate z-score
    mean = spread.rolling(window=20).mean().fillna(method='bfill')
    std = spread.rolling(window=20).std().fillna(method='bfill')
    zscore = (spread - mean) / std
    
    # Create spreads dataframe
    spreads_df = pd.DataFrame({
        'spread': spread,
        'mean': mean,
        'std': std,
        'zscore': zscore,
        'hedge_ratio': hedge_ratio
    })
    
    # Create labels (entry signals)
    labels = ((zscore > 2) | (zscore < -2)).astype(int)
    
    return prices_df, spreads_df, volumes_df, labels


def test_ensemble_models(prices_df, spreads_df, volumes_df, labels):
    """
    Test ensemble models for improved accuracy.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Price data
    spreads_df : pd.DataFrame
        Spread data
    volumes_df : pd.DataFrame
        Volume data
    labels : pd.Series
        Target labels
    """
    logger.info("Testing ensemble models...")
    
    # Split data into train/test sets
    split_idx = int(len(prices_df) * 0.7)
    
    train_prices = prices_df.iloc[:split_idx]
    train_spreads = spreads_df.iloc[:split_idx]
    train_volumes = volumes_df.iloc[:split_idx] if volumes_df is not None else None
    train_labels = labels.iloc[:split_idx]
    
    test_prices = prices_df.iloc[split_idx:]
    test_spreads = spreads_df.iloc[split_idx:]
    test_volumes = volumes_df.iloc[split_idx:] if volumes_df is not None else None
    test_labels = labels.iloc[split_idx:]
    
    # Generate features
    # First, use basic features
    feature_eng = AdvancedFeatureEngineering()
    
    # Generate base features
    base_features = feature_eng.base_features.generate_intraday_features(
        prices_df=train_prices,
        spreads_df=train_spreads,
        volumes_df=train_volumes
    )
    
    # Generate advanced features
    advanced_features = feature_eng.generate_advanced_features(
        prices_df=train_prices,
        spreads_df=train_spreads,
        volumes_df=train_volumes
    )
    
    # Generate test features
    test_base_features = feature_eng.base_features.generate_intraday_features(
        prices_df=test_prices,
        spreads_df=test_spreads,
        volumes_df=test_volumes
    )
    
    test_advanced_features = feature_eng.generate_advanced_features(
        prices_df=test_prices,
        spreads_df=test_spreads,
        volumes_df=test_volumes
    )
    
    # Initialize ensemble factory
    ensemble_factory = EnsembleModelFactory(models_dir="models/intraday/ensemble")
    
    # Create ensemble models with different techniques
    ensembles = {}
    ensemble_types = ["voting", "stacking", "boosting", "bagging", "diversified"]
    
    for ensemble_type in ensemble_types:
        model_name = f"signal_filter_{ensemble_type}"
        
        if ensemble_type == "voting":
            model = ensemble_factory.create_voting_ensemble(
                model_name=model_name,
                base_models=[
                    ('rf', ensemble_factory.base_models.get('random_forest', None)),
                    ('gb', ensemble_factory.base_models.get('gradient_boosting', None)),
                    ('lr', ensemble_factory.base_models.get('logistic_regression', None))
                ]
            )
        elif ensemble_type == "stacking":
            model = ensemble_factory.create_stacking_ensemble(
                model_name=model_name,
                base_models=[
                    ('rf', ensemble_factory.base_models.get('random_forest', None)),
                    ('gb', ensemble_factory.base_models.get('gradient_boosting', None)),
                    ('lr', ensemble_factory.base_models.get('logistic_regression', None))
                ]
            )
        elif ensemble_type == "boosting":
            model = ensemble_factory.create_boosting_ensemble(model_name=model_name)
        elif ensemble_type == "bagging":
            model = ensemble_factory.create_bagging_ensemble(model_name=model_name)
        else:  # diversified
            model = ensemble_factory.create_diversified_ensemble(model_name=model_name)
        
        ensembles[ensemble_type] = model
    
    # Train and evaluate ensemble models
    results = {}
    
    # Evaluate on both base and advanced features
    for feature_set_name, (train_features, test_features) in [
        ("base", (base_features, test_base_features)),
        ("advanced", (advanced_features, test_advanced_features))
    ]:
        # Make sure no NaN values
        train_features = train_features.fillna(0)
        test_features = test_features.fillna(0)
        
        for ensemble_type, model in ensembles.items():
            # Train model
            model.fit(train_features, train_labels)
            
            # Evaluate model
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = model.predict(test_features)
            
            metrics = {
                "accuracy": accuracy_score(test_labels, y_pred),
                "precision": precision_score(test_labels, y_pred, zero_division=0),
                "recall": recall_score(test_labels, y_pred, zero_division=0),
                "f1": f1_score(test_labels, y_pred, zero_division=0)
            }
            
            results[f"{feature_set_name}_{ensemble_type}"] = metrics
            
            logger.info(f"[{feature_set_name}] {ensemble_type} ensemble: {metrics}")
    
    # Visualize ensemble model comparison
    plt.figure(figsize=(12, 8))
    
    metric = "f1"  # Primary metric for comparison
    
    # Compare base vs advanced features
    base_scores = [results[f"base_{t}"][metric] for t in ensemble_types]
    advanced_scores = [results[f"advanced_{t}"][metric] for t in ensemble_types]
    
    x = np.arange(len(ensemble_types))
    width = 0.35
    
    plt.bar(x - width/2, base_scores, width, label='Base Features')
    plt.bar(x + width/2, advanced_scores, width, label='Advanced Features')
    
    plt.xlabel('Ensemble Type')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title('Ensemble Model Comparison')
    plt.xticks(x, ensemble_types)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y')
    
    # Save plot
    os.makedirs("output/model_refinement", exist_ok=True)
    plt.savefig("output/model_refinement/ensemble_comparison.png")
    logger.info("Saved ensemble comparison plot to output/model_refinement/ensemble_comparison.png")
    
    # Return the best ensemble model
    best_model_key = max(results.keys(), key=lambda k: results[k][metric])
    best_model_type = best_model_key.split('_')[1]
    best_feature_set = best_model_key.split('_')[0]
    
    logger.info(f"Best model: {best_model_key} with {metric}={results[best_model_key][metric]:.4f}")
    
    return ensembles[best_model_type], results


def test_advanced_feature_engineering(prices_df, spreads_df, volumes_df, labels):
    """
    Test advanced feature engineering capabilities.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Price data
    spreads_df : pd.DataFrame
        Spread data
    volumes_df : pd.DataFrame
        Volume data
    labels : pd.Series
        Target labels
    """
    logger.info("Testing advanced feature engineering...")
    
    # Initialize advanced feature engineering
    feature_eng = AdvancedFeatureEngineering()
    
    # Generate advanced features
    advanced_features = feature_eng.generate_advanced_features(
        prices_df=prices_df,
        spreads_df=spreads_df,
        volumes_df=volumes_df
    )
    
    # Generate basic features for comparison
    basic_features = feature_eng.base_features.generate_intraday_features(
        prices_df=prices_df,
        spreads_df=spreads_df,
        volumes_df=volumes_df
    )
    
    logger.info(f"Generated {len(advanced_features.columns)} advanced features vs {len(basic_features.columns)} basic features")
    
    # Select top features based on importance to target
    selected_features = feature_eng.select_top_features(
        features=advanced_features,
        target=labels,
        n_features=20,
        method='mutual_info'
    )
    
    logger.info(f"Selected top {len(selected_features.columns)} features:")
    for i, col in enumerate(selected_features.columns, 1):
        importance = feature_eng.feature_importance.get(col, 0)
        logger.info(f"{i}. {col}: {importance:.4f}")
    
    # Visualize feature importances
    feature_eng.visualize_feature_importance("output/model_refinement/feature_importance.png")
    
    # Test feature engineering with different configurations
    feature_configs = {
        "full": {
            "include_base_features": True,
            "include_nonlinear": True,
            "include_temporal": True,
            "include_interaction": True
        },
        "no_nonlinear": {
            "include_base_features": True,
            "include_nonlinear": False,
            "include_temporal": True,
            "include_interaction": True
        },
        "no_temporal": {
            "include_base_features": True,
            "include_nonlinear": True,
            "include_temporal": False,
            "include_interaction": True
        },
        "no_interaction": {
            "include_base_features": True,
            "include_nonlinear": True,
            "include_temporal": True,
            "include_interaction": False
        },
        "basic_only": {
            "include_base_features": True,
            "include_nonlinear": False,
            "include_temporal": False,
            "include_interaction": False
        }
    }
    
    # Generate feature sets for each configuration
    feature_sets = {}
    feature_counts = {}
    
    for config_name, config in feature_configs.items():
        features = feature_eng.generate_advanced_features(
            prices_df=prices_df,
            spreads_df=spreads_df,
            volumes_df=volumes_df,
            **config
        )
        feature_sets[config_name] = features
        feature_counts[config_name] = len(features.columns)
    
    # Test dimensionality reduction
    reduced_features = feature_eng.reduce_dimensionality(advanced_features)
    feature_sets["pca_reduced"] = reduced_features
    feature_counts["pca_reduced"] = len(reduced_features.columns)
    
    # Plot feature counts
    plt.figure(figsize=(10, 6))
    plt.bar(feature_counts.keys(), feature_counts.values())
    plt.xlabel('Feature Configuration')
    plt.ylabel('Feature Count')
    plt.title('Feature Count by Configuration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/model_refinement/feature_counts.png")
    logger.info("Saved feature counts plot to output/model_refinement/feature_counts.png")
    
    return selected_features, feature_sets


def test_automated_retraining(prices_df, spreads_df, volumes_df, labels, best_model=None):
    """
    Test automated model retraining functionality.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Price data
    spreads_df : pd.DataFrame
        Spread data
    volumes_df : pd.DataFrame
        Volume data
    labels : pd.Series
        Target labels
    best_model : object, optional
        Best model from ensemble testing
    """
    logger.info("Testing automated model retraining...")
    
    # Split data into initial training, validation, and new data
    split1 = int(len(prices_df) * 0.5)
    split2 = int(len(prices_df) * 0.7)
    
    # Initial training data
    train_prices = prices_df.iloc[:split1]
    train_spreads = spreads_df.iloc[:split1]
    train_volumes = volumes_df.iloc[:split1] if volumes_df is not None else None
    train_labels = labels.iloc[:split1]
    
    # Validation data
    val_prices = prices_df.iloc[split1:split2]
    val_spreads = spreads_df.iloc[split1:split2]
    val_volumes = volumes_df.iloc[split1:split2] if volumes_df is not None else None
    val_labels = labels.iloc[split1:split2]
    
    # New data for retraining
    new_prices = prices_df.iloc[split2:]
    new_spreads = spreads_df.iloc[split2:]
    new_volumes = volumes_df.iloc[split2:] if volumes_df is not None else None
    new_labels = labels.iloc[split2:]
    
    # Generate features
    feature_eng = AdvancedFeatureEngineering()
    
    # Get training features
    train_features = feature_eng.generate_advanced_features(
        prices_df=train_prices,
        spreads_df=train_spreads,
        volumes_df=train_volumes
    )
    
    # Get validation features
    val_features = feature_eng.generate_advanced_features(
        prices_df=val_prices,
        spreads_df=val_spreads,
        volumes_df=val_volumes
    )
    
    # Get new data features
    new_features = feature_eng.generate_advanced_features(
        prices_df=new_prices,
        spreads_df=new_spreads,
        volumes_df=new_volumes
    )
    
    # Fill missing values
    train_features = train_features.fillna(0)
    val_features = val_features.fillna(0)
    new_features = new_features.fillna(0)
    
    # Define a simple training function
    def train_model(X, y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    
    # Initialize retraining manager
    manager = ModelRetrainingManager(
        models_dir="models/intraday/retraining",
        output_dir="output/model_refinement/retraining"
    )
    
    # Train initial model
    initial_model = train_model(train_features, train_labels)
    
    # Calculate initial metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    val_pred = initial_model.predict(val_features)
    
    initial_metrics = {
        "accuracy": accuracy_score(val_labels, val_pred),
        "precision": precision_score(val_labels, val_pred, zero_division=0),
        "recall": recall_score(val_labels, val_pred, zero_division=0),
        "f1": f1_score(val_labels, val_pred, zero_division=0)
    }
    
    # Register model
    model_id = manager.register_model(
        model_name="signal_filter",
        model_type="classification",
        model_path="models/intraday/retraining/initial_model.joblib",
        initial_metrics=initial_metrics
    )
    
    # Save initial model and reference data
    import joblib
    os.makedirs("models/intraday/retraining", exist_ok=True)
    joblib.dump(initial_model, "models/intraday/retraining/initial_model.joblib")
    
    # Simulate performance degradation by adding noise to new data
    from sklearn.utils import shuffle
    
    # Create noisy data (5% of labels flipped)
    noise_idx = np.random.choice(len(new_labels), size=int(len(new_labels) * 0.05), replace=False)
    noisy_labels = new_labels.copy()
    noisy_labels.iloc[noise_idx] = 1 - noisy_labels.iloc[noise_idx]
    
    # Evaluate on noisy data
    noisy_pred = initial_model.predict(new_features)
    
    noisy_metrics = {
        "accuracy": accuracy_score(noisy_labels, noisy_pred),
        "precision": precision_score(noisy_labels, noisy_pred, zero_division=0),
        "recall": recall_score(noisy_labels, noisy_pred, zero_division=0),
        "f1": f1_score(noisy_labels, noisy_pred, zero_division=0)
    }
    
    # Update metrics with degraded performance
    manager.update_model_metrics(model_id, noisy_metrics, event="performance_degradation")
    
    # Check if retraining is needed
    retrain_needed = manager.check_retraining_needed(model_id)
    logger.info(f"Retraining needed: {retrain_needed}")
    
    # Perform retraining if needed, or force retrain
    if retrain_needed or True:  # Force retrain for demo
        success, new_model_id, metrics = manager.retrain_model(
            model_id=model_id,
            training_func=train_model,
            X_train=new_features,
            y_train=new_labels,
            X_val=val_features,
            y_val=val_labels
        )
        
        logger.info(f"Retraining {'successful' if success else 'failed'}")
        if success:
            logger.info(f"New model ID: {new_model_id}")
            logger.info(f"New metrics: {metrics}")
    
    # Visualize performance history
    manager.visualize_performance_history(
        model_id=model_id,
        metric="f1",
        output_file="output/model_refinement/retraining/performance_history.png"
    )
    
    # Test auto_retrain_if_needed helper function
    retrained, final_model_id = auto_retrain_if_needed(
        model_id=model_id,
        manager=manager,
        training_func=train_model,
        X_new=new_features,
        y_new=new_labels,
        validation_data=(val_features, val_labels),
        force_retrain=True
    )
    
    logger.info(f"Auto retrain result: {retrained}, new model ID: {final_model_id}")
    
    return manager, final_model_id


def main():
    """Main function to run model refinement tests."""
    try:
        # Create output directory
        os.makedirs("output/model_refinement", exist_ok=True)
        
        # Load/generate test data
        prices_df, spreads_df, volumes_df, labels = load_test_data()
        
        if len(prices_df) == 0:
            logger.error("No data available for testing")
            return
        
        logger.info(f"Loaded test data with {len(prices_df)} samples")
        
        # Test ensemble models
        best_model, ensemble_results = test_ensemble_models(
            prices_df, spreads_df, volumes_df, labels
        )
        
        # Test advanced feature engineering
        selected_features, feature_sets = test_advanced_feature_engineering(
            prices_df, spreads_df, volumes_df, labels
        )
        
        # Test automated retraining
        retraining_manager, final_model_id = test_automated_retraining(
            prices_df, spreads_df, volumes_df, labels, best_model
        )
        
        logger.info("Model refinement tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model refinement tests: {e}", exc_info=True)


if __name__ == "__main__":
    main() 