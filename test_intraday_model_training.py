"""
Test script for intraday model training modules.

This script demonstrates how to use the training utilities, data augmentation,
and model trainer modules for intraday trading.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_processor import IntradayDataProcessor
from src.ml_enhancements.training_utils import WalkForwardValidator, generate_labels, create_train_test_labels
from src.ml_enhancements.data_augmentation import TimeSeriesAugmentation
from src.ml_enhancements.intraday_model_trainer import IntradayModelTrainer
from src.ml_enhancements.feature_engineering import IntradayFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data():
    """Prepare training data for testing model training components."""
    # Initialize data processor
    data_dir = "data/processed"
    data_processor = IntradayDataProcessor(data_dir=data_dir)
    
    # Initialize feature engineering
    feature_engineering = IntradayFeatureEngineering(output_dir="output/feature_analysis")
    
    # Get available symbols
    symbols = data_processor.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return None, None, None
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Preparing training data for pair: {pair_id}")
    
    # Set date range for training
    start_date = "2023-01-01"
    end_date = "2023-06-30"
    
    # Load pair data
    pair_data = data_processor.load_pair_data(pair_id, start_date, end_date, timeframe="5min")
    
    if pair_data is None:
        logger.error("Failed to load pair data")
        return None, None, None
    
    # Create test configuration
    test_config = {
        "lookback": 20,
        "hedge_ratio": 1.0,
        "use_rolling_regression": True,
        "regression_window": 30
    }
    
    # Calculate spread
    symbol1, symbol2 = pair_id.split('_')
    spread_df = data_processor.calculate_spread(
        pair_data['prices'], symbol1, symbol2, test_config
    )
    
    if spread_df is None:
        logger.error("Failed to calculate spread")
        return None, None, None
    
    # Generate features
    features = feature_engineering.generate_intraday_features(
        prices_df=pair_data['prices'],
        spreads_df=spread_df,
        volumes_df=pair_data['volumes'],
        include_time_features=True,
        include_liquidity_features=True,
        include_microstructure=False,
        scale_output=False  # We'll scale during model training
    )
    
    if features.empty:
        logger.error("Failed to generate features")
        return None, None, None
    
    # Drop NaN values
    features = features.dropna()
    
    # Align spread_df and pair_data with features
    common_idx = features.index
    spread_df = spread_df.loc[common_idx]
    
    prices_aligned = {}
    for symbol, data in pair_data['prices'].items():
        prices_aligned[symbol] = data.loc[common_idx]
    pair_data['prices'] = pd.DataFrame(prices_aligned)
    
    if pair_data['volumes'] is not None:
        volumes_aligned = {}
        for symbol, data in pair_data['volumes'].items():
            volumes_aligned[symbol] = data.loc[common_idx]
        pair_data['volumes'] = pd.DataFrame(volumes_aligned)
    
    logger.info(f"Prepared {len(features)} rows of training data with {len(features.columns)} features")
    
    return features, pair_data, spread_df

def test_walk_forward_validation(features, pair_data, spread_df):
    """Test the walk-forward validation functionality."""
    if features is None or pair_data is None or spread_df is None:
        logger.error("No data available for testing walk-forward validation")
        return
    
    logger.info("Testing walk-forward validation")
    
    # Create a synthetic target for testing (direction of next 10-period return)
    symbol1 = pair_data['prices'].columns[0]
    future_return = pair_data['prices'][symbol1].pct_change(10).shift(-10)
    
    # Create binary labels
    labels = pd.Series(0, index=future_return.index)
    labels[future_return > 0.001] = 1  # Up
    labels[future_return < -0.001] = -1  # Down
    
    # Drop NaN values
    valid_idx = ~labels.isna()
    features_valid = features.loc[valid_idx]
    labels_valid = labels.loc[valid_idx]
    
    # Create validator
    validator = WalkForwardValidator(n_splits=5, train_size=1000, test_size=200)
    
    # Get fold information
    fold_info = validator.get_fold_info(features_valid)
    logger.info(f"Walk-forward validation folds: {len(fold_info)}")
    
    # Display fold info
    for fold, row in fold_info.iterrows():
        logger.info(f"Fold {row['fold']}: Train {row['train_start']} to {row['train_end']}, Test {row['test_start']} to {row['test_end']}")
    
    # Get splits
    splits = validator.split(features_valid)
    logger.info(f"Generated {len(splits)} train/test splits")
    
    # Create a simple model for testing
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    
    # Test each split
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train = features_valid.iloc[train_idx]
        y_train = labels_valid.iloc[train_idx]
        X_test = features_valid.iloc[test_idx]
        y_test = labels_valid.iloc[test_idx]
        
        logger.info(f"Split {i}: Train {len(X_train)} samples, Test {len(X_test)} samples")
        
        # Class distribution in train and test
        train_counts = y_train.value_counts()
        test_counts = y_test.value_counts()
        logger.info(f"Train class distribution: {train_counts}")
        logger.info(f"Test class distribution: {test_counts}")
    
    # Visualize folds
    output_dir = "output/model_training"
    os.makedirs(output_dir, exist_ok=True)
    
    validator.plot_folds(features_valid, labels_valid, 
                       output_file=os.path.join(output_dir, "walk_forward_folds.png"))
    
    logger.info(f"Walk-forward visualization saved to {os.path.join(output_dir, 'walk_forward_folds.png')}")

def test_label_generation(features, pair_data, spread_df):
    """Test the label generation functionality."""
    if features is None or pair_data is None or spread_df is None:
        logger.error("No data available for testing label generation")
        return
    
    logger.info("Testing label generation")
    
    # Generate various types of labels
    label_types = ['direction', 'return', 'zscore_reversal', 'price_reversal']
    lookforward_periods = [5, 10, 20]
    
    # Generate labels for each type and period
    all_labels = {}
    
    for label_type in label_types:
        for period in lookforward_periods:
            try:
                labels = generate_labels(
                    prices_df=pair_data['prices'],
                    spreads_df=spread_df,
                    lookforward=period,
                    threshold=0.001,
                    label_type=label_type
                )
                
                label_name = f"{label_type}_{period}"
                all_labels[label_name] = labels
                
                # Get class distribution for categorical labels
                if label_type in ['direction', 'zscore_reversal', 'price_reversal']:
                    counts = labels.value_counts()
                    logger.info(f"{label_name} class distribution: {counts}")
                else:
                    # For continuous labels, show statistics
                    stats = labels.describe()
                    logger.info(f"{label_name} statistics: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            except Exception as e:
                logger.error(f"Error generating {label_type} labels for period {period}: {e}")
    
    # Use the create_train_test_labels function
    all_labels_at_once = create_train_test_labels(
        features=features,
        prices_df=pair_data['prices'],
        spreads_df=spread_df,
        lookforward_periods=lookforward_periods,
        label_types=label_types
    )
    
    logger.info(f"Generated {len(all_labels_at_once)} label sets using create_train_test_labels")
    
    # Compare a few labels to ensure consistency
    for key in list(all_labels.keys())[:2]:
        if key in all_labels_at_once:
            is_equal = all_labels[key].equals(all_labels_at_once[key])
            logger.info(f"Labels {key} equal: {is_equal}")
    
    # Return direction labels for later use
    direction_labels = all_labels.get('direction_10')
    
    return direction_labels

def test_data_augmentation(features, labels):
    """Test the data augmentation functionality."""
    if features is None or labels is None:
        logger.error("No data available for testing data augmentation")
        return
    
    logger.info("Testing data augmentation")
    
    # Create augmenter
    augmenter = TimeSeriesAugmentation(
        preserve_features=['hour', 'minute', 'day_of_week'],
        random_seed=42
    )
    
    # Test simple jitter
    X_aug, y_aug = augmenter.jitter(features.iloc[:100], labels.iloc[:100], std=0.01)
    logger.info(f"Jitter augmentation: {len(X_aug)} samples")
    
    # Test scaling
    X_aug, y_aug = augmenter.scaling(features.iloc[:100], labels.iloc[:100], scaling_factor=1.2)
    logger.info(f"Scaling augmentation: {len(X_aug)} samples")
    
    # Test multiple augmentations
    X_aug, y_aug = augmenter.augment_dataset(
        features.iloc[:100], 
        labels.iloc[:100],
        methods=['jitter', 'scaling', 'magnitude_warp'],
        n_augmentations=3
    )
    logger.info(f"Multiple augmentations: {len(X_aug)} samples (original + {len(X_aug) - 100} augmented)")
    
    # Test financial specific augmentation
    X_aug, y_aug = augmenter.financial_specific_augmentation(
        features.iloc[:100],
        labels.iloc[:100],
        vol_scaling=True,
        trend_reset=True
    )
    logger.info(f"Financial specific augmentation: {len(X_aug)} samples")
    
    # Test balanced augmentation
    # Filter to non-NaN labels
    valid_idx = ~labels.isna()
    X_valid = features.loc[valid_idx]
    y_valid = labels.loc[valid_idx]
    
    # Get original class distribution
    original_counts = y_valid.value_counts()
    logger.info(f"Original class distribution: {original_counts}")
    
    # Balance classes
    X_balanced, y_balanced = augmenter.balanced_augmentation(
        X_valid,
        y_valid,
        target_class_ratio=None,  # Auto-balance
        methods=['jitter', 'scaling']
    )
    
    balanced_counts = y_balanced.value_counts()
    logger.info(f"Balanced class distribution: {balanced_counts}")
    
    return X_balanced, y_balanced

def test_model_trainer(features, labels):
    """Test the model trainer functionality."""
    if features is None or labels is None:
        logger.error("No data available for testing model trainer")
        return
    
    logger.info("Testing model trainer")
    
    # Create model trainer
    models_dir = "models/intraday_test"
    os.makedirs(models_dir, exist_ok=True)
    
    trainer = IntradayModelTrainer(models_dir=models_dir, random_seed=42)
    
    # Filter to non-NaN labels
    valid_idx = ~labels.isna()
    X_valid = features.loc[valid_idx]
    y_valid = labels.loc[valid_idx]
    
    # Train a simple signal filter model
    model, metrics = trainer.train_model(
        X=X_valid,
        y=y_valid,
        model_name="test_signal_filter",
        model_type="signal_filter",
        use_walk_forward=True,
        n_splits=3,
        scale_features=True
    )
    
    logger.info(f"Trained model test_signal_filter with metrics: {metrics}")
    
    # Make predictions
    y_pred = trainer.predict(X_valid, model_name="test_signal_filter")
    
    # Get predicted class distribution
    pred_counts = pd.Series(y_pred).value_counts()
    logger.info(f"Predicted class distribution: {pred_counts}")
    
    # Compare predicted vs actual
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_valid, y_pred)
    conf_matrix = confusion_matrix(y_valid, y_pred)
    
    logger.info(f"Classification report:\n{report}")
    logger.info(f"Confusion matrix:\n{conf_matrix}")
    
    # Save feature importances if it's a tree-based model
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Get feature names
        feature_names = X_valid.columns
        
        # Print feature ranking
        logger.info("Feature ranking:")
        for i, idx in enumerate(indices[:20]):  # Top 20 features
            logger.info(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature importances")
        plt.bar(range(20), importances[indices[:20]], align="center")
        plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        
        # Save plot
        output_dir = "output/model_training"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "feature_importances.png"))
        plt.close()
        
        logger.info(f"Feature importances saved to {os.path.join(output_dir, 'feature_importances.png')}")
    
    # Test model loading
    trainer.models = {}  # Clear models
    loaded_model = trainer.get_model("test_signal_filter")
    
    if loaded_model is not None:
        logger.info("Successfully loaded model from disk")
    else:
        logger.error("Failed to load model from disk")

if __name__ == "__main__":
    logger.info("Testing intraday model training components")
    
    # Prepare training data
    features, pair_data, spread_df = prepare_training_data()
    
    if features is not None:
        # Test walk-forward validation
        test_walk_forward_validation(features, pair_data, spread_df)
        
        # Test label generation
        labels = test_label_generation(features, pair_data, spread_df)
        
        if labels is not None:
            # Test data augmentation
            X_balanced, y_balanced = test_data_augmentation(features, labels)
            
            # Test model trainer
            test_model_trainer(features, labels)
    
    logger.info("Tests completed") 