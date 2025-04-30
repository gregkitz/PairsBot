"""
Test script for IntradayFeatureEngineering.

This script demonstrates how to generate and analyze intraday features for ML models.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_processor import IntradayDataProcessor
from src.ml_enhancements.feature_engineering import IntradayFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_generation():
    """Test feature generation for intraday data."""
    data_dir = "data/processed"
    output_dir = "output/feature_analysis"
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Initialize data processor
    data_processor = IntradayDataProcessor(data_dir=data_dir)
    
    # Get available symbols
    symbols = data_processor.get_available_symbols()
    logger.info(f"Available symbols: {symbols}")
    
    if len(symbols) < 2:
        logger.error("Not enough symbols available for pair analysis")
        return
    
    # Create a test pair
    pair_id = f"{symbols[0]}_{symbols[1]}"
    logger.info(f"Generating features for pair: {pair_id}")
    
    # Set test date range
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    
    # Load pair data
    pair_data = data_processor.load_pair_data(pair_id, start_date, end_date, timeframe="5min")
    
    if pair_data is None:
        logger.error("Failed to load pair data")
        return
    
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
        return
    
    # Initialize feature engineering
    feature_engineering = IntradayFeatureEngineering(
        base_lookback=20,
        custom_windows=[5, 10, 20, 50],
        output_dir=output_dir
    )
    
    # Generate features
    features = feature_engineering.generate_intraday_features(
        prices_df=pair_data['prices'],
        spreads_df=spread_df,
        volumes_df=pair_data['volumes'],
        include_time_features=True,
        include_liquidity_features=True,
        include_microstructure=True,
        scale_output=False  # Don't scale for better interpretability
    )
    
    # Log feature info
    logger.info(f"Generated {len(features.columns)} features for {len(features)} data points")
    
    # Show sample features by category
    feature_categories = {
        "Basic Spread Features": ["spread", "zscore", "hedge_ratio"],
        "Time Features": [col for col in features.columns if any(x in col for x in ["hour", "minute", "day_", "session"])],
        "Volatility Features": [col for col in features.columns if "vol" in col],
        "Correlation Features": [col for col in features.columns if "corr" in col],
        "Mean Reversion Features": [col for col in features.columns if any(x in col for x in ["reversion", "zscore_cross", "extreme"])],
        "Regime Features": [col for col in features.columns if "regime" in col]
    }
    
    for category, cols in feature_categories.items():
        present_cols = [col for col in cols if col in features.columns]
        if present_cols:
            logger.info(f"{category}: {len(present_cols)} features - {present_cols[:5]} ...")
    
    # Save features to CSV
    feature_file = os.path.join(output_dir, f"{pair_id}_features.csv")
    features.to_csv(feature_file)
    logger.info(f"Saved {len(features)} rows of features to {feature_file}")
    
    return features, pair_data

def test_feature_importance(features, pair_data):
    """Test feature importance analysis."""
    if features is None or pair_data is None:
        logger.error("No features or pair data available for importance analysis")
        return
    
    output_dir = "output/feature_analysis"
    
    # Initialize feature engineering
    feature_engineering = IntradayFeatureEngineering(output_dir=output_dir)
    
    # Create a synthetic target for testing
    # Here we'll use future return of the first instrument in the pair as target
    symbol1 = pair_data['prices'].columns[0]
    symbol1_returns = pair_data['prices'][symbol1].pct_change(5).shift(-5)  # 5-period future return
    
    # Align target with features
    common_idx = features.index.intersection(symbol1_returns.index)
    features_subset = features.loc[common_idx]
    target = symbol1_returns.loc[common_idx]
    
    # Drop NaN values
    valid_idx = ~target.isna()
    features_subset = features_subset.loc[valid_idx]
    target = target.loc[valid_idx]
    
    # Analyze feature importance
    importance = feature_engineering.analyze_feature_importance(
        features=features_subset,
        target=target,
        method='mutual_info',
        output_file=os.path.join(output_dir, "feature_importance.png"),
        n_features=30
    )
    
    # Display top features
    top_features = list(importance.items())[:10]
    logger.info("Top 10 features by importance:")
    for feature, score in top_features:
        logger.info(f"  {feature}: {score:.6f}")
    
    # Test feature selection
    selected_features = feature_engineering.select_important_features(
        features=features_subset,
        importance_threshold=np.percentile(list(importance.values()), 75),  # Top 25% of features
        max_features=20
    )
    
    logger.info(f"Selected {len(selected_features.columns)} important features")
    
    return importance

def visualize_features(features, pair_data, output_dir="output/feature_analysis"):
    """Visualize selected features."""
    if features is None or pair_data is None:
        logger.error("No features or pair data available for visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Get the pair symbols
    symbol1, symbol2 = pair_data['prices'].columns
    
    # 1. Plot prices
    pair_data['prices'].plot(ax=axes[0])
    axes[0].set_title(f"Prices: {symbol1} and {symbol2}")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Plot spread and z-score
    if 'spread' in features.columns and 'zscore' in features.columns:
        ax_spread = axes[1]
        ax_zscore = ax_spread.twinx()
        
        ax_spread.plot(features.index, features['spread'], 'b-', label='Spread')
        ax_zscore.plot(features.index, features['zscore'], 'r-', label='Z-Score')
        
        ax_spread.set_ylabel('Spread', color='b')
        ax_zscore.set_ylabel('Z-Score', color='r')
        
        # Add z-score bands
        ax_zscore.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
        ax_zscore.axhline(y=2, color='red', linestyle='--', alpha=0.5)
        ax_zscore.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
        
        ax_spread.set_title(f"Spread and Z-Score")
        ax_spread.grid(True, alpha=0.3)
        
        # Add combined legend
        lines_spread, labels_spread = ax_spread.get_legend_handles_labels()
        lines_zscore, labels_zscore = ax_zscore.get_legend_handles_labels()
        ax_spread.legend(lines_spread + lines_zscore, labels_spread + labels_zscore, loc='upper right')
    
    # 3. Plot time-based features
    if 'session_progress' in features.columns:
        time_features = ['session_progress']
        
        # Add session indicators if available
        if 'session_open' in features.columns:
            time_features.extend(['session_open', 'session_mid', 'session_close'])
        
        features[time_features].plot(ax=axes[2])
        axes[2].set_title("Time-Based Features")
        axes[2].grid(True, alpha=0.3)
    
    # 4. Plot volatility features
    vol_features = [col for col in features.columns if 'vol_' in col and not 'vol_imbalance' in col][:5]
    if vol_features:
        features[vol_features].plot(ax=axes[3])
        axes[3].set_title("Volatility Features")
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "feature_visualization.png"))
    plt.close()
    
    logger.info(f"Saved feature visualization to {os.path.join(output_dir, 'feature_visualization.png')}")

if __name__ == "__main__":
    logger.info("Testing IntradayFeatureEngineering module")
    
    # Test feature generation
    features, pair_data = test_feature_generation()
    
    # Test feature importance
    importance = test_feature_importance(features, pair_data)
    
    # Visualize features
    visualize_features(features, pair_data)
    
    logger.info("Tests completed") 