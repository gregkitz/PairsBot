"""
Integration test for the intraday ML system.

This test verifies that all components of the intraday ML system work together correctly,
from data loading to signal generation with ML enhancements and regime-aware parameters.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.ml_enhancements.intraday_integration import IntradayMLSystem
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from src.ml_enhancements.feature_engineering.advanced_features import AdvancedFeatureEngineering
from src.optimization.adaptive_parameter_manager import AdaptiveParameterManager
from src.backtest.intraday_backtest_engine import IntradayBacktestEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data():
    """Create synthetic data for testing."""
    # Generate dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Create price data for two symbols
    symbols = ["GC", "SI"]
    prices = {}
    volumes = {}
    
    # Generate price data with a cointegrated relationship
    base_price = 100 + np.random.normal(0, 1, size=len(date_range)).cumsum()
    
    for i, symbol in enumerate(symbols):
        # Create price with some variation from base
        price = base_price + (i+1) * 10 + np.random.normal(0, 0.5, size=len(date_range))
        prices[symbol] = pd.Series(price, index=date_range)
        
        # Create volume data
        volume = np.abs(np.random.normal(1000, 200, size=len(date_range)))
        volumes[symbol] = pd.Series(volume, index=date_range)
    
    # Create dataframes
    prices_df = pd.DataFrame(prices)
    volumes_df = pd.DataFrame(volumes)
    
    # Calculate spread
    hedge_ratio = 0.5
    spread = prices_df["GC"] - hedge_ratio * prices_df["SI"]
    
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
        'hedge_ratio': pd.Series(hedge_ratio, index=date_range)
    })
    
    return prices_df, spreads_df, volumes_df


@pytest.fixture
def ml_system():
    """Create an instance of the IntradayMLSystem."""
    config = {
        "feature_engineering": {
            "use_advanced_features": True,
            "include_nonlinear": True,
            "include_temporal": True
        },
        "regime_detection": {
            "n_regimes": 3,
            "lookback_window": 60,
            "update_frequency": 60  # minutes
        },
        "parameter_adaptation": {
            "enable_adaptation": True,
            "adaptation_frequency": "daily"
        },
        "model_settings": {
            "signal_filter": {
                "enabled": True,
                "confidence_threshold": 0.6
            },
            "entry_timing": {
                "enabled": True,
                "confidence_threshold": 0.7
            },
            "exit_timing": {
                "enabled": True,
                "confidence_threshold": 0.7
            }
        }
    }
    
    # Create system
    system = IntradayMLSystem(config=config)
    
    return system


@pytest.mark.integration
def test_end_to_end_intraday_ml_workflow(sample_data, ml_system):
    """Test the complete workflow from data to ML-enhanced signals."""
    prices_df, spreads_df, volumes_df = sample_data
    
    # 1. Initialize feature engineering
    feature_eng = AdvancedFeatureEngineering()
    
    # 2. Generate features
    features = feature_eng.generate_advanced_features(
        prices_df=prices_df,
        spreads_df=spreads_df,
        volumes_df=volumes_df
    )
    
    # Verify feature generation
    assert not features.empty, "Features DataFrame should not be empty"
    assert len(features) == len(prices_df), "Features should have same length as input data"
    assert len(features.columns) > 10, "Should generate multiple feature columns"
    
    # 3. Detect market regimes
    regime_detector = MarketRegimeClassifier(n_regimes=3)
    regime_features = regime_detector.calculate_features(prices_df)
    regime_detector.fit(regime_features)
    regimes = regime_detector.predict(regime_features)
    
    # Verify regime detection
    assert not regimes.empty, "Regimes series should not be empty"
    assert len(regimes) == len(prices_df), "Regimes should have same length as input data"
    assert regimes.nunique() > 1, "Should detect multiple regimes"
    
    # 4. Create adaptive parameters
    param_manager = AdaptiveParameterManager(
        config_dict={
            "regime_responses": {
                "default": {
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "stop_loss_std": 2.5,
                    "position_size_factor": 0.8
                }
            }
        }
    )
    
    # 5. Adapt parameters based on regime
    regime_idx, regime_name = param_manager.detect_regime(prices_df)
    adapted_params = param_manager.adapt_parameters(prices_df)
    
    # Verify parameter adaptation
    assert regime_idx is not None, "Should detect a regime index"
    assert regime_name is not None, "Should provide a regime name"
    assert adapted_params, "Should return adapted parameters"
    assert "entry_zscore" in adapted_params, "Should include entry threshold parameter"
    
    # 6. Generate ML-enhanced signals
    # First, get base signals from z-score
    base_signals = (abs(spreads_df['zscore']) > adapted_params.get("entry_zscore", 2.0)).astype(int)
    base_signals = base_signals * np.sign(-spreads_df['zscore'])  # Direction: negative z-score = long
    
    # Apply ML signal enhancements through the ML system
    ml_enhanced_signals = ml_system.enhance_signals(
        base_signals=base_signals,
        features=features,
        regimes=regimes
    )
    
    # Verify ML-enhanced signals
    assert len(ml_enhanced_signals) == len(base_signals), "ML signals should have same length as base signals"
    assert not ml_enhanced_signals.equals(base_signals), "ML signals should differ from base signals"
    
    # 7. Run backtest with ML-enhanced signals
    backtest_engine = IntradayBacktestEngine(
        signals=ml_enhanced_signals,
        prices=prices_df,
        account_size=100000,
        transaction_cost_model={
            "commission_model": "fixed",
            "commission_params": {
                "per_contract": 2.0
            }
        }
    )
    
    # Run backtest
    backtest_results = backtest_engine.run_backtest()
    
    # Verify backtest results
    assert backtest_results is not None, "Should produce backtest results"
    assert "equity_curve" in backtest_results, "Should include equity curve"
    assert "trades" in backtest_results, "Should include trade list"
    assert "performance_metrics" in backtest_results, "Should include performance metrics"
    
    # Check that backtest metrics make sense
    metrics = backtest_results["performance_metrics"]
    assert "total_return" in metrics, "Should calculate total return"
    assert "sharpe_ratio" in metrics, "Should calculate Sharpe ratio"
    assert "max_drawdown" in metrics, "Should calculate max drawdown"
    
    # 8. Verify integration of all components
    # Check that the test flow completed without errors
    logger.info("End-to-end intraday ML workflow test completed successfully")
    logger.info(f"Detected regime: {regime_name}")
    logger.info(f"Adapted parameters: {adapted_params}")
    logger.info(f"Performance metrics: {metrics}")
    
    # If we got this far without exceptions, the test is successful
    assert True


@pytest.mark.integration
def test_regime_specific_parameter_adaptation(sample_data):
    """Test that parameters adapt correctly to different market regimes."""
    prices_df, spreads_df, volumes_df = sample_data
    
    # Create three distinct market regimes in the data
    third = len(prices_df) // 3
    
    # Modify the data to simulate three different regimes
    # 1. Low volatility regime
    prices_df.iloc[:third, :] = prices_df.iloc[:third, :] * 0.2 + prices_df.iloc[0, :] * 0.8
    
    # 2. High volatility regime
    vol_factor = np.linspace(1, 2, third)
    for i in range(third):
        prices_df.iloc[third + i, :] = prices_df.iloc[third, :] * (1 + 0.01 * np.random.randn() * vol_factor[i])
    
    # 3. Trending regime
    trend = np.linspace(0, 1, third)
    prices_df.iloc[2*third:, 0] = prices_df.iloc[2*third, 0] * (1 + trend * 0.2)
    prices_df.iloc[2*third:, 1] = prices_df.iloc[2*third, 1] * (1 + trend * 0.1)
    
    # Create regime detector
    regime_detector = MarketRegimeClassifier(n_regimes=3)
    regime_features = regime_detector.calculate_features(prices_df)
    regime_detector.fit(regime_features)
    regimes = regime_detector.predict(regime_features)
    
    # Verify we have three different regimes
    unique_regimes = regimes.unique()
    assert len(unique_regimes) == 3, "Should detect three different regimes"
    
    # Create parameter manager with regime-specific settings
    param_manager = AdaptiveParameterManager(
        config_dict={
            "regime_responses": {
                "high_volatility": {
                    "entry_zscore": 2.5,
                    "exit_zscore": 0.3,
                    "stop_loss_std": 2.0,
                    "position_size_factor": 0.5
                },
                "low_volatility": {
                    "entry_zscore": 1.8,
                    "exit_zscore": 0.8,
                    "stop_loss_std": 3.0,
                    "position_size_factor": 1.0
                },
                "trending": {
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "stop_loss_std": 2.5,
                    "position_size_factor": 0.8
                }
            }
        }
    )
    
    # Test parameters for each regime
    params_by_regime = {}
    
    # Check first part of data (low volatility)
    low_vol_data = prices_df.iloc[:third, :]
    _, regime_name = param_manager.detect_regime(low_vol_data)
    params = param_manager.adapt_parameters(low_vol_data)
    params_by_regime[regime_name] = params
    
    # Check middle part of data (high volatility)
    high_vol_data = prices_df.iloc[third:2*third, :]
    _, regime_name = param_manager.detect_regime(high_vol_data)
    params = param_manager.adapt_parameters(high_vol_data)
    params_by_regime[regime_name] = params
    
    # Check last part of data (trending)
    trend_data = prices_df.iloc[2*third:, :]
    _, regime_name = param_manager.detect_regime(trend_data)
    params = param_manager.adapt_parameters(trend_data)
    params_by_regime[regime_name] = params
    
    # Verify that parameters differ between regimes
    assert len(params_by_regime) >= 2, "Should detect at least 2 different regimes"
    
    # Check that entry thresholds are different for different regimes
    entry_thresholds = [params.get("entry_zscore", 0) for params in params_by_regime.values()]
    assert len(set(entry_thresholds)) > 1, "Entry thresholds should differ between regimes"
    
    # Verify high volatility regime has higher entry threshold and smaller position size
    if "high_volatility" in params_by_regime:
        high_vol_params = params_by_regime["high_volatility"]
        for regime, params in params_by_regime.items():
            if regime != "high_volatility":
                assert high_vol_params.get("entry_zscore", 0) >= params.get("entry_zscore", 0), \
                    "High volatility regime should have higher entry threshold"
                assert high_vol_params.get("position_size_factor", 1) <= params.get("position_size_factor", 1), \
                    "High volatility regime should have smaller position size"
    
    logger.info("Regime-specific parameter adaptation test completed successfully")
    logger.info(f"Detected regimes: {list(params_by_regime.keys())}")
    for regime, params in params_by_regime.items():
        logger.info(f"Parameters for {regime}: {params}")


@pytest.mark.integration
def test_model_retraining_trigger(sample_data, ml_system):
    """Test that model retraining is triggered correctly based on performance or drift."""
    from src.ml_enhancements.model_retraining import ModelRetrainingManager, auto_retrain_if_needed
    
    prices_df, spreads_df, volumes_df = sample_data
    
    # Split data for initial training and new data
    split_idx = int(len(prices_df) * 0.7)
    
    train_prices = prices_df.iloc[:split_idx]
    train_spreads = spreads_df.iloc[:split_idx]
    train_volumes = volumes_df.iloc[:split_idx] if volumes_df is not None else None
    
    new_prices = prices_df.iloc[split_idx:]
    new_spreads = spreads_df.iloc[split_idx:]
    new_volumes = volumes_df.iloc[split_idx:] if volumes_df is not None else None
    
    # Generate features
    feature_eng = AdvancedFeatureEngineering()
    
    train_features = feature_eng.generate_advanced_features(
        prices_df=train_prices,
        spreads_df=train_spreads,
        volumes_df=train_volumes
    )
    
    new_features = feature_eng.generate_advanced_features(
        prices_df=new_prices,
        spreads_df=new_spreads,
        volumes_df=new_volumes
    )
    
    # Create labels (entry signals based on z-score)
    train_labels = (abs(train_spreads['zscore']) > 2.0).astype(int)
    new_labels = (abs(new_spreads['zscore']) > 2.0).astype(int)
    
    # Define a simple training function
    def train_model(X, y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        return model
    
    # Initialize retraining manager
    manager = ModelRetrainingManager(
        models_dir="models/intraday/test",
        output_dir="output/test_retraining"
    )
    
    # Train initial model
    initial_model = train_model(train_features, train_labels)
    
    # Calculate initial metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    val_pred = initial_model.predict(train_features)
    
    initial_metrics = {
        "accuracy": accuracy_score(train_labels, val_pred),
        "f1": f1_score(train_labels, val_pred, zero_division=0)
    }
    
    # Register model
    model_id = manager.register_model(
        model_name="test_signal_filter",
        model_type="classification",
        model_path="models/intraday/test/initial_model.joblib",
        initial_metrics=initial_metrics
    )
    
    # Create degraded metrics to trigger retraining
    degraded_metrics = {
        "accuracy": initial_metrics["accuracy"] * 0.7,  # 30% worse
        "f1": initial_metrics["f1"] * 0.7  # 30% worse
    }
    
    # Update with degraded metrics
    manager.update_model_metrics(model_id, degraded_metrics, event="performance_degradation")
    
    # Check if retraining is needed - should be true with degraded performance
    retrain_needed = manager.check_retraining_needed(model_id)
    assert retrain_needed, "Should trigger retraining with degraded performance"
    
    # Test auto_retrain_if_needed
    retrained, new_model_id = auto_retrain_if_needed(
        model_id=model_id,
        manager=manager,
        training_func=train_model,
        X_new=new_features,
        y_new=new_labels,
        validation_data=(train_features, train_labels)
    )
    
    assert retrained, "Should have retrained the model"
    assert new_model_id != model_id, "Should have a new model ID after retraining"
    
    # Verify new model exists in registry
    assert new_model_id in manager.model_registry, "New model should be in registry"
    
    # Verify that new model has better metrics than degraded ones
    new_metrics = manager.model_registry[new_model_id]["current_metrics"]
    assert new_metrics["accuracy"] > degraded_metrics["accuracy"], "New model should have better accuracy"
    
    logger.info("Model retraining test completed successfully")
    logger.info(f"Initial metrics: {initial_metrics}")
    logger.info(f"Degraded metrics: {degraded_metrics}")
    logger.info(f"New model metrics: {new_metrics}")


if __name__ == "__main__":
    # For running tests directly with pytest
    pytest.main(["-xvs", __file__]) 