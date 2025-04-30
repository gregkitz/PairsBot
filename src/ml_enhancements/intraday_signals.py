"""
Intraday Signal Enhancement Module

This module provides ML-based signal enhancements for intraday pairs trading,
focusing on improving entry/exit timing, filtering false signals, and adapting
to changing market conditions within the trading session.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import logging
import joblib
import os
from datetime import datetime, time, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntradaySignalEnhancer:
    """
    Enhances statistical arbitrage signals with machine learning for intraday trading.
    """
    
    def __init__(self, config=None):
        """
        Initialize the intraday signal enhancer.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for signal enhancement
        """
        self.config = config or {
            "feature_lookback": 20,         # Lookback window for feature calculation
            "models_dir": "models/intraday", # Directory to save/load models
            "min_training_samples": 500,     # Minimum samples required for training
            "prediction_threshold": 0.6,     # Probability threshold for signal confirmation
            "retrain_frequency": "weekly",   # How often to retrain models
            "use_rsi_filter": True,          # Use RSI confirmation filter
            "use_volume_filter": True,       # Use volume confirmation filter
            "use_volatility_filter": True,   # Use volatility confirmation filter
            "enable_ml_filtering": True,     # Use ML for signal filtering
            "enable_ml_timing": True,        # Use ML for entry/exit timing
            "enable_ml_adaptation": True     # Use ML for intraday adaptation
        }
        
        # Initialize models
        self.signal_filter_model = None
        self.entry_timing_model = None
        self.exit_timing_model = None
        self.volume_prediction_model = None
        self.correlation_model = None
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Create models directory
        os.makedirs(self.config["models_dir"], exist_ok=True)
        
        # Track model performance
        self.model_metrics = {}
    
    def calculate_features(self, prices_df, spreads_df, volumes_df=None):
        """
        Calculate features for ML signal enhancement.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data for each instrument
        spreads_df : pd.DataFrame
            DataFrame with spread data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data for each instrument
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated features
        """
        features = pd.DataFrame(index=spreads_df.index)
        lookback = self.config["feature_lookback"]
        
        # 1. Spread features
        if 'zscore' in spreads_df.columns:
            features['zscore'] = spreads_df['zscore']
            features['zscore_abs'] = np.abs(spreads_df['zscore'])
            
            # Z-score derivatives
            features['zscore_change'] = spreads_df['zscore'].diff()
            features['zscore_acceleration'] = features['zscore_change'].diff()
            
            # Z-score momentum
            features['zscore_mom_5'] = spreads_df['zscore'].diff(5)
            features['zscore_mom_10'] = spreads_df['zscore'].diff(10)
            
            # Z-score smoothing
            features['zscore_ma5'] = spreads_df['zscore'].rolling(5).mean()
            features['zscore_ma10'] = spreads_df['zscore'].rolling(10).mean()
            
            # Z-score volatility
            features['zscore_vol_5'] = spreads_df['zscore'].rolling(5).std()
            features['zscore_vol_10'] = spreads_df['zscore'].rolling(10).std()
            
            # Z-score extremes
            features['zscore_max_5'] = spreads_df['zscore'].rolling(5).max()
            features['zscore_min_5'] = spreads_df['zscore'].rolling(5).min()
        
        # 2. Time features for intraday patterns
        if isinstance(spreads_df.index, pd.DatetimeIndex):
            # Hour of day
            features['hour'] = spreads_df.index.hour
            
            # Minute of hour
            features['minute'] = spreads_df.index.minute
            
            # Time of day (decimal hours)
            features['tod'] = features['hour'] + features['minute'] / 60.0
            
            # Session progress (0-1 range)
            market_open = 9.5  # 9:30 AM
            market_close = 16.0  # 4:00 PM
            session_length = market_close - market_open
            features['session_progress'] = (features['tod'] - market_open) / session_length
            features['session_progress'] = features['session_progress'].clip(0, 1)
            
            # Session phase (early, mid, late)
            features['early_session'] = (features['session_progress'] < 0.33).astype(int)
            features['mid_session'] = ((features['session_progress'] >= 0.33) & 
                                      (features['session_progress'] < 0.67)).astype(int)
            features['late_session'] = (features['session_progress'] >= 0.67).astype(int)
            
            # Day of week
            features['day_of_week'] = spreads_df.index.dayofweek
        
        # 3. Instrument-specific features
        for col in prices_df.columns:
            prices = prices_df[col]
            
            # Price momentum
            features[f'{col}_mom_5'] = prices.pct_change(5)
            features[f'{col}_mom_10'] = prices.pct_change(10)
            
            # RSI
            # Use the internal implementation instead of TA-Lib
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features[f'{col}_rsi'] = 100 - (100 / (1 + rs))
            
            # Volatility
            features[f'{col}_vol_5'] = prices.pct_change().rolling(5).std()
            features[f'{col}_vol_10'] = prices.pct_change().rolling(10).std()
            
            # Volume features if available
            if volumes_df is not None and col in volumes_df.columns:
                vol = volumes_df[col]
                features[f'{col}_vol_ratio'] = vol / vol.rolling(20).mean()
                features[f'{col}_vol_change'] = vol.pct_change(5)
        
        # 4. Correlation features
        price_returns = prices_df.pct_change()
        
        if len(prices_df.columns) >= 2:
            # Rolling correlation between pairs
            for i, col1 in enumerate(prices_df.columns):
                for j, col2 in enumerate(prices_df.columns):
                    if i < j:  # Only calculate each pair once
                        corr = price_returns[col1].rolling(lookback).corr(price_returns[col2])
                        features[f'corr_{col1}_{col2}'] = corr
        
        # 5. Mean reversion strength
        if 'zscore' in spreads_df.columns:
            # Half-life calculation (simplified for feature)
            zscore_lag = spreads_df['zscore'].shift(1)
            zscore_diff = spreads_df['zscore'] - zscore_lag
            
            # Regression of diff on lag to get mean reversion speed
            # We'll calculate this over rolling windows
            for window in [20, 50]:
                if len(zscore_diff) > window:
                    mr_speed = []
                    
                    for i in range(window, len(zscore_diff)):
                        y = zscore_diff.iloc[i-window:i].values
                        x = zscore_lag.iloc[i-window:i].values
                        x = np.reshape(x, (len(x), 1))
                        
                        # Add constant
                        x_with_const = np.column_stack([np.ones(len(x)), x])
                        
                        try:
                            # OLS regression
                            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                            mr_speed.append(beta[1])
                        except:
                            mr_speed.append(np.nan)
                    
                    # Pad with NaNs for the initial periods
                    mr_speed = [np.nan] * window + mr_speed
                    features[f'mr_speed_{window}'] = mr_speed
        
        # Drop rows with NaN values
        features = features.dropna()
        
        return features 

    def train_signal_filter_model(self, features, labels):
        """
        Train a model to filter false signals.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        labels : pd.Series
            Series with true/false labels for successful signals
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        logger.info("Training signal filter model")
        
        if len(features) < self.config["min_training_samples"]:
            logger.warning(f"Not enough samples for training. Got {len(features)}, need {self.config['min_training_samples']}")
            return False
        
        # Split data into training and validation sets
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]
            
            # Only use the last split for validation
            pass
        
        # Scale features
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Train model
        self.signal_filter_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        
        self.signal_filter_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_preds = self.signal_filter_model.predict(X_val_scaled)
        val_probs = self.signal_filter_model.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        
        # Store metrics
        self.model_metrics["signal_filter"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": len(X_train)
        }
        
        logger.info(f"Signal filter model: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "signal_filter.joblib")
        scaler_path = os.path.join(self.config["models_dir"], "signal_filter_scaler.joblib")
        
        joblib.dump(self.signal_filter_model, model_path)
        joblib.dump(self.feature_scaler, scaler_path)
        
        logger.info(f"Saved signal filter model to {model_path}")
        
        return True
    
    def train_entry_timing_model(self, features, optimal_entries):
        """
        Train a model to improve entry timing.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        optimal_entries : pd.Series
            Series with 1 for optimal entry points, 0 otherwise
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        logger.info("Training entry timing model")
        
        if len(features) < self.config["min_training_samples"]:
            logger.warning(f"Not enough samples for training. Got {len(features)}, need {self.config['min_training_samples']}")
            return False
        
        # Split data into training and validation sets
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = optimal_entries.iloc[train_idx], optimal_entries.iloc[val_idx]
            
            # Only use the last split for validation
            pass
        
        # Scale features
        entry_scaler = StandardScaler()
        entry_scaler.fit(X_train)
        X_train_scaled = entry_scaler.transform(X_train)
        X_val_scaled = entry_scaler.transform(X_val)
        
        # Train model - higher weight on positive class due to imbalance
        self.entry_timing_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        self.entry_timing_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_preds = self.entry_timing_model.predict(X_val_scaled)
        val_probs = self.entry_timing_model.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        
        # Store metrics
        self.model_metrics["entry_timing"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": len(X_train)
        }
        
        logger.info(f"Entry timing model: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "entry_timing.joblib")
        scaler_path = os.path.join(self.config["models_dir"], "entry_timing_scaler.joblib")
        
        joblib.dump(self.entry_timing_model, model_path)
        joblib.dump(entry_scaler, scaler_path)
        
        logger.info(f"Saved entry timing model to {model_path}")
        
        return True
    
    def train_exit_timing_model(self, features, optimal_exits):
        """
        Train a model to improve exit timing.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        optimal_exits : pd.Series
            Series with 1 for optimal exit points, 0 otherwise
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        logger.info("Training exit timing model")
        
        if len(features) < self.config["min_training_samples"]:
            logger.warning(f"Not enough samples for training. Got {len(features)}, need {self.config['min_training_samples']}")
            return False
        
        # Split data into training and validation sets
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = optimal_exits.iloc[train_idx], optimal_exits.iloc[val_idx]
            
            # Only use the last split for validation
            pass
        
        # Scale features
        exit_scaler = StandardScaler()
        exit_scaler.fit(X_train)
        X_train_scaled = exit_scaler.transform(X_train)
        X_val_scaled = exit_scaler.transform(X_val)
        
        # Train model - higher weight on positive class due to imbalance
        self.exit_timing_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        self.exit_timing_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_preds = self.exit_timing_model.predict(X_val_scaled)
        val_probs = self.exit_timing_model.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)
        
        # Store metrics
        self.model_metrics["exit_timing"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": len(X_train)
        }
        
        logger.info(f"Exit timing model: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "exit_timing.joblib")
        scaler_path = os.path.join(self.config["models_dir"], "exit_timing_scaler.joblib")
        
        joblib.dump(self.exit_timing_model, model_path)
        joblib.dump(exit_scaler, scaler_path)
        
        logger.info(f"Saved exit timing model to {model_path}")
        
        return True
    
    def train_volume_prediction_model(self, features, future_volumes):
        """
        Train a model to predict future volume patterns.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        future_volumes : pd.Series
            Series with future volume values
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        logger.info("Training volume prediction model")
        
        if len(features) < self.config["min_training_samples"]:
            logger.warning(f"Not enough samples for training. Got {len(features)}, need {self.config['min_training_samples']}")
            return False
        
        # Split data into training and validation sets
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = future_volumes.iloc[train_idx], future_volumes.iloc[val_idx]
            
            # Only use the last split for validation
            pass
        
        # Scale features
        volume_scaler = StandardScaler()
        volume_scaler.fit(X_train)
        X_train_scaled = volume_scaler.transform(X_train)
        X_val_scaled = volume_scaler.transform(X_val)
        
        # Train model
        self.volume_prediction_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        self.volume_prediction_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_preds = self.volume_prediction_model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        
        # Store metrics
        self.model_metrics["volume_prediction"] = {
            "rmse": rmse,
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": len(X_train)
        }
        
        logger.info(f"Volume prediction model: RMSE={rmse:.4f}")
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "volume_prediction.joblib")
        scaler_path = os.path.join(self.config["models_dir"], "volume_prediction_scaler.joblib")
        
        joblib.dump(self.volume_prediction_model, model_path)
        joblib.dump(volume_scaler, scaler_path)
        
        logger.info(f"Saved volume prediction model to {model_path}")
        
        return True
    
    def train_correlation_prediction_model(self, features, future_correlation):
        """
        Train a model to predict future correlation changes.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
        future_correlation : pd.Series
            Series with future correlation values
            
        Returns:
        --------
        bool
            True if training was successful, False otherwise
        """
        logger.info("Training correlation prediction model")
        
        if len(features) < self.config["min_training_samples"]:
            logger.warning(f"Not enough samples for training. Got {len(features)}, need {self.config['min_training_samples']}")
            return False
        
        # Split data into training and validation sets
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = future_correlation.iloc[train_idx], future_correlation.iloc[val_idx]
            
            # Only use the last split for validation
            pass
        
        # Scale features
        corr_scaler = StandardScaler()
        corr_scaler.fit(X_train)
        X_train_scaled = corr_scaler.transform(X_train)
        X_val_scaled = corr_scaler.transform(X_val)
        
        # Train model
        self.correlation_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        self.correlation_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_preds = self.correlation_model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        
        # Store metrics
        self.model_metrics["correlation_prediction"] = {
            "rmse": rmse,
            "trained_date": datetime.now().strftime("%Y-%m-%d"),
            "samples": len(X_train)
        }
        
        logger.info(f"Correlation prediction model: RMSE={rmse:.4f}")
        
        # Save model
        model_path = os.path.join(self.config["models_dir"], "correlation_prediction.joblib")
        scaler_path = os.path.join(self.config["models_dir"], "correlation_prediction_scaler.joblib")
        
        joblib.dump(self.correlation_model, model_path)
        joblib.dump(corr_scaler, scaler_path)
        
        logger.info(f"Saved correlation prediction model to {model_path}")
        
        return True
    
    def load_models(self):
        """
        Load pre-trained models.
        
        Returns:
        --------
        bool
            True if all models were loaded successfully, False otherwise
        """
        try:
            logger.info("Loading pre-trained models")
            
            # Signal filter model
            model_path = os.path.join(self.config["models_dir"], "signal_filter.joblib")
            scaler_path = os.path.join(self.config["models_dir"], "signal_filter_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.signal_filter_model = joblib.load(model_path)
                    self.feature_scaler = joblib.load(scaler_path)
                    logger.info("Loaded signal filter model")
                except Exception as e:
                    logger.warning(f"Could not load signal filter model: {e}")
                    self.signal_filter_model = None
            
            # Entry timing model
            model_path = os.path.join(self.config["models_dir"], "entry_timing.joblib")
            scaler_path = os.path.join(self.config["models_dir"], "entry_timing_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.entry_timing_model = joblib.load(model_path)
                    # Use the same scaler for all models
                    logger.info("Loaded entry timing model")
                except Exception as e:
                    logger.warning(f"Could not load entry timing model: {e}")
                    self.entry_timing_model = None
            
            # Exit timing model
            model_path = os.path.join(self.config["models_dir"], "exit_timing.joblib")
            scaler_path = os.path.join(self.config["models_dir"], "exit_timing_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.exit_timing_model = joblib.load(model_path)
                    logger.info("Loaded exit timing model")
                except Exception as e:
                    logger.warning(f"Could not load exit timing model: {e}")
                    self.exit_timing_model = None
            
            # Volume prediction model
            model_path = os.path.join(self.config["models_dir"], "volume_prediction.joblib")
            scaler_path = os.path.join(self.config["models_dir"], "volume_prediction_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.volume_prediction_model = joblib.load(model_path)
                    logger.info("Loaded volume prediction model")
                except Exception as e:
                    logger.warning(f"Could not load volume prediction model: {e}")
                    self.volume_prediction_model = None
            
            # Correlation prediction model
            model_path = os.path.join(self.config["models_dir"], "correlation_prediction.joblib")
            scaler_path = os.path.join(self.config["models_dir"], "correlation_prediction_scaler.joblib")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.correlation_model = joblib.load(model_path)
                    logger.info("Loaded correlation prediction model")
                except Exception as e:
                    logger.warning(f"Could not load correlation prediction model: {e}")
                    self.correlation_model = None
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Initialize all models to None to ensure we have a fallback
            self.signal_filter_model = None
            self.entry_timing_model = None
            self.exit_timing_model = None
            self.volume_prediction_model = None
            self.correlation_model = None
            return False
    
    def predict_signal_quality(self, features):
        """
        Predict quality of the signal (probability of being profitable).
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.Series
            Series with signal quality prediction (probability)
        """
        if self.signal_filter_model is None:
            logger.warning("Signal filter model not available")
            # Return default values (all 0.5 means neutral)
            return pd.Series(0.5, index=features.index)
        
        try:
            # Get expected feature names from the model
            if hasattr(self.feature_scaler, 'feature_names_in_'):
                expected_feature_names = self.feature_scaler.feature_names_in_
            else:
                # Use a fallback approach - infer from number of features
                n_features = self.feature_scaler.n_features_in_
                logger.warning(f"Feature names not available, using fallback with {n_features} features")
                # Just use the columns we have, but make sure we have the right number
                if features.shape[1] < n_features:
                    logger.error(f"Not enough features: have {features.shape[1]}, need {n_features}")
                    return pd.Series(0.5, index=features.index)
                expected_feature_names = features.columns[:n_features]
            
            # Check which expected features are missing
            missing_features = [feat for feat in expected_feature_names if feat not in features.columns]
            
            # Create a new feature dataframe with the expected columns
            adapted_features = pd.DataFrame(index=features.index)
            
            # For missing features, use zeros
            for feat in expected_feature_names:
                if feat in features.columns:
                    adapted_features[feat] = features[feat]
                else:
                    logger.warning(f"Missing feature: {feat}, using zero values")
                    adapted_features[feat] = 0.0
            
            # Scale features
            X_scaled = self.feature_scaler.transform(adapted_features)
            
            # Apply prediction model
            probabilities = self.signal_filter_model.predict_proba(X_scaled)
            
            # Return probability of positive class
            return pd.Series(probabilities[:, 1], index=features.index)
        except Exception as e:
            logger.error(f"Error in signal quality prediction: {e}")
            # Return neutral probabilities on error
            return pd.Series(0.5, index=features.index)
    
    def predict_optimal_entry(self, features):
        """
        Predict optimal entry points using the trained model.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.Series
            Series with entry quality prediction (probability)
        """
        if self.entry_timing_model is None:
            logger.warning("Entry timing model not available")
            # Return default values (all 0.5 means neutral)
            return pd.Series(0.5, index=features.index)
        
        try:
            # Get expected feature names from the model - use the same scaler as signal filter
            if hasattr(self.feature_scaler, 'feature_names_in_'):
                expected_feature_names = self.feature_scaler.feature_names_in_
            else:
                # Use a fallback approach - infer from number of features
                n_features = self.feature_scaler.n_features_in_
                expected_feature_names = features.columns[:n_features]
            
            # Create a new feature dataframe with the expected columns
            adapted_features = pd.DataFrame(index=features.index)
            
            # For missing features, use zeros
            for feat in expected_feature_names:
                if feat in features.columns:
                    adapted_features[feat] = features[feat]
                else:
                    adapted_features[feat] = 0.0
            
            # Scale features
            X_scaled = self.feature_scaler.transform(adapted_features)
            
            # Apply prediction model
            probabilities = self.entry_timing_model.predict_proba(X_scaled)
            
            # Return probability of positive class
            return pd.Series(probabilities[:, 1], index=features.index)
        except Exception as e:
            logger.error(f"Error in entry timing prediction: {e}")
            # Return neutral probabilities on error
            return pd.Series(0.5, index=features.index)
    
    def predict_optimal_exit(self, features):
        """
        Predict optimal exit points using the trained model.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.Series
            Series with exit quality prediction (probability)
        """
        if self.exit_timing_model is None:
            logger.warning("Exit timing model not available")
            # Return default values (all 0.5 means neutral)
            return pd.Series(0.5, index=features.index)
        
        try:
            # Get expected feature names from the model - use the same scaler as signal filter
            if hasattr(self.feature_scaler, 'feature_names_in_'):
                expected_feature_names = self.feature_scaler.feature_names_in_
            else:
                # Use a fallback approach - infer from number of features
                n_features = self.feature_scaler.n_features_in_
                expected_feature_names = features.columns[:n_features]
            
            # Create a new feature dataframe with the expected columns
            adapted_features = pd.DataFrame(index=features.index)
            
            # For missing features, use zeros
            for feat in expected_feature_names:
                if feat in features.columns:
                    adapted_features[feat] = features[feat]
                else:
                    adapted_features[feat] = 0.0
            
            # Scale features
            X_scaled = self.feature_scaler.transform(adapted_features)
            
            # Apply prediction model
            probabilities = self.exit_timing_model.predict_proba(X_scaled)
            
            # Return probability of positive class
            return pd.Series(probabilities[:, 1], index=features.index)
        except Exception as e:
            logger.error(f"Error in exit timing prediction: {e}")
            # Return neutral probabilities on error
            return pd.Series(0.5, index=features.index)
    
    def predict_volume_pattern(self, features):
        """
        Predict future volume patterns.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.Series
            Series with volume predictions (relative to average)
        """
        if self.volume_prediction_model is None:
            logger.warning("Volume prediction model not available")
            # Return default values (all 1.0 means average volume)
            return pd.Series(1.0, index=features.index)
        
        try:
            # Get expected feature names - using same approach as other methods
            if hasattr(self.feature_scaler, 'feature_names_in_'):
                expected_feature_names = self.feature_scaler.feature_names_in_
            else:
                # Use a fallback approach - infer from number of features
                n_features = self.feature_scaler.n_features_in_
                expected_feature_names = features.columns[:n_features]
            
            # Create a new feature dataframe with the expected columns
            adapted_features = pd.DataFrame(index=features.index)
            
            # For missing features, use zeros
            for feat in expected_feature_names:
                if feat in features.columns:
                    adapted_features[feat] = features[feat]
                else:
                    adapted_features[feat] = 0.0
            
            # Scale features
            X_scaled = self.feature_scaler.transform(adapted_features)
            
            # Apply prediction model
            predictions = self.volume_prediction_model.predict(X_scaled)
            
            # Return predictions
            return pd.Series(predictions, index=features.index)
        except Exception as e:
            logger.error(f"Error in volume prediction: {e}")
            # Return neutral predictions on error
            return pd.Series(1.0, index=features.index)
    
    def predict_correlation_change(self, features):
        """
        Predict future correlation changes.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with features
            
        Returns:
        --------
        pd.Series
            Series with correlation change predictions
        """
        if self.correlation_model is None:
            logger.warning("Correlation prediction model not available")
            # Return default values (all 0 means no change expected)
            return pd.Series(0.0, index=features.index)
        
        try:
            # Get expected feature names - using same approach as other methods
            if hasattr(self.feature_scaler, 'feature_names_in_'):
                expected_feature_names = self.feature_scaler.feature_names_in_
            else:
                # Use a fallback approach - infer from number of features
                n_features = self.feature_scaler.n_features_in_
                expected_feature_names = features.columns[:n_features]
            
            # Create a new feature dataframe with the expected columns
            adapted_features = pd.DataFrame(index=features.index)
            
            # For missing features, use zeros
            for feat in expected_feature_names:
                if feat in features.columns:
                    adapted_features[feat] = features[feat]
                else:
                    adapted_features[feat] = 0.0
            
            # Scale features
            X_scaled = self.feature_scaler.transform(adapted_features)
            
            # Apply prediction model
            predictions = self.correlation_model.predict(X_scaled)
            
            # Return predictions
            return pd.Series(predictions, index=features.index)
        except Exception as e:
            logger.error(f"Error in correlation prediction: {e}")
            # Return neutral predictions on error
            return pd.Series(0.0, index=features.index)
    
    def enhance_signals(self, original_signals, features, prices_df=None, volumes_df=None):
        """
        Enhance trading signals using ML models and filters.
        
        Parameters:
        -----------
        original_signals : pd.Series or pd.DataFrame
            Series or DataFrame with original signals (1 for long, -1 for short, 0 for no position)
        features : pd.DataFrame
            DataFrame with calculated features
        prices_df : pd.DataFrame, optional
            DataFrame with price data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
            
        Returns:
        --------
        pd.Series
            Series with enhanced signals
        pd.DataFrame
            DataFrame with signal quality metrics
        """
        # Ensure features and original_signals have the same index
        common_index = original_signals.index.intersection(features.index)
        
        if len(common_index) == 0:
            logger.warning("No common timestamps between signals and features")
            return original_signals.copy(), pd.DataFrame(index=original_signals.index)
        
        # Filter both to contain only common timestamps
        features = features.loc[common_index]
        
        # Initialize enhanced signals with original signals (maintain original index)
        enhanced_signals = original_signals.copy()
        
        # Initialize metrics
        metrics = pd.DataFrame(index=original_signals.index)
        
        # Handle the case where original_signals is a DataFrame with multiple columns
        if isinstance(original_signals, pd.DataFrame) and original_signals.shape[1] > 1:
            if 'signal' in original_signals.columns:
                # Extract just the signal column
                signal_series = original_signals['signal']
                metrics['original_signal'] = signal_series
                original_signals_filtered = signal_series.loc[common_index]
                enhanced_signals_filtered = signal_series.loc[common_index].copy()
            else:
                # Use the first column as the signal
                signal_series = original_signals.iloc[:, 0]
                metrics['original_signal'] = signal_series
                original_signals_filtered = signal_series.loc[common_index]
                enhanced_signals_filtered = signal_series.loc[common_index].copy()
        else:
            # Original implementation for when original_signals is a Series
            metrics['original_signal'] = original_signals
            original_signals_filtered = original_signals.loc[common_index]
            enhanced_signals_filtered = enhanced_signals.loc[common_index]
        
        # Align prices_df if provided
        if prices_df is not None:
            common_price_index = common_index.intersection(prices_df.index)
            if len(common_price_index) > 0:
                prices_df = prices_df.loc[common_price_index]
                features = features.loc[common_price_index]
                common_index = common_price_index
                original_signals_filtered = original_signals_filtered.loc[common_price_index]
                enhanced_signals_filtered = enhanced_signals_filtered.loc[common_price_index]
        
        # Align volumes_df if provided
        if volumes_df is not None:
            common_volume_index = common_index.intersection(volumes_df.index)
            if len(common_volume_index) > 0:
                volumes_df = volumes_df.loc[common_volume_index]
                features = features.loc[common_volume_index]
                common_index = common_volume_index
                original_signals_filtered = original_signals_filtered.loc[common_volume_index]
                enhanced_signals_filtered = enhanced_signals_filtered.loc[common_volume_index]
        
        # Refilter to final common index
        original_signals_filtered = original_signals_filtered.loc[common_index]
        enhanced_signals_filtered = enhanced_signals_filtered.loc[common_index]
        
        # 1. Apply ML signal filter
        if self.config["enable_ml_filtering"] and self.signal_filter_model is not None:
            signal_quality = self.predict_signal_quality(features)
            metrics.loc[common_index, 'signal_quality'] = signal_quality
            
            # Filter out low quality signals
            filter_mask = signal_quality < self.config["prediction_threshold"]
            enhanced_signals_filtered[filter_mask] = 0
        
        # 2. Apply ML timing enhancements
        if self.config["enable_ml_timing"]:
            # Entry timing
            if self.entry_timing_model is not None:
                entry_quality = self.predict_optimal_entry(features)
                metrics.loc[common_index, 'entry_quality'] = entry_quality
                
                # Only enter when entry quality is high
                entry_mask = (original_signals_filtered != 0) & (enhanced_signals_filtered == 0) & (entry_quality >= self.config["prediction_threshold"])
                enhanced_signals_filtered[entry_mask] = original_signals_filtered[entry_mask]
            
            # Exit timing
            if self.exit_timing_model is not None:
                exit_quality = self.predict_optimal_exit(features)
                metrics.loc[common_index, 'exit_quality'] = exit_quality
                
                # Exit when exit quality is high and we have a position
                exit_mask = (enhanced_signals_filtered != 0) & (exit_quality >= self.config["prediction_threshold"])
                enhanced_signals_filtered[exit_mask] = 0
        
        # 3. Apply traditional filters
        # RSI filter
        if self.config["use_rsi_filter"] and prices_df is not None:
            for col in prices_df.columns:
                rsi_col = f'{col}_rsi'
                if rsi_col in features.columns:
                    rsi = features[rsi_col]
                    
                    # Oversold/overbought filter
                    oversold = rsi < 30
                    overbought = rsi > 70
                    
                    # Don't short oversold, don't go long overbought
                    enhanced_signals_filtered[(enhanced_signals_filtered < 0) & oversold] = 0
                    enhanced_signals_filtered[(enhanced_signals_filtered > 0) & overbought] = 0
        
        # Volume filter
        if self.config["use_volume_filter"] and volumes_df is not None:
            # Use volume prediction if available
            if self.volume_prediction_model is not None:
                predicted_volume = self.predict_volume_pattern(features)
                metrics.loc[common_index, 'predicted_volume'] = predicted_volume
                
                # Don't trade when predicted volume is extremely low
                volume_mask = predicted_volume < 0.5  # Previously 0.8, now 0.5 (below 50% of average volume)
                enhanced_signals_filtered[volume_mask] = 0
        
        # Volatility filter
        if self.config["use_volatility_filter"]:
            # Use volatility features if available
            vol_columns = [col for col in features.columns if 'vol_' in col]
            
            if vol_columns:
                # Calculate average volatility
                avg_vol = features[vol_columns].mean(axis=1)
                metrics.loc[common_index, 'avg_volatility'] = avg_vol
                
                # Don't trade in extremely low volatility only
                low_vol_mask = avg_vol < 0.0005  # Previously 0.001, now 0.0005 (half as restrictive)
                enhanced_signals_filtered[low_vol_mask] = 0
                
                # Adjust position size in high volatility (metadata)
                high_vol_mask = avg_vol > 0.03  # Previously 0.02, now 0.03 (less restrictive)
                metrics.loc[common_index[high_vol_mask], 'position_size_factor'] = 0.5  # Reduce position size
                metrics.loc[common_index[~high_vol_mask], 'position_size_factor'] = 1.0  # Normal position size
        
        # 4. Apply time-of-day filters
        if isinstance(features.index, pd.DatetimeIndex):
            # Early session filter (avoid first 15 minutes)
            if 'early_session' in features.columns:
                early_minutes_mask = (features.index.hour == 9) & (features.index.minute < 45)
                enhanced_signals_filtered[early_minutes_mask] = 0
            
            # Late session filter (close positions before end of day)
            if 'late_session' in features.columns:
                late_minutes_mask = (features.index.hour == 15) & (features.index.minute > 45)
                enhanced_signals_filtered[late_minutes_mask] = 0
            
            # Lunch hour filter (reduce activity during lunch)
            lunch_mask = (features.index.hour == 12)
            enhanced_signals_filtered[lunch_mask & (enhanced_signals_filtered != 0)] = 0  # Close positions
        
        # 5. Apply correlation prediction
        if self.correlation_model is not None:
            predicted_correlation = self.predict_correlation_change(features)
            metrics.loc[common_index, 'predicted_correlation'] = predicted_correlation
            
            # Don't enter when correlation is predicted to break down
            corr_breakdown_mask = predicted_correlation < 0.5  # Below 0.5 correlation
            new_entry_mask = (original_signals_filtered != 0) & (enhanced_signals_filtered == 0) & corr_breakdown_mask
            enhanced_signals_filtered[new_entry_mask] = 0
        
        # Calculate how many original signals were present and modified
        total_original_signals = (original_signals_filtered != 0).sum()
        total_enhanced_signals = (enhanced_signals_filtered != 0).sum()
        
        # If we've filtered out too many signals, preserve at least 10% of the original ones
        if total_original_signals > 10 and total_enhanced_signals < total_original_signals * 0.1:
            logger.warning(f"Enhanced signals filtered too aggressively: {total_enhanced_signals} of {total_original_signals} original signals remain")
            
            # Find indices of the original signals
            original_signal_indices = original_signals_filtered[original_signals_filtered != 0].index
            
            # Randomly select 10% of original signals to preserve if total_original_signals is large enough
            min_preserve = max(1, int(total_original_signals * 0.1))
            preserve_indices = original_signal_indices[:min_preserve]
            
            # Restore these signals
            enhanced_signals_filtered.loc[preserve_indices] = original_signals_filtered.loc[preserve_indices]
            logger.info(f"Preserved {len(preserve_indices)} original signals to maintain trading activity")
        
        # Track modifications
        modifications = sum((enhanced_signals_filtered != original_signals_filtered).astype(int))
        total_signals = len(original_signals_filtered)
        
        modification_rate = modifications / total_signals if total_signals > 0 else 0
        logger.info(f"Modified {modifications} out of {total_signals} signals ({modification_rate:.2%})")
        
        # Add enhanced signals to metrics
        metrics.loc[common_index, 'enhanced_signal'] = enhanced_signals_filtered
        
        # Update enhanced signals
        if isinstance(enhanced_signals, pd.DataFrame):
            # When enhanced_signals is a DataFrame
            if 'signal' in enhanced_signals.columns:
                # Create a temporary Series to hold the updates
                temp_series = pd.Series(enhanced_signals['signal'])
                temp_series.loc[common_index] = enhanced_signals_filtered
                enhanced_signals['signal'] = temp_series
            else:
                # Just update the first column
                col_name = enhanced_signals.columns[0]
                temp_series = pd.Series(enhanced_signals[col_name])
                temp_series.loc[common_index] = enhanced_signals_filtered
                enhanced_signals[col_name] = temp_series
        else:
            # When enhanced_signals is a Series
            enhanced_signals.loc[common_index] = enhanced_signals_filtered
        
        return enhanced_signals, metrics
    
    def apply_intraday_adaptations(self, signals, features, prices_df, current_time=None, volumes_df=None):
        """
        Apply intraday adaptations to signals based on time of day and market conditions.
        
        Parameters:
        -----------
        signals : pd.Series
            Series with trading signals
        features : pd.DataFrame
            DataFrame with features
        prices_df : pd.DataFrame
            DataFrame with price data
        current_time : datetime, optional
            Current time for time-based adaptations
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
            
        Returns:
        --------
        pd.Series
            Series with adapted signals
        dict
            Dictionary with adaptation metadata
        """
        # Use the last timestamp in the data if current_time is not provided
        if current_time is None and isinstance(features.index, pd.DatetimeIndex):
            current_time = features.index[-1]
        
        adapted_signals = signals.copy()
        adaptations = {}
        
        # Skip if no time information is available
        if current_time is None:
            logger.warning("No time information provided for intraday adaptations")
            return adapted_signals, {"adapted": False, "reason": "No time information"}
        
        # 1. Time-based position size scaling
        if isinstance(current_time, datetime):
            hour = current_time.hour
            minute = current_time.minute
            
            # Convert to decimal time
            decimal_time = hour + minute / 60.0
            
            # Market session progress (9:30 AM to 4:00 PM)
            market_open = 9.5  # 9:30 AM
            market_close = 16.0  # 4:00 PM
            session_length = market_close - market_open
            
            # Calculate session progress (0 to 1)
            if decimal_time < market_open:
                session_progress = 0.0
            elif decimal_time > market_close:
                session_progress = 1.0
            else:
                session_progress = (decimal_time - market_open) / session_length
            
            # Time decay factor (reduce position size as day progresses)
            time_decay = 1.0 - (session_progress * 0.3)  # Max 30% reduction
            adaptations["time_decay_factor"] = time_decay
            
            # Force close near end of day
            if decimal_time > (market_close - 0.25):  # Last 15 minutes
                # Instead of setting all signals to 0, just set a flag but don't modify the signals
                # This will allow some trades to continue
                adaptations["approaching_close"] = True
                adaptations["reason"] = "End of day approaching"
                
                # Only close positions but don't open new ones
                if len(adapted_signals[adapted_signals != 0]) > 0:
                    # Keep 20% of positions open for smoother trading
                    open_position_indices = adapted_signals[adapted_signals != 0].index
                    positions_to_close = open_position_indices[:(len(open_position_indices) * 8) // 10]
                    
                    if len(positions_to_close) > 0:
                        # Handle both Series and DataFrame cases
                        if isinstance(adapted_signals, pd.DataFrame):
                            # For DataFrame, we need to handle columns individually
                            if 'signal' in adapted_signals.columns:
                                adapted_signals.loc[positions_to_close, 'signal'] = 0
                            else:  # Use the first column
                                col_name = adapted_signals.columns[0]
                                adapted_signals.loc[positions_to_close, col_name] = 0
                        else:
                            # For Series, we can use the dtype attribute
                            adapted_signals.loc[positions_to_close] = 0
                        
                        logger.info(f"Closing {len(positions_to_close)} positions due to approaching market close")
        
        # High impact times (market open, lunch, market close)
        high_impact_times = [
            (9.5, 9.75),   # 9:30-9:45 AM (market open)
            (12.0, 13.0),  # 12:00-1:00 PM (lunch hour)
            (15.75, 16.0)  # 3:45-4:00 PM (market close)
        ]

        for start_time, end_time in high_impact_times:
            if start_time <= decimal_time < end_time:
                # More conservative parameters during high impact times
                adaptations["in_high_impact_time"] = True
                adaptations["high_impact_period"] = f"{start_time:.2f}-{end_time:.2f}"
                
                # Less aggressive during high impact times
                # Don't open new positions, but don't force close existing ones
                # Handle both Series and DataFrame cases
                if isinstance(adapted_signals, pd.DataFrame):
                    # For DataFrame, we need to handle columns individually
                    if 'signal' in adapted_signals.columns:
                        # Get indices where signals is 0
                        zero_indices = signals.index[signals['signal'] == 0]
                        adapted_signals.loc[zero_indices, 'signal'] = 0
                    else:  # Use the first column
                        col_name = adapted_signals.columns[0]
                        # Get indices where signals is 0
                        zero_indices = signals.index[signals.iloc[:, 0] == 0]
                        adapted_signals.loc[zero_indices, col_name] = 0
                else:
                    # For Series, we can set directly
                    adapted_signals[signals == 0] = 0
        
        # 2. Volume-based adaptations
        if 'predicted_volume' in features.columns or (volumes_df is not None and len(volumes_df.columns) > 0):
            # Use predicted volume if available, otherwise use actual volume
            if 'predicted_volume' in features.columns:
                volume_indicator = features['predicted_volume']
            else:
                # Use average volume across all instruments
                volume_indicator = volumes_df.mean(axis=1)
            
            # Get the latest volume indicator
            latest_volume = volume_indicator.iloc[-1] if len(volume_indicator) > 0 else 1.0
            
            # Volume-based position size scaling
            volume_scale = min(max(latest_volume, 0.5), 1.5)  # Between 0.5 and 1.5
            adaptations["volume_scale_factor"] = volume_scale
            
            # Avoid trading in very low volume
            if latest_volume < 0.5:  # Less than 50% of average volume
                adapted_signals[signals == 0] = 0  # No new entries
                adaptations["low_volume_filter"] = True
        
        # 3. Volatility-based adaptations
        vol_columns = [col for col in features.columns if 'vol_' in col]
        
        if vol_columns:
            # Calculate average volatility
            avg_vol = features[vol_columns].mean(axis=1).iloc[-1]
            adaptations["current_volatility"] = float(avg_vol)
            
            # Volatility-based position size scaling
            vol_scale = 1.0 / max(avg_vol / 0.01, 0.5)  # Inverse relationship with volatility
            vol_scale = min(max(vol_scale, 0.5), 1.5)  # Between 0.5 and 1.5
            adaptations["volatility_scale_factor"] = vol_scale
            
            # Increase signal threshold in high volatility
            if avg_vol > 0.02:  # High volatility threshold
                adaptations["high_volatility"] = True
                adaptations["signal_threshold_adjustment"] = "+0.5"  # Increase threshold
        
        # 4. Combined adaptation factor
        adaptation_factor = 1.0
        
        if "time_decay_factor" in adaptations:
            adaptation_factor *= adaptations["time_decay_factor"]
        
        if "volume_scale_factor" in adaptations:
            adaptation_factor *= adaptations["volume_scale_factor"]
        
        if "volatility_scale_factor" in adaptations:
            adaptation_factor *= adaptations["volatility_scale_factor"]
        
        # Ensure factor is within reasonable bounds
        adaptation_factor = min(max(adaptation_factor, 0.3), 1.5)
        adaptations["final_adaptation_factor"] = adaptation_factor
        
        # Apply adaptation by scaling signal intensity (for position sizing)
        # Note: This is metadata for position sizing, not changing the signal direction
        adaptations["original_signal_count"] = (signals != 0).sum()
        adaptations["adapted_signal_count"] = (adapted_signals != 0).sum()
        
        return adapted_signals, adaptations

class IntradaySignalProcessor:
    """
    Process and apply intraday signal enhancements to a pairs trading strategy.
    """
    
    def __init__(self, config=None):
        """Initialize the signal processor."""
        self.config = config or {}
        self.signal_enhancer = IntradaySignalEnhancer(config)
        
    def process_intraday_signals(self, prices_df, spreads_df, original_signals, volumes_df=None):
        """
        Process original signals with intraday enhancements.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data
        spreads_df : pd.DataFrame
            DataFrame with spread data
        original_signals : pd.Series
            Series with original trading signals
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
            
        Returns:
        --------
        pd.Series
            Series with enhanced signals
        pd.DataFrame
            DataFrame with signal metrics
        """
        # Calculate features
        features = self.signal_enhancer.calculate_features(prices_df, spreads_df, volumes_df)
        
        # Load models if not already loaded
        if self.signal_enhancer.signal_filter_model is None:
            self.signal_enhancer.load_models()
        
        # Enhance signals
        enhanced_signals, metrics = self.signal_enhancer.enhance_signals(
            original_signals, features, prices_df, volumes_df
        )
        
        # Apply intraday adaptations
        if isinstance(spreads_df.index, pd.DatetimeIndex):
            current_time = spreads_df.index[-1]
            adapted_signals, adaptations = self.signal_enhancer.apply_intraday_adaptations(
                enhanced_signals, features, prices_df, current_time, volumes_df
            )
            metrics['adaptation_factor'] = pd.Series(adaptations.get("final_adaptation_factor", 1.0), index=metrics.index)
            enhanced_signals = adapted_signals
        
        return enhanced_signals, metrics
    
    def train_models(self, prices_df, spreads_df, signals_df, performance_df, volumes_df=None):
        """
        Train all ML models for signal enhancement.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data
        spreads_df : pd.DataFrame
            DataFrame with spread data
        signals_df : pd.DataFrame
            DataFrame with trading signals
        performance_df : pd.DataFrame
            DataFrame with trade performance data
        volumes_df : pd.DataFrame, optional
            DataFrame with volume data
            
        Returns:
        --------
        dict
            Dictionary with training results
        """
        # Calculate features
        features = self.signal_enhancer.calculate_features(prices_df, spreads_df, volumes_df)
        
        # Prepare training data
        
        # 1. Signal filter - predict if a signal will be profitable
        signal_mask = signals_df['signal'] != 0
        profitable_mask = performance_df['trade_pnl'] > 0
        
        # Create labels: 1 for profitable trades, 0 for unprofitable
        labels = pd.Series(0, index=signals_df.index)
        labels[signal_mask & profitable_mask] = 1
        
        # Train signal filter model
        signal_filter_success = self.signal_enhancer.train_signal_filter_model(
            features[signal_mask], labels[signal_mask]
        )
        
        # 2. Entry timing - predict optimal entry points
        # Identify optimal entry points (e.g., those close to local extremes)
        optimal_entries = pd.Series(0, index=spreads_df.index)
        
        # Define optimal entry as points where the spread is near local extremes
        if 'zscore' in spreads_df.columns:
            zscore = spreads_df['zscore']
            
            # Long entries (negative z-score extremes)
            long_entries = (zscore < -2.0) & (zscore.shift(1) >= -2.0)
            
            # Short entries (positive z-score extremes)
            short_entries = (zscore > 2.0) & (zscore.shift(1) <= 2.0)
            
            optimal_entries[long_entries | short_entries] = 1
        
        # Train entry timing model
        entry_timing_success = self.signal_enhancer.train_entry_timing_model(
            features, optimal_entries
        )
        
        # 3. Exit timing - predict optimal exit points
        optimal_exits = pd.Series(0, index=spreads_df.index)
        
        # Define optimal exit as points where the spread reverts to mean
        if 'zscore' in spreads_df.columns:
            zscore = spreads_df['zscore']
            
            # Exit long positions (zscore crosses back above 0)
            exit_longs = (zscore > 0) & (zscore.shift(1) <= 0)
            
            # Exit short positions (zscore crosses back below 0)
            exit_shorts = (zscore < 0) & (zscore.shift(1) >= 0)
            
            optimal_exits[exit_longs | exit_shorts] = 1
        
        # Train exit timing model
        exit_timing_success = self.signal_enhancer.train_exit_timing_model(
            features, optimal_exits
        )
        
        # 4. Volume prediction - predict future volume
        if volumes_df is not None and len(volumes_df.columns) > 0:
            # Use average volume across instruments
            avg_volume = volumes_df.mean(axis=1)
            
            # Target is future volume (e.g., 30 minutes ahead)
            future_volume = avg_volume.shift(-6)  # Assuming 5-minute bars, shift by 6 bars = 30 minutes
            
            # Train volume prediction model
            volume_prediction_success = self.signal_enhancer.train_volume_prediction_model(
                features[:-6], future_volume[:-6].dropna()
            )
        else:
            volume_prediction_success = False
        
        # 5. Correlation prediction - predict future correlation changes
        if len(prices_df.columns) >= 2:
            # Calculate rolling correlation between first two instruments
            returns1 = prices_df.iloc[:, 0].pct_change()
            returns2 = prices_df.iloc[:, 1].pct_change()
            
            rolling_corr = returns1.rolling(20).corr(returns2)
            
            # Target is future correlation
            future_corr = rolling_corr.shift(-6)  # 30 minutes ahead
            
            # Train correlation prediction model
            correlation_prediction_success = self.signal_enhancer.train_correlation_prediction_model(
                features[:-6], future_corr[:-6].dropna()
            )
        else:
            correlation_prediction_success = False
        
        # Return training results
        return {
            "signal_filter": signal_filter_success,
            "entry_timing": entry_timing_success,
            "exit_timing": exit_timing_success,
            "volume_prediction": volume_prediction_success,
            "correlation_prediction": correlation_prediction_success
        } 