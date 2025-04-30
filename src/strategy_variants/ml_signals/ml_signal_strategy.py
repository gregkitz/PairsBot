"""
Machine Learning Signal Generation Strategy for the Intraday Statistical Arbitrage System.

This module implements machine learning-based approaches to signal generation
for pairs trading, including feature engineering, model training, and prediction.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load

from src.pairs_trading_strategy import PairsTradingStrategy
from src.data_processor.data_processor import DataProcessor
from src.signal_generation.signal_generator import SignalGenerator
from src.risk_management.risk_manager import RiskManager
from src.spread_analytics.spread_analyzer import SpreadAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class MLSignalStrategy(PairsTradingStrategy):
    """
    Machine Learning Signal Strategy for pairs trading that extends the base
    pairs trading strategy with advanced ML-based signal generation.
    
    This strategy uses machine learning models to predict the direction of
    the spread and generate trading signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML signal strategy.
        
        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary for the strategy
        """
        super().__init__(config)
        
        # ML-specific configuration
        self.ml_config = config.get('ml_signals', {})
        self.model_type = self.ml_config.get('model_type', 'random_forest')  # 'random_forest', 'gbm', 'logistic', 'svm', 'nn'
        self.prediction_horizon = self.ml_config.get('prediction_horizon', 5)  # bars ahead to predict
        self.min_history_bars = self.ml_config.get('min_history_bars', 1000)
        self.training_frequency = self.ml_config.get('training_frequency', 7)  # days
        self.threshold = self.ml_config.get('threshold', 0.65)  # prediction probability threshold
        self.feature_window = self.ml_config.get('feature_window', 20)  # lookback for feature calculation
        
        # Model parameters
        self.random_forest_params = self.ml_config.get('random_forest_params', {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        })
        
        self.gbm_params = self.ml_config.get('gbm_params', {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        })
        
        self.logistic_params = self.ml_config.get('logistic_params', {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        })
        
        self.svm_params = self.ml_config.get('svm_params', {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        })
        
        self.nn_params = self.ml_config.get('nn_params', {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 1000,
            'random_state': 42
        })
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.last_training = {}
        
        # Feature importance
        self.feature_importance = {}
        
        logger.info(f"ML Signal Strategy initialized with model type: {self.model_type}")
    
    def analyze_pair(self, pair_id: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze a pair using ML models and generate trading signals.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        data : Dict[str, pd.DataFrame]
            Historical price data for each symbol in the pair
        
        Returns:
        --------
        Dict[str, Any]
            Analysis results including ML-based signals
        """
        # Get the standard pairs trading analysis first
        base_analysis = super().analyze_pair(pair_id, data)
        
        # Check if we have enough data for ML analysis
        leg1_data = data[self.pairs[pair_id]['leg1']]
        leg2_data = data[self.pairs[pair_id]['leg2']]
        
        if len(leg1_data) < self.min_history_bars or len(leg2_data) < self.min_history_bars:
            logger.warning(f"Not enough data for ML analysis of pair {pair_id}")
            return base_analysis
        
        # Get the spread series from the base analysis
        spread_series = base_analysis['spread_series']
        z_score_series = base_analysis['z_score_series']
        
        # Check if we need to train or load the model
        if self._should_train_model(pair_id):
            self._train_model(pair_id, spread_series, z_score_series, leg1_data, leg2_data)
        
        # Generate features for prediction
        features = self._generate_features(pair_id, spread_series, z_score_series, leg1_data, leg2_data)
        
        # Make predictions
        prediction_result = self._make_prediction(pair_id, features)
        
        # Update the base analysis with ML predictions
        base_analysis.update({
            'ml_prediction': prediction_result.get('prediction', 0),
            'ml_probability': prediction_result.get('probability', 0.5),
            'ml_model_type': self.model_type,
            'ml_feature_importance': self.feature_importance.get(pair_id, {})
        })
        
        # Generate ML-enhanced signals
        signals = self._generate_ml_signals(pair_id, base_analysis, prediction_result)
        base_analysis['signals'] = signals
        
        return base_analysis
    
    def _should_train_model(self, pair_id: str) -> bool:
        """
        Determine if the model should be trained or retrained.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        
        Returns:
        --------
        bool
            True if the model should be trained, False otherwise
        """
        # Check if the model exists
        if pair_id not in self.models:
            # Try to load the model from disk
            model_path = os.path.join('models', 'ml_signals', f"{pair_id}_{self.model_type}.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[pair_id] = load(model_path)
                    
                    # Load scaler if available
                    scaler_path = os.path.join('models', 'ml_signals', f"{pair_id}_scaler.joblib")
                    if os.path.exists(scaler_path):
                        self.scalers[pair_id] = load(scaler_path)
                    
                    # Load last training date if available
                    meta_path = os.path.join('models', 'ml_signals', f"{pair_id}_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            self.last_training[pair_id] = datetime.fromisoformat(meta.get('last_training', '2000-01-01'))
                            self.feature_importance[pair_id] = meta.get('feature_importance', {})
                    
                    logger.info(f"Loaded existing model for pair {pair_id}")
                    
                    # Still check if we should retrain
                    current_time = datetime.now()
                    last_train_time = self.last_training.get(pair_id, datetime.min)
                    
                    return (current_time - last_train_time) > timedelta(days=self.training_frequency)
                except Exception as e:
                    logger.error(f"Error loading model for pair {pair_id}: {str(e)}")
                    return True
            else:
                # No existing model, should train
                return True
        else:
            # Check if it's time to retrain the model
            current_time = datetime.now()
            last_train_time = self.last_training.get(pair_id, datetime.min)
            
            return (current_time - last_train_time) > timedelta(days=self.training_frequency)
    
    def _train_model(self, pair_id: str, spread_series: pd.Series, z_score_series: pd.Series, 
                    leg1_data: pd.DataFrame, leg2_data: pd.DataFrame) -> None:
        """
        Train an ML model for signal generation.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        z_score_series : pd.Series
            Historical z-score data
        leg1_data : pd.DataFrame
            Historical data for the first leg
        leg2_data : pd.DataFrame
            Historical data for the second leg
        """
        logger.info(f"Training ML model for pair {pair_id}")
        
        try:
            # Generate features
            X, y = self._prepare_training_data(spread_series, z_score_series, leg1_data, leg2_data)
            
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"Not enough training data for pair {pair_id}")
                return
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Create and fit a scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Save the scaler
            self.scalers[pair_id] = scaler
            
            # Create the model
            model = self._create_model()
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, average='weighted')
            recall = recall_score(y_val, y_pred, average='weighted')
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            logger.info(f"Model performance for {pair_id}: "
                       f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                       f"Recall={recall:.4f}, F1={f1:.4f}")
            
            # Store the model
            self.models[pair_id] = model
            self.last_training[pair_id] = datetime.now()
            
            # Extract feature importance if available
            feature_names = self._get_feature_names(spread_series, z_score_series, leg1_data, leg2_data)
            self.feature_importance[pair_id] = self._extract_feature_importance(model, feature_names)
            
            # Save the model, scaler, and metadata
            os.makedirs(os.path.join('models', 'ml_signals'), exist_ok=True)
            
            model_path = os.path.join('models', 'ml_signals', f"{pair_id}_{self.model_type}.joblib")
            scaler_path = os.path.join('models', 'ml_signals', f"{pair_id}_scaler.joblib")
            meta_path = os.path.join('models', 'ml_signals', f"{pair_id}_meta.json")
            
            dump(model, model_path)
            dump(scaler, scaler_path)
            
            meta = {
                'last_training': self.last_training[pair_id].isoformat(),
                'model_type': self.model_type,
                'performance': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'feature_importance': self.feature_importance[pair_id]
            }
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"Model training and saving successful for pair {pair_id}")
            
        except Exception as e:
            logger.error(f"Error training model for pair {pair_id}: {str(e)}")
    
    def _prepare_training_data(self, spread_series: pd.Series, z_score_series: pd.Series, 
                               leg1_data: pd.DataFrame, leg2_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the ML model.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Historical spread data
        z_score_series : pd.Series
            Historical z-score data
        leg1_data : pd.DataFrame
            Historical data for the first leg
        leg2_data : pd.DataFrame
            Historical data for the second leg
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Features (X) and labels (y) for training
        """
        # Generate features
        features_df = self._generate_features_df(spread_series, z_score_series, leg1_data, leg2_data)
        
        # Create labels (target variable)
        # Label 1 if spread grows in next 'prediction_horizon' bars, 0 if flat or decreases
        future_spread_change = spread_series.shift(-self.prediction_horizon) - spread_series
        # Convert to classes: 1 for positive change, -1 for negative change, 0 for no change
        future_spread_direction = np.sign(future_spread_change)
        
        # Combine features and labels
        combined_df = pd.concat([features_df, future_spread_direction], axis=1)
        combined_df.columns = list(features_df.columns) + ['target']
        
        # Drop NaN values
        combined_df = combined_df.dropna()
        
        # Split into features and labels
        X = combined_df.drop('target', axis=1).values
        y = combined_df['target'].values
        
        return X, y
    
    def _generate_features_df(self, spread_series: pd.Series, z_score_series: pd.Series, 
                             leg1_data: pd.DataFrame, leg2_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features dataframe for the ML model.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Historical spread data
        z_score_series : pd.Series
            Historical z-score data
        leg1_data : pd.DataFrame
            Historical data for the first leg
        leg2_data : pd.DataFrame
            Historical data for the second leg
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing features
        """
        # Start with basic statistics on spread and z-score
        features = {}
        
        # Current z-score
        features['z_score'] = z_score_series
        
        # Z-score changes over different windows
        for window in [1, 3, 5, 10]:
            features[f'z_score_change_{window}'] = z_score_series.diff(window)
        
        # Z-score moving averages
        for window in [5, 10, 20]:
            features[f'z_score_ma_{window}'] = z_score_series.rolling(window=window).mean()
        
        # Z-score volatility (standard deviation)
        for window in [5, 10, 20]:
            features[f'z_score_std_{window}'] = z_score_series.rolling(window=window).std()
        
        # Z-score momentum
        for window in [3, 5, 10]:
            features[f'z_score_momentum_{window}'] = z_score_series - z_score_series.shift(window)
        
        # Z-score acceleration
        features['z_score_accel'] = z_score_series.diff().diff()
        
        # Relative Strength Index (RSI) on z-score
        def rsi(series, window=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        features['z_score_rsi'] = rsi(z_score_series, window=14)
        
        # Technical indicators based on spread
        # MACD
        def macd(series, fast=12, slow=26, signal=9):
            fast_ema = series.ewm(span=fast, min_periods=fast).mean()
            slow_ema = series.ewm(span=slow, min_periods=slow).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
            return macd_line, signal_line
        
        macd_line, signal_line = macd(spread_series)
        features['macd_line'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line
        
        # Cross-pair features
        # Correlation between legs
        for window in [10, 20, 30]:
            features[f'correlation_{window}'] = leg1_data['Close'].rolling(window=window).corr(leg2_data['Close'])
        
        # Volume imbalance
        if 'Volume' in leg1_data.columns and 'Volume' in leg2_data.columns:
            for window in [1, 5, 10]:
                features[f'volume_imbalance_{window}'] = (
                    leg1_data['Volume'].rolling(window=window).sum() / 
                    leg2_data['Volume'].rolling(window=window).sum()
                )
        
        # Market regime indicators
        # Bollinger Bands width (as volatility indicator)
        for window in [10, 20]:
            bb_std = spread_series.rolling(window=window).std()
            bb_middle = spread_series.rolling(window=window).mean()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            features[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(features)
        
        # Add time-of-day features if timestamp is available
        if hasattr(spread_series.index, 'hour'):
            features_df['hour_of_day'] = spread_series.index.hour
            features_df['minute_of_hour'] = spread_series.index.minute
            # Add daily cyclical features using sine and cosine transformations
            features_df['hour_sin'] = np.sin(2 * np.pi * spread_series.index.hour / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * spread_series.index.hour / 24)
        
        # Add day-of-week features if timestamp is available
        if hasattr(spread_series.index, 'dayofweek'):
            features_df['day_of_week'] = spread_series.index.dayofweek
            # Add weekly cyclical features
            features_df['day_sin'] = np.sin(2 * np.pi * spread_series.index.dayofweek / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * spread_series.index.dayofweek / 7)
        
        return features_df
    
    def _get_feature_names(self, spread_series: pd.Series, z_score_series: pd.Series, 
                          leg1_data: pd.DataFrame, leg2_data: pd.DataFrame) -> List[str]:
        """
        Get the feature names for the ML model.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Historical spread data
        z_score_series : pd.Series
            Historical z-score data
        leg1_data : pd.DataFrame
            Historical data for the first leg
        leg2_data : pd.DataFrame
            Historical data for the second leg
        
        Returns:
        --------
        List[str]
            List of feature names
        """
        # Generate features dataframe to get column names
        features_df = self._generate_features_df(spread_series, z_score_series, leg1_data, leg2_data)
        return list(features_df.columns)
    
    def _create_model(self):
        """
        Create an ML model based on the configuration.
        
        Returns:
        --------
        object
            Scikit-learn model
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.random_forest_params)
        elif self.model_type == 'gbm':
            return GradientBoostingClassifier(**self.gbm_params)
        elif self.model_type == 'logistic':
            return LogisticRegression(**self.logistic_params)
        elif self.model_type == 'svm':
            return SVC(**self.svm_params)
        elif self.model_type == 'nn':
            return MLPClassifier(**self.nn_params)
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using Random Forest as fallback")
            return RandomForestClassifier(**self.random_forest_params)
    
    def _extract_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from the model if available.
        
        Parameters:
        -----------
        model : object
            Scikit-learn model
        feature_names : List[str]
            List of feature names
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of feature importance
        """
        feature_importance = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                importances = model.feature_importances_
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
            elif hasattr(model, 'coef_'):
                # For linear models
                importances = np.abs(model.coef_[0])
                for name, importance in zip(feature_names, importances):
                    feature_importance[name] = float(importance)
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
        
        return feature_importance
    
    def _generate_features(self, pair_id: str, spread_series: pd.Series, z_score_series: pd.Series, 
                          leg1_data: pd.DataFrame, leg2_data: pd.DataFrame) -> np.ndarray:
        """
        Generate features for prediction.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        spread_series : pd.Series
            Historical spread data
        z_score_series : pd.Series
            Historical z-score data
        leg1_data : pd.DataFrame
            Historical data for the first leg
        leg2_data : pd.DataFrame
            Historical data for the second leg
        
        Returns:
        --------
        np.ndarray
            Features for prediction
        """
        # Generate features dataframe
        features_df = self._generate_features_df(spread_series, z_score_series, leg1_data, leg2_data)
        
        # Get the latest data point
        latest_features = features_df.iloc[-1].values.reshape(1, -1)
        
        # Scale features if scaler exists
        if pair_id in self.scalers:
            latest_features = self.scalers[pair_id].transform(latest_features)
        
        return latest_features
    
    def _make_prediction(self, pair_id: str, features: np.ndarray) -> Dict[str, Any]:
        """
        Make a prediction using the ML model.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        features : np.ndarray
            Features for prediction
        
        Returns:
        --------
        Dict[str, Any]
            Prediction results
        """
        if pair_id not in self.models:
            logger.warning(f"No model found for pair {pair_id}")
            return {'prediction': 0, 'probability': 0.5}
        
        model = self.models[pair_id]
        
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probability
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                if prediction == 1:
                    probability = probabilities[1]  # Probability of class 1
                elif prediction == -1:
                    probability = probabilities[0]  # Probability of class -1
                else:
                    # If using a model with 3 classes (0 for no change)
                    probability = probabilities[np.where(model.classes_ == prediction)[0][0]]
            else:
                # If model doesn't have predict_proba, use decision function if available
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(features)[0]
                    probability = 1 / (1 + np.exp(-decision))  # Sigmoid function
                else:
                    # Default to 0.5 if no probability method is available
                    probability = 0.5
            
            return {
                'prediction': prediction,
                'probability': probability
            }
        except Exception as e:
            logger.error(f"Error making prediction for pair {pair_id}: {str(e)}")
            return {'prediction': 0, 'probability': 0.5}
    
    def _generate_ml_signals(self, pair_id: str, analysis: Dict[str, Any], 
                            prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals using ML predictions.
        
        Parameters:
        -----------
        pair_id : str
            Identifier for the pair
        analysis : Dict[str, Any]
            Analysis results including z-score based signals
        prediction_result : Dict[str, Any]
            ML prediction results
        
        Returns:
        --------
        Dict[str, Any]
            Trading signals
        """
        # Get base signals from z-score
        base_signals = analysis.get('signals', {})
        
        # Get ML prediction and probability
        prediction = prediction_result.get('prediction', 0)
        probability = prediction_result.get('probability', 0.5)
        
        # Get current signal
        current_signal = base_signals.get('signal', 0)
        
        # Only use ML signal if confidence is high enough
        if probability > self.threshold:
            # Use ML prediction to enhance the signal
            if prediction == 1:  # Spread expected to increase
                # If spread is already high (positive z-score), strengthen short signal
                # If spread is low (negative z-score), weaken long signal
                if current_signal < 0:  # Short signal
                    enhanced_signal = current_signal * 1.2  # 20% stronger
                elif current_signal > 0:  # Long signal
                    enhanced_signal = current_signal * 0.8  # 20% weaker
                else:
                    enhanced_signal = current_signal
            elif prediction == -1:  # Spread expected to decrease
                # If spread is already low (negative z-score), strengthen long signal
                # If spread is high (positive z-score), weaken short signal
                if current_signal > 0:  # Long signal
                    enhanced_signal = current_signal * 1.2  # 20% stronger
                elif current_signal < 0:  # Short signal
                    enhanced_signal = current_signal * 0.8  # 20% weaker
                else:
                    enhanced_signal = current_signal
            else:
                # No clear prediction, use base signal
                enhanced_signal = current_signal
        else:
            # Not confident enough, use base signal
            enhanced_signal = current_signal
        
        # Update signals
        enhanced_signals = base_signals.copy()
        enhanced_signals['signal'] = enhanced_signal
        enhanced_signals['ml_prediction'] = prediction
        enhanced_signals['ml_probability'] = probability
        
        return enhanced_signals
    
    def save_state(self, directory: str = None) -> Dict[str, Any]:
        """
        Save the strategy state to file.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to save the state
        
        Returns:
        --------
        Dict[str, Any]
            Summary of saved state
        """
        # Get base state
        base_state = super().save_state(directory)
        
        # Save any additional ML-specific state
        ml_state = {
            'model_type': self.model_type,
            'pairs_with_models': list(self.models.keys()),
            'last_training': {pair_id: date.isoformat() for pair_id, date in self.last_training.items()},
            'feature_importance': self.feature_importance
        }
        
        # Save ML state to file
        if directory:
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, 'ml_signals_state.json'), 'w') as f:
                json.dump(ml_state, f, indent=2)
        
        return {
            **base_state,
            'ml_signals': ml_state
        }
    
    def load_state(self, directory: str) -> Dict[str, Any]:
        """
        Load the strategy state from file.
        
        Parameters:
        -----------
        directory : str
            Directory containing saved state
        
        Returns:
        --------
        Dict[str, Any]
            Summary of loaded state
        """
        # Get base state
        base_state = super().load_state(directory)
        
        # Load ML state
        ml_state_file = os.path.join(directory, 'ml_signals_state.json')
        if os.path.exists(ml_state_file):
            with open(ml_state_file, 'r') as f:
                ml_state = json.load(f)
            
            # Update state
            self.model_type = ml_state.get('model_type', self.model_type)
            
            # Update last training times
            last_training = ml_state.get('last_training', {})
            for pair_id, date_str in last_training.items():
                try:
                    self.last_training[pair_id] = datetime.fromisoformat(date_str)
                except (ValueError, TypeError):
                    # If date parsing fails, default to requiring model update
                    self.last_training[pair_id] = datetime.now() - timedelta(days=self.training_frequency)
            
            # Update feature importance
            self.feature_importance = ml_state.get('feature_importance', {})
            
            # Load models for pairs
            for pair_id in ml_state.get('pairs_with_models', []):
                if pair_id not in self.models:
                    model_path = os.path.join('models', 'ml_signals', f"{pair_id}_{self.model_type}.joblib")
                    scaler_path = os.path.join('models', 'ml_signals', f"{pair_id}_scaler.joblib")
                    
                    if os.path.exists(model_path):
                        try:
                            self.models[pair_id] = load(model_path)
                            logger.info(f"Loaded model for pair {pair_id}")
                        except Exception as e:
                            logger.error(f"Error loading model for pair {pair_id}: {str(e)}")
                    
                    if os.path.exists(scaler_path):
                        try:
                            self.scalers[pair_id] = load(scaler_path)
                            logger.info(f"Loaded scaler for pair {pair_id}")
                        except Exception as e:
                            logger.error(f"Error loading scaler for pair {pair_id}: {str(e)}")
        
        return {
            **base_state,
            'ml_signals_loaded': True
        } 