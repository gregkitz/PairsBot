"""
Intraday ML Integration Module

This module integrates various ML components for intraday trading:
1. Connects ML models to signal generation
2. Integrates regime detection with parameter adaptation
3. Links the backtesting system with ML enhancements
"""

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import logging
import os
import json
from typing import Dict, List, Tuple, Union, Optional

from src.ml_enhancements.intraday_signals import IntradaySignalEnhancer, IntradaySignalProcessor
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from src.backtest.intraday_backtest_engine import IntradayBacktestEngine
from src.backtest.intraday_performance_visualization import create_intraday_performance_dashboard, save_performance_dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class IntradayMLSystem:
    """
    Integrated ML system for intraday trading.
    
    This class connects ML model components with signal generation, regime detection,
    and the backtesting engine for a complete ML-enhanced intraday trading system.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 config: Optional[Dict] = None, 
                 models_dir: str = "models/intraday",
                 output_dir: str = "output/intraday"):
        """
        Initialize the integrated ML system.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        config : dict, optional
            Configuration dictionary (used if config_path is None)
        models_dir : str
            Directory for model storage
        output_dir : str
            Directory for output files
        """
        self.config = self._load_config(config_path) if config_path else config or {}
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # State tracking
        self.current_regime = None
        self.regimes_history = pd.DataFrame()
        self.last_adaptation_time = None
        self.adaptation_frequency = self.config.get("adaptation_frequency", "daily")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def _init_components(self):
        """Initialize the ML and trading components."""
        # Initialize signal enhancer
        signal_enhancer_config = self.config.get("signal_enhancer", {})
        signal_enhancer_config["models_dir"] = self.models_dir
        self.signal_enhancer = IntradaySignalEnhancer(config=signal_enhancer_config)
        
        # Initialize signal processor
        signal_processor_config = self.config.get("signal_processor", {})
        self.signal_processor = IntradaySignalProcessor(config=signal_processor_config)
        
        # Initialize regime classifier
        regime_config = self.config.get("regime_detection", {})
        self.regime_classifier = MarketRegimeClassifier(
            n_regimes=regime_config.get("n_regimes", 3),
            lookback_window=regime_config.get("lookback_window", 60),
            features=regime_config.get("features", None),
            method=regime_config.get("method", "kmeans")
        )
        
        # Initialize backtest engine
        backtest_config = self.config.get("backtest", {})
        intraday_params = backtest_config.get("intraday_params", {})
        transaction_cost_model = backtest_config.get("transaction_cost_model", {})
        
        self.backtest_engine = IntradayBacktestEngine(
            account_size=backtest_config.get("account_size", 100000),
            intraday_params=intraday_params,
            transaction_cost_model=transaction_cost_model
        )
    
    def load_models(self):
        """Load pre-trained ML models for signal enhancement and regime detection."""
        # Load signal enhancer models
        self.signal_enhancer.load_models()
        
        # Load regime classifier model
        regime_model_path = os.path.join(self.models_dir, "regime_classifier.pkl")
        if os.path.exists(regime_model_path):
            try:
                import joblib
                regime_model = joblib.load(regime_model_path)
                self.regime_classifier.model = regime_model
                logger.info("Loaded regime classifier model")
            except Exception as e:
                logger.error(f"Error loading regime classifier model: {e}")
    
    def train_models(self, 
                     prices_data: Dict[str, pd.DataFrame], 
                     spreads_data: pd.DataFrame, 
                     signals_data: pd.DataFrame, 
                     performance_data: Optional[pd.DataFrame] = None,
                     volumes_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Train all ML models in the system.
        
        Parameters:
        -----------
        prices_data : dict
            Dictionary of price DataFrames for each symbol
        spreads_data : pd.DataFrame
            DataFrame with spread data
        signals_data : pd.DataFrame
            DataFrame with trading signals
        performance_data : pd.DataFrame, optional
            DataFrame with trading performance
        volumes_data : dict, optional
            Dictionary of volume DataFrames for each symbol
        """
        logger.info("Training signal enhancement models")
        self.signal_processor.train_models(
            prices_df=prices_data,
            spreads_df=spreads_data,
            signals_df=signals_data,
            performance_df=performance_data,
            volumes_df=volumes_data
        )
        
        logger.info("Training regime detection model")
        # Extract main price series for regime detection
        main_symbol = list(prices_data.keys())[0]
        main_prices = prices_data[main_symbol]
        
        # Calculate features for regime detection
        regime_features = self.regime_classifier.calculate_features(
            prices=main_prices,
            volumes=volumes_data.get(main_symbol) if volumes_data else None
        )
        
        # Fit regime classifier
        self.regime_classifier.fit(regime_features)
        
        # Save regime classifier model
        try:
            import joblib
            regime_model_path = os.path.join(self.models_dir, "regime_classifier.pkl")
            joblib.dump(self.regime_classifier.model, regime_model_path)
            logger.info(f"Saved regime classifier model to {regime_model_path}")
        except Exception as e:
            logger.error(f"Error saving regime classifier model: {e}")
    
    def detect_market_regime(self, 
                            prices_data: Dict[str, pd.DataFrame],
                            volumes_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Detect market regime for intraday trading.
        
        Parameters:
        -----------
        prices_data : dict
            Dictionary of price DataFrames for each symbol
        volumes_data : dict, optional
            Dictionary of volume DataFrames for each symbol
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with detected regimes over time
        """
        logger.info("Detecting market regime")
        
        # Extract main price series
        main_symbol = list(prices_data.keys())[0]
        main_prices = prices_data[main_symbol]
        
        # Calculate features for regime detection
        regime_features = self.regime_classifier.calculate_features(
            prices=main_prices,
            volumes=volumes_data.get(main_symbol) if volumes_data else None
        )
        
        # Predict regimes
        regimes = self.regime_classifier.predict(regime_features)
        
        # Store regime history
        self.regimes_history = regimes
        
        # Set current regime to the most recent one
        if not regimes.empty:
            self.current_regime = regimes.iloc[-1]['regime']
            logger.info(f"Current regime: {self.current_regime}")
            
            # Get regime description
            regime_desc = self.regime_classifier.get_regime_description(self.current_regime)
            logger.info(f"Regime description: {regime_desc}")
        
        return regimes
    
    def adapt_parameters(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Adapt trading parameters based on current market regime.
        
        Parameters:
        -----------
        current_time : datetime, optional
            Current time (defaults to now)
            
        Returns:
        --------
        dict
            Adapted parameters
        """
        current_time = current_time or datetime.now()
        
        # Check if we need to adapt parameters yet
        if self.last_adaptation_time is not None:
            if self.adaptation_frequency == "daily":
                if (current_time.date() == self.last_adaptation_time.date()):
                    # We already adapted today
                    return self.current_parameters
            elif self.adaptation_frequency == "hourly":
                if (current_time - self.last_adaptation_time).total_seconds() < 3600:
                    # We already adapted within the last hour
                    return self.current_parameters
        
        logger.info("Adapting parameters based on current regime")
        
        # If no current regime detected, use default parameters
        if self.current_regime is None:
            logger.warning("No regime detected, using default parameters")
            self.current_parameters = self.config.get("default_parameters", {})
            return self.current_parameters
        
        # Get parameters for current regime
        regime_parameters = self.regime_classifier.get_regime_parameters(self.current_regime)
        
        # Merge with default parameters
        default_parameters = self.config.get("default_parameters", {})
        adapted_parameters = {**default_parameters, **regime_parameters}
        
        # Update current parameters
        self.current_parameters = adapted_parameters
        
        # Update last adaptation time
        self.last_adaptation_time = current_time
        
        logger.info(f"Adapted parameters: {adapted_parameters}")
        return adapted_parameters
    
    def enhance_signals(self, 
                       original_signals: pd.DataFrame, 
                       prices_data: Dict[str, pd.DataFrame], 
                       spreads_data: pd.DataFrame,
                       volumes_data: Optional[Dict[str, pd.DataFrame]] = None,
                       current_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Enhance trading signals using ML models.
        
        Parameters:
        -----------
        original_signals : pd.DataFrame
            DataFrame with original trading signals
        prices_data : dict
            Dictionary of price DataFrames for each symbol
        spreads_data : pd.DataFrame
            DataFrame with spread data
        volumes_data : dict, optional
            Dictionary of volume DataFrames for each symbol
        current_time : datetime, optional
            Current time (defaults to now)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with enhanced signals
        """
        logger.info("Enhancing signals with ML models")
        
        # Process intraday signals
        enhanced_signals = self.signal_processor.process_intraday_signals(
            prices_df=prices_data,
            spreads_df=spreads_data,
            original_signals=original_signals,
            volumes_df=volumes_data
        )
        
        # Apply intraday adaptations based on current regime
        adapted_parameters = self.adapt_parameters(current_time)
        
        # Calculate features for signal enhancement
        features = self.signal_enhancer.calculate_features(
            prices_df=prices_data,
            spreads_df=spreads_data,
            volumes_df=volumes_data
        )
        
        # Apply regime-specific adaptations
        adapted_signals = self.signal_enhancer.apply_intraday_adaptations(
            signals=enhanced_signals,
            features=features,
            prices_df=prices_data,
            current_time=current_time,
            volumes_df=volumes_data
        )
        
        return adapted_signals
    
    def run_backtest(self, 
                    original_signals: pd.DataFrame, 
                    prices_data: Dict[str, pd.DataFrame], 
                    spreads_data: pd.DataFrame,
                    volumes_data: Optional[Dict[str, pd.DataFrame]] = None,
                    save_results: bool = True) -> Dict:
        """
        Run ML-enhanced intraday backtest.
        
        Parameters:
        -----------
        original_signals : pd.DataFrame
            DataFrame with original trading signals
        prices_data : dict
            Dictionary of price DataFrames for each symbol
        spreads_data : pd.DataFrame
            DataFrame with spread data
        volumes_data : dict, optional
            Dictionary of volume DataFrames for each symbol
        save_results : bool
            Whether to save backtest results
            
        Returns:
        --------
        dict
            Backtest results
        """
        logger.info("Running ML-enhanced intraday backtest")
        
        # Detect market regimes
        regimes = self.detect_market_regime(prices_data, volumes_data)
        
        # Enhance signals with ML
        enhanced_signals = self.enhance_signals(
            original_signals=original_signals,
            prices_data=prices_data,
            spreads_data=spreads_data,
            volumes_data=volumes_data
        )
        
        # Run backtest
        self.backtest_engine.signals = enhanced_signals
        self.backtest_engine.prices = prices_data
        self.backtest_engine.volume_data = volumes_data
        
        backtest_results = self.backtest_engine.run_backtest()
        
        # Calculate detailed metrics
        detailed_metrics = self.backtest_engine.calculate_detailed_metrics()
        backtest_results['detailed_metrics'] = detailed_metrics
        
        if save_results:
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save signals
            signal_file = os.path.join(self.output_dir, f"enhanced_signals_{timestamp}.csv")
            enhanced_signals.to_csv(signal_file)
            
            # Save regimes
            regime_file = os.path.join(self.output_dir, f"regimes_{timestamp}.csv")
            regimes.to_csv(regime_file)
            
            # Save detailed metrics
            metrics_file = os.path.join(self.output_dir, f"metrics_{timestamp}.json")
            with open(metrics_file, 'w') as f:
                import json
                json.dump(detailed_metrics, f, indent=4, default=str)
            
            # Create visualization dashboard
            dashboard_figures = create_intraday_performance_dashboard(
                backtest_results=backtest_results,
                regime_data=regimes,
                figsize=(12, 8)
            )
            
            # Save dashboard
            dashboard_dir = os.path.join(self.output_dir, f"dashboard_{timestamp}")
            os.makedirs(dashboard_dir, exist_ok=True)
            save_performance_dashboard(
                figures=dashboard_figures,
                output_dir=dashboard_dir,
                prefix="intraday_ml"
            )
            
            logger.info(f"Saved backtest results to {self.output_dir}")
            
            # Add saved file paths to results
            backtest_results['saved_files'] = {
                'signals': signal_file,
                'regimes': regime_file,
                'metrics': metrics_file,
                'dashboard': dashboard_dir
            }
        
        return backtest_results
    
    def generate_intraday_trading_plan(self, current_date: Optional[datetime] = None) -> Dict:
        """
        Generate intraday trading plan for the current day.
        
        Parameters:
        -----------
        current_date : datetime, optional
            Current date (defaults to today)
            
        Returns:
        --------
        dict
            Trading plan with adapted parameters and schedules
        """
        current_date = current_date or datetime.now().date()
        logger.info(f"Generating intraday trading plan for {current_date}")
        
        # Ensure we have adapted parameters
        adapted_parameters = self.adapt_parameters(datetime.now())
        
        # Extract time filters from parameters
        time_filters = adapted_parameters.get("time_filters", {})
        
        # Create trading schedule
        trading_schedule = {}
        market_open = time(9, 30)  # Default market open (9:30 AM)
        market_close = time(16, 0)  # Default market close (4:00 PM)
        
        # Avoid first 15 minutes if configured
        if time_filters.get("avoid_first_15min", True):
            first_entry = (datetime.combine(current_date, market_open) + timedelta(minutes=15)).time()
        else:
            first_entry = market_open
        
        # Set last exit before market close
        exit_buffer = adapted_parameters.get("exit_buffer_minutes", 15)
        last_exit = (datetime.combine(current_date, market_close) - timedelta(minutes=exit_buffer)).time()
        
        # Calculate max holding period in minutes
        max_holding = adapted_parameters.get("max_holding_period", 180)  # Default 3 hours
        
        # Create high liquidity windows
        high_liquidity = time_filters.get("high_liquidity_windows", [
            {"start": "09:45", "end": "11:30"},
            {"start": "13:30", "end": "15:45"}
        ])
        
        # Convert string times to time objects
        liquidity_windows = []
        for window in high_liquidity:
            start_hour, start_minute = map(int, window["start"].split(":"))
            end_hour, end_minute = map(int, window["end"].split(":"))
            liquidity_windows.append({
                "start": time(start_hour, start_minute),
                "end": time(end_hour, end_minute)
            })
        
        # Build trading schedule
        trading_schedule = {
            "date": current_date.strftime("%Y-%m-%d"),
            "market_open": market_open.strftime("%H:%M"),
            "market_close": market_close.strftime("%H:%M"),
            "first_entry": first_entry.strftime("%H:%M"),
            "last_exit": last_exit.strftime("%H:%M"),
            "max_holding_minutes": max_holding,
            "high_liquidity_windows": [
                {
                    "start": window["start"].strftime("%H:%M"),
                    "end": window["end"].strftime("%H:%M")
                }
                for window in liquidity_windows
            ],
            "avoid_lunch_hour": time_filters.get("avoid_lunch_hour", True)
        }
        
        # Build trading plan
        trading_plan = {
            "date": current_date.strftime("%Y-%m-%d"),
            "current_regime": int(self.current_regime) if self.current_regime is not None else None,
            "regime_description": self.regime_classifier.get_regime_description(self.current_regime) if self.current_regime is not None else "Unknown",
            "trading_schedule": trading_schedule,
            "trading_parameters": {
                "entry_threshold": adapted_parameters.get("entry_threshold", 2.0),
                "exit_threshold": adapted_parameters.get("exit_threshold", 0.5),
                "stop_loss": adapted_parameters.get("stop_loss", 3.0),
                "position_size": adapted_parameters.get("position_size", 0.1)
            }
        }
        
        # Save trading plan
        plan_file = os.path.join(self.output_dir, f"trading_plan_{current_date.strftime('%Y%m%d')}.json")
        with open(plan_file, 'w') as f:
            import json
            json.dump(trading_plan, f, indent=4)
        
        logger.info(f"Saved trading plan to {plan_file}")
        
        return trading_plan
    

def integrate_with_paper_trading(ml_system: IntradayMLSystem, paper_trading_config: Dict) -> Dict:
    """
    Integrate ML system with paper trading.
    
    Parameters:
    -----------
    ml_system : IntradayMLSystem
        The ML system to integrate
    paper_trading_config : dict
        Paper trading configuration
        
    Returns:
    --------
    dict
        Updated paper trading configuration
    """
    # Generate trading plan
    trading_plan = ml_system.generate_intraday_trading_plan()
    
    # Extract current regime
    current_regime = trading_plan.get("current_regime")
    
    # Get regime-specific parameters
    if current_regime is not None:
        regime_parameters = ml_system.regime_classifier.get_regime_parameters(current_regime)
    else:
        regime_parameters = {}
    
    # Update paper trading configuration
    updated_config = paper_trading_config.copy()
    
    # Add ML-specific configuration
    updated_config["ml_config"] = {
        "use_ml_signals": True,
        "current_regime": current_regime,
        "trading_plan": trading_plan
    }
    
    # Add regime-specific parameters
    if regime_parameters:
        # Entry/exit thresholds
        if "entry_threshold" in regime_parameters:
            for pair in updated_config.get("pairs", []):
                pair["config"]["entry_zscore"] = regime_parameters["entry_threshold"]
        
        if "exit_threshold" in regime_parameters:
            for pair in updated_config.get("pairs", []):
                pair["config"]["exit_zscore"] = regime_parameters["exit_threshold"]
        
        # Stop loss
        if "stop_loss" in regime_parameters:
            for pair in updated_config.get("pairs", []):
                pair["config"]["stop_loss_std"] = regime_parameters["stop_loss"]
        
        # Position size
        if "position_size" in regime_parameters:
            updated_config["max_allocation_per_pair"] = regime_parameters["position_size"]
    
    # Add trading schedule
    trading_schedule = trading_plan.get("trading_schedule", {})
    if trading_schedule:
        # Add time filters
        intraday_params = {
            "max_holding_period": trading_schedule.get("max_holding_minutes", 180),
            "time_filters": {
                "avoid_first_15min": True,
                "avoid_lunch_hour": trading_schedule.get("avoid_lunch_hour", True),
                "high_liquidity_windows": trading_schedule.get("high_liquidity_windows", [])
            },
            "exit_buffer_minutes": 15
        }
        
        # Update pairs configuration
        for pair in updated_config.get("pairs", []):
            pair["config"]["intraday_params"] = intraday_params
    
    return updated_config 