"""
Signal Enhancer Component for the Paper Trading module.

This component is responsible for generating and enhancing trading signals
using ML models and market regime detection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from src.ml_enhancements.intraday_integration import IntradayMLSystem
from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Configure logging
logger = logging.getLogger(__name__)


class SignalEnhancer:
    """
    Component responsible for generating and enhancing trading signals.
    
    This class extracts signal generation and enhancement functionality from
    the IntradayMLPaperTrader class, providing a focused interface for
    working with trading signals.
    """
    
    def __init__(
        self,
        ml_system: Optional[IntradayMLSystem] = None,
        regime_classifier: Optional[MarketRegimeClassifier] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the signal enhancer component.
        
        Parameters:
        -----------
        ml_system : IntradayMLSystem, optional
            ML system for signal enhancement
        regime_classifier : MarketRegimeClassifier, optional
            Classifier for market regime detection
        config : dict, optional
            Configuration options
        """
        self.ml_system = ml_system
        self.regime_classifier = regime_classifier
        self.config = config or {}
        self.regime = "unknown"
        self.regime_parameters = {}
    
    def process_market_data(
        self,
        pair_id: str,
        prices_df: pd.DataFrame,
        timeframe: str = '5min'
    ) -> pd.DataFrame:
        """
        Process market data for a trading pair.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        prices_df : pd.DataFrame
            DataFrame with price data
        timeframe : str, optional
            Timeframe for processing (e.g., '5min', '1hour')
            
        Returns:
        --------
        pd.DataFrame
            Processed market data with signals
        """
        if self.ml_system is None:
            logger.warning("ML system not available for processing market data")
            return pd.DataFrame()
        
        try:
            # Update market regime
            self.detect_regime(prices_df)
            
            # Process data with ML system
            processed_data = self.ml_system.process_data(
                pair_id=pair_id,
                prices_df=prices_df,
                timeframe=timeframe
            )
            
            logger.info(f"Processed market data for {pair_id}: {len(processed_data)} rows")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing market data for {pair_id}: {str(e)}")
            return pd.DataFrame()
    
    def detect_regime(self, prices_df: pd.DataFrame) -> str:
        """
        Detect the current market regime.
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        str
            Market regime identifier
        """
        if self.regime_classifier is None:
            logger.warning("Regime classifier not available")
            return "unknown"
        
        try:
            # Detect regime
            self.regime = self.regime_classifier.predict_regime(prices_df)
            
            # Get parameters for the regime
            self.regime_parameters = self.regime_classifier.get_regime_parameters(self.regime)
            
            logger.info(f"Detected market regime: {self.regime}")
            return self.regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown"
    
    def get_current_regime(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the current market regime and parameters.
        
        Returns:
        --------
        tuple
            (regime identifier, regime parameters)
        """
        return self.regime, self.regime_parameters
    
    def generate_signals(
        self,
        pair_id: str,
        processed_data: pd.DataFrame
    ) -> pd.Series:
        """
        Generate trading signals for a pair.
        
        Parameters:
        -----------
        pair_id : str
            Trading pair identifier
        processed_data : pd.DataFrame
            Processed market data
            
        Returns:
        --------
        pd.Series
            Series with trading signals
        """
        if self.ml_system is None:
            logger.warning("ML system not available for generating signals")
            return pd.Series()
        
        try:
            # Generate baseline signals
            baseline_signals = processed_data.get('signal', pd.Series())
            
            # Apply ML-based enhancements
            enhanced_signals, metrics = self.ml_system.enhance_signals(
                pair_id=pair_id,
                signals=baseline_signals,
                regime=self.regime,
                regime_parameters=self.regime_parameters
            )
            
            # Log enhancement statistics
            if len(baseline_signals) > 0 and len(enhanced_signals) > 0:
                changes = sum((baseline_signals != enhanced_signals).astype(int))
                logger.info(f"Enhanced signals for {pair_id}: {changes} changes out of {len(baseline_signals)} signals")
            
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {pair_id}: {str(e)}")
            return pd.Series()
    
    def adapt_signals_to_regime(
        self,
        signals: pd.Series,
        pair_config: Dict[str, Any]
    ) -> pd.Series:
        """
        Adapt signals based on current market regime.
        
        Parameters:
        -----------
        signals : pd.Series
            Original trading signals
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        pd.Series
            Adapted trading signals
        """
        if self.regime == "unknown":
            return signals
        
        try:
            # Apply different adaptations based on regime
            if self.regime == "volatile":
                # In volatile regimes, we're more conservative
                return self._adapt_to_volatile_regime(signals, pair_config)
                
            elif self.regime == "trending":
                # In trending regimes, we can be more aggressive
                return self._adapt_to_trending_regime(signals, pair_config)
                
            elif self.regime == "low_vol":
                # In low volatility regimes, we can widen thresholds
                return self._adapt_to_low_vol_regime(signals, pair_config)
            
            # Default: return original signals
            return signals
            
        except Exception as e:
            logger.error(f"Error adapting signals to regime: {str(e)}")
            return signals
    
    def _adapt_to_volatile_regime(
        self, 
        signals: pd.Series, 
        pair_config: Dict[str, Any]
    ) -> pd.Series:
        """
        Adapt signals for volatile market regime.
        
        In volatile regimes, we reduce position size and increase thresholds.
        
        Parameters:
        -----------
        signals : pd.Series
            Original trading signals
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        pd.Series
            Adapted trading signals
        """
        # In volatile regimes, we might want to be more conservative
        # One approach: filter out some signals to reduce trading frequency
        
        # For example, skip every other signal
        adapted_signals = signals.copy()
        adapted_signals.iloc[::2] = 0
        
        return adapted_signals
    
    def _adapt_to_trending_regime(
        self, 
        signals: pd.Series, 
        pair_config: Dict[str, Any]
    ) -> pd.Series:
        """
        Adapt signals for trending market regime.
        
        In trending regimes, we might hold positions longer.
        
        Parameters:
        -----------
        signals : pd.Series
            Original trading signals
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        pd.Series
            Adapted trading signals
        """
        # In trending regimes, we might want to hold positions longer
        # For now, we just return the original signals
        return signals
    
    def _adapt_to_low_vol_regime(
        self, 
        signals: pd.Series, 
        pair_config: Dict[str, Any]
    ) -> pd.Series:
        """
        Adapt signals for low volatility market regime.
        
        In low volatility regimes, we might use tighter thresholds.
        
        Parameters:
        -----------
        signals : pd.Series
            Original trading signals
        pair_config : dict
            Pair configuration
            
        Returns:
        --------
        pd.Series
            Adapted trading signals
        """
        # In low volatility regimes, we might want to trade more frequently
        # For now, we just return the original signals
        return signals 