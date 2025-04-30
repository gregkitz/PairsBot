"""
Intraday ML-Enhanced Paper Trader

This module extends the base PaperTrader with ML capabilities, real-time
monitoring, and regime-change alerting for intraday trading.
"""

import os
import time
import logging
import threading
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
from pathlib import Path

from .paper_trader import PaperTrader
from ..ml_enhancements.intraday_integration import IntradayMLSystem
from ..ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier
from ..connectors.ib.contract_utils import create_futures_contract, parse_contract_string
from .components.position_manager import PositionManager

# Configure logging
logger = logging.getLogger(__name__)


class IntradayMLPaperTrader:
    """
    Paper trading implementation with ML enhancements for intraday trading.
    
    This class integrates the base PaperTrader with ML models for signal enhancement,
    regime detection, parameter adaptation, and real-time performance monitoring.
    """
    
    def __init__(self,
                initial_capital: float = 100000.0,
                ib_host: str = '127.0.0.1',
                ib_port: int = 7497,
                ib_client_id: int = 1,
                ml_config_path: Optional[str] = None,
                ml_config: Optional[Dict] = None,
                models_dir: str = "models/intraday",
                output_dir: str = "output/paper_trading",
                refresh_interval_seconds: int = 60,
                enable_dashboard: bool = True,
                enable_alerts: bool = True,
                alert_channels: Optional[List[str]] = None,
                dashboard_update_interval_seconds: int = 300,
                auto_shutdown_time: Optional[str] = None,
                test_mode: bool = False):
        """
        Initialize the ML-enhanced paper trader.
        
        Args:
            initial_capital (float): Initial capital for paper trading
            ib_host (str): IB TWS/Gateway host address
            ib_port (int): IB TWS/Gateway port
            ib_client_id (int): IB client ID
            ml_config_path (str, optional): Path to ML configuration file
            ml_config (dict, optional): ML configuration dictionary (overrides file)
            models_dir (str): Directory containing ML models
            output_dir (str): Directory for output files
            refresh_interval_seconds (int): Market data refresh interval
            enable_dashboard (bool): Whether to enable the dashboard
            enable_alerts (bool): Whether to enable alerts
            alert_channels (list): List of alert channels
            dashboard_update_interval_seconds (int): Dashboard update interval
            auto_shutdown_time (str, optional): Time to auto-shutdown (HH:MM)
            test_mode (bool): Whether to run in test mode without real IB connection
        """
        # Initialize core attributes
        self.output_dir = output_dir
        self.models_dir = os.path.abspath(models_dir)
        self.enable_dashboard = enable_dashboard
        self.enable_alerts = enable_alerts
        self.alert_channels = alert_channels or ["console"]
        self.refresh_interval_seconds = refresh_interval_seconds
        self.dashboard_update_interval_seconds = dashboard_update_interval_seconds
        self.test_mode = test_mode
        
        # Create output directories
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "dashboard"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "signals"), exist_ok=True)
        
        # Load ML configuration
        self.ml_config = ml_config or self._load_ml_config(ml_config_path)
        
        # Create base paper trader
        if test_mode:
            # In test mode, don't try to connect to IB
            self.paper_trader = PaperTrader(
                initial_capital=initial_capital,
                ib_host='127.0.0.1',  # Doesn't matter in test mode
                ib_port=7497,         # Doesn't matter in test mode
                ib_client_id=0,       # Doesn't matter in test mode
                data_directory=os.path.join(output_dir, "data"),
                auto_shutdown_time=auto_shutdown_time
            )
            # Manually set the paper trader as running for test mode
            self.paper_trader.is_running = lambda: True
        else:
            # In normal mode, connect to IB
            self.paper_trader = PaperTrader(
                initial_capital=initial_capital,
                ib_host=ib_host,
                ib_port=ib_port,
                ib_client_id=ib_client_id,
                data_directory=os.path.join(output_dir, "data"),
                auto_shutdown_time=auto_shutdown_time
            )
        
        # Initialize PositionManager
        self.position_manager = PositionManager(
            paper_trader=self.paper_trader,
            config={
                'default_position_size': self.current_trading_plan.get('trading_parameters', {}).get('position_size', 0.1)
            }
        )
        
        # Setup ML components
        self._setup_ml_components()
        
        # Setup callbacks
        self._setup_callbacks()
        
        # Initialize trading state
        self.pairs = []
        self.symbols = set()
        self.regime = "Unknown"
        self.regime_history = []
        self.signal_history = {}
        self.parameter_history = {}
        
        # Initialize threading
        self.is_running = False
        self.refresh_thread = None
        self.stop_event = threading.Event()
    
    def _load_ml_config(self, config_path: Optional[str]) -> Dict:
        """
        Load ML configuration from file or use defaults.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to ML configuration file
            
        Returns:
        --------
        dict
            ML configuration
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded ML configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading ML configuration: {e}")
        
        # Use default configuration
        logger.info("Using default ML configuration")
        return {
            "signal_enhancer": {
                "feature_lookback": 20,
                "prediction_threshold": 0.6,
                "use_rsi_filter": True,
                "use_volume_filter": True,
                "use_volatility_filter": True,
                "enable_ml_filtering": True
            },
            "signal_processor": {
                "enable_ml_enhancement": True,
                "enable_regime_adaptation": True
            },
            "regime_detection": {
                "n_regimes": 3,
                "lookback_window": 60,
                "method": "kmeans",
                "features": [
                    "volatility_20d",
                    "rsi_14d",
                    "atr_14d",
                    "hurst_exponent",
                    "mean_reversion_strength"
                ]
            },
            "default_parameters": {
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "stop_loss": 3.0,
                "position_size": 0.1,
                "time_filters": {
                    "avoid_first_15min": True,
                    "avoid_lunch_hour": True,
                    "high_liquidity_windows": [
                        {"start": "09:45", "end": "11:30"},
                        {"start": "13:30", "end": "15:45"}
                    ]
                }
            },
            "adaptation_frequency": "daily"
        }
    
    def _setup_callbacks(self):
        """Set up callbacks for the paper trader."""
        self.paper_trader.on_account_update = self._on_account_update
        self.paper_trader.on_position_change = self._on_position_change
        self.paper_trader.on_order_update = self._on_order_update
        self.paper_trader.on_market_data_update = self._on_market_data_update
        self.paper_trader.on_error = self._on_error
    
    def _on_account_update(self, account_data: Dict):
        """
        Handle account updates.
        
        Parameters:
        -----------
        account_data : dict
            Account data including equity, cash, and P&L
        """
        # Log account update
        logger.info(f"Account update: Equity=${account_data['equity']:.2f}, "
                   f"Cash=${account_data['cash']:.2f}, "
                   f"Unrealized P&L=${account_data['unrealized_pnl']:.2f}")
        
        # Track performance
        timestamp = datetime.now()
        performance_entry = {
            "timestamp": timestamp,
            "equity": account_data['equity'],
            "cash": account_data['cash'],
            "unrealized_pnl": account_data['unrealized_pnl'],
            "realized_pnl": account_data.get('realized_pnl', 0.0),
            "regime": self.current_regime
        }
        self.performance_history.append(performance_entry)
        
        # Check if dashboard update is needed
        if self.enable_dashboard:
            if (self.last_dashboard_update is None or 
                (timestamp - self.last_dashboard_update).total_seconds() > self.dashboard_update_interval_seconds):
                self._update_dashboard()
                self.last_dashboard_update = timestamp
    
    def _on_position_change(self, symbol: str, position_data: Optional[Dict]):
        """
        Handle position changes.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the instrument
        position_data : dict, optional
            Position data (None if position closed)
        """
        if position_data:
            logger.info(f"Position updated for {symbol}: "
                      f"Quantity={position_data['quantity']}, "
                      f"Entry Price=${position_data['entry_price']:.2f}, "
                      f"Current Price=${position_data['current_price']:.2f}, "
                      f"Unrealized P&L=${position_data['unrealized_pnl']:.2f}")
        else:
            logger.info(f"Position closed for {symbol}")
        
        # Track trade
        timestamp = datetime.now()
        trade_entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "position": position_data['quantity'] if position_data else 0,
            "entry_price": position_data['entry_price'] if position_data else None,
            "current_price": position_data['current_price'] if position_data else None,
            "unrealized_pnl": position_data['unrealized_pnl'] if position_data else 0.0,
            "regime": self.current_regime
        }
        self.trade_history.append(trade_entry)
    
    def _on_order_update(self, order_id: str, order_data: Dict):
        """
        Handle order updates.
        
        Parameters:
        -----------
        order_id : str
            Order ID
        order_data : dict
            Order data
        """
        logger.info(f"Order update for {order_id}: "
                  f"Status={order_data['status']}, "
                  f"Filled={order_data['filled']}/{order_data['quantity']}")
    
    def _on_market_data_update(self, symbol: str, market_data: Dict):
        """
        Handle market data updates.
        
        Parameters:
        -----------
        symbol : str
            Symbol of the instrument
        market_data : dict
            Market data
        """
        # Check if refresh is needed
        current_time = datetime.now()
        
        if (self.last_refresh_time is None or 
            (current_time - self.last_refresh_time).total_seconds() > self.refresh_interval_seconds):
            
            # Refresh ML system
            self._refresh_ml_system()
            self.last_refresh_time = current_time
    
    def _on_error(self, error_msg: str, error_code: Optional[int] = None):
        """
        Handle errors.
        
        Parameters:
        -----------
        error_msg : str
            Error message
        error_code : int, optional
            Error code
        """
        logger.error(f"Error: {error_msg} (Code: {error_code})")
        
        # Send alert if enabled
        if self.enable_alerts:
            self._send_alert(
                title="Trading Error",
                message=f"Error: {error_msg} (Code: {error_code})",
                level="error"
            )
    
    def _refresh_ml_system(self):
        """Refresh ML system with latest market data and update trading plan."""
        try:
            # Collect market data for ML analysis
            market_data = self._collect_market_data()
            
            if not market_data:
                logger.warning("No market data available for ML refresh")
                return
            
            # Detect market regime
            regime_data = self._detect_market_regime(market_data)
            
            # Check for regime change
            prev_regime = self.current_regime
            if regime_data is not None and not regime_data.empty:
                self.current_regime = regime_data.iloc[-1]['regime']
                
                # Send alert if regime changed
                if prev_regime is not None and self.current_regime != prev_regime:
                    regime_desc = self.regime_classifier.get_regime_description(self.current_regime)
                    self._send_alert(
                        title="Regime Change Detected",
                        message=f"Market regime changed from {prev_regime} to {self.current_regime}: {regime_desc}",
                        level="info"
                    )
            
            # Update trading plan
            self.current_trading_plan = self.ml_system.generate_intraday_trading_plan()
            
            # Generate trading signals
            self._generate_trading_signals(market_data)
            
            # Log refresh
            logger.info(f"ML system refreshed. Current regime: {self.current_regime}")
            
        except Exception as e:
            logger.error(f"Error refreshing ML system: {e}")
            if self.enable_alerts:
                self._send_alert(
                    title="ML Refresh Error",
                    message=f"Error refreshing ML system: {e}",
                    level="error"
                )
    
    def _collect_market_data(self) -> Dict:
        """
        Collect market data from paper trader for ML analysis.
        
        Returns:
        --------
        dict
            Dictionary with market data
        """
        # Get subscribed symbols
        symbols = self.paper_trader.get_subscribed_symbols()
        
        if not symbols:
            logger.warning("No subscribed symbols found")
            return {}
        
        # Initialize data structures
        prices = {}
        volumes = {}
        spreads = {}
        
        # Get current date for lookback calculation
        current_date = datetime.now().date()
        lookback_days = self.ml_config.get("regime_detection", {}).get("lookback_window", 60)
        start_date = (current_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = current_date.strftime("%Y-%m-%d")
        
        # Get historical data for each symbol
        for symbol in symbols:
            try:
                # Get historical data from paper trader
                historical_data = self.paper_trader.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    bar_size="5 mins"
                )
                
                if historical_data is not None and not historical_data.empty:
                    prices[symbol] = historical_data
                    volumes[symbol] = pd.DataFrame({'volume': historical_data['volume']}, index=historical_data.index)
            except Exception as e:
                logger.error(f"Error getting historical data for {symbol}: {e}")
        
        # Calculate spreads for pairs
        for pair in self.current_pairs:
            try:
                symbol1 = pair["symbol1"]
                symbol2 = pair["symbol2"]
                
                if symbol1 in prices and symbol2 in prices:
                    # Extract price data for the pair
                    price_df1 = prices[symbol1]
                    price_df2 = prices[symbol2]
                    
                    # Calculate spread (delegate to ML system)
                    spread_df = self._calculate_spread(price_df1, price_df2, pair.get("hedge_ratio"))
                    
                    if spread_df is not None and not spread_df.empty:
                        pair_id = f"{symbol1}_{symbol2}"
                        spreads[pair_id] = spread_df
            except Exception as e:
                logger.error(f"Error calculating spread for pair {symbol1}_{symbol2}: {e}")
        
        return {
            'prices': prices,
            'volumes': volumes,
            'spreads': spreads
        }
    
    def _calculate_spread(self, price_df1, price_df2, hedge_ratio=None):
        """
        Calculate spread between two price series.
        
        Parameters:
        -----------
        price_df1 : pd.DataFrame
            Price data for first instrument
        price_df2 : pd.DataFrame
            Price data for second instrument
        hedge_ratio : float, optional
            Hedge ratio (calculated if None)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread data
        """
        try:
            import statsmodels.api as sm
            
            # Align price data
            common_index = price_df1.index.intersection(price_df2.index)
            price1 = price_df1.loc[common_index, 'close']
            price2 = price_df2.loc[common_index, 'close']
            
            # Calculate hedge ratio if not provided
            if hedge_ratio is None:
                # Simple OLS regression
                X = sm.add_constant(price1)
                model = sm.OLS(price2, X).fit()
                hedge_ratio = model.params[1]
            
            # Calculate spread
            spread = price2 - hedge_ratio * price1
            
            # Calculate z-score (standardized spread)
            window = min(30, len(spread))
            mean = spread.rolling(window=window).mean()
            std = spread.rolling(window=window).std()
            zscore = (spread - mean) / std
            
            # Combine into DataFrame
            spread_df = pd.DataFrame({
                'price1': price1,
                'price2': price2,
                'spread': spread,
                'mean': mean,
                'std': std,
                'zscore': zscore
            })
            
            return spread_df
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return None
    
    def _detect_market_regime(self, market_data):
        """
        Detect current market regime.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary with market data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime data
        """
        try:
            # Detect market regime using ML system
            regimes = self.ml_system.detect_market_regime(
                prices_data=market_data['prices'],
                volumes_data=market_data['volumes']
            )
            
            # Track regime history
            if regimes is not None and not regimes.empty:
                current_regime = regimes.iloc[-1]
                timestamp = datetime.now()
                
                regime_entry = {
                    "timestamp": timestamp,
                    "regime": current_regime['regime'],
                    "description": self.regime_classifier.get_regime_description(current_regime['regime'])
                }
                self.regime_history.append(regime_entry)
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return None
    
    def _generate_trading_signals(self, market_data):
        """
        Generate trading signals using ML models.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary with market data
        """
        try:
            # Generate original signals from spreads
            original_signals = {}
            
            for pair_id, spread_df in market_data['spreads'].items():
                # Get trading parameters from current trading plan
                trading_params = self.current_trading_plan.get('trading_parameters', {})
                entry_threshold = trading_params.get('entry_threshold', 2.0)
                exit_threshold = trading_params.get('exit_threshold', 0.5)
                
                # Generate signals
                signals = self._generate_signals_from_spread(
                    spread_df, 
                    entry_threshold=entry_threshold, 
                    exit_threshold=exit_threshold
                )
                
                if signals is not None and not signals.empty:
                    original_signals[pair_id] = signals
            
            if not original_signals:
                logger.warning("No original signals generated")
                return
            
            # Enhance signals with ML
            enhanced_signals = {}
            
            for pair_id, signals in original_signals.items():
                try:
                    # Get pair symbols
                    symbols = pair_id.split('_')
                    if len(symbols) != 2:
                        continue
                    
                    symbol1, symbol2 = symbols
                    
                    # Check if we have data for both symbols
                    if symbol1 not in market_data['prices'] or symbol2 not in market_data['prices']:
                        continue
                    
                    # Get data for this pair
                    pair_prices = {
                        symbol1: market_data['prices'][symbol1],
                        symbol2: market_data['prices'][symbol2]
                    }
                    
                    pair_volumes = {
                        symbol1: market_data['volumes'][symbol1],
                        symbol2: market_data['volumes'][symbol2]
                    }
                    
                    # Enhance signals
                    enhanced = self.ml_system.enhance_signals(
                        original_signals=signals,
                        prices_data=pair_prices,
                        spreads_data=market_data['spreads'][pair_id],
                        volumes_data=pair_volumes
                    )
                    
                    if enhanced is not None and not enhanced.empty:
                        enhanced_signals[pair_id] = enhanced
                        
                        # Track signal changes
                        self._track_signal_changes(
                            pair_id=pair_id,
                            original_signals=signals,
                            enhanced_signals=enhanced
                        )
                        
                except Exception as e:
                    logger.error(f"Error enhancing signals for {pair_id}: {e}")
            
            # Apply signals to paper trader
            self._apply_trading_signals(enhanced_signals)
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
    
    def _generate_signals_from_spread(self, spread_df, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate trading signals from spread data.
        
        Parameters:
        -----------
        spread_df : pd.DataFrame
            DataFrame with spread data
        entry_threshold : float
            Z-score threshold for entries
        exit_threshold : float
            Z-score threshold for exits
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        try:
            # Initialize signals
            signals = pd.DataFrame(0, index=spread_df.index, columns=['pair_signal'])
            
            # Generate signals based on z-score
            position = 0
            
            for i in range(len(spread_df)):
                zscore = spread_df['zscore'].iloc[i]
                
                # Skip if zscore is NaN (e.g., during warmup period)
                if pd.isna(zscore):
                    continue
                
                # Long entry
                if position == 0 and zscore < -entry_threshold:
                    signals.iloc[i] = 1
                    position = 1
                
                # Short entry
                elif position == 0 and zscore > entry_threshold:
                    signals.iloc[i] = -1
                    position = -1
                
                # Long exit
                elif position == 1 and zscore > -exit_threshold:
                    signals.iloc[i] = 0
                    position = 0
                
                # Short exit
                elif position == -1 and zscore < exit_threshold:
                    signals.iloc[i] = 0
                    position = 0
                
                # Maintain position
                else:
                    signals.iloc[i] = position
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals from spread: {e}")
            return None
    
    def _track_signal_changes(self, pair_id, original_signals, enhanced_signals):
        """
        Track changes between original and enhanced signals.
        
        Parameters:
        -----------
        pair_id : str
            Pair identifier
        original_signals : pd.DataFrame
            Original trading signals
        enhanced_signals : pd.DataFrame
            Enhanced trading signals
        """
        # Get the latest signals
        latest_idx = original_signals.index[-1]
        original = original_signals.iloc[-1, 0]
        enhanced = enhanced_signals.iloc[-1, 0]
        
        # Track changes
        if original != enhanced:
            timestamp = datetime.now()
            signal_entry = {
                "timestamp": timestamp,
                "pair_id": pair_id,
                "original_signal": original,
                "enhanced_signal": enhanced,
                "regime": self.current_regime
            }
            self.signal_history.append(signal_entry)
            
            # Log signal change
            logger.info(f"Signal change for {pair_id}: Original={original}, Enhanced={enhanced}")
    
    def _apply_trading_signals(self, enhanced_signals):
        """
        Apply enhanced trading signals to the paper trader.
        
        Parameters:
        -----------
        enhanced_signals : dict
            Dictionary of enhanced signals for each pair
        """
        try:
            # Get active pairs
            for pair in self.current_pairs:
                pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
                
                # Skip if no signals for this pair
                if pair_id not in enhanced_signals:
                    continue
                
                signals = enhanced_signals[pair_id]
                
                # Execute signals using position manager
                result = self.position_manager.execute_signals(pair_id, signals)
                
                # Log execution results
                if result['executed'] > 0:
                    logger.info(f"Executed {result['executed']} signal(s) for {pair_id}")
                elif result['errors'] > 0:
                    logger.warning(f"Errors executing signals for {pair_id}: {result['errors']}")
        
        except Exception as e:
            logger.error(f"Error applying trading signals: {e}")
    
    def _get_current_pair_position(self, pair):
        """
        Get current position for a pair.
        
        Parameters:
        -----------
        pair : dict
            Pair configuration
            
        Returns:
        --------
        int
            Current position (1 for long, -1 for short, 0 for no position)
        """
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        position = self.position_manager.get_position(pair_id)
        
        if position is not None:
            return position['direction']
        
        return 0  # No position
    
    def _close_pair_position(self, pair):
        """
        Close a pair position.
        
        Parameters:
        -----------
        pair : dict
            Pair configuration
        """
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        self.position_manager._exit_position(pair_id, "manual")
    
    def _open_pair_position(self, pair, signal):
        """
        Open a new pair position.
        
        Parameters:
        -----------
        pair : dict
            Pair configuration
        signal : int
            Signal (1 for long, -1 for short)
        """
        pair_id = f"{pair['symbol1']}_{pair['symbol2']}"
        pair_config = self.position_manager.pair_configs.get(pair_id)
        
        if pair_config:
            self.position_manager._enter_position(pair_id, signal, pair_config)
    
    def _update_dashboard(self):
        """Update performance dashboard."""
        if not self.enable_dashboard:
            return
        
        try:
            # Create dashboard directory
            dashboard_dir = os.path.join(self.output_dir, "dashboard")
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Convert performance history to DataFrame
            if not self.performance_history:
                logger.warning("No performance data to update dashboard")
                return
                
            perf_df = pd.DataFrame(self.performance_history)
            
            # Create equity curve plot
            self._create_equity_curve_plot(perf_df, dashboard_dir)
            
            # Create regime performance plot
            self._create_regime_performance_plot(perf_df, dashboard_dir)
            
            # Create trade analysis plot
            self._create_trade_analysis_plot(perf_df, dashboard_dir)
            
            # Create performance metrics table
            self._create_performance_metrics(perf_df, dashboard_dir)
            
            # Create HTML dashboard
            self._create_html_dashboard(dashboard_dir)
            
            logger.info(f"Dashboard updated at {dashboard_dir}")
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    def _create_equity_curve_plot(self, perf_df, dashboard_dir):
        """
        Create equity curve plot.
        
        Parameters:
        -----------
        perf_df : pd.DataFrame
            Performance data
        dashboard_dir : str
            Directory to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(perf_df['timestamp'], perf_df['equity'], label='Equity', linewidth=2)
        
        # Add annotations for regime changes
        prev_regime = None
        for i, row in perf_df.iterrows():
            regime = row['regime']
            if regime != prev_regime and prev_regime is not None:
                plt.axvline(x=row['timestamp'], color='r', linestyle='--', alpha=0.5)
                plt.text(row['timestamp'], perf_df['equity'].max(),
                       f"Regime {regime}", rotation=90, va='top')
            prev_regime = regime
        
        # Add labels and grid
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.title('ML-Enhanced Paper Trading Equity Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plt.savefig(os.path.join(dashboard_dir, 'equity_curve.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    def _create_regime_performance_plot(self, perf_df, dashboard_dir):
        """
        Create regime performance plot.
        
        Parameters:
        -----------
        perf_df : pd.DataFrame
            Performance data
        dashboard_dir : str
            Directory to save the plot
        """
        if 'regime' not in perf_df.columns or perf_df['regime'].isna().all():
            logger.warning("No regime data available for performance plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create a grid of subplots
        gs = plt.GridSpec(2, 2)
        
        # Subplot 1: Performance by regime
        ax1 = plt.subplot(gs[0, 0])
        
        # Calculate returns
        perf_df['returns'] = perf_df['equity'].pct_change()
        
        # Group by regime and calculate metrics
        regime_stats = perf_df.groupby('regime')['returns'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('sum', 'sum'),
            ('count', 'count')
        ])
        
        regime_stats['sharpe'] = regime_stats['mean'] / regime_stats['std']
        
        # Bar chart of Sharpe ratio by regime
        ax1.bar(regime_stats.index.astype(str), regime_stats['sharpe'],
               color='purple', alpha=0.7)
        ax1.set_title('Sharpe Ratio by Regime')
        ax1.set_xlabel('Regime')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Returns by regime
        ax2 = plt.subplot(gs[0, 1])
        
        # Calculate cumulative returns by regime
        regimes = perf_df['regime'].unique()
        for regime in regimes:
            regime_mask = perf_df['regime'] == regime
            regime_returns = (1 + perf_df.loc[regime_mask, 'returns']).cumprod()
            ax2.plot(perf_df.loc[regime_mask, 'timestamp'], regime_returns,
                   label=f'Regime {regime}')
        
        ax2.set_title('Cumulative Returns by Regime')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.gcf().autofmt_xdate()
        
        # Subplot 3: Regime distribution
        ax3 = plt.subplot(gs[1, 0])
        
        regime_counts = perf_df['regime'].value_counts()
        ax3.pie(regime_counts, labels=[f'Regime {r}' for r in regime_counts.index],
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('Time Distribution by Regime')
        
        # Subplot 4: Volatility by regime
        ax4 = plt.subplot(gs[1, 1])
        
        ax4.bar(regime_stats.index.astype(str), regime_stats['std'] * 100,
               color='red', alpha=0.7)
        ax4.set_title('Return Volatility by Regime (%)')
        ax4.set_xlabel('Regime')
        ax4.set_ylabel('Volatility (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dashboard_dir, 'regime_performance.png'), dpi=120, bbox_inches='tight')
        plt.close()
    
    def _create_trade_analysis_plot(self, perf_df, dashboard_dir):
        """Create trade analysis plot."""
        try:
            # Create a new figure
            plt.figure(figsize=(15, 12))
            
            # Create 4 subplots
            gs = plt.GridSpec(2, 2)
            
            # Get position data from position manager
            position_summary = self.position_manager.get_position_summary()
            position_history = self.position_manager.get_position_history()
            
            # Convert trade history to DataFrame
            if not self.trade_history:
                logger.warning("No trade history available for analysis")
                return
            
            trade_df = pd.DataFrame(self.trade_history)
            
            # Subplot 1: P&L distribution
            ax1 = plt.subplot(gs[0, 0])
            if 'pnl' in trade_df.columns:
                ax1.hist(trade_df['pnl'], bins=20, alpha=0.7, color='royalblue')
                ax1.axvline(x=0, color='r', linestyle='--')
                ax1.set_title('P&L Distribution')
                ax1.set_xlabel('P&L ($)')
                ax1.set_ylabel('Frequency')
                
                # Add stats
                mean_pnl = trade_df['pnl'].mean()
                median_pnl = trade_df['pnl'].median()
                win_rate = (trade_df['pnl'] > 0).mean() * 100
                
                ax1.text(0.05, 0.95, f"Mean P&L: ${mean_pnl:.2f}\nMedian P&L: ${median_pnl:.2f}\nWin Rate: {win_rate:.1f}%",
                       transform=ax1.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            
            # Subplot 2: Trade duration
            ax2 = plt.subplot(gs[0, 1])
            if 'duration_minutes' in trade_df.columns:
                ax2.hist(trade_df['duration_minutes'], bins=20, alpha=0.7, color='green')
                ax2.set_title('Trade Duration Distribution')
                ax2.set_xlabel('Duration (minutes)')
                ax2.set_ylabel('Frequency')
                
                # Add stats
                mean_duration = trade_df['duration_minutes'].mean()
                median_duration = trade_df['duration_minutes'].median()
                
                ax2.text(0.05, 0.95, f"Mean Duration: {mean_duration:.1f} min\nMedian Duration: {median_duration:.1f} min",
                       transform=ax2.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            
            # Subplot 3: Position distribution
            ax3 = plt.subplot(gs[1, 0])
            
            # Get current positions from position manager
            current_positions = self.position_manager.get_positions()
            position_count = len(current_positions)
            
            # Count of positions by direction
            position_direction_counts = {'long': 0, 'short': 0, 'no_position': 0}
            for pair_id, pos in current_positions.items():
                if pos['direction'] > 0:
                    position_direction_counts['long'] += 1
                elif pos['direction'] < 0:
                    position_direction_counts['short'] += 1
            
            # Add count of pairs with no position
            position_direction_counts['no_position'] = len(self.current_pairs) - position_count
            
            # Create pie chart of position distribution
            if sum(position_direction_counts.values()) > 0:
                labels = list(position_direction_counts.keys())
                sizes = list(position_direction_counts.values())
                ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                       colors=['forestgreen', 'firebrick', 'lightgray'])
                ax3.set_title('Current Position Distribution')
            
            # Subplot 4: Exit reason distribution
            ax4 = plt.subplot(gs[1, 1])
            
            # Count exit reasons from position history
            exit_reasons = {}
            for pos in position_history:
                reason = pos.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            if exit_reasons:
                # Sort by frequency
                sorted_reasons = sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True)
                labels = [item[0] for item in sorted_reasons]
                counts = [item[1] for item in sorted_reasons]
                
                ax4.bar(labels, counts, color='darkorange')
                ax4.set_title('Exit Reason Distribution')
                ax4.set_ylabel('Count')
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(dashboard_dir, 'trade_analysis.png'), dpi=100, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating trade analysis plot: {e}")
    
    def _create_performance_metrics(self, perf_df, dashboard_dir):
        """
        Create performance metrics.
        
        Parameters:
        -----------
        perf_df : pd.DataFrame
            Performance data
        dashboard_dir : str
            Directory to save the metrics
        """
        # Calculate performance metrics
        metrics = {}
        
        # Basic metrics
        initial_equity = perf_df['equity'].iloc[0] if not perf_df.empty else self.initial_capital
        final_equity = perf_df['equity'].iloc[-1] if not perf_df.empty else initial_equity
        
        metrics['start_date'] = perf_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S') if not perf_df.empty else 'N/A'
        metrics['end_date'] = perf_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') if not perf_df.empty else 'N/A'
        metrics['initial_equity'] = initial_equity
        metrics['final_equity'] = final_equity
        metrics['total_return'] = (final_equity / initial_equity - 1) * 100
        
        # Calculate returns
        if len(perf_df) > 1:
            perf_df['returns'] = perf_df['equity'].pct_change()
            
            # Return statistics
            metrics['mean_return'] = perf_df['returns'].mean() * 100
            metrics['std_return'] = perf_df['returns'].std() * 100
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['std_return'] if metrics['std_return'] > 0 else 0
            
            # Drawdown
            perf_df['high_water_mark'] = perf_df['equity'].cummax()
            perf_df['drawdown'] = (perf_df['equity'] / perf_df['high_water_mark'] - 1) * 100
            metrics['max_drawdown'] = perf_df['drawdown'].min()
        
        # Trade statistics
        if self.trade_history:
            trade_df = pd.DataFrame(self.trade_history)
            metrics['total_trades'] = len(trade_df)
            
            # Win/loss statistics
            if 'unrealized_pnl' in trade_df.columns:
                winning_trades = trade_df[trade_df['unrealized_pnl'] > 0]
                losing_trades = trade_df[trade_df['unrealized_pnl'] < 0]
                
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                metrics['win_rate'] = len(winning_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0
                
                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    metrics['avg_win'] = winning_trades['unrealized_pnl'].mean()
                    metrics['avg_loss'] = losing_trades['unrealized_pnl'].mean()
                    metrics['profit_factor'] = abs(winning_trades['unrealized_pnl'].sum() / losing_trades['unrealized_pnl'].sum()) if losing_trades['unrealized_pnl'].sum() != 0 else float('inf')
        
        # Add regime information
        if 'regime' in perf_df.columns:
            current_regime = perf_df['regime'].iloc[-1] if not perf_df.empty else None
            if current_regime is not None:
                metrics['current_regime'] = int(current_regime)
                regime_desc = self.regime_classifier.get_regime_description(current_regime)
                metrics['regime_description'] = regime_desc
        
        # Save metrics to JSON
        with open(os.path.join(dashboard_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        # Create metrics table HTML
        html = "<html><head><style>"
        html += "table { border-collapse: collapse; width: 100%; }"
        html += "th, td { text-align: left; padding: 8px; }"
        html += "tr:nth-child(even) { background-color: #f2f2f2; }"
        html += "th { background-color: #4CAF50; color: white; }"
        html += ".positive { color: green; }"
        html += ".negative { color: red; }"
        html += "</style></head><body>"
        
        html += "<h1>Performance Metrics</h1>"
        html += "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                if key in ['total_return', 'mean_return', 'std_return', 'max_drawdown', 'win_rate']:
                    formatted_value = f"{value:.2f}%"
                else:
                    formatted_value = f"{value:.2f}"
                    
                # Add color class
                if key in ['total_return', 'mean_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'avg_win']:
                    class_name = "positive" if value > 0 else "negative"
                    formatted_value = f"<span class='{class_name}'>{formatted_value}</span>"
                elif key in ['max_drawdown', 'avg_loss']:
                    class_name = "negative" if value < 0 else "positive"
                    formatted_value = f"<span class='{class_name}'>{formatted_value}</span>"
            else:
                formatted_value = str(value)
            
            # Format key for display
            display_key = ' '.join(word.capitalize() for word in key.split('_'))
            
            html += f"<tr><td>{display_key}</td><td>{formatted_value}</td></tr>"
        
        html += "</table></body></html>"
        
        # Save HTML
        with open(os.path.join(dashboard_dir, 'performance_metrics.html'), 'w') as f:
            f.write(html)
    
    def _create_html_dashboard(self, dashboard_dir):
        """
        Create HTML dashboard.
        
        Parameters:
        -----------
        dashboard_dir : str
            Directory to save the dashboard
        """
        # Get timestamp for refresh
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML-Enhanced Paper Trading Dashboard</title>
            <meta http-equiv="refresh" content="{self.dashboard_update_interval_seconds}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-container {{ display: flex; flex-wrap: wrap; }}
                .dashboard-item {{ margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .dashboard-item img {{ max-width: 100%; }}
                h1, h2 {{ color: #333; }}
                .metrics-container {{ margin: 20px 0; }}
                iframe {{ border: none; width: 100%; height: 400px; }}
                .timestamp {{ color: #666; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <h1>ML-Enhanced Paper Trading Dashboard</h1>
            <p class="timestamp">Last updated: {timestamp}</p>
            
            <div class="dashboard-container">
                <div class="dashboard-item">
                    <h2>Equity Curve</h2>
                    <img src="equity_curve.png" alt="Equity Curve">
                </div>
                
                <div class="dashboard-item">
                    <h2>Regime Performance</h2>
                    <img src="regime_performance.png" alt="Regime Performance">
                </div>
                
                <div class="dashboard-item">
                    <h2>Trade Analysis</h2>
                    <img src="trade_analysis.png" alt="Trade Analysis">
                </div>
            </div>
            
            <div class="metrics-container">
                <h2>Performance Metrics</h2>
                <iframe src="performance_metrics.html"></iframe>
            </div>
        </body>
        </html>
        """
        
        # Save HTML
        with open(os.path.join(dashboard_dir, 'index.html'), 'w') as f:
            f.write(html)
    
    def _send_alert(self, title, message, level='info'):
        """
        Send alert to configured channels.
        
        Parameters:
        -----------
        title : str
            Alert title
        message : str
            Alert message
        level : str
            Alert level ('info', 'warning', 'error')
        """
        if not self.enable_alerts:
            return
        
        # Log the alert
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"[{level.upper()}] {timestamp} - {title}: {message}"
        
        if level == 'info':
            logger.info(alert_message)
        elif level == 'warning':
            logger.warning(alert_message)
        elif level == 'error':
            logger.error(alert_message)
        else:
            logger.info(alert_message)
        
        # Send to console if enabled
        if 'console' in self.alert_channels:
            # Already logged, no need to do anything else
            pass
        
        # Send to file if enabled
        if 'file' in self.alert_channels:
            alerts_file = os.path.join(self.output_dir, "logs", "alerts.log")
            with open(alerts_file, 'a') as f:
                f.write(f"{alert_message}\n")
        
        # Send to email if enabled (simplified implementation)
        if 'email' in self.alert_channels:
            email_config = self.ml_config.get("alerts", {}).get("email", {})
            if email_config.get("enabled", False):
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    
                    sender = email_config.get("sender")
                    recipient = email_config.get("recipient")
                    smtp_server = email_config.get("smtp_server")
                    smtp_port = email_config.get("smtp_port", 587)
                    username = email_config.get("username")
                    password = email_config.get("password")
                    
                    if sender and recipient and smtp_server and username and password:
                        msg = MIMEText(message)
                        msg['Subject'] = f"[{level.upper()}] {title}"
                        msg['From'] = sender
                        msg['To'] = recipient
                        
                        server = smtplib.SMTP(smtp_server, smtp_port)
                        server.starttls()
                        server.login(username, password)
                        server.send_message(msg)
                        server.quit()
                        
                        logger.info(f"Alert email sent to {recipient}")
                    else:
                        logger.warning("Incomplete email configuration, alert not sent")
                except Exception as e:
                    logger.error(f"Error sending email alert: {e}")
    
    def set_pairs(self, pairs):
        """
        Set the pairs to trade.
        
        Args:
            pairs (list): List of pairs configuration dictionaries
        """
        self.pairs = pairs
        self.current_pairs = pairs
        self.symbols = set()
        
        # Add pairs to the system
        for pair in pairs:
            pair_id = pair.get('pair_id')
            symbol1 = pair.get('symbol1')
            symbol2 = pair.get('symbol2')
            full_symbol1 = pair.get('full_symbol1')
            full_symbol2 = pair.get('full_symbol2')
            
            if not all([pair_id, symbol1, symbol2]):
                logger.warning(f"Skipping invalid pair: {pair}")
                continue
            
            # Add the symbols to our set
            self.symbols.add(symbol1)
            self.symbols.add(symbol2)
            
            if not self.test_mode:
                # Normal mode - attempt to subscribe to market data
                # If full symbol specifications are provided, log them but use basic subscription
                if full_symbol1:
                    try:
                        logger.info(f"Using contract specification for {symbol1}: {full_symbol1}")
                        # Just subscribe with the symbol for now, later we can enhance the PaperTrader
                        # to support contract specifications
                        self.paper_trader.subscribe_market_data(symbol1)
                    except Exception as e:
                        logger.error(f"Error subscribing to {symbol1}: {e}")
                else:
                    try:
                        self.paper_trader.subscribe_market_data(symbol1)
                    except Exception as e:
                        logger.error(f"Error subscribing to {symbol1}: {e}")
                
                if full_symbol2:
                    try:
                        logger.info(f"Using contract specification for {symbol2}: {full_symbol2}")
                        self.paper_trader.subscribe_market_data(symbol2)
                    except Exception as e:
                        logger.error(f"Error subscribing to {symbol2}: {e}")
                else:
                    try:
                        self.paper_trader.subscribe_market_data(symbol2)
                    except Exception as e:
                        logger.error(f"Error subscribing to {symbol2}: {e}")
            
            logger.info(f"Added pair {pair_id} ({symbol1}/{symbol2}) to trading system")
    
    def start(self):
        """
        Start the ML-enhanced paper trader.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        # Don't start the paper trader if it's already running
        if self.is_running:
            logger.warning("ML-enhanced paper trader is already running")
            return True
        
        if not self.test_mode:
            # In normal mode, start the paper trader
            if not self.paper_trader.start():
                logger.error("Failed to start paper trader")
                return False
        
        # Start the refresh loop
        self.is_running = True
        self.stop_event.clear()
        self.refresh_thread = threading.Thread(target=self._refresh_loop)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()
        
        logger.info("ML-enhanced paper trader started")
        return True
    
    def stop(self):
        """
        Stop the ML-enhanced paper trader.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            logger.warning("ML-enhanced paper trader is not running")
            return True
        
        # Signal the refresh thread to stop
        self.stop_event.set()
        
        # Wait for the refresh thread to finish
        if self.refresh_thread and self.refresh_thread.is_alive():
            self.refresh_thread.join(timeout=5.0)
        
        # Stop the paper trader if not in test mode
        if not self.test_mode:
            self.paper_trader.stop()
        
        # Set running flag
        self.is_running = False
        
        logger.info("ML-enhanced paper trader stopped")
        return True
    
    def _refresh_loop(self):
        """
        Background thread that refreshes ML system periodically.
        """
        last_dashboard_update = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Refresh ML system
                self._refresh_ml_system()
                
                # Update dashboard if enabled and it's time
                if self.enable_dashboard and (time.time() - last_dashboard_update) >= self.dashboard_update_interval_seconds:
                    self._update_dashboard()
                    last_dashboard_update = time.time()
                
                # Sleep for refresh interval
                self.stop_event.wait(self.refresh_interval_seconds)
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}", exc_info=True)
                self.stop_event.wait(self.refresh_interval_seconds)
        
        # Final dashboard update
        if self.enable_dashboard:
            try:
                self._update_dashboard()
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}", exc_info=True)
    
    def _setup_ml_components(self):
        """
        Initialize ML components including regime detection and signal enhancement.
        """
        # Initialize ML system
        self.ml_system = IntradayMLSystem(
            config=self.ml_config,
            models_dir=self.models_dir,
            output_dir=os.path.join(self.output_dir, "ml")
        )
        
        # Initialize regime classifier
        regime_config = self.ml_config.get("regime_detection", {})
        self.regime_classifier = MarketRegimeClassifier(
            n_regimes=regime_config.get("n_regimes", 3),
            lookback_window=regime_config.get("lookback_window", 60),
            features=regime_config.get("features", None),
            method=regime_config.get("method", "kmeans")
        )
        
        # Load trained regime classifier model if available
        regime_model_path = os.path.join(self.models_dir, "market_regime_classifier.joblib")
        if os.path.exists(regime_model_path):
            try:
                import joblib
                logger.info(f"Loading market regime classifier from {regime_model_path}")
                self.regime_classifier = joblib.load(regime_model_path)
                logger.info("Successfully loaded market regime classifier model")
                
                # Try to load current regime information if available
                current_regime_path = os.path.join(self.models_dir, "current_regime.json")
                if os.path.exists(current_regime_path):
                    try:
                        with open(current_regime_path, 'r') as f:
                            regime_info = json.load(f)
                            self.current_regime = regime_info.get('regime_description', "Low Volatility / Weak Trend")
                            logger.info(f"Loaded current regime: {self.current_regime}")
                    except Exception as regime_err:
                        logger.warning(f"Could not load current regime: {regime_err}")
            except Exception as e:
                logger.warning(f"Could not load market regime classifier model: {e}")
        
        # Load pre-trained ML models
        self.ml_system.load_models()
        
        # Performance tracking
        self.performance_history = []
        self.last_refresh_time = None
        self.last_dashboard_update = None
        self.trade_history = []
        self.current_regime = None
        self.current_trading_plan = None
        self.current_pairs = []

    def _load_trading_plan(self):
        """Load trading plan from configuration."""
        try:
            # Load trading plan from ML config
            self.current_trading_plan = self.ml_config.get('trading_plan', {})
            
            # Apply defaults if needed
            if 'trading_parameters' not in self.current_trading_plan:
                self.current_trading_plan['trading_parameters'] = {}
            
            if 'max_positions' not in self.current_trading_plan['trading_parameters']:
                self.current_trading_plan['trading_parameters']['max_positions'] = 3
            
            if 'position_size' not in self.current_trading_plan['trading_parameters']:
                self.current_trading_plan['trading_parameters']['position_size'] = 0.1
            
            logger.info(f"Loaded trading plan: {self.current_trading_plan}")
            
        except Exception as e:
            logger.error(f"Error loading trading plan: {e}")
            self.current_trading_plan = {
                'trading_parameters': {
                    'max_positions': 3,
                    'position_size': 0.1
                }
            }

    def _setup_trading_pairs(self):
        """Set up trading pairs from configuration."""
        # Load pairs from config file
        self.current_pairs = self._load_pairs_from_config()
        
        # Register each pair with the position manager
        for pair in self.current_pairs:
            # Create pair config for position manager
            pair_config = {
                'pair_id': f"{pair['symbol1']}_{pair['symbol2']}",
                'leg1': pair['symbol1'],
                'leg2': pair['symbol2'],
                'hedge_ratio': pair.get('hedge_ratio', 1.0),
                'leg1_multiplier': pair.get('leg1_multiplier', 1.0),
                'leg2_multiplier': pair.get('leg2_multiplier', 1.0),
                'z_entry': pair.get('z_entry', 2.0),
                'z_exit': pair.get('z_exit', 0.5),
                'stop_loss_z': pair.get('stop_loss_z', 3.0),
                'max_holding_period': pair.get('max_holding_period', 180),
                'min_correlation': pair.get('min_correlation', 0.5),
                'half_life': pair.get('half_life', 24)
            }
            
            # Add pair to position manager
            self.position_manager.add_pair(pair_config)
        
        # Log setup completion
        logger.info(f"Set up {len(self.current_pairs)} trading pairs")

    def _run_trading_cycle(self):
        """Run a single trading cycle."""
        try:
            # Check if trading is paused
            if self.is_trading_paused:
                return
            
            # Check trading hours
            if not self._is_trading_hours():
                if not self.outside_hours_notified:
                    logger.info("Outside trading hours - waiting for next trading session")
                    self.outside_hours_notified = True
                return
            
            self.outside_hours_notified = False
            
            # Process market data
            market_data = self._process_market_data()
            
            # Generate and apply signals
            self._generate_and_apply_signals(market_data)
            
            # Monitor positions (new addition)
            self._monitor_positions(market_data)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check for regime changes
            self._check_for_regime_change()
            
            # Update dashboard (if enabled and time to update)
            current_time = datetime.now()
            if (self.enable_dashboard and 
                (self.last_dashboard_update is None or 
                 (current_time - self.last_dashboard_update).total_seconds() >= self.dashboard_update_interval_seconds)):
                self._update_dashboard()
                self.last_dashboard_update = current_time
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    def _monitor_positions(self, market_data):
        """
        Monitor all open positions.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary of market data by pair
        """
        try:
            # Format market data for position manager
            formatted_data = {}
            for pair_id, data in market_data.items():
                formatted_data[pair_id] = data
            
            # Monitor positions
            results = self.position_manager.monitor_all_positions(formatted_data)
            
            # Log monitoring results
            if results['stop_losses_triggered'] > 0:
                logger.info(f"Stop losses triggered: {results['stop_losses_triggered']}")
            
            if results['take_profits_triggered'] > 0:
                logger.info(f"Take profits triggered: {results['take_profits_triggered']}")
            
            if results['holding_limits_triggered'] > 0:
                logger.info(f"Max holding periods exceeded: {results['holding_limits_triggered']}")
            
            if results['correlation_breakdowns'] > 0:
                logger.info(f"Correlation breakdowns detected: {results['correlation_breakdowns']}")
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}")


def create_ml_paper_trader(config_path=None, initial_capital=100000.0):
    """
    Create and configure an ML-enhanced paper trader.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file
    initial_capital : float
        Initial capital for paper trading
        
    Returns:
    --------
    IntradayMLPaperTrader
        Configured ML-enhanced paper trader
    """
    # Load configuration
    config = None
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Create ML paper trader
    trader = IntradayMLPaperTrader(
        initial_capital=initial_capital,
        ml_config=config,
        enable_dashboard=True,
        enable_alerts=True
    )
    
    # Configure pairs
    if config and 'pairs' in config:
        trader.set_pairs(config['pairs'])
    
    return trader 