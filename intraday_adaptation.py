"""
Script to adapt our pairs trading system for intraday trading.

This script enhances our existing system to focus on shorter timeframes,
higher frequency signals, and prop firm risk management requirements.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
import logging

from src.ml_enhancements.regime_detection.market_regime_classifier import MarketRegimeClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def load_portfolio_config(config_file):
    """
    Load portfolio configuration from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the portfolio configuration file
        
    Returns:
    --------
    dict
        Portfolio configuration
    """
    logger.info(f"Loading portfolio configuration from {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration with {len(config['pairs'])} pairs")
        return config
    except Exception as e:
        logger.error(f"Error loading portfolio configuration: {e}")
        return None

def adapt_for_intraday(portfolio_config, prop_firm_rules=None):
    """
    Adapt the portfolio configuration for intraday trading.
    
    Parameters:
    -----------
    portfolio_config : dict
        Original portfolio configuration
    prop_firm_rules : dict, optional
        Rules specific to prop firm requirements
        
    Returns:
    --------
    dict
        Intraday-adapted portfolio configuration
    """
    # Set default prop firm rules if not provided
    if prop_firm_rules is None:
        prop_firm_rules = {
            "max_daily_loss": 0.05,  # 5% max daily loss
            "max_drawdown": 0.10,    # 10% max drawdown
            "required_win_rate": 0.55, # Minimum win rate
            "session_start": "09:30", # Market open (EST)
            "session_end": "16:00",   # Market close (EST)
            "no_overnight": True      # No overnight positions
        }
    
    # Create a new configuration based on the original
    intraday_config = portfolio_config.copy()
    
    # Add intraday-specific settings
    intraday_config["intraday_settings"] = {
        "prop_firm_rules": prop_firm_rules,
        "timeframes": ["1min", "5min", "15min", "30min", "1h"],
        "primary_timeframe": "5min",
        "session_times": {
            "start": prop_firm_rules["session_start"],
            "end": prop_firm_rules["session_end"]
        },
        "no_overnight_positions": prop_firm_rules["no_overnight"],
        "force_close_time": "15:45",  # Close all positions 15 minutes before session end
        "data_resample_method": "volume_weighted"  # Higher quality price sampling
    }
    
    # Adjust risk parameters for intraday trading
    intraday_config["max_allocation_per_pair"] = min(
        0.1,  # Max 10% per pair for intraday
        intraday_config.get("max_allocation_per_pair", 0.2)
    )
    
    # Adjust pairs parameters for intraday trading
    for i, pair in enumerate(intraday_config["pairs"]):
        # Create intraday-specific configuration
        intraday_pair = pair.copy()
        
        # Shorter lookback for faster response
        intraday_pair["config"]["lookback"] = min(20, intraday_pair["config"].get("lookback", 50))
        
        # Tighter stop-loss for intraday risk management
        intraday_pair["config"]["stop_loss_std"] = min(2.5, intraday_pair["config"].get("stop_loss_std", 3.0))
        
        # Add intraday-specific parameters
        intraday_pair["config"]["intraday_params"] = {
            "mean_reversion_exit_time": 120,  # Close if not mean-reverted within 120 minutes
            "max_holding_period": 180,        # Max holding time of 3 hours
            "time_decay_factor": 0.9,         # Reduce position size as day progresses
            "volume_filter": True,            # Only trade during sufficient volume
            "min_volume_percentile": 40,      # Minimum volume percentile to enter
            "exit_buffer_minutes": 15,        # Exit 15 minutes before session close
        }
        
        # Add time-of-day filters
        intraday_pair["config"]["time_filters"] = {
            "avoid_first_15min": True,       # Avoid trading in first 15 minutes
            "avoid_lunch_hour": True,        # Avoid 12:00-13:00 EST
            "avoid_pre_news": True,          # Avoid trading before scheduled news
            "high_liquidity_windows": [      # Focus on high liquidity periods
                {"start": "09:45", "end": "11:30"},
                {"start": "13:30", "end": "15:45"}
            ]
        }
        
        # Add ML signal enhancements
        intraday_pair["config"]["ml_enhancements"] = {
            "use_regime_detection": True,     # Use regime detection
            "use_volume_prediction": True,    # Predict volume patterns
            "use_volatility_forecast": True,  # Forecast volatility
            "use_correlation_monitor": True,  # Monitor correlation changes
            "signal_confirmation": {
                "rsi_filter": True,           # RSI confirmation
                "momentum_filter": True,      # Momentum confirmation
                "volume_imbalance": True      # Volume imbalance confirmation
            }
        }
        
        # Update the pair in the config
        intraday_config["pairs"][i] = intraday_pair
    
    # Add prop firm specific risk management
    intraday_config["risk_management"] = {
        "max_daily_loss_pct": prop_firm_rules["max_daily_loss"],
        "max_drawdown_pct": prop_firm_rules["max_drawdown"],
        "position_scaling": {
            "scale_in": False,             # No scaling in for intraday
            "scale_out": True,             # Scale out allowed
            "scale_out_levels": [0.5, 0.3, 0.2]  # Scale out in thirds
        },
        "correlation_monitoring": {
            "min_correlation": 0.6,        # Minimum correlation to maintain
            "check_interval_minutes": 15   # Check correlation every 15 minutes
        },
        "drawdown_recovery": {
            "reduce_size_after_loss": True,   # Reduce size after losses
            "recovery_factor": 0.7            # Reduce to 70% size after loss
        }
    }
    
    # Add execution optimization for intraday
    intraday_config["execution"] = {
        "order_types": {
            "entry": "limit",              # Use limit orders for entry
            "exit": "market",              # Use market orders for exit
            "stop_loss": "stop_market"     # Use stop market for stop loss
        },
        "smart_execution": {
            "use_twap": True,              # Time-weighted average price
            "use_volume_profile": True,    # Use volume profile for execution
            "limit_price_offset_ticks": 1  # Limit order 1 tick from mid
        },
        "slippage_model": {
            "base_points": 1.0,            # Base slippage in price points
            "volume_factor": 0.5,          # Volume impact on slippage
            "volatility_factor": 0.3       # Volatility impact on slippage
        }
    }
    
    # Add intraday monitoring settings
    intraday_config["monitoring"] = {
        "check_interval_seconds": 60,      # Check every minute
        "metrics_to_track": [
            "open_pnl",
            "realized_pnl",
            "position_size",
            "correlation",
            "spread_zscore",
            "regime_state"
        ],
        "alerting": {
            "correlation_breakdown": True,   # Alert on correlation breakdown
            "regime_change": True,           # Alert on regime change
            "profit_target_hit": True,       # Alert on profit target hit
            "stop_loss_hit": True,           # Alert on stop loss hit
            "drawdown_threshold": True       # Alert on drawdown threshold
        }
    }
    
    return intraday_config

def generate_session_schedule(intraday_config):
    """
    Generate an optimal intraday trading schedule based on config.
    
    Parameters:
    -----------
    intraday_config : dict
        Intraday configuration with session times
        
    Returns:
    --------
    dict
        Dictionary with session timing information
    """
    session_settings = intraday_config["intraday_settings"]
    session_start_str = session_settings["session_times"]["start"]
    session_end_str = session_settings["session_times"]["end"]
    force_close_time_str = session_settings["force_close_time"]
    
    # Convert to datetime.time objects
    session_start = datetime.strptime(session_start_str, "%H:%M").time()
    session_end = datetime.strptime(session_end_str, "%H:%M").time()
    force_close_time = datetime.strptime(force_close_time_str, "%H:%M").time()
    
    # High liquidity windows from first pair (assuming all pairs use same windows)
    high_liquidity_windows = intraday_config["pairs"][0]["config"]["time_filters"]["high_liquidity_windows"]
    
    # Calculate ideal trading windows
    trading_windows = []
    
    for window in high_liquidity_windows:
        window_start = datetime.strptime(window["start"], "%H:%M").time()
        window_end = datetime.strptime(window["end"], "%H:%M").time()
        
        # Only include if within session
        if window_start >= session_start and window_end <= session_end:
            trading_windows.append({
                "start": window_start,
                "end": window_end
            })
    
    # Generate actual schedule
    schedule = {
        "session_start": session_start,
        "session_end": session_end,
        "force_close_time": force_close_time,
        "trading_windows": trading_windows,
        "avoid_times": []
    }
    
    # Add lunch hour if configured to avoid it
    if intraday_config["pairs"][0]["config"]["time_filters"]["avoid_lunch_hour"]:
        schedule["avoid_times"].append({
            "start": time(12, 0),
            "end": time(13, 0),
            "reason": "Lunch hour"
        })
    
    # Add opening 15 minutes if configured to avoid it
    if intraday_config["pairs"][0]["config"]["time_filters"]["avoid_first_15min"]:
        schedule["avoid_times"].append({
            "start": session_start,
            "end": (datetime.combine(datetime.today(), session_start) + timedelta(minutes=15)).time(),
            "reason": "Market open volatility"
        })
    
    return schedule

def create_intraday_backtest_config(intraday_config):
    """
    Create a configuration specifically for backtesting intraday strategies.
    
    Parameters:
    -----------
    intraday_config : dict
        Intraday trading configuration
        
    Returns:
    --------
    dict
        Backtest configuration for intraday
    """
    backtest_config = {
        "data_settings": {
            "timeframe": intraday_config["intraday_settings"]["primary_timeframe"],
            "start_date": "2023-01-01",   # Use recent data for intraday backtest
            "end_date": "2023-12-31",     # Use recent data for intraday backtest
            "include_after_hours": False,  # Exclude after-hours trading
            "use_volume_profile": True     # Include volume profile for realistic execution
        },
        "backtest_parameters": {
            "account_size": 100000,        # Standard prop firm account size
            "commission_model": {
                "per_contract": 0.85,      # Standard commission per contract
                "per_share": 0.005,        # Standard commission per share
                "minimum": 1.0             # Minimum commission
            },
            "slippage_model": {
                "fixed_pips": 1,           # 1 pip fixed slippage
                "percentage": 0.0001,      # 0.01% percentage slippage
                "volume_based": True       # Volume impacts slippage
            },
            "session_times": {
                "start": intraday_config["intraday_settings"]["session_times"]["start"],
                "end": intraday_config["intraday_settings"]["session_times"]["end"]
            }
        },
        "validation": {
            "walk_forward": True,            # Use walk-forward optimization
            "walk_forward_windows": 10,      # Number of walk-forward windows
            "monte_carlo_simulations": 1000, # Number of Monte Carlo simulations
            "robustness_tests": [
                "slippage_impact",          # Test with higher slippage
                "commission_impact",        # Test with higher commissions
                "execution_delay",          # Test with execution delays
                "correlation_breakdown"     # Test with correlation breakdowns
            ]
        }
    }
    
    return backtest_config

def adapt_regime_detection_for_intraday(intraday_config):
    """
    Adapt regime detection parameters for intraday trading.
    
    Parameters:
    -----------
    intraday_config : dict
        Intraday configuration to adapt
        
    Returns:
    --------
    dict
        Adapted regime detection configuration
    """
    regime_config = {
        "timeframes": [
            "5min",   # Ultra-short-term regime
            "30min",  # Short-term regime
            "1h",     # Medium-term regime
        ],
        "detection_features": [
            "volatility",            # Volatility is crucial for intraday
            "trend_strength",        # Trend strength
            "correlation",           # Correlation stability
            "mean_reversion",        # Mean reversion potential
            "spread_volatility",     # Spread volatility
            "volume_profile"         # Volume profile
        ],
        "regime_responses": {
            "high_volatility": {
                "entry_zscore": 2.5,         # More conservative entry
                "exit_zscore": 0.3,          # Faster exit
                "stop_loss_std": 2.0,        # Tighter stop loss
                "position_size_factor": 0.7  # Smaller positions
            },
            "trending": {
                "entry_zscore": 2.2,         # Standard entry
                "exit_zscore": 0.5,          # Standard exit
                "stop_loss_std": 2.5,        # Standard stop loss
                "position_size_factor": 0.8  # Slightly reduced size
            },
            "mean_reverting": {
                "entry_zscore": 2.0,         # More aggressive entry
                "exit_zscore": 0.7,          # Longer hold for reversion
                "stop_loss_std": 2.8,        # Slightly wider stop
                "position_size_factor": 1.0  # Full position size
            },
            "low_correlation": {
                "entry_zscore": 2.8,         # Very conservative entry
                "exit_zscore": 0.2,          # Very quick exit
                "stop_loss_std": 1.8,        # Tight stop loss
                "position_size_factor": 0.5  # Half position size
            }
        },
        "update_frequency": {
            "regime_check_minutes": 15,      # Check regime every 15 minutes
            "parameter_update_minutes": 30,  # Update parameters every 30 minutes
            "correlation_check_minutes": 10  # Check correlation every 10 minutes
        }
    }
    
    # Add to intraday config
    intraday_config["regime_detection"] = regime_config
    
    return intraday_config

def save_intraday_config(intraday_config, output_dir="data/configs"):
    """
    Save the intraday configuration to a file.
    
    Parameters:
    -----------
    intraday_config : dict
        Intraday configuration to save
    output_dir : str
        Directory to save the configuration
        
    Returns:
    --------
    str
        Path to the saved configuration file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"intraday_config_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(intraday_config, f, indent=2, default=str)
    
    logger.info(f"Saved intraday configuration to {output_file}")
    
    return output_file

def main():
    """Main function to adapt portfolio for intraday trading."""
    # Find most recent portfolio file
    portfolio_dir = "data/results/portfolio"
    
    # Try with regime-adapted portfolios first
    portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.json') and f.startswith('regime_adapted_portfolio_')]
    
    # If no regime-adapted portfolios, use simple portfolios
    if not portfolio_files:
        portfolio_files = [f for f in os.listdir(portfolio_dir) if f.endswith('.json') and f.startswith('simple_portfolio_')]
    
    if not portfolio_files:
        logger.error("No portfolio configuration files found")
        return
    
    # Sort by timestamp in filename (most recent first)
    portfolio_files.sort(reverse=True)
    latest_portfolio = os.path.join(portfolio_dir, portfolio_files[0])
    
    logger.info(f"Using latest portfolio: {latest_portfolio}")
    
    # Load portfolio configuration
    portfolio_config = load_portfolio_config(latest_portfolio)
    
    if portfolio_config is None:
        return
    
    # Define prop firm rules
    prop_firm_rules = {
        "max_daily_loss": 0.05,     # 5% max daily loss
        "max_drawdown": 0.10,       # 10% max drawdown
        "required_win_rate": 0.55,  # Minimum win rate
        "session_start": "09:30",   # Market open (EST)
        "session_end": "16:00",     # Market close (EST)
        "no_overnight": True        # No overnight positions
    }
    
    # Adapt portfolio for intraday trading
    intraday_config = adapt_for_intraday(portfolio_config, prop_firm_rules)
    
    # Generate optimal trading schedule
    schedule = generate_session_schedule(intraday_config)
    intraday_config["trading_schedule"] = schedule
    
    # Create backtest configuration
    backtest_config = create_intraday_backtest_config(intraday_config)
    intraday_config["backtest_config"] = backtest_config
    
    # Adapt regime detection for intraday
    intraday_config = adapt_regime_detection_for_intraday(intraday_config)
    
    # Save intraday configuration
    config_file = save_intraday_config(intraday_config)
    
    # Print summary
    print("\nIntraday Trading Configuration Created")
    print(f"Base portfolio: {os.path.basename(latest_portfolio)}")
    print(f"Configuration file: {os.path.basename(config_file)}")
    print("\nKey Adaptations:")
    print("  • Shorter lookback periods for faster signal response")
    print("  • Time-of-day trading filters for optimal market conditions")
    print("  • Tighter risk management for prop firm compliance")
    print("  • ML signal enhancements for intraday edge")
    print("  • No overnight position holding")
    print("  • Regime-specific parameter adaptations")
    
    print("\nTrading Schedule:")
    for window in schedule["trading_windows"]:
        print(f"  • {window['start'].strftime('%H:%M')} - {window['end'].strftime('%H:%M')}")
    
    print("\nAvoid Times:")
    for avoid in schedule["avoid_times"]:
        print(f"  • {avoid['start'].strftime('%H:%M')} - {avoid['end'].strftime('%H:%M')}: {avoid['reason']}")
    
    print(f"\nForce close all positions by: {schedule['force_close_time'].strftime('%H:%M')}")
    
    print("\nIntraday Regime Detection:")
    for regime, params in intraday_config["regime_detection"]["regime_responses"].items():
        print(f"  • {regime.replace('_', ' ').title()}: Entry z-score {params['entry_zscore']}, Exit z-score {params['exit_zscore']}")

if __name__ == "__main__":
    main() 