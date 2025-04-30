# Live Trading API Documentation

The Live Trading module provides functionality for executing trading strategies with real money through the Interactive Brokers (IB) API. This module enables actual order execution, position tracking, risk management, and monitoring of account performance.

## Overview

Live trading is the final step in deploying a trading strategy, following successful backtesting and paper trading. The `LiveTrader` class connects to Interactive Brokers' TWS (Trader Workstation) or IB Gateway to:

1. Retrieve real-time market data
2. Execute actual orders with real money
3. Monitor positions and account status
4. Implement risk management features
5. Track trading performance

**⚠️ WARNING: Live trading involves real money and financial risk. Always thoroughly test strategies in paper trading before deploying to live trading.**

## Importing the Module

```python
from src.live_trading import LiveTrader
```

## LiveTrader Class

### Initialization

```python
trader = LiveTrader(
    ib_host='127.0.0.1',            # IB TWS/Gateway hostname or IP
    ib_port=7496,                   # 7496 for live, 7497 for paper
    ib_client_id=1,                 # Client ID
    account=None,                   # Use first account if None
    data_directory=None,            # Directory to store trading data
    use_emergency_stop=True,        # Enable emergency stop on excessive loss
    max_daily_loss_pct=1.0,         # Maximum daily loss percentage
    position_check_interval=10,     # Interval to check positions (seconds)
    confirmation_required=True,     # Require confirmation before execution
    risk_level='low',               # Risk level ('low', 'medium', 'high')
    heartbeat_interval=30,          # Heartbeat interval (seconds)
    auto_shutdown_time=None,        # Auto-shutdown time (format: "HH:MM")
    debug_mode=False                # Enable debug mode
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ib_host` | str | The hostname or IP address of the IB TWS/Gateway instance. Default is '127.0.0.1'. |
| `ib_port` | int | The port used by IB TWS/Gateway. Port 7496 is for live trading, 7497 for paper trading. |
| `ib_client_id` | int | A unique client ID for the connection. |
| `account` | str | The IB account ID to use. If None, the first available account will be used. |
| `data_directory` | str | Directory to store trading data (positions, orders, trades). If None, a default directory will be created. |
| `use_emergency_stop` | bool | Whether to use emergency stop on excessive loss. |
| `max_daily_loss_pct` | float | Maximum allowed daily loss as percentage of account equity. |
| `position_check_interval` | int | Interval in seconds to check positions. |
| `confirmation_required` | bool | Whether to require confirmation before executing orders. |
| `risk_level` | str | Risk level affecting position sizing ('low', 'medium', 'high'). |
| `heartbeat_interval` | int | Interval in seconds to send heartbeat signals. |
| `auto_shutdown_time` | str | Time to automatically shutdown trading (format: "HH:MM"). |
| `debug_mode` | bool | Whether to run in debug mode with additional logging. |

## Core Methods

### Connection Management

```python
# Start live trading
success = trader.start()

# Check if running
is_running = trader.is_running()

# Pause trading (continues receiving market data but stops new orders)
trader.pause_trading()

# Resume trading after pause
trader.resume_trading()

# Check if trading is paused
is_paused = trader.is_paused()

# Stop trading and close connection
trader.stop()
```

### Order Execution

```python
# Place a market order
order_id = trader.place_order(
    symbol='ES',         # The symbol to trade
    quantity=1,          # Positive for buy, negative for sell
    order_type='MKT',    # Order type ('MKT', 'LMT', 'STP', 'STP LMT')
    time_in_force='DAY'  # Time in force ('DAY', 'GTC', 'IOC')
)

# Place a limit order
order_id = trader.place_order(
    symbol='ES',
    quantity=1,
    order_type='LMT',
    limit_price=4000.50,
    time_in_force='DAY'
)

# Place a stop order
order_id = trader.place_order(
    symbol='ES',
    quantity=-1,          # Negative for sell
    order_type='STP',
    stop_price=3950.00,
    time_in_force='GTC'   # Good Till Cancelled
)
```

### Market Data

```python
# Subscribe to market data for a symbol
trader.subscribe_market_data('ES')

# Get the latest market data for a symbol
market_data = trader.get_market_data('ES')

# Unsubscribe from market data
trader.unsubscribe_market_data('ES')
```

### Position and Account Information

```python
# Get all current positions
positions = trader.get_positions()

# Get all pending orders
pending_orders = trader.get_pending_orders()

# Get all executed orders
executed_orders = trader.get_executed_orders()

# Get current risk metrics
risk_metrics = trader.get_risk_metrics()

# Get account values
account_values = trader.get_account_values()
```

## Event Callbacks

The LiveTrader uses an event-based architecture to notify about system events. You can register callback functions for different event types:

```python
# Register a callback for position changes
def on_position_change(data):
    print(f"Position change: {data['symbol']} from {data['old_position']} to {data['new_position']}")

trader.register_callback('position_change', on_position_change)
```

### Available Event Types

| Event Type | Description | Data Structure |
|------------|-------------|----------------|
| `order_status` | Order status updates | `{'order_id': str, 'status': str, 'filled': int, 'remaining': int, 'avg_fill_price': float}` |
| `position_change` | Position changes | `{'symbol': str, 'old_position': float, 'new_position': float, 'timestamp': str}` |
| `account_update` | Account value updates | Dictionary with account values (e.g., 'NetLiquidation_USD', 'AvailableFunds_USD') |
| `market_data` | Market data updates | `{'symbol': str, 'data': dict}` |
| `trade` | Completed trades | `{'symbol': str, 'quantity': float, 'price': float, 'timestamp': str}` |
| `error` | Error events | `{'code': int, 'message': str, 'source': str}` |
| `heartbeat` | Periodic system status | `{'timestamp': str, 'is_running': bool, 'is_paused': bool, 'emergency_stop': bool, 'connected': bool}` |
| `emergency_stop` | Emergency stop triggered | `{'timestamp': str, 'reason': str, 'daily_loss': float, 'max_daily_loss': float}` |

## Risk Management Features

### Emergency Stop

The LiveTrader includes an emergency stop feature that automatically halts trading if the daily loss exceeds a specified threshold. This is a critical safety feature to prevent excessive losses:

```python
# Configure emergency stop with a 1% maximum daily loss
trader = LiveTrader(
    # ...other parameters...
    use_emergency_stop=True,
    max_daily_loss_pct=1.0
)
```

### Risk Levels

The risk level parameter affects position sizing:

- **Low**: 2% max position size
- **Medium**: 5% max position size
- **High**: 10% max position size

```python
# Configure low risk level
trader = LiveTrader(
    # ...other parameters...
    risk_level='low'
)
```

### Auto-Shutdown

The auto-shutdown feature allows the system to automatically stop trading at a specified time:

```python
# Configure auto-shutdown at 4:00 PM
trader = LiveTrader(
    # ...other parameters...
    auto_shutdown_time="16:00"
)
```

## Complete Example

Here's a complete example of how to use the LiveTrader:

```python
import logging
from src.live_trading import LiveTrader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Callback for position changes
def on_position_change(data):
    print(f"Position change: {data['symbol']} from {data['old_position']} to {data['new_position']}")

# Create and configure LiveTrader
trader = LiveTrader(
    ib_host='127.0.0.1',
    ib_port=7496,
    ib_client_id=1,
    use_emergency_stop=True,
    max_daily_loss_pct=1.0,
    risk_level='low'
)

# Register callbacks
trader.register_callback('position_change', on_position_change)

# Start the trader
if trader.start():
    try:
        # Subscribe to market data
        trader.subscribe_market_data('ES')
        
        # Place a market order to buy 1 contract
        order_id = trader.place_order(
            symbol='ES',
            quantity=1,
            order_type='MKT'
        )
        
        if order_id:
            print(f"Order placed: {order_id}")
        
        # Main trading loop
        while trader.is_running():
            # Trading logic goes here
            pass
            
    finally:
        # Stop the trader
        trader.stop()
else:
    print("Failed to start trader")
```

## Best Practices

1. **Test thoroughly in paper trading** before going live
2. **Start with small position sizes** when transitioning to live trading
3. **Always use the emergency stop** feature to limit daily losses
4. **Monitor the system closely** during initial live trading
5. **Have a clear exit strategy** for all positions
6. **Create detailed logs** for post-trade analysis
7. **Use limit orders** when possible to control execution costs
8. **Implement alerts** for important events
9. **Have backup connectivity** to Interactive Brokers
10. **Regularly verify account status** and position consistency

## Troubleshooting

### Common IB Connection Issues

- Ensure TWS or IB Gateway is running
- Check that the correct port is being used (7496 for live, 7497 for paper)
- Verify that API connections are enabled in TWS/Gateway settings
- Make sure the client ID is not being used by another application
- Confirm your account has the necessary permissions for trading

### Handling Disconnections

The LiveTrader automatically attempts to reconnect if the connection to IB is lost. You can monitor connection status through the heartbeat callback.

### Error Handling

All errors are logged and also sent through the 'error' callback. Make sure to register a callback for error events to be informed about issues:

```python
def on_error(data):
    print(f"Error {data['code']}: {data['message']}")

trader.register_callback('error', on_error)
```

## Limitations and Considerations

- The LiveTrader is designed for use with Interactive Brokers only
- Order execution is subject to market conditions and may differ from expectations
- Risk management features are safety nets but do not guarantee against losses
- Position and order tracking relies on proper connectivity to IB
- Auto-shutdown only works while the program is running

By following these guidelines and using the LiveTrader responsibly, you can implement systematic trading strategies with appropriate risk controls. 