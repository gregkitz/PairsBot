# Interactive Brokers Connector

The Interactive Brokers (IB) connector provides a high-level interface for connecting to Interactive Brokers TWS or Gateway for market data retrieval, order execution, and account management.

## Components

### IBConnector

The `IBConnector` class is the main entry point for interacting with Interactive Brokers.

```python
from src.connectors.ib import IBConnector
```

#### Initialization

```python
connector = IBConnector(
    host='127.0.0.1',          # IB TWS/Gateway hostname or IP
    port=7497,                 # Port (7497 for TWS Paper, 7496 for TWS Live)
    client_id=1,               # Client ID
    account=None,              # Account ID (None to use first available)
    timeout=30,                # Connection timeout in seconds
    read_only=False,           # Read-only mode (no order execution)
    max_retry_count=3,         # Max retry attempts for connection
    retry_wait_time=5,         # Seconds to wait between retries
    auto_reconnect=True,       # Auto reconnect on disconnection
    use_async=False            # Use async/await pattern
)
```

#### Connection Management

```python
# Connect to IB
connected = connector.connect()

# Check if connected
is_connected = connector.is_connected()

# Disconnect from IB
connector.disconnect()
```

#### Market Data

```python
# Get real-time market data
ticker = connector.get_market_data('AAPL-STK-SMART', subscribe=True)
# ticker contains bid, ask, last, volume, etc.

# Cancel market data subscription
connector.cancel_market_data('AAPL-STK-SMART')

# Get historical data
historical_data = connector.get_historical_data(
    symbol='AAPL-STK-SMART',
    start=datetime(2023, 1, 1),  # Optional
    end=datetime(2023, 1, 31),   # Optional
    duration='1 D',              # Duration if start not specified
    bar_size='1 min',            # Bar size (e.g., '1 min', '1 hour', '1 day')
    what_to_show='TRADES',       # Type of data ('TRADES', 'MIDPOINT', 'BID', 'ASK')
    use_rth=True                 # Use regular trading hours only
)
# Returns pandas DataFrame with OHLCV data
```

#### Contract Management

```python
# Get contract details
details = connector.get_contract_details('AAPL-STK-SMART')
# Returns dictionary with contract details
```

#### Account Management

```python
# Get account information
account_info = connector.get_account_info()
# Returns dictionary with account values

# Get positions
positions = connector.get_positions()
# Returns dictionary of positions by symbol

# Get specific position
aapl_position = connector.get_position('AAPL-STK-SMART')
# Returns dictionary with position information
```

#### Order Management

```python
# Place a market order
trade = connector.place_order(
    symbol='AAPL-STK-SMART',
    action='BUY',                 # 'BUY' or 'SELL'
    quantity=100,
    order_type='MKT',             # Market order
    time_in_force='GTC'           # Good Till Cancelled
)

# Place a limit order
trade = connector.place_order(
    symbol='AAPL-STK-SMART',
    action='BUY',
    quantity=100,
    order_type='LMT',            # Limit order
    limit_price=150.00,
    time_in_force='DAY'
)

# Place a stop order
trade = connector.place_order(
    symbol='AAPL-STK-SMART',
    action='SELL',
    quantity=100,
    order_type='STP',            # Stop order
    stop_price=145.00
)

# Place a bracket order (entry with profit target and stop loss)
trades = connector.place_bracket_order(
    symbol='AAPL-STK-SMART',
    action='BUY',
    quantity=100,
    entry_order_type='MKT',
    profit_price=155.00,         # Profit target price
    stop_price=145.00            # Stop loss price
)
# Returns list of trades [entry, profit, stop]

# Cancel an order
cancelled = connector.cancel_order(order_id=12345)

# Cancel all orders
connector.cancel_all_orders()

# Get order status
status = connector.get_order_status(order_id=12345)
# Returns dictionary with order status

# Get all active orders
active_orders = connector.get_active_orders()
# Returns dictionary of active orders by order ID
```

#### Event Processing

```python
# Process events (non-blocking)
connector.process_pending_events()

# Run event loop for specified time
connector.run_event_loop(timeout=0.1)  # 100ms
```

#### Event Callbacks

```python
# Add callback for order status updates
def on_order_status(trade):
    print(f"Order {trade.order.orderId} status: {trade.orderStatus.status}")

connector.add_callback('order_status', on_order_status)

# Add callback for position updates
def on_position_change(symbol, position, avg_cost):
    print(f"Position: {symbol}, Quantity: {position}, Avg Cost: {avg_cost}")

connector.add_callback('position_change', on_position_change)

# Add callback for account updates
def on_account_update(tag, value, currency):
    print(f"Account {tag}: {value} {currency}")

connector.add_callback('account_update', on_account_update)

# Add callback for market data updates
def on_market_data(symbol, ticker):
    print(f"{symbol} price: {ticker.last if hasattr(ticker, 'last') and ticker.last else 'N/A'}")

connector.add_callback('market_data', on_market_data)

# Add callback for errors
def on_error(req_id, error_code, error_string, contract):
    print(f"Error {error_code}: {error_string}")

connector.add_callback('error', on_error)

# Remove a callback
connector.remove_callback('order_status', on_order_status)
```

### Utility Functions

The module also provides utility functions for working with IB contracts:

```python
from src.connectors.ib import contract_to_symbol, symbol_to_contract

# Convert IB contract to symbol string
symbol = contract_to_symbol(contract)
# Example: 'AAPL-STK-SMART'

# Convert symbol string to IB contract
contract = symbol_to_contract('AAPL-STK-SMART')
# Returns an IB contract object
```

## Symbol Format

The system uses a standardized symbol format to identify instruments:

- **Stocks**: `{ticker}-STK-{exchange}`, e.g., `AAPL-STK-SMART`
- **Futures**: `{symbol}-{expiry}-FUT-{exchange}`, e.g., `ES-202306-FUT-GLOBEX`
- **Forex**: `{base}{quote}-CASH` or `{base}-{quote}-CASH`, e.g., `EURUSD-CASH` or `EUR-USD-CASH`
- **Indices**: `{symbol}-IND-{exchange}`, e.g., `SPX-IND-CBOE`

The `exchange` part is optional for stocks, defaulting to `SMART`. The system will also attempt to infer the asset type if a simplified format is used.

## Examples

### Basic Market Data Example

```python
from src.connectors.ib import IBConnector
import time

# Create connector
connector = IBConnector(
    host='127.0.0.1',
    port=7497,  # TWS Paper Trading
    client_id=1
)

# Connect to IB
if connector.connect():
    try:
        # Get real-time market data
        connector.get_market_data('AAPL-STK-SMART')
        connector.get_market_data('MSFT-STK-SMART')
        
        # Process events for 10 seconds
        for _ in range(10):
            # Get current prices
            aapl_ticker = connector.get_market_data('AAPL-STK-SMART', subscribe=False)
            msft_ticker = connector.get_market_data('MSFT-STK-SMART', subscribe=False)
            
            print(f"AAPL: {aapl_ticker.last if hasattr(aapl_ticker, 'last') and aapl_ticker.last else 'N/A'}, "
                  f"MSFT: {msft_ticker.last if hasattr(msft_ticker, 'last') and msft_ticker.last else 'N/A'}")
            
            # Process events
            connector.run_event_loop(1.0)  # 1 second
    
    finally:
        # Clean up
        connector.disconnect()
```

### Using Callbacks

```python
from src.connectors.ib import IBConnector
import time

# Create connector
connector = IBConnector(
    host='127.0.0.1',
    port=7497,
    client_id=1
)

# Define callbacks
def on_market_data(symbol, ticker):
    print(f"{symbol} price: {ticker.last if hasattr(ticker, 'last') and ticker.last else 'N/A'}")

def on_account_update(tag, value, currency):
    if tag in ['NetLiquidation', 'AvailableFunds']:
        print(f"Account {tag}: {value} {currency}")

def on_error(req_id, error_code, error_string, contract):
    print(f"Error {error_code}: {error_string}")

# Register callbacks
connector.add_callback('market_data', on_market_data)
connector.add_callback('account_update', on_account_update)
connector.add_callback('error', on_error)

# Connect to IB
if connector.connect():
    try:
        # Subscribe to market data
        connector.get_market_data('AAPL-STK-SMART')
        connector.get_market_data('MSFT-STK-SMART')
        
        # Run event loop for 60 seconds
        for _ in range(60):
            connector.run_event_loop(1.0)
    
    finally:
        # Clean up
        connector.disconnect()
```

### Portfolio Information

```python
from src.connectors.ib import IBConnector
import pandas as pd

# Create connector
connector = IBConnector(
    host='127.0.0.1',
    port=7497,
    client_id=1
)

# Connect to IB
if connector.connect():
    try:
        # Get account information
        account_info = connector.get_account_info()
        
        # Print account summary
        print("Account Summary:")
        print(f"Net Liquidation: ${account_info.get('NetLiquidation_USD', 'N/A')}")
        print(f"Available Funds: ${account_info.get('AvailableFunds_USD', 'N/A')}")
        print(f"Margin Used: ${account_info.get('MaintMarginReq_USD', 'N/A')}")
        
        # Get positions
        positions = connector.get_positions()
        
        # Convert to DataFrame for display
        if positions:
            positions_df = pd.DataFrame.from_dict(
                {symbol: {
                    'Quantity': pos['position'],
                    'Avg Cost': pos['avg_cost'],
                    'Market Price': connector.get_price(symbol) or 0,
                    'Market Value': pos['position'] * (connector.get_price(symbol) or 0),
                    'Unrealized P&L': pos['position'] * (connector.get_price(symbol) or 0) - pos['position'] * pos['avg_cost']
                } for symbol, pos in positions.items()
            }).T
            
            print("\nPositions:")
            print(positions_df)
        else:
            print("\nNo positions found.")
    
    finally:
        connector.disconnect()
```

### Historical Data Analysis

```python
from src.connectors.ib import IBConnector
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create connector
connector = IBConnector(
    host='127.0.0.1',
    port=7497,
    client_id=1
)

# Connect to IB
if connector.connect():
    try:
        # Get historical data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = connector.get_historical_data(
            symbol='SPY-STK-SMART',
            start=start_date,
            end=end_date,
            bar_size='1 day',
            what_to_show='TRADES',
            use_rth=True
        )
        
        # Calculate simple moving averages
        data['SMA20'] = data['close'].rolling(window=20).mean()
        data['SMA50'] = data['close'].rolling(window=50).mean()
        
        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['close'], label='Close')
        plt.plot(data.index, data['SMA20'], label='20-day SMA')
        plt.plot(data.index, data['SMA50'], label='50-day SMA')
        plt.title('SPY Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('spy_analysis.png')
        plt.close()
        
        print(f"Analysis saved to spy_analysis.png")
        
    finally:
        connector.disconnect()
```

### Placing Orders

```python
from src.connectors.ib import IBConnector
import time

# Create connector
connector = IBConnector(
    host='127.0.0.1',
    port=7497,
    client_id=1
)

# Connect to IB
if connector.connect():
    try:
        # Get current price
        ticker = connector.get_market_data('AAPL-STK-SMART')
        current_price = ticker.last if hasattr(ticker, 'last') and ticker.last else None
        
        if current_price:
            print(f"Current AAPL price: ${current_price:.2f}")
            
            # Place a limit buy order below current price
            limit_price = current_price * 0.98  # 2% below current price
            
            order_trade = connector.place_order(
                symbol='AAPL-STK-SMART',
                action='BUY',
                quantity=10,
                order_type='LMT',
                limit_price=limit_price,
                time_in_force='DAY'
            )
            
            if order_trade and order_trade.order and order_trade.order.orderId:
                order_id = order_trade.order.orderId
                print(f"Limit buy order placed: {order_id}")
                
                # Wait for a while
                for _ in range(10):
                    # Check order status
                    status = connector.get_order_status(order_id)
                    print(f"Order status: {status.get('status', 'Unknown')}")
                    
                    # If filled, break
                    if status.get('status') == 'Filled':
                        print(f"Order filled at ${status.get('avg_fill_price', 0):.2f}")
                        break
                        
                    # Process events for 1 second
                    connector.run_event_loop(1.0)
                
                # Cancel order if not filled
                if connector.get_order_status(order_id).get('status') != 'Filled':
                    print("Cancelling order...")
                    connector.cancel_order(order_id)
            else:
                print("Failed to place order")
        else:
            print("Could not get current price")
    
    finally:
        connector.disconnect()
```

## Best Practices

### Connection Management

1. **Automatic Reconnection**:
   - Use `auto_reconnect=True` for production systems to handle connection loss
   - Implement additional error handling for critical operations

2. **Connection Timeout**:
   - Set an appropriate `timeout` value based on network conditions
   - Use longer timeouts for initial connection and shorter timeouts for operations

### Market Data

1. **Subscribe Only When Needed**:
   - Use `subscribe=True` only for symbols you need to monitor continuously
   - For one-time requests, use `subscribe=False` to avoid resource consumption

2. **Historical Data Requests**:
   - Be mindful of IB's data limitations and request throttling
   - Use the appropriate `bar_size` for your analysis to minimize data volume

### Order Execution

1. **Read-Only Mode for Testing**:
   - Use `read_only=True` during development and testing to prevent accidental orders
   - Only disable read-only mode when ready for actual trading

2. **Order Verification**:
   - Always verify order parameters before submitting
   - Use `get_order_status` to confirm orders are processed correctly

3. **Order Batching**:
   - For multiple related orders, use bracket orders rather than individual orders
   - This ensures proper parent-child relationships for stop loss and profit targets

### Event Handling

1. **Event Loop Management**:
   - Call `process_pending_events` or `run_event_loop` regularly to handle IB events
   - For GUI applications, integrate this with your main event loop

2. **Callback Design**:
   - Keep callbacks lightweight and non-blocking
   - For complex processing, use a queue to defer work outside the callback

### Error Handling

1. **Error Callback**:
   - Always register an error callback to catch and log IB API errors
   - Some "errors" are informational and can be filtered by error code

2. **Reconnection Logic**:
   - Implement backoff logic for reconnection attempts
   - Consider the nature of the error before automatic reconnection

## Troubleshooting

### Common Issues

1. **Connection Refused**:
   - Ensure TWS/Gateway is running and configured to accept API connections
   - Check `Edit > Global Configuration > API > Settings` in TWS
   - Verify the port number matches (typically 7496 for live, 7497 for paper)

2. **Authentication Failed**:
   - Ensure the `trusted IPs` list in TWS/Gateway includes your client IP

3. **Data Permissions**:
   - Some market data requires additional subscriptions
   - Check TWS messages for permission warnings

4. **Order Placement Failures**:
   - Verify sufficient buying power/margin
   - Check for trading restrictions on the instrument
   - Look for order precondition failures in the TWS log

### Debugging Tips

1. **Enable TWS/Gateway Logging**:
   - Configure TWS to log API messages: `Global Configuration > API > Settings > Create API message log file`

2. **TWS Setting Recommendations**:
   - Disable auto-restart of TWS
   - Enable API precautions bypass for production systems
   - Configure appropriate socket port and timeout values

3. **API Version Compatibility**:
   - Ensure TWS/Gateway and the ib_insync library versions are compatible
   - The minimum supported TWS version is typically mentioned in ib_insync documentation 