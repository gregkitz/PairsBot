# Paper Trading Module

The Paper Trading module provides a simulation environment for testing trading strategies with real-time market data but simulated executions. This allows for validating strategies in realistic market conditions without risking real capital.

## Components

### PaperTrader

The `PaperTrader` class is the main entry point for paper trading. It connects to Interactive Brokers for market data while simulating order execution and position management.

```python
from src.paper_trading import PaperTrader
```

#### Initialization

```python
paper_trader = PaperTrader(
    initial_capital=100000.0,  # Starting capital
    ib_host='127.0.0.1',       # IB TWS/Gateway host
    ib_port=7497,              # Port (7497 for TWS Paper, 7496 for TWS Live)
    ib_client_id=1,            # Client ID
    data_directory=None,       # Directory for storing data (default: './paper_trading_data')
    commission_model='ibkr_pro',  # Commission model ('ibkr_pro', 'ibkr_lite', 'flat', 'none')
    slippage_model='fixed',    # Slippage model ('fixed', 'variable', 'none')
    slippage_factor=0.0001,    # Slippage factor (percentage for fixed)
    latency_model='fixed',     # Latency model ('fixed', 'variable', 'none')
    latency_ms=100,            # Latency in milliseconds
    auto_shutdown_time=None    # Auto-shutdown time (format: "HH:MM")
)
```

#### Connection Management

```python
# Start the paper trading environment
paper_trader.start()

# Check if running
is_running = paper_trader.is_running()

# Stop the paper trading environment
paper_trader.stop()
```

#### Market Data

```python
# Subscribe to market data
paper_trader.subscribe_market_data('AAPL-STK-SMART')

# Get market data
market_data = paper_trader.get_market_data('AAPL-STK-SMART')

# Get current quote
quote = paper_trader.get_quote('AAPL-STK-SMART')

# Get current price
price = paper_trader.get_price('AAPL-STK-SMART')

# Unsubscribe from market data
paper_trader.unsubscribe_market_data('AAPL-STK-SMART')
```

#### Order Management

```python
# Place a market order
order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='BUY',           # 'BUY' or 'SELL'
    quantity=100,
    order_type='MKT'        # Market order
)

# Place a limit order
order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='BUY',
    quantity=100,
    order_type='LMT',       # Limit order
    limit_price=150.00
)

# Place a stop order
order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='SELL',
    quantity=100,
    order_type='STP',       # Stop order
    stop_price=145.00
)

# Place a stop-limit order
order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='SELL',
    quantity=100,
    order_type='STP LMT',   # Stop-limit order
    stop_price=145.00,
    limit_price=144.50
)

# Place a bracket order (entry with profit target and stop loss)
bracket_orders = paper_trader.place_bracket_order(
    symbol='AAPL-STK-SMART',
    action='BUY',
    quantity=100,
    entry_order_type='MKT',
    profit_price=155.00,    # Profit target
    stop_price=145.00       # Stop loss
)
# Returns dict with 'entry', 'profit', and 'stop' order IDs

# Get order details
order = paper_trader.get_order(order_id)

# Get all orders (optionally filtered)
all_orders = paper_trader.get_orders()
open_orders = paper_trader.get_orders(status='PENDING')
aapl_orders = paper_trader.get_orders(symbol='AAPL-STK-SMART')

# Cancel an order
paper_trader.cancel_order(order_id)
```

#### Account and Position Management

```python
# Get account information
account_info = paper_trader.get_account_info()
# {
#   'cash': 95000.0,
#   'equity': 100000.0,
#   'margin_used': 5000.0,
#   'pnl_day': 500.0,
#   'unrealized_pnl': 500.0,
#   ...
# }

# Get positions
positions = paper_trader.get_positions()
# {
#   'AAPL-STK-SMART': {
#     'quantity': 100,
#     'avg_cost': 150.0,
#     'market_price': 152.0,
#     'market_value': 15200.0,
#     'unrealized_pnl': 200.0,
#     ...
#   },
#   ...
# }

# Get specific position
aapl_position = paper_trader.get_position('AAPL-STK-SMART')
```

#### Event Callbacks

```python
# Add callback for account updates
def on_account_update(account_data):
    print(f"Account equity: ${account_data['equity']:.2f}")

paper_trader.add_callback('account_update', on_account_update)

# Add callback for position changes
def on_position_change(symbol, position_data):
    if not position_data:
        print(f"Position closed: {symbol}")
    else:
        print(f"Position: {symbol}, Quantity: {position_data['quantity']}")

paper_trader.add_callback('position_change', on_position_change)

# Add callback for order status updates
def on_order_status(order_data):
    print(f"Order {order_data['order_id']} status: {order_data['status']}")

paper_trader.add_callback('order_status', on_order_status)

# Add callback for trade executions
def on_trade(trade_data):
    print(f"Trade executed: {trade_data['action']} {trade_data['quantity']} "
          f"{trade_data['symbol']} @ ${trade_data['price']:.2f}")

paper_trader.add_callback('trade', on_trade)

# Add callback for market data updates
def on_market_data(symbol, market_data):
    print(f"{symbol} price: ${market_data['last_price']:.2f}")

paper_trader.add_callback('market_data', on_market_data)

# Remove a callback
paper_trader.remove_callback('account_update', on_account_update)
```

## Interactive Brokers Connector

The paper trading module uses the Interactive Brokers connector (`IBConnector`) internally for market data. The connector can also be used directly:

```python
from src.connectors.ib import IBConnector

ib = IBConnector(
    host='127.0.0.1',
    port=7497,           # 7497 for TWS Paper
    client_id=1,
    account=None,        # Auto-select first account if None
    timeout=30,
    read_only=True,      # Only for data retrieval, not order execution
    max_retry_count=3,
    retry_wait_time=5,
    auto_reconnect=True
)

# Connect to IB
ib.connect()

# Get market data
ticker = ib.get_market_data('AAPL-STK-SMART')

# Get historical data
data = ib.get_historical_data(
    symbol='AAPL-STK-SMART',
    duration='1 D',
    bar_size='1 min'
)

# Get contract details
details = ib.get_contract_details('AAPL-STK-SMART')

# Disconnect
ib.disconnect()
```

## Symbol Format

The system uses a standardized symbol format to identify instruments:

- **Stocks**: `{ticker}-STK-{exchange}`, e.g., `AAPL-STK-SMART`
- **Futures**: `{symbol}-{expiry}-FUT-{exchange}`, e.g., `ES-202306-FUT-GLOBEX`
- **Forex**: `{base}{quote}-CASH` or `{base}-{quote}-CASH`, e.g., `EURUSD-CASH` or `EUR-USD-CASH`
- **Indices**: `{symbol}-IND-{exchange}`, e.g., `SPX-IND-CBOE`

The `exchange` part is optional for stocks, defaulting to `SMART`. The system will also attempt to infer the asset type if a simplified format is used.

## Examples

### Basic Paper Trading Example

```python
from src.paper_trading import PaperTrader
import time

# Create paper trader
paper_trader = PaperTrader(initial_capital=100000.0)

# Start paper trading
paper_trader.start()

# Subscribe to market data
paper_trader.subscribe_market_data('AAPL-STK-SMART')

# Wait for market data
time.sleep(2)

# Place a market order
order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='BUY',
    quantity=100,
    order_type='MKT'
)

# Wait for execution
time.sleep(5)

# Check position
position = paper_trader.get_position('AAPL-STK-SMART')
print(f"Position: {position['quantity']} shares at ${position['avg_cost']:.2f}")

# Place a limit sell order
sell_order_id = paper_trader.place_order(
    symbol='AAPL-STK-SMART',
    action='SELL',
    quantity=100,
    order_type='LMT',
    limit_price=position['avg_cost'] * 1.01  # 1% profit target
)

# Run for a while
time.sleep(60)

# Stop paper trading
paper_trader.stop()
```

### Using Callbacks

```python
from src.paper_trading import PaperTrader
import time

# Create paper trader
paper_trader = PaperTrader(initial_capital=100000.0)

# Setup callbacks
def on_account_update(account_data):
    print(f"Account equity: ${account_data['equity']:.2f}")

def on_order_status(order_data):
    print(f"Order {order_data['order_id']} status: {order_data['status']}")

def on_trade(trade_data):
    print(f"Trade executed: {trade_data['action']} {trade_data['quantity']} "
          f"{trade_data['symbol']} @ ${trade_data['price']:.2f}")

# Register callbacks
paper_trader.add_callback('account_update', on_account_update)
paper_trader.add_callback('order_status', on_order_status)
paper_trader.add_callback('trade', on_trade)

# Start paper trading
paper_trader.start()

# Rest of the example...
```

### Integrated with Strategy Components

```python
from src.paper_trading import PaperTrader
from src.signals import SignalGenerator
from src.risk import RiskManager
import time

# Initialize components
paper_trader = PaperTrader(initial_capital=100000.0)
signal_generator = SignalGenerator()
risk_manager = RiskManager(initial_capital=100000.0)

# Start paper trading
paper_trader.start()

# Subscribe to pair symbols
paper_trader.subscribe_market_data('ES-202306-FUT-GLOBEX')
paper_trader.subscribe_market_data('NQ-202306-FUT-GLOBEX')

# Main trading loop
try:
    while True:
        # Get current market data
        es_data = paper_trader.get_market_data('ES-202306-FUT-GLOBEX')
        nq_data = paper_trader.get_market_data('NQ-202306-FUT-GLOBEX')
        
        # Generate signals (simplified example)
        signal = signal_generator.generate_signal(es_data, nq_data)
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(signal)
        
        # Execute trades based on signal
        if signal['action'] == 'BUY':
            paper_trader.place_order(
                symbol=signal['symbol'],
                action='BUY',
                quantity=position_size,
                order_type='MKT'
            )
        elif signal['action'] == 'SELL':
            paper_trader.place_order(
                symbol=signal['symbol'],
                action='SELL',
                quantity=position_size,
                order_type='MKT'
            )
        
        # Wait for next update
        time.sleep(10)
        
except KeyboardInterrupt:
    # Stop paper trading on Ctrl+C
    paper_trader.stop() 