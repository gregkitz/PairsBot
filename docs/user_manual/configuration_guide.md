# Configuration Guide: Intraday Statistical Arbitrage System

This guide provides detailed instructions for setting up and configuring the Intraday Statistical Arbitrage System for futures pairs trading.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Interactive Brokers Setup](#interactive-brokers-setup)
4. [Configuration Options](#configuration-options)
5. [Operating Modes](#operating-modes)
6. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended for parallel processing
- **Memory**: 8GB minimum, 16GB+ recommended
- **Disk**: 50GB+ free space for data storage

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.8 or higher
- **Interactive Brokers**:
  - TWS (Trader Workstation) version 981+, or
  - IB Gateway version 981+

### Network Requirements
- **Internet Connection**: Stable broadband connection
- **Latency**: Lower is better, especially for live trading

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/quant-trader.git
cd quant-trader
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Data Directories

```bash
mkdir -p data/historical
mkdir -p data/market
mkdir -p data/results
```

## Interactive Brokers Setup

### 1. Install TWS or IB Gateway

Download and install the latest version of TWS or IB Gateway from the [Interactive Brokers website](https://www.interactivebrokers.com).

### 2. API Configuration

1. Launch TWS or IB Gateway
2. Go to **File** > **Global Configuration** (TWS) or **Settings** (Gateway)
3. Select **API** > **Settings**
4. Configure the following settings:
   - Enable Active X and Socket Clients: **Yes**
   - Socket port: **7496** (live) or **7497** (paper)
   - Allow connections from localhost only: **Recommended for security**
   - Read-Only API: **No** (system needs to place orders)
   - Create detailed logs: **Yes** (useful for troubleshooting)

### 3. Authentication

For live trading, ensure you have:
- Valid IB account credentials
- Market data subscriptions for the futures you intend to trade
- Permissions for futures trading enabled on your account

### 4. Paper Trading Configuration

For paper trading:
- Use port 7497 instead of 7496
- Log in with your IB credentials but select "Paper Trading" mode
- Configure your paper trading account with realistic initial capital

## Configuration Options

The system uses JSON configuration files located in the `config/` directory:

### Main Configuration (`config/config.json`)

```json
{
  "mode": "backtest",  // Options: "backtest", "paper", "live"
  "data_directory": "data/",
  "log_level": "INFO",
  "timezone": "UTC",
  "api": {
    "broker": "interactive_brokers",
    "host": "127.0.0.1",
    "port": 7496,
    "client_id": 1,
    "read_only": false
  }
}
```

### Strategy Configuration (`config/strategy.json`)

```json
{
  "strategy_name": "intraday_pairs_stat_arb",
  "pairs": [
    {"leg1": "ES", "leg2": "NQ", "ratio": 1.0},
    {"leg1": "GC", "leg2": "SI", "ratio": 0.1}
  ],
  "entry_threshold": 2.0,
  "exit_threshold": 0.5,
  "stop_loss_threshold": 3.0,
  "max_holding_period_minutes": 180,
  "entry_time_filter": {
    "start": "09:30:00",
    "end": "15:30:00"
  },
  "position_sizing": {
    "method": "volatility",
    "max_allocation_pct": 15.0,
    "max_risk_pct": 1.0,
    "volatility_lookback_days": [10, 20, 30]
  }
}
```

### Risk Management Configuration (`config/risk.json`)

```json
{
  "max_daily_loss_pct": 1.0,
  "max_position_pct": 15.0,
  "max_pair_correlation_min": 0.5,
  "emergency_stop": true,
  "slippage_model": {
    "fixed_ticks": 1,
    "percent": 0.0,
    "market_impact": 0.0
  },
  "commission_model": {
    "per_contract": 0.85,
    "per_order": 0.0,
    "percent": 0.0,
    "minimum": 0.0
  }
}
```

### Monitoring Configuration (`config/monitoring.json`)

```json
{
  "enable_email_alerts": false,
  "email_settings": {
    "smtp_server": "smtp.gmail.com",
    "port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "sender": "your-email@gmail.com",
    "recipients": ["alerts-recipient@example.com"]
  },
  "enable_sms_alerts": false,
  "sms_settings": {
    "provider": "twilio",
    "account_sid": "your-twilio-account-sid",
    "auth_token": "your-twilio-auth-token",
    "from_number": "your-twilio-number",
    "to_numbers": ["your-phone-number"]
  },
  "alert_levels": {
    "daily_loss_pct": 0.5,
    "drawdown_pct": 2.0,
    "error_count": 3,
    "missed_heartbeats": 2
  },
  "heartbeat_interval": 30,
  "check_interval": 60,
  "auto_shutdown_time": "16:00"
}
```

## Operating Modes

### Backtesting Mode

Run historical backtests to validate strategy performance:

```bash
# Basic backtest
python run.py --mode backtest

# Backtest with custom configuration
python run.py --mode backtest --config custom_config.json

# Backtest with specific date range
python run.py --mode backtest --start-date 2022-01-01 --end-date 2022-12-31
```

### Paper Trading Mode

Test with real-time market data but simulated executions:

```bash
# Start paper trading
python run.py --mode paper

# Start paper trading with TWS on a different port
python run.py --mode paper --port 4002
```

### Live Trading Mode

Execute trades with real money (use with caution):

```bash
# Start live trading
python run.py --mode live

# Start live trading with specific account
python run.py --mode live --account U12345678
```

## Troubleshooting

### Common Issues

#### Connection Problems

```
Error: Failed to connect to Interactive Brokers
```

**Solutions**:
- Confirm TWS/Gateway is running
- Verify API settings in TWS/Gateway
- Check port configuration matches in TWS and your config file
- Make sure TWS is allowing API connections
- Check your firewall isn't blocking connections

#### Data Issues

```
Error: No market data received for symbol XYZ
```

**Solutions**:
- Verify you have market data subscriptions for the requested symbols
- Check contract specifications are correct
- Ensure your account has permissions for the requested market data

#### Order Execution Problems

```
Error: Order rejected - XXX
```

**Solutions**:
- Check account has sufficient funds
- Verify trading permissions for the instrument
- Ensure order parameters are valid
- Check for any exchange-specific restrictions

#### System Performance

If the system is running slowly:

- Reduce the number of pairs being monitored
- Increase the RAM allocation if possible
- Close other CPU-intensive applications
- Check disk space isn't running low
- Consider using a more powerful machine for production

## Advanced Configuration

### Custom Indicators

To add custom indicators, create Python files in the `src/indicators/` directory:

```python
# src/indicators/custom_indicator.py
def calculate_indicator(data, param1=10, param2=20):
    # Implementation here
    return result
```

Then reference them in your strategy configuration:

```json
"indicators": {
  "custom_indicator": {
    "param1": 15,
    "param2": 25
  }
}
```

### Custom Slippage Models

Create custom slippage models in `src/execution_optimization/slippage.py`:

```python
def custom_slippage_model(quantity, price, volatility):
    # Implementation here
    return slippage_cost
```

Reference in configuration:

```json
"slippage_model": {
  "model": "custom",
  "volatility_factor": 1.5
}
```

### Performance Tuning

For high-frequency operations:

```json
"performance": {
  "use_parallel": true,
  "num_workers": 8,
  "cache_size_mb": 1024,
  "use_numpy_optimization": true
}
```

## Security Recommendations

1. **API Credentials**: Never store API credentials in config files directly; use environment variables
2. **Network Security**: Use localhost connections when possible
3. **Access Control**: Limit system access to authorized users only
4. **Regular Updates**: Keep TWS/Gateway and system dependencies updated
5. **Logs**: Regularly review logs for unauthorized access attempts

## Backup Procedures

1. Regularly backup configuration files
2. Export and backup trading history monthly
3. Consider version control for configuration changes
4. Backup position data to prevent discrepancies

---

This guide provides the foundation for configuring the Intraday Statistical Arbitrage System. For advanced use cases or specific questions, please refer to the API documentation or contact the development team. 