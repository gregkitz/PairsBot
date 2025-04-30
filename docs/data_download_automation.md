# Data Download Automation

This document provides detailed instructions for automating the data download process from Interactive Brokers (IB).

## Overview

The data download automation system periodically retrieves market data from Interactive Brokers and stores it in a structured format for later use by the analysis and trading systems. The process is designed to run daily after market close.

## Prerequisites

- Interactive Brokers account
- IB TWS or IB Gateway installed and configured
- Python 3.8+ with required packages from `requirements.txt`
- Windows OS (for Task Scheduler automation)

## Manual Data Download

You can manually run the data download script as follows:

```bash
# Basic usage (uses defaults from config/data_config.yaml)
python scripts/download_data.py

# Download specific symbols
python scripts/download_data.py --symbols ES,NQ,GC

# Download for a specific number of days
python scripts/download_data.py --days 5

# Specify IB port (if different than default 7497)
python scripts/download_data.py --port 7496  # For live TWS

# Use sequential processing instead of parallel (more stable)
python scripts/download_data.py --sequential
```

## Configuration

The data download process is configured via the `config/data_config.yaml` file. Key settings include:

- **IB connection settings**: Host, port, client ID
- **Symbols to download**: List of futures symbols (ES, NQ, etc.)
- **Timeframes**: 1min, 5min, 15min, 1hour, 1day
- **Parallel processing**: Enable/disable parallel downloading

Example configuration:

```yaml
data_download:
  ib_host: "127.0.0.1"
  ib_port: 7497  # 7497 for TWS Paper, 7496 for TWS Live
  client_id_base: 10000
  client_id_spread: 1000
  data_dir: "data/historical"
  symbols:
    - ES  # E-mini S&P 500
    - NQ  # E-mini NASDAQ-100
  timeframes:
    - 1min
    - 5min
    - 1hour
    - 1day
  days_to_download: 10
  parallel: false
```

## Windows Task Scheduler Automation

The script `scripts/schedule_data_download.ps1` will create a Windows Task Scheduler task to run the data download process daily at 5:30 PM (17:30) local time.

To set up the scheduled task:

1. Open PowerShell as Administrator
2. Navigate to the project directory
3. Run the script:

```powershell
cd \path\to\quant-trader
powershell -ExecutionPolicy Bypass -File scripts\schedule_data_download.ps1
```

The script will create a task named "QuantTrader_DataDownload" that executes the download script daily.

## IB Gateway Setup

For reliable automation, it's recommended to use IB Gateway rather than TWS:

1. Configure IB Gateway to auto-login and stay running
2. Set it to automatically start with Windows
3. Ensure it's set to accept API connections on port 7497
4. Disable the "Auto logout" feature

## Data Storage

Downloaded data is stored in the following directory structure:

```
data/
└── historical/
    ├── ES/
    │   ├── 1min/
    │   │   └── data.csv
    │   ├── 5min/
    │   │   └── data.csv
    │   └── 1day/
    │       └── data.csv
    ├── NQ/
    │   └── ...
    └── ...
```

Each CSV file contains OHLCV data for the respective symbol and timeframe.

## Troubleshooting

### Common Issues

1. **Connection Errors**: Ensure IB Gateway/TWS is running and accepting API connections.
2. **Client ID Conflicts**: If you see "client ID already in use" errors, try running with a different client ID using `--client-id`.
3. **Symbol Not Found**: Verify the symbol exists and is accessible in your IB account.

### Logs

Logs are stored in the `logs/` directory:
- `data_download.log`: Main logging output
- `data_download_task.log`: Scheduled task execution log

## Next Steps

After the data is downloaded, you should run the feature generation process:

```bash
python scripts/generate_features.py
```

This will calculate technical indicators and prepare the data for the ML modeling step. 