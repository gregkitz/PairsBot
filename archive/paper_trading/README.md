# Archived Paper Trading Scripts

This directory contains archived versions of paper trading scripts that have been superseded by newer, more modular implementations.

## Archived Scripts

1. **run_paper_trade_fixed.py**: 
   - Large monolithic script (~2185 lines) that was developed to address issues with JSON serialization and contract handling
   - Contains many fixes and workarounds for Interactive Brokers API issues
   - Was the most actively used script for paper trading as of March 2025
   - Archived on: 2025-03-22

2. **run_paper_trader.py**:
   - Simple script (~32 lines) that just runs the base PaperTrader class without ML enhancements
   - Was an early version/starting point for the paper trading system
   - Archived on: 2025-03-22

## Improved Implementation

The key functionality from these scripts has been extracted and integrated into a more modular codebase:

1. **JSON serialization** has been moved to `src/utils/json_utils.py`
2. **Contract handling** has been moved to `src/connectors/ib/contract_utils.py`
3. **The main entry point** is now `run_ml_paper_trader.py` which uses these utility modules

## Usage

Please use the newer script `run_ml_paper_trader.py` for all paper trading activities. It provides a more maintainable and extensible solution with the same functionality as the archived scripts.

Example:
```bash
python run_ml_paper_trader.py --config config/paper_trading.json
``` 