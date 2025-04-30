# Paper Trading Script Organization

## Current Status

We currently have multiple scripts for paper trading:

1. **run_paper_trade_fixed.py** (~2185 lines):
   - Currently the most actively used script (based on logs)
   - Includes fixes for JSON serialization with datetime objects
   - Contains extensive custom code for IB contract handling
   - Has been heavily tested and debugged based on log history
   - Functions as a monolithic script rather than using modular architecture

2. **run_ml_paper_trader.py** (~200 lines):
   - Well-structured entry point script
   - Uses proper argument parsing and config loading
   - Calls the underlying `IntradayMLPaperTrader` class
   - Appears to be less actively used based on log history

3. **src/paper_trading/intraday_ml_paper_trader.py** (~1546 lines):
   - Well-structured class implementation using proper software engineering principles
   - Contains ML enhancement functionality
   - Called by both scripts above

4. **run_paper_trader.py** (~32 lines):
   - Basic script for running the base PaperTrader without ML enhancements
   - Appears to be an early/simple version or starting point

## Recommended Organization

To improve codebase cohesion and prevent confusion, I recommend the following organization:

1. **Keep and enhance**:
   - `src/paper_trading/intraday_ml_paper_trader.py`: This is the well-structured core implementation.
   - `run_ml_paper_trader.py`: This is a clean entry point for running the system.

2. **Move and archive**:
   - `run_paper_trade_fixed.py`: Move to `archive/run_paper_trade_fixed.py` with a comment header explaining it was an earlier version.
   - `run_paper_trader.py`: Move to `archive/run_paper_trader.py` as it's just the basic version.

3. **Refactor**:
   - Extract the datetime JSON serialization fixes from `run_paper_trade_fixed.py` into a utility function in `src/utils/json_utils.py`
   - Extract any specialized contract handling from `run_paper_trade_fixed.py` that isn't already in the proper classes and move it into appropriate modules

4. **Update documentation**:
   - Update all documentation to refer to `run_ml_paper_trader.py` as the primary entry point
   - Create a consolidated help/guide for paper trading that explains the workflow

## Implementation Plan

1. First, ensure that all fixes from `run_paper_trade_fixed.py` are properly integrated into the modular codebase.
2. Test that `run_ml_paper_trader.py` works properly with the full functionality.
3. Move the deprecated scripts to the archive folder.
4. Update documentation to prevent confusion.

## Notes on Specific Fixes to Transfer

The critical functionality from `run_paper_trade_fixed.py` that needs to be transferred:

1. JSON datetime serialization (for compatibility with the API and data storage)
2. The specialized futures contract handling and continuous contract representation
3. Any error handling or timeout mechanisms for IB API calls
4. Any enhanced signal processing not already in the main implementation

Once these elements are properly integrated into the modular codebase, the fixed script can be safely archived. 