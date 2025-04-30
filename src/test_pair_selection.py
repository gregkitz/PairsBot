import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

# Add the project root to the path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.data_processor import DataProcessor
from src.cointegration.pair_finder import PairFinder
from src.cointegration.cointegration_tests import test_cointegration, rolling_cointegration

def main():
    """
    Test the pair selection framework by finding and analyzing cointegrated futures pairs.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze futures pairs for cointegration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--tickers", nargs='+', help="List of tickers to analyze")
    parser.add_argument("--ticker-file", type=str, help="File containing tickers to analyze")
    parser.add_argument("--min-correlation", type=float, default=0.7, help="Minimum correlation threshold")
    parser.add_argument("--output-file", type=str, default="output/pairs_analysis.json", help="Output file path")
    parser.add_argument("--lookback-days", type=int, default=60, help="Lookback window in days")
    parser.add_argument("--timeframe", type=str, default="1day", help="Timeframe for analysis")
    parser.add_argument("--start-date", type=str, default="2018-01-01", help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date for analysis (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    print("Starting pair selection test...")
    print(f"Arguments: {args}")
    
    # Initialize data processor
    data_processor = DataProcessor()
    print(f"Data processor initialized with paths: historical_dir={data_processor.historical_dir}")
    
    # Define futures to test (if not provided via command line)
    futures_to_test = args.tickers if args.tickers else [
        # Metals
        'GC',  # Gold
        'SI',  # Silver
        'HG',  # Copper
        'PL',  # Platinum
        'PA',  # Palladium
        
        # Energy
        'CL',  # Crude Oil
        'NG',  # Natural Gas
        'RB',  # Gasoline
        'HO',  # Heating Oil
        'BZ',  # Brent Crude
        
        # Interest Rates / Bonds
        'ZN',  # 10Y Treasury Note
        'ZF',  # 5Y Treasury Note
        'ZT',  # 2Y Treasury Note
        'ZB',  # 30Y Treasury Bond
        'UB',  # Ultra Treasury Bond
        
        # Equity Indices
        'ES',  # S&P 500
        'NQ',  # Nasdaq
        'YM',  # Dow Jones
        'RTY', # Russell 2000
        
        # Agricultural
        'ZC',  # Corn
        'ZW',  # Wheat
        'ZS',  # Soybeans
        'ZL',  # Soybean Oil
        'ZM',  # Soybean Meal
        
        # Currencies
        'EC',  # Euro
        'BP',  # British Pound
        'JY',  # Japanese Yen
        'AD',  # Australian Dollar
        'CD'   # Canadian Dollar
    ]
    
    print(f"Futures to test: {futures_to_test}")
    
    # Filter to include only available futures
    available_futures = []
    for ticker in futures_to_test:
        path = Path(f"data/historical/{ticker}/1day/data.csv")
        if path.exists():
            available_futures.append(ticker)
        else:
            print(f"Data file not found for {ticker}: {path}")
    
    print(f"Available futures: {available_futures}")
    
    if not available_futures:
        print("No futures data found. Please check data directory.")
        return
    
    # Load daily data for the available futures
    try:
        print(f"Loading data for {len(available_futures)} futures...")
        daily_data = data_processor.load_daily_data(
            available_futures,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if daily_data is None or daily_data.empty:
            print("Error: No data loaded. Check if futures data exists.")
            return
        
        print(f"Loaded data for {len(daily_data.columns)} futures from {daily_data.index[0]} to {daily_data.index[-1]}")
        print(f"Available futures: {list(daily_data.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize pair finder
    print(f"Initializing PairFinder with min_correlation={args.min_correlation}...")
    pair_finder = PairFinder(
        min_correlation=args.min_correlation,
        max_half_life=20,
        min_half_life=1,
        min_cointegration_pct=0.6,
        train_test_split=0.7
    )
    
    # Find cointegrated pairs
    print("Finding cointegrated pairs...")
    pairs = pair_finder.find_cointegrated_pairs(daily_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Save pairs to JSON file
    print(f"Saving {len(pairs)} pairs to {args.output_file}...")
    pair_finder.save_pairs_to_json(pairs, args.output_file)
    
    print(f"Found {len(pairs)} cointegrated pairs. Results saved to {args.output_file}")
    
    # Analyze the top N pairs in more detail
    top_n = min(5, len(pairs))
    if top_n > 0:
        print(f"\nAnalyzing top {top_n} pairs in detail...")
        
        # Sort pairs by correlation
        sorted_pairs = sorted(pairs, key=lambda x: x['correlation'], reverse=True)
        
        for i, pair in enumerate(sorted_pairs[:top_n]):
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            
            print(f"\n{i+1}. {ticker1}-{ticker2}: " + 
                f"correlation={pair['correlation']:.4f}, " + 
                f"p_value={pair['p_value']:.4f}, " + 
                f"hedge_ratio={pair['hedge_ratio']:.4f}, " + 
                f"half_life={pair['half_life']:.2f} days")
    else:
        print("No cointegrated pairs found.")
    
    print("Pair selection test completed.")

if __name__ == "__main__":
    main() 