"""
Benchmark metrics for paper trading validation.

This module provides benchmark calculations and comparison tools
to evaluate trading strategies against standard benchmarks.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ValidationBenchmarks:
    """
    Benchmark calculations for trading strategy performance comparison.
    
    This class provides tools to compare trading strategy performance
    against common benchmarks like buy-and-hold, S&P 500, and others.
    """
    
    @staticmethod
    def buy_and_hold_benchmark(price_data, initial_capital=100000.0):
        """
        Calculate benchmark for simple buy and hold strategy.
        
        Args:
            price_data (pd.Series): Price data for asset
            initial_capital (float): Initial capital
            
        Returns:
            pd.Series: Equity curve for buy and hold strategy
        """
        # Calculate number of units that could be purchased initially
        first_price = price_data.iloc[0]
        units = initial_capital / first_price
        
        # Generate equity curve
        equity_curve = price_data * units
        
        return equity_curve
    
    @staticmethod
    def rebalanced_portfolio_benchmark(price_data_dict, weights=None, 
                                      initial_capital=100000.0, 
                                      rebalance_frequency='M'):
        """
        Calculate benchmark for periodically rebalanced portfolio.
        
        Args:
            price_data_dict (dict): Dictionary of price series for assets
            weights (dict): Asset allocation weights (defaults to equal weighting)
            initial_capital (float): Initial capital
            rebalance_frequency (str): Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            pd.Series: Equity curve for rebalanced portfolio
        """
        # Convert all series to DataFrame with common index
        assets = list(price_data_dict.keys())
        
        # Create DataFrame with aligned dates
        df = pd.DataFrame({asset: price_data_dict[asset] for asset in assets})
        
        # Equal weights if not specified
        if weights is None:
            weights = {asset: 1.0 / len(assets) for asset in assets}
        
        # Initialize portfolio
        equity = pd.Series(index=df.index)
        equity.iloc[0] = initial_capital
        
        # Get rebalance dates based on frequency
        if rebalance_frequency == 'D':
            rebalance_dates = df.index
        else:
            # Convert index to period and find period boundaries
            rebalance_dates = df.index[df.index.to_period(rebalance_frequency).to_timestamp() == df.index]
            
            # Ensure first and last dates are included
            if rebalance_dates[0] != df.index[0]:
                rebalance_dates = df.index[[0]].append(rebalance_dates)
            
            if rebalance_dates[-1] != df.index[-1]:
                rebalance_dates = rebalance_dates.append(df.index[[-1]])
        
        # Calculate units at each rebalance date
        units = pd.DataFrame(0, index=df.index, columns=assets)
        
        # Process each rebalance period
        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i+1]
            
            # Capital at start of period
            current_capital = equity.loc[start_date]
            
            # Calculate units for each asset based on weights
            for asset in assets:
                units.loc[start_date:end_date, asset] = weights[asset] * current_capital / df.loc[start_date, asset]
            
            # Calculate equity for this period
            period_equity = (df.loc[start_date:end_date] * units.loc[start_date:end_date]).sum(axis=1)
            equity.loc[start_date:end_date] = period_equity
        
        return equity
    
    @staticmethod
    def generate_market_index_benchmark(price_data, beta=1.0, 
                                       volatility_ratio=1.0, 
                                       correlation=0.7,
                                       initial_value=100.0):
        """
        Generate a benchmark that behaves like a market index with specified properties.
        
        Args:
            price_data (pd.Series): Original price data to match dates
            beta (float): Beta relative to original price data
            volatility_ratio (float): Ratio of benchmark volatility to original data
            correlation (float): Correlation between benchmark and original data
            initial_value (float): Starting value of benchmark
            
        Returns:
            pd.Series: Synthetic market index benchmark
        """
        # Calculate returns of original data
        returns = price_data.pct_change().dropna()
        
        # Generate correlated returns for benchmark
        # Using beta * returns + uncorrelated component weighted by desired correlation
        benchmark_returns = pd.Series(index=returns.index)
        
        # Mean and standard deviation of original returns
        orig_mean = returns.mean()
        orig_std = returns.std()
        
        # Target standard deviation for benchmark
        target_std = orig_std * volatility_ratio
        
        # Generate correlated component
        benchmark_returns = beta * returns
        
        # Generate uncorrelated component
        uncorrelated = pd.Series(np.random.normal(orig_mean, orig_std, len(returns)), index=returns.index)
        
        # Combine correlated and uncorrelated components based on desired correlation
        benchmark_returns = correlation * benchmark_returns + (1 - correlation) * uncorrelated
        
        # Scale to desired volatility
        benchmark_returns = benchmark_returns * (target_std / benchmark_returns.std())
        
        # Convert returns to prices
        benchmark_prices = (1 + benchmark_returns).cumprod() * initial_value
        
        # Add initial value
        benchmark_prices = pd.Series([initial_value], index=[price_data.index[0]]).append(benchmark_prices)
        
        return benchmark_prices
    
    @staticmethod
    def mean_reversion_benchmark(price_data, lookback=20, 
                               entry_z=2.0, exit_z=0.0, 
                               initial_capital=100000.0):
        """
        Calculate benchmark for basic mean reversion strategy.
        
        Args:
            price_data (pd.Series): Price data for asset
            lookback (int): Lookback period for z-score calculation
            entry_z (float): Z-score threshold for entry
            exit_z (float): Z-score threshold for exit
            initial_capital (float): Initial capital
            
        Returns:
            tuple: (pd.Series, list) - Equity curve and trades for mean reversion strategy
        """
        # Calculate rolling mean and standard deviation
        rolling_mean = price_data.rolling(window=lookback).mean()
        rolling_std = price_data.rolling(window=lookback).std()
        
        # Calculate z-scores
        z_scores = (price_data - rolling_mean) / rolling_std
        
        # Initialize position and equity arrays
        position = pd.Series(0, index=price_data.index)
        equity = pd.Series(initial_capital, index=price_data.index)
        cash = initial_capital
        units = 0
        
        # List to track trades
        trades = []
        
        # Simulate trading
        for i in range(lookback, len(price_data)):
            yesterday = price_data.index[i-1]
            today = price_data.index[i]
            price = price_data.iloc[i]
            z = z_scores.iloc[i]
            
            # Check for position entry/exit
            if position.iloc[i-1] == 0:  # No position
                if z <= -abs(entry_z):  # Go long on negative z-score crossing threshold
                    units = cash / price
                    cash = 0
                    position.iloc[i] = 1
                    trade = {
                        'entry_date': today,
                        'entry_price': price,
                        'direction': 'LONG',
                        'units': units
                    }
                    trades.append(trade)
                elif z >= abs(entry_z):  # Go short on positive z-score crossing threshold
                    units = cash / price
                    cash = initial_capital + initial_capital  # Account for short selling
                    position.iloc[i] = -1
                    trade = {
                        'entry_date': today,
                        'entry_price': price,
                        'direction': 'SHORT',
                        'units': units
                    }
                    trades.append(trade)
                else:
                    position.iloc[i] = 0
            
            elif position.iloc[i-1] == 1:  # Long position
                if z >= exit_z:  # Exit long
                    cash = units * price
                    trades[-1]['exit_date'] = today
                    trades[-1]['exit_price'] = price
                    trades[-1]['pnl'] = units * (price - trades[-1]['entry_price'])
                    trades[-1]['return'] = price / trades[-1]['entry_price'] - 1
                    position.iloc[i] = 0
                    units = 0
                else:
                    position.iloc[i] = 1
            
            elif position.iloc[i-1] == -1:  # Short position
                if z <= exit_z:  # Exit short
                    cash = initial_capital - units * (price - trades[-1]['entry_price'])
                    trades[-1]['exit_date'] = today
                    trades[-1]['exit_price'] = price
                    trades[-1]['pnl'] = units * (trades[-1]['entry_price'] - price)
                    trades[-1]['return'] = trades[-1]['entry_price'] / price - 1
                    position.iloc[i] = 0
                    units = 0
                else:
                    position.iloc[i] = -1
            
            # Update equity
            if position.iloc[i] == 0:
                equity.iloc[i] = cash
            elif position.iloc[i] == 1:
                equity.iloc[i] = units * price
            else:  # Short position
                equity.iloc[i] = initial_capital + units * (trades[-1]['entry_price'] - price)
        
        # Close any open positions at the end
        if position.iloc[-1] != 0:
            last_day = price_data.index[-1]
            last_price = price_data.iloc[-1]
            
            if position.iloc[-1] == 1:  # Long position
                trades[-1]['exit_date'] = last_day
                trades[-1]['exit_price'] = last_price
                trades[-1]['pnl'] = units * (last_price - trades[-1]['entry_price'])
                trades[-1]['return'] = last_price / trades[-1]['entry_price'] - 1
            else:  # Short position
                trades[-1]['exit_date'] = last_day
                trades[-1]['exit_price'] = last_price
                trades[-1]['pnl'] = units * (trades[-1]['entry_price'] - last_price)
                trades[-1]['return'] = trades[-1]['entry_price'] / last_price - 1
        
        return equity, trades
    
    @staticmethod
    def trend_following_benchmark(price_data, fast_ma=10, slow_ma=50, 
                                initial_capital=100000.0):
        """
        Calculate benchmark for basic trend following strategy.
        
        Args:
            price_data (pd.Series): Price data for asset
            fast_ma (int): Fast moving average period
            slow_ma (int): Slow moving average period
            initial_capital (float): Initial capital
            
        Returns:
            tuple: (pd.Series, list) - Equity curve and trades for trend following strategy
        """
        # Calculate moving averages
        fast = price_data.rolling(window=fast_ma).mean()
        slow = price_data.rolling(window=slow_ma).mean()
        
        # Generate trading signals
        signal = pd.Series(0, index=price_data.index)
        signal[fast > slow] = 1  # Long when fast MA above slow MA
        signal[fast < slow] = -1  # Short when fast MA below slow MA
        
        # Initialize position and equity arrays
        position = pd.Series(0, index=price_data.index)
        equity = pd.Series(initial_capital, index=price_data.index)
        cash = initial_capital
        units = 0
        
        # List to track trades
        trades = []
        
        # Simulate trading
        for i in range(slow_ma, len(price_data)):
            yesterday = price_data.index[i-1]
            today = price_data.index[i]
            price = price_data.iloc[i]
            sig = signal.iloc[i]
            
            # Check for position changes
            if position.iloc[i-1] != sig:  # Position change
                # Close existing position if any
                if position.iloc[i-1] != 0:
                    last_trade = trades[-1]
                    last_trade['exit_date'] = today
                    last_trade['exit_price'] = price
                    
                    if position.iloc[i-1] == 1:  # Exiting long
                        cash = units * price
                        last_trade['pnl'] = units * (price - last_trade['entry_price'])
                        last_trade['return'] = price / last_trade['entry_price'] - 1
                    else:  # Exiting short
                        cash = initial_capital - units * (price - last_trade['entry_price'])
                        last_trade['pnl'] = units * (last_trade['entry_price'] - price)
                        last_trade['return'] = last_trade['entry_price'] / price - 1
                    
                    units = 0
                
                # Enter new position if signal is not zero
                if sig != 0:
                    units = cash / price
                    
                    if sig == 1:  # Enter long
                        cash = 0
                        trade = {
                            'entry_date': today,
                            'entry_price': price,
                            'direction': 'LONG',
                            'units': units
                        }
                        trades.append(trade)
                    else:  # Enter short
                        cash = initial_capital + initial_capital  # Account for short selling
                        trade = {
                            'entry_date': today,
                            'entry_price': price,
                            'direction': 'SHORT',
                            'units': units
                        }
                        trades.append(trade)
                
                # Update position
                position.iloc[i] = sig
            else:
                # Maintain current position
                position.iloc[i] = position.iloc[i-1]
            
            # Update equity
            if position.iloc[i] == 0:
                equity.iloc[i] = cash
            elif position.iloc[i] == 1:
                equity.iloc[i] = units * price
            else:  # Short position
                equity.iloc[i] = initial_capital + units * (trades[-1]['entry_price'] - price)
        
        # Close any open positions at the end
        if position.iloc[-1] != 0:
            last_day = price_data.index[-1]
            last_price = price_data.iloc[-1]
            
            if position.iloc[-1] == 1:  # Long position
                trades[-1]['exit_date'] = last_day
                trades[-1]['exit_price'] = last_price
                trades[-1]['pnl'] = units * (last_price - trades[-1]['entry_price'])
                trades[-1]['return'] = last_price / trades[-1]['entry_price'] - 1
            else:  # Short position
                trades[-1]['exit_date'] = last_day
                trades[-1]['exit_price'] = last_price
                trades[-1]['pnl'] = units * (trades[-1]['entry_price'] - last_price)
                trades[-1]['return'] = trades[-1]['entry_price'] / last_price - 1
        
        return equity, trades
    
    @staticmethod
    def run_all_benchmarks(price_data, initial_capital=100000.0):
        """
        Run all benchmark strategies on price data.
        
        Args:
            price_data (pd.Series): Price data for asset
            initial_capital (float): Initial capital
            
        Returns:
            dict: Dictionary of equity curves and trades for all benchmarks
        """
        results = {}
        
        # Buy and hold
        results['buy_and_hold'] = {
            'equity': ValidationBenchmarks.buy_and_hold_benchmark(price_data, initial_capital),
            'trades': [{
                'entry_date': price_data.index[0],
                'entry_price': price_data.iloc[0],
                'exit_date': price_data.index[-1],
                'exit_price': price_data.iloc[-1],
                'pnl': initial_capital * (price_data.iloc[-1] / price_data.iloc[0] - 1),
                'return': price_data.iloc[-1] / price_data.iloc[0] - 1
            }]
        }
        
        # Market index (synthetic)
        market_index = ValidationBenchmarks.generate_market_index_benchmark(price_data)
        results['market_index'] = {
            'equity': market_index * (initial_capital / market_index.iloc[0]),
            'trades': [{
                'entry_date': market_index.index[0],
                'entry_price': market_index.iloc[0],
                'exit_date': market_index.index[-1],
                'exit_price': market_index.iloc[-1],
                'pnl': initial_capital * (market_index.iloc[-1] / market_index.iloc[0] - 1),
                'return': market_index.iloc[-1] / market_index.iloc[0] - 1
            }]
        }
        
        # Mean reversion
        mean_rev_equity, mean_rev_trades = ValidationBenchmarks.mean_reversion_benchmark(
            price_data, initial_capital=initial_capital
        )
        results['mean_reversion'] = {
            'equity': mean_rev_equity,
            'trades': mean_rev_trades
        }
        
        # Trend following
        trend_equity, trend_trades = ValidationBenchmarks.trend_following_benchmark(
            price_data, initial_capital=initial_capital
        )
        results['trend_following'] = {
            'equity': trend_equity,
            'trades': trend_trades
        }
        
        return results
    
    @staticmethod
    def save_benchmark_results(results, output_dir):
        """
        Save benchmark results to output directory.
        
        Args:
            results (dict): Benchmark results from run_all_benchmarks
            output_dir (str): Directory to save results
            
        Returns:
            str: Path to saved results file
        """
        # Create benchmarks directory if it doesn't exist
        benchmark_dir = os.path.join(output_dir, "benchmarks")
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Prepare data for serialization
        serializable_results = {}
        
        for benchmark, data in results.items():
            # Convert equity curve to list
            equity_list = data['equity'].reset_index()
            equity_list.columns = ['date', 'equity']
            equity_list['date'] = equity_list['date'].astype(str)
            
            # Convert trades to list
            trades_list = []
            for trade in data['trades']:
                # Convert dates to strings
                serializable_trade = {k: (v.strftime('%Y-%m-%d') if 'date' in k else v) 
                                    for k, v in trade.items()}
                trades_list.append(serializable_trade)
            
            serializable_results[benchmark] = {
                'equity': equity_list.to_dict(orient='records'),
                'trades': trades_list
            }
        
        # Save to JSON file
        output_file = os.path.join(benchmark_dir, "benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        return output_file


# Test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tempfile
    from datetime import datetime
    
    # Create test data
    dates = pd.date_range(start=datetime(2020, 1, 1), periods=252)
    
    # Trending market with some noise
    price_data = pd.Series(100 + np.linspace(0, 30, len(dates)) + np.random.normal(0, 3, len(dates)), 
                          index=dates)
    
    # Test all benchmarks
    with tempfile.TemporaryDirectory() as temp_dir:
        benchmark_results = ValidationBenchmarks.run_all_benchmarks(price_data)
        output_file = ValidationBenchmarks.save_benchmark_results(benchmark_results, temp_dir)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        
        for benchmark, data in benchmark_results.items():
            plt.plot(data['equity'].index, data['equity'], label=benchmark)
        
        plt.title('Benchmark Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file = os.path.join(temp_dir, "benchmarks.png")
        plt.savefig(plot_file)
        
        print(f"Benchmark results saved to {output_file}")
        print(f"Benchmark plot saved to {plot_file}") 