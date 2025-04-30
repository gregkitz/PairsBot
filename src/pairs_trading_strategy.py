import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, time
from pathlib import Path

from src.data_processing.data_processor import DataProcessor
from src.cointegration.pair_finder import PairFinder
from src.signal_generation.spread_calculator import SpreadCalculator
from src.signal_generation.signal_generator import SignalGenerator
from src.risk_management.position_sizer import PositionSizer
from src.risk_management.risk_manager import RiskManager


class PairsTradingStrategy:
    """
    Main class for the pairs trading strategy. Integrates all components:
    - Pair selection and cointegration testing
    - Spread calculation with dynamic hedge ratio
    - Signal generation
    - Position sizing and risk management
    """
    
    def __init__(self, account_size=25000, max_risk_per_trade=0.01, max_allocation=0.15,
                entry_threshold=2.0, exit_threshold=0.5, max_holding_period=180,
                time_filters=None, daily_loss_limit=0.01, correlation_threshold=0.5):
        """
        Initialize the pairs trading strategy.
        
        Parameters:
        -----------
        account_size : float
            Size of trading account
        max_risk_per_trade : float
            Maximum risk per trade as fraction of account
        max_allocation : float
            Maximum allocation to any pair as fraction of account
        entry_threshold : float
            Z-score threshold for entry
        exit_threshold : float
            Z-score threshold for exit
        max_holding_period : int
            Maximum holding period in minutes
        time_filters : dict, optional
            Time-of-day filters
        daily_loss_limit : float
            Daily loss limit as fraction of account
        correlation_threshold : float
            Minimum correlation to maintain position
        """
        # Initialize account parameters
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_allocation = max_allocation
        
        # Initialize strategy parameters
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_holding_period = max_holding_period
        self.time_filters = time_filters
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.pair_finder = PairFinder()
        self.spread_calculator = SpreadCalculator()
        self.signal_generator = SignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            max_holding_period=max_holding_period,
            time_filters=time_filters
        )
        self.position_sizer = PositionSizer(
            account_size=account_size,
            max_risk=max_risk_per_trade,
            max_allocation=max_allocation
        )
        self.risk_manager = RiskManager(
            account_size=account_size,
            daily_loss_limit=daily_loss_limit,
            correlation_threshold=correlation_threshold
        )
        
        # Strategy state
        self.pairs = []
        self.pair_data = {}
        self.active_signals = {}
        self.active_positions = {}
        self.trade_history = []
        self.performance_metrics = {}
    
    def find_pairs(self, tickers=None, start_date=None, end_date=None, use_log_prices=True):
        """
        Find suitable pairs for trading.
        
        Parameters:
        -----------
        tickers : list, optional
            List of tickers to search for pairs. If None, use available futures.
        start_date : str, optional
            Start date for analysis in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for analysis in 'YYYY-MM-DD' format
        use_log_prices : bool
            Whether to use log prices for analysis
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with pair analysis results
        """
        # Use default futures if tickers not provided
        if tickers is None:
            tickers = self._get_default_futures()
        
        # Load data
        daily_data = self.data_processor.load_daily_data(tickers, start_date, end_date)
        
        if daily_data is None or daily_data.empty:
            print("No data available for pair finding.")
            return pd.DataFrame()
        
        # Find pairs
        pairs_results = self.pair_finder.find_pairs(daily_data, use_log_prices=use_log_prices)
        
        # Store pairs for strategy
        self.pairs = []
        for _, row in pairs_results.iterrows():
            self.pairs.append((row['ticker1'], row['ticker2'], row['hedge_ratio']))
        
        return pairs_results
    
    def initialize_strategy(self, pairs=None, start_date=None, end_date=None, timeframe='1hour'):
        """
        Initialize the strategy with pairs and data.
        
        Parameters:
        -----------
        pairs : list of tuples, optional
            List of (ticker1, ticker2, hedge_ratio) tuples. If None, use predefined pairs.
        start_date : str, optional
            Start date for data in 'YYYY-MM-DD' format
        end_date : str, optional
            End date for data in 'YYYY-MM-DD' format
        timeframe : str
            Timeframe for data ('1min', '5min', '1hour', etc.)
        """
        # Use provided pairs or find new ones
        if pairs is not None:
            self.pairs = pairs
        elif not self.pairs:
            print("No pairs defined. Finding pairs...")
            self.find_pairs(start_date=start_date, end_date=end_date)
        
        if not self.pairs:
            print("No suitable pairs found for trading.")
            return False
        
        # Load data for all unique tickers in pairs
        all_tickers = set()
        for ticker1, ticker2, _ in self.pairs:
            all_tickers.add(ticker1)
            all_tickers.add(ticker2)
        
        # Load data for the specified timeframe
        print(f"Loading {timeframe} data for {len(all_tickers)} tickers...")
        ticker_data = self.data_processor.load_intraday_data(
            list(all_tickers),
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if not ticker_data:
            print("Failed to load ticker data.")
            return False
        
        # Prepare pair data
        self.pair_data = {}
        for ticker1, ticker2, hedge_ratio in self.pairs:
            if ticker1 not in ticker_data or ticker2 not in ticker_data:
                print(f"Data missing for pair {ticker1}/{ticker2}. Skipping.")
                continue
            
            # Align data for the pair
            aligned_data = self.data_processor.align_pairs_data(
                ticker_data, 
                tickers=[ticker1, ticker2]
            )
            
            if aligned_data.empty or len(aligned_data.columns) < 2:
                print(f"Failed to align data for pair {ticker1}/{ticker2}. Skipping.")
                continue
            
            # Store the pair data
            pair_id = f"{ticker1}_{ticker2}"
            self.pair_data[pair_id] = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'hedge_ratio': hedge_ratio,
                'data': aligned_data,
                'is_active': True
            }
        
        print(f"Strategy initialized with {len(self.pair_data)} pairs.")
        return True
    
    def calculate_spreads(self, use_kalman=True):
        """
        Calculate spreads for all pairs.
        
        Parameters:
        -----------
        use_kalman : bool
            Whether to use Kalman filter for dynamic hedge ratio
            
        Returns:
        --------
        dict
            Dictionary of pair_id -> spread_data
        """
        for pair_id, pair_info in self.pair_data.items():
            if not pair_info['is_active']:
                continue
                
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            data = pair_info['data']
            
            if ticker1 not in data.columns or ticker2 not in data.columns:
                print(f"Missing data for {ticker1} or {ticker2} in pair {pair_id}.")
                continue
            
            # Calculate spread with static or dynamic hedge ratio
            if use_kalman:
                spread, hedge_ratio_series, _, _ = self.spread_calculator.calculate_kalman_spread(
                    data[ticker1], data[ticker2]
                )
                pair_info['hedge_ratio_series'] = hedge_ratio_series
            else:
                # Use the predefined hedge ratio
                hedge_ratio = pair_info['hedge_ratio']
                spread = data[ticker1] - hedge_ratio * data[ticker2]
                pair_info['hedge_ratio_series'] = pd.Series(hedge_ratio, index=data.index)
            
            # Calculate z-score
            zscore = self.spread_calculator.calculate_zscore(spread)
            
            # Store spread and z-score
            pair_info['spread'] = spread
            pair_info['zscore'] = zscore
        
        return {pair_id: pair_info['spread'] for pair_id, pair_info in self.pair_data.items() if 'spread' in pair_info}
    
    def generate_signals(self):
        """
        Generate trading signals for all pairs.
        
        Returns:
        --------
        dict
            Dictionary of pair_id -> signals
        """
        self.active_signals = {}
        
        for pair_id, pair_info in self.pair_data.items():
            if not pair_info['is_active'] or 'zscore' not in pair_info:
                continue
            
            # Generate signals from z-score
            try:
                signals = self.signal_generator.generate_signals(
                    pair_info['zscore'],
                    volume_data=None,  # Could use volume if available
                    time_data=pair_info['data']
                )
                
                # Apply holding period constraints
                signals = self.signal_generator.apply_holding_period(
                    signals,
                    frequency='1hour'  # Adjust based on the data timeframe
                )
                
                # Apply stop loss
                signals = self.signal_generator.apply_stop_loss(
                    signals,
                    pair_info['zscore'],
                    stop_threshold=3.0
                )
                
                # Apply trailing exit
                signals = self.signal_generator.apply_trailing_exit(
                    signals,
                    pair_info['zscore'],
                    trail_percent=0.5
                )
                
                # Store signals
                self.active_signals[pair_id] = signals
            except Exception as e:
                print(f"Error generating signals for pair {pair_id}: {e}")
        
        return self.active_signals
    
    def backtest_strategy(self, commission=2.0, slippage=1.0):
        """
        Backtest the strategy with the current configuration.
        
        Parameters:
        -----------
        commission : float
            Commission per trade (per side)
        slippage : float
            Slippage per trade in price points
            
        Returns:
        --------
        dict
            Dictionary with backtest results
        """
        # Ensure we have signals
        if not self.active_signals:
            print("No active signals. Run generate_signals() first.")
            return None
        
        # Reset performance tracking
        self.performance_metrics = {}
        self.trade_history = []
        
        # Set up tracking variables
        equity_curve = {}
        daily_returns = {}
        positions = {}
        
        # Process each pair
        for pair_id, signals in self.active_signals.items():
            if pair_id not in self.pair_data or 'spread' not in self.pair_data[pair_id]:
                continue
            
            pair_info = self.pair_data[pair_id]
            spread = pair_info['spread']
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            
            # Initialize equity curve for this pair
            equity_curve[pair_id] = pd.Series(0.0, index=signals.index)
            positions[pair_id] = pd.Series(0, index=signals.index)
            
            # Trade P&L tracking
            trade_entry_price = None
            trade_entry_time = None
            trade_direction = 0
            trade_size = 0
            
            # Size for this pair
            initial_volatility = spread.rolling(window=20).std().iloc[20] if len(spread) > 20 else spread.std()
            position_size, _ = self.position_sizer.calculate_position_size(spread)
            
            # Process signals chronologically
            for i in range(1, len(signals)):
                current_time = signals.index[i]
                current_spread = spread.loc[current_time]
                current_position = positions[pair_id].iloc[i-1]
                
                # Check for entry signals
                if current_position == 0:
                    # Long entry (spread is undervalued)
                    if signals.loc[current_time, 'entry_long'] == 1:
                        # Calculate position size based on current volatility
                        position_size, _ = self.position_sizer.calculate_position_size(
                            spread.iloc[:i]
                        )
                        
                        # Record entry
                        trade_entry_price = current_spread
                        trade_entry_time = current_time
                        trade_direction = 1  # Long
                        trade_size = position_size
                        positions[pair_id].iloc[i] = 1
                        
                        # Apply transaction costs
                        equity_curve[pair_id].iloc[i] = equity_curve[pair_id].iloc[i-1] - commission
                    
                    # Short entry (spread is overvalued)
                    elif signals.loc[current_time, 'entry_short'] == 1:
                        # Calculate position size based on current volatility
                        position_size, _ = self.position_sizer.calculate_position_size(
                            spread.iloc[:i]
                        )
                        
                        # Record entry
                        trade_entry_price = current_spread
                        trade_entry_time = current_time
                        trade_direction = -1  # Short
                        trade_size = position_size
                        positions[pair_id].iloc[i] = -1
                        
                        # Apply transaction costs
                        equity_curve[pair_id].iloc[i] = equity_curve[pair_id].iloc[i-1] - commission
                    
                    else:
                        # No position
                        positions[pair_id].iloc[i] = 0
                        equity_curve[pair_id].iloc[i] = equity_curve[pair_id].iloc[i-1]
                
                # Check for exit signals
                elif current_position == 1:  # Long position
                    # Update P&L
                    price_change = current_spread - trade_entry_price
                    equity_curve[pair_id].iloc[i] = equity_curve[pair_id].iloc[i-1] + price_change * trade_size
                    
                    # Exit signal
                    if signals.loc[current_time, 'exit_long'] == 1:
                        # Record trade
                        self._record_trade(
                            pair_id=pair_id,
                            entry_time=trade_entry_time,
                            exit_time=current_time,
                            entry_price=trade_entry_price,
                            exit_price=current_spread,
                            direction=trade_direction,
                            size=trade_size
                        )
                        
                        # Apply transaction costs
                        equity_curve[pair_id].iloc[i] -= commission
                        
                        # Reset position
                        positions[pair_id].iloc[i] = 0
                        trade_entry_price = None
                        trade_entry_time = None
                        trade_direction = 0
                    else:
                        # Continue holding
                        positions[pair_id].iloc[i] = 1
                
                elif current_position == -1:  # Short position
                    # Update P&L
                    price_change = trade_entry_price - current_spread
                    equity_curve[pair_id].iloc[i] = equity_curve[pair_id].iloc[i-1] + price_change * trade_size
                    
                    # Exit signal
                    if signals.loc[current_time, 'exit_short'] == 1:
                        # Record trade
                        self._record_trade(
                            pair_id=pair_id,
                            entry_time=trade_entry_time,
                            exit_time=current_time,
                            entry_price=trade_entry_price,
                            exit_price=current_spread,
                            direction=trade_direction,
                            size=trade_size
                        )
                        
                        # Apply transaction costs
                        equity_curve[pair_id].iloc[i] -= commission
                        
                        # Reset position
                        positions[pair_id].iloc[i] = 0
                        trade_entry_price = None
                        trade_entry_time = None
                        trade_direction = 0
                    else:
                        # Continue holding
                        positions[pair_id].iloc[i] = -1
        
        # Combine equity curves
        total_equity = pd.Series(0.0, index=list(equity_curve.values())[0].index)
        for pair_equity in equity_curve.values():
            total_equity = total_equity.add(pair_equity, fill_value=0)
        
        # Calculate daily returns
        daily_equity = total_equity.resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(
            total_equity, daily_returns
        )
        
        # Add detailed data to results
        results = {
            'equity_curve': total_equity,
            'daily_returns': daily_returns,
            'positions': positions,
            'metrics': self.performance_metrics,
            'trade_history': self.trade_history
        }
        
        return results
    
    def _record_trade(self, pair_id, entry_time, exit_time, entry_price, exit_price, direction, size):
        """Record a completed trade in the trade history."""
        if direction == 1:  # Long
            pnl = (exit_price - entry_price) * size
        else:  # Short
            pnl = (entry_price - exit_price) * size
        
        # Calculate holding period in minutes
        holding_minutes = (exit_time - entry_time).total_seconds() / 60
        
        trade = {
            'pair_id': pair_id,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,
            'size': size,
            'pnl': pnl,
            'holding_period_minutes': holding_minutes
        }
        
        self.trade_history.append(trade)
    
    def _calculate_performance_metrics(self, equity_curve, daily_returns):
        """Calculate performance metrics from equity curve and returns."""
        if len(daily_returns) == 0:
            return {}
        
        # Extract trade statistics
        if self.trade_history:
            trade_pnls = [trade['pnl'] for trade in self.trade_history]
            win_trades = [pnl for pnl in trade_pnls if pnl > 0]
            loss_trades = [pnl for pnl in trade_pnls if pnl <= 0]
            
            win_rate = len(win_trades) / len(trade_pnls) if trade_pnls else 0
            avg_win = np.mean(win_trades) if win_trades else 0
            avg_loss = np.mean(loss_trades) if loss_trades else 0
            profit_factor = sum(win_trades) / abs(sum(loss_trades)) if loss_trades and sum(loss_trades) != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate returns and drawdowns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if equity_curve.iloc[0] != 0 else 0
        annual_return = total_return * (252 / len(daily_returns))
        daily_std = daily_returns.std()
        annual_volatility = daily_std * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        is_drawdown = equity_curve < rolling_max
        drawdown_ends = (~is_drawdown).astype(int).diff()
        drawdown_starts = is_drawdown.astype(int).diff()
        
        # Store metrics
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history),
            'winning_trades': len(win_trades),
            'losing_trades': len(loss_trades),
            'average_win': avg_win,
            'average_loss': avg_loss
        }
        
        return metrics
    
    def plot_results(self, results, save_path=None):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        results : dict
            Results from backtest_strategy
        save_path : str, optional
            Path to save the plot
        """
        if not results or 'equity_curve' not in results:
            print("No results to plot.")
            return
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        equity_curve = results['equity_curve']
        axes[0].plot(equity_curve.index, equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True)
        
        # Plot drawdowns
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100  # as percentage
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True)
        
        # Plot daily returns
        daily_returns = results['daily_returns'] * 100  # as percentage
        axes[2].bar(daily_returns.index, daily_returns.values, color='blue', alpha=0.6)
        axes[2].set_title('Daily Returns (%)')
        axes[2].set_ylabel('Returns (%)')
        axes[2].grid(True)
        
        # Add text with performance metrics
        metrics = results['metrics']
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Annual Volatility: {metrics['annual_volatility']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Total Trades: {metrics['total_trades']}"
        )
        
        # Add a text box with metrics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(0.02, 0.05, metrics_text, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='bottom', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        return fig, axes
    
    def _get_default_futures(self):
        """Get a list of default futures to use for pair finding."""
        default_futures = [
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
        ]
        
        # Filter to include only available futures
        available_futures = []
        for ticker in default_futures:
            path = Path(f"data/historical/{ticker}/1day/data.csv")
            if path.exists():
                available_futures.append(ticker)
        
        return available_futures


def main():
    """Run a sample backtest of the pairs trading strategy."""
    # Initialize strategy
    strategy = PairsTradingStrategy(
        account_size=100000,
        max_risk_per_trade=0.01,
        max_allocation=0.15,
        entry_threshold=2.0,
        exit_threshold=0.5,
        max_holding_period=180,  # 3 hours in minutes
        daily_loss_limit=0.01
    )
    
    # Find pairs
    pairs_results = strategy.find_pairs(start_date='2018-01-01')
    
    if pairs_results.empty:
        print("No suitable pairs found.")
        return
    
    print(f"Found {len(pairs_results)} suitable pairs:")
    print(pairs_results[['ticker1', 'ticker2', 'correlation', 'half_life_train', 'pair_score']].head())
    
    # Take top 3 pairs for testing
    top_pairs = []
    for i in range(min(3, len(pairs_results))):
        row = pairs_results.iloc[i]
        top_pairs.append((row['ticker1'], row['ticker2'], row['hedge_ratio']))
    
    # Initialize with historical data
    strategy.initialize_strategy(
        pairs=top_pairs,
        start_date='2020-01-01',
        end_date='2022-12-31',
        timeframe='1hour'
    )
    
    # Calculate spreads
    spreads = strategy.calculate_spreads(use_kalman=True)
    
    # Generate signals
    signals = strategy.generate_signals()
    
    # Run backtest
    results = strategy.backtest_strategy(commission=2.0, slippage=1.0)
    
    if results:
        # Print performance metrics
        print("\nBacktest Results:")
        for key, value in results['metrics'].items():
            print(f"{key}: {value}")
        
        # Plot results
        fig, axes = strategy.plot_results(results, save_path='output/backtest_results.png')
        print("\nPlot saved to output/backtest_results.png")
        plt.show()
    else:
        print("Backtest failed.")

if __name__ == "__main__":
    main() 