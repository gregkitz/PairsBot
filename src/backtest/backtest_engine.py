import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json


class BacktestEngine:
    """
    Backtesting engine for pairs trading strategies.
    
    This class provides functionality to simulate pairs trading strategies
    using historical data and calculate performance metrics.
    """
    
    def __init__(self, signals=None, prices=None, position_sizes=None, 
                 account_size=100000, commission=0.0, slippage=0.0,
                 trade_delay=0, allow_simultaneous_positions=True,
                 pairs_data=None):
        """
        Initialize the BacktestEngine.
        
        Parameters:
        -----------
        signals : pandas.DataFrame, optional
            DataFrame containing trading signals
        prices : dict, optional
            Dictionary of price data for each instrument
        position_sizes : pandas.DataFrame, optional
            DataFrame containing position sizes
        account_size : float
            Initial account size
        commission : float
            Commission per trade as percentage
        slippage : float
            Slippage per trade as percentage
        trade_delay : int
            Delay in bars between signal and execution
        allow_simultaneous_positions : bool
            Whether to allow multiple positions at the same time
        pairs_data : dict, optional
            Dictionary containing pairs trading data
        """
        self.signals = signals
        self.prices = prices
        self.position_sizes = position_sizes
        self.initial_account_size = account_size
        self.commission = commission
        self.slippage = slippage
        self.trade_delay = trade_delay
        self.allow_simultaneous_positions = allow_simultaneous_positions
        self.pairs_data = pairs_data
        
        # Results
        self.equity_curve = None
        self.trade_history = []
        self.metrics = None
        self.daily_returns = None
    
    def run_backtest(self, signals=None, prices=None, position_sizes=None, pairs_data=None):
        """
        Run backtest for the pairs trading strategy.
        
        Parameters:
        -----------
        signals : pandas.DataFrame, optional
            DataFrame containing trading signals
        prices : dict, optional
            Dictionary of price data for each instrument
        position_sizes : pandas.DataFrame, optional
            DataFrame containing position sizes
        pairs_data : dict, optional
            Dictionary containing pairs trading data
            
        Returns:
        --------
        dict
            Dictionary containing backtest results
        """
        # Update with provided data if available
        self.signals = signals if signals is not None else self.signals
        self.prices = prices if prices is not None else self.prices
        self.position_sizes = position_sizes if position_sizes is not None else self.position_sizes
        self.pairs_data = pairs_data if pairs_data is not None else self.pairs_data
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize backtest
        self._initialize_backtest()
        
        # Process each trading day
        self._process_trading_days()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Prepare results
        results = {
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'trade_history': self.trade_history,
            'metrics': self.metrics
        }
        
        return results
    
    def _validate_inputs(self):
        """Validate input data."""
        if self.signals is None:
            raise ValueError("Signals data is required")
        
        if self.prices is None and self.pairs_data is None:
            raise ValueError("Either price data or pairs data is required")
        
        # Check if pairs_data has necessary components
        if self.pairs_data is not None:
            required_keys = ['ticker1', 'ticker2', 'hedge_ratio']
            if not all(key in self.pairs_data for key in required_keys):
                raise ValueError(f"pairs_data must contain {required_keys}")
    
    def _initialize_backtest(self):
        """Initialize backtest variables."""
        if self.pairs_data is not None:
            # Extract prices if provided in pairs_data
            ticker1 = self.pairs_data['ticker1']
            ticker2 = self.pairs_data['ticker2']
            
            if self.prices is None:
                self.prices = {}
                
            # Check if prices need to be added
            if 'ticker1_prices' in self.pairs_data and ticker1 not in self.prices:
                self.prices[ticker1] = self.pairs_data['ticker1_prices']
            
            if 'ticker2_prices' in self.pairs_data and ticker2 not in self.prices:
                self.prices[ticker2] = self.pairs_data['ticker2_prices']
            
            # Create ticker pairs for easier reference
            self.ticker_pairs = [(ticker1, ticker2, self.pairs_data['hedge_ratio'])]
        else:
            # Determine ticker pairs from the available price data
            tickers = list(self.prices.keys())
            
            if len(tickers) < 2:
                raise ValueError("At least two tickers are required in the price data")
            
            # Default to simple pairs with 1:1 ratio if not specified
            self.ticker_pairs = [(tickers[0], tickers[1], 1.0)]
        
        # Initialize equity curve
        self.equity_curve = pd.Series(
            index=self.signals.index,
            data=self.initial_account_size
        )
        
        # Initialize daily returns
        self.daily_returns = pd.Series(index=self.signals.index, dtype=float)
        self.daily_returns.iloc[0] = 0
        
        # Initialize positions
        self.current_positions = {}
        for ticker in self.prices:
            self.current_positions[ticker] = 0
    
    def _process_trading_days(self):
        """Process each trading day in the backtest."""
        # Extract the position column if it exists
        position_col = 'position' if 'position' in self.signals.columns else None
        
        # Initialize variables for tracking
        equity = self.initial_account_size
        
        for i in range(1, len(self.signals)):
            current_date = self.signals.index[i]
            prev_date = self.signals.index[i-1]
            
            # Check for trades
            if position_col is not None:
                # Simple mode using position column
                self._process_position_trade(i, position_col)
            else:
                # Detailed mode using entry/exit signals
                self._process_signal_trade(i)
            
            # Update equity based on current positions
            new_equity = self._calculate_current_equity(current_date)
            self.equity_curve.iloc[i] = new_equity
            
            # Calculate daily return
            self.daily_returns.iloc[i] = (new_equity / equity) - 1
            equity = new_equity
    
    def _process_position_trade(self, index, position_col):
        """
        Process trades based on position column.
        
        Parameters:
        -----------
        index : int
            Current index in the signals DataFrame
        position_col : str
            Name of the position column
        """
        current_position = self.signals[position_col].iloc[index]
        previous_position = self.signals[position_col].iloc[index-1]
        
        if current_position != previous_position:
            # Position has changed, execute trade
            current_date = self.signals.index[index]
            ticker1, ticker2, hedge_ratio = self.ticker_pairs[0]
            
            if previous_position == 0 and current_position == 1:
                # Long entry
                self._execute_pair_trade('LONG', ticker1, ticker2, hedge_ratio, index)
            
            elif previous_position == 0 and current_position == -1:
                # Short entry
                self._execute_pair_trade('SHORT', ticker1, ticker2, hedge_ratio, index)
            
            elif previous_position == 1 and current_position == 0:
                # Long exit
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)
            
            elif previous_position == -1 and current_position == 0:
                # Short exit
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)
            
            elif previous_position == 1 and current_position == -1:
                # Flip from long to short
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)  # Close long
                self._execute_pair_trade('SHORT', ticker1, ticker2, hedge_ratio, index)  # Open short
            
            elif previous_position == -1 and current_position == 1:
                # Flip from short to long
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)  # Close short
                self._execute_pair_trade('LONG', ticker1, ticker2, hedge_ratio, index)  # Open long
    
    def _process_signal_trade(self, index):
        """
        Process trades based on entry/exit signals.
        
        Parameters:
        -----------
        index : int
            Current index in the signals DataFrame
        """
        signals = self.signals.iloc[index]
        
        # Process each pair
        for ticker1, ticker2, hedge_ratio in self.ticker_pairs:
            # Check for long entry
            if ('entry_long' in signals and signals.entry_long == 1 and 
                    (ticker1, ticker2) not in self.current_positions):
                self._execute_pair_trade('LONG', ticker1, ticker2, hedge_ratio, index)
            
            # Check for short entry
            if ('entry_short' in signals and signals.entry_short == 1 and 
                    (ticker1, ticker2) not in self.current_positions):
                self._execute_pair_trade('SHORT', ticker1, ticker2, hedge_ratio, index)
            
            # Check for long exit
            if ('exit_long' in signals and signals.exit_long == 1 and 
                    (ticker1, ticker2) in self.current_positions and 
                    self.current_positions[(ticker1, ticker2)] > 0):
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)
            
            # Check for short exit
            if ('exit_short' in signals and signals.exit_short == 1 and 
                    (ticker1, ticker2) in self.current_positions and 
                    self.current_positions[(ticker1, ticker2)] < 0):
                self._execute_pair_trade('FLAT', ticker1, ticker2, hedge_ratio, index)
    
    def _execute_pair_trade(self, action, ticker1, ticker2, hedge_ratio, index):
        """
        Execute a pairs trade.
        
        Parameters:
        -----------
        action : str
            Trade action: 'LONG', 'SHORT', or 'FLAT'
        ticker1 : str
            First ticker symbol
        ticker2 : str
            Second ticker symbol
        hedge_ratio : float
            Hedge ratio for the pair
        index : int
            Current index in the signals DataFrame
        """
        # Get current date and prices
        current_date = self.signals.index[index]
        price1 = self._get_price(ticker1, current_date)
        price2 = self._get_price(ticker2, current_date)
        
        # Determine position size
        if self.position_sizes is not None and index < len(self.position_sizes):
            position_size = self.position_sizes.iloc[index]
        else:
            # Default to 50% of account for ticker1
            position_size = self.equity_curve.iloc[index] * 0.5 / price1
        
        # Determine trade details based on action
        if action == 'LONG':
            # Buy ticker1, sell ticker2
            qty1 = position_size
            qty2 = -position_size * hedge_ratio
            
            # Record entry in current positions
            self.current_positions[(ticker1, ticker2)] = 1
            
            # Record trade
            trade = {
                'entry_date': current_date,
                'ticker1': ticker1,
                'ticker2': ticker2,
                'action': 'LONG',
                'qty1': qty1,
                'qty2': qty2,
                'price1_entry': price1,
                'price2_entry': price2,
                'hedge_ratio': hedge_ratio,
                'status': 'OPEN',
                'exit_date': None,
                'price1_exit': None,
                'price2_exit': None,
                'pnl': 0,
                'trade_id': len(self.trade_history)
            }
            
            self.trade_history.append(trade)
        
        elif action == 'SHORT':
            # Sell ticker1, buy ticker2
            qty1 = -position_size
            qty2 = position_size * hedge_ratio
            
            # Record entry in current positions
            self.current_positions[(ticker1, ticker2)] = -1
            
            # Record trade
            trade = {
                'entry_date': current_date,
                'ticker1': ticker1,
                'ticker2': ticker2,
                'action': 'SHORT',
                'qty1': qty1,
                'qty2': qty2,
                'price1_entry': price1,
                'price2_entry': price2,
                'hedge_ratio': hedge_ratio,
                'status': 'OPEN',
                'exit_date': None,
                'price1_exit': None,
                'price2_exit': None,
                'pnl': 0,
                'trade_id': len(self.trade_history)
            }
            
            self.trade_history.append(trade)
        
        elif action == 'FLAT':
            # Find open trade for this pair
            for i, trade in enumerate(self.trade_history):
                if (trade['ticker1'] == ticker1 and trade['ticker2'] == ticker2 and 
                        trade['status'] == 'OPEN'):
                    # Close the trade
                    self.trade_history[i]['exit_date'] = current_date
                    self.trade_history[i]['price1_exit'] = price1
                    self.trade_history[i]['price2_exit'] = price2
                    self.trade_history[i]['status'] = 'CLOSED'
                    
                    # Calculate PnL
                    pnl = 0
                    
                    if trade['action'] == 'LONG':
                        # PnL = (exit_price1 - entry_price1) * qty1 + (entry_price2 - exit_price2) * qty2
                        pnl = ((price1 - trade['price1_entry']) * trade['qty1'] + 
                              (trade['price2_entry'] - price2) * trade['qty2'])
                    else:  # 'SHORT'
                        # PnL = (entry_price1 - exit_price1) * qty1 + (exit_price2 - entry_price2) * qty2
                        pnl = ((trade['price1_entry'] - price1) * abs(trade['qty1']) + 
                              (price2 - trade['price2_entry']) * abs(trade['qty2']))
                    
                    # Subtract commissions and slippage
                    commission_cost = (abs(trade['qty1']) * price1 + abs(trade['qty2']) * price2) * self.commission
                    slippage_cost = (abs(trade['qty1']) * price1 + abs(trade['qty2']) * price2) * self.slippage
                    pnl -= (commission_cost + slippage_cost)
                    
                    self.trade_history[i]['pnl'] = pnl
                    
                    # Remove from current positions
                    if (ticker1, ticker2) in self.current_positions:
                        del self.current_positions[(ticker1, ticker2)]
                    
                    break
    
    def _calculate_current_equity(self, date):
        """
        Calculate current equity based on cash and open positions.
        
        Parameters:
        -----------
        date : datetime
            Current date
            
        Returns:
        --------
        float
            Current equity
        """
        # Start with initial account size
        equity = self.initial_account_size
        
        # Add PnL from closed trades
        for trade in self.trade_history:
            if trade['status'] == 'CLOSED':
                equity += trade['pnl']
            elif trade['status'] == 'OPEN':
                # Calculate unrealized PnL for open trades
                ticker1 = trade['ticker1']
                ticker2 = trade['ticker2']
                price1 = self._get_price(ticker1, date)
                price2 = self._get_price(ticker2, date)
                
                if trade['action'] == 'LONG':
                    # Unrealized PnL = (current_price1 - entry_price1) * qty1 + (entry_price2 - current_price2) * qty2
                    unrealized_pnl = ((price1 - trade['price1_entry']) * trade['qty1'] + 
                                     (trade['price2_entry'] - price2) * trade['qty2'])
                else:  # 'SHORT'
                    # Unrealized PnL = (entry_price1 - current_price1) * qty1 + (current_price2 - entry_price2) * qty2
                    unrealized_pnl = ((trade['price1_entry'] - price1) * abs(trade['qty1']) + 
                                     (price2 - trade['price2_entry']) * abs(trade['qty2']))
                
                # Subtract commissions for eventual closing (estimate)
                commission_cost = (abs(trade['qty1']) * price1 + abs(trade['qty2']) * price2) * self.commission
                unrealized_pnl -= commission_cost
                
                equity += unrealized_pnl
        
        return equity
    
    def _get_price(self, ticker, date):
        """
        Get price for a ticker at a specific date.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        date : datetime
            Date for which to get the price
            
        Returns:
        --------
        float
            Price for the ticker at the specified date
        """
        if ticker not in self.prices:
            raise ValueError(f"No price data found for ticker {ticker}")
        
        # Get price series for the ticker
        price_series = self.prices[ticker]
        
        # Check if date exists in the price series
        if date not in price_series.index:
            # Find nearest date
            nearest_date = self._find_nearest_date(price_series.index, date)
            return price_series.loc[nearest_date]
        
        return price_series.loc[date]
    
    def _find_nearest_date(self, date_index, target_date):
        """
        Find nearest date in index.
        
        Parameters:
        -----------
        date_index : pandas.DatetimeIndex
            Index of dates
        target_date : datetime
            Target date to find
            
        Returns:
        --------
        datetime
            Nearest date in the index
        """
        # Convert to numpy array for faster search
        dates = date_index.to_numpy()
        target = np.datetime64(target_date)
        
        # Find index of nearest date
        nearest_idx = np.abs(dates - target).argmin()
        
        return date_index[nearest_idx]
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        # Calculate returns
        returns = self.daily_returns
        
        # Total return
        total_return = (self.equity_curve.iloc[-1] / self.initial_account_size) - 1
        
        # Calculate annualized return
        days = (self.signals.index[-1] - self.signals.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate volatility
        daily_std = returns.std()
        annual_volatility = daily_std * np.sqrt(252)  # Assuming 252 trading days in a year
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.0  # Simplified, can be adjusted
        sharpe_ratio = ((annual_return - risk_free_rate) / annual_volatility 
                        if annual_volatility > 0 else 0)
        
        # Calculate maximum drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Calculate trade statistics
        if self.trade_history:
            winning_trades = sum(1 for trade in self.trade_history if trade['status'] == 'CLOSED' and trade['pnl'] > 0)
            losing_trades = sum(1 for trade in self.trade_history if trade['status'] == 'CLOSED' and trade['pnl'] <= 0)
            total_trades = winning_trades + losing_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['status'] == 'CLOSED' and trade['pnl'] > 0)
            gross_loss = sum(trade['pnl'] for trade in self.trade_history if trade['status'] == 'CLOSED' and trade['pnl'] <= 0)
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
            
            # Calculate average win and loss
            average_win = gross_profit / winning_trades if winning_trades > 0 else 0
            average_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        else:
            winning_trades = 0
            losing_trades = 0
            total_trades = 0
            win_rate = 0
            profit_factor = 0
            average_win = 0
            average_loss = 0
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss
        }
    
    def plot_results(self, figsize=(15, 10), save_path=None):
        """
        Plot backtest results.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if self.equity_curve is None:
            raise ValueError("Run backtest first to generate results")
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot equity curve
        axes[0].plot(self.equity_curve.index, self.equity_curve.values)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity')
        axes[0].grid(True)
        
        # Calculate and plot drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max * 100  # as percentage
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True)
        
        # Plot daily returns
        axes[2].bar(self.daily_returns.index, self.daily_returns.values * 100)  # as percentage
        axes[2].set_title('Daily Returns (%)')
        axes[2].set_ylabel('Return %')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def save_results(self, file_path):
        """
        Save backtest results to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the results
        """
        if self.metrics is None:
            raise ValueError("Run backtest first to generate results")
        
        # Prepare results
        results = {
            'metrics': self.metrics,
            'trade_history': self.trade_history
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)  # default=str handles datetime
        
        print(f"Results saved to {file_path}")
    
    def generate_trade_statistics(self):
        """
        Generate detailed trade statistics.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trade statistics
        """
        if not self.trade_history:
            raise ValueError("No trades found in the backtest")
        
        # Convert trade history to DataFrame
        trade_df = pd.DataFrame(self.trade_history)
        
        # Filter closed trades
        closed_trades = trade_df[trade_df['status'] == 'CLOSED']
        
        if closed_trades.empty:
            raise ValueError("No closed trades found in the backtest")
        
        # Calculate trade duration
        closed_trades['duration'] = (pd.to_datetime(closed_trades['exit_date']) - 
                                    pd.to_datetime(closed_trades['entry_date']))
        
        # Calculate returns
        closed_trades['return'] = closed_trades['pnl'] / self.initial_account_size
        
        # Group by action (LONG/SHORT)
        grouped = closed_trades.groupby('action')
        
        # Calculate statistics
        stats = {
            'count': grouped.size(),
            'win_rate': grouped['pnl'].apply(lambda x: (x > 0).mean()),
            'avg_profit': grouped['pnl'].mean(),
            'avg_win': grouped['pnl'].apply(lambda x: x[x > 0].mean()),
            'avg_loss': grouped['pnl'].apply(lambda x: x[x <= 0].mean()),
            'profit_factor': grouped['pnl'].apply(lambda x: abs(x[x > 0].sum() / x[x <= 0].sum()) 
                                               if (x <= 0).any() else float('inf')),
            'max_profit': grouped['pnl'].max(),
            'max_loss': grouped['pnl'].min(),
            'avg_duration': grouped['duration'].mean()
        }
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        
        # Add overall statistics
        overall = pd.Series({
            'count': closed_trades.shape[0],
            'win_rate': (closed_trades['pnl'] > 0).mean(),
            'avg_profit': closed_trades['pnl'].mean(),
            'avg_win': closed_trades.loc[closed_trades['pnl'] > 0, 'pnl'].mean(),
            'avg_loss': closed_trades.loc[closed_trades['pnl'] <= 0, 'pnl'].mean(),
            'profit_factor': abs(closed_trades.loc[closed_trades['pnl'] > 0, 'pnl'].sum() / 
                              closed_trades.loc[closed_trades['pnl'] <= 0, 'pnl'].sum()) 
                              if (closed_trades['pnl'] <= 0).any() else float('inf'),
            'max_profit': closed_trades['pnl'].max(),
            'max_loss': closed_trades['pnl'].min(),
            'avg_duration': closed_trades['duration'].mean()
        }, name='OVERALL')
        
        # Append overall statistics
        # pd.DataFrame.append was removed in pandas 2.0; use concat instead
        stats_df = pd.concat([stats_df, overall.to_frame().T])
        
        return stats_df 