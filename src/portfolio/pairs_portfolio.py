"""
Pairs Trading Portfolio Module

This module provides functionality for managing a portfolio of pairs trades.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PairsPortfolio:
    """
    Manage a portfolio of pairs trades.
    """
    
    def __init__(self, account_size=1000000, max_allocation_per_pair=0.2,
                 max_correlation=0.5, target_volatility=0.1):
        """
        Initialize the PairsPortfolio with parameters.
        
        Parameters:
        -----------
        account_size : float
            Initial account size
        max_allocation_per_pair : float
            Maximum allocation per pair as a percentage of account size
        max_correlation : float
            Maximum allowed correlation between pairs
        target_volatility : float
            Target portfolio volatility
        """
        self.account_size = account_size
        self.max_allocation_per_pair = max_allocation_per_pair
        self.max_correlation = max_correlation
        self.target_volatility = target_volatility
        
        # Store portfolio constituents
        self.pairs = []
        self.signals = {}
        self.returns = {}
        self.weights = {}
        self.allocation = {}
        
        # Performance tracking
        self.portfolio_equity = None
        self.portfolio_returns = None
        
    def add_pair(self, symbol1, symbol2, signals, config=None):
        """
        Add a pair to the portfolio.
        
        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        signals : pd.DataFrame
            DataFrame with trading signals
        config : dict, optional
            Configuration for the pair
            
        Returns:
        --------
        bool
            True if pair was added successfully, False otherwise
        """
        pair_id = f"{symbol1}_{symbol2}"
        
        # Check if pair already exists
        if pair_id in [p['pair_id'] for p in self.pairs]:
            logger.warning(f"Pair {pair_id} already exists in portfolio")
            return False
        
        # Calculate returns if not in signals
        if 'returns' not in signals.columns:
            # Assume positions are already in signals
            signals['returns'] = signals['position'].shift(1) * signals['z_score'].diff()
        
        # Store pair information
        pair_info = {
            'pair_id': pair_id,
            'symbol1': symbol1,
            'symbol2': symbol2,
            'config': config or {},
            'start_date': signals.index[0],
            'end_date': signals.index[-1]
        }
        
        self.pairs.append(pair_info)
        self.signals[pair_id] = signals
        self.returns[pair_id] = signals['returns']
        
        logger.info(f"Added pair {pair_id} to portfolio")
        return True
    
    def calculate_correlation_matrix(self):
        """
        Calculate the correlation matrix between pair returns.
        
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        # Get returns for each pair
        returns_df = pd.DataFrame()
        
        for pair_id, returns_series in self.returns.items():
            returns_df[pair_id] = returns_series
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    def filter_pairs_by_correlation(self):
        """
        Filter pairs based on correlation.
        
        Returns:
        --------
        list
            List of pairs that meet the correlation criteria
        """
        if len(self.pairs) <= 1:
            return self.pairs
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix()
        
        # Start with an empty set of selected pairs
        selected_pairs = []
        remaining_pairs = self.pairs.copy()
        
        # Helper function to check if a pair can be added
        def can_add_pair(pair_id, selected_ids):
            if not selected_ids:
                return True
            
            # Check correlation with already selected pairs
            for selected_id in selected_ids:
                if abs(corr_matrix.loc[pair_id, selected_id]) > self.max_correlation:
                    return False
            
            return True
        
        # Sort pairs by Sharpe ratio (if available)
        sorted_pairs = sorted(
            remaining_pairs, 
            key=lambda p: self.signals[p['pair_id']].get('stat_sharpe_ratio', 0).iloc[-1] 
                if 'stat_sharpe_ratio' in self.signals[p['pair_id']].columns 
                else 0,
            reverse=True
        )
        
        # Add pairs one by one, checking correlation
        for pair in sorted_pairs:
            pair_id = pair['pair_id']
            
            if can_add_pair(pair_id, [p['pair_id'] for p in selected_pairs]):
                selected_pairs.append(pair)
        
        logger.info(f"Selected {len(selected_pairs)} pairs after correlation filtering")
        return selected_pairs
    
    def calculate_weights(self, method='equal', selected_pairs=None):
        """
        Calculate portfolio weights for each pair.
        
        Parameters:
        -----------
        method : str
            Weight calculation method ('equal', 'volatility', 'sharpe')
        selected_pairs : list, optional
            List of pairs to consider. If None, use filtered pairs.
            
        Returns:
        --------
        dict
            Dictionary of weights by pair_id
        """
        # If no pairs selected, use filtered pairs
        if selected_pairs is None:
            selected_pairs = self.filter_pairs_by_correlation()
        
        # If no pairs remaining, return empty weights
        if not selected_pairs:
            return {}
        
        # Get pair IDs
        pair_ids = [p['pair_id'] for p in selected_pairs]
        
        if method == 'equal':
            # Equal weights
            weight = 1.0 / len(pair_ids)
            weights = {pair_id: weight for pair_id in pair_ids}
            
        elif method == 'volatility':
            # Inverse volatility weighting
            volatilities = {
                pair_id: self.returns[pair_id].std() for pair_id in pair_ids
            }
            
            # Calculate inverse volatility
            inv_vols = {pair_id: 1.0 / vol if vol > 0 else 0.0 for pair_id, vol in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            
            # Normalize to get weights
            if total_inv_vol > 0:
                weights = {pair_id: inv_vol / total_inv_vol for pair_id, inv_vol in inv_vols.items()}
            else:
                # Fall back to equal weighting
                weight = 1.0 / len(pair_ids)
                weights = {pair_id: weight for pair_id in pair_ids}
                
        elif method == 'sharpe':
            # Sharpe ratio weighting
            sharpes = {}
            
            for pair_id in pair_ids:
                signals = self.signals[pair_id]
                if 'stat_sharpe_ratio' in signals.columns:
                    sharpe = signals['stat_sharpe_ratio'].iloc[-1]
                else:
                    returns = self.returns[pair_id]
                    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
                
                # Only use positive Sharpe ratios
                sharpes[pair_id] = max(0.0, sharpe)
            
            # Calculate weights proportional to Sharpe ratio
            total_sharpe = sum(sharpes.values())
            
            if total_sharpe > 0:
                weights = {pair_id: sharpe / total_sharpe for pair_id, sharpe in sharpes.items()}
            else:
                # Fall back to equal weighting
                weight = 1.0 / len(pair_ids)
                weights = {pair_id: weight for pair_id in pair_ids}
        else:
            # Default to equal weighting
            weight = 1.0 / len(pair_ids)
            weights = {pair_id: weight for pair_id in pair_ids}
        
        # Store weights
        self.weights = weights
        
        logger.info(f"Calculated weights using {method} method")
        return weights
    
    def allocate_capital(self, weights=None, account_size=None):
        """
        Allocate capital to each pair based on weights.
        
        Parameters:
        -----------
        weights : dict, optional
            Dictionary of weights by pair_id. If None, use stored weights.
        account_size : float, optional
            Account size to allocate. If None, use stored account_size.
            
        Returns:
        --------
        dict
            Dictionary of allocation by pair_id
        """
        # Use stored values if not provided
        if weights is None:
            weights = self.weights
            
            if not weights:
                # Calculate weights if needed
                weights = self.calculate_weights()
        
        if account_size is None:
            account_size = self.account_size
        
        # Calculate allocation based on weights and max_allocation_per_pair
        allocation = {}
        
        for pair_id, weight in weights.items():
            # Cap at max allocation
            alloc = min(weight, self.max_allocation_per_pair) * account_size
            allocation[pair_id] = alloc
        
        # Store allocation
        self.allocation = allocation
        
        logger.info(f"Allocated capital to {len(allocation)} pairs")
        return allocation
    
    def simulate_portfolio(self, start_date=None, end_date=None):
        """
        Simulate portfolio performance over time.
        
        Parameters:
        -----------
        start_date : datetime or str, optional
            Start date for simulation. If None, use earliest common date.
        end_date : datetime or str, optional
            End date for simulation. If None, use latest common date.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio equity curve
        """
        # Ensure we have weights and allocation
        if not self.weights:
            self.calculate_weights()
            
        if not self.allocation:
            self.allocate_capital()
        
        # Get returns for each pair
        returns_df = pd.DataFrame()
        
        for pair_id, returns_series in self.returns.items():
            if pair_id in self.weights:
                # Clean returns to avoid extreme values that could cause numerical instability
                clean_returns = returns_series.copy()
                # Replace NaN, inf, -inf with 0
                clean_returns = clean_returns.replace([np.inf, -np.inf, np.nan], 0)
                # Clip extreme values
                clean_returns = clean_returns.clip(lower=-0.25, upper=0.25)
                returns_df[pair_id] = clean_returns
        
        # Determine date range
        if start_date is None:
            start_date = max(series.index[0] for series in self.returns.values() if not series.empty)
            
        if end_date is None:
            end_date = min(series.index[-1] for series in self.returns.values() if not series.empty)
        
        # Filter to specified date range
        returns_df = returns_df.loc[start_date:end_date]
        
        # Calculate weighted returns
        weighted_returns = pd.DataFrame()
        
        for pair_id, weight in self.weights.items():
            if pair_id in returns_df.columns:
                weighted_returns[pair_id] = returns_df[pair_id] * weight
        
        # Calculate portfolio returns
        portfolio_returns = weighted_returns.sum(axis=1)
        
        # Clean portfolio returns
        portfolio_returns = portfolio_returns.replace([np.inf, -np.inf, np.nan], 0)
        portfolio_returns = portfolio_returns.clip(lower=-0.25, upper=0.25)
        
        # Calculate equity curve - use a more stable calculation method
        portfolio_equity = (1 + portfolio_returns).cumprod() * self.account_size
        
        # Store results
        self.portfolio_returns = portfolio_returns
        self.portfolio_equity = portfolio_equity
        
        # Create output dataframe
        result_df = pd.DataFrame({
            'equity': portfolio_equity,
            'returns': portfolio_returns
        })
        
        logger.info(f"Simulated portfolio performance from {start_date} to {end_date}")
        return result_df
    
    def calculate_portfolio_metrics(self):
        """
        Calculate performance metrics for the portfolio.
        
        Returns:
        --------
        dict
            Dictionary of portfolio performance metrics
        """
        if self.portfolio_returns is None:
            logger.warning("No portfolio returns available. Run simulate_portfolio first.")
            return {}
        
        # Calculate performance metrics
        returns = self.portfolio_returns
        
        # Total return
        total_return = (self.portfolio_equity.iloc[-1] / self.account_size) - 1
        
        # Cap extreme values
        if np.isnan(total_return) or np.isinf(total_return):
            total_return = 0.0
        total_return = max(-0.99, min(10.0, total_return))
        
        # Annualized return
        n_years = max(len(returns) / 252, 0.1)  # Avoid division by zero
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Cap extreme values
        if np.isnan(annual_return) or np.isinf(annual_return):
            annual_return = 0.0
        annual_return = max(-0.99, min(10.0, annual_return))
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Cap extreme values
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0.0
        sharpe_ratio = max(-10.0, min(10.0, sharpe_ratio))
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        wins = returns[returns > 0].sum()
        losses = returns[returns < 0].sum()
        
        # Avoid division by zero
        if losses < 0:
            profit_factor = abs(wins / losses)
        else:
            profit_factor = 1.0 if wins > 0 else 0.0
        
        # Cap extreme values
        if np.isnan(profit_factor) or np.isinf(profit_factor):
            profit_factor = 10.0 if wins > 0 else 0.0
        profit_factor = min(profit_factor, 10.0)
        
        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
            
        # Cap extreme values
        if np.isnan(calmar_ratio) or np.isinf(calmar_ratio):
            calmar_ratio = 0.0
        calmar_ratio = min(calmar_ratio, 10.0)
        
        # Create metrics dictionary
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(returns)
        }
        
        return metrics
    
    def plot_portfolio_performance(self, save_path=None):
        """
        Plot portfolio performance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, display the plot.
            
        Returns:
        --------
        None
        """
        if self.portfolio_equity is None:
            logger.warning("No portfolio equity curve available. Run simulate_portfolio first.")
            return
        
        # Calculate metrics
        metrics = self.calculate_portfolio_metrics()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot equity curve
        ax.plot(self.portfolio_equity, label='Portfolio Equity')
        
        # Add metrics to plot
        title = "Portfolio Performance\n"
        title += f"Total Return: {metrics['total_return']:.2%}, "
        title += f"Annual Return: {metrics['annual_return']:.2%}, "
        title += f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
        title += f"Max DD: {metrics['max_drawdown']:.2%}"
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True)
        ax.legend()
        
        # Plot drawdown on lower axis
        returns = self.portfolio_returns
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        
        ax2 = ax.twinx()
        ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        
        # Save or display plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def save_portfolio(self, filepath):
        """
        Save portfolio configuration to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the portfolio configuration
            
        Returns:
        --------
        None
        """
        # Create portfolio config
        # Convert all pair info to serializable format
        serializable_pairs = []
        for pair in self.pairs:
            serializable_pair = pair.copy()
            # Convert timestamps to strings
            if 'start_date' in serializable_pair and hasattr(serializable_pair['start_date'], 'strftime'):
                serializable_pair['start_date'] = serializable_pair['start_date'].strftime('%Y-%m-%d %H:%M:%S')
            if 'end_date' in serializable_pair and hasattr(serializable_pair['end_date'], 'strftime'):
                serializable_pair['end_date'] = serializable_pair['end_date'].strftime('%Y-%m-%d %H:%M:%S')
            serializable_pairs.append(serializable_pair)
        
        # Get metrics and ensure they're serializable
        metrics = self.calculate_portfolio_metrics()
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_metrics[k] = float(v)
            else:
                serializable_metrics[k] = v
        
        config = {
            'account_size': float(self.account_size),
            'max_allocation_per_pair': float(self.max_allocation_per_pair),
            'max_correlation': float(self.max_correlation),
            'target_volatility': float(self.target_volatility),
            'pairs': serializable_pairs,
            'weights': {k: float(v) for k, v in self.weights.items()},
            'allocation': {k: float(v) for k, v in self.allocation.items()},
            'metrics': serializable_metrics
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved portfolio configuration to {filepath}")
    
    @classmethod
    def load_portfolio(cls, filepath, signals_dir=None):
        """
        Load portfolio configuration from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the portfolio configuration file
        signals_dir : str, optional
            Directory containing signal files
            
        Returns:
        --------
        PairsPortfolio
            Loaded portfolio object
        """
        # Load config from file
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Create portfolio object
        portfolio = cls(
            account_size=config.get('account_size', 1000000),
            max_allocation_per_pair=config.get('max_allocation_per_pair', 0.2),
            max_correlation=config.get('max_correlation', 0.5),
            target_volatility=config.get('target_volatility', 0.1)
        )
        
        # Load pairs
        if signals_dir:
            for pair_info in config.get('pairs', []):
                symbol1 = pair_info['symbol1']
                symbol2 = pair_info['symbol2']
                pair_id = pair_info['pair_id']
                
                # Load signals from file
                signals_file = os.path.join(signals_dir, f"{pair_id}_signals.csv")
                
                if os.path.exists(signals_file):
                    signals = pd.read_csv(signals_file)
                    signals['timestamp'] = pd.to_datetime(signals['timestamp'])
                    signals.set_index('timestamp', inplace=True)
                    
                    # Add pair to portfolio
                    portfolio.add_pair(symbol1, symbol2, signals, pair_info.get('config'))
                else:
                    logger.warning(f"Signals file not found: {signals_file}")
        
        # Set weights and allocation
        portfolio.weights = config.get('weights', {})
        portfolio.allocation = config.get('allocation', {})
        
        logger.info(f"Loaded portfolio from {filepath}")
        return portfolio 