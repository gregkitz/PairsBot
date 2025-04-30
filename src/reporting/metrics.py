"""
Performance Metrics for the Intraday Statistical Arbitrage System.

This module provides functions for calculating performance metrics
from backtest results, including returns, drawdowns, risk-adjusted metrics,
and trade statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from collections import defaultdict
import scipy.stats as stats


def calculate_performance_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    benchmark: pd.Series = None,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from backtest results.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Time series of portfolio equity values
    trades : pd.DataFrame, optional
        DataFrame containing individual trade information
    benchmark : pd.Series, optional
        Time series of benchmark values for comparison
    risk_free_rate : float
        Annualized risk-free rate for risk-adjusted metrics
    trading_days_per_year : int
        Number of trading days per year
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing calculated performance metrics
    """
    # Create metrics dictionary
    metrics = {}
    
    # Calculate basic metrics from equity curve
    if equity_curve is not None and len(equity_curve) > 0:
        metrics.update(_calculate_equity_metrics(
            equity_curve, 
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year
        ))
        
        # Calculate drawdown metrics
        metrics.update(_calculate_drawdown_metrics(equity_curve))
    
    # Calculate benchmark comparison metrics
    if benchmark is not None and len(benchmark) > 0:
        metrics.update(_calculate_benchmark_metrics(
            equity_curve, 
            benchmark, 
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year
        ))
    
    # Calculate trade-based metrics
    if trades is not None and len(trades) > 0:
        metrics.update(_calculate_trade_metrics(trades))
    
    # Add summary metrics
    metrics.update(_create_summary_metrics(metrics))
    
    return metrics


def _calculate_equity_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate metrics based on the equity curve.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Time series of portfolio equity values
    risk_free_rate : float
        Annualized risk-free rate
    trading_days_per_year : int
        Number of trading days per year
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of equity-based metrics
    """
    # Clean equity curve
    equity = equity_curve.dropna()
    
    # Cannot calculate metrics with less than 2 data points
    if len(equity) < 2:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0, 
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }
    
    # Calculate returns
    returns = equity.pct_change().dropna()
    log_returns = np.log(equity / equity.shift(1)).dropna()
    
    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    # Calculate trading days
    trading_days = (equity.index[-1] - equity.index[0]).days
    trading_days = max(1, trading_days)  # Avoid division by zero
    
    # Calculate annualized return
    years = trading_days / 365.25
    annualized_return = (1 + total_return) ** (1 / max(years, 1/365.25)) - 1
    
    # Calculate volatility (annualized standard deviation of returns)
    daily_volatility = float(returns.std())
    annualized_volatility = daily_volatility * np.sqrt(trading_days_per_year)
    
    # Calculate downside returns for Sortino ratio
    downside_returns = returns[returns < 0]
    downside_volatility = float(downside_returns.std()) if len(downside_returns) > 0 else 0
    annualized_downside_volatility = downside_volatility * np.sqrt(trading_days_per_year)
    
    # Calculate Sharpe ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_volatility if isinstance(annualized_volatility, (int, float)) and annualized_volatility > 0 else 0
    
    # Calculate Sortino ratio
    sortino_ratio = excess_return / annualized_downside_volatility if isinstance(annualized_downside_volatility, (int, float)) and annualized_downside_volatility > 0 else 0
    
    # Calculate daily and cumulative returns
    daily_returns = returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Calculate additional statistics
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    
    avg_daily_return = float(returns.mean())
    avg_positive_return = float(pos_returns.mean()) if len(pos_returns) > 0 else 0
    avg_negative_return = float(neg_returns.mean()) if len(neg_returns) > 0 else 0
    
    # Winning days statistics
    winning_days = len(pos_returns)
    losing_days = len(neg_returns)
    total_days = len(returns)
    
    win_rate = winning_days / total_days if total_days > 0 else 0
    # Calculate sums safely
    pos_sum = float(pos_returns.sum()) if not pos_returns.empty else 0
    neg_sum = float(neg_returns.sum()) if not neg_returns.empty else 0
    profit_factor = abs(pos_sum / neg_sum) if neg_sum != 0 else np.inf
    
    # Return all metrics
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'avg_daily_return': avg_daily_return,
        'avg_positive_return': avg_positive_return,
        'avg_negative_return': avg_negative_return,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'win_rate_days': win_rate,
        'profit_factor': profit_factor,
        'daily_returns': daily_returns,
        'cumulative_returns': cumulative_returns
    }


def _calculate_drawdown_metrics(equity_curve: pd.Series) -> Dict[str, float]:
    """
    Calculate drawdown related metrics.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Time series of portfolio equity values
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of drawdown metrics
    """
    # Clean equity curve
    equity = equity_curve.dropna()
    
    # Cannot calculate metrics with less than 2 data points
    if len(equity) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0,
            'avg_drawdown': 0.0,
            'avg_drawdown_duration': 0,
            'calmar_ratio': 0.0,
            'drawdowns': pd.Series(),
            'underwater': pd.Series()
        }
    
    # Calculate rolling maximum
    rolling_max = equity.cummax()
    
    # Calculate drawdown percentage
    drawdown = (equity / rolling_max) - 1
    
    # Calculate underwater (drawdown) periods
    underwater = drawdown.copy()
    
    # Find drawdown periods
    is_in_drawdown = underwater < 0
    
    # Identify individual drawdown periods
    # Identify individual drawdown periods safely
    # Identify individual drawdown periods safely
    shifted = is_in_drawdown.shift(1).fillna(False)
    drawdown_starts = (is_in_drawdown == True) & (shifted == False)
    drawdown_ends = (is_in_drawdown == False) & (shifted == True)
    
    # Get start and end dates for each drawdown
    # Convert to boolean mask and get indices
    mask_starts = drawdown_starts.values if hasattr(drawdown_starts, 'values') else drawdown_starts
    start_indices = [i for i, val in enumerate(mask_starts) if val]
    start_dates = equity.index[start_indices]
    mask_ends = drawdown_ends.values if hasattr(drawdown_ends, 'values') else drawdown_ends
    end_indices = [i for i, val in enumerate(mask_ends) if val]
    end_dates = equity.index[end_indices]
    
    # Handle case where we end in a drawdown
    if len(start_dates) > len(end_dates):
        end_dates = end_dates.append(pd.Index([equity.index[-1]]))
    
    # Calculate drawdown metrics
    drawdown_info = []
    
    for i in range(len(start_dates)):
        if i < len(end_dates):
            start_date = start_dates[i]
            end_date = end_dates[i]
            
            # Get drawdown period data
            period_equity = equity.loc[start_date:end_date]
            period_max = period_equity.iloc[0]  # Drawdown starts from previous peak
            period_min = period_equity.min()
            
            # Calculate drawdown amount and percentage
            drawdown_amount = period_max - period_min
            drawdown_pct = drawdown_amount / period_max
            
            # Calculate duration in trading days
            duration = len(period_equity)
            
            # Recovery date is the first date after the minimum when equity returns to peak
            try:
                min_date = period_equity.idxmin()
                recovery_equity = equity.loc[min_date:]
                recovery_mask = recovery_equity >= period_max
                
                if any(recovery_mask):
                    recovery_date = recovery_equity[recovery_mask].index[0]
                    recovery_duration = (recovery_date - min_date).days
                else:
                    recovery_date = None
                    recovery_duration = None
            except:
                min_date = None
                recovery_date = None
                recovery_duration = None
            
            # Append drawdown information
            drawdown_info.append({
                'start_date': start_date,
                'end_date': end_date,
                'min_date': min_date,
                'recovery_date': recovery_date,
                'drawdown_amount': drawdown_amount,
                'drawdown_pct': drawdown_pct,
                'duration': duration,
                'recovery_duration': recovery_duration
            })
    
    # Create drawdown DataFrame
    drawdowns = pd.DataFrame(drawdown_info)
    
    # Calculate key metrics
    # Get minimum drawdown safely
    max_drawdown = float(drawdown.min()) if hasattr(drawdown, 'min') else drawdown
    
    # Calmar ratio (annualized return / max drawdown)
    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (365 / max((equity.index[-1] - equity.index[0]).days, 1)) - 1
    # Ensure max_drawdown is a scalar
    max_dd = float(max_drawdown) if hasattr(max_drawdown, '__float__') else max_drawdown
    calmar_ratio = abs(annualized_return / max_dd) if max_dd < 0 else 0
    
    # Get drawdown statistics
    if len(drawdowns) > 0:
        avg_drawdown = drawdowns['drawdown_pct'].mean() if 'drawdown_pct' in drawdowns else 0
        avg_drawdown_duration = drawdowns['duration'].mean() if 'duration' in drawdowns else 0
        max_drawdown_duration = drawdowns['duration'].max() if 'duration' in drawdowns else 0
    else:
        avg_drawdown = 0
        avg_drawdown_duration = 0
        max_drawdown_duration = 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'avg_drawdown': avg_drawdown,
        'avg_drawdown_duration': avg_drawdown_duration,
        'calmar_ratio': calmar_ratio,
        'drawdowns': drawdowns,
        'underwater': underwater
    }


def _calculate_benchmark_metrics(
    equity_curve: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate benchmark comparison metrics.
    
    Parameters:
    -----------
    equity_curve : pd.Series
        Time series of portfolio equity values
    benchmark : pd.Series
        Time series of benchmark values
    risk_free_rate : float
        Annualized risk-free rate
    trading_days_per_year : int
        Number of trading days per year
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of benchmark comparison metrics
    """
    # Align dates
    aligned_data = pd.concat([equity_curve, benchmark], axis=1).dropna()
    
    if len(aligned_data) < 2:
        return {
            'alpha': 0.0,
            'beta': 0.0,
            'r_squared': 0.0,
            'tracking_error': 0.0,
            'information_ratio': 0.0,
            'excess_return': 0.0
        }
    
    # Rename columns
    aligned_data.columns = ['strategy', 'benchmark']
    
    # Calculate returns
    returns = aligned_data.pct_change().dropna()
    strategy_returns = returns['strategy']
    benchmark_returns = returns['benchmark']
    
    # Calculate beta using covariance and variance
    covariance = strategy_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    beta = covariance / variance if variance > 0 else 0
    
    # Calculate alpha
    # Alpha = Portfolio Return - [Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate)]
    strategy_return = (aligned_data['strategy'].iloc[-1] / aligned_data['strategy'].iloc[0]) - 1
    benchmark_return = (aligned_data['benchmark'].iloc[-1] / aligned_data['benchmark'].iloc[0]) - 1
    
    daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
    alpha = strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    # Calculate R-squared (correlation squared)
    correlation = strategy_returns.corr(benchmark_returns)
    r_squared = correlation ** 2
    
    # Calculate tracking error (standard deviation of excess returns)
    excess_returns = strategy_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(trading_days_per_year)
    
    # Calculate information ratio (excess return / tracking error)
    excess_return = strategy_return - benchmark_return
    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
    
    return {
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'excess_return': excess_return,
        'excess_returns': excess_returns,
        'benchmark_return': benchmark_return
    }


def _calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate trade-based performance metrics.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame containing individual trade information
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of trade-based metrics
    """
    # Check if we have required columns
    required_columns = ['entry_time', 'exit_time', 'pnl', 'side']
    
    # Initialize default metrics
    metrics = {
        'total_trades': 0,
        'win_rate': 0.0,
        'avg_trade_pnl': 0.0,
        'avg_winner_pnl': 0.0,
        'avg_loser_pnl': 0.0,
        'largest_winner': 0.0,
        'largest_loser': 0.0,
        'profit_factor': 0.0,
        'avg_trade_duration': 0.0
    }
    
    if trades is None or len(trades) == 0:
        return metrics
    
    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in trades.columns]
    if missing_columns:
        # Fill with default values if columns are missing
        return metrics
    
    # Calculate metrics from trades
    total_trades = len(trades)
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    # Basic metrics
    metrics['total_trades'] = total_trades
    metrics['winning_trades'] = len(winning_trades)
    metrics['losing_trades'] = len(losing_trades)
    metrics['win_rate'] = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # PnL metrics
    metrics['total_pnl'] = trades['pnl'].sum()
    metrics['avg_trade_pnl'] = trades['pnl'].mean()
    metrics['avg_winner_pnl'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    metrics['avg_loser_pnl'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    metrics['largest_winner'] = trades['pnl'].max() if total_trades > 0 else 0
    metrics['largest_loser'] = trades['pnl'].min() if total_trades > 0 else 0
    
    # Calculate profit factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Duration metrics
    if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
        try:
            # Convert string columns to datetime if necessary
            if not pd.api.types.is_datetime64_dtype(trades['entry_time']):
                trades['entry_time'] = pd.to_datetime(trades['entry_time'])
            
            if not pd.api.types.is_datetime64_dtype(trades['exit_time']):
                trades['exit_time'] = pd.to_datetime(trades['exit_time'])
            
            # Calculate trade durations
            trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 60  # in minutes
            
            metrics['avg_trade_duration'] = trades['duration'].mean()
            metrics['avg_winner_duration'] = winning_trades['duration'].mean() if len(winning_trades) > 0 else 0
            metrics['avg_loser_duration'] = losing_trades['duration'].mean() if len(losing_trades) > 0 else 0
        except:
            # Handle any errors in duration calculation
            metrics['avg_trade_duration'] = 0
            metrics['avg_winner_duration'] = 0
            metrics['avg_loser_duration'] = 0
    
    # Analyze by trade side if available
    if 'side' in trades.columns:
        long_trades = trades[trades['side'] == 'long']
        short_trades = trades[trades['side'] == 'short']
        
        metrics['long_trades'] = len(long_trades)
        metrics['short_trades'] = len(short_trades)
        
        metrics['long_win_rate'] = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        metrics['short_win_rate'] = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        metrics['long_pnl'] = long_trades['pnl'].sum()
        metrics['short_pnl'] = short_trades['pnl'].sum()
        
        metrics['avg_long_pnl'] = long_trades['pnl'].mean() if len(long_trades) > 0 else 0
        metrics['avg_short_pnl'] = short_trades['pnl'].mean() if len(short_trades) > 0 else 0
    
    # Calculate expectancy
    avg_win = metrics['avg_winner_pnl']
    avg_loss = abs(metrics['avg_loser_pnl'])
    win_rate = metrics['win_rate']
    
    metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    metrics['risk_reward_ratio'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Calculate statistical significance
    if total_trades >= 30:  # Only if we have enough trades
        # Use binomial test to determine if win rate is significantly different from random
    
        # Use binomtest instead of binom_test for newer scipy versions

        try:

            p_value = stats.binom_test(metrics['winning_trades'], n=total_trades, p=0.5)

        except AttributeError:

            # For newer scipy versions

            result = stats.binomtest(metrics['winning_trades'], n=total_trades, p=0.5)

            p_value = result.pvalue

        metrics['win_rate_p_value'] = p_value
        metrics['statistically_significant'] = p_value < 0.05
    
    return metrics


def _create_summary_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create summary metrics for quick overview.
    
    Parameters:
    -----------
    metrics : Dict[str, Any]
        Dictionary of all calculated metrics
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary of summary metrics
    """
    summary = {}
    
    # Return if metrics is empty
    if not metrics:
        return summary
    
    # Extract key metrics
    if 'total_return' in metrics:
        summary['total_return'] = metrics['total_return']
    
    if 'annualized_return' in metrics:
        summary['annualized_return'] = metrics['annualized_return']
    
    if 'sharpe_ratio' in metrics:
        summary['sharpe_ratio'] = metrics['sharpe_ratio']
    
    if 'sortino_ratio' in metrics:
        summary['sortino_ratio'] = metrics['sortino_ratio']
    
    if 'max_drawdown' in metrics:
        summary['max_drawdown'] = metrics['max_drawdown']
    
    if 'calmar_ratio' in metrics:
        summary['calmar_ratio'] = metrics['calmar_ratio']
    
    if 'win_rate' in metrics:
        summary['win_rate'] = metrics['win_rate']
    
    if 'profit_factor' in metrics:
        summary['profit_factor'] = metrics['profit_factor']
    
    if 'expectancy' in metrics:
        summary['expectancy'] = metrics['expectancy']
    
    # Calculate risk-adjusted CAGR
    if 'annualized_return' in metrics and 'max_drawdown' in metrics:
        max_dd = abs(metrics['max_drawdown'])
        if float(max_dd) > 0:
            summary['risk_adjusted_cagr'] = metrics['annualized_return'] / max_dd
        else:
            summary['risk_adjusted_cagr'] = float('inf')
    
    # Add strategy rating
    if 'sharpe_ratio' in metrics and 'sortino_ratio' in metrics and 'calmar_ratio' in metrics:
        # Calculate weighted average of key metrics
        sharpe = metrics['sharpe_ratio']
        sortino = metrics['sortino_ratio']
        calmar = metrics['calmar_ratio']
        
        # Normalize to 0-5 scale
        # Ensure sharpe is a scalar

        sharpe_val = float(sharpe) if isinstance(sharpe, (pd.Series, pd.DataFrame)) else sharpe

        sharpe_score = min(5, max(0, sharpe_val))

        sortino_score = min(5, max(0, sortino / 2))
        calmar_score = min(5, max(0, calmar * 2))
        
        # Calculate weighted score (0-5 scale)
        score = (sharpe_score * 0.4) + (sortino_score * 0.3) + (calmar_score * 0.3)
        
        # Convert to A, B, C, D, F rating
        if score >= 4:
            rating = 'A'
        elif score >= 3:
            rating = 'B'
        elif score >= 2:
            rating = 'C'
        elif score >= 1:
            rating = 'D'
        else:
            rating = 'F'
        
        summary['strategy_score'] = score
        summary['strategy_rating'] = rating
    
    return {'summary': summary} 