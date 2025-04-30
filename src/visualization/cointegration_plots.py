"""
Cointegration Visualization Tools

This module provides specialized visualization functions for cointegration analysis.
Functions include plotting cointegration relationships, spread behavior visualization,
diagnostic plots for residuals, and other statistical visualizations relevant to
pairs trading analysis.

These visualization tools enhance the Phase 1 deliverables by providing clear graphical
representations of cointegration relationships and statistical properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from typing import Dict, List, Tuple, Union, Optional
import os
from datetime import datetime

# Import statistical methods
from src.cointegration.statistical_methods import johansen_test, engle_granger_test
from src.cointegration.cointegration_tests import calculate_half_life

# Set default plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def plot_cointegration_relationship(
    price1: pd.Series, 
    price2: pd.Series,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Comprehensive visualization of a cointegration relationship between two price series.
    
    Creates a multi-panel plot showing:
    1. Original price series
    2. Scatter plot with regression line showing relationship
    3. Residual spread and its distribution
    4. Rolling statistics for the spread
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    title : str, optional
        Title for the plot
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object for further customization if needed
    """
    # Ensure we have aligned data
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Run Engle-Granger test to get the cointegration parameters
    try:
        eg_result = engle_granger_test(s2, s1)
        hedge_ratio = eg_result['hedge_ratio']
        intercept = eg_result['intercept']
        residuals = eg_result['residuals']
        is_cointegrated = eg_result['is_cointegrated']
        half_life = eg_result.get('half_life', np.nan)
        p_value = eg_result.get('p_value', np.nan)
    except Exception as e:
        print(f"Error in cointegration test: {str(e)}")
        hedge_ratio = np.polyfit(s1, s2, 1)[0]
        intercept = np.polyfit(s1, s2, 1)[1]
        residuals = s2 - (hedge_ratio * s1 + intercept)
        is_cointegrated = False
        half_life = np.nan
        p_value = np.nan
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    
    # Price series plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(s1.index, s1, label=f'{price1.name if hasattr(price1, "name") else "Series 1"}')
    ax1.plot(s2.index, s2, label=f'{price2.name if hasattr(price2, "name") else "Series 2"}')
    ax1.set_title('Price Series')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Scatter plot with regression line
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(s1, s2, alpha=0.5, s=10)
    ax2.plot(s1, hedge_ratio * s1 + intercept, 'r--', linewidth=2)
    ax2.set_title(f'Price Relationship\nHedge Ratio: {hedge_ratio:.4f}, Intercept: {intercept:.4f}')
    ax2.set_xlabel(f'{price1.name if hasattr(price1, "name") else "Series 1"}')
    ax2.set_ylabel(f'{price2.name if hasattr(price2, "name") else "Series 2"}')
    
    # Residual spread
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(residuals.index, residuals)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax3.axhline(y=residuals.mean(), color='g', linestyle='--')
    ax3.set_title(f'Residual Spread\nHalf-Life: {half_life:.2f} days, p-value: {p_value:.4f}')
    
    # Spread distribution
    ax4 = fig.add_subplot(gs[2, 0])
    sns.histplot(residuals, kde=True, ax=ax4)
    ax4.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    ax4.set_title('Spread Distribution')
    
    # Rolling statistics
    ax5 = fig.add_subplot(gs[2, 1])
    window_size = min(30, len(residuals) // 10)
    rolling_mean = residuals.rolling(window=window_size).mean()
    rolling_std = residuals.rolling(window=window_size).std()
    ax5.plot(residuals.index, residuals, label='Spread')
    ax5.plot(rolling_mean.index, rolling_mean, label=f'{window_size}-day Moving Average')
    ax5.plot(rolling_std.index, rolling_std, label=f'{window_size}-day Moving Std')
    ax5.legend()
    ax5.set_title('Rolling Statistics')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        cointegration_status = "Cointegrated" if is_cointegrated else "Not Cointegrated"
        fig.suptitle(f'Cointegration Analysis: {cointegration_status}', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mean_reversion_zones(
    spread: pd.Series,
    z_scores: Optional[pd.Series] = None,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    window: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the mean-reversion behavior of a spread with trading zones.
    
    Creates a multi-panel plot showing:
    1. Original spread with rolling mean and standard deviation bands
    2. Z-scores with entry/exit thresholds for trading signals
    3. Mean-reversion strength metrics
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series
    z_scores : pd.Series, optional
        Pre-calculated z-scores. If None, they will be calculated
    entry_threshold : float, default=2.0
        Z-score threshold for trade entry
    exit_threshold : float, default=0.5
        Z-score threshold for trade exit
    window : int, default=20
        Rolling window size for z-score calculation
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object for further customization if needed
    """
    # Calculate z-scores if not provided
    if z_scores is None:
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_scores = (spread - rolling_mean) / rolling_std
    
    # Calculate half-life
    try:
        half_life_result = calculate_half_life(spread)
        half_life = half_life_result.get('half_life', np.nan)
        
        # Check if half-life result contains more metrics
        if isinstance(half_life_result, dict):
            r_squared = half_life_result.get('r_squared', np.nan)
            hurst_exponent = half_life_result.get('hurst_exponent', np.nan)
        else:
            r_squared = np.nan
            hurst_exponent = np.nan
    except Exception:
        half_life = np.nan
        r_squared = np.nan
        hurst_exponent = np.nan
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.5, 1])
    
    # Spread with rolling mean and bands
    ax1 = fig.add_subplot(gs[0])
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    ax1.plot(spread.index, spread, label='Spread', color='blue')
    ax1.plot(rolling_mean.index, rolling_mean, label=f'{window}-day Mean', color='red', 
             linestyle='--', linewidth=2)
    
    # Add standard deviation bands
    for n_std in [1, 2]:
        upper_band = rolling_mean + n_std * rolling_std
        lower_band = rolling_mean - n_std * rolling_std
        ax1.fill_between(rolling_mean.index, lower_band, upper_band, 
                         alpha=0.1 if n_std == 2 else 0.2, color='gray',
                         label=f'{n_std} Std Dev' if n_std == 1 else None)
    
    ax1.set_title('Spread with Rolling Mean and Standard Deviation Bands')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Z-scores with trading zones
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(z_scores.index, z_scores, label='Z-Score', color='blue')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add entry and exit thresholds
    ax2.axhline(y=entry_threshold, color='red', linestyle='--', label=f'Entry (+{entry_threshold})')
    ax2.axhline(y=-entry_threshold, color='red', linestyle='--')
    ax2.axhline(y=exit_threshold, color='green', linestyle='--', label=f'Exit (+{exit_threshold})')
    ax2.axhline(y=-exit_threshold, color='green', linestyle='--')
    
    # Color trading zones
    ax2.fill_between(z_scores.index, entry_threshold, max(z_scores.max(), entry_threshold + 1), 
                     alpha=0.1, color='red', label='Short Zone')
    ax2.fill_between(z_scores.index, min(z_scores.min(), -entry_threshold - 1), -entry_threshold, 
                     alpha=0.1, color='green', label='Long Zone')
    
    ax2.set_title('Z-Score with Trading Zones')
    ax2.legend()
    
    # Mean-reversion metrics
    ax3 = fig.add_subplot(gs[2])
    metrics = pd.Series({
        'Half-Life (days)': half_life,
        'R-Squared': r_squared,
        'Hurst Exponent': hurst_exponent,
        'Mean-Reversion Strength': 1-hurst_exponent if not np.isnan(hurst_exponent) else np.nan
    })
    
    # Create a table in the plot
    ax3.axis('off')
    ax3.set_title('Mean-Reversion Metrics')
    
    # Color-code mean-reversion strength
    mean_rev_strength = 1-hurst_exponent if not np.isnan(hurst_exponent) else np.nan
    mean_rev_color = 'green' if mean_rev_strength > 0.5 else 'orange' if mean_rev_strength > 0.3 else 'red'
    
    # Create custom table text with interpretation
    table_text = [
        f"Half-Life: {half_life:.2f} days" + 
        (" (Good)" if 1 <= half_life <= 20 else " (Too Fast)" if half_life < 1 else " (Too Slow)"),
        
        f"R-Squared: {r_squared:.4f}" +
        (" (Strong)" if r_squared > 0.7 else " (Moderate)" if r_squared > 0.3 else " (Weak)"),
        
        f"Hurst Exponent: {hurst_exponent:.4f}" +
        (" (Mean-Reverting)" if hurst_exponent < 0.5 else " (Random)" if hurst_exponent == 0.5 else " (Trending)"),
        
        f"Mean-Reversion Strength: {mean_rev_strength:.4f}" +
        (" (Strong)" if mean_rev_strength > 0.5 else " (Moderate)" if mean_rev_strength > 0.3 else " (Weak)")
    ]
    
    # Add a visual interpretation box
    interpretation = ""
    if not np.isnan(half_life) and not np.isnan(hurst_exponent):
        if half_life < 1:
            interpretation += "⚠️ Half-life too short - spread reverts too quickly to trade effectively.\n"
        elif half_life > 20:
            interpretation += "⚠️ Half-life too long - spread takes too long to revert to mean.\n"
        else:
            interpretation += "✓ Half-life in optimal trading range.\n"
            
        if hurst_exponent < 0.4:
            interpretation += "✓ Strong mean-reversion characteristics.\n"
        elif hurst_exponent < 0.5:
            interpretation += "✓ Moderate mean-reversion characteristics.\n"
        else:
            interpretation += "⚠️ Weak or no mean-reversion characteristics.\n"
    else:
        interpretation = "⚠️ Insufficient data for reliable mean-reversion assessment."
    
    # Display table text and interpretation
    y_pos = 0.8
    for line in table_text:
        ax3.text(0.1, y_pos, line, fontsize=12, ha='left')
        y_pos -= 0.2
    
    # Add interpretation box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax3.text(0.5, 0.3, interpretation, transform=ax3.transAxes, fontsize=10,
             verticalalignment='center', horizontalalignment='center', bbox=props)
    
    # Overall title
    half_life_desc = f"Half-Life: {half_life:.2f} days" if not np.isnan(half_life) else "Half-Life: Unknown"
    fig.suptitle(f'Mean-Reversion Analysis\n{half_life_desc}', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residual_diagnostics(
    residuals: pd.Series,
    lags: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create diagnostic plots for residuals from a cointegration regression.
    
    Creates a multi-panel plot showing:
    1. Time series of residuals
    2. Q-Q plot to assess normality
    3. Autocorrelation function (ACF)
    4. Partial autocorrelation function (PACF)
    5. Histogram with normal distribution fit
    
    Parameters
    ----------
    residuals : pd.Series
        Time series of residuals from cointegration regression
    lags : int, default=20
        Number of lags for ACF and PACF plots
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object for further customization if needed
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Time series of residuals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(residuals.index, residuals)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residual Time Series')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Plot 2: Q-Q plot
    ax2 = fig.add_subplot(gs[1, 0])
    qqplot(residuals, line='s', ax=ax2)
    ax2.set_title('Q-Q Plot (Check for Normality)')
    
    # Plot 3: ACF
    ax3 = fig.add_subplot(gs[1, 1])
    plot_acf(residuals.dropna(), lags=lags, ax=ax3)
    ax3.set_title('Autocorrelation Function (ACF)')
    
    # Plot 4: PACF
    ax4 = fig.add_subplot(gs[2, 0])
    plot_pacf(residuals.dropna(), lags=lags, ax=ax4)
    ax4.set_title('Partial Autocorrelation Function (PACF)')
    
    # Plot 5: Histogram with normal fit
    ax5 = fig.add_subplot(gs[2, 1])
    sns.histplot(residuals, kde=True, ax=ax5)
    # Add normal distribution fit
    x = np.linspace(residuals.min(), residuals.max(), 100)
    mu, std = stats.norm.fit(residuals.dropna())
    p = stats.norm.pdf(x, mu, std)
    ax5.plot(x, p * len(residuals) * (residuals.max() - residuals.min()) / 10, 
             'r-', linewidth=2, label=f'Normal: $\mu$={mu:.2f}, $\sigma$={std:.2f}')
    ax5.legend()
    ax5.set_title('Histogram with Normal Fit')
    
    # Run statistical tests for the residuals
    # Shapiro-Wilk test for normality
    try:
        shapiro_test = stats.shapiro(residuals.dropna())
        shapiro_pval = shapiro_test[1]
        is_normal = shapiro_pval > 0.05
    except:
        shapiro_pval = np.nan
        is_normal = None
    
    # Ljung-Box test for autocorrelation
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals.dropna(), lags=[10])
        lb_pval = lb_test.iloc[0, 1]  # p-value
        has_autocorr = lb_pval < 0.05
    except:
        lb_pval = np.nan
        has_autocorr = None
    
    # ADF test for stationarity
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_test = adfuller(residuals.dropna())
        adf_pval = adf_test[1]
        is_stationary = adf_pval < 0.05
    except:
        adf_pval = np.nan
        is_stationary = None
    
    # Add a summary box with test results
    summary_text = f"""
    Statistical Tests:
    -----------------
    Normality (Shapiro-Wilk): p={shapiro_pval:.4f} ({'Normal' if is_normal else 'Non-normal' if is_normal is not None else 'Unknown'})
    Autocorrelation (Ljung-Box): p={lb_pval:.4f} ({'Present' if has_autocorr else 'Absent' if has_autocorr is not None else 'Unknown'})
    Stationarity (ADF): p={adf_pval:.4f} ({'Stationary' if is_stationary else 'Non-stationary' if is_stationary is not None else 'Unknown'})
    """
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    fig.text(0.5, 0.01, summary_text, fontsize=10, 
             bbox=props, horizontalalignment='center')
    
    # Overall title
    fig.suptitle('Residual Diagnostics', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rolling_window_analysis(
    price1: pd.Series,
    price2: pd.Series, 
    window_size: int = 252,
    step_size: int = 20,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling window analysis of cointegration parameters.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    window_size : int, default=252
        Size of rolling window in days
    step_size : int, default=20
        Number of days to step forward between windows
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Ensure index alignment
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Need enough data
    if len(s1) < window_size:
        raise ValueError(f"Not enough data. Need at least {window_size} points.")
    
    # Create windows
    windows = []
    for start in range(0, len(s1) - window_size, step_size):
        end = start + window_size
        windows.append((start, end))
    
    # Calculate parameters for each window
    dates = []
    hedge_ratios = []
    half_lives = []
    p_values = []
    is_cointegrated = []
    
    for start, end in windows:
        window_s1 = s1.iloc[start:end]
        window_s2 = s2.iloc[start:end]
        
        # Get end date for this window
        end_date = window_s1.index[-1]
        dates.append(end_date)
        
        # Run Engle-Granger test
        try:
            result = engle_granger_test(window_s2, window_s1)
            hedge_ratios.append(result['hedge_ratio'])
            half_lives.append(result.get('half_life', np.nan))
            p_values.append(result.get('p_value', np.nan))
            is_cointegrated.append(result['is_cointegrated'])
        except Exception:
            hedge_ratios.append(np.nan)
            half_lives.append(np.nan)
            p_values.append(np.nan)
            is_cointegrated.append(False)
    
    # Convert to Series for easier plotting
    hedge_ratio_series = pd.Series(hedge_ratios, index=dates)
    half_life_series = pd.Series(half_lives, index=dates)
    p_value_series = pd.Series(p_values, index=dates)
    cointegrated_series = pd.Series(is_cointegrated, index=dates).astype(int)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Hedge ratios
    axes[0].plot(hedge_ratio_series.index, hedge_ratio_series, 'b-')
    axes[0].set_title('Rolling Hedge Ratio')
    
    # Calculate hedge ratio stability metrics
    hr_mean = hedge_ratio_series.mean()
    hr_std = hedge_ratio_series.std()
    hr_cv = hr_std / hr_mean if hr_mean != 0 else np.nan
    
    # Add stability annotation
    stability_text = f"Mean: {hr_mean:.4f}, Std: {hr_std:.4f}, CV: {hr_cv:.4f}"
    stability_rating = "Stable" if hr_cv < 0.1 else "Moderate" if hr_cv < 0.2 else "Unstable"
    axes[0].annotate(f"{stability_text}\nStability: {stability_rating}", 
                    xy=(0.02, 0.85), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot 2: Half-lives
    axes[1].plot(half_life_series.index, half_life_series, 'g-')
    axes[1].set_title('Rolling Half-Life (days)')
    axes[1].set_ylim(bottom=0)
    
    # Add horizontal lines for optimal half-life range
    axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.3)
    axes[1].axhline(y=20, color='r', linestyle='--', alpha=0.3)
    axes[1].fill_between(half_life_series.index, 1, 20, color='g', alpha=0.1)
    
    # Plot 3: p-values
    axes[2].plot(p_value_series.index, p_value_series, 'r-')
    axes[2].axhline(y=0.05, color='k', linestyle='--')
    axes[2].set_title('Rolling p-value')
    axes[2].set_ylim(0, min(1, p_value_series.max() * 1.1))
    
    # Plot 4: Cointegration status
    axes[3].plot(cointegrated_series.index, cointegrated_series, 'ko-')
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['No', 'Yes'])
    axes[3].set_title('Cointegration Status')
    
    # Calculate stability percentage
    stability_pct = cointegrated_series.mean() * 100
    axes[3].annotate(f"Stability: {stability_pct:.1f}% of windows show cointegration", 
                    xy=(0.02, 0.8), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Overall title
    title = f"Rolling Window Analysis ({window_size} days window, {step_size} days step)"
    if stability_pct > 90:
        title += " - VERY STABLE Cointegration"
    elif stability_pct > 75:
        title += " - STABLE Cointegration"
    elif stability_pct > 50:
        title += " - MODERATELY STABLE Cointegration"
    else:
        title += " - UNSTABLE Cointegration"
    
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_structural_breaks(
    spread: pd.Series,
    break_dates: Optional[List] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize structural breaks in a cointegration relationship.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series
    break_dates : List, optional
        List of dates where structural breaks occur. If None, breaks will be estimated
    figsize : tuple, default=(14, 8)
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Estimate break dates if not provided
    if break_dates is None:
        try:
            # Try to import structural break detection from statsmodels
            from statsmodels.tsa.stattools import breakvar_heteroskedasticity
            
            # Check if we have enough data
            if len(spread) > 50:
                # Detect structural breaks
                break_test = breakvar_heteroskedasticity(spread.dropna())
                # Get top 3 most significant breaks
                if hasattr(break_test, 'breakpoints'):
                    break_indices = break_test.breakpoints[:3]
                    break_dates = [spread.index[i] for i in break_indices if i < len(spread)]
                else:
                    break_dates = []
            else:
                break_dates = []
        except ImportError:
            # Simple heuristic approach if statsmodels functions not available
            # Look for points where spread crosses beyond 3 standard deviations
            roll_mean = spread.rolling(window=30).mean()
            roll_std = spread.rolling(window=30).std()
            upper = roll_mean + 3 * roll_std
            lower = roll_mean - 3 * roll_std
            
            # Find potential breaks
            breaks = (
                ((spread > upper) & (spread.shift(1) <= upper)) | 
                ((spread < lower) & (spread.shift(1) >= lower))
            )
            
            # Get dates where breaks occur
            break_dates = spread.index[breaks]
            
            # Limit to top 3 breaks
            if len(break_dates) > 3:
                # Get the most extreme deviations
                z_scores = np.abs((spread - roll_mean) / roll_std)
                extreme_scores = z_scores.loc[break_dates]
                break_dates = extreme_scores.nlargest(3).index.tolist()
    
    # Plot 1: Original spread with break points
    ax1.plot(spread.index, spread)
    ax1.set_title('Spread with Structural Breaks')
    
    # Add vertical lines for breaks
    for break_date in break_dates:
        ax1.axvline(x=break_date, color='r', linestyle='--', alpha=0.7)
        # Add annotation
        ax1.annotate('Break', xy=(mdates.date2num(break_date), min(spread)),
                    xytext=(0, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                    ha='center')
    
    # Plot 2: Moving window statistics to show changing properties
    window = min(60, len(spread) // 4)  # Reasonable window size
    roll_mean = spread.rolling(window=window).mean()
    roll_std = spread.rolling(window=window).std()
    
    ax2.plot(roll_mean.index, roll_mean, label=f'{window}-day Mean')
    ax2.plot(roll_std.index, roll_std, label=f'{window}-day Std Dev')
    ax2.set_title('Rolling Statistics')
    ax2.legend()
    
    # Add segment shading if breaks exist
    if break_dates:
        # Sort break dates
        break_dates = sorted(break_dates)
        
        # Create segments including start and end of series
        all_points = [spread.index[0]] + break_dates + [spread.index[-1]]
        
        # Color alternating segments
        segment_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i+1]
            color = segment_colors[i % len(segment_colors)]
            
            # Shade area in both plots
            for ax in [ax1, ax2]:
                ax.axvspan(start, end, alpha=0.2, color=color)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add break summary
    if break_dates:
        fig.suptitle(f'Structural Break Analysis: {len(break_dates)} breaks detected', fontsize=16)
        break_summary = "Break dates: " + ", ".join([d.strftime('%Y-%m-%d') for d in break_dates])
        fig.text(0.5, 0.01, break_summary, ha='center', fontsize=10)
    else:
        fig.suptitle('Structural Break Analysis: No significant breaks detected', fontsize=16)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, bottom=0.1)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_cointegration_report(
    price1: pd.Series,
    price2: pd.Series,
    output_dir: str = 'reports',
    prefix: str = '',
    show_plots: bool = False
) -> Dict[str, str]:
    """
    Create a comprehensive cointegration analysis report with multiple visualization plots.
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    output_dir : str, default='reports'
        Directory to save output plots
    prefix : str, default=''
        Prefix for filenames
    show_plots : bool, default=False
        Whether to display plots in addition to saving them
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot descriptions to file paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate report timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Format prefix
    if prefix and not prefix.endswith('_'):
        prefix = prefix + '_'
    
    # Dictionary to store file paths
    plots_info = {}
    
    # Create names for the series
    series1_name = getattr(price1, 'name', 'Series_1')
    series2_name = getattr(price2, 'name', 'Series_2')
    pair_name = f"{series1_name}_{series2_name}"
    
    # 1. Cointegration relationship plot
    filename = f"{prefix}cointegration_{pair_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    fig = plot_cointegration_relationship(price1, price2, save_path=filepath)
    if not show_plots:
        plt.close(fig)
    plots_info['Cointegration Relationship'] = filepath
    
    # Run Engle-Granger test to get residuals and other parameters
    try:
        eg_result = engle_granger_test(price2, price1)
        residuals = eg_result['residuals']
        is_cointegrated = eg_result['is_cointegrated']
        
        # 2. Mean reversion zones plot
        filename = f"{prefix}mean_reversion_{pair_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig = plot_mean_reversion_zones(residuals, save_path=filepath)
        if not show_plots:
            plt.close(fig)
        plots_info['Mean Reversion Analysis'] = filepath
        
        # 3. Residual diagnostics plot
        filename = f"{prefix}residual_diagnostics_{pair_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig = plot_residual_diagnostics(residuals, save_path=filepath)
        if not show_plots:
            plt.close(fig)
        plots_info['Residual Diagnostics'] = filepath
        
        # 4. Rolling window analysis
        if len(price1) >= 252:  # Need at least a year of data
            filename = f"{prefix}rolling_window_{pair_name}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            fig = plot_rolling_window_analysis(price1, price2, save_path=filepath)
            if not show_plots:
                plt.close(fig)
            plots_info['Rolling Window Analysis'] = filepath
        
        # 5. Structural breaks
        filename = f"{prefix}structural_breaks_{pair_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        fig = plot_structural_breaks(residuals, save_path=filepath)
        if not show_plots:
            plt.close(fig)
        plots_info['Structural Break Analysis'] = filepath
    
    except Exception as e:
        print(f"Error generating additional plots: {str(e)}")
    
    # Create HTML report if matplotlib has this capability
    try:
        # Create a simple HTML report
        html_content = f"""
        <html>
        <head>
            <title>Cointegration Analysis: {pair_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .report-section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Cointegration Analysis Report: {pair_name}</h1>
            <div class="summary">
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Series 1:</strong> {series1_name}</p>
                <p><strong>Series 2:</strong> {series2_name}</p>
                <p><strong>Cointegration Result:</strong> {"Cointegrated" if is_cointegrated else "Not Cointegrated"}</p>
            </div>
        """
        
        # Add each plot section
        for title, path in plots_info.items():
            filename = os.path.basename(path)
            html_content += f"""
            <div class="report-section">
                <h2>{title}</h2>
                <img src="{filename}" alt="{title}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        html_filename = f"{prefix}cointegration_report_{pair_name}_{timestamp}.html"
        html_filepath = os.path.join(output_dir, html_filename)
        with open(html_filepath, 'w') as f:
            f.write(html_content)
        
        plots_info['HTML Report'] = html_filepath
    
    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
    
    return plots_info


def plot_interactive_kalman_filter(
    price1: pd.Series,
    price2: pd.Series,
    kalman_results: Optional[pd.DataFrame] = None,
    add_intercept: bool = True,
    transition_covariance: float = 1e-4,
    observation_covariance: float = 1e-2,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create an interactive visualization of Kalman filter results for cointegration analysis.
    
    This function creates a multi-panel plot with interactive elements showing:
    1. Original price series
    2. Time-varying hedge ratio
    3. Spread calculated using time-varying hedge ratio
    4. Z-scores with regime highlighting
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    kalman_results : pd.DataFrame, optional
        Pre-calculated Kalman filter results. If None, they will be calculated
    add_intercept : bool, default=True
        Whether to add an intercept term to the regression
    transition_covariance : float, default=1e-4
        Transition covariance for Kalman filter
    observation_covariance : float, default=1e-2
        Observation covariance for Kalman filter
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object
        
    Notes
    -----
    This function requires the Kalman filter implementation from 
    src.cointegration.kalman_filter to calculate the time-varying hedge ratio.
    The plot has interactive elements if displayed in a notebook environment.
    """
    try:
        from src.cointegration.kalman_filter import estimate_timevarying_hedge_ratio
        from matplotlib.widgets import Slider, CheckButtons
    except ImportError as e:
        print(f"Required module not found: {e}")
        return None
    
    # Ensure we have aligned data
    common_index = price1.index.intersection(price2.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    
    # Calculate Kalman filter results if not provided
    if kalman_results is None:
        kalman_results = estimate_timevarying_hedge_ratio(
            s2, s1, 
            add_intercept=add_intercept,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance
        )
    
    # Extract data from Kalman results
    hedge_ratios = kalman_results['hedge_ratio']
    intercepts = kalman_results.get('intercept', pd.Series(0, index=hedge_ratios.index))
    spreads = kalman_results['spread']
    
    # Calculate z-scores using a rolling window
    window_size = min(20, len(spreads) // 10)
    z_scores = (spreads - spreads.rolling(window=window_size).mean()) / spreads.rolling(window=window_size).std()
    
    # Create figure with gridspec for proper spacing
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 1, figure=fig, height_ratios=[1, 0.3, 1, 0.3, 1, 1])
    
    # Price series plot
    ax1 = fig.add_subplot(gs[0])
    line1, = ax1.plot(s1.index, s1, label=f'{price1.name if hasattr(price1, "name") else "Series 1"}')
    line2, = ax1.plot(s2.index, s2, label=f'{price2.name if hasattr(price2, "name") else "Series 2"}')
    ax1.set_title('Price Series')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Add slider for price scaling
    ax_slider1 = fig.add_subplot(gs[1])
    price_slider = Slider(ax_slider1, 'Normalize', 0, 1, valinit=0)
    
    # Time-varying hedge ratio
    ax2 = fig.add_subplot(gs[2])
    line3, = ax2.plot(hedge_ratios.index, hedge_ratios, label='Hedge Ratio', color='green')
    ax2.set_title('Time-Varying Hedge Ratio')
    ax2.set_ylim(min(hedge_ratios) * 0.9, max(hedge_ratios) * 1.1)
    
    # Add slider for smoothing hedge ratio
    ax_slider2 = fig.add_subplot(gs[3])
    smooth_slider = Slider(ax_slider2, 'Smoothing', 1, 30, valinit=1, valstep=1)
    
    # Spread plot
    ax3 = fig.add_subplot(gs[4], sharex=ax1)
    line4, = ax3.plot(spreads.index, spreads, label='Spread', color='purple')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Kalman Filter Spread')
    
    # Z-score plot with regime highlighting
    ax4 = fig.add_subplot(gs[5], sharex=ax1)
    line5, = ax4.plot(z_scores.index, z_scores, label='Z-Score', color='blue')
    
    # Add threshold lines
    threshold = 2.0
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax4.axhline(y=threshold, color='r', linestyle='--', alpha=0.8, label=f'+{threshold} (Short Entry)')
    ax4.axhline(y=-threshold, color='g', linestyle='--', alpha=0.8, label=f'-{threshold} (Long Entry)')
    ax4.axhline(y=threshold/2, color='orange', linestyle=':', alpha=0.8, label=f'+{threshold/2} (Short Exit)')
    ax4.axhline(y=-threshold/2, color='cyan', linestyle=':', alpha=0.8, label=f'-{threshold/2} (Long Exit)')
    
    # Color the regions based on z-score
    long_positions = z_scores < -threshold
    short_positions = z_scores > threshold
    neutral_positions = (~long_positions) & (~short_positions)
    
    # Create regions to highlight
    ax4.fill_between(z_scores.index, -threshold, z_scores.where(z_scores < -threshold), color='green', alpha=0.2)
    ax4.fill_between(z_scores.index, threshold, z_scores.where(z_scores > threshold), color='red', alpha=0.2)
    
    ax4.set_title('Z-Scores with Trading Zones')
    ax4.legend()
    
    # Add checkboxes for showing/hiding elements
    ax_check = plt.axes([0.02, 0.01, 0.15, 0.08])
    check_buttons = CheckButtons(ax_check, ['Show Intercept', 'Highlight Regimes', 'Show Bands'], 
                                [False, True, True])
    
    # Add slider for threshold adjustment
    ax_thresh = plt.axes([0.25, 0.01, 0.65, 0.03])
    thresh_slider = Slider(ax_thresh, 'Threshold', 0.5, 3.0, valinit=threshold, valstep=0.1)
    
    # Function to update the plots based on slider/checkbox changes
    def update(val):
        # Update price normalization
        if price_slider.val > 0:
            # Normalize prices
            s1_norm = s1 / s1.iloc[0]
            s2_norm = s2 / s2.iloc[0]
            # Interpolate between original and normalized
            s1_plot = s1 * (1 - price_slider.val) + s1_norm * price_slider.val
            s2_plot = s2 * (1 - price_slider.val) + s2_norm * price_slider.val
            line1.set_ydata(s1_plot)
            line2.set_ydata(s2_plot)
            # Adjust y-limits
            ax1.set_ylim(min(min(s1_plot), min(s2_plot)) * 0.95, 
                         max(max(s1_plot), max(s2_plot)) * 1.05)
        else:
            # Original prices
            line1.set_ydata(s1)
            line2.set_ydata(s2)
            ax1.set_ylim(min(min(s1), min(s2)) * 0.95, 
                         max(max(s1), max(s2)) * 1.05)
        
        # Update hedge ratio smoothing
        if smooth_slider.val > 1:
            # Apply smoothing
            hedge_ratios_smooth = hedge_ratios.rolling(window=int(smooth_slider.val)).mean()
            line3.set_ydata(hedge_ratios_smooth)
        else:
            # Original hedge ratios
            line3.set_ydata(hedge_ratios)
        
        # Update threshold value
        new_threshold = thresh_slider.val
        # Update threshold lines
        ax4.lines[1].set_ydata([new_threshold, new_threshold])
        ax4.lines[2].set_ydata([-new_threshold, -new_threshold])
        ax4.lines[3].set_ydata([new_threshold/2, new_threshold/2])
        ax4.lines[4].set_ydata([-new_threshold/2, -new_threshold/2])
        # Update threshold labels
        ax4.lines[1].set_label(f'+{new_threshold} (Short Entry)')
        ax4.lines[2].set_label(f'-{new_threshold} (Long Entry)')
        ax4.lines[3].set_label(f'+{new_threshold/2} (Short Exit)')
        ax4.lines[4].set_label(f'-{new_threshold/2} (Long Exit)')
        
        # Clear previous highlighting
        for collection in ax4.collections:
            collection.remove()
        
        # Add new highlighting if enabled
        if check_buttons.get_status()[1]:  # Highlight Regimes checkbox
            ax4.fill_between(z_scores.index, -new_threshold, z_scores.where(z_scores < -new_threshold), 
                             color='green', alpha=0.2)
            ax4.fill_between(z_scores.index, new_threshold, z_scores.where(z_scores > new_threshold), 
                             color='red', alpha=0.2)
        
        # Show/hide intercept
        if check_buttons.get_status()[0]:  # Show Intercept checkbox
            if not hasattr(ax2, 'intercept_line') or ax2.intercept_line is None:
                ax2.intercept_line, = ax2.plot(intercepts.index, intercepts, label='Intercept', 
                                              color='brown', alpha=0.7, linestyle=':')
                ax2.legend()
        else:
            if hasattr(ax2, 'intercept_line') and ax2.intercept_line is not None:
                ax2.intercept_line.remove()
                ax2.intercept_line = None
                ax2.legend()
        
        # Show/hide bands
        if check_buttons.get_status()[2]:  # Show Bands checkbox
            if not hasattr(ax3, 'bands') or not ax3.bands:
                # Calculate rolling mean and std
                rolling_mean = spreads.rolling(window=window_size).mean()
                rolling_std = spreads.rolling(window=window_size).std()
                # Plot bands
                ax3.mean_line, = ax3.plot(rolling_mean.index, rolling_mean, color='blue', 
                                         linestyle='--', alpha=0.7, label='Mean')
                ax3.band1 = ax3.fill_between(rolling_mean.index, 
                                            rolling_mean - rolling_std, 
                                            rolling_mean + rolling_std, 
                                            color='gray', alpha=0.2)
                ax3.band2 = ax3.fill_between(rolling_mean.index, 
                                            rolling_mean - 2*rolling_std, 
                                            rolling_mean + 2*rolling_std, 
                                            color='gray', alpha=0.1)
                ax3.bands = True
                ax3.legend()
        else:
            if hasattr(ax3, 'bands') and ax3.bands:
                ax3.mean_line.remove()
                ax3.band1.remove()
                ax3.band2.remove()
                ax3.bands = False
                ax3.legend()
        
        # Update z-score legend
        ax4.legend()
        
        fig.canvas.draw_idle()
    
    # Connect the update function to slider and checkbox events
    price_slider.on_changed(update)
    smooth_slider.on_changed(update)
    thresh_slider.on_changed(update)
    check_buttons.on_clicked(update)
    
    # Initialize with attributes
    ax2.intercept_line = None
    ax3.bands = False
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12, hspace=0.5)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_detection(
    spread: pd.Series,
    window_size: int = 60,
    n_regimes: int = 3,
    lookback: int = 252,
    regime_method: str = 'hmm',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Detect and visualize different regimes in spread behavior.
    
    This function implements regime detection using either Hidden Markov Models (HMM)
    or K-means clustering, and visualizes the regimes alongside the spread.
    
    Parameters
    ----------
    spread : pd.Series
        The spread time series
    window_size : int, default=60
        Rolling window size for feature calculation
    n_regimes : int, default=3
        Number of regimes to detect
    lookback : int, default=252
        Lookback period for regime classification
    regime_method : str, default='hmm'
        Method for regime detection: 'hmm' (Hidden Markov Model) or 'kmeans'
    figsize : tuple, default=(14, 10)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        try:
            from hmmlearn import hmm
            hmm_available = True
        except ImportError:
            hmm_available = False
            if regime_method == 'hmm':
                print("hmmlearn not available, falling back to K-means clustering")
                regime_method = 'kmeans'
    except ImportError as e:
        print(f"Required module not found: {e}")
        return None
    
    # Create features for regime detection
    features = pd.DataFrame(index=spread.index)
    
    # Add features based on spread behavior
    features['volatility'] = spread.rolling(window=window_size).std()
    features['trend'] = spread.rolling(window=window_size).mean()
    features['momentum'] = spread.diff(window_size)
    features['mean_reversion'] = spread - spread.rolling(window=window_size).mean()
    features['zscore'] = features['mean_reversion'] / features['volatility']
    
    # Calculate autocorrelation
    features['autocorr'] = spread.rolling(window=window_size).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else np.nan
    )
    
    # Drop NaN values
    features = features.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Detect regimes
    regimes = pd.Series(index=features.index)
    
    if regime_method == 'hmm' and hmm_available:
        # Use Hidden Markov Model for regime detection
        model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
        model.fit(features_scaled)
        regimes_values = model.predict(features_scaled)
        regimes.loc[:] = regimes_values
    else:
        # Use K-means clustering for regime detection
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes_values = kmeans.fit_predict(features_scaled)
        regimes.loc[:] = regimes_values
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1, 1])
    
    # Spread plot with regimes highlighted
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(spread.index, spread, label='Spread', color='blue')
    
    # Color-code the regimes
    colors = ['green', 'red', 'orange', 'purple', 'brown'][:n_regimes]
    regime_names = [f'Regime {i+1}' for i in range(n_regimes)]
    
    # Highlight regime periods
    prev_regime = None
    start_idx = None
    
    for idx, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None:
                # End of a regime period, highlight it
                ax1.axvspan(start_idx, idx, alpha=0.2, color=colors[int(prev_regime)], 
                           label=regime_names[int(prev_regime)] if start_idx == regimes.index[0] else "")
            # Start of a new regime period
            start_idx = idx
            prev_regime = regime
    
    # Highlight the last regime period
    if prev_regime is not None and start_idx is not None:
        ax1.axvspan(start_idx, regimes.index[-1], alpha=0.2, color=colors[int(prev_regime)], 
                   label=regime_names[int(prev_regime)] if start_idx == regimes.index[0] else "")
    
    ax1.set_title('Spread with Regime Detection')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Regime plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for i in range(n_regimes):
        ax2.plot(regimes.index, (regimes == i).astype(int), label=regime_names[i], 
                color=colors[i], linewidth=1.5)
    ax2.set_title('Regime Classification')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Off', 'On'])
    ax2.legend()
    
    # Feature importance plot
    ax3 = fig.add_subplot(gs[2])
    feature_names = features.columns
    
    if regime_method == 'hmm' and hmm_available:
        # For HMM, use means of the different states as a proxy for feature importance
        feature_importance = np.std(model.means_, axis=0)
    else:
        # For K-means, use the distance between cluster centers as importance
        feature_importance = np.std(kmeans.cluster_centers_, axis=0)
    
    # Normalize feature importance
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Create bar plot
    bars = ax3.bar(range(len(feature_names)), feature_importance, color='skyblue')
    ax3.set_xticks(range(len(feature_names)))
    ax3.set_xticklabels(feature_names, rotation=45, ha='right')
    ax3.set_title('Feature Importance for Regime Detection')
    ax3.set_ylabel('Relative Importance')
    
    # Add regime characteristics table as text
    regime_stats = {}
    for i in range(n_regimes):
        regime_mask = regimes == i
        if regime_mask.sum() > 0:
            regime_stats[i] = {
                'mean': spread[regime_mask].mean(),
                'std': spread[regime_mask].std(),
                'count': regime_mask.sum(),
                'pct': 100 * regime_mask.sum() / len(regimes)
            }
    
    # Add text description
    stats_text = "Regime Statistics:\n"
    for i, stats in regime_stats.items():
        stats_text += f"{regime_names[i]}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, " \
                     f"Days={stats['count']}, ({stats['pct']:.1f}%)\n"
    
    fig.text(0.02, 0.01, stats_text, fontsize=9, verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, bottom=0.15)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_strategy_entry_exit_points(
    price1: pd.Series,
    price2: pd.Series,
    signals: pd.DataFrame,
    spread: Optional[pd.Series] = None,
    z_scores: Optional[pd.Series] = None,
    kalman_results: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize entry and exit points for a trading strategy on price series and spread.
    
    This function creates a comprehensive visualization of strategy execution showing:
    1. Price series of both assets with entry/exit points
    2. Spread with entry/exit points
    3. Z-scores with threshold crossings
    4. Position size over time
    
    Parameters
    ----------
    price1 : pd.Series
        First price series
    price2 : pd.Series
        Second price series
    signals : pd.DataFrame
        DataFrame with trading signals containing at minimum:
        - 'entry_long', 'exit_long' columns for long positions
        - 'entry_short', 'exit_short' columns for short positions
        - 'position' column for overall position size
    spread : pd.Series, optional
        Spread series. If None, will use signals index
    z_scores : pd.Series, optional
        Z-score series. If None, will use signals index
    kalman_results : pd.DataFrame, optional
        Results from Kalman filter analysis, if available
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Ensure we have aligned data
    common_index = price1.index.intersection(price2.index).intersection(signals.index)
    s1 = price1.loc[common_index]
    s2 = price2.loc[common_index]
    sig = signals.loc[common_index]
    
    # If spread and z_scores are not provided, create placeholder series with the right index
    if spread is None:
        if kalman_results is not None and 'spread' in kalman_results:
            spread = kalman_results['spread'].loc[common_index]
        else:
            spread = pd.Series(0, index=common_index)
    else:
        spread = spread.loc[common_index]
    
    if z_scores is None:
        if 'zscore' in sig.columns:
            z_scores = sig['zscore']
        else:
            # Calculate simple z-score if not provided
            window = min(20, len(spread) // 10)
            rolling_mean = spread.rolling(window=window).mean()
            rolling_std = spread.rolling(window=window).std()
            z_scores = (spread - rolling_mean) / rolling_std
    else:
        z_scores = z_scores.loc[common_index]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 0.5])
    
    # 1. Price series with entry/exit points
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(s1.index, s1, label=f'{price1.name if hasattr(price1, "name") else "Series 1"}')
    ax1.plot(s2.index, s2, label=f'{price2.name if hasattr(price2, "name") else "Series 2"}')
    
    # Add entry/exit points for price series
    long_entries = sig[sig['entry_long'] == 1].index
    long_exits = sig[sig['exit_long'] == 1].index
    short_entries = sig[sig['entry_short'] == 1].index
    short_exits = sig[sig['exit_short'] == 1].index
    
    # Plot markers on price series
    for entry in long_entries:
        ax1.plot(entry, s1.loc[entry], '^', color='green', markersize=10, alpha=0.7, label='Long Entry' if entry == long_entries[0] else "")
        ax1.plot(entry, s2.loc[entry], '^', color='green', markersize=10, alpha=0.7)
    
    for exit in long_exits:
        ax1.plot(exit, s1.loc[exit], 'v', color='blue', markersize=10, alpha=0.7, label='Long Exit' if exit == long_exits[0] else "")
        ax1.plot(exit, s2.loc[exit], 'v', color='blue', markersize=10, alpha=0.7)
    
    for entry in short_entries:
        ax1.plot(entry, s1.loc[entry], 'v', color='red', markersize=10, alpha=0.7, label='Short Entry' if entry == short_entries[0] else "")
        ax1.plot(entry, s2.loc[entry], 'v', color='red', markersize=10, alpha=0.7)
    
    for exit in short_exits:
        ax1.plot(exit, s1.loc[exit], '^', color='orange', markersize=10, alpha=0.7, label='Short Exit' if exit == short_exits[0] else "")
        ax1.plot(exit, s2.loc[exit], '^', color='orange', markersize=10, alpha=0.7)
    
    ax1.set_title('Price Series with Entry/Exit Points')
    # Only add legend entries if there are signals of that type
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. Spread with entry/exit points
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(spread.index, spread, label='Spread', color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add spread mean
    window = min(20, len(spread) // 10)
    rolling_mean = spread.rolling(window=window).mean()
    ax2.plot(rolling_mean.index, rolling_mean, 'r--', alpha=0.5, label=f'{window}-day Mean')
    
    # Add entry/exit points for spread
    for entry in long_entries:
        ax2.plot(entry, spread.loc[entry], '^', color='green', markersize=10, alpha=0.7)
    
    for exit in long_exits:
        ax2.plot(exit, spread.loc[exit], 'v', color='blue', markersize=10, alpha=0.7)
    
    for entry in short_entries:
        ax2.plot(entry, spread.loc[entry], 'v', color='red', markersize=10, alpha=0.7)
    
    for exit in short_exits:
        ax2.plot(exit, spread.loc[exit], '^', color='orange', markersize=10, alpha=0.7)
    
    ax2.set_title('Spread with Entry/Exit Points')
    ax2.legend()
    
    # 3. Z-scores with threshold lines
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(z_scores.index, z_scores, label='Z-Score', color='blue')
    
    # Add threshold lines
    entry_threshold = 2.0
    exit_threshold = 0.5
    
    # Try to extract thresholds from signals if provided
    if hasattr(signals, 'attrs') and 'entry_threshold' in signals.attrs:
        entry_threshold = signals.attrs['entry_threshold']
    if hasattr(signals, 'attrs') and 'exit_threshold' in signals.attrs:
        exit_threshold = signals.attrs['exit_threshold']
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=entry_threshold, color='red', linestyle='--', alpha=0.8, label=f'Entry (+{entry_threshold})')
    ax3.axhline(y=-entry_threshold, color='green', linestyle='--', alpha=0.8, label=f'Entry (-{entry_threshold})')
    ax3.axhline(y=exit_threshold, color='orange', linestyle=':', alpha=0.8, label=f'Exit (+{exit_threshold})')
    ax3.axhline(y=-exit_threshold, color='cyan', linestyle=':', alpha=0.8, label=f'Exit (-{exit_threshold})')
    
    # Add entry/exit points for z-scores
    for entry in long_entries:
        ax3.plot(entry, z_scores.loc[entry], '^', color='green', markersize=10, alpha=0.7)
    
    for exit in long_exits:
        ax3.plot(exit, z_scores.loc[exit], 'v', color='blue', markersize=10, alpha=0.7)
    
    for entry in short_entries:
        ax3.plot(entry, z_scores.loc[entry], 'v', color='red', markersize=10, alpha=0.7)
    
    for exit in short_exits:
        ax3.plot(exit, z_scores.loc[exit], '^', color='orange', markersize=10, alpha=0.7)
    
    ax3.set_title('Z-Scores with Entry/Exit Thresholds')
    ax3.legend()
    
    # 4. Position size over time
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    # Use 'position' column if available, otherwise construct from signals
    if 'position' in sig.columns:
        position = sig['position']
    else:
        position = pd.Series(0, index=sig.index)
        pos = 0
        for i, row in sig.iterrows():
            if row['entry_long'] == 1:
                pos = 1
            elif row['entry_short'] == 1:
                pos = -1
            elif (row['exit_long'] == 1 and pos == 1) or (row['exit_short'] == 1 and pos == -1):
                pos = 0
            position.loc[i] = pos
    
    # Create step plot for position
    ax4.step(position.index, position, 'k-', label='Position', where='post')
    ax4.fill_between(position.index, 0, position, step='post', alpha=0.2, 
                    color='green', where=(position > 0))
    ax4.fill_between(position.index, 0, position, step='post', alpha=0.2, 
                    color='red', where=(position < 0))
    
    ax4.set_title('Position Size')
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Short', 'Flat', 'Long'])
    ax4.legend()
    
    # Add trade statistics
    long_trades = []
    short_trades = []
    current_long = None
    current_short = None
    
    # Extract trades from signals
    for i, row in sig.iterrows():
        if row['entry_long'] == 1:
            current_long = {'entry_date': i, 'entry_spread': spread.loc[i]}
        elif row['exit_long'] == 1 and current_long is not None:
            current_long['exit_date'] = i
            current_long['exit_spread'] = spread.loc[i]
            current_long['pnl'] = current_long['entry_spread'] - current_long['exit_spread']
            current_long['duration'] = (current_long['exit_date'] - current_long['entry_date']).days
            long_trades.append(current_long)
            current_long = None
        
        if row['entry_short'] == 1:
            current_short = {'entry_date': i, 'entry_spread': spread.loc[i]}
        elif row['exit_short'] == 1 and current_short is not None:
            current_short['exit_date'] = i
            current_short['exit_spread'] = spread.loc[i]
            current_short['pnl'] = current_short['exit_spread'] - current_short['entry_spread']
            current_short['duration'] = (current_short['exit_date'] - current_short['entry_date']).days
            short_trades.append(current_short)
            current_short = None
    
    # Calculate trade statistics
    n_long = len(long_trades)
    n_short = len(short_trades)
    
    if n_long > 0:
        long_win_rate = sum(1 for t in long_trades if t['pnl'] > 0) / n_long
        long_avg_pnl = sum(t['pnl'] for t in long_trades) / n_long
        long_avg_duration = sum(t['duration'] for t in long_trades) / n_long
    else:
        long_win_rate = long_avg_pnl = long_avg_duration = 0
    
    if n_short > 0:
        short_win_rate = sum(1 for t in short_trades if t['pnl'] > 0) / n_short
        short_avg_pnl = sum(t['pnl'] for t in short_trades) / n_short
        short_avg_duration = sum(t['duration'] for t in short_trades) / n_short
    else:
        short_win_rate = short_avg_pnl = short_avg_duration = 0
    
    # Add statistics text
    stats_text = f"Long Trades: {n_long}, Win Rate: {long_win_rate:.1%}, Avg PnL: {long_avg_pnl:.4f}, Avg Duration: {long_avg_duration:.1f} days\n" \
                f"Short Trades: {n_short}, Win Rate: {short_win_rate:.1%}, Avg PnL: {short_avg_pnl:.4f}, Avg Duration: {short_avg_duration:.1f} days"
    
    fig.text(0.02, 0.01, stats_text, fontsize=9, verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3, bottom=0.12)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_attribution(
    returns: pd.Series,
    signals: pd.DataFrame,
    z_scores: Optional[pd.Series] = None,
    market_returns: Optional[pd.Series] = None,
    regime_labels: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize performance attribution analysis for a trading strategy.
    
    This function creates a multi-panel plot showing:
    1. Cumulative returns with drawdown shading
    2. Returns attribution by regime or signal type
    3. Performance metrics breakdown
    4. Return distribution analysis
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns series
    signals : pd.DataFrame
        DataFrame with trading signals
    z_scores : pd.Series, optional
        Z-score series for additional analysis
    market_returns : pd.Series, optional
        Market returns for comparison
    regime_labels : pd.Series, optional
        Labels for market regimes to use in attribution
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])
    
    # 1. Cumulative returns with drawdown shading
    ax1 = fig.add_subplot(gs[0, :])
    cumulative_returns = (1 + returns).cumprod()
    ax1.plot(cumulative_returns.index, cumulative_returns, label='Strategy', linewidth=2)
    
    # Add market returns if provided
    if market_returns is not None:
        market_returns = market_returns.loc[returns.index]
        cumulative_market = (1 + market_returns).cumprod()
        ax1.plot(cumulative_market.index, cumulative_market, label='Market', 
                linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Highlight drawdowns
    previous_peak = cumulative_returns.iloc[0]
    drawdown_start = None
    
    for idx, value in cumulative_returns.items():
        if value > previous_peak:
            previous_peak = value
            if drawdown_start is not None:
                # End of drawdown
                ax1.axvspan(drawdown_start, idx, alpha=0.2, color='red')
                drawdown_start = None
        elif value < previous_peak and drawdown_start is None:
            # Start of drawdown
            drawdown_start = idx
    
    # If still in drawdown at the end of the series
    if drawdown_start is not None:
        ax1.axvspan(drawdown_start, cumulative_returns.index[-1], alpha=0.2, color='red')
    
    ax1.set_title('Cumulative Returns with Drawdowns')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.set_ylabel('Cumulative Return')
    
    # 2. Returns attribution by regime
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Use regime labels if provided, otherwise use position sign
    if regime_labels is not None:
        regimes = regime_labels.loc[returns.index]
    else:
        # Use position sign as regime
        if 'position' in signals.columns:
            position = signals['position'].loc[returns.index]
            regimes = pd.Series('Neutral', index=returns.index)
            regimes[position > 0] = 'Long'
            regimes[position < 0] = 'Short'
        else:
            # Default to a dummy regime
            regimes = pd.Series('All', index=returns.index)
    
    # Calculate returns by regime
    regime_returns = {}
    for regime in regimes.unique():
        regime_mask = regimes == regime
        if regime_mask.sum() > 0:
            regime_returns[regime] = returns[regime_mask].sum()
    
    # Create bar chart
    bars = ax2.bar(range(len(regime_returns)), 
                  list(regime_returns.values()), 
                  color=plt.cm.tab10.colors[:len(regime_returns)])
    ax2.set_xticks(range(len(regime_returns)))
    ax2.set_xticklabels(list(regime_returns.keys()), rotation=45, ha='right')
    ax2.set_title('Returns Attribution by Regime/Position')
    ax2.set_ylabel('Cumulative Return')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2%}', ha='center', va='bottom')
    
    # 3. Performance metrics
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate various performance metrics
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    
    # Calculate drawdowns
    drawdowns = 1 - cumulative_returns / cumulative_returns.cummax()
    max_drawdown = drawdowns.max()
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    daily_sharpe = returns.mean() / returns.std()
    annualized_sharpe = daily_sharpe * np.sqrt(252)
    
    # Calculate percent positive days
    pct_positive = (returns > 0).mean()
    
    # Calculate win/loss ratio
    if (returns < 0).sum() > 0:
        win_loss_ratio = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
    else:
        win_loss_ratio = float('inf')
    
    # Calculate maximum consecutive wins/losses
    streaks = (returns > 0).astype(int).diff().ne(0).cumsum()
    streak_groups = returns.groupby(streaks)
    max_win_streak = max([len(g) for k, g in streak_groups if (g > 0).all()], default=0)
    max_loss_streak = max([len(g) for k, g in streak_groups if (g < 0).all()], default=0)
    
    # Calculate correlation with market if provided
    if market_returns is not None:
        market_correlation = returns.corr(market_returns)
        beta = returns.cov(market_returns) / market_returns.var()
    else:
        market_correlation = None
        beta = None
    
    # Calculate volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Prepare metrics for display
    metrics = {
        'Total Return': f'{total_return:.2%}',
        'Annualized Return': f'{annualized_return:.2%}',
        'Volatility': f'{volatility:.2%}',
        'Sharpe Ratio': f'{annualized_sharpe:.2f}',
        'Max Drawdown': f'{max_drawdown:.2%}',
        'Win Rate': f'{pct_positive:.2%}',
        'Win/Loss Ratio': f'{win_loss_ratio:.2f}',
        'Max Win Streak': f'{max_win_streak}',
        'Max Loss Streak': f'{max_loss_streak}'
    }
    
    if market_correlation is not None:
        metrics['Market Correlation'] = f'{market_correlation:.2f}'
    if beta is not None:
        metrics['Beta'] = f'{beta:.2f}'
    
    # Create table-like display
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(
        cellText=[[v] for v in metrics.values()],
        rowLabels=list(metrics.keys()),
        colLabels=["Value"],
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0, 0.8, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax3.set_title('Performance Metrics')
    
    # 4. Return distribution analysis
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Plot return distribution histogram
    sns.histplot(returns, kde=True, ax=ax4)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=returns.mean(), color='green', linestyle='-', alpha=0.5, 
               label=f'Mean: {returns.mean():.2%}')
    
    ax4.set_title('Return Distribution')
    ax4.set_xlabel('Daily Return')
    ax4.legend()
    
    # 5. Drawdown chart
    ax5 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax5.fill_between(drawdowns.index, 0, -drawdowns, color='red', alpha=0.3)
    ax5.plot(drawdowns.index, -drawdowns, color='red')
    ax5.set_title('Drawdowns')
    ax5.set_ylabel('Drawdown (%)')
    ax5.set_ylim(bottom=-max(max_drawdown * 1.1, 0.01))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 