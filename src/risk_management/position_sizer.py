"""
Position sizer module for the Intraday Statistical Arbitrage System.

This module implements position sizing functionality.
"""

from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Position sizer class for calculating appropriate position sizes.
    """
    
    def __init__(self, max_position_size: float = 0.2, 
                volatility_lookback: int = 20):
        """
        Initialize the position sizer.
        
        Parameters:
        -----------
        max_position_size : float, optional
            Maximum position size as a fraction of account value
        volatility_lookback : int, optional
            Lookback period for volatility calculation
        """
        self.max_position_size = max_position_size
        self.volatility_lookback = volatility_lookback
    
    def calculate_position_size(self, spread_volatility: float, 
                             account_value: float,
                             max_risk: float = 0.01) -> float:
        """
        Calculate position size based on volatility.
        
        Parameters:
        -----------
        spread_volatility : float
            Volatility of the spread
        account_value : float
            Current account value
        max_risk : float, optional
            Maximum risk per trade as a fraction of account value
            
        Returns:
        --------
        float
            Position size in currency units
        """
        # Calculate position size inversely proportional to volatility
        if spread_volatility <= 0:
            return 0
            
        # Base position size on account value and max risk
        position_size = (account_value * max_risk) / spread_volatility
        
        # Cap at max position size
        max_amount = account_value * self.max_position_size
        return min(position_size, max_amount)

    def calculate_kelly_position_size(self, spread_data, win_rate, profit_loss_ratio):
        """
        Calculate position size using Kelly Criterion.
        
        Parameters:
        -----------
        spread_data : pandas.Series
            Historical spread data
        win_rate : float
            Historical win rate (0.0 to 1.0)
        profit_loss_ratio : float
            Ratio of average win to average loss
            
        Returns:
        --------
        float
            Kelly position size as a fraction of account
        """
        # Calculate Kelly fraction
        kelly_fraction = (win_rate * profit_loss_ratio - (1 - win_rate)) / profit_loss_ratio
        
        # Apply half-Kelly for conservatism
        half_kelly = kelly_fraction / 2
        
        # Ensure maximum risk limit
        kelly_size = min(half_kelly, self.max_risk)
        
        # Ensure maximum allocation limit
        kelly_size = min(kelly_size, self.max_allocation)
        
        # Calculate the position size in account units
        position_size = self.account_size * kelly_size
        
        return position_size
    
    def monte_carlo_risk_analysis(self, spread_data, hedge_ratio, price_data, 
                                 num_simulations=1000, holding_period=180, 
                                 confidence_level=0.95):
        """
        Perform Monte Carlo simulation to estimate worst-case scenarios.
        
        Parameters:
        -----------
        spread_data : pandas.Series
            Historical spread data
        hedge_ratio : float
            Current hedge ratio
        price_data : pandas.DataFrame
            Price data for the assets in the pair
        num_simulations : int
            Number of Monte Carlo simulations to run
        holding_period : int
            Maximum holding period in minutes
        confidence_level : float
            Confidence level for risk metrics (e.g., 0.95 for 95%)
            
        Returns:
        --------
        dict
            Dictionary with risk metrics (VaR, maximum drawdown, etc.)
        """
        # Fit parameters to the spread using Ornstein-Uhlenbeck process
        lag_spread = spread_data.shift(1)
        delta_spread = spread_data - lag_spread
        lag_spread = lag_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        X = pd.DataFrame({'lag_spread': lag_spread})
        X = sm.add_constant(X)
        model = sm.OLS(delta_spread, X).fit()
        
        # Extract Ornstein-Uhlenbeck parameters
        mu = -model.params[0] / model.params[1]  # Long-run mean
        theta = -model.params[1]                 # Mean reversion speed
        sigma = np.sqrt(model.mse_resid)         # Volatility
        
        # Run simulations
        simulations = []
        for _ in range(num_simulations):
            sim_path = self._simulate_ou_process(
                s0=spread_data.iloc[-1],
                theta=theta,
                mu=mu,
                sigma=sigma,
                n_steps=holding_period,
                dt=1/60  # Assuming 1-minute data
            )
            simulations.append(sim_path)
        
        # Convert to DataFrame for analysis
        sim_df = pd.DataFrame(simulations)
        
        # Calculate spread impact on P&L
        latest_prices = price_data.iloc[-1]
        pair_value = latest_prices[0] + (hedge_ratio * latest_prices[1])
        spread_value_impact = pair_value * sim_df
        
        # Calculate risk metrics
        var_95 = np.percentile(spread_value_impact.min(axis=1), 100 - confidence_level * 100)
        var_99 = np.percentile(spread_value_impact.min(axis=1), 1)
        
        # Expected shortfall (average of losses beyond VaR)
        expected_shortfall = spread_value_impact.min(axis=1)[
            spread_value_impact.min(axis=1) <= var_95
        ].mean()
        
        # Probability of various outcomes
        probability_loss = (spread_value_impact.iloc[:, -1] < 0).mean()
        probability_stop_hit = (spread_value_impact.min(axis=1) < -self.max_risk * self.account_size).mean()
        
        risk_metrics = {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'max_favorable': np.percentile(spread_value_impact.max(axis=1), 95),
            'probability_loss': probability_loss,
            'probability_stop_hit': probability_stop_hit
        }
        
        return risk_metrics
    
    def _simulate_ou_process(self, s0, theta, mu, sigma, n_steps, dt=1.0):
        """
        Simulate Ornstein-Uhlenbeck process for a spread.
        
        Parameters:
        -----------
        s0 : float
            Initial spread value
        theta : float
            Mean reversion speed
        mu : float
            Long-run mean
        sigma : float
            Volatility
        n_steps : int
            Number of steps to simulate
        dt : float
            Time step size
            
        Returns:
        --------
        numpy.ndarray
            Simulated spread path
        """
        path = np.zeros(n_steps + 1)
        path[0] = s0
        
        for t in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = path[t-1] + theta * (mu - path[t-1]) * dt + sigma * dW
        
        return path 