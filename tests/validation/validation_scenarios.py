"""
Validation scenarios for paper trading testing.

This module generates different market scenarios for comprehensive
validation of the MVTS (Minimal Viable Trading Strategy) performance.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Scenario types
SCENARIO_TRENDING = "trending"
SCENARIO_MEAN_REVERTING = "mean_reverting"
SCENARIO_VOLATILE = "volatile"
SCENARIO_BREAKOUT = "breakout"
SCENARIO_REGIME_CHANGE = "regime_change"
SCENARIO_STRESSED = "stressed"


class ValidationScenarioGenerator:
    """
    Generate various market scenarios for comprehensive trading strategy validation.
    
    This class creates synthetic price data with different statistical properties
    to test how the trading strategy performs under various market conditions.
    """
    
    def __init__(self, output_dir, start_date=None, days=30, timeframe="1h"):
        """
        Initialize the scenario generator.
        
        Args:
            output_dir (str): Directory to save generated scenario data
            start_date (datetime, optional): Start date for scenarios. Defaults to 30 days ago.
            days (int, optional): Number of days to generate. Defaults to 30.
            timeframe (str, optional): Data timeframe. Defaults to "1h".
        """
        self.output_dir = output_dir
        self.start_date = start_date or (datetime.now() - timedelta(days=days))
        self.days = days
        self.timeframe = timeframe
        
        # Create scenario directory
        self.scenario_dir = os.path.join(output_dir, "scenarios")
        os.makedirs(self.scenario_dir, exist_ok=True)
        
        # Initialize scenario configuration
        self.scenario_config = {}
    
    def _generate_dates(self, freq="1h"):
        """Generate date range for scenarios."""
        periods = self.days * 24 if freq == "1h" else self.days
        return pd.date_range(start=self.start_date, periods=periods, freq=freq)
    
    def generate_trending_scenario(self, symbol1="GC", symbol2="SI", 
                                  trend_strength=0.5, noise_level=0.2,
                                  cointegration_strength=0.9):
        """
        Generate a trending market scenario with cointegrated pairs.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            trend_strength (float): Strength of the trend (0-1)
            noise_level (float): Amount of noise to add (0-1)
            cointegration_strength (float): Strength of cointegration (0-1)
            
        Returns:
            dict: Dictionary with scenario data
        """
        dates = self._generate_dates(self.timeframe)
        n = len(dates)
        
        # Create trending price series with noise
        trend = np.linspace(0, trend_strength * 100, n)
        symbol1_prices = 1000 + trend + np.random.normal(0, noise_level * 20, n)
        
        # Create cointegrated series
        symbol2_prices = (symbol1_prices / 70) * cointegration_strength + \
                        (1 - cointegration_strength) * np.random.normal(15, noise_level * 5, n)
        
        # Create DataFrames
        symbol1_df = pd.DataFrame({'timestamp': dates, 'close': symbol1_prices})
        symbol2_df = pd.DataFrame({'timestamp': dates, 'close': symbol2_prices})
        
        # Save to files
        scenario_id = f"trending_{symbol1}_{symbol2}"
        symbol1_file = f"{symbol1}_trending_{self.timeframe}.csv"
        symbol2_file = f"{symbol2}_trending_{self.timeframe}.csv"
        
        symbol1_df.to_csv(os.path.join(self.scenario_dir, symbol1_file), index=False)
        symbol2_df.to_csv(os.path.join(self.scenario_dir, symbol2_file), index=False)
        
        # Save configuration
        self.scenario_config[scenario_id] = {
            "type": SCENARIO_TRENDING,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": self.timeframe,
            "parameters": {
                "trend_strength": trend_strength,
                "noise_level": noise_level,
                "cointegration_strength": cointegration_strength
            },
            "files": {
                "symbol1": symbol1_file,
                "symbol2": symbol2_file
            }
        }
        
        return {
            "scenario_id": scenario_id,
            "symbol1_data": symbol1_df,
            "symbol2_data": symbol2_df,
            "config": self.scenario_config[scenario_id]
        }
    
    def generate_mean_reverting_scenario(self, symbol1="GC", symbol2="SI",
                                        cycle_count=3, amplitude=50,
                                        noise_level=0.15, cointegration_strength=0.95):
        """
        Generate a mean-reverting market scenario.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            cycle_count (int): Number of cycles in the period
            amplitude (float): Amplitude of oscillations
            noise_level (float): Amount of noise to add (0-1)
            cointegration_strength (float): Strength of cointegration (0-1)
            
        Returns:
            dict: Dictionary with scenario data
        """
        dates = self._generate_dates(self.timeframe)
        n = len(dates)
        
        # Create oscillating price series with noise
        x = np.linspace(0, cycle_count * 2 * np.pi, n)
        symbol1_prices = 1000 + np.sin(x) * amplitude + np.random.normal(0, noise_level * 10, n)
        
        # Create cointegrated series
        symbol2_prices = (symbol1_prices / 70) * cointegration_strength + \
                        (1 - cointegration_strength) * np.random.normal(15, noise_level * 3, n)
        
        # Create DataFrames
        symbol1_df = pd.DataFrame({'timestamp': dates, 'close': symbol1_prices})
        symbol2_df = pd.DataFrame({'timestamp': dates, 'close': symbol2_prices})
        
        # Save to files
        scenario_id = f"mean_reverting_{symbol1}_{symbol2}"
        symbol1_file = f"{symbol1}_mean_reverting_{self.timeframe}.csv"
        symbol2_file = f"{symbol2}_mean_reverting_{self.timeframe}.csv"
        
        symbol1_df.to_csv(os.path.join(self.scenario_dir, symbol1_file), index=False)
        symbol2_df.to_csv(os.path.join(self.scenario_dir, symbol2_file), index=False)
        
        # Save configuration
        self.scenario_config[scenario_id] = {
            "type": SCENARIO_MEAN_REVERTING,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": self.timeframe,
            "parameters": {
                "cycle_count": cycle_count,
                "amplitude": amplitude,
                "noise_level": noise_level,
                "cointegration_strength": cointegration_strength
            },
            "files": {
                "symbol1": symbol1_file,
                "symbol2": symbol2_file
            }
        }
        
        return {
            "scenario_id": scenario_id,
            "symbol1_data": symbol1_df,
            "symbol2_data": symbol2_df,
            "config": self.scenario_config[scenario_id]
        }
    
    def generate_volatile_scenario(self, symbol1="GC", symbol2="SI",
                                  volatility=2.0, jumps=True,
                                  cointegration_strength=0.8):
        """
        Generate a volatile market scenario with price jumps.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            volatility (float): Volatility multiplier
            jumps (bool): Whether to include price jumps
            cointegration_strength (float): Strength of cointegration (0-1)
            
        Returns:
            dict: Dictionary with scenario data
        """
        dates = self._generate_dates(self.timeframe)
        n = len(dates)
        
        # Create random walk with high volatility
        random_changes = np.random.normal(0, volatility * 5, n)
        
        # Add occasional jumps if requested
        if jumps:
            jump_points = np.random.randint(0, n, 5)  # 5 random jump points
            jump_sizes = np.random.choice([-30, -20, 20, 30], 5)
            
            for point, size in zip(jump_points, jump_sizes):
                random_changes[point] += size
        
        # Cumulative sum to create a price series
        symbol1_prices = 1000 + np.cumsum(random_changes)
        
        # Create cointegrated series with varying relationship
        symbol2_prices = (symbol1_prices / 70) * cointegration_strength + \
                        (1 - cointegration_strength) * np.random.normal(15, volatility * 2, n)
        
        # Create DataFrames
        symbol1_df = pd.DataFrame({'timestamp': dates, 'close': symbol1_prices})
        symbol2_df = pd.DataFrame({'timestamp': dates, 'close': symbol2_prices})
        
        # Save to files
        scenario_id = f"volatile_{symbol1}_{symbol2}"
        symbol1_file = f"{symbol1}_volatile_{self.timeframe}.csv"
        symbol2_file = f"{symbol2}_volatile_{self.timeframe}.csv"
        
        symbol1_df.to_csv(os.path.join(self.scenario_dir, symbol1_file), index=False)
        symbol2_df.to_csv(os.path.join(self.scenario_dir, symbol2_file), index=False)
        
        # Save configuration
        self.scenario_config[scenario_id] = {
            "type": SCENARIO_VOLATILE,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": self.timeframe,
            "parameters": {
                "volatility": volatility,
                "jumps": jumps,
                "cointegration_strength": cointegration_strength
            },
            "files": {
                "symbol1": symbol1_file,
                "symbol2": symbol2_file
            }
        }
        
        return {
            "scenario_id": scenario_id,
            "symbol1_data": symbol1_df,
            "symbol2_data": symbol2_df,
            "config": self.scenario_config[scenario_id]
        }
    
    def generate_regime_change_scenario(self, symbol1="GC", symbol2="SI",
                                       num_regimes=3, cointegration_changes=True):
        """
        Generate a scenario with regime changes.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            num_regimes (int): Number of different regimes
            cointegration_changes (bool): Whether cointegration changes between regimes
            
        Returns:
            dict: Dictionary with scenario data
        """
        dates = self._generate_dates(self.timeframe)
        n = len(dates)
        
        # Create regime change points
        regime_points = [0] + sorted(np.random.choice(range(1, n-1), num_regimes-1, replace=False)) + [n]
        
        # Initialize price arrays
        symbol1_prices = np.zeros(n)
        symbol2_prices = np.zeros(n)
        
        # Create different regimes
        regimes_info = []
        
        for i in range(num_regimes):
            start_idx = regime_points[i]
            end_idx = regime_points[i+1]
            regime_len = end_idx - start_idx
            
            # Randomly select regime type
            regime_type = np.random.choice(["trending", "mean_reverting", "volatile"])
            
            # Different parameters for different regimes
            if regime_type == "trending":
                trend_strength = np.random.uniform(0.3, 0.7)
                trend = np.linspace(0, trend_strength * 50, regime_len)
                regime_prices = 1000 + trend + np.random.normal(0, 5, regime_len)
            
            elif regime_type == "mean_reverting":
                amplitude = np.random.uniform(20, 50)
                x = np.linspace(0, 2 * np.pi, regime_len)
                regime_prices = 1000 + np.sin(x) * amplitude + np.random.normal(0, 3, regime_len)
            
            else:  # volatile
                volatility = np.random.uniform(1.5, 3.0)
                changes = np.random.normal(0, volatility * 5, regime_len)
                regime_prices = 1000 + np.cumsum(changes)
            
            # Set price level to continue from previous regime
            if i > 0:
                level_adjustment = symbol1_prices[start_idx-1] - regime_prices[0]
                regime_prices += level_adjustment
            
            # Set prices for this regime
            symbol1_prices[start_idx:end_idx] = regime_prices
            
            # Cointegration strength varies between regimes if requested
            if cointegration_changes:
                cointegration_strength = np.random.uniform(0.7, 0.98)
            else:
                cointegration_strength = 0.9
                
            # Create cointegrated prices for this regime
            symbol2_regime_prices = (regime_prices / 70) * cointegration_strength + \
                                   (1 - cointegration_strength) * np.random.normal(15, 2, regime_len)
            
            # Set price level to continue from previous regime
            if i > 0:
                level_adjustment = symbol2_prices[start_idx-1] - symbol2_regime_prices[0]
                symbol2_regime_prices += level_adjustment
            
            # Set prices for this regime
            symbol2_prices[start_idx:end_idx] = symbol2_regime_prices
            
            # Store regime info
            regimes_info.append({
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "type": regime_type,
                "cointegration_strength": float(cointegration_strength)
            })
        
        # Create DataFrames
        symbol1_df = pd.DataFrame({'timestamp': dates, 'close': symbol1_prices})
        symbol2_df = pd.DataFrame({'timestamp': dates, 'close': symbol2_prices})
        
        # Save to files
        scenario_id = f"regime_change_{symbol1}_{symbol2}"
        symbol1_file = f"{symbol1}_regime_change_{self.timeframe}.csv"
        symbol2_file = f"{symbol2}_regime_change_{self.timeframe}.csv"
        
        symbol1_df.to_csv(os.path.join(self.scenario_dir, symbol1_file), index=False)
        symbol2_df.to_csv(os.path.join(self.scenario_dir, symbol2_file), index=False)
        
        # Save configuration
        self.scenario_config[scenario_id] = {
            "type": SCENARIO_REGIME_CHANGE,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": self.timeframe,
            "parameters": {
                "num_regimes": num_regimes,
                "cointegration_changes": cointegration_changes,
                "regimes": regimes_info
            },
            "files": {
                "symbol1": symbol1_file,
                "symbol2": symbol2_file
            }
        }
        
        return {
            "scenario_id": scenario_id,
            "symbol1_data": symbol1_df,
            "symbol2_data": symbol2_df,
            "config": self.scenario_config[scenario_id],
            "regimes": regimes_info
        }
    
    def generate_stressed_scenario(self, symbol1="GC", symbol2="SI",
                                  crash_magnitude=0.15, recovery=True,
                                  cointegration_breakdown=True):
        """
        Generate a stressed market scenario with crashes and potential recovery.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            crash_magnitude (float): Size of market crash as percentage
            recovery (bool): Whether market recovers after crash
            cointegration_breakdown (bool): Whether cointegration breaks during stress
            
        Returns:
            dict: Dictionary with scenario data
        """
        dates = self._generate_dates(self.timeframe)
        n = len(dates)
        
        # Generate normal prices for first half
        first_half = n // 2
        symbol1_prices_first = 1000 + np.linspace(0, 50, first_half) + np.random.normal(0, 5, first_half)
        
        # Generate crash and recovery for second half
        crash_start = first_half
        crash_duration = n // 10  # 10% of the time period
        
        # Create crash
        crash = np.linspace(0, -crash_magnitude * 1000, crash_duration)
        
        # Second half with crash
        second_half = n - first_half
        symbol1_prices_second = np.zeros(second_half)
        symbol1_prices_second[:crash_duration] = crash
        
        # Add recovery if specified
        if recovery:
            recovery_start = crash_duration
            recovery_duration = second_half - crash_duration
            recovery = np.linspace(-crash_magnitude * 1000, 0, recovery_duration)
            symbol1_prices_second[recovery_start:] = recovery
        else:
            # Stay at depressed level
            symbol1_prices_second[crash_duration:] = -crash_magnitude * 1000
        
        # Add some noise
        symbol1_prices_second += np.random.normal(0, 10, second_half)
        
        # Combine first and second half
        symbol1_prices = np.concatenate([
            symbol1_prices_first, 
            symbol1_prices_first[-1] + symbol1_prices_second
        ])
        
        # Create cointegrated series with breakdown during crash if specified
        symbol2_prices = np.zeros(n)
        
        # First half - normal cointegration
        symbol2_prices[:first_half] = (symbol1_prices[:first_half] / 70) * 0.95 + \
                                    0.05 * np.random.normal(15, 2, first_half)
        
        # Second half - potential cointegration breakdown
        if cointegration_breakdown:
            # During crash, relationship breaks down
            cointegration_strength = np.linspace(0.95, 0.2, second_half)
            
            for i in range(second_half):
                cs = cointegration_strength[i]
                symbol2_prices[first_half + i] = (symbol1_prices[first_half + i] / 70) * cs + \
                                              (1 - cs) * np.random.normal(15, 5, 1)[0]
        else:
            # Maintain cointegration through crash
            symbol2_prices[first_half:] = (symbol1_prices[first_half:] / 70) * 0.95 + \
                                        0.05 * np.random.normal(15, 2, second_half)
        
        # Create DataFrames
        symbol1_df = pd.DataFrame({'timestamp': dates, 'close': symbol1_prices})
        symbol2_df = pd.DataFrame({'timestamp': dates, 'close': symbol2_prices})
        
        # Save to files
        scenario_id = f"stressed_{symbol1}_{symbol2}"
        symbol1_file = f"{symbol1}_stressed_{self.timeframe}.csv"
        symbol2_file = f"{symbol2}_stressed_{self.timeframe}.csv"
        
        symbol1_df.to_csv(os.path.join(self.scenario_dir, symbol1_file), index=False)
        symbol2_df.to_csv(os.path.join(self.scenario_dir, symbol2_file), index=False)
        
        # Save configuration
        self.scenario_config[scenario_id] = {
            "type": SCENARIO_STRESSED,
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": self.timeframe,
            "parameters": {
                "crash_magnitude": crash_magnitude,
                "recovery": recovery,
                "cointegration_breakdown": cointegration_breakdown,
                "crash_start_idx": first_half,
                "crash_duration": crash_duration
            },
            "files": {
                "symbol1": symbol1_file,
                "symbol2": symbol2_file
            }
        }
        
        return {
            "scenario_id": scenario_id,
            "symbol1_data": symbol1_df,
            "symbol2_data": symbol2_df,
            "config": self.scenario_config[scenario_id]
        }
        
    def generate_all_scenarios(self, symbol1="GC", symbol2="SI"):
        """
        Generate all validation scenarios for a given pair.
        
        Args:
            symbol1 (str): First symbol name
            symbol2 (str): Second symbol name
            
        Returns:
            dict: Dictionary with all scenario data
        """
        scenarios = {}
        
        # Generate all scenario types
        scenarios["trending"] = self.generate_trending_scenario(symbol1, symbol2)
        scenarios["mean_reverting"] = self.generate_mean_reverting_scenario(symbol1, symbol2)
        scenarios["volatile"] = self.generate_volatile_scenario(symbol1, symbol2)
        scenarios["regime_change"] = self.generate_regime_change_scenario(symbol1, symbol2)
        scenarios["stressed"] = self.generate_stressed_scenario(symbol1, symbol2)
        
        # Save overall configuration
        with open(os.path.join(self.scenario_dir, "scenario_config.json"), "w") as f:
            # Convert numpy values to native Python types for JSON serialization
            config_json = json.dumps(self.scenario_config, default=lambda x: float(x) if isinstance(x, np.number) else x)
            f.write(config_json)
        
        return scenarios
    

# Test code to generate and visualize scenarios
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create scenario generator
        generator = ValidationScenarioGenerator(temp_dir)
        
        # Generate all scenarios
        scenarios = generator.generate_all_scenarios()
        
        # Plot the scenarios
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        
        for i, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]
            
            # Plot both symbols
            symbol1_data = scenario_data["symbol1_data"]
            symbol2_data = scenario_data["symbol2_data"]
            
            ax1.plot(symbol1_data["close"], label="GC")
            ax1.set_title(f"{scenario_name} - GC")
            ax1.legend()
            
            ax2.plot(symbol2_data["close"], label="SI", color="orange")
            ax2.set_title(f"{scenario_name} - SI")
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(temp_dir, "scenarios.png"))
        
        print(f"Scenarios generated and saved to {temp_dir}")
        print(f"Configuration saved to {os.path.join(temp_dir, 'scenarios', 'scenario_config.json')}")
        print(f"Visualization saved to {os.path.join(temp_dir, 'scenarios.png')}") 