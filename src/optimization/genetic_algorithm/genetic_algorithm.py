"""
Genetic Algorithm Optimization for Pairs Trading Strategy.

This module implements a genetic algorithm optimizer for finding optimal parameters
for the pairs trading strategy through evolutionary methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Any, Optional, Tuple
import time
import os
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pairs_trading_strategy import PairsTradingStrategy
from src.optimization.parameter_space import ParameterSpace, create_default_parameter_space


class GeneticOptimizer:
    """
    Genetic Algorithm Optimizer for Pairs Trading Strategy.
    
    This class implements a genetic algorithm approach to find optimal parameters
    for the pairs trading strategy by evolving generations of parameter settings.
    """
    
    def __init__(self, 
                param_space: Optional[ParameterSpace] = None,
                objective_function: Optional[Callable] = None,
                population_size: int = 50,
                n_generations: int = 10,
                elite_size: int = 5,
                mutation_rate: float = 0.2,
                crossover_rate: float = 0.8,
                tournament_size: int = 3,
                n_jobs: int = 1,
                verbose: bool = True):
        """
        Initialize the genetic algorithm optimizer.
        
        Parameters:
        -----------
        param_space : ParameterSpace, optional
            Parameter space for genetic algorithm. If None, uses default parameter space.
        objective_function : callable, optional
            Function to maximize. If None, uses Sharpe ratio.
        population_size : int
            Size of the population in each generation
        n_generations : int
            Number of generations to evolve
        elite_size : int
            Number of top individuals to preserve in each generation
        mutation_rate : float
            Probability of mutation for each parameter
        crossover_rate : float
            Probability of crossover (vs. keeping parents unchanged)
        tournament_size : int
            Size of tournaments for parent selection
        n_jobs : int
            Number of parallel jobs for evaluation
        verbose : bool
            Whether to print progress
        """
        self.param_space = param_space if param_space is not None else create_default_parameter_space()
        self.objective_function = objective_function if objective_function is not None else self._default_objective
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Results storage
        self.results = None
        self.best_solution = None
        self.generation_stats = []
    
    def _default_objective(self, strategy: PairsTradingStrategy) -> float:
        """
        Default objective function - Sharpe ratio.
        
        Parameters:
        -----------
        strategy : PairsTradingStrategy
            The strategy instance with results from backtesting
        
        Returns:
        --------
        float
            Sharpe ratio
        """
        if not hasattr(strategy, 'performance_metrics') or not strategy.performance_metrics:
            return -np.inf
        
        metrics = strategy.performance_metrics
        return metrics.get('sharpe_ratio', -np.inf)
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """
        Initialize a random population of parameter configurations.
        
        Returns:
        --------
        list
            List of parameter configurations
        """
        population = []
        for _ in range(self.population_size):
            individual = self.param_space.sample()
            population.append(individual)
        
        return population
    
    def _tournament_selection(self, 
                            population: List[Dict[str, Any]], 
                            fitness_values: List[float]) -> Dict[str, Any]:
        """
        Select an individual using tournament selection.
        
        Parameters:
        -----------
        population : list
            List of parameter configurations
        fitness_values : list
            List of fitness values for each individual
        
        Returns:
        --------
        dict
            Selected individual
        """
        # Select random individuals for tournament
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        # Find the winner (highest fitness)
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform crossover between two parents to create a child.
        
        Parameters:
        -----------
        parent1 : dict
            First parent
        parent2 : dict
            Second parent
        
        Returns:
        --------
        dict
            Child configuration
        """
        # If no crossover, return a copy of parent1
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = {}
        for param_name in parent1.keys():
            # Perform crossover
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual.
        
        Parameters:
        -----------
        individual : dict
            Individual to mutate
        
        Returns:
        --------
        dict
            Mutated individual
        """
        mutated = individual.copy()
        for param_name in individual.keys():
            # Apply mutation with some probability
            if random.random() < self.mutation_rate:
                param_type = self.param_space.parameter_types[param_name]
                
                if param_type == 'discrete':
                    values = self.param_space.parameters[param_name]
                    mutated[param_name] = random.choice(values)
                
                elif param_type == 'continuous':
                    low, high = self.param_space.parameters[param_name]
                    # Add small Gaussian noise
                    delta = (high - low) * np.random.normal(0, 0.1)
                    new_value = individual[param_name] + delta
                    # Ensure within bounds
                    new_value = max(low, min(high, new_value))
                    mutated[param_name] = new_value
                
                elif param_type == 'integer':
                    low, high = self.param_space.parameters[param_name]
                    # Randomly choose a new integer in the range
                    mutated[param_name] = random.randint(low, high)
        
        return mutated
    
    def optimize(self, 
                pairs: List[Tuple[str, str, float]],
                start_date: str,
                end_date: str,
                timeframe: str = '1hour',
                commission: float = 2.0,
                slippage: float = 1.0,
                account_size: float = 25000) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Parameters:
        -----------
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for backtest in 'YYYY-MM-DD' format
        end_date : str
            End date for backtest in 'YYYY-MM-DD' format
        timeframe : str
            Timeframe for data ('1min', '5min', '1hour', etc.)
        commission : float
            Commission per contract in currency units
        slippage : float
            Slippage per contract in currency units
        account_size : float
            Initial account size
            
        Returns:
        --------
        dict
            Best parameter configuration found
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"Genetic Algorithm Optimization")
            print(f"Population size: {self.population_size}")
            print(f"Number of generations: {self.n_generations}")
            print(f"Elite size: {self.elite_size}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Crossover rate: {self.crossover_rate}")
            print(f"Running with {self.n_jobs} parallel jobs")
            print("=" * 50)
        
        # Initialize population
        population = self._initialize_population()
        
        # Store results from all generations
        all_results = []
        
        # Evolution loop
        for generation in range(self.n_generations):
            gen_start_time = time.time()
            
            if self.verbose:
                print(f"\nGeneration {generation+1}/{self.n_generations}")
            
            # Evaluate current population
            fitness_values = self._evaluate_population(
                population=population,
                pairs=pairs,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                commission=commission,
                slippage=slippage,
                account_size=account_size
            )
            
            # Store results
            generation_results = []
            for i, individual in enumerate(population):
                result = {**individual, 'fitness': fitness_values[i], 'generation': generation + 1}
                generation_results.append(result)
                all_results.append(result)
            
            # Find best solution in this generation
            best_idx = np.argmax(fitness_values)
            best_fitness = fitness_values[best_idx]
            best_individual = population[best_idx]
            
            # Store generation statistics
            gen_stats = {
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_values),
                'median_fitness': np.median(fitness_values),
                'std_fitness': np.std(fitness_values),
                'best_params': best_individual.copy()
            }
            self.generation_stats.append(gen_stats)
            
            if self.verbose:
                print(f"Best fitness: {best_fitness:.4f}")
                print(f"Mean fitness: {np.mean(fitness_values):.4f}")
                print(f"Best parameters: {best_individual}")
            
            # If this is the last generation, break
            if generation == self.n_generations - 1:
                break
            
            # Create next generation
            next_population = []
            
            # Elitism: Keep best individuals
            combined = list(zip(population, fitness_values))
            sorted_population = [x[0] for x in sorted(combined, key=lambda x: x[1], reverse=True)]
            next_population.extend(sorted_population[:self.elite_size])
            
            # Fill the rest of the population with offspring
            while len(next_population) < self.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Create child through crossover and mutation
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                
                next_population.append(child)
            
            # Update population
            population = next_population
            
            gen_end_time = time.time()
            if self.verbose:
                print(f"Generation completed in {gen_end_time - gen_start_time:.2f} seconds")
        
        # Convert all results to DataFrame
        self.results = pd.DataFrame(all_results)
        
        # Get best overall solution
        best_row = self.results.loc[self.results['fitness'].idxmax()]
        self.best_solution = {k: best_row[k] for k in best_row.index if k not in ['fitness', 'generation']}
        
        end_time = time.time()
        
        if self.verbose:
            print("\nOptimization completed")
            print(f"Total time: {end_time - start_time:.2f} seconds")
            print(f"Best fitness value: {best_row['fitness']:.4f}")
            print(f"Best parameters: {self.best_solution}")
        
        return self.best_solution
    
    def _evaluate_population(self, 
                            population: List[Dict[str, Any]],
                            pairs: List[Tuple[str, str, float]],
                            start_date: str,
                            end_date: str,
                            timeframe: str,
                            commission: float,
                            slippage: float,
                            account_size: float) -> List[float]:
        """
        Evaluate all individuals in the population.
        
        Parameters:
        -----------
        population : list
            List of parameter configurations to evaluate
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        timeframe : str
            Timeframe for data
        commission : float
            Commission per contract
        slippage : float
            Slippage per contract
        account_size : float
            Initial account size
            
        Returns:
        --------
        list
            List of fitness values
        """
        if self.n_jobs > 1:
            # Parallel evaluation
            fitness_values = []
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for params in population:
                    future = executor.submit(
                        self._evaluate_params,
                        params=params,
                        pairs=pairs,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe,
                        commission=commission,
                        slippage=slippage,
                        account_size=account_size
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    fitness_values.append(future.result())
        else:
            # Sequential evaluation
            fitness_values = []
            for i, params in enumerate(population):
                fitness = self._evaluate_params(
                    params=params,
                    pairs=pairs,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    commission=commission,
                    slippage=slippage,
                    account_size=account_size
                )
                fitness_values.append(fitness)
                
                if self.verbose and (i+1) % 10 == 0:
                    print(f"Evaluated {i+1}/{len(population)} individuals")
        
        return fitness_values
    
    def _evaluate_params(self, 
                        params: Dict[str, Any],
                        pairs: List[Tuple[str, str, float]],
                        start_date: str,
                        end_date: str,
                        timeframe: str,
                        commission: float,
                        slippage: float,
                        account_size: float) -> float:
        """
        Evaluate a single parameter set.
        
        Parameters:
        -----------
        params : dict
            Parameter configuration to evaluate
        pairs : list of tuples
            List of (ticker1, ticker2, hedge_ratio) tuples to trade
        start_date : str
            Start date for backtest
        end_date : str
            End date for backtest
        timeframe : str
            Timeframe for data
        commission : float
            Commission per contract
        slippage : float
            Slippage per contract
        account_size : float
            Initial account size
            
        Returns:
        --------
        float
            Fitness value
        """
        try:
            # Create and initialize strategy with parameters
            strategy = PairsTradingStrategy(account_size=account_size, **params)
            success = strategy.initialize_strategy(pairs=pairs, start_date=start_date, end_date=end_date, timeframe=timeframe)
            
            if not success:
                return -np.inf
            
            # Calculate spreads
            strategy.calculate_spreads(use_kalman=params.get('use_kalman', True))
            
            # Generate signals
            strategy.generate_signals()
            
            # Backtest strategy
            results = strategy.backtest_strategy(commission=commission, slippage=slippage)
            
            if results is None or not strategy.performance_metrics:
                return -np.inf
            
            # Evaluate objective
            return self.objective_function(strategy)
        
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating parameters {params}: {str(e)}")
            return -np.inf
    
    def plot_evolution(self, save_path=None):
        """
        Plot the evolution of fitness across generations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.generation_stats:
            print("No evolution data to plot")
            return
        
        stats_df = pd.DataFrame(self.generation_stats)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot best, mean, and std fitness
        ax.plot(stats_df['generation'], stats_df['best_fitness'], 'b-', label='Best Fitness')
        ax.plot(stats_df['generation'], stats_df['mean_fitness'], 'g--', label='Mean Fitness')
        
        # Plot standard deviation as a shaded area
        mean = stats_df['mean_fitness'].values
        std = stats_df['std_fitness'].values
        ax.fill_between(stats_df['generation'], mean - std, mean + std, alpha=0.2, color='g')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Objective Value)')
        ax.set_title('Evolution of Fitness Across Generations')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'evolution_plot.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Plot parameter evolution for best individual in each generation
        param_names = list(self.generation_stats[0]['best_params'].keys())
        n_params = len(param_names)
        
        fig, axs = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
        
        if n_params == 1:
            axs = [axs]
        
        for i, param in enumerate(param_names):
            ax = axs[i]
            
            # Extract parameter values from best individual in each generation
            param_values = [gen['best_params'][param] for gen in self.generation_stats]
            
            ax.plot(stats_df['generation'], param_values, marker='o')
            ax.set_xlabel('Generation')
            ax.set_ylabel(param)
            ax.set_title(f'Evolution of {param}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'parameter_evolution.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, directory=None):
        """
        Save optimization results to CSV files.
        
        Parameters:
        -----------
        directory : str, optional
            Directory to save results. If None, uses current directory.
        """
        if self.results is None or self.results.empty:
            print("No optimization results to save")
            return
        
        if directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = f"genetic_optimization_{timestamp}"
            os.makedirs(directory, exist_ok=True)
        
        # Save all evaluations
        self.results.to_csv(os.path.join(directory, 'all_evaluations.csv'), index=False)
        
        # Save generation statistics
        stats_df = pd.DataFrame(self.generation_stats)
        # Extract nested parameters
        for gen_idx, gen in enumerate(self.generation_stats):
            for param, value in gen['best_params'].items():
                stats_df.at[gen_idx, f'best_{param}'] = value
        
        # Drop the best_params column
        stats_df = stats_df.drop(columns=['best_params'])
        stats_df.to_csv(os.path.join(directory, 'generation_stats.csv'), index=False)
        
        # Save best solution
        best_df = pd.DataFrame([{**self.best_solution, 'fitness': self.results['fitness'].max()}])
        best_df.to_csv(os.path.join(directory, 'best_solution.csv'), index=False)
        
        if self.verbose:
            print(f"Optimization results saved to {directory}")


if __name__ == "__main__":
    # Example usage
    from src.data_processor.data_processor import DataProcessor
    from src.cointegration.pair_finder import PairFinder
    
    # Load data and find pairs
    data_processor = DataProcessor()
    pair_finder = PairFinder()
    
    # Define sample futures
    futures = ['ES', 'NQ', 'GC', 'SI', 'CL', 'NG']
    
    # Load data
    daily_data = data_processor.load_daily_data(futures, '2020-01-01', '2022-12-31')
    
    # Find pairs
    pairs_results = pair_finder.find_pairs(daily_data)
    
    # Extract pairs for optimization
    pairs = []
    for _, row in pairs_results.iterrows():
        pairs.append((row['ticker1'], row['ticker2'], row['hedge_ratio']))
    
    # Create optimizer with smaller population for testing
    optimizer = GeneticOptimizer(
        population_size=20,
        n_generations=5,
        n_jobs=4
    )
    
    # Run optimization
    best_params = optimizer.optimize(
        pairs=pairs[:2],  # Use top 2 pairs
        start_date='2021-01-01',
        end_date='2022-12-31',
        timeframe='1hour',
        account_size=50000
    )
    
    # Plot evolution
    optimizer.plot_evolution()
    
    # Save results
    optimizer.save_results() 