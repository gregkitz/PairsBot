import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime
import json


def create_directory(path):
    """
    Create a directory if it doesn't exist.
    
    Parameters:
    -----------
    path : str
        Path to directory
    """
    os.makedirs(path, exist_ok=True)


def save_results(results, filename, directory='output'):
    """
    Save results to a file.
    
    Parameters:
    -----------
    results : dict
        Results to save
    filename : str
        Filename to save as
    directory : str
        Directory to save in
    """
    # Create directory if it doesn't exist
    create_directory(directory)
    
    # Full path
    path = os.path.join(directory, filename)
    
    # Save based on file extension
    if filename.endswith('.json'):
        # Convert non-serializable objects to string
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                continue  # Skip pandas objects
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (datetime.datetime, datetime.date)):
                serializable_results[key] = value.isoformat()
            elif isinstance(value, dict):
                # Recursively check for non-serializable objects
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray, datetime.datetime, datetime.date)):
                        serializable_dict[k] = str(v)
                    else:
                        serializable_dict[k] = v
                serializable_results[key] = serializable_dict
            else:
                serializable_results[key] = value
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    elif filename.endswith('.csv'):
        # Assume the results contains a DataFrame or can be converted to one
        if isinstance(results, dict) and all(isinstance(v, (pd.DataFrame, pd.Series)) for v in results.values()):
            # Save each DataFrame to a separate file
            for key, df in results.items():
                df_path = os.path.join(directory, f"{key}_{filename}")
                df.to_csv(df_path)
        elif isinstance(results, (pd.DataFrame, pd.Series)):
            results.to_csv(path)
        else:
            pd.DataFrame(results).to_csv(path)
    
    elif filename.endswith('.pkl'):
        # Use pickle for pandas objects
        if isinstance(results, (pd.DataFrame, pd.Series)):
            results.to_pickle(path)
        else:
            pd.to_pickle(results, path)
    
    else:
        # Default to JSON
        with open(path, 'w') as f:
            if isinstance(results, dict):
                json.dump({k: str(v) for k, v in results.items()}, f, indent=4)
            else:
                f.write(str(results))


def load_results(filename, directory='output'):
    """
    Load results from a file.
    
    Parameters:
    -----------
    filename : str
        Filename to load from
    directory : str
        Directory to load from
        
    Returns:
    --------
    object
        Loaded results
    """
    # Full path
    path = os.path.join(directory, filename)
    
    # Load based on file extension
    if filename.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    
    elif filename.endswith('.csv'):
        return pd.read_csv(path, index_col=0)
    
    elif filename.endswith('.pkl'):
        return pd.read_pickle(path)
    
    else:
        # Default to text
        with open(path, 'r') as f:
            return f.read()


def format_number(number, precision=4):
    """
    Format a number for display.
    
    Parameters:
    -----------
    number : float
        Number to format
    precision : int
        Decimal precision
        
    Returns:
    --------
    str
        Formatted number
    """
    if isinstance(number, (int, float)):
        if number == 0:
            return '0'
        elif abs(number) < 0.001:
            return f"{number:.{precision}e}"
        else:
            return f"{number:.{precision}f}"
    else:
        return str(number)


def print_results_table(results, title=None):
    """
    Print a results table.
    
    Parameters:
    -----------
    results : dict
        Results to print
    title : str, optional
        Title for the table
    """
    if title:
        print(f"\n{title}")
        print("=" * len(title))
    
    if isinstance(results, dict):
        # Find the maximum key length for alignment
        max_key_len = max(len(str(key)) for key in results.keys())
        
        for key, value in results.items():
            # Format numbers, keep strings as is
            if isinstance(value, (int, float)):
                formatted_value = format_number(value)
            elif isinstance(value, dict):
                # For nested dictionaries, show a summary
                formatted_value = f"<dict with {len(value)} items>"
            elif isinstance(value, (list, tuple)):
                formatted_value = f"<{type(value).__name__} with {len(value)} items>"
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                formatted_value = f"<{type(value).__name__} with shape {value.shape}>"
            else:
                formatted_value = str(value)
            
            print(f"{str(key):{max_key_len}}: {formatted_value}")
    else:
        print(results)


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
    --------
    str
        Project root directory
    """
    # Assuming this file is in src/utils/utils.py
    return str(Path(__file__).resolve().parent.parent.parent)


def plot_comparison(series_list, labels=None, title=None, figsize=(12, 6)):
    """
    Plot multiple time series for comparison.
    
    Parameters:
    -----------
    series_list : list
        List of pandas Series to plot
    labels : list, optional
        List of labels for the series
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    tuple
        (fig, ax) matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(series_list))]
    
    for i, series in enumerate(series_list):
        ax.plot(series, label=labels[i])
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True)
    
    return fig, ax 