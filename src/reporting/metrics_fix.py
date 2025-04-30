"""
Patch to fix metrics calculation in the reporting module.

This script applies a patch to fix the issue with Series comparison in 
the metrics calculation code.
"""

import os
import sys
import re
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists, creating it if necessary."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {directory_path}")

def apply_patch():
    """Apply the patch to fix metrics calculation."""
    metrics_file = os.path.abspath(os.path.join('src', 'reporting', 'metrics.py'))
    
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file not found at {metrics_file}")
        return False
    
    # Read the file
    with open(metrics_file, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_dir = os.path.abspath('patches/backups')
    ensure_directory_exists(backup_dir)
    
    backup_file = os.path.join(backup_dir, 'metrics.py.bak')
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print(f"Created backup of metrics.py at {backup_file}")
    
    # Apply the patch - fix Series comparison in _calculate_equity_metrics function
    patched_content = re.sub(
        r'(\s+sharpe_ratio = excess_return / annualized_volatility) if annualized_volatility > 0 else 0',
        r'\1 if isinstance(annualized_volatility, (int, float)) and annualized_volatility > 0 else 0',
        content
    )
    
    patched_content = re.sub(
        r'(\s+sortino_ratio = excess_return / annualized_downside_volatility) if annualized_downside_volatility > 0 else 0',
        r'\1 if isinstance(annualized_downside_volatility, (int, float)) and annualized_downside_volatility > 0 else 0',
        patched_content
    )
    
    # Make Series-related operations safe throughout the module
    safer_ops = [
        # Fix Series comparison operators
        (r'(\w+)\.std\(\) ([<>=!]+) 0', r'float(\1.std() or 0) \2 0'),
        (r'(\w+)\.sum\(\) ([<>=!]+) 0', r'float(\1.sum() or 0) \2 0'),
        (r'(\w+)\.mean\(\) ([<>=!]+) 0', r'float(\1.mean() or 0) \2 0'),
        # Convert Series to scalar where needed
        (r'if ([\w\.]+) > 0:', r'if float(\1) > 0:'),
        (r'if ([\w\.]+) < 0:', r'if float(\1) < 0:'),
        (r'if ([\w\.]+) == 0:', r'if float(\1) == 0:'),
        (r'if ([\w\.]+) != 0:', r'if float(\1) != 0:'),
    ]
    
    for pattern, replacement in safer_ops:
        patched_content = re.sub(pattern, replacement, patched_content)
    
    # Write the patched file
    with open(metrics_file, 'w') as f:
        f.write(patched_content)
    
    print(f"Applied patch to {metrics_file}")
    
    # Also fix the MarketRegimeClassifier
    regime_file = os.path.abspath(os.path.join('src', 'ml_enhancements', 'regime_detection', 'market_regime_classifier.py'))
    
    if os.path.exists(regime_file):
        # Add a detect_regime method if it doesn't exist
        with open(regime_file, 'r') as f:
            regime_content = f.read()
        
        if 'def detect_regime' not in regime_content:
            # Create a backup
            regime_backup = os.path.join(backup_dir, 'market_regime_classifier.py.bak')
            with open(regime_backup, 'w') as f:
                f.write(regime_content)
            
            print(f"Created backup of market_regime_classifier.py at {regime_backup}")
            
            # Add the missing methods
            if 'def describe_regime' not in regime_content:
                regime_content += """
    def detect_regime(self, data):
        \"\"\"
        Detect the current market regime based on the provided data.
        
        This is a simple placeholder implementation. In practice, this would use
        more sophisticated methods like clustering, HMM, or other statistical techniques.
        
        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            Historical price or returns data
            
        Returns:
        --------
        str
            Identified market regime ('normal', 'volatile', 'trending', 'mean_reverting')
        \"\"\"
        # Basic implementation - in practice, would be more sophisticated
        if data is None or len(data) < 20:
            return 'unknown'
            
        # Convert to returns if prices were provided
        if isinstance(data, pd.Series):
            returns = data.pct_change().dropna()
        else:
            # Assuming the last column is the price
            returns = data.iloc[:, -1].pct_change().dropna()
            
        volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
        
        if volatility > 0.25:
            return 'volatile'
        elif volatility < 0.10:
            return 'low_volatility'
        
        # Check for trend using simple moving average crossover
        if len(returns) >= 50:
            ma_short = returns.rolling(20).mean()
            ma_long = returns.rolling(50).mean()
            
            # Check last 5 days of crossover
            crossover_points = (ma_short.iloc[-5:] > ma_long.iloc[-5:]).sum()
            
            if crossover_points >= 4:
                return 'trending_up'
            elif crossover_points <= 1:
                return 'trending_down'
        
        # Check mean reversion using autocorrelation
        if len(returns) >= 20:
            autocorr = returns.autocorr(lag=1)
            if autocorr < -0.2:
                return 'mean_reverting'
                
        return 'normal'
        
    def describe_regime(self, regime):
        \"\"\"
        Provide a description of the given market regime.
        
        Parameters:
        -----------
        regime : str
            Market regime identifier
            
        Returns:
        --------
        str
            Description of the market regime
        \"\"\"
        descriptions = {
            'volatile': 'High volatility environment with large price swings',
            'low_volatility': 'Periods of low volatility with small price movements',
            'trending_up': 'Upward trending market with persistent price increases',
            'trending_down': 'Downward trending market with persistent price decreases',
            'mean_reverting': 'Mean-reverting market with frequent reversals',
            'normal': 'Normal market conditions with mixed characteristics',
            'unknown': 'Insufficient data to determine market regime'
        }
        
        return descriptions.get(regime, 'Unknown market regime')
"""
            
            # Add missing import if needed
            if 'import pandas as pd' not in regime_content:
                regime_content = "import pandas as pd\n" + regime_content
            
            # Write the updated file
            with open(regime_file, 'w') as f:
                f.write(regime_content)
            
            print(f"Added missing methods to {regime_file}")
    
    return True

def main():
    """Main function to apply patches."""
    print("Applying patches to fix metrics calculation and market regime classifier...")
    
    # Ensure patches directory exists
    ensure_directory_exists('patches')
    
    if apply_patch():
        print("Patches applied successfully.")
    else:
        print("Failed to apply patches.")

if __name__ == '__main__':
    main() 