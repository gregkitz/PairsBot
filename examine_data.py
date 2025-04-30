import pandas as pd
import os

def preview_data_file(file_path, num_rows=10):
    """
    Preview the first few rows of a data file.
    """
    print(f"Examining file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    # Try to read the file using different methods
    try:
        # First try reading as CSV with proper column names
        df = pd.read_csv(file_path, nrows=num_rows, header=None,
                         names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print("\nRead as CSV with column names:")
        print(df.head(num_rows))
        print(f"\nData types: {df.dtypes}")
        
        # Convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("\nTimestamp range:")
            print(f"Start: {df['timestamp'].min()}")
            print(f"End: {df['timestamp'].max()}")
        except Exception as e:
            print(f"Error converting timestamp: {e}")
        
        return df
    except Exception as e:
        print(f"Error reading as CSV: {e}")
    
    try:
        # Try reading as text and splitting
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()[:num_rows]]
        
        print("\nFirst few lines:")
        for i, line in enumerate(lines):
            print(f"{i+1}: {line}")
        
        # Try to determine delimiter
        delimiters = [',', '\t', '|', ';', ' ']
        for delimiter in delimiters:
            if delimiter in lines[0]:
                fields = lines[0].split(delimiter)
                print(f"\nPossible fields using '{delimiter}' as delimiter: {len(fields)}")
                print(fields)
                break
    except Exception as e:
        print(f"Error reading as text: {e}")
    
    return None

if __name__ == "__main__":
    data_dir = r"data\raw\futures_full_1min_contin_adj_ratio_wiihg54"
    files_to_examine = [
        "ES_full_1min_continuous_ratio_adjusted.txt",  # S&P 500 E-mini
        "NQ_full_1min_continuous_ratio_adjusted.txt",  # Nasdaq E-mini
        "ZN_full_1min_continuous_ratio_adjusted.txt",  # 10-Year Treasury Note
        "GC_full_1min_continuous_ratio_adjusted.txt"   # Gold
    ]
    
    for file_name in files_to_examine:
        file_path = os.path.join(data_dir, file_name)
        print("\n" + "="*80)
        df = preview_data_file(file_path)
        if df is not None:
            print(f"\nBasic statistics:\n{df.describe()}")
        print("="*80 + "\n") 