import os
import pandas as pd
import glob

def fix_headers():
    # Find all data.csv files
    csv_files = glob.glob('data/historical/**/**/data.csv', recursive=True)
    
    for file_path in csv_files:
        print(f"Processing {file_path}")
        
        # Check if this is a 1day timeframe file
        is_1day = '/1day/' in file_path
        
        try:
            # Read the first few lines to determine structure
            with open(file_path, 'r') as f:
                header = f.readline().strip()
                first_data_row = f.readline().strip()
            
            # Expected headers
            expected_header = "datetime,open,high,low,close,volume"
            if is_1day:
                # For 1day timeframes, check if we have the extra column (7 columns total)
                if first_data_row and first_data_row.count(',') == 6:
                    # The expected header should include open_interest for the extra column
                    expected_header = "date,open,high,low,close,volume,open_interest"
            
            # Make expected header lowercase
            expected_header = expected_header.lower()
            
            # Compare current header with expected header
            if header.lower() != expected_header:
                print(f"  Updating header in {file_path}")
                print(f"  Old: {header}")
                print(f"  New: {expected_header}")
                
                # Read the entire file as text
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Replace the header
                updated_content = content.replace(header, expected_header, 1)
                
                # Write the updated content back to the file
                with open(file_path, 'w') as f:
                    f.write(updated_content)
            else:
                print(f"  Header already correct in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    fix_headers() 