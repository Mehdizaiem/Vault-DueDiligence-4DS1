from extract import fetch_crypto_data
from transform import transform_data
from load import load_to_postgresql
import sys
from pathlib import Path

# Add the project root to Python path
root_dir = Path(__file__).parents[3]  # Go up 3 levels from current file
sys.path.append(str(root_dir))

from config import load_config

def main():
    try:
        # Load configuration from environment
        config = load_config()
        
        # ETL Process
        print("Starting ETL process...")
        
        # Extract
        print("Fetching data from CoinGecko...")
        raw_data = fetch_crypto_data(config['API_KEY'])
        if not raw_data:
            print("Failed to fetch data. Exiting...")
            return
        
        # Transform
        print("Transforming data...")
        transformed_data = transform_data(raw_data)
        if transformed_data is None:
            print("Failed to transform data. Exiting...")
            return
        
        # Load
        print("Loading data to PostgreSQL...")
        success = load_to_postgresql(transformed_data, config['DB_CONNECTION'])
        
        if success:
            print("ETL process completed successfully!")
        else:
            print("ETL process completed with errors.")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()