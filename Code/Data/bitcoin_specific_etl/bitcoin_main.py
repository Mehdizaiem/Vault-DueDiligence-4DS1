# Code/Data/bitcoin_specific_etl/bitcoin_main.py
from bitcoin_extract import fetch_bitcoin_data
from bitcoin_transform import transform_bitcoin_data
from bitcoin_load import load_to_postgresql
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
        print("Starting Bitcoin historical data ETL process...")
        
        # Extract
        print("Fetching Bitcoin data from CoinGecko...")
        raw_data = fetch_bitcoin_data()  # No API key needed for public endpoint
        if not raw_data:
            print("Failed to fetch data. Exiting...")
            return
        
        # Transform
        print("Transforming Bitcoin data...")
        transformed_data = transform_bitcoin_data(raw_data)
        if transformed_data is None:
            print("Failed to transform data. Exiting...")
            return
        
        # Load
        print("Loading data to PostgreSQL...")
        success = load_to_postgresql(transformed_data, config['DB_CONNECTION'])
        
        if success:
            print("Bitcoin ETL process completed successfully!")
            print(f"Loaded {len(transformed_data)} days of Bitcoin historical data")
        else:
            print("Bitcoin ETL process completed with errors.")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()