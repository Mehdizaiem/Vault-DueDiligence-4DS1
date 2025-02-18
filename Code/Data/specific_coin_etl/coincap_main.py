import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from coincap_extract import fetch_coincap_data
from coincap_transform import transform_coincap_data
from coincap_load import load_to_postgresql

# Load config from parent directory
sys.path.append(str(Path(__file__).parents[3]))  # Go up to project root
from config import load_config

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetch historical data for a specific cryptocurrency')
    parser.add_argument('coin', type=str, help='Name of the cryptocurrency (e.g., bitcoin, ethereum, dogecoin)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to fetch (default: 365)')
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        coin_name = args.coin.lower()
        days = args.days
        
        print(f"Fetching {days} days of historical data...")
        
        # Load configuration
        config = load_config()
        
        # ETL Process
        print(f"Starting {coin_name.title()} CoinCap ETL process...")
        
        # Extract
        print(f"Fetching {coin_name.title()} data from CoinCap...")
        # Explicitly pass the days parameter
        raw_data = fetch_coincap_data(asset_id=coin_name, days=int(days))
        if raw_data is None:
            print("Failed to fetch data. Exiting...")
            return
            
        print(f"Fetched {len(raw_data)} records")
        
        # Transform
        print(f"Transforming {coin_name.title()} data...")
        transformed_data = transform_coincap_data(raw_data)
        if transformed_data is None:
            print("Failed to transform data. Exiting...")
            return
            
        if len(transformed_data) > days:
            # Trim to exact number of days requested
            transformed_data = transformed_data.tail(days)
            print(f"Trimmed data to {days} days")
        
        # Load
        print("Loading data to PostgreSQL...")
        table_name = f"{coin_name}_historical"
        success = load_to_postgresql(transformed_data, config['DB_CONNECTION'], table_name=table_name)
        
        if success:
            print(f"{coin_name.title()} ETL process completed successfully!")
            print(f"Loaded {len(transformed_data)} records of {coin_name.title()} historical data")
        else:
            print(f"{coin_name.title()} ETL process completed with errors.")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full error trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
