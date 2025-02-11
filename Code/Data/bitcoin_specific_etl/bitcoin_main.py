from bitcoin_extract import fetch_bitcoin_data
from bitcoin_transform import transform_bitcoin_data
from bitcoin_load import load_to_postgresql

def main():
    # Configuration
    db_connection_string = 'postgresql://postgres:hamza@localhost:5432/crypto_db'
    #changed to the correct ones depending on api/password for post
    
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
    success = load_to_postgresql(transformed_data, db_connection_string)
    
    if success:
        print("Bitcoin ETL process completed successfully!")
        print(f"Loaded {len(transformed_data)} days of Bitcoin historical data")
    else:
        print("Bitcoin ETL process completed with errors.")

if __name__ == "__main__":
    main()