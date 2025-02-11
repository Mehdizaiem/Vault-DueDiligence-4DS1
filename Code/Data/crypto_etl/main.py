from extract import fetch_crypto_data
from transform import transform_data
from load import load_to_postgresql

def main():
    # Configuration
    api_key = 'CG-phcgaJ2qinoHHZRXdU4e8exv'
    db_connection_string = 'postgresql://postgres:hamza@localhost:5432/crypto_db'

    #change both of those to the correct ones depending on api/password for post
    
    # ETL Process
    print("Starting ETL process...")
    
    # Extract
    print("Fetching data from CoinGecko...")
    raw_data = fetch_crypto_data(api_key)
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
    success = load_to_postgresql(transformed_data, db_connection_string)
    
    if success:
        print("ETL process completed successfully!")
    else:
        print("ETL process completed with errors.")

if __name__ == "__main__":
    main()