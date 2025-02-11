from sqlalchemy import create_engine, text

def create_crypto_table(db_connection_string):
    """Create the crypto prices table if it doesn't exist"""
    engine = create_engine(db_connection_string)
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS crypto_prices (
        id VARCHAR(100),
        symbol VARCHAR(20),
        name VARCHAR(100),
        current_price NUMERIC,
        market_cap NUMERIC,
        market_cap_rank INTEGER,
        total_volume NUMERIC,
        price_change_percentage_24h NUMERIC,
        circulating_supply NUMERIC,
        total_supply NUMERIC,
        max_supply NUMERIC,
        last_updated TIMESTAMP,
        etl_timestamp TIMESTAMP,
        CONSTRAINT pk_crypto_prices PRIMARY KEY (id, etl_timestamp)
    );
    """
    
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))
        connection.commit()

def load_to_postgresql(df, db_connection_string):
    """Load the transformed data into PostgreSQL"""
    if df is None or df.empty:
        print("No data to load")
        return False
        
    try:
        # Create table if it doesn't exist
        create_crypto_table(db_connection_string)
        
        # Create database connection
        engine = create_engine(db_connection_string)
        
        # Load data to 'crypto_prices' table
        df.to_sql('crypto_prices', 
                 engine, 
                 if_exists='append', 
                 index=False)
        
        print(f"Successfully loaded {len(df)} records to database")
        return True
        
    except Exception as e:
        print(f"Error loading data to database: {e}")
        return False
