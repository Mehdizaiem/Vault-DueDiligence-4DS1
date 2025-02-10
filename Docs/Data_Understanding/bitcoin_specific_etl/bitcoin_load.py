from sqlalchemy import create_engine, text

def create_bitcoin_table(db_connection_string):
    """Create the Bitcoin historical data table if it doesn't exist"""
    engine = create_engine(db_connection_string)
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS bitcoin_historical (
        timestamp TIMESTAMP,
        price NUMERIC,
        market_cap NUMERIC,
        volume NUMERIC,
        MA_7 NUMERIC,
        MA_30 NUMERIC,
        MA_90 NUMERIC,
        daily_return NUMERIC,
        volatility_30d NUMERIC,
        volume_MA_7 NUMERIC,
        RSI NUMERIC,
        etl_timestamp TIMESTAMP,
        CONSTRAINT pk_bitcoin_historical PRIMARY KEY (timestamp)
    );
    """
    
    with engine.connect() as connection:
        connection.execute(text(create_table_sql))
        connection.commit()

def load_to_postgresql(df, db_connection_string):
    """Load the transformed Bitcoin data into PostgreSQL"""
    if df is None or df.empty:
        print("No data to load")
        return False
        
    try:
        # Create table if it doesn't exist
        create_bitcoin_table(db_connection_string)
        
        # Create database connection
        engine = create_engine(db_connection_string)
        
        # Load data
        df.to_sql('bitcoin_historical', 
                 engine, 
                 if_exists='replace',  # Replace existing data
                 index=False)
        
        print(f"Successfully loaded {len(df)} records to database")
        return True
        
    except Exception as e:
        print(f"Error loading data to database: {e}")
        return False