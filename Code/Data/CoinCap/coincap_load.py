from sqlalchemy import create_engine, text, MetaData

def load_to_postgresql(df, db_connection_string, table_name=None):
    """
    Load the transformed crypto data into PostgreSQL
    Args:
        df: DataFrame to load
        db_connection_string: Database connection string
        table_name: Name of the table to create/update
    """
    if df is None or df.empty:
        print("No data to load")
        return False
    
    if table_name is None:
        table_name = 'crypto_historical'
        
    try:
        # Create database connection
        engine = create_engine(db_connection_string)
        
        # First, drop the table if it exists
        print(f"Dropping table {table_name} if it exists...")
        with engine.connect() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            connection.commit()
        
        print(f"Creating new table {table_name}...")
        # Let pandas create the table with the correct schema
        df.to_sql(
            table_name,
            engine,
            if_exists='fail',  # Fail if table exists (it shouldn't, we just dropped it)
            index=False,
            method='multi',
            chunksize=1000  # Load in chunks for better performance
        )
        
        # Verify the data was loaded
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            print(f"Verified {count} records in {table_name}")
            
            # Print the first few rows for verification
            result = connection.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
            print("\nFirst 5 rows of data:")
            for row in result:
                print(row)
        
        return True
        
    except Exception as e:
        print(f"Error loading data to database: {e}")
        print("DataFrame info:")
        print(df.info())
        print("\nDataFrame head:")
        print(df.head())
        return False