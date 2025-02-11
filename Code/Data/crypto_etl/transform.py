import pandas as pd
from datetime import datetime

def transform_data(data):
    """Transform the raw API data into a structured DataFrame"""
    if not data:
        print("No data to transform")
        return None
        
    df = pd.DataFrame(data)
    
    # Select relevant columns
    columns_to_keep = [
        'id', 'symbol', 'name', 'current_price', 'market_cap',
        'market_cap_rank', 'total_volume', 'price_change_percentage_24h',
        'circulating_supply', 'total_supply', 'max_supply',
        'last_updated'
    ]
    
    # Check if all columns exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    
    # Add ETL timestamp
    df['etl_timestamp'] = datetime.now()
    
    # Convert numeric columns
    numeric_columns = ['current_price', 'market_cap', 'total_volume', 
                      'price_change_percentage_24h', 'circulating_supply', 
                      'total_supply', 'max_supply']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df