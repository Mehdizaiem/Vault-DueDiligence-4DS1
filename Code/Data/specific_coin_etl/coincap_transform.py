import pandas as pd
import numpy as np
from datetime import datetime

def transform_coincap_data(df):
    """Transform CoinCap data into a structured DataFrame"""
    if df is None or df.empty:
        print("No data to transform")
        return None
        
    try:
        print("Original columns:", df.columns.tolist())
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        
        # Convert price to numeric and rename
        df['price'] = pd.to_numeric(df['priceUsd'], errors='coerce')
        
        # Calculate technical indicators
        # Moving averages
        df['MA_7'] = df['price'].rolling(window=7).mean()
        df['MA_30'] = df['price'].rolling(window=30).mean()
        df['MA_90'] = df['price'].rolling(window=90).mean()
        
        # Daily returns
        df['daily_return'] = df['price'].pct_change()
        
        # Volatility (30-day rolling standard deviation)
        df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
        
        # RSI
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Add ETL timestamp
        df['etl_timestamp'] = datetime.now()
        
        # Select and reorder columns
        columns_to_keep = [
            'timestamp', 'price', 'MA_7', 'MA_30', 'MA_90',
            'daily_return', 'volatility_30d', 'RSI', 'etl_timestamp'
        ]
        
        df = df[columns_to_keep]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Round numeric columns
        numeric_columns = ['price', 'MA_7', 'MA_30', 'MA_90',
                         'daily_return', 'volatility_30d', 'RSI']
        df[numeric_columns] = df[numeric_columns].round(8)
        
        print(f"Transformed data shape: {df.shape}")
        print("Final columns:", df.columns.tolist())
        return df
        
    except Exception as e:
        print(f"Error transforming data: {e}")
        print("Current df head:")
        print(df.head())
        return None