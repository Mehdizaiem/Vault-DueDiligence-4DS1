import pandas as pd
from datetime import datetime

def transform_bitcoin_data(data):
    """Transform Bitcoin historical data into a structured DataFrame"""
    if not data or 'prices' not in data:
        print("No data to transform")
        return None
        
    # Create DataFrames for each metric
    prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    market_caps_df = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
    volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    
    # Convert timestamp (milliseconds) to datetime for all dataframes
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
    market_caps_df['timestamp'] = pd.to_datetime(market_caps_df['timestamp'], unit='ms')
    volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
    
    # Merge all metrics
    df = prices_df.merge(market_caps_df[['timestamp', 'market_cap']], on='timestamp')
    df = df.merge(volumes_df[['timestamp', 'volume']], on='timestamp')
    
    # Add technical indicators
    # Moving averages
    df['MA_7'] = df['price'].rolling(window=7).mean()
    df['MA_30'] = df['price'].rolling(window=30).mean()
    df['MA_90'] = df['price'].rolling(window=90).mean()
    
    # Volatility (standard deviation of returns)
    df['daily_return'] = df['price'].pct_change()
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
    
    # Volume moving average
    df['volume_MA_7'] = df['volume'].rolling(window=7).mean()
    
    # Relative Strength Index (RSI)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add ETL timestamp
    df['etl_timestamp'] = datetime.now()
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Round numeric columns to handle floating point precision
    numeric_columns = ['price', 'market_cap', 'volume', 'MA_7', 'MA_30', 'MA_90', 
                      'daily_return', 'volatility_30d', 'volume_MA_7', 'RSI']
    df[numeric_columns] = df[numeric_columns].round(8)
    
    print(f"Transformed data shape: {df.shape}")
    print("Data columns:", df.columns.tolist())
    return df