import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for price data.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate moving averages
    df_copy['ma_7'] = df_copy['price'].rolling(window=7).mean()
    df_copy['ma_14'] = df_copy['price'].rolling(window=14).mean()
    
    # Calculate rate of change
    df_copy['price_roc'] = df_copy['price'].pct_change(periods=1)
    df_copy['volume_roc'] = df_copy['volume_24h'].pct_change(periods=1)
    
    # Relative Strength Index (RSI) - simplified
    delta = df_copy['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df_copy['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df_copy['ema_12'] = df_copy['price'].ewm(span=12, adjust=False).mean()
    df_copy['ema_26'] = df_copy['price'].ewm(span=26, adjust=False).mean()
    df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=9, adjust=False).mean()
    
    # Handle NaN values
    df_copy.fillna(method='bfill', inplace=True)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(0, inplace=True)
    
    return df_copy

def _aggregate_sentiment(sentiment_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Dict[str, Union[float, int]]:
    """
    Aggregate sentiment data for a specific date range.
    
    Args:
        sentiment_df (pd.DataFrame): DataFrame with sentiment data
        start_date (datetime): Start date for aggregation
        end_date (datetime): End date for aggregation
        
    Returns:
        dict: Aggregated sentiment metrics
    """
    if sentiment_df is None or sentiment_df.empty:
        return {"avg_sentiment": 0.5, "sentiment_count": 0}
    
    # Filter by date range
    mask = (sentiment_df['date'] >= start_date) & (sentiment_df['date'] <= end_date)
    filtered_df = sentiment_df[mask]
    
    if filtered_df.empty:
        return {"avg_sentiment": 0.5, "sentiment_count": 0}
    
    # Calculate average sentiment score
    avg_sentiment = filtered_df['sentiment_score'].mean()
    
    # Count articles
    sentiment_count = len(filtered_df)
    
    return {
        "avg_sentiment": avg_sentiment,
        "sentiment_count": sentiment_count
    }

def prepare_features(market_df: pd.DataFrame, sentiment_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepare features by combining market and sentiment data.
    
    Args:
        market_df (pd.DataFrame): DataFrame with market data
        sentiment_df (pd.DataFrame, optional): DataFrame with sentiment data
        
    Returns:
        pd.DataFrame: DataFrame with prepared features
    """
    if market_df is None:
        return None
        
    # Create copy to avoid modifying original data
    df = market_df.copy()
    
    # Add sentiment features if available
    if sentiment_df is not None and not sentiment_df.empty:
        # Create a new column for sentiment data
        df['avg_sentiment'] = 0.5
        df['sentiment_count'] = 0
        
        # Add sentiment data for each day
        for idx, row in df.iterrows():
            current_date = row['timestamp'].date()
            start_date = datetime.combine(current_date, datetime.min.time())
            end_date = datetime.combine(current_date, datetime.max.time())
            
            sentiment_metrics = _aggregate_sentiment(sentiment_df, start_date, end_date)
            df.at[idx, 'avg_sentiment'] = sentiment_metrics['avg_sentiment']
            df.at[idx, 'sentiment_count'] = sentiment_metrics['sentiment_count']
    else:
        # Add placeholder sentiment features
        df['avg_sentiment'] = 0.5
        df['sentiment_count'] = 0
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Create target variable (next day's price)
    df['next_price'] = df['price'].shift(-1)
    
    # Drop last row (which will have NaN for next_price)
    df = df[:-1]
    
    return df

def plot_forecast(symbol: str, predictions: pd.DataFrame, market_df: Optional[pd.DataFrame] = None, 
                 output_path: str = None) -> None:
    """
    Plot price predictions with historical data.
    
    Args:
        symbol (str): Crypto symbol (e.g., 'BTCUSDT')
        predictions (pd.DataFrame): DataFrame with predictions
        market_df (pd.DataFrame, optional): DataFrame with historical market data
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.plot(predictions['date'], predictions['predicted_price'], 'b-o', label='Predicted Price')
    
    # Plot historical data if available
    if market_df is not None and not market_df.empty:
        plt.plot(market_df['timestamp'], market_df['price'], 'r-', label='Historical Price')
    
    # Format plot
    plt.title(f'{symbol} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter('${x:,.2f}')
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Add price annotations
    for i, row in predictions.iterrows():
        plt.annotate(
            f"${row['predicted_price']:.2f}\n({row['change_pct']:+.1f}%)",
            (row['date'], row['predicted_price']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        
    plt.close()

def evaluate_forecast(predictions: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate forecast performance against actual values.
    
    Args:
        predictions (pd.DataFrame): DataFrame with predictions
        actual_df (pd.DataFrame): DataFrame with actual values
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Merge predictions with actuals on date
    predictions['date'] = pd.to_datetime(predictions['date']).dt.date
    actual_df['date'] = pd.to_datetime(actual_df['timestamp']).dt.date
    
    merged = pd.merge(predictions, actual_df[['date', 'price']], on='date', how='inner')
    
    if merged.empty:
        return {"mae": None, "rmse": None, "mape": None}
    
    # Calculate metrics
    mae = np.mean(np.abs(merged['predicted_price'] - merged['price']))
    rmse = np.sqrt(np.mean((merged['predicted_price'] - merged['price'])**2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((merged['price'] - merged['predicted_price']) / merged['price'])) * 100
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }