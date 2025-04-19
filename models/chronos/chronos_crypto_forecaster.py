#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal Cryptocurrency Forecaster with Chronos

This script can forecast any cryptocurrency pair using Amazon's Chronos model,
automatically adapting to different coins, data formats, and price scales.

Features:
- Works with any cryptocurrency (BTC, ETH, ADA, SOL, etc.)
- Handles different price denominations (USD, BTC, USDT, etc.)
- Automatically detects and adapts to various CSV formats from exchanges
- Provides appropriate visualization and scaling based on the coin
- Stores forecasts in Weaviate for future analysis
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import torch
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_forecast.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def check_chronos_availability():
    """Check if Chronos package is installed and install if needed."""
    try:
        import chronos
        logger.info("Chronos package is already installed")
        return True
    except ImportError:
        logger.warning("Chronos package not found. Installing...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "chronos-forecasting"])
            logger.info("Successfully installed Chronos")
            return True
        except Exception as e:
            logger.error(f"Failed to install Chronos: {e}")
            return False

def check_cuda_availability():
    """Check if CUDA is available for GPU acceleration."""
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} devices.")
        logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
        return True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) is available.")
        return True
    else:
        logger.warning("No GPU acceleration available. Using CPU.")
        return False

def detect_data_type(df, filename=""):
    """
    Detect data type, price denomination, and appropriate scaling.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        filename (str): Original filename
        
    Returns:
        dict: Data type information including denomination and scaling
    """
    # Check price column exists
    price_col = None
    for col_name in ['price', 'close', 'Close', 'last', 'Last', 'lastPrice']:
        if col_name in df.columns:
            price_col = col_name
            break
    
    if price_col is None:
        logger.warning("No price column found, using first numeric column")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]
            df['price'] = df[price_col]
        else:
            logger.error("No numeric columns found in data")
            return {"type": "unknown", "denomination": "unknown", "scale": 1.0, "symbol": "$"}
    elif price_col != 'price':
        df['price'] = df[price_col]
    
    # Extract base and quote currency from filename
    base_currency = "UNKNOWN"
    quote_currency = "UNKNOWN"
    
    # Parse filename for currency info
    if filename:
        # Common patterns: BASE_QUOTE, BASE-QUOTE, BASEvsQUOTE
        currency_patterns = [
            r'([A-Z]+)[-_/]([A-Z]+)',  # Matches BASE_QUOTE or BASE-QUOTE
            r'([A-Z]+)vs([A-Z]+)',     # Matches BASEvsQUOTE
            r'([A-Z]+)([A-Z]{3,4})$'   # Matches BASEUSD or BASEUSDT with no separator
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, filename)
            if match:
                base_currency = match.group(1)
                quote_currency = match.group(2)
                break
    
    # If still unknown, try to detect from data
    if quote_currency == "UNKNOWN":
        # Check for common quote currencies in filename
        for quote in ["USD", "USDT", "USDC", "BTC", "ETH"]:
            if quote in filename:
                quote_currency = quote
                # Extract base (everything before quote)
                base_parts = filename.split(quote)[0].strip('_-/')
                # Take the last part (in case there are directory names)
                base_currency = base_parts.split('/')[-1]
                break
    
    # Get mean price to help with detection
    mean_price = df['price'].mean()
    median_price = df['price'].median()
    max_price = df['price'].max()
    
    # Determine price denomination based on values and filename
    if 'BTC' in quote_currency or (mean_price < 0.1 and 'BTC' in filename):
        logger.info(f"Detected BTC denomination (mean price: {mean_price:.6f} BTC)")
        price_denomination = "BTC"
        price_symbol = "₿"
        price_scale = 1.0
    elif any(quote in quote_currency for quote in ['USD', 'USDT', 'USDC']):
        logger.info(f"Detected USD denomination (mean price: ${mean_price:.2f})")
        price_denomination = "USD"
        price_symbol = "$"
        price_scale = 1.0
    else:
        # Default to USD if unclear
        logger.info(f"Defaulting to USD denomination (mean price: ${mean_price:.2f})")
        price_denomination = "USD"
        price_symbol = "$"
        price_scale = 1.0
    
    return {
        "type": f"{base_currency}/{quote_currency}",
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "denomination": price_denomination,
        "scale": price_scale,
        "symbol": price_symbol,
        "mean_price": mean_price,
        "median_price": median_price,
        "max_price": max_price
    }

def find_csv_file(symbol, data_dir="data/time series cryptos"):
    """
    Find a CSV file for the given symbol.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC", "ETH")
        data_dir (str): Directory containing CSV files
        
    Returns:
        str: Path to the CSV file or None if not found
    """
    # Normalize path
    data_dir = os.path.join(project_root, data_dir)
    
    # Make sure directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return None
    
    # Normalize symbol to uppercase
    symbol = symbol.upper()
    
    # Look for exact match first
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            if symbol in filename.upper():
                return os.path.join(data_dir, filename)
    
    # If no exact match, look for partial match
    base_symbol = symbol.replace("USDT", "").replace("USD", "")
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            if base_symbol in filename.upper():
                return os.path.join(data_dir, filename)
    
    logger.error(f"No CSV file found for symbol: {symbol}")
    return None

def load_csv_data(file_path, lookback_days=365):
    """
    Load cryptocurrency data from a CSV file with automatic format detection.
    
    Args:
        file_path (str): Path to CSV file
        lookback_days (int): Number of days of historical data to use
        
    Returns:
        tuple: (DataFrame, data_type_info)
    """
    logger.info(f"Loading data from: {file_path}")
    filename = os.path.basename(file_path)
    
    try:
        # Try to determine CSV dialect
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(1024)
            
        # Check for common delimiters
        delimiter = ',' if ',' in sample else ';' if ';' in sample else '\t'
        
        # Try parsing with pandas
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Look for date/timestamp column
        date_columns = ['Date', 'date', 'Timestamp', 'timestamp', 'Time', 'time', 'datetime']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Try to find a column that looks like a date
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check first value to see if it looks like a date
                    first_val = str(df[col].iloc[0])
                    if re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', first_val) or re.search(r'\d{2}:\d{2}', first_val):
                        date_col = col
                        break
        
        if date_col is None:
            logger.warning("No date column found, using row index")
            df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df))
        else:
            # Try to parse the date column
            try:
                df['timestamp'] = pd.to_datetime(df[date_col])
            except Exception as e:
                logger.warning(f"Error parsing date column: {e}")
                # Try multiple date formats
                for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                    try:
                        df['timestamp'] = pd.to_datetime(df[date_col], format=date_format)
                        break
                    except:
                        continue
                
                # If still failed, use row index
                if 'timestamp' not in df.columns:
                    logger.warning("Could not parse date column, using row index")
                    df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df))
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Sort by index
        df = df.sort_index()
        
        # Detect data type and price format
        data_type_info = detect_data_type(df, filename)
        
        # Extract the lookback period
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:]
        
        logger.info(f"Loaded {len(df)} data points from CSV")
        return df, data_type_info
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def load_time_series_data(symbol, lookback_days=365, use_csv=True):
    """
    Load cryptocurrency data from either CSV files or Weaviate.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC", "ETH")
        lookback_days (int): Number of days of historical data to use
        use_csv (bool): Whether to load from CSV (True) or Weaviate (False)
        
    Returns:
        tuple: (DataFrame, data_type_info)
    """
    logger.info(f"Loading historical data for {symbol}")
    
    if use_csv:
        # Find a CSV file for this symbol
        csv_file = find_csv_file(symbol)
        if csv_file:
            return load_csv_data(csv_file, lookback_days)
    
    # If CSV loading failed or not requested, try Weaviate
    try:
        from Sample_Data.vector_store.storage_manager import StorageManager
        
        storage = StorageManager()
        try:
            # Retrieve time series data
            data = storage.retrieve_time_series(symbol, limit=lookback_days)
            
            if not data or len(data) == 0:
                logger.error(f"No data found for {symbol} in Weaviate")
                return None, None
                
            # Convert to DataFrame and prepare
            df = pd.DataFrame(data)
            
            # Create timestamp index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Create 'price' column if needed
            if 'price' not in df.columns:
                if 'close' in df.columns:
                    df['price'] = df['close']
                else:
                    logger.warning("No price or close column found in Weaviate data")
                    return None, None
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Detect data type
            data_type_info = detect_data_type(df, symbol)
            
            logger.info(f"Loaded {len(df)} data points from Weaviate")
            return df, data_type_info
            
        finally:
            storage.close()
            
    except Exception as e:
        logger.error(f"Error loading data from Weaviate: {e}")
        return None, None

def generate_forecast(data, data_info, model_name, days_ahead, num_samples=100):
    """
    Generate cryptocurrency forecasts using Amazon's Chronos model.
    
    Args:
        data (pd.DataFrame): Historical price data
        data_info (dict): Information about the data type and format
        model_name (str): Chronos model to use
        days_ahead (int): Number of days to forecast
        num_samples (int): Number of samples for uncertainty quantification
        
    Returns:
        tuple: (forecast_results, market_insights, plot_path)
    """
    try:
        # Import the ChronosForecaster
        from models.chronos.chronos_crypto_forecaster import ChronosForecaster
        
        # Check for GPU availability and set appropriate device
        use_gpu = check_cuda_availability()
        
        # Initialize the forecaster
        logger.info(f"Initializing Chronos forecaster with model: {model_name}")
        forecaster = ChronosForecaster(
            model_name=model_name,
            use_gpu=use_gpu
        )
        
        # Generate forecast
        symbol = data_info['type']
        logger.info(f"Generating {days_ahead}-day forecast for {symbol}")
        forecast_results = forecaster.forecast(
            data,
            prediction_length=days_ahead,
            num_samples=num_samples,
            quantile_levels=[0.1, 0.5, 0.9]  # 80% prediction interval
        )
        
        # Create visualization with custom formatting for this data type
        logger.info("Creating forecast visualization")
        plot_path = create_forecast_visualization(
            data,
            forecast_results,
            data_info,
            days_ahead
        )
        
        # Generate market insights
        logger.info("Generating market insights")
        market_insights = generate_market_insights(
            data,
            forecast_results,
            data_info
        )
        
        return forecast_results, market_insights, plot_path
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None

def create_forecast_visualization(data, forecast_results, data_info, days_ahead):
    """
    Create visualization of forecast results with proper formatting.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast_results (dict): Results from forecast
        data_info (dict): Information about the data type
        days_ahead (int): Number of days in forecast
        
    Returns:
        str: Path to saved visualization
    """
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Extract symbol and pricing information
    symbol = data_info['type']
    price_symbol = data_info['symbol']
    base_currency = data_info['base_currency']
    quote_currency = data_info['quote_currency']
    
    # Extract forecast components
    dates = forecast_results['dates']
    
    # Get quantiles and mean
    quantiles = forecast_results['quantiles'][0]
    quantile_levels = forecast_results['quantile_levels']
    mean = forecast_results['mean'][0]
    
    # Find index of median or closest to median
    if 0.5 in quantile_levels:
        median_idx = quantile_levels.index(0.5)
    else:
        median_idx = len(quantile_levels) // 2
        
    # Get lower, median, and upper bounds
    low_idx = 0  # Lowest quantile
    high_idx = len(quantile_levels) - 1  # Highest quantile
    
    median_forecast = quantiles[:, median_idx]
    lower_bound = quantiles[:, low_idx]
    upper_bound = quantiles[:, high_idx]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot historical data (last 120 days or all if less)
    history_len = min(120, len(data))
    recent_data = data.iloc[-history_len:]
    plt.plot(recent_data.index, recent_data['price'], 'b-', linewidth=2, label='Historical Price')
    
    # Plot forecast
    plt.plot(dates, median_forecast, 'r-', linewidth=2, label='Median Forecast')
    
    # Plot prediction interval
    plt.fill_between(
        dates,
        lower_bound,
        upper_bound,
        color='red',
        alpha=0.2,
        label=f'{int((max(quantile_levels) - min(quantile_levels)) * 100)}% Prediction Interval'
    )
    
    # Add some sample paths from the full distribution
    samples = forecast_results['samples'][0]
    num_paths = min(10, samples.shape[0])
    for i in range(num_paths):
        plt.plot(dates, samples[i], 'r-', linewidth=0.5, alpha=0.3)
    
    # Get current price and final forecast for annotations
    current_price = data['price'].iloc[-1]
    final_forecast = median_forecast[-1]
    change_pct = ((final_forecast / current_price) - 1) * 100
    
    # Format plot
    title = f"{base_currency}/{quote_currency} Price Forecast (Chronos)"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    
    if data_info['denomination'] == 'BTC':
        plt.ylabel('Price (BTC)', fontsize=12)
    else:
        plt.ylabel('Price (USD)', fontsize=12)
        
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    from matplotlib.ticker import FuncFormatter
    
    if data_info['denomination'] == 'BTC':
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₿{x:.6f}'))
    else:
        # Determine appropriate format based on price range
        if data_info['median_price'] < 0.1:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.4f}'))
        elif data_info['median_price'] < 1.0:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.2f}'))
        else:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Add annotations for forecast starting point
    plt.annotate(
        f"Last observed: {price_symbol}{current_price:.2f}" if current_price >= 1 else f"Last observed: {price_symbol}{current_price:.6f}",
        (data.index[-1], current_price),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Add annotation for median forecast endpoint
    plt.annotate(
        f"Forecast: {price_symbol}{final_forecast:.2f}" if final_forecast >= 1 else f"Forecast: {price_symbol}{final_forecast:.6f}",
        (dates[-1], final_forecast),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plots/{base_currency}_{quote_currency}_forecast_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Forecast plot saved to {filename}")
    plt.close()
    
    return filename

def generate_market_insights(data, forecast_results, data_info):
    """
    Generate market insights from forecast results.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast_results (dict): Results from forecast
        data_info (dict): Information about the data type
        
    Returns:
        dict: Market insights
    """
    # Extract necessary information
    symbol = data_info['type']
    price_symbol = data_info['symbol']
    base_currency = data_info['base_currency']
    quote_currency = data_info['quote_currency']
    
    # Get current price
    current_price = data['price'].iloc[-1]
    
    # Extract forecast components
    quantiles = forecast_results['quantiles'][0]
    quantile_levels = forecast_results['quantile_levels']
    mean = forecast_results['mean'][0]
    
    # Find index of median or closest to median
    if 0.5 in quantile_levels:
        median_idx = quantile_levels.index(0.5)
    else:
        median_idx = len(quantile_levels) // 2
        
    # Get lower, median, and upper bounds
    low_idx = 0  # Lowest quantile
    high_idx = len(quantile_levels) - 1  # Highest quantile
    
    median_forecast = quantiles[:, median_idx]
    final_forecast = median_forecast[-1]
    change_pct = ((final_forecast / current_price) - 1) * 100
    
    # Calculate uncertainty
    uncertainty = (quantiles[:, high_idx] - quantiles[:, low_idx]) / median_forecast * 100
    avg_uncertainty = np.mean(uncertainty)
    
    # Get samples for probability calculation
    samples = forecast_results['samples'][0]
    prob_increase = np.mean(samples[:, -1] > current_price) * 100
    
    # Determine trend
    if change_pct > 5:
        trend = "strongly bullish"
    elif change_pct > 1:
        trend = "bullish"
    elif change_pct < -5:
        trend = "strongly bearish"
    elif change_pct < -1:
        trend = "bearish"
    else:
        trend = "neutral"
    
    # Generate insight text
    if trend in ["strongly bullish", "bullish"]:
        insight = f"Chronos forecasts a {trend} outlook for {base_currency}/{quote_currency} with a projected increase of {change_pct:.2f}% over the next {len(median_forecast)} days."
    elif trend in ["strongly bearish", "bearish"]:
        insight = f"Chronos forecasts a {trend} outlook for {base_currency}/{quote_currency} with a projected decrease of {abs(change_pct):.2f}% over the next {len(median_forecast)} days."
    else:
        insight = f"Chronos forecasts a {trend} outlook for {base_currency}/{quote_currency} with minimal price movement (expected change: {change_pct:.2f}%) over the next {len(median_forecast)} days."
        
    # Add probability context
    if prob_increase > 75:
        insight += f" There is a strong probability ({prob_increase:.1f}%) that the price will increase."
    elif prob_increase > 60:
        insight += f" There is a moderate probability ({prob_increase:.1f}%) that the price will increase."
    elif prob_increase < 25:
        insight += f" There is a strong probability ({100-prob_increase:.1f}%) that the price will decrease."
    elif prob_increase < 40:
        insight += f" There is a moderate probability ({100-prob_increase:.1f}%) that the price will decrease."
    else:
        insight += f" The price direction is uncertain with {prob_increase:.1f}% probability of increase."
    
    # Comment on uncertainty
    if avg_uncertainty > 20:
        insight += f" The forecast shows high uncertainty (±{avg_uncertainty:.1f}% on average), suggesting caution."
    elif avg_uncertainty > 10:
        insight += f" The forecast shows moderate uncertainty (±{avg_uncertainty:.1f}% on average)."
    else:
        insight += f" The forecast shows relatively low uncertainty (±{avg_uncertainty:.1f}% on average)."
    
    # Compile insights
    results = {
        "symbol": symbol,
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "current_price": float(current_price),
        "final_forecast": float(final_forecast),
        "change_pct": float(change_pct),
        "trend": trend,
        "probability_increase": float(prob_increase),
        "average_uncertainty": float(avg_uncertainty),
        "insight": insight,
        "generated_at": datetime.now().isoformat()
    }
    
    return results

def store_forecast(forecast_results, market_insights, data_info, model_name, days_ahead, plot_path):
    """
    Store forecast results in Weaviate.
    
    Args:
        forecast_results (dict): Results from Chronos forecast
        market_insights (dict): Market insights generated from forecast
        data_info (dict): Information about the data type and format
        model_name (str): Name of the model used
        days_ahead (int): Number of days in forecast horizon
        plot_path (str): Path to saved visualization
        
    Returns:
        bool: Success status
    """
    try:
        #from Sample_Data.vector_store.forecast_storage import store_chronos_forecast
        
        base_currency = data_info['base_currency']
        quote_currency = data_info['quote_currency']
        symbol = f"{base_currency}{quote_currency}"
        
        logger.info(f"Storing forecast for {symbol} in Weaviate")
        success = store_chronos_forecast(
            forecast_results=forecast_results,
            market_insights=market_insights,
            symbol=symbol,
            model_name=model_name,
            days_ahead=days_ahead,
            plot_path=plot_path
        )
        
        if success:
            logger.info("Forecast successfully stored")
        else:
            logger.error("Failed to store forecast")
            
        return success
        
    except Exception as e:
        logger.error(f"Error storing forecast: {e}")
        return False

def analyze_forecast_history(symbol, date_range=30):
    """
    Analyze forecast history and performance for a symbol.
    
    Args:
        symbol (str): Trading symbol
        date_range (int): Number of days to look back
        
    Returns:
        dict: Forecast comparison analysis
    """
    try:
        from Sample_Data.vector_store.forecast_storage import compare_forecasts
        
        logger.info(f"Analyzing forecast history for {symbol}")
        results = compare_forecasts(symbol, date_range)
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing forecast history: {e}")
        return {"error": str(e)}

def list_available_coins():
    """
    List available cryptocurrency data files.
    
    Returns:
        list: List of available cryptocurrencies
    """
    data_dir = os.path.join(project_root, "data", "time series cryptos")
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return []
    
    available_coins = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        # Try to extract base currency from filename
        base_currency = "UNKNOWN"
        quote_currency = "UNKNOWN"
        
        # Common patterns: BASE_QUOTE, BASE-QUOTE, BASEvsQUOTE
        currency_patterns = [
            r'([A-Z]+)[-_/]([A-Z]+)',  # Matches BASE_QUOTE or BASE-QUOTE
            r'([A-Z]+)vs([A-Z]+)',     # Matches BASEvsQUOTE
            r'([A-Z]+)([A-Z]{3,4})$'   # Matches BASEUSD or BASEUSDT with no separator
        ]
        
        for pattern in currency_patterns:
            match = re.search(pattern, filename.upper())
            if match:
                base_currency = match.group(1)
                quote_currency = match.group(2)
                break
        
        # If still unknown, check for common quote currencies
        if base_currency == "UNKNOWN":
            for quote in ["USD", "USDT", "USDC", "BTC", "ETH"]:
                if quote in filename.upper():
                    quote_currency = quote
                    # Extract base (everything before quote)
                    parts = filename.upper().split(quote)[0]
                    base_currency = parts.strip('_-/')
                    break
        
        # Add to list if we found a valid currency
        if base_currency != "UNKNOWN":
            coin_info = {
                "symbol": f"{base_currency}/{quote_currency}",
                "base": base_currency,
                "quote": quote_currency,
                "filename": filename
            }
            available_coins.append(coin_info)
    
    return available_coins

def run_demo(args):
    """
    Run the complete forecasting demo.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Check Chronos availability
    if not check_chronos_availability():
        logger.error("Chronos package is required but could not be installed")
        return 1
    
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    
    # List available coins if requested
    if args.list_coins:
        coins = list_available_coins()
        print("\nAvailable cryptocurrencies:\n")
        
        if not coins:
            print("No cryptocurrency data files found!")
            return 1
            
        print(f"{'Symbol':<15} {'Base':<8} {'Quote':<8} {'File'}")
        print("-" * 50)
        
        for coin in coins:
            print(f"{coin['symbol']:<15} {coin['base']:<8} {coin['quote']:<8} {coin['filename']}")
            
        return 0
    
    # Load historical data
    data, data_info = load_time_series_data(args.symbol, args.lookback, not args.use_weaviate)
    
    if data is None or data_info is None:
        logger.error(f"Failed to load historical data for {args.symbol}")
        return 1
    
    logger.info(f"Loaded {len(data)} data points for {args.symbol}")
    
    # Generate forecast
    forecast_results, market_insights, plot_path = generate_forecast(
        data,
        data_info,
        args.model,
        args.days_ahead,
        args.samples
    )
    
    if forecast_results is None:
        logger.error("Failed to generate forecast")
        return 1
    
    # Store forecast if requested
    if args.store:
        success = store_forecast(
            forecast_results,
            market_insights,
            data_info,
            args.model,
            args.days_ahead,
            plot_path
        )
        if not success:
            logger.warning("Failed to store forecast in Weaviate")
    
    # Analyze forecast history if requested
    if args.analyze_history:
        symbol = f"{data_info['base_currency']}{data_info['quote_currency']}"
        forecast_history = analyze_forecast_history(symbol, args.history_days)
        
        if "error" not in forecast_history:
            # Print analysis summary
            print("\n===== FORECAST HISTORY ANALYSIS =====")
            print(f"Symbol: {symbol}")
            print(f"Total forecasts analyzed: {forecast_history.get('forecasts_count', 0)}")
            print(f"Most common trend: {forecast_history.get('most_common_trend', 'unknown')}")
            print(f"Trend consistency: {forecast_history.get('trend_consistency', 0):.1f}%")
            print(f"Average forecasted change: {forecast_history.get('avg_forecasted_change', 0):.2f}%")
            print(f"Direction changes: {forecast_history.get('direction_changes', 0)}")
            print("=====================================\n")
    
    # Print forecast summary
    print("\n===== CHRONOS FORECAST SUMMARY =====")
    print(f"Symbol: {data_info['base_currency']}/{data_info['quote_currency']}")
    print(f"Model: {args.model}")
    print(f"Forecast horizon: {args.days_ahead} days")
    
    # Format price based on denomination
    if data_info['denomination'] == 'BTC':
        current_price_str = f"₿{market_insights['current_price']:.6f}"
        final_forecast_str = f"₿{market_insights['final_forecast']:.6f}"
    else:
        # Use appropriate precision based on price magnitude
        if market_insights['current_price'] < 0.01:
            current_price_str = f"${market_insights['current_price']:.6f}"
            final_forecast_str = f"${market_insights['final_forecast']:.6f}"
        elif market_insights['current_price'] < 1.0:
            current_price_str = f"${market_insights['current_price']:.4f}"
            final_forecast_str = f"${market_insights['final_forecast']:.4f}"
        else:
            current_price_str = f"${market_insights['current_price']:.2f}"
            final_forecast_str = f"${market_insights['final_forecast']:.2f}"
    
    print(f"Current price: {current_price_str}")
    print(f"Final forecast: {final_forecast_str} ({market_insights['change_pct']:+.2f}%)")
    print(f"Forecast trend: {market_insights['trend']}")
    print(f"Probability of increase: {market_insights['probability_increase']:.1f}%")
    print(f"Average uncertainty: ±{market_insights['average_uncertainty']:.1f}%")
    print(f"Forecast visualization: {plot_path}")
    print("\nInsight:")
    print(market_insights['insight'])
    print("===================================\n")
    
    return 0

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Universal Cryptocurrency Forecaster")
    
    # Basic options
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Cryptocurrency symbol to forecast")
    parser.add_argument("--days-ahead", type=int, default=14, help="Number of days to forecast")
    parser.add_argument("--lookback", type=int, default=365, help="Days of historical data to use")
    
    # Model options
    parser.add_argument("--model", type=str, default="amazon/chronos-t5-small",
                      choices=[
                          "amazon/chronos-t5-tiny",
                          "amazon/chronos-t5-mini",
                          "amazon/chronos-t5-small",
                          "amazon/chronos-t5-base",
                          "amazon/chronos-bolt-tiny",
                          "amazon/chronos-bolt-mini", 
                          "amazon/chronos-bolt-small"
                      ],
                      help="Chronos model to use")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples for uncertainty quantification")
    
    # Storage and data options
    parser.add_argument("--store", action="store_true", help="Store forecast in Weaviate")
    parser.add_argument("--use-weaviate", action="store_true", help="Load data from Weaviate instead of CSV")
    
    # Analysis options
    parser.add_argument("--analyze-history", action="store_true", help="Analyze forecast history")
    parser.add_argument("--history-days", type=int, default=30, help="Days of forecast history to analyze")
    
    # Utility options
    parser.add_argument("--list-coins", action="store_true", help="List available cryptocurrency data files")
    
    args = parser.parse_args()
    
    try:
        return run_demo(args)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())