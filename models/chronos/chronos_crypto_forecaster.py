#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos Cryptocurrency Forecaster
This script uses the Chronos forecasting model to predict cryptocurrency prices.
It loads historical data, trains the model, and generates forecasts with visualizations.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import re
import glob
from typing import Dict, Any, List, Optional, Union, Tuple
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_project_root():
    """Find the project root directory that contains the data folder."""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    test_dirs = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir))
    ]
    
    for directory in test_dirs:
        data_dir = os.path.join(directory, 'data')
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            return directory
    
    return current_dir

# Find project root
PROJECT_ROOT = find_project_root()
logger.info(f"Using project root: {PROJECT_ROOT}")

# Add project root to path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import storage manager if available
try:
    from Sample_Data.vector_store.storage_manager import StorageManager
    STORAGE_AVAILABLE = True
    logger.info("StorageManager successfully imported")
except ImportError:
    logger.warning("StorageManager not available - results will not be stored in Weaviate")
    STORAGE_AVAILABLE = False

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
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (MPS) is available.")
        return "mps"
    else:
        logger.warning("No GPU acceleration available. Using CPU.")
        return "cpu"

def convert_symbol_for_api(symbol):
    """
    Convert internal symbol format to API-compatible format.
    
    Args:
        symbol (str): Internal symbol format (e.g., "BTC_USD")
        
    Returns:
        tuple: (coin_id, base_symbol, quote_currency)
    """
    # Extract base and quote
    if "_" in symbol:
        parts = symbol.split("_")
        base_symbol = parts[0].upper()
        quote_currency = parts[1].lower() if len(parts) > 1 else "usd"
    else:
        # Handle symbols without separator
        matches = re.match(r"([A-Z]+)([A-Z]{3,4})$", symbol)
        if matches:
            base_symbol = matches.group(1).upper()
            quote_currency = matches.group(2).lower()
        else:
            # Default assumption
            base_symbol = symbol.replace("USD", "").replace("USDT", "").upper()
            quote_currency = "usd"
    
    # Map common symbols to CoinGecko IDs
    coin_id_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple",
        "BNB": "binancecoin",
        "ADA": "cardano",
        "DOT": "polkadot",
        "DOGE": "dogecoin",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
        "LTC": "litecoin"
    }
    
    coin_id = coin_id_map.get(base_symbol, base_symbol.lower())
    
    return coin_id, base_symbol, quote_currency

def format_symbol_for_storage(symbol):
    """
    Format symbol for storage in Weaviate.
    Converts any symbol format to the standard format used in the database.
    
    Args:
        symbol (str): Symbol in any format (e.g. "BTC", "BTC_USD", "BTCUSD")
        
    Returns:
        str: Formatted symbol for storage (e.g. "BTCUSD")
    """
    # Get base and quote parts
    _, base_symbol, quote_currency = convert_symbol_for_api(symbol)
    
    # Format quote currency
    if quote_currency.lower() in ["usd", "usdt", "usdc"]:
        quote_formatted = "USD"
    else:
        quote_formatted = quote_currency.upper()
    
    # Combine into standard format
    return f"{base_symbol}{quote_formatted}"

def fetch_api_data(symbol, days=365, retries=3, delay=1):
    """
    Fetch cryptocurrency data from CoinGecko API.
    
    Args:
        symbol (str): Symbol to fetch (e.g. "BTC_USD")
        days (int): Number of days of data to fetch
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    logger.info(f"Fetching {days} days of data for {symbol} from CoinGecko API")
    
    # Convert symbol to API format
    coin_id, base_symbol, quote_currency = convert_symbol_for_api(symbol)
    
    # CoinGecko has limits on number of days for free tier
    if days > 90:
        logger.warning(f"CoinGecko may limit historical data to 90 days for free tier. Requested: {days} days")
    
    for attempt in range(retries):
        try:
            # Get market data including historical prices
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": quote_currency,
                "days": str(min(days, 365)),  # CoinGecko has limits on days
                "interval": "daily"
            }
            
            # Add user agent to reduce chance of being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Process price data (format: [timestamp, price])
                if "prices" in data and data["prices"]:
                    price_data = data["prices"]
                    
                    # Get volume data if available
                    volume_data = data.get("total_volumes", [])
                    has_volume = len(volume_data) == len(price_data)
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(price_data, columns=["timestamp", "price"])
                    
                    # Add volume if available
                    if has_volume:
                        df["volume"] = [v[1] for v in volume_data]
                    
                    # Convert timestamp (milliseconds) to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    
                    # Set timestamp as index
                    df = df.set_index("timestamp")
                    
                    # Sort by date
                    df = df.sort_index()
                    
                    # Fill missing data using forward and backward fill
                    df = df.ffill().bfill()
                    
                    logger.info(f"Successfully fetched {len(df)} days of data from CoinGecko API")
                    return df
                
                logger.warning(f"No price data found in API response for {coin_id}")
                
            elif response.status_code == 429:
                # Rate limit hit, wait longer before retry
                wait_time = delay * (attempt + 1) * 2
                logger.warning(f"Rate limit hit (429). Waiting {wait_time} seconds before retry {attempt+1}/{retries}")
                time.sleep(wait_time)
                continue
                
            else:
                logger.warning(f"Failed to fetch data from CoinGecko (status: {response.status_code})")
                
            # Wait before retry
            if attempt < retries - 1:
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error fetching API data (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    
    # If all attempts failed, try alternative API provider
    logger.warning("Failed to fetch data from CoinGecko, trying alternative API...")
    return fetch_alternative_api_data(symbol, days)

def fetch_alternative_api_data(symbol, days=365):
    """
    Fetch cryptocurrency data from an alternative API (CryptoCompare).
    
    Args:
        symbol (str): Symbol to fetch (e.g. "BTC_USD")
        days (int): Number of days of data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with latest price data
    """
    try:
        # Convert symbol to API format
        _, base_symbol, quote_currency = convert_symbol_for_api(symbol)
        
        # Format for CryptoCompare
        base_sym = base_symbol.upper()
        quote_sym = quote_currency.upper()
        
        # Get daily historical data (OHLCV)
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            "fsym": base_sym,
            "tsym": quote_sym,
            "limit": min(days, 2000),  # Maximum limit is 2000
            "aggregate": 1
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("Response") == "Success" and "Data" in data:
                price_data = data["Data"]["Data"]
                
                # Convert to DataFrame
                df = pd.DataFrame(price_data)
                
                if "time" in df.columns:
                    # Convert Unix timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                    df = df.set_index("timestamp")
                    
                    # Use close price as the main price
                    if "close" in df.columns:
                        df["price"] = df["close"]
                    
                    # Sort by date
                    df = df.sort_index()
                    
                    # Fill missing data
                    df = df.fillna(method="ffill").fillna(method="bfill")
                    
                    logger.info(f"Successfully fetched {len(df)} days of data from CryptoCompare API")
                    return df
        
        logger.warning(f"Failed to fetch data from alternative API (status: {response.status_code})")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching from alternative API: {e}")
        return None

def find_csv_file(symbol):
    """
    Find a CSV file for the given symbol using a robust approach.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC", "ETH")
        
    Returns:
        str: Path to the CSV file or None if not found
    """
    symbol = symbol.upper()
    
    # Try all possible data directories
    possible_dirs = [
        os.path.join(PROJECT_ROOT, "data", "time series cryptos"),
        os.path.join(PROJECT_ROOT, "data", "time_series_cryptos"),
        os.path.join(PROJECT_ROOT, "Sample_Data", "data", "time_series"),
        os.path.join(PROJECT_ROOT, "data")
    ]
    
    # Search for CSV files in all directories
    for data_dir in possible_dirs:
        if not os.path.exists(data_dir):
            continue
            
        logger.info(f"Searching for {symbol} in {data_dir}")
        
        # Search pattern based on symbol
        patterns = [
            f"*{symbol}*.csv",
            f"*{symbol.replace('USDT', '')}*.csv",
            f"*{symbol.replace('USD', '')}*.csv"
        ]
        
        for pattern in patterns:
            matching_files = glob.glob(os.path.join(data_dir, pattern))
            if matching_files:
                logger.info(f"Found matching files: {matching_files}")
                return matching_files[0]
    
    # Use a more aggressive search approach if needed
    for data_dir in possible_dirs:
        if not os.path.exists(data_dir):
            continue
            
        all_csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # Try to find partial matches
        for csv_file in all_csv_files:
            filename = os.path.basename(csv_file).upper()
            base_symbol = symbol.replace("USDT", "").replace("USD", "")
            if base_symbol in filename:
                logger.info(f"Found partial match: {csv_file}")
                return csv_file
    
    logger.error(f"No CSV file found for symbol: {symbol}")
    return None

def load_csv_data(file_path, lookback_days=365):
    """
    Load cryptocurrency data from a CSV file with automatic format detection.
    
    Args:
        file_path (str): Path to CSV file
        lookback_days (int): Number of days to look back
        
    Returns:
        pd.DataFrame: DataFrame with price data
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
        
        # Look for price columns
        price_columns = ['price', 'close', 'Close', 'last', 'Last', 'Price', 'value', 'Value']
        price_col = None
        
        for col in price_columns:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            # Try to find a column that might contain price data
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                logger.warning(f"No clear price column found, using first numeric column: {price_col}")
            else:
                logger.error("No numeric columns found for price data")
                return None
        
        # Create a standardized 'price' column
        if price_col != 'price':
            # Convert price column to numeric, handling comma-separated numbers
            df['price'] = pd.to_numeric(df[price_col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Fill any NaNs using forward and backward fill
        df['price'] = df['price'].ffill().bfill()
        
        # Detect if this is USD or BTC denominated
        mean_price = df['price'].mean()
        
        if 'BTC' in filename or mean_price < 0.1:
            logger.info(f"Detected BTC denomination (mean price: {mean_price:.6f} BTC)")
            price_denomination = "BTC"
            currency_symbol = "₿"
        else:
            logger.info(f"Detected USD denomination (mean price: ${mean_price:.2f})")
            price_denomination = "USD"
            currency_symbol = "$"
        
        # Sort by index
        df = df.sort_index()
        
        # Extract the lookback period
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:]
        
        # Add metadata to DataFrame
        df.attrs['currency_symbol'] = currency_symbol
        df.attrs['price_denomination'] = price_denomination
        df.attrs['mean_price'] = mean_price
        df.attrs['median_price'] = df['price'].median()
        df.attrs['filename'] = filename
        
        # Log the first and last row for verification
        logger.info(f"First row: {df.iloc[0]['price']} at {df.index[0]}")
        logger.info(f"Last row: {df.iloc[-1]['price']} at {df.index[-1]}")
        logger.info(f"Loaded {len(df)} data points from CSV")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_historical_data(symbol, lookback_days=365, api_days=90):
    """
    Load historical data from API and/or CSV.
    
    Args:
        symbol (str): Trading symbol (e.g., "BTC_USD")
        lookback_days (int): Total days of historical data to use
        api_days (int): Days of API data to fetch
        
    Returns:
        pd.DataFrame: DataFrame with price data
    """
    logger.info(f"Loading historical data for {symbol}")
    
    # First try to find a CSV file
    csv_file = find_csv_file(symbol)
    
    if csv_file:
        # Load data from CSV
        data = load_csv_data(csv_file, lookback_days)
        
        if data is not None:
            logger.info(f"Loaded {len(data)} data points for {symbol}")
            return data
    
    # If CSV file not found or loading failed, try API
    logger.info(f"No CSV data available, fetching from API...")
    api_data = fetch_api_data(symbol, days=min(lookback_days, 365))
    
    if api_data is not None:
        logger.info(f"Loaded {len(api_data)} data points for {symbol}")
        
        # Add metadata to DataFrame
        _, base_symbol, quote_currency = convert_symbol_for_api(symbol)
        price_denomination = "BTC" if "BTC" in quote_currency.upper() else "USD"
        currency_symbol = "₿" if price_denomination == "BTC" else "$"
        
        mean_price = api_data['price'].mean()
        
        api_data.attrs['currency_symbol'] = currency_symbol
        api_data.attrs['price_denomination'] = price_denomination
        api_data.attrs['mean_price'] = mean_price
        api_data.attrs['median_price'] = api_data['price'].median()
        api_data.attrs['filename'] = f"{symbol}_API_Data"
        
        return api_data
    
    logger.error(f"Failed to load any data for {symbol}")
    return None

def initialize_chronos_model(model_name="amazon/chronos-t5-small", device="cpu"):
    """
    Initialize Chronos model.
    
    Args:
        model_name (str): Name of the Chronos model to use
        device (str): Device to use for inference
        
    Returns:
        model: Initialized Chronos model
    """
    logger.info(f"Initializing Chronos forecaster with model: {model_name}")
    
    try:
        from chronos import BaseChronosPipeline
        
        # Determine torch dtype
        if device == "cuda" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16
        else:
            torch_dtype = torch.float32
        
        logger.info(f"Loading Chronos model: {model_name}")
        model = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else device,
            torch_dtype=torch_dtype
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error initializing Chronos model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_forecast(model, data, days_ahead=7, num_samples=10):
    """
    Generate forecast using Chronos model.
    
    Args:
        model: Chronos model
        data (pd.DataFrame): Historical price data
        days_ahead (int): Number of days to forecast
        num_samples (int): Number of samples to generate
        
    Returns:
        dict: Forecast results
    """
    logger.info(f"Generating {days_ahead}-day forecast for {data.attrs['filename']}")
    
    try:
        # Prepare input data
        prices = data['price'].values
        
        # Use the last 90 days of data as context
        context_length = min(90, len(prices))
        context = prices[-context_length:]
        
        # Normalize context
        context_mean = np.mean(context)
        context_std = np.std(context)
        
        if context_std > 0:
            normalized_context = (context - context_mean) / context_std
        else:
            # Handle zero variance case
            normalized_context = context - context_mean
            logger.warning("Zero standard deviation in context data, using mean subtraction only")
        
        # Convert to torch tensor
        context_tensor = torch.tensor(normalized_context, dtype=torch.float32).unsqueeze(0)
        
        # Generate forecast
        samples = model.predict(
            context=context_tensor,
            prediction_length=days_ahead,
            num_samples=num_samples
        )
        
        # Get quantiles
        quantiles, mean = model.predict_quantiles(
            context=context_tensor,
            prediction_length=days_ahead,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Convert predictions back to original scale
        def denormalize(norm_values):
            if isinstance(norm_values, torch.Tensor):
                return norm_values.cpu().numpy() * context_std + context_mean
            return norm_values * context_std + context_mean
        
        # Extract predictions
        median_forecast = quantiles[0, :, 1].cpu().numpy()  # 0.5 quantile (median)
        mean_forecast = mean[0].cpu().numpy()
        lower_bound = quantiles[0, :, 0].cpu().numpy()  # 0.1 quantile
        upper_bound = quantiles[0, :, 2].cpu().numpy()  # 0.9 quantile
        
        # Denormalize predictions
        median_forecast_denorm = denormalize(median_forecast)
        mean_forecast_denorm = denormalize(mean_forecast)
        lower_bound_denorm = denormalize(lower_bound)
        upper_bound_denorm = denormalize(upper_bound)
        
        # Generate forecast dates
        last_date = data.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create forecast visualization
        plot_path = create_forecast_visualization(
            data, 
            forecast_dates, 
            median_forecast_denorm, 
            lower_bound_denorm, 
            upper_bound_denorm
        )
        
        # Generate market insights
        market_insights = generate_market_insights(
            data, 
            median_forecast_denorm, 
            quantiles.cpu().numpy(),
            samples.cpu().numpy(),
            context_std,
            context_mean
        )
        
        # Prepare results
        forecast_results = {
            'symbol': data.attrs.get('filename', 'Unknown').split('.')[0],
            'dates': forecast_dates,
            'median_forecast': median_forecast_denorm.tolist(),
            'mean_forecast': mean_forecast_denorm.tolist(),
            'lower_bound': lower_bound_denorm.tolist(),
            'upper_bound': upper_bound_denorm.tolist(),
            'current_price': float(data['price'].iloc[-1]),
            'forecast_price': float(median_forecast_denorm[-1]),
            'change_pct': float(market_insights['change_pct']),
            'market_insights': market_insights,
            'plot_path': plot_path
        }
        
        return forecast_results
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_forecast_visualization(data, forecast_dates, forecast, lower_bound, upper_bound):
    """
    Create visualization of forecast.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast_dates (list): List of dates for forecast period
        forecast (np.array): Median forecast values
        lower_bound (np.array): Lower bound of forecast (10th percentile)
        upper_bound (np.array): Upper bound of forecast (90th percentile)
        
    Returns:
        str: Path to saved plot
    """
    logger.info("Creating forecast visualization")
    
    try:
        # Create output directory
        os.makedirs("plots", exist_ok=True)
        
        # Get metadata from DataFrame
        symbol = data.attrs.get('filename', 'Unknown').split('.')[0]
        currency_symbol = data.attrs.get('currency_symbol', '$')
        price_denomination = data.attrs.get('price_denomination', 'USD')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 90 days to keep it readable)
        history_len = min(90, len(data))
        recent_data = data.iloc[-history_len:]
        plt.plot(recent_data.index, recent_data['price'], 'b-', linewidth=2, label='Historical Price')
        
        # Plot forecast
        plt.plot(forecast_dates, forecast, 'g-', linewidth=2, label='Forecast (Median)')
        
        # Plot uncertainty bands
        plt.fill_between(forecast_dates, lower_bound, upper_bound, color='g', alpha=0.2, label='80% Confidence Interval')
        
        # Format plot
        plt.title(f"{symbol} Price Forecast", fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'Price ({price_denomination})', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format y-axis based on price denomination
        from matplotlib.ticker import FuncFormatter
        
        if price_denomination == 'BTC':
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₿{x:.6f}'))
        else:
            mean_price = data.attrs.get('mean_price', 0)
            if mean_price < 0.1:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.4f}'))
            elif mean_price < 1.0:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.2f}'))
            else:
                plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
        
        # Rotate date labels
        plt.gcf().autofmt_xdate()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"plots/{symbol}_forecast_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Forecast plot saved to {filename}")
        plt.close()
        
        return filename
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return None

def generate_market_insights(data, forecast, quantiles, samples, context_std, context_mean):
    """
    Generate market insights from forecast.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast (np.array): Median forecast values
        quantiles (np.array): Quantile forecasts
        samples (np.array): Sample forecasts
        context_std (float): Standard deviation of context data
        context_mean (float): Mean of context data
        
    Returns:
        dict: Market insights
    """
    # Get current price and forecast final price
    current_price = data['price'].iloc[-1]
    final_forecast = forecast[-1]
    
    # Calculate percent change
    change_pct = ((final_forecast / current_price) - 1) * 100
    
    # Calculate uncertainty (using numpy operations)
    # Fix for the error: Handle the quantiles shape correctly
    if len(quantiles.shape) == 3:
        # Assuming shape is (batch, time, quantile)
        lower_values = quantiles[0, :, 0] * context_std + context_mean  # 0.1 quantile
        upper_values = quantiles[0, :, 2] * context_std + context_mean  # 0.9 quantile
        median_values = quantiles[0, :, 1] * context_std + context_mean  # 0.5 quantile
        
        # Calculate uncertainty as percentage of median
        uncertainty_values = np.zeros(len(median_values))
        for i in range(len(median_values)):
            if median_values[i] != 0:  # Avoid division by zero
                uncertainty_values[i] = ((upper_values[i] - lower_values[i]) / median_values[i]) * 100
            else:
                uncertainty_values[i] = 0
        
        # Calculate average uncertainty
        avg_uncertainty = float(np.mean(uncertainty_values))
    else:
        # Fallback if the shape is unexpected
        avg_uncertainty = 10.0  # Default value
        logger.warning(f"Unexpected quantiles shape: {quantiles.shape}, using default uncertainty")
    
    # Calculate probability of increase
    # Fix for potential shape issues with samples
    if isinstance(samples, np.ndarray) and len(samples.shape) >= 2:
        # Get the last prediction point from each sample
        final_samples = samples[:, -1] if len(samples.shape) == 2 else samples[:, 0, -1]
        # Denormalize
        final_samples = final_samples * context_std + context_mean
        # Calculate probability of increase
        prob_increase = float(np.mean(final_samples > current_price) * 100)
    else:
        prob_increase = 50.0  # Default value
        logger.warning(f"Unexpected samples shape: {type(samples)}, using default probability")
    
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
    
    # Format coin name
    symbol = data.attrs.get('filename', 'Unknown').split('.')[0]
    if "_" in symbol:
        parts = symbol.split("_")
        base_symbol = parts[0].upper()
        quote_currency = parts[1].upper()
    else:
        base_symbol = symbol.replace("USD", "").replace("USDT", "").upper()
        quote_currency = "USD"
    
    # Generate insight text
    if trend in ["strongly bullish", "bullish"]:
        insight = f"Chronos forecasts a {trend} outlook for {base_symbol}/{quote_currency} with a projected increase of {change_pct:.2f}% in the forecast period."
    elif trend in ["strongly bearish", "bearish"]:
        insight = f"Chronos forecasts a {trend} outlook for {base_symbol}/{quote_currency} with a projected decrease of {abs(change_pct):.2f}% in the forecast period."
    else:
        insight = f"Chronos forecasts a {trend} outlook for {base_symbol}/{quote_currency} with minimal price movement (expected change: {change_pct:.2f}%) in the forecast period."
        
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
    
    # Return market insights
    return {
        'trend': trend,
        'change_pct': change_pct,
        'prob_increase': prob_increase,
        'avg_uncertainty': avg_uncertainty,
        'insight': insight
    }

def store_forecast_results(forecast_data):
    """
    Store forecast results in Weaviate.
    
    Args:
        forecast_data (dict): Forecast data
        
    Returns:
        bool: Success or failure
    """
    if not STORAGE_AVAILABLE:
        logger.warning("StorageManager not available - forecast not stored")
        return False
        
    storage_manager = None
    try:
        # Initialize storage manager
        storage_manager = StorageManager()
        storage_manager.connect()
        logger.info("Connected to Weaviate for storage")
        
        # Prepare forecast data for storage
        formatted_symbol = format_symbol_for_storage(forecast_data['symbol'])
        
        storage_data = {
            "symbol": formatted_symbol,
            "forecast_timestamp": datetime.now().isoformat(),
            "model_name": "chronos",
            "model_type": "transformer",
            "days_ahead": len(forecast_data['dates']),
            "current_price": forecast_data['current_price'],
            "forecast_dates": [d.isoformat() for d in forecast_data['dates']],
            "forecast_values": forecast_data['median_forecast'],
            "lower_bounds": forecast_data['lower_bound'],
            "upper_bounds": forecast_data['upper_bound'],
            "final_forecast": forecast_data['forecast_price'],
            "change_pct": forecast_data['change_pct'],
            "trend": forecast_data['market_insights']['trend'],
            "probability_increase": forecast_data['market_insights']['prob_increase'],
            "average_uncertainty": forecast_data['market_insights']['avg_uncertainty'],
            "insight": forecast_data['market_insights']['insight']
        }
        
        # Store the forecast
        success = storage_manager.store_forecast(storage_data)
        
        if success:
            logger.info(f"Successfully stored forecast for {formatted_symbol}")
        else:
            logger.error(f"Failed to store forecast for {formatted_symbol}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error storing forecast: {e}")
        return False
    finally:
        if storage_manager is not None:
            try:
                storage_manager.close()
                logger.info("Closed Weaviate connection after storage")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Chronos Cryptocurrency Forecaster")
    
    # Basic options
    parser.add_argument("--symbol", type=str, default="BTC_USD", 
                      help="Cryptocurrency symbol to forecast (e.g., BTC_USD)")
    parser.add_argument("--days-ahead", type=int, default=7, 
                      help="Number of days to forecast")
    parser.add_argument("--lookback", type=int, default=3650, 
                      help="Days of historical data to use")
    
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
    
    # Forecast options
    parser.add_argument("--samples", type=int, default=10, 
                      help="Number of samples to generate for uncertainty estimation")
    
    # Storage options
    parser.add_argument("--no-store", action="store_true", 
                      help="Disable storage of results in Weaviate")
    
    args = parser.parse_args()
    
    # Override storage availability if requested
    if args.no_store:
        global STORAGE_AVAILABLE
        STORAGE_AVAILABLE = False
        logger.info("Storage has been disabled via --no-store flag")
    
    try:
        # Check Chronos availability
        if not check_chronos_availability():
            logger.error("Chronos package is required but could not be installed")
            return 1
        
        # Check GPU availability and get device
        device = check_cuda_availability()
        
        # Load historical data
        data = load_historical_data(args.symbol, lookback_days=args.lookback)
        
        if data is None:
            logger.error(f"Failed to load data for {args.symbol}")
            return 1
            
        # Initialize Chronos model
        model = initialize_chronos_model(args.model, device)
        
        if model is None:
            logger.error("Failed to initialize Chronos model")
            return 1
            
        # Generate forecast
        forecast = generate_forecast(
            model, 
            data, 
            days_ahead=args.days_ahead, 
            num_samples=args.samples
        )
        
        if forecast is None:
            logger.error("Failed to generate forecast")
            return 1
            
        # Store forecast results if enabled
        if STORAGE_AVAILABLE:
            success = store_forecast_results(forecast)
            storage_status = "successfully stored" if success else "not stored"
        else:
            storage_status = "storage disabled"
        
        # Print results
        print("\n===== FORECAST RESULTS =====")
        print(f"Symbol: {args.symbol}")
        print(f"Model: {args.model}")
        print(f"Current price: {forecast['current_price']:.2f}")
        print(f"Forecast price: {forecast['forecast_price']:.2f}")
        print(f"Change: {forecast['change_pct']:.2f}%")
        print(f"Trend: {forecast['market_insights']['trend']}")
        print(f"Probability of increase: {forecast['market_insights']['prob_increase']:.1f}%")
        print(f"Average uncertainty: {forecast['market_insights']['avg_uncertainty']:.1f}%")
        print(f"\nInsight: {forecast['market_insights']['insight']}")
        print(f"\nForecast visualization: {forecast['plot_path']}")
        print(f"Forecast {storage_status} in Weaviate database")
        print("=============================\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())