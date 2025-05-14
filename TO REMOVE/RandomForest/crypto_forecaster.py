#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cryptocurrency Forecaster with Random Forest
This script implements a Random Forest model for cryptocurrency forecasting.
Uses the same data fetching logic as Chronos fine-tune.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import glob
import requests
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

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
        
        # Extract base and quote currency from filename
        base_currency = "Unknown"
        quote_currency = "Unknown"
        
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
        
        # Normalize prices to improve training stability
        price_mean = df['price'].mean()
        price_std = df['price'].std()
        df['normalized_price'] = (df['price'] - price_mean) / price_std
        
        # Add metadata to DataFrame
        df.attrs['currency_symbol'] = currency_symbol
        df.attrs['price_denomination'] = price_denomination
        df.attrs['mean_price'] = mean_price
        df.attrs['median_price'] = df['price'].median()
        df.attrs['filename'] = filename
        df.attrs['price_mean'] = price_mean
        df.attrs['price_std'] = price_std
        
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

def load_combined_data(symbol, lookback_days=3650, api_days=365):
    """
    Load and combine historical CSV data with latest API data.
    
    Args:
        symbol (str): Trading symbol
        lookback_days (int): Total days of historical data to use
        api_days (int): Days of API data to fetch
        
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    logger.info(f"Loading combined data for {symbol}")
    logger.info(f"Parameters: lookback_days={lookback_days}, api_days={api_days}")
    
    # Fetch latest data from API first
    logger.info(f"Fetching {api_days} days of data from API...")
    api_data = fetch_api_data(symbol, days=api_days)
    
    # Try to find and load CSV file
    csv_file = find_csv_file(symbol)
    
    # If API data is sufficient, we can use it directly
    if api_data is not None and len(api_data) >= lookback_days:
        logger.info(f"API data is sufficient ({len(api_data)} days), no need for CSV data")
        
        # Normalize the data
        price_mean = api_data['price'].mean()
        price_std = api_data['price'].std()
        api_data['normalized_price'] = (api_data['price'] - price_mean) / price_std
        
        # Add metadata to DataFrame
        _, base_symbol, quote_currency = convert_symbol_for_api(symbol)
        price_denomination = "BTC" if "BTC" in quote_currency.upper() else "USD"
        currency_symbol = "₿" if price_denomination == "BTC" else "$"
        
        api_data.attrs['currency_symbol'] = currency_symbol
        api_data.attrs['price_denomination'] = price_denomination
        api_data.attrs['mean_price'] = price_mean
        api_data.attrs['median_price'] = api_data['price'].median()
        api_data.attrs['filename'] = f"{symbol}_API_Data"
        api_data.attrs['price_mean'] = price_mean
        api_data.attrs['price_std'] = price_std
        
        # Take the most recent lookback_days
        if len(api_data) > lookback_days:
            api_data = api_data.iloc[-lookback_days:]
            logger.info(f"Trimmed API data to last {lookback_days} days")
        
        price_data = api_data
        logger.info("Successfully prepared API data")
    
    # If we couldn't get API data, try to use CSV data only
    elif api_data is None or api_data.empty:
        if csv_file is None:
            logger.error(f"No data available for {symbol} from either API or CSV")
            return None
            
        logger.warning("Could not fetch API data, using CSV data only")
        price_data = load_csv_data(csv_file, lookback_days)
        if price_data is not None:
            logger.info(f"Successfully loaded {len(price_data)} days from CSV")
    
    # If we need to combine CSV and API data
    else:
        if csv_file is not None:
            # Load CSV data
            logger.info("Loading CSV data to combine with API data...")
            csv_data = load_csv_data(csv_file, lookback_days=lookback_days)
            
            if csv_data is None:
                logger.error(f"Failed to load CSV data for {symbol}")
                
                # Use API data only if CSV loading failed
                price_mean = api_data['price'].mean()
                price_std = api_data['price'].std()
                api_data['normalized_price'] = (api_data['price'] - price_mean) / price_std
                
                api_data.attrs['price_mean'] = price_mean
                api_data.attrs['price_std'] = price_std
                
                price_data = api_data
                logger.info("Using API data only due to CSV loading failure")
            else:
                # Check if we need to combine (might already have up-to-date data)
                csv_last_date = csv_data.index[-1].date()
                api_last_date = api_data.index[-1].date()
                
                logger.info(f"CSV data last date: {csv_last_date}")
                logger.info(f"API data last date: {api_last_date}")
                
                if csv_last_date >= api_last_date:
                    logger.info("CSV data is already up-to-date, no need to combine with API data")
                    price_data = csv_data
                else:
                    # Check for overlap to align data
                    api_first_date = api_data.index[0].date()
                    
                    if api_first_date <= csv_last_date:
                        # We have overlap, find the cutoff point
                        logger.info(f"Data overlap found from {api_first_date} to {csv_last_date}")
                        
                        # Remove overlapping days from CSV data to avoid duplicates
                        csv_data = csv_data[csv_data.index.date < api_first_date]
                        logger.info(f"Removed overlapping data from CSV, now has {len(csv_data)} days")
                    
                    # Now combine the datasets
                    logger.info(f"Combining CSV data (until {csv_last_date}) with API data (from {api_first_date} to {api_last_date})")
                    
                    # Ensure API data has normalized_price column
                    price_mean = csv_data.attrs['price_mean']
                    price_std = csv_data.attrs['price_std']
                    api_data['normalized_price'] = (api_data['price'] - price_mean) / price_std
                    
                    # Combine the dataframes
                    combined_data = pd.concat([csv_data, api_data])
                    
                    # Copy attributes from CSV data
                    for attr_name in csv_data.attrs:
                        combined_data.attrs[attr_name] = csv_data.attrs[attr_name]
                    
                    # Take the most recent lookback_days
                    if len(combined_data) > lookback_days:
                        combined_data = combined_data.iloc[-lookback_days:]
                        logger.info(f"Trimmed combined data to last {lookback_days} days")
                    
                    logger.info(f"Created combined dataset with {len(combined_data)} data points")
                    price_data = combined_data
        else:
            # If we only have API data and no CSV
            logger.warning("No CSV data found, using API data only")
            
            # Normalize the data
            price_mean = api_data['price'].mean()
            price_std = api_data['price'].std()
            api_data['normalized_price'] = (api_data['price'] - price_mean) / price_std
            
            # Add metadata to DataFrame
            _, base_symbol, quote_currency = convert_symbol_for_api(symbol)
            price_denomination = "BTC" if "BTC" in quote_currency.upper() else "USD"
            currency_symbol = "₿" if price_denomination == "BTC" else "$"
            
            api_data.attrs['currency_symbol'] = currency_symbol
            api_data.attrs['price_denomination'] = price_denomination
            api_data.attrs['mean_price'] = price_mean
            api_data.attrs['median_price'] = api_data['price'].median()
            api_data.attrs['filename'] = f"{symbol}_API_Data"
            api_data.attrs['price_mean'] = price_mean
            api_data.attrs['price_std'] = price_std
            
            price_data = api_data
    
    return price_data

def prepare_datasets(data, context_window=30, prediction_length=14, test_size=0.2):
    """
    Prepare train and test datasets for time series forecasting.
    
    Args:
        data (pd.DataFrame): Historical price data with 'normalized_price' column
        context_window (int): Number of days to use as features
        prediction_length (int): Number of days to predict ahead
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, validation_data)
    """
    logger.info(f"Preparing datasets with context_window={context_window}, prediction_length={prediction_length}")
    
    # Create features and target
    X, y = [], []
    
    # Use normalized_price for model input if available, otherwise use price
    if 'normalized_price' in data.columns:
        price_column = 'normalized_price'
    else:
        price_column = 'price'
    
    # Create sliding windows for features
    for i in range(len(data) - context_window - prediction_length + 1):
        # Extract context window
        window = data[price_column].iloc[i:i + context_window].values
        
        # Add features - can add more sophisticated feature engineering here
        features = []
        
        # Use past prices as features
        features.extend(window)
        
        # Add simple features like moving averages, volatility
        ma7 = np.mean(window[-7:]) if len(window) >= 7 else np.mean(window)
        ma14 = np.mean(window[-14:]) if len(window) >= 14 else np.mean(window)
        ma30 = np.mean(window) if len(window) >= 30 else np.mean(window)
        
        # Volatility (standard deviation)
        vol7 = np.std(window[-7:]) if len(window) >= 7 else np.std(window)
        vol14 = np.std(window[-14:]) if len(window) >= 14 else np.std(window)
        
        # Price momentum (rate of change)
        momentum = (window[-1] - window[0]) / window[0] if window[0] != 0 else 0
        
        # Add calculated features
        features.extend([ma7, ma14, ma30, vol7, vol14, momentum])
        
        # For multi-step forecasting, target is the next prediction_length prices
        target = data[price_column].iloc[i + context_window:i + context_window + prediction_length].values
        
        X.append(features)
        y.append(target)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    train_size = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Created {len(X_train)} training samples and {len(X_test)} testing samples")
    
    # Prepare validation data for final prediction
    last_window = data[price_column].iloc[-context_window:].values
    
    # Create the same features for validation
    validation_features = []
    validation_features.extend(last_window)
    
    ma7 = np.mean(last_window[-7:]) if len(last_window) >= 7 else np.mean(last_window)
    ma14 = np.mean(last_window[-14:]) if len(last_window) >= 14 else np.mean(last_window)
    ma30 = np.mean(last_window) if len(last_window) >= 30 else np.mean(last_window)
    
    vol7 = np.std(last_window[-7:]) if len(last_window) >= 7 else np.std(last_window)
    vol14 = np.std(last_window[-14:]) if len(last_window) >= 14 else np.std(last_window)
    
    momentum = (last_window[-1] - last_window[0]) / last_window[0] if last_window[0] != 0 else 0
    
    validation_features.extend([ma7, ma14, ma30, vol7, vol14, momentum])
    
    # Scale validation features
    validation_features = scaler.transform(np.array([validation_features]))
    
    # Actual future values if available
    actual_future = None
    if len(data) >= context_window + prediction_length:
        actual_future = data[price_column].iloc[-prediction_length:].values
    
    validation_data = {
        'features': validation_features,
        'actual_future': actual_future
    }
    
    return X_train, X_test, y_train, y_test, scaler, validation_data

def create_random_forest_model(X_train, y_train):
    """
    Create and train a Random Forest model for crypto price prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        model: Trained Random Forest model
    """
    logger.info("Training Random Forest model...")
    
    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("Random Forest model training complete")
    
    return model

def evaluate_model(model, X_test, y_test, data, validation_data, prediction_length):
    """
    Evaluate the trained model and make future predictions.
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test targets
        data: Original DataFrame
        validation_data: Validation data dictionary
        prediction_length: Number of days to predict ahead
        
    Returns:
        dict: Evaluation results
    """
    logger.info("Evaluating Random Forest model...")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test.reshape(-1), y_pred.reshape(-1))
    rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1)))
    
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    # Make future predictions
    future_normalized = model.predict(validation_data['features'])[0]
    
    # Convert normalized predictions back to original scale
    price_mean = data.attrs['price_mean']
    price_std = data.attrs['price_std']
    
    future_prices = future_normalized * price_std + price_mean
    
    # Calculate actual future prices if available
    actual_future_prices = None
    if validation_data['actual_future'] is not None:
        actual_future_normalized = validation_data['actual_future']
        actual_future_prices = actual_future_normalized * price_std + price_mean
    
    evaluation_results = {
        'mae': mae,
        'rmse': rmse,
        'future_forecast': future_prices,
        'actual_future': actual_future_prices
    }
    
    return evaluation_results

def create_forecast_plot(data, evaluation_results, prediction_length):
    """
    Create a plot of historical data and forecast.
    
    Args:
        data: Original DataFrame with price data
        evaluation_results: Results from model evaluation
        prediction_length: Number of days in forecast horizon
        
    Returns:
        str: Path to saved plot
    """
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
    
    # Generate future dates
    last_date = data.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(prediction_length)]
    
    # Plot forecast
    future_forecast = evaluation_results.get('future_forecast')
    if future_forecast is not None:
        plt.plot(forecast_dates, future_forecast, 'g-', linewidth=2, label='Random Forest Forecast')
    
    # Plot actual future values if available
    actual_future = evaluation_results.get('actual_future')
    if actual_future is not None:
        plt.plot(forecast_dates, actual_future, 'k--', linewidth=2, label='Actual Values')
    
    # Add metrics as text in the plot
    if 'mae' in evaluation_results and 'rmse' in evaluation_results:
        metrics_text = (
            f"Test MAE: {evaluation_results['mae']:.4f}\n"
            f"Test RMSE: {evaluation_results['rmse']:.4f}"
        )
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Format plot
    title = f"{symbol} Random Forest Forecast"
    plt.title(title, fontsize=16, fontweight='bold')
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
    filename = f"plots/{symbol}_random_forest_forecast_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Forecast plot saved to {filename}")
    plt.close()
    
    return filename

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Cryptocurrency Forecaster with Random Forest")
    
    # Basic options
    parser.add_argument("--symbol", type=str, default="BTC_USD", help="Cryptocurrency symbol to forecast")
    parser.add_argument("--days-ahead", type=int, default=32, help="Number of days to forecast")
    parser.add_argument("--lookback", type=int, default=3650, help="Days of historical data to use")
    parser.add_argument("--context-window", type=int, default=60, help="Context window length for features")
    
    args = parser.parse_args()
    
    try:
        # Load combined data from CSV and API (no sentiment)
        data = load_combined_data(args.symbol, lookback_days=args.lookback, api_days=365)
        
        if data is None:
            logger.error(f"Failed to load data for {args.symbol}")
            return 1
        
        logger.info(f"Preparing datasets for {args.symbol}")
        
        # Prepare datasets
        X_train, X_test, y_train, y_test, scaler, validation_data = prepare_datasets(
            data, 
            context_window=args.context_window,
            prediction_length=args.days_ahead
        )
        
        # Create and train model
        model = create_random_forest_model(X_train, y_train)
        
        # Evaluate model and make predictions
        evaluation_results = evaluate_model(
            model,
            X_test,
            y_test,
            data,
            validation_data,
            args.days_ahead
        )
        
        # Create forecast plot
        plot_path = create_forecast_plot(
            data, 
            evaluation_results, 
            args.days_ahead
        )
        
        # Print results
        print("\n===== RANDOM FOREST FORECASTER RESULTS =====")
        print(f"Symbol: {args.symbol}")
        print(f"Context Window: {args.context_window}")
        print(f"Days Ahead: {args.days_ahead}")
        print(f"Historical Data: {len(data)} days")
        
        # Print metrics
        print("\nError Metrics:")
        print(f"  Test MAE: {evaluation_results['mae']:.4f}")
        print(f"  Test RMSE: {evaluation_results['rmse']:.4f}")
        
        # Print forecast summary
        future_forecast = evaluation_results.get('future_forecast')
        if future_forecast is not None:
            current_price = data['price'].iloc[-1]
            final_price = future_forecast[-1]
            change_pct = ((final_price / current_price) - 1) * 100
            
            print("\nForecast Summary:")
            print(f"  Current Price: {current_price:.2f}")
            print(f"  Forecasted Price ({args.days_ahead} days): {final_price:.2f}")
            print(f"  Forecasted Change: {change_pct:.2f}%")
            
            if change_pct > 5:
                trend = "Strongly Bullish"
            elif change_pct > 1:
                trend = "Bullish"
            elif change_pct < -5:
                trend = "Strongly Bearish"
            elif change_pct < -1:
                trend = "Bearish"
            else:
                trend = "Neutral"
                
            print(f"  Market Trend: {trend}")
        
        print(f"\nForecast plot: {plot_path}")
        print("================================\n")
        
        return 0
            
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())