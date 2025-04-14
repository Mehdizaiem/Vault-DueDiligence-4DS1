#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos Cryptocurrency Forecasting Demo

A simplified demo script to forecast cryptocurrency prices using Amazon's Chronos model.
This version has minimum dependencies and finds data files automatically.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_project_root():
    """Find the project root directory that contains the data folder."""
    # Start from the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Look up the directory tree for the data folder
    test_dirs = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.dirname(os.path.dirname(current_dir))
    ]
    
    for directory in test_dirs:
        # Check if 'data' exists in this directory
        data_dir = os.path.join(directory, 'data')
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            return directory
    
    # If we didn't find it, default to the current directory
    return current_dir

# Find project root
PROJECT_ROOT = find_project_root()
logger.info(f"Using project root: {PROJECT_ROOT}")

# Add project root to path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

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

def find_csv_file(symbol):
    """
    Find a CSV file for the given symbol using a more robust approach.
    
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
        lookback_days (int): Number of days of historical data to use
        
    Returns:
        pd.DataFrame: Processed DataFrame with price data
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
            df['price'] = df[price_col]
        
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

def run_chronos_forecast(data, model_name, days_ahead, num_samples=100):
    """
    Generate cryptocurrency forecasts using Chronos.
    
    Args:
        data (pd.DataFrame): Historical price data
        model_name (str): Chronos model to use
        days_ahead (int): Number of days to forecast
        num_samples (int): Number of samples for uncertainty quantification
        
    Returns:
        tuple: (forecast_results, plot_path)
    """
    try:
        # Import Chronos
        from chronos import BaseChronosPipeline
        
        # Determine device
        if torch.cuda.is_available():
            device_map = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"
        
        # Determine precision
        if device_map == "cuda" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        
        # Load model
        logger.info(f"Loading Chronos model: {model_name} on {device_map}")
        pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        
        # Prepare data for Chronos
        price_data = data['price'].values
        context = torch.tensor(price_data, dtype=torch.float32)
        
        # Generate forecast
        logger.info(f"Generating {days_ahead}-day forecast")
        samples = pipeline.predict(
            context=context,
            prediction_length=days_ahead,
            num_samples=num_samples
        )
        
        # Generate quantiles
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=days_ahead,
            quantile_levels=[0.1, 0.5, 0.9]  # 80% prediction interval
        )
        
        # Get forecast dates
        last_date = data.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create results dictionary
        forecast_results = {
            'samples': samples.cpu().numpy(),
            'quantiles': quantiles.cpu().numpy(),
            'mean': mean.cpu().numpy(),
            'dates': forecast_dates,
            'quantile_levels': [0.1, 0.5, 0.9]
        }
        
        # Create visualization
        plot_path = create_forecast_plot(data, forecast_results)
        
        return forecast_results, plot_path
        
    except Exception as e:
        logger.error(f"Error in Chronos forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def create_forecast_plot(data, forecast_results):
    """
    Create visualization of forecast results.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast_results (dict): Results from Chronos forecast
        
    Returns:
        str: Path to saved plot
    """
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Get metadata from DataFrame
    symbol = data.attrs.get('filename', 'Unknown').split('.')[0]
    currency_symbol = data.attrs.get('currency_symbol', '$')
    price_denomination = data.attrs.get('price_denomination', 'USD')
    mean_price = data.attrs.get('mean_price', 0)
    
    # Extract forecast components
    dates = forecast_results['dates']
    quantiles = forecast_results['quantiles'][0]
    samples = forecast_results['samples'][0]
    
    # Get median and bounds
    lower_bound = quantiles[:, 0]  # 10th percentile
    median_forecast = quantiles[:, 1]  # 50th percentile
    upper_bound = quantiles[:, 2]  # 90th percentile
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 90 days to keep it readable)
    history_len = min(90, len(data))
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
        label='80% Prediction Interval'
    )
    
    # Add some sample paths
    num_paths = min(10, samples.shape[0])
    for i in range(num_paths):
        plt.plot(dates, samples[i], 'r-', linewidth=0.5, alpha=0.3)
    
    # Format plot based on price denomination
    if price_denomination == 'BTC':
        title = f"{symbol} Price Forecast (in BTC)"
        plt.ylabel('Price (BTC)', fontsize=12)
        
        # Format y-axis with BTC symbol
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'₿{x:.6f}'))
        
    else:
        title = f"{symbol} Price Forecast (in USD)"
        plt.ylabel('Price (USD)', fontsize=12)
        
        # Format y-axis based on price magnitude
        from matplotlib.ticker import FuncFormatter
        if mean_price < 0.1:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.4f}'))
        elif mean_price < 1.0:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.2f}'))
        else:
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
    
    # Common formatting
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for last observed and forecast
    current_price = data['price'].iloc[-1]
    final_forecast = median_forecast[-1]
    change_pct = ((final_forecast / current_price) - 1) * 100
    
    # Format price based on denomination
    if price_denomination == 'BTC':
        current_price_str = f"₿{current_price:.6f}"
        forecast_str = f"₿{final_forecast:.6f}"
    else:
        if mean_price < 0.1:
            current_price_str = f"${current_price:.4f}"
            forecast_str = f"${final_forecast:.4f}"
        else:
            current_price_str = f"${current_price:.2f}"
            forecast_str = f"${final_forecast:.2f}"
    
    # Add annotations
    plt.annotate(
        f"Last observed: {current_price_str}",
        (data.index[-1], current_price),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.annotate(
        f"Forecast: {forecast_str} ({change_pct:+.2f}%)",
        (dates[-1], final_forecast),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"plots/{symbol}_forecast_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Forecast plot saved to {filename}")
    plt.close()
    
    return filename

def generate_insights(data, forecast_results):
    """
    Generate market insights from forecast results.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast_results (dict): Results from Chronos forecast
        
    Returns:
        dict: Market insights
    """
    # Get metadata from DataFrame
    symbol = data.attrs.get('filename', 'Unknown').split('.')[0]
    currency_symbol = data.attrs.get('currency_symbol', '$')
    price_denomination = data.attrs.get('price_denomination', 'USD')
    
    # Get current price
    current_price = data['price'].iloc[-1]
    
    # Extract forecast components
    dates = forecast_results['dates']
    quantiles = forecast_results['quantiles'][0]
    samples = forecast_results['samples'][0]
    
    # Get median and bounds
    lower_bound = quantiles[:, 0]  # 10th percentile
    median_forecast = quantiles[:, 1]  # 50th percentile
    upper_bound = quantiles[:, 2]  # 90th percentile
    
    # Calculate metrics
    final_forecast = median_forecast[-1]
    change_pct = ((final_forecast / current_price) - 1) * 100
    
    # Calculate uncertainty
    uncertainty = (upper_bound - lower_bound) / median_forecast * 100
    avg_uncertainty = np.mean(uncertainty)
    
    # Calculate probability of price increase
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
        insight = f"Chronos forecasts a {trend} outlook for {symbol} with a projected increase of {change_pct:.2f}% over the next {len(median_forecast)} days."
    elif trend in ["strongly bearish", "bearish"]:
        insight = f"Chronos forecasts a {trend} outlook for {symbol} with a projected decrease of {abs(change_pct):.2f}% over the next {len(median_forecast)} days."
    else:
        insight = f"Chronos forecasts a {trend} outlook for {symbol} with minimal price movement (expected change: {change_pct:.2f}%) over the next {len(median_forecast)} days."
        
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
    
    # Format price strings based on denomination
    if price_denomination == 'BTC':
        current_price_str = f"₿{current_price:.6f}"
        final_forecast_str = f"₿{final_forecast:.6f}"
    else:
        if current_price < 0.1:
            current_price_str = f"${current_price:.4f}"
            final_forecast_str = f"${final_forecast:.4f}"
        else:
            current_price_str = f"${current_price:.2f}"
            final_forecast_str = f"${final_forecast:.2f}"
    
    # Compile insights
    results = {
        "symbol": symbol,
        "current_price": current_price,
        "current_price_str": current_price_str,
        "final_forecast": final_forecast,
        "final_forecast_str": final_forecast_str,
        "change_pct": change_pct,
        "trend": trend,
        "probability_increase": prob_increase,
        "average_uncertainty": avg_uncertainty,
        "insight": insight,
        "price_denomination": price_denomination,
        "currency_symbol": currency_symbol
    }
    
    return results

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
    
    # Check GPU availability
    check_cuda_availability()
    
    # Create output directories
    os.makedirs("plots", exist_ok=True)
    
    # Find and load data
    csv_file = find_csv_file(args.symbol)
    if csv_file is None:
        logger.error(f"Could not find data file for {args.symbol}")
        return 1
    
    # Load data
    data = load_csv_data(csv_file, args.lookback)
    if data is None:
        logger.error(f"Failed to load data for {args.symbol}")
        return 1
    
    # Run forecast
    forecast_results, plot_path = run_chronos_forecast(
        data, 
        args.model, 
        args.days_ahead,
        args.samples
    )
    
    if forecast_results is None:
        logger.error("Failed to generate forecast")
        return 1
    
    # Generate insights
    insights = generate_insights(data, forecast_results)
    
    # Print forecast summary
    print("\n===== CHRONOS FORECAST SUMMARY =====")
    print(f"Symbol: {insights['symbol']}")
    print(f"Model: {args.model}")
    print(f"Forecast horizon: {args.days_ahead} days")
    print(f"Current price: {insights['current_price_str']}")
    print(f"Final forecast: {insights['final_forecast_str']} ({insights['change_pct']:+.2f}%)")
    print(f"Forecast trend: {insights['trend']}")
    print(f"Probability of increase: {insights['probability_increase']:.1f}%")
    print(f"Average uncertainty: ±{insights['average_uncertainty']:.1f}%")
    print(f"Forecast visualization: {plot_path}")
    print("\nInsight:")
    print(insights['insight'])
    print("===================================\n")
    
    return 0

def main():
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Chronos Cryptocurrency Forecasting Demo")
    
    # Basic options
    parser.add_argument("--symbol", type=str, default="BTC", help="Cryptocurrency symbol to forecast")
    parser.add_argument("--days-ahead", type=int, default=30, help="Number of days to forecast")
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