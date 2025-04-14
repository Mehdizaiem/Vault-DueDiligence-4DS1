# chronos_crypto_forecaster.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
import logging
import os
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if chronos is installed, if not install it
try:
    from chronos import BaseChronosPipeline
except ImportError:
    logger.info("Installing chronos-forecasting package...")
    import subprocess
    subprocess.check_call(["pip", "install", "chronos-forecasting"])
    from chronos import BaseChronosPipeline

class ChronosForecaster:
    """
    Cryptocurrency forecaster using Amazon's Chronos model
    """
    
    def __init__(self, model_name="amazon/chronos-t5-small", use_gpu=True, precision="bfloat16"):
        """
        Initialize the forecaster with a pre-trained Chronos model
        
        Args:
            model_name (str): The name of the Chronos model to use
            use_gpu (bool): Whether to use GPU for inference
            precision (str): Precision for model (bfloat16, float16, or float32)
        """
        logger.info(f"Initializing Chronos forecaster with model: {model_name}")
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            device_map = "cuda"
            logger.info("Using GPU for inference")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"  # Apple Silicon GPU
            logger.info("Using Apple Silicon GPU for inference")
        else:
            device_map = "cpu"
            logger.info("Using CPU for inference")
        
        # Set precision
        if precision == "bfloat16" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif precision == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
            
        # Load pre-trained model
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        logger.info("Chronos model loaded successfully")
    
    def preprocess_data(self, data):
        """
        Preprocess market data for the Chronos model
        
        Args:
            data (pd.DataFrame): DataFrame with price data
             
        Returns:
            torch.Tensor: Processed data ready for the model
        """
        # Check if data is None or empty
        if data is None or len(data) == 0:
            raise ValueError("No market data provided")
        
        # Try multiple column names for price
        price_columns = ['price', 'close', 'Price', 'Close']
        price_col = None
        
        for col in price_columns:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found. Tried: {price_columns}")
        
        # Extract price data
        price_data = data[price_col].values
        
        # Check for NaN values
        if np.isnan(price_data).any():
            logger.warning("NaN values found in data, filling with forward fill then backward fill")
            price_data = pd.Series(price_data).fillna(method='ffill').fillna(method='bfill').values
        
        # Convert to tensor
        return torch.tensor(price_data, dtype=torch.float32)
    
    def forecast(self, data, prediction_length=7, num_samples=100, quantile_levels=None):
        """
        Generate forecasts using the Chronos model
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            prediction_length (int): Number of time steps to forecast
            num_samples (int): Number of samples to draw
            quantile_levels (list): Quantile levels for prediction intervals
            
        Returns:
            dict: Forecast results including samples, quantiles, mean, dates
        """
        if quantile_levels is None:
            quantile_levels = [0.1, 0.5, 0.9]  # Default 80% prediction interval
        
        # Preprocess data
        context = self.preprocess_data(data)
        logger.info(f"Forecasting {prediction_length} steps with context length {len(context)}")
        
        # Generate full sample distribution
        samples = self.pipeline.predict(
            context=context, 
            prediction_length=prediction_length,
            num_samples=num_samples
        )
        
        # Generate quantiles from samples
        quantiles, mean = self.pipeline.predict_quantiles(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels
        )
        
        # Get forecast dates
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(prediction_length)]
        
        # Prepare results
        results = {
            'samples': samples.cpu().numpy(),
            'quantiles': quantiles.cpu().numpy(),
            'mean': mean.cpu().numpy(),
            'dates': forecast_dates,
            'quantile_levels': quantile_levels
        }
        
        return results
    
    def plot_forecast(self, data, forecast_results, symbol, output_path=None):
        """
        Plot forecast results with historical data
        
        Args:
            data (pd.DataFrame): Historical price data
            forecast_results (dict): Results from forecast method
            symbol (str): Trading symbol
            output_path (str): Path to save plot or None
            
        Returns:
            str: Path to saved plot or None
        """
        # Extract price data
        price_columns = ['price', 'close', 'Price', 'Close']
        price_col = None
        
        for col in price_columns:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found. Tried: {price_columns}")
        
        price_series = data[price_col]
        
        # Extract forecast components
        dates = forecast_results['dates']
        mean = forecast_results['mean'][0]
        quantiles = forecast_results['quantiles'][0]
        quantile_levels = forecast_results['quantile_levels']
        
        # Get the median and prediction interval
        q_low_idx = quantile_levels.index(min(quantile_levels))
        q_mid_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
        q_high_idx = quantile_levels.index(max(quantile_levels))
        
        low = quantiles[:, q_low_idx]
        median = quantiles[:, q_mid_idx]
        high = quantiles[:, q_high_idx]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot historical data
        plt.plot(data.index, price_series, 'b-', linewidth=2, label='Historical Price')
        
        # Plot forecast
        plt.plot(dates, median, 'r-', linewidth=2, label='Median Forecast')
        
        # Plot prediction interval
        plt.fill_between(
            dates,
            low,
            high,
            color='red',
            alpha=0.2,
            label=f'{int((max(quantile_levels) - min(quantile_levels)) * 100)}% Prediction Interval'
        )
        
        # Add some sample paths from the full distribution
        samples = forecast_results['samples'][0]
        num_paths = min(10, samples.shape[0])
        for i in range(num_paths):
            plt.plot(dates, samples[i], 'r-', linewidth=0.5, alpha=0.3)
        
        # Format plot
        plt.title(f'{symbol} Price Forecast (Chronos)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
        
        # Add annotations for forecast starting point
        plt.annotate(
            f"Last observed: ${price_series.iloc[-1]:.2f}",
            (data.index[-1], price_series.iloc[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Add annotation for median forecast endpoint
        plt.annotate(
            f"Forecast: ${median[-1]:.2f}",
            (dates[-1], median[-1]),
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
        
        # Save plot if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {output_path}")
        else:
            # Save to default location
            os.makedirs('plots', exist_ok=True)
            filename = f"plots/{symbol}_chronos_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {filename}")
            output_path = filename
        
        plt.close()
        return output_path
    
    def generate_market_insights(self, data, forecast_results, symbol):
        """
        Generate market insights from forecast results
        
        Args:
            data (pd.DataFrame): Historical price data
            forecast_results (dict): Results from forecast method
            symbol (str): Trading symbol
            
        Returns:
            dict: Market insights
        """
        # Extract price data
        price_columns = ['price', 'close', 'Price', 'Close']
        price_col = None
        
        for col in price_columns:
            if col in data.columns:
                price_col = col
                break
        
        if price_col is None:
            raise ValueError(f"No price column found. Tried: {price_columns}")
        
        price_series = data[price_col]
            
        # Get current price and calculate metrics
        current_price = price_series.iloc[-1]
        
        # Extract forecast components
        quantiles = forecast_results['quantiles'][0]
        mean = forecast_results['mean'][0]
        quantile_levels = forecast_results['quantile_levels']
        
        # Get median and prediction interval
        low_idx = quantile_levels.index(min(quantile_levels))
        med_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
        high_idx = quantile_levels.index(max(quantile_levels))
        
        # Calculate forecast metrics
        median_forecast = quantiles[:, med_idx]
        final_forecast = median_forecast[-1]
        change_pct = ((final_forecast / current_price) - 1) * 100
        
        # Calculate uncertainty
        uncertainty = (quantiles[:, high_idx] - quantiles[:, low_idx]) / median_forecast * 100
        avg_uncertainty = np.mean(uncertainty)
        
        # Get prediction trajectories
        samples = forecast_results['samples'][0]
        
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
        
        # Compile insights
        results = {
            "symbol": symbol,
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

def prepare_crypto_data(symbol, lookback_days=365):
    """
    Fetch historical market data for a specific cryptocurrency symbol
    
    Args:
        symbol (str): Crypto symbol (e.g., "BTCUSDT", "ETHUSDT")
        lookback_days (int): Number of days of historical data to fetch
        
    Returns:
        pd.DataFrame: Market data with timestamp index
    """
    try:
        import os
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Determine data directory
        data_dir = os.path.join(os.getcwd(), 'data', 'time series cryptos')
        
        # Find CSV files matching the symbol
        csv_files = [f for f in os.listdir(data_dir) if f.lower().startswith(symbol.lower()) and f.endswith('.csv')]
        
        if not csv_files:
            logger.error(f"No CSV file found for symbol {symbol}")
            return None
        
        # Use the first matching file
        csv_path = os.path.join(data_dir, csv_files[0])
        logger.info(f"Loading data from: {csv_path}")
        
        # Read the first row to inspect columns
        sample_data = pd.read_csv(csv_path, nrows=1)
        logger.info(f"CSV columns: {list(sample_data.columns)}")
        
        # Read CSV file with thousands separator specified
        df = pd.read_csv(csv_path, thousands=',')
        
        # Log the data type of the 'Price' column to confirm
        logger.info(f"Price column type after reading: {df['Price'].dtype}")
        
        # Check for date/timestamp column
        date_column = None
        date_candidates = ['Date', 'date', 'Timestamp', 'timestamp', 'Time', 'time']
        for col in date_candidates:
            if col in df.columns:
                date_column = col
                break
        
        if date_column:
            df['timestamp'] = pd.to_datetime(df[date_column], errors='coerce')
        else:
            logger.warning("No date column found, creating synthetic timestamps")
            df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        
        # Look for price column with more flexible matching
        price_column = None
        price_candidates = ['Close', 'close', 'Price', 'price', 'Last', 'last', 'Value', 'value']
        
        for col in price_candidates:
            if col in df.columns:
                price_column = col
                logger.info(f"Found price column: {col}")
                break
        
        # If no direct price column, try to identify it by looking for columns with numeric data
        if price_column is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            logger.info(f"Numeric columns: {list(numeric_cols)}")
            
            # Exclude volume or other non-price columns
            exclude_keywords = ['volume', 'vol', 'qty', 'quantity', 'market cap', 'marketcap', 'open', 'high', 'low']
            potential_price_cols = [col for col in numeric_cols if not any(kw in col.lower() for kw in exclude_keywords)]
            
            if potential_price_cols:
                # Use the first potential price column
                price_column = potential_price_cols[0]
                logger.info(f"Using {price_column} as price column")
        
        # If still no price column found, look for the column that might contain price data
        if price_column is None:
            # Try to identify a price column by looking for values typical of cryptocurrency prices
            for col in df.select_dtypes(include=['number']).columns:
                col_mean = df[col].mean()
                # Crypto prices typically range from <$1 to tens of thousands
                if 0.01 <= col_mean <= 100000:
                    price_column = col
                    logger.info(f"Guessing {price_column} as price column (mean value: {col_mean})")
                    break
        
        # If we still don't have a price column, as a last resort, pick the first numeric column
        if price_column is None and len(df.select_dtypes(include=['number']).columns) > 0:
            price_column = df.select_dtypes(include=['number']).columns[0]
            logger.warning(f"No ideal price column identified, using {price_column} as fallback")
        
        if price_column is None:
            logger.error("No valid price column found in CSV")
            return None
        
        # Create standardized 'price' column
        df['price'] = df[price_column]
        
        # Set timestamp as index and sort
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Handle duplicated indices
        if df.index.duplicated().any():
            logger.warning("Duplicate timestamps found, keeping last occurrence")
            df = df[~df.index.duplicated(keep='last')]
        
        # Limit to lookback days
        if len(df) > lookback_days:
            df = df.tail(lookback_days)
        
        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # Log loaded data details
        logger.info(f"Loaded {len(df)} data points for {symbol}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Price range: {df['price'].min()} to {df['price'].max()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in prepare_crypto_data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency forecasting with Amazon Chronos")
    parser.add_argument("--symbol", type=str, default="BTC", help="Cryptocurrency symbol (e.g., BTC, ETH)")
    parser.add_argument("--days_ahead", type=int, default=14, help="Number of days to forecast")
    parser.add_argument("--model", type=str, default="amazon/chronos-t5-small", 
                        choices=["amazon/chronos-t5-tiny", "amazon/chronos-t5-mini", 
                                "amazon/chronos-t5-small", "amazon/chronos-t5-base", 
                                "amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini", 
                                "amazon/chronos-bolt-small"],
                        help="Chronos model to use")
    parser.add_argument("--lookback", type=int, default=365, help="Days of historical data to use")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")
    parser.add_argument("--output", type=str, help="Output path for forecast plot")
    
    args = parser.parse_args()
    
    try:
        # Load cryptocurrency data
        data = prepare_crypto_data(args.symbol, args.lookback)
        
        # Validate loaded data
        if data is None or len(data) == 0:
            logger.error(f"Failed to load data for {args.symbol}")
            return 1
        
        # Initialize forecaster
        forecaster = ChronosForecaster(
            model_name=args.model,
            use_gpu=args.gpu
        )
        
        # Generate forecast
        forecast_results = forecaster.forecast(
            data,
            prediction_length=args.days_ahead,
            num_samples=100,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Plot forecast
        plot_path = forecaster.plot_forecast(
            data,
            forecast_results,
            args.symbol,
            output_path=args.output
        )
        
        # Generate insights
        insights = forecaster.generate_market_insights(
            data,
            forecast_results,
            args.symbol
        )
        
        # Print insights
        print("\n--------- CHRONOS FORECAST INSIGHTS ---------")
        print(f"Symbol: {args.symbol}")
        print(f"Current Price: ${insights['current_price']:.2f}")
        print(f"Forecast (in {args.days_ahead} days): ${insights['final_forecast']:.2f} ({insights['change_pct']:+.2f}%)")
        print(f"Trend: {insights['trend'].upper()}")
        print(f"Probability of price increase: {insights['probability_increase']:.1f}%")
        print(f"Average forecast uncertainty: ±{insights['average_uncertainty']:.1f}%")
        print("\nInsight:")
        print(insights['insight'])
        print("\nForecast plot saved to:", plot_path)
        print("----------------------------------------------")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in main forecasting process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())