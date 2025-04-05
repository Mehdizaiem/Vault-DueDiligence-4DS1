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
        # Extract price data
        if 'price' in data.columns:
            price_data = data['price'].values
        elif 'close' in data.columns:
            price_data = data['close'].values
        else:
            raise ValueError("No price or close column found in data")
        
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
        if 'price' in data.columns:
            price_series = data['price']
        elif 'close' in data.columns:
            price_series = data['close']
        else:
            raise ValueError("No price or close column found in data")
        
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
        if 'price' in data.columns:
            price_series = data['price']
        elif 'close' in data.columns:
            price_series = data['close']
        else:
            raise ValueError("No price or close column found in data")
            
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
    Download or load crypto price data for a given symbol
    
    Args:
        symbol (str): Crypto symbol (e.g., "BTC", "ETH")
        lookback_days (int): Number of days of historical data
        
    Returns:
        pd.DataFrame: Market data with datetime index
    """
    try:
        # Try to use yfinance to get data
        import yfinance as yf
        ticker = f"{symbol}-USD"
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if len(df) > 0:
            # Rename columns to lower case
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Successfully downloaded {len(df)} days of data for {symbol}")
            return df
    except Exception as e:
        logger.warning(f"Error downloading data with yfinance: {e}")
    
    # If yfinance fails, try to use a sample dataset or ask user to provide data
    logger.warning(f"Cannot download {symbol} data. Using synthetic data for demonstration.")
    
    # Generate synthetic data that resembles crypto prices
    dates = pd.date_range(end=datetime.now(), periods=lookback_days)
    
    # Start with a reasonable price for the chosen crypto
    if symbol.upper() == "BTC":
        base_price = 50000
    elif symbol.upper() == "ETH":
        base_price = 3000
    else:
        base_price = 100
    
    # Generate a random walk with drift and volatility that resembles crypto
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.0005, 0.02, lookback_days)  # Daily returns with positive drift
    
    # Add some volatility clusters to simulate crypto behavior
    volatility_factor = np.ones(lookback_days)
    for i in range(3):  # Create 3 volatility clusters
        cluster_start = np.random.randint(0, lookback_days - 30)
        cluster_length = np.random.randint(10, 30)
        volatility_factor[cluster_start:cluster_start+cluster_length] = np.random.uniform(1.5, 3.0)
    
    returns = returns * volatility_factor
    
    # Convert returns to prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Create synthetic OHLC data
    synthetic_data = {
        'open': prices * np.random.uniform(0.98, 1.02, lookback_days),
        'high': prices * np.random.uniform(1.01, 1.05, lookback_days),
        'low': prices * np.random.uniform(0.95, 0.99, lookback_days),
        'close': prices,
        'volume': np.random.lognormal(15, 1, lookback_days)  # Log-normal distribution for volume
    }
    
    df = pd.DataFrame(synthetic_data, index=dates)
    logger.info(f"Created synthetic data with {len(df)} days for {symbol}")
    
    return df

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
    
    # Load cryptocurrency data
    data = prepare_crypto_data(args.symbol, args.lookback)
    
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

if __name__ == "__main__":
    main()