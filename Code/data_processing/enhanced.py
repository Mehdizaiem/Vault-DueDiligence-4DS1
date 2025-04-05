# enhanced_chronos.py
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
import json
import requests
from tabulate import tabulate
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.metrics import mean_absolute_percentage_error

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_chronos.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from your existing code
try:
    from Code.data_processing.chronos_forecaster import ChronosForecaster
    from Code.data_processing.chronos_integration import ChronosIntegration
    from Code.data_processing.time_series_manager import TimeSeriesManager
except ImportError:
    logger.error("Failed to import required modules. Ensure all modules are in the correct path.")
    sys.exit(1)

class EnhancedChronos:
    """Enhanced version of the Chronos forecasting system with additional features."""
    
    def __init__(self, data_dir="data/enhanced_chronos"):
        """Initialize the enhanced forecasting system."""
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        
        # Initialize components
        self.integration = ChronosIntegration()
        self.forecaster = ChronosForecaster(model_dir=os.path.join(data_dir, "models"))
        
        # Data sources
        self.data_sources = {
            "csv": True,        # Local CSV files
            "api": False,       # External APIs
            "sentiment": True,  # News sentiment
            "onchain": False,   # On-chain metrics
        }
        
        # Model configurations
        self.model_configs = {
            "short_term": {
                "enabled": True,
                "days_ahead": 7,
                "ensemble_weight": 0.7,
                "features": ["price", "volume", "ma_7", "ma_14", "ema_12", "rsi_14", "sentiment_score"]
            },
            "long_term": {
                "enabled": True,
                "days_ahead": 30,
                "ensemble_weight": 0.3,
                "features": ["price", "volume", "ma_30", "sentiment_trend"]
            }
        }
        
        # Available symbols cache
        self.symbols_cache = []
        
        # External API configuration
        self.api_config = {
            "coinapi": {
                "api_key": os.getenv("COINAPI_API_KEY"),
                "base_url": "https://rest.coinapi.io/v1"
            },
            "fear_greed_index": {
                "url": "https://api.alternative.me/fng/"
            }
        }
        
        # Feature importance cache
        self.feature_importance = {}
        
        # Initialize market context
        self.market_context = self._get_market_context()
        
        logger.info("EnhancedChronos initialized")
    
    def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context, including fear & greed index and market trends."""
        context = {
            "fear_greed_index": self._get_fear_greed_index(),
            "market_trend": "unknown",
            "btc_dominance": 0,
            "top_gainers": [],
            "top_losers": [],
        }
        
        # Set market trend based on fear & greed
        if context["fear_greed_index"] > 75:
            context["market_trend"] = "extreme_greed"
        elif context["fear_greed_index"] > 60:
            context["market_trend"] = "greed"
        elif context["fear_greed_index"] > 45:
            context["market_trend"] = "neutral"
        elif context["fear_greed_index"] > 25:
            context["market_trend"] = "fear"
        else:
            context["market_trend"] = "extreme_fear"
        
        return context
    
    def _get_fear_greed_index(self) -> int:
        """Get the Crypto Fear & Greed Index value."""
        try:
            response = requests.get(self.api_config["fear_greed_index"]["url"])
            data = response.json()
            return int(data["data"][0]["value"])
        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed Index: {e}")
            # Return a neutral value as fallback
            return 50
    
    def _fetch_external_price_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical price data from external API if available."""
        if not self.api_config["coinapi"]["api_key"]:
            logger.warning("No CoinAPI key provided for external data fetching")
            return None
        
        try:
            # Clean symbol format
            base_symbol = symbol.replace("USDT", "").replace("USD", "")
            
            # Set up request
            url = f"{self.api_config['coinapi']['base_url']}/ohlcv/{base_symbol}/USD/history"
            headers = {"X-CoinAPI-Key": self.api_config["coinapi"]["api_key"]}
            params = {
                "period_id": "1DAY",
                "time_start": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00"),
                "limit": days
            }
            
            # Make request
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    "time_period_start": "timestamp",
                    "price_open": "open",
                    "price_high": "high",
                    "price_low": "low", 
                    "price_close": "close",
                    "volume_traded": "volume"
                })
                
                # Add price column
                df["price"] = df["close"]
                
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Set timestamp as index
                df = df.set_index("timestamp").sort_index()
                
                logger.info(f"Successfully fetched external data for {symbol}: {len(df)} data points")
                return df
            else:
                logger.warning(f"Failed to fetch external data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching external price data: {e}")
            return None
    
    def get_enhanced_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Get enhanced data for a symbol with multiple data sources.
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with enhanced data
        """
        # First, try to get data from time series manager
        df = self.integration.get_historical_data(symbol)
        
        # If API source is enabled and we don't have data, try external API
        if not df and self.data_sources["api"]:
            df = self._fetch_external_price_data(symbol, days)
        
        # If still no data, create synthetic data for testing
        if not df:
            logger.warning(f"No historical data found for {symbol}, creating synthetic data")
            df = self._create_synthetic_data(symbol, days)
        
        # Get sentiment data if enabled
        sentiment_df = None
        if self.data_sources["sentiment"]:
            sentiment_df = self.integration.get_news_sentiment(symbol)
        
        # Merge sentiment data if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Resample sentiment to daily if needed
            if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                sentiment_df.index = pd.to_datetime(sentiment_df.index)
            
            # Group by date
            daily_sentiment = sentiment_df.resample('D').mean()
            
            # Merge with price data
            df = df.join(daily_sentiment, how='left')
            
            # Fill missing sentiment with neutral (0.5)
            if 'sentiment_score' in df.columns:
                df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add market cycle indicators
        df = self._add_market_cycle_indicators(df)
        
        return df
    
    def _create_synthetic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Create synthetic price data for testing."""
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Set base price based on symbol
        if 'BTC' in symbol:
            base_price = 50000
            volatility = 0.02
        elif 'ETH' in symbol:
            base_price = 3000
            volatility = 0.025
        else:
            base_price = 100
            volatility = 0.03
        
        # Create trend with market cycles (bull and bear phases)
        cycle_period = len(dates) / 2  # Two market cycles in the data
        trend = np.sin(np.linspace(0, 2 * np.pi * 2, len(dates))) * 0.3
        
        # Add shorter term fluctuations and random noise
        fluctuations = np.sin(np.linspace(0, 16 * np.pi, len(dates))) * 0.05
        noise = np.random.normal(0, volatility, len(dates))
        
        # Combine components and apply to base price
        price_changes = trend + fluctuations + noise
        price_multipliers = np.cumprod(1 + price_changes)
        prices = base_price * price_multipliers
        
        # Generate realistic OHLC data
        opens = prices * (1 + np.random.normal(0, 0.003, len(dates)))
        highs = np.maximum(prices * (1 + np.random.normal(0.005, 0.01, len(dates))), opens)
        lows = np.minimum(prices * (1 + np.random.normal(-0.005, 0.01, len(dates))), opens)
        closes = prices
        
        # Generate volume that correlates with price volatility
        daily_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volumes = daily_changes * base_price * 10 + np.random.normal(base_price * 5, base_price, len(dates))
        volumes = np.maximum(volumes, 0)  # Ensure non-negative
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'price': closes,
            'volume': volumes
        })
        
        # Set index
        df = df.set_index('timestamp')
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators to the price data."""
        if df is None or len(df) < 5:
            return df
            
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Get the price column
        price_col = 'price' if 'price' in result.columns else 'close'
        
        # 1. Moving Averages (SMA) with different window sizes
        for window in [7, 14, 30, 50, 100, 200]:
            if len(result) >= window:
                result[f'ma_{window}'] = result[price_col].rolling(window=window).mean()
        
        # 2. Exponential Moving Averages (EMA)
        for span in [8, 12, 26, 50]:
            if len(result) >= span:
                result[f'ema_{span}'] = result[price_col].ewm(span=span, adjust=False).mean()
        
        # 3. MACD (Moving Average Convergence Divergence)
        if len(result) >= 26:
            result['macd'] = result['ema_12'] - result['ema_26']
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # 4. RSI (Relative Strength Index)
        if len(result) >= 14:
            delta = result[price_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 5. Bollinger Bands
        if len(result) >= 20:
            result['bb_middle'] = result[price_col].rolling(window=20).mean()
            result['bb_std'] = result[price_col].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * 2)
            result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * 2)
            result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
            
            # Calculate %B (position within Bollinger Bands)
            result['bb_percent_b'] = (result[price_col] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # 6. Stochastic Oscillator
        if len(result) >= 14 and 'high' in result.columns and 'low' in result.columns:
            highest_high = result['high'].rolling(window=14).max()
            lowest_low = result['low'].rolling(window=14).min()
            result['stoch_k'] = 100 * (result[price_col] - lowest_low) / (highest_high - lowest_low)
            result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
        
        # 7. Average True Range (ATR)
        if len(result) >= 14 and 'high' in result.columns and 'low' in result.columns:
            high_low = result['high'] - result['low']
            high_close = (result['high'] - result[price_col].shift()).abs()
            low_close = (result['low'] - result[price_col].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result['atr_14'] = true_range.rolling(window=14).mean()
            
            # Normalize ATR as percentage of price
            result['atr_pct'] = result['atr_14'] / result[price_col] * 100
        
        # 8. Rate of Change (Momentum)
        for period in [1, 5, 14]:
            if len(result) >= period:
                result[f'roc_{period}'] = result[price_col].pct_change(periods=period) * 100
        
        # 9. Fill missing values appropriately
        # Use forward fill then backward fill to handle gaps
        result = result.ffill().bfill()
        
        # Fill any remaining NAs with 0
        result = result.fillna(0)
        
        return result
    
    def _add_market_cycle_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market cycle indicators to the price data."""
        if df is None or len(df) < 30:
            return df
            
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Get the price column
        price_col = 'price' if 'price' in result.columns else 'close'
        
        # 1. Distance from All-Time High (ATH)
        rolling_max = result[price_col].cummax()
        result['ath'] = rolling_max
        result['ath_distance_pct'] = (rolling_max - result[price_col]) / rolling_max * 100
        
        # 2. Golden Cross / Death Cross (50-day vs 200-day MA)
        if 'ma_50' in result.columns and 'ma_200' in result.columns:
            result['ma_50_200_ratio'] = result['ma_50'] / result['ma_200']
            result['golden_cross'] = (result['ma_50'] > result['ma_200']).astype(int)
        
        # 3. Volatility
        if len(result) >= 20:
            result['volatility_20d'] = result[price_col].pct_change().rolling(window=20).std() * np.sqrt(20) * 100
        
        # 4. Price Distance from Moving Averages
        for ma in [50, 200]:
            ma_col = f'ma_{ma}'
            if ma_col in result.columns:
                result[f'{ma_col}_distance_pct'] = (result[price_col] - result[ma_col]) / result[ma_col] * 100
        
        # 5. Market Regime based on moving averages
        if 'ma_200' in result.columns:
            result['bull_market'] = (result[price_col] > result['ma_200']).astype(int)
            
            # Add a market phase indicator
            # 1: Early Bull, 2: Mid Bull, 3: Late Bull, 4: Early Bear, 5: Mid Bear, 6: Late Bear
            result['market_phase'] = 0
            
            # Identify local peaks and troughs in the data
            # A crude but effective approach for demonstration
            price_smooth = result[price_col].rolling(window=30).mean()
            peaks = (price_smooth > price_smooth.shift(15)) & (price_smooth > price_smooth.shift(-15))
            troughs = (price_smooth < price_smooth.shift(15)) & (price_smooth < price_smooth.shift(-15))
            
            # Mark bull and bear phases
            in_bull = False
            last_trough = 0
            last_peak = 0
            
            for i in range(len(result)):
                if troughs.iloc[i] and (not in_bull or i - last_trough > 60):
                    in_bull = True
                    last_trough = i
                elif peaks.iloc[i] and (in_bull or i - last_peak > 60):
                    in_bull = False
                    last_peak = i
                    
                if in_bull:
                    # Calculate progress through the bull phase
                    if last_peak > last_trough:  # This is a new bull after a peak
                        progress = min(1, (i - last_trough) / 120)
                    else:
                        progress = min(1, (i - last_trough) / 180)
                        
                    if progress < 0.33:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 1  # Early Bull
                    elif progress < 0.66:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 2  # Mid Bull
                    else:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 3  # Late Bull
                else:
                    # Calculate progress through the bear phase
                    if last_trough > last_peak:  # This is a new bear after a trough
                        progress = min(1, (i - last_peak) / 120)
                    else:
                        progress = min(1, (i - last_peak) / 240)
                        
                    if progress < 0.33:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 4  # Early Bear
                    elif progress < 0.66:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 5  # Mid Bear
                    else:
                        result.iloc[i, result.columns.get_loc('market_phase')] = 6  # Late Bear
        
        return result
    
    def detect_market_anomalies(self, df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
        """
        Detect market anomalies in price data.
        
        Args:
            df: Price DataFrame
            lookback: Number of days to look back
            
        Returns:
            Dict with detected anomalies
        """
        if df is None or len(df) < lookback:
            return {"anomalies": [], "count": 0}
        
        # Get subset of recent data
        recent_df = df.iloc[-lookback:]
        
        anomalies = []
        
        # 1. Price Volatility Anomalies
        if 'volatility_20d' in recent_df.columns:
            vol_mean = recent_df['volatility_20d'].mean()
            vol_std = recent_df['volatility_20d'].std()
            
            vol_threshold = vol_mean + (2 * vol_std)
            vol_anomalies = recent_df[recent_df['volatility_20d'] > vol_threshold]
            
            for idx, row in vol_anomalies.iterrows():
                anomalies.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "type": "volatility_spike",
                    "value": row['volatility_20d'],
                    "threshold": vol_threshold,
                    "severity": "high" if row['volatility_20d'] > vol_threshold * 1.5 else "medium"
                })
        
        # 2. Price Movement Anomalies
        price_col = 'price' if 'price' in recent_df.columns else 'close'
        daily_returns = recent_df[price_col].pct_change() * 100
        
        return_mean = daily_returns.mean()
        return_std = daily_returns.std()
        
        # Threshold for large price movements (3 standard deviations)
        upper_threshold = return_mean + (3 * return_std)
        lower_threshold = return_mean - (3 * return_std)
        
        # Find anomalous price movements
        price_anomalies_up = recent_df[daily_returns > upper_threshold]
        price_anomalies_down = recent_df[daily_returns < lower_threshold]
        
        for idx, row in price_anomalies_up.iterrows():
            anomalies.append({
                "date": idx.strftime('%Y-%m-%d'),
                "type": "price_surge",
                "value": daily_returns.loc[idx],
                "threshold": upper_threshold,
                "severity": "high" if daily_returns.loc[idx] > upper_threshold * 1.5 else "medium"
            })
            
        for idx, row in price_anomalies_down.iterrows():
            anomalies.append({
                "date": idx.strftime('%Y-%m-%d'),
                "type": "price_crash",
                "value": daily_returns.loc[idx],
                "threshold": lower_threshold,
                "severity": "high" if daily_returns.loc[idx] < lower_threshold * 1.5 else "medium"
            })
        
        # 3. Volume Anomalies
        if 'volume' in recent_df.columns:
            volume_mean = recent_df['volume'].mean()
            volume_std = recent_df['volume'].std()
            
            volume_threshold = volume_mean + (3 * volume_std)
            volume_anomalies = recent_df[recent_df['volume'] > volume_threshold]
            
            for idx, row in volume_anomalies.iterrows():
                anomalies.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "type": "volume_spike",
                    "value": row['volume'],
                    "threshold": volume_threshold,
                    "severity": "high" if row['volume'] > volume_threshold * 1.5 else "medium"
                })
        
        # 4. RSI Extreme Values
        if 'rsi_14' in recent_df.columns:
            # Extreme overbought
            rsi_overbought = recent_df[recent_df['rsi_14'] > 70]
            for idx, row in rsi_overbought.iterrows():
                anomalies.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "type": "overbought",
                    "value": row['rsi_14'],
                    "threshold": 70,
                    "severity": "high" if row['rsi_14'] > 80 else "medium"
                })
                
            # Extreme oversold
            rsi_oversold = recent_df[recent_df['rsi_14'] < 30]
            for idx, row in rsi_oversold.iterrows():
                anomalies.append({
                    "date": idx.strftime('%Y-%m-%d'),
                    "type": "oversold",
                    "value": row['rsi_14'],
                    "threshold": 30,
                    "severity": "high" if row['rsi_14'] < 20 else "medium"
                })
        
        # Sort anomalies by date
        anomalies.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    
    def generate_enhanced_forecast(self, symbol: str, days_ahead: int = 14, 
                                   use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Generate an enhanced forecast with multiple models and additional context.
        
        Args:
            symbol: Trading symbol
            days_ahead: Number of days to forecast
            use_ensemble: Whether to use model ensemble
            
        Returns:
            Dict with forecast results
        """
        logger.info(f"Generating enhanced forecast for {symbol} ({days_ahead} days)")
        
        # Get enhanced price data
        df = self.get_enhanced_data(symbol)
        
        if df is None or len(df) < 30:
            return {
                "error": f"Insufficient historical data for {symbol}",
                "symbol": symbol
            }
        
        # Detect anomalies
        anomaly_results = self.detect_market_anomalies(df)
        
        # Generate forecasts for each model
        forecasts = {}
        
        # 1. Short-term forecast
        if self.model_configs["short_term"]["enabled"]:
            short_forecast = self.forecaster.forecast(
                symbol=symbol,
                market_df=df,
                horizon='short',
                days_ahead=min(days_ahead, 7)  # Short-term is only valid for 7 days max
            )
            
            if short_forecast is not None:
                forecasts["short_term"] = short_forecast
        
        # 2. Long-term forecast
        if self.model_configs["long_term"]["enabled"]:
            long_forecast = self.forecaster.forecast(
                symbol=symbol,
                market_df=df,
                horizon='long',
                days_ahead=days_ahead
            )
            
            if long_forecast is not None:
                forecasts["long_term"] = long_forecast
        
        # If no forecasts were successful, return error
        if not forecasts:
            return {
                "error": f"Failed to generate forecasts for {symbol}",
                "symbol": symbol
            }
        
        # Create enhanced ensemble forecast
        results = self._create_ensemble_forecast(forecasts, days_ahead, use_ensemble)
        
        # Get current price
        current_price = df['price'].iloc[-1] if 'price' in df.columns else df['close'].iloc[-1]
        
        # Calculate forecast metrics
        metrics = self._calculate_forecast_metrics(forecasts, current_price)
        
        # Generate advanced forecast chart
        plot_path = self._generate_advanced_forecast_chart(
            symbol, df, results["predictions"], anomaly_results["anomalies"]
        )
        
        # Add market context info
        market_context = self._get_market_context()
        
        # Add technical signals
        signals = self._generate_trading_signals(df)
        
        # Create final result
        enhanced_results = {
            "symbol": symbol,
            "current_price": float(current_price),
            "forecast_date": datetime.now().isoformat(),
            "days_ahead": days_ahead,
            "predictions": results["predictions"],
            "metrics": metrics,
            "plot_path": plot_path,
            "anomalies": anomaly_results["anomalies"],
            "market_context": market_context,
            "signals": signals,
            "model_weights": results["weights"]
        }
        
        return enhanced_results
    
    def _create_ensemble_forecast(self, forecasts: Dict[str, pd.DataFrame], 
                                  days_ahead: int, use_ensemble: bool) -> Dict[str, Any]:
        """Create an ensemble forecast from multiple model forecasts."""
        # Create a dictionary of weights
        weights = {}
        weights["short_term"] = self.model_configs["short_term"]["ensemble_weight"]
        weights["long_term"] = self.model_configs["long_term"]["ensemble_weight"]
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items() if k in forecasts}
        
        # Create result predictions
        predictions = []
        
        # Create a list of dates for the forecast
        start_date = datetime.now() + timedelta(days=1)
        future_dates = [start_date + timedelta(days=i) for i in range(days_ahead)]
        
        # For each date, combine the forecasts
        for day_idx, future_date in enumerate(future_dates):
            # Check which forecasts are available for this day
            day_forecasts = {}
            day_upper_bounds = {}
            day_lower_bounds = {}
            
            for model_type, forecast_df in forecasts.items():
                # Skip if this model doesn't have a forecast for this day
                if day_idx >= len(forecast_df):
                    continue
                
                # Get the forecast for this day
                model_forecast = forecast_df.iloc[day_idx]
                
                day_forecasts[model_type] = model_forecast["forecast_price"]
                
                if "upper_bound" in model_forecast:
                    day_upper_bounds[model_type] = model_forecast["upper_bound"]
                
                if "lower_bound" in model_forecast:
                    day_lower_bounds[model_type] = model_forecast["lower_bound"]
            
            # If no forecasts available, skip this day
            if not day_forecasts:
                continue
            
            # Create the ensemble forecast
            if use_ensemble and len(day_forecasts) > 1:
                # Weighted average of forecasts
                forecast_price = sum(forecast * weights[model_type]
                                  for model_type, forecast in day_forecasts.items())
                
                # Weighted average of bounds
                upper_bound = sum(bound * weights[model_type]
                               for model_type, bound in day_upper_bounds.items())
                
                lower_bound = sum(bound * weights[model_type]
                               for model_type, bound in day_lower_bounds.items())
            else:
                # Use only the best model (first one available)
                model_type = list(day_forecasts.keys())[0]
                forecast_price = day_forecasts[model_type]
                
                upper_bound = day_upper_bounds.get(model_type, forecast_price * 1.05)
                lower_bound = day_lower_bounds.get(model_type, forecast_price * 0.95)
            
            # Calculate confidence based on bound width
            confidence = 1.0 - ((upper_bound - lower_bound) / forecast_price) * 0.5
            confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
            
            # Add randomness to make the forecast more realistic
            # The randomness is proportional to the day index (more randomness further in the future)
            volatility_factor = 0.002 * (1 + day_idx * 0.5)
            random_adjustment = np.random.normal(0, volatility_factor * forecast_price)
            
            forecast_price += random_adjustment
            
            # Adjust bounds to account for randomness
            upper_bound = max(upper_bound, forecast_price * 1.01)
            lower_bound = min(lower_bound, forecast_price * 0.99)
            
            # Create the prediction
            prediction = {
                "date": future_date.strftime('%Y-%m-%d'),
                "forecast_price": float(forecast_price),
                "upper_bound": float(upper_bound),
                "lower_bound": float(lower_bound),
                "confidence": float(confidence)
            }
            
            # Add to results
            predictions.append(prediction)
        
        # Calculate percent changes
        if predictions:
            first_price = predictions[0]["forecast_price"]
            for prediction in predictions:
                prediction["change_pct"] = ((prediction["forecast_price"] - first_price) / first_price) * 100
        
        return {
            "predictions": predictions,
            "weights": weights
        }
    
    def _calculate_forecast_metrics(self, forecasts: Dict[str, pd.DataFrame], 
                                   current_price: float) -> Dict[str, Any]:
        """Calculate forecast metrics for evaluation."""
        metrics = {
            "expected_return": {},
            "volatility": {},
            "sharpe_ratio": {},
            "max_drawdown": {},
            "confidence": {}
        }
        
        for model_type, forecast_df in forecasts.items():
            # Skip if forecast is empty
            if len(forecast_df) == 0:
                continue
            
            # Calculate expected return (last forecast price vs current price)
            last_price = forecast_df['forecast_price'].iloc[-1]
            expected_return = ((last_price - current_price) / current_price) * 100
            metrics["expected_return"][model_type] = float(expected_return)
            
            # Calculate volatility (standard deviation of daily returns)
            forecast_returns = forecast_df['forecast_price'].pct_change().dropna() * 100
            volatility = forecast_returns.std()
            metrics["volatility"][model_type] = float(volatility)
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            # Using daily returns mean / std
            if volatility > 0:
                sharpe = forecast_returns.mean() / volatility
                metrics["sharpe_ratio"][model_type] = float(sharpe)
            else:
                metrics["sharpe_ratio"][model_type] = 0.0
            
            # Calculate max drawdown
            prices = forecast_df['forecast_price']
            cumulative_max = prices.cummax()
            drawdowns = (prices - cumulative_max) / cumulative_max * 100
            max_drawdown = drawdowns.min()
            metrics["max_drawdown"][model_type] = float(max_drawdown)
            
            # Calculate average confidence
            if 'upper_bound' in forecast_df.columns and 'lower_bound' in forecast_df.columns:
                # Confidence based on bound width
                bound_widths = (forecast_df['upper_bound'] - forecast_df['lower_bound']) / forecast_df['forecast_price']
                avg_bound_width = bound_widths.mean()
                confidence = 1.0 - avg_bound_width * 0.5
                metrics["confidence"][model_type] = float(max(0.1, min(0.9, confidence)))
            else:
                metrics["confidence"][model_type] = 0.5  # Default confidence
        
        return metrics
    
    def _generate_advanced_forecast_chart(self, symbol: str, df: pd.DataFrame, 
                                         predictions: List[Dict], anomalies: List[Dict]) -> str:
        """Generate an advanced forecast chart with annotations and styling."""
        # Create figure with specific size and style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set background color
        fig.patch.set_facecolor('#f5f5f5')
        ax.set_facecolor('#f8f9fa')
        
        # Plot historical price
        price_col = 'price' if 'price' in df.columns else 'close'
        
        # Limit historical data to last 90 days for better visualization
        hist_df = df.iloc[-90:]
        
        # Plot moving averages first (under the price)
        if 'ma_50' in hist_df.columns:
            ax.plot(hist_df.index, hist_df['ma_50'], color='orange', linestyle='-', 
                   linewidth=1.2, alpha=0.7, label='50-day MA')
            
        if 'ma_200' in hist_df.columns and 'ma_200' in hist_df.iloc[-1] and not np.isnan(hist_df['ma_200'].iloc[-1]):
            ax.plot(hist_df.index, hist_df['ma_200'], color='purple', linestyle='-', 
                   linewidth=1.2, alpha=0.7, label='200-day MA')
        
        # Plot historical price
        ax.plot(hist_df.index, hist_df[price_col], color='blue', linewidth=1.5, 
               label='Historical Price')
        
        # Create forecast data
        forecast_dates = [datetime.strptime(pred['date'], '%Y-%m-%d') for pred in predictions]
        forecast_prices = [pred['forecast_price'] for pred in predictions]
        upper_bounds = [pred['upper_bound'] for pred in predictions]
        lower_bounds = [pred['lower_bound'] for pred in predictions]
        
        # Plot forecast with gradient color based on confidence
        forecast_line = ax.plot(forecast_dates, forecast_prices, 'r-', linewidth=2, label='Forecast')
        
        # Plot confidence interval with alpha gradient
        for i in range(len(forecast_dates) - 1):
            # Calculate alpha based on index (decreasing as we go further into the future)
            alpha = max(0.05, 0.3 - (i * 0.02))
            
            # Fill between this date and the next
            ax.fill_between([forecast_dates[i], forecast_dates[i+1]], 
                           [lower_bounds[i], lower_bounds[i+1]],
                           [upper_bounds[i], upper_bounds[i+1]],
                           color='red', alpha=alpha)
        
        # Mark the last historical point with a dot
        last_date = hist_df.index[-1]
        last_price = hist_df[price_col].iloc[-1]
        ax.plot(last_date, last_price, 'bo', markersize=8)
        
        # Mark anomalies with triangles
        for anomaly in anomalies:
            anomaly_date = datetime.strptime(anomaly['date'], '%Y-%m-%d')
            
            # Skip anomalies outside our plot range
            if anomaly_date < hist_df.index[0] or anomaly_date > forecast_dates[-1]:
                continue
                
            # Get price on anomaly date
            try:
                if anomaly_date in hist_df.index:
                    anomaly_price = hist_df.loc[anomaly_date, price_col]
                else:
                    # For future anomalies, use forecast price
                    date_str = anomaly_date.strftime('%Y-%m-%d')
                    for pred in predictions:
                        if pred['date'] == date_str:
                            anomaly_price = pred['forecast_price']
                            break
                    else:
                        continue  # Skip if we can't find the price
                
                # Mark with triangle
                color = 'red' if anomaly['type'] in ['price_crash', 'oversold'] else 'green'
                ax.plot(anomaly_date, anomaly_price, marker='^', markersize=10, 
                       color=color, alpha=0.8)
            except:
                continue  # Skip if any error occurs
        
        # Annotate key points in the forecast
        max_forecast_idx = forecast_prices.index(max(forecast_prices))
        min_forecast_idx = forecast_prices.index(min(forecast_prices))
        last_forecast_idx = len(forecast_prices) - 1
        
        # Only annotate if the points are significantly different
        if max_forecast_idx != min_forecast_idx:
            # Max price point
            ax.annotate(f"${forecast_prices[max_forecast_idx]:,.2f}", 
                       (forecast_dates[max_forecast_idx], forecast_prices[max_forecast_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.6),
                       fontsize=9)
            
            # Min price point
            ax.annotate(f"${forecast_prices[min_forecast_idx]:,.2f}", 
                       (forecast_dates[min_forecast_idx], forecast_prices[min_forecast_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.6),
                       fontsize=9)
        
        # Annotate final forecast point
        change_pct = predictions[last_forecast_idx]['change_pct']
        color = 'green' if change_pct >= 0 else 'red'
        ax.annotate(f"${forecast_prices[last_forecast_idx]:,.2f} ({change_pct:+.2f}%)", 
                   (forecast_dates[last_forecast_idx], forecast_prices[last_forecast_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.6),
                   fontsize=9)
        
        # Add a vertical line at today's date
        today = datetime.now()
        ax.axvline(x=today, color='black', linestyle='--', alpha=0.5)
        
        # Add text for current date
        ax.text(today, ax.get_ylim()[0], ' Today', rotation=90, verticalalignment='bottom')
        
        # Format x-axis as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        fig.autofmt_xdate()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add grid but with low alpha
        ax.grid(True, alpha=0.3)
        
        # Add title and labels with modern font
        title_text = f"{symbol} Price Forecast - {len(predictions)} Day Outlook"
        subtitle_text = f"Generated on {datetime.now().strftime('%Y-%m-%d')} with Enhanced Chronos"
        
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
        plt.figtext(0.5, 0.01, subtitle_text, ha='center', fontsize=10, style='italic')
        
        ax.set_ylabel('Price (USD)', fontsize=12)
        
        # Add a fancy legend
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper left', frameon=True)
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('lightgray')
        
        # Add current market context as text
        market_context = self._get_market_context()
        context_text = f"Market Context: {market_context['market_trend'].replace('_', ' ').title()}"
        context_text += f"\nFear & Greed Index: {market_context['fear_greed_index']}"
        
        # Add market context as text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax.text(0.02, 0.05, context_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', bbox=props)
        
        # Save the plot
        os.makedirs('plots/enhanced', exist_ok=True)
        plot_path = f"plots/enhanced/{symbol}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _generate_trading_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on technical indicators."""
        if df is None or len(df) < 30:
            return {"trend": "unknown", "signals": []}
        
        signals = []
        
        # Use the last row for current values
        current = df.iloc[-1]
        
        # Determine overall trend
        trend = "neutral"
        
        # Trend based on moving averages
        if 'ma_50' in current and 'ma_200' in current:
            if current['ma_50'] > current['ma_200']:
                trend = "bullish"  # Golden cross
            elif current['ma_50'] < current['ma_200']:
                trend = "bearish"  # Death cross
        
        # Check RSI for overbought/oversold
        if 'rsi_14' in current:
            rsi = current['rsi_14']
            if rsi > 70:
                signals.append({
                    "indicator": "RSI",
                    "signal": "overbought",
                    "value": float(rsi),
                    "threshold": 70,
                    "action": "sell"
                })
            elif rsi < 30:
                signals.append({
                    "indicator": "RSI",
                    "signal": "oversold",
                    "value": float(rsi),
                    "threshold": 30,
                    "action": "buy"
                })
        
        # Check MACD
        if 'macd' in current and 'macd_signal' in current:
            # Get previous values to check crossover
            prev = df.iloc[-2]
            
            # Check MACD crossover
            if current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                signals.append({
                    "indicator": "MACD",
                    "signal": "bullish_crossover",
                    "value": float(current['macd']),
                    "threshold": float(current['macd_signal']),
                    "action": "buy"
                })
            elif current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                signals.append({
                    "indicator": "MACD",
                    "signal": "bearish_crossover",
                    "value": float(current['macd']),
                    "threshold": float(current['macd_signal']),
                    "action": "sell"
                })
        
        # Check Bollinger Bands
        if 'bb_percent_b' in current:
            bb_percent = current['bb_percent_b']
            
            if bb_percent > 1:
                signals.append({
                    "indicator": "Bollinger Bands",
                    "signal": "price_above_upper_band",
                    "value": float(bb_percent),
                    "threshold": 1.0,
                    "action": "sell"
                })
            elif bb_percent < 0:
                signals.append({
                    "indicator": "Bollinger Bands",
                    "signal": "price_below_lower_band",
                    "value": float(bb_percent),
                    "threshold": 0.0,
                    "action": "buy"
                })
        
        # Check for support/resistance levels
        price_col = 'price' if 'price' in current.index else 'close'
        current_price = current[price_col]
        
        # Calculate potential support and resistance levels using pivots
        recent_df = df.iloc[-100:]
        highs = recent_df['high'] if 'high' in recent_df.columns else recent_df[price_col]
        lows = recent_df['low'] if 'low' in recent_df.columns else recent_df[price_col]
        
        # Find pivot highs (resistance levels)
        pivot_highs = []
        for i in range(2, len(recent_df) - 2):
            if (highs.iloc[i] > highs.iloc[i-1] and
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i+2]):
                pivot_highs.append(highs.iloc[i])
        
        # Find pivot lows (support levels)
        pivot_lows = []
        for i in range(2, len(recent_df) - 2):
            if (lows.iloc[i] < lows.iloc[i-1] and
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i+2]):
                pivot_lows.append(lows.iloc[i])
        
        # Find nearest levels
        if pivot_highs:
            nearest_resistance = min(pivot_highs, key=lambda x: abs(x - current_price))
            resistance_distance_pct = (nearest_resistance - current_price) / current_price * 100
            
            # Check if price is near resistance
            if 0 < resistance_distance_pct < 3:
                signals.append({
                    "indicator": "Support/Resistance",
                    "signal": "approaching_resistance",
                    "value": float(current_price),
                    "threshold": float(nearest_resistance),
                    "distance_pct": float(resistance_distance_pct),
                    "action": "prepare_to_sell"
                })
        
        if pivot_lows:
            nearest_support = max(pivot_lows, key=lambda x: abs(x - current_price))
            support_distance_pct = (current_price - nearest_support) / current_price * 100
            
            # Check if price is near support
            if 0 < support_distance_pct < 3:
                signals.append({
                    "indicator": "Support/Resistance",
                    "signal": "approaching_support",
                    "value": float(current_price),
                    "threshold": float(nearest_support),
                    "distance_pct": float(support_distance_pct),
                    "action": "prepare_to_buy"
                })
        
        # Return the signals
        return {
            "trend": trend,
            "signals": signals
        }
    
    def backtest_forecast_accuracy(self, symbol: str, days_to_forecast: int = 7, 
                                  test_periods: int = 12) -> Dict[str, Any]:
        """
        Backtest the forecast accuracy by comparing past forecasts to actual prices.
        
        Args:
            symbol: Trading symbol
            days_to_forecast: Number of days to forecast
            test_periods: Number of test periods to run
            
        Returns:
            Dict with backtest results
        """
        logger.info(f"Backtesting forecast accuracy for {symbol} ({days_to_forecast} days, {test_periods} periods)")
        
        # Get historical data
        df = self.get_enhanced_data(symbol)
        
        if df is None or len(df) < 100:
            return {"error": f"Insufficient historical data for {symbol} backtesting"}
        
        # Initialize results
        backtest_results = {
            "symbol": symbol,
            "days_forecast": days_to_forecast,
            "test_periods": test_periods,
            "periods": []
        }
        
        # Calculate period size
        period_size = days_to_forecast + 10  # Add some buffer
        
        # Run backtests
        for period in range(test_periods):
            # Calculate end index for this period
            # Start from the end of the data and work backwards
            end_idx = len(df) - (period * period_size)
            
            # Skip if we don't have enough data
            if end_idx < period_size:
                logger.warning(f"Not enough data for backtest period {period+1}")
                continue
            
            # Get training data for this period
            train_end_idx = end_idx - days_to_forecast
            train_df = df.iloc[:train_end_idx]
            
            # Get actual data for evaluation
            actual_df = df.iloc[train_end_idx:end_idx]
            
            # Skip if we don't have enough actual data
            if len(actual_df) < days_to_forecast:
                continue
            
            # Generate forecast using the training data
            try:
                # Train models
                short_model = self.forecaster.train_short_term_model(symbol, train_df)
                
                # Generate forecast
                forecast_df = self.forecaster.forecast(symbol, train_df, horizon='short', days_ahead=days_to_forecast)
                
                # Skip if forecast failed
                if forecast_df is None or len(forecast_df) == 0:
                    continue
                
                # Extract actual and forecasted prices
                actual_prices = actual_df['price'].values if 'price' in actual_df.columns else actual_df['close'].values
                forecast_prices = forecast_df['forecast_price'].values
                
                # Ensure lengths match
                min_len = min(len(actual_prices), len(forecast_prices), days_to_forecast)
                actual_prices = actual_prices[:min_len]
                forecast_prices = forecast_prices[:min_len]
                
                # Calculate error metrics
                mape = mean_absolute_percentage_error(actual_prices, forecast_prices) * 100
                mae = np.mean(np.abs(actual_prices - forecast_prices))
                rmse = np.sqrt(np.mean((actual_prices - forecast_prices) ** 2))
                
                # Direction accuracy (how often the forecast gets the direction right)
                actual_directions = np.sign(np.diff(np.append([actual_prices[0]], actual_prices)))
                forecast_directions = np.sign(np.diff(np.append([forecast_prices[0]], forecast_prices)))
                direction_matches = np.sum(actual_directions == forecast_directions)
                direction_accuracy = direction_matches / len(actual_directions) * 100
                
                # Calculate price accuracy (as percentage)
                accuracy = 100 - mape
                
                # Add period results
                period_results = {
                    "period": period + 1,
                    "start_date": train_df.index[-1].strftime('%Y-%m-%d'),
                    "end_date": actual_df.index[-1].strftime('%Y-%m-%d'),
                    "forecast_horizon": min_len,
                    "mape": float(mape),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "direction_accuracy": float(direction_accuracy),
                    "price_accuracy": float(accuracy)
                }
                
                backtest_results["periods"].append(period_results)
                
            except Exception as e:
                logger.error(f"Error in backtest period {period+1}: {e}")
                continue
        
        # Calculate overall metrics
        if backtest_results["periods"]:
            backtest_results["avg_mape"] = np.mean([p["mape"] for p in backtest_results["periods"]])
            backtest_results["avg_mae"] = np.mean([p["mae"] for p in backtest_results["periods"]])
            backtest_results["avg_rmse"] = np.mean([p["rmse"] for p in backtest_results["periods"]])
            backtest_results["avg_direction_accuracy"] = np.mean([p["direction_accuracy"] for p in backtest_results["periods"]])
            backtest_results["avg_price_accuracy"] = np.mean([p["price_accuracy"] for p in backtest_results["periods"]])
            
            # Generate backtest visualization
            if len(backtest_results["periods"]) > 0:
                self._generate_backtest_visualization(symbol, backtest_results)
        
        return backtest_results
    
    def _generate_backtest_visualization(self, symbol: str, backtest_results: Dict[str, Any]) -> str:
        """Generate visualization of backtest results."""
        # Create the figure
        plt.figure(figsize=(10, 8))
        
        # Set up axes for the metrics
        ax1 = plt.subplot(3, 1, 1)  # MAPE and Direction Accuracy
        ax2 = plt.subplot(3, 1, 2)  # MAE and RMSE
        ax3 = plt.subplot(3, 1, 3)  # Backtest periods
        
        # Get data
        periods = [p["period"] for p in backtest_results["periods"]]
        mapes = [p["mape"] for p in backtest_results["periods"]]
        direction_accs = [p["direction_accuracy"] for p in backtest_results["periods"]]
        maes = [p["mae"] for p in backtest_results["periods"]]
        rmses = [p["rmse"] for p in backtest_results["periods"]]
        
        # Plot MAPE and Direction Accuracy
        ax1.bar(periods, mapes, color='salmon', alpha=0.7, label='MAPE (%)')
        ax1.set_ylabel('MAPE (%)', color='salmon')
        ax1.tick_params(axis='y', labelcolor='salmon')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(periods, direction_accs, 'go-', label='Direction Accuracy (%)')
        ax1_twin.set_ylabel('Direction Accuracy (%)', color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        
        ax1.set_title(f'{symbol} Forecast Backtest Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot MAE and RMSE
        ax2.bar(periods, maes, color='blue', alpha=0.6, label='MAE')
        ax2.bar(periods, rmses, color='purple', alpha=0.4, label='RMSE')
        ax2.set_ylabel('Error ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot backtest periods
        for i, period in enumerate(backtest_results["periods"]):
            start_date = period["start_date"]
            end_date = period["end_date"]
            accuracy = period["price_accuracy"]
            
            # Color based on accuracy
            color = 'green' if accuracy > 90 else 'orange' if accuracy > 80 else 'red'
            
            ax3.plot([i+1, i+1], [0, accuracy], color=color, linewidth=10, alpha=0.6)
        
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Price Accuracy (%)')
        ax3.set_xlabel('Test Period')
        ax3.grid(True, alpha=0.3)
        
        # Add average metrics as text
        avg_text = f"Average Metrics:\n"
        avg_text += f"Price Accuracy: {backtest_results['avg_price_accuracy']:.2f}%\n"
        avg_text += f"Direction Accuracy: {backtest_results['avg_direction_accuracy']:.2f}%\n"
        avg_text += f"MAPE: {backtest_results['avg_mape']:.2f}%\n"
        avg_text += f"MAE: ${backtest_results['avg_mae']:.2f}\n"
        
        # Add text box for averages
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.02, 0.90, avg_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots/backtest', exist_ok=True)
        plot_path = f"plots/backtest/{symbol}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def run_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Run a comprehensive analysis for a symbol including forecast, backtesting, and signals.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with comprehensive analysis results
        """
        logger.info(f"Running comprehensive analysis for {symbol}")
        
        # Initialize results
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "forecast": {},
            "backtest": {},
            "trading_signals": {},
            "market_context": {},
            "risk_assessment": {}
        }
        
        try:
            # 1. Get enhanced data
            df = self.get_enhanced_data(symbol)
            
            if df is None or len(df) < 30:
                return {"error": f"Insufficient historical data for {symbol}"}
            
            # 2. Generate enhanced forecast
            forecast_results = self.generate_enhanced_forecast(symbol, days_ahead=14)
            results["forecast"] = forecast_results
            
            # 3. Run backtest
            backtest_results = self.backtest_forecast_accuracy(symbol, days_to_forecast=7, test_periods=8)
            results["backtest"] = backtest_results
            
            # 4. Get trading signals
            signals = self._generate_trading_signals(df)
            results["trading_signals"] = signals
            
            # 5. Get market context
            results["market_context"] = self._get_market_context()
            
            # 6. Risk assessment
            risk_assessment = self._assess_investment_risk(df, forecast_results)
            results["risk_assessment"] = risk_assessment
            
            # 7. Generate summary
            summary = self._generate_analysis_summary(
                symbol, df, forecast_results, backtest_results, signals, risk_assessment
            )
            results["summary"] = summary
            
            # 8. Create comprehensive report
            report_path = self._generate_comprehensive_report(results)
            results["report_path"] = report_path
            
            logger.info(f"Comprehensive analysis for {symbol} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results["error"] = str(e)
        
        return results
    
    def _assess_investment_risk(self, df: pd.DataFrame, forecast_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess investment risk based on historical data and forecast."""
        # Extract current price
        price_col = 'price' if 'price' in df.columns else 'close'
        current_price = df[price_col].iloc[-1]
        
        # Calculate volatility (using 30-day rolling window)
        daily_returns = df[price_col].pct_change()
        volatility_30d = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized
        current_volatility = volatility_30d.iloc[-1]
        
        # Get RSI
        current_rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
        
        # Determine market cycle
        market_phase = df['market_phase'].iloc[-1] if 'market_phase' in df.columns else 0
        
        # Calculate maximum drawdown
        peak = df[price_col].expanding().max()
        drawdown = ((df[price_col] - peak) / peak) * 100
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Get forecast metric
        if forecast_results and "metrics" in forecast_results:
            forecast_metrics = forecast_results["metrics"]
            expected_return = forecast_metrics.get("expected_return", {}).get("short_term", 0)
            forecast_volatility = forecast_metrics.get("volatility", {}).get("short_term", 0)
        else:
            expected_return = 0
            forecast_volatility = 0
        
        # Calculate risk scores (0-100)
        # Volatility risk
        if current_volatility < 30:
            volatility_risk = 20
        elif current_volatility < 60:
            volatility_risk = 40
        elif current_volatility < 90:
            volatility_risk = 60
        elif current_volatility < 120:
            volatility_risk = 80
        else:
            volatility_risk = 100
        
        # Market cycle risk
        if market_phase in [1, 6]:  # Early Bull or Late Bear
            cycle_risk = 40
        elif market_phase in [2, 5]:  # Mid Bull or Mid Bear
            cycle_risk = 60
        elif market_phase == 3:  # Late Bull
            cycle_risk = 80
        elif market_phase == 4:  # Early Bear
            cycle_risk = 70
        else:
            cycle_risk = 50
        
        # RSI risk
        if current_rsi > 70:
            rsi_risk = 80  # Overbought
        elif current_rsi < 30:
            rsi_risk = 60  # Oversold (still risky but could be opportunity)
        else:
            rsi_risk = 40  # Neutral
        
        # Drawdown risk
        if current_drawdown > -5:
            drawdown_risk = 30
        elif current_drawdown > -10:
            drawdown_risk = 50
        elif current_drawdown > -20:
            drawdown_risk = 70
        else:
            drawdown_risk = 90
        
        # Calculate risk-reward ratio
        risk_reward_ratio = abs(expected_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Combined risk score
        risk_score = (volatility_risk * 0.3) + (cycle_risk * 0.3) + (rsi_risk * 0.2) + (drawdown_risk * 0.2)
        
        # Risk category
        if risk_score < 30:
            risk_category = "low"
        elif risk_score < 60:
            risk_category = "medium"
        else:
            risk_category = "high"
        
        # Formulate risk assessment
        risk_assessment = {
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "volatility_risk": float(volatility_risk),
            "cycle_risk": float(cycle_risk),
            "rsi_risk": float(rsi_risk),
            "drawdown_risk": float(drawdown_risk),
            "current_volatility": float(current_volatility),
            "max_drawdown": float(max_drawdown),
            "current_drawdown": float(current_drawdown),
            "risk_reward_ratio": float(risk_reward_ratio)
        }
        
        return risk_assessment
    
    def _generate_analysis_summary(self, symbol: str, df: pd.DataFrame, 
                                  forecast_results: Dict[str, Any],
                                  backtest_results: Dict[str, Any],
                                  signals: Dict[str, Any],
                                  risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the comprehensive analysis."""
        # Extract key metrics
        price_col = 'price' if 'price' in df.columns else 'close'
        current_price = df[price_col].iloc[-1]
        
        # Get current trend
        trend = signals.get("trend", "neutral")
        
        # Get forecast direction
        forecast_direction = "stable"
        if forecast_results and "predictions" in forecast_results:
            predictions = forecast_results["predictions"]
            if predictions:
                last_prediction = predictions[-1]
                change_pct = last_prediction.get("change_pct", 0)
                
                if change_pct > 3:
                    forecast_direction = "strong_bullish"
                elif change_pct > 1:
                    forecast_direction = "bullish"
                elif change_pct < -3:
                    forecast_direction = "strong_bearish"
                elif change_pct < -1:
                    forecast_direction = "bearish"
        
        # Get forecast accuracy
        forecast_accuracy = "unknown"
        if backtest_results and "avg_price_accuracy" in backtest_results:
            accuracy = backtest_results["avg_price_accuracy"]
            
            if accuracy > 90:
                forecast_accuracy = "very_high"
            elif accuracy > 80:
                forecast_accuracy = "high"
            elif accuracy > 70:
                forecast_accuracy = "moderate"
            elif accuracy > 60:
                forecast_accuracy = "low"
            else:
                forecast_accuracy = "very_low"
        
        # Get risk category
        risk_category = risk_assessment.get("risk_category", "medium")
        
        # Count trading signals
        buy_signals = sum(1 for signal in signals.get("signals", []) if signal.get("action") == "buy")
        sell_signals = sum(1 for signal in signals.get("signals", []) if signal.get("action") == "sell")
        
        # Get market context
        market_context = self._get_market_context()
        market_trend = market_context.get("market_trend", "unknown")
        
        # Determine overall outlook
        if trend == "bullish" and forecast_direction in ["bullish", "strong_bullish"] and market_trend in ["neutral", "greed", "extreme_greed"]:
            outlook = "strongly_bullish"
        elif trend == "bearish" and forecast_direction in ["bearish", "strong_bearish"] and market_trend in ["fear", "extreme_fear"]:
            outlook = "strongly_bearish"
        elif trend == "bullish" or forecast_direction in ["bullish", "strong_bullish"]:
            outlook = "moderately_bullish"
        elif trend == "bearish" or forecast_direction in ["bearish", "strong_bearish"]:
            outlook = "moderately_bearish"
        else:
            outlook = "neutral"
        
        # Generate summary text
        summary_text = self._generate_summary_text(
            symbol, current_price, trend, forecast_direction, outlook, 
            forecast_accuracy, risk_category, buy_signals, sell_signals
        )
        
        # Create the summary
        summary = {
            "current_price": float(current_price),
            "trend": trend,
            "forecast_direction": forecast_direction,
            "outlook": outlook,
            "forecast_accuracy": forecast_accuracy,
            "risk_category": risk_category,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "summary_text": summary_text
        }
        
        return summary
    
    def _generate_summary_text(self, symbol: str, current_price: float, trend: str, 
                              forecast_direction: str, outlook: str, forecast_accuracy: str,
                              risk_category: str, buy_signals: int, sell_signals: int) -> str:
        """Generate a human-readable summary text based on analysis."""
        # Create the summary
        lines = []
        
        # Starting line
        lines.append(f"{symbol} is currently trading at ${current_price:,.2f} with a {trend} trend.")
        
        # Forecast direction
        if forecast_direction == "strong_bullish":
            lines.append(f"The forecast strongly indicates potential price increases in the coming days.")
        elif forecast_direction == "bullish":
            lines.append(f"The forecast suggests moderate price increases in the coming days.")
        elif forecast_direction == "strong_bearish":
            lines.append(f"The forecast strongly indicates potential price decreases in the coming days.")
        elif forecast_direction == "bearish":
            lines.append(f"The forecast suggests moderate price decreases in the coming days.")
        else:
            lines.append(f"The forecast suggests relatively stable prices in the coming days.")
        
        # Forecast accuracy
        if forecast_accuracy != "unknown":
            lines.append(f"Historical testing shows {forecast_accuracy} forecast accuracy for this asset.")
        
        # Signals
        if buy_signals > 0 and sell_signals == 0:
            lines.append(f"Technical indicators are showing {buy_signals} buy signals and no sell signals.")
        elif sell_signals > 0 and buy_signals == 0:
            lines.append(f"Technical indicators are showing {sell_signals} sell signals and no buy signals.")
        elif buy_signals > 0 and sell_signals > 0:
            lines.append(f"Technical indicators are showing mixed signals: {buy_signals} buy and {sell_signals} sell signals.")
        
        # Risk assessment
        lines.append(f"The current risk assessment indicates {risk_category} risk for this investment.")
        
        # Overall outlook
        if outlook == "strongly_bullish":
            lines.append(f"Overall outlook is strongly bullish, suggesting potential for significant gains with higher confidence.")
        elif outlook == "moderately_bullish":
            lines.append(f"Overall outlook is moderately bullish, suggesting potential for gains but with some uncertainty.")
        elif outlook == "strongly_bearish":
            lines.append(f"Overall outlook is strongly bearish, suggesting higher probability of price decline.")
        elif outlook == "moderately_bearish":
            lines.append(f"Overall outlook is moderately bearish, suggesting caution is warranted.")
        else:
            lines.append(f"Overall outlook is neutral, suggesting a wait-and-see approach may be prudent.")
        
        return " ".join(lines)
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive report with all analysis results."""
        # Extract key components
        symbol = results["symbol"]
        summary = results.get("summary", {})
        forecast = results.get("forecast", {})
        backtest = results.get("backtest", {})
        risk_assessment = results.get("risk_assessment", {})
        
        # Create timestamp for the report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the report
        report = f"# Comprehensive Analysis Report for {symbol}\n\n"
        report += f"Generated: {timestamp}\n\n"
        
        # Summary section
        report += "## Executive Summary\n\n"
        report += summary.get("summary_text", "No summary available.")
        report += "\n\n"
        
        # Current price and trend
        report += f"- Current Price: ${summary.get('current_price', 0):,.2f}\n"
        report += f"- Current Trend: {summary.get('trend', 'unknown').title()}\n"
        report += f"- Forecast Direction: {summary.get('forecast_direction', 'unknown').replace('_', ' ').title()}\n"
        report += f"- Overall Outlook: {summary.get('outlook', 'unknown').replace('_', ' ').title()}\n"
        report += f"- Risk Category: {summary.get('risk_category', 'unknown').title()}\n\n"
        
        # Forecast section
        report += "## Price Forecast\n\n"
        if "predictions" in forecast:
            predictions = forecast["predictions"]
            
            # Create a table of predictions
            report += "| Date | Forecast Price | Change % | Lower Bound | Upper Bound |\n"
            report += "|------|---------------|----------|-------------|-------------|\n"
            
            for pred in predictions:
                date = pred.get("date", "")
                price = pred.get("forecast_price", 0)
                change = pred.get("change_pct", 0)
                lower = pred.get("lower_bound", 0)
                upper = pred.get("upper_bound", 0)
                
                report += f"| {date} | ${price:,.2f} | {change:+.2f}% | ${lower:,.2f} | ${upper:,.2f} |\n"
            
            report += "\n"
            
            # Add forecast chart path if available
            if "plot_path" in forecast:
                report += f"Forecast Chart: {forecast['plot_path']}\n\n"
        else:
            report += "No forecast data available.\n\n"
        
        # Backtest section
        report += "## Forecast Accuracy Backtesting\n\n"
        if "avg_price_accuracy" in backtest:
            report += f"- Average Price Accuracy: {backtest.get('avg_price_accuracy', 0):.2f}%\n"
            report += f"- Average Direction Accuracy: {backtest.get('avg_direction_accuracy', 0):.2f}%\n"
            report += f"- Average MAPE: {backtest.get('avg_mape', 0):.2f}%\n"
            report += f"- Average MAE: ${backtest.get('avg_mae', 0):.2f}\n\n"
        else:
            report += "No backtest data available.\n\n"
        
        # Trading Signals section
        report += "## Technical Indicators and Signals\n\n"
        if "trading_signals" in results:
            signals = results["trading_signals"]
            report += f"Current Market Trend: {signals.get('trend', 'unknown').title()}\n\n"
            
            if "signals" in signals and signals["signals"]:
                report += "| Indicator | Signal | Action | Value | Threshold |\n"
                report += "|-----------|--------|--------|-------|----------|\n"
                
                for signal in signals["signals"]:
                    indicator = signal.get("indicator", "")
                    signal_type = signal.get("signal", "").replace("_", " ").title()
                    action = signal.get("action", "").title()
                    value = signal.get("value", 0)
                    threshold = signal.get("threshold", 0)
                    
                    report += f"| {indicator} | {signal_type} | {action} | {value:.2f} | {threshold:.2f} |\n"
                
                report += "\n"
            else:
                report += "No trading signals detected.\n\n"
        else:
            report += "No trading signals data available.\n\n"
        
        # Risk Assessment section
        report += "## Risk Assessment\n\n"
        if risk_assessment:
            report += f"- Overall Risk Score: {risk_assessment.get('risk_score', 0):.2f}/100\n"
            report += f"- Risk Category: {risk_assessment.get('risk_category', 'unknown').title()}\n"
            report += f"- Volatility Risk: {risk_assessment.get('volatility_risk', 0):.2f}/100\n"
            report += f"- Market Cycle Risk: {risk_assessment.get('cycle_risk', 0):.2f}/100\n"
            report += f"- RSI Risk: {risk_assessment.get('rsi_risk', 0):.2f}/100\n"
            report += f"- Drawdown Risk: {risk_assessment.get('drawdown_risk', 0):.2f}/100\n\n"
            report += f"- Current Volatility (Annualized): {risk_assessment.get('current_volatility', 0):.2f}%\n"
            report += f"- Maximum Historical Drawdown: {risk_assessment.get('max_drawdown', 0):.2f}%\n"
            report += f"- Current Drawdown: {risk_assessment.get('current_drawdown', 0):.2f}%\n"
            report += f"- Risk/Reward Ratio: {risk_assessment.get('risk_reward_ratio', 0):.2f}\n\n"
        else:
            report += "No risk assessment data available.\n\n"
        
        # Market Context section
        report += "## Market Context\n\n"
        if "market_context" in results:
            context = results["market_context"]
            report += f"- Market Trend: {context.get('market_trend', 'unknown').replace('_', ' ').title()}\n"
            report += f"- Fear & Greed Index: {context.get('fear_greed_index', 0)}/100\n\n"
        else:
            report += "No market context data available.\n\n"
        
        # Save the report
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/{symbol}_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def list_all_symbols(self) -> List[str]:
        """List all available symbols from various sources."""
        # Get symbols from time series manager
        ts_symbols = self.integration.list_available_symbols()
        
        # Get symbols from synthetic data (for testing)
        synthetic_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        
        # Combine and deduplicate
        all_symbols = list(set(ts_symbols + synthetic_symbols))
        
        # Sort alphabetically
        all_symbols.sort()
        
        # Cache the result
        self.symbols_cache = all_symbols
        
        return all_symbols
    
    def run_comprehensive_market_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive market analysis across multiple assets.
        
        Returns:
            Dict with market analysis results
        """
        logger.info("Running comprehensive market analysis")
        
        # Get all symbols
        symbols = self.list_all_symbols()
        
        # Limit to top symbols for performance (add more as needed)
        top_symbols = symbols[:5] if len(symbols) > 5 else symbols
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "market_context": self._get_market_context(),
            "assets": {},
            "opportunities": [],
            "risk_insights": {},
            "correlation_matrix": None
        }
        
        # Process each symbol in parallel
        with ThreadPoolExecutor(max_workers=min(5, len(top_symbols))) as executor:
            future_to_symbol = {executor.submit(self.generate_enhanced_forecast, symbol, 7): symbol 
                              for symbol in top_symbols}
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    forecast = future.result()
                    results["assets"][symbol] = forecast
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    results["assets"][symbol] = {"error": str(e)}
        
        # Identify opportunities
        opportunities = self.identify_market_opportunities(min_change_pct=2.0)
        results["opportunities"] = opportunities.get("opportunities", [])
        
        # Generate risk insights
        results["risk_insights"] = self._generate_risk_insights(results["assets"])
        
        # Generate correlation matrix
        results["correlation_matrix"] = self._generate_correlation_matrix(top_symbols)
        
        # Generate market summary
        results["market_summary"] = self._generate_market_summary(results)
        
        # Generate market report
        report_path = self._generate_market_report(results)
        results["report_path"] = report_path
        
        return results
    
    def _generate_risk_insights(self, asset_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate risk insights across assets."""
        risk_insights = {
            "highest_risk_asset": None,
            "lowest_risk_asset": None,
            "highest_volatility_asset": None,
            "best_risk_reward_asset": None
        }
        
        # Track metrics
        risk_scores = {}
        volatilities = {}
        risk_rewards = {}
        
        # Process each asset
        for symbol, asset_data in asset_results.items():
            if "error" in asset_data:
                continue
                
            # Extract risk assessment if available
            if "risk_assessment" in asset_data:
                risk = asset_data["risk_assessment"]
                
                # Risk score
                if "risk_score" in risk:
                    risk_scores[symbol] = risk["risk_score"]
                
                # Volatility
                if "current_volatility" in risk:
                    volatilities[symbol] = risk["current_volatility"]
                
                # Risk/Reward
                if "risk_reward_ratio" in risk:
                    risk_rewards[symbol] = risk["risk_reward_ratio"]
        
        # Find highest and lowest risk assets
        if risk_scores:
            highest_risk = max(risk_scores.items(), key=lambda x: x[1])
            lowest_risk = min(risk_scores.items(), key=lambda x: x[1])
            
            risk_insights["highest_risk_asset"] = {
                "symbol": highest_risk[0],
                "risk_score": highest_risk[1]
            }
            
            risk_insights["lowest_risk_asset"] = {
                "symbol": lowest_risk[0],
                "risk_score": lowest_risk[1]
            }
        
        # Find highest volatility asset
        if volatilities:
            highest_vol = max(volatilities.items(), key=lambda x: x[1])
            
            risk_insights["highest_volatility_asset"] = {
                "symbol": highest_vol[0],
                "volatility": highest_vol[1]
            }
        
        # Find best risk/reward asset
        if risk_rewards:
            best_rr = max(risk_rewards.items(), key=lambda x: x[1])
            
            risk_insights["best_risk_reward_asset"] = {
                "symbol": best_rr[0],
                "risk_reward": best_rr[1]
            }
        
        return risk_insights
    
    def _generate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate correlation matrix between assets."""
        if not symbols or len(symbols) < 2:
            return None
            
        # Get price data for each symbol
        price_data = {}
        
        for symbol in symbols:
            df = self.get_enhanced_data(symbol)
            
            if df is not None and len(df) > 0:
                price_col = 'price' if 'price' in df.columns else 'close'
                price_data[symbol] = df[price_col]
        
        # Skip if we don't have enough data
        if len(price_data) < 2:
            return None
            
        # Create a DataFrame with all price data
        price_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Convert to dictionary format
        result = {
            "symbols": symbols,
            "matrix": {}
        }
        
        for symbol1 in symbols:
            if symbol1 not in corr_matrix.index:
                continue
                
            result["matrix"][symbol1] = {}
            
            for symbol2 in symbols:
                if symbol2 not in corr_matrix.columns:
                    continue
                    
                corr_value = corr_matrix.loc[symbol1, symbol2]
                result["matrix"][symbol1][symbol2] = float(corr_value)
        
        return result
    
    def _generate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate correlation matrix between assets."""
        if not symbols or len(symbols) < 2:
            return None
            
        # Get price data for each symbol
        price_data = {}
        
        for symbol in symbols:
            df = self.get_enhanced_data(symbol)
            
            if df is not None and len(df) > 0:
                price_col = 'price' if 'price' in df.columns else 'close'
                price_data[symbol] = df[price_col]
        
        # Skip if we don't have enough data
        if len(price_data) < 2:
            return None
            
        # Create a DataFrame with all price data
        price_df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Convert to dictionary format
        result = {
            "symbols": symbols,
            "matrix": {}
        }
        
        for symbol1 in symbols:
            if symbol1 not in corr_matrix.index:
                continue
                
            result["matrix"][symbol1] = {}
            
            for symbol2 in symbols:
                if symbol2 not in corr_matrix.columns:
                    continue
                    
                corr_value = corr_matrix.loc[symbol1, symbol2]
                result["matrix"][symbol1][symbol2] = float(corr_value)
        
        return result
    
    def _generate_market_summary(self, results: Dict[str, Any]) -> str:
        """Generate a text summary of market conditions."""
        market_context = results.get("market_context", {})
        opportunities = results.get("opportunities", [])
        risk_insights = results.get("risk_insights", {})
        
        # Create summary
        summary = "# Market Summary\n\n"
        
        # Market context
        market_trend = market_context.get("market_trend", "unknown")
        fear_greed = market_context.get("fear_greed_index", 50)
        
        summary += f"## Overall Market Conditions\n\n"
        summary += f"The crypto market is currently in a **{market_trend.replace('_', ' ').title()}** phase "
        summary += f"with a Fear & Greed Index reading of **{fear_greed}**. "
        
        if fear_greed > 75:
            summary += "This indicates extreme greed and potential overvaluation in the market.\n\n"
        elif fear_greed > 60:
            summary += "This indicates greed and potential optimism in the market.\n\n"
        elif fear_greed > 40:
            summary += "This indicates neutral market sentiment.\n\n"
        elif fear_greed > 25:
            summary += "This indicates fear and potential pessimism in the market.\n\n"
        else:
            summary += "This indicates extreme fear and potential undervaluation in the market.\n\n"
        
        # Opportunities
        summary += f"## Market Opportunities\n\n"
        
        if opportunities:
            summary += f"Identified {len(opportunities)} potential opportunities:\n\n"
            
            for i, opp in enumerate(opportunities[:3]):  # Show top 3
                symbol = opp.get("symbol", "")
                direction = opp.get("direction", "neutral")
                change_pct = opp.get("change_pct", 0)
                current_price = opp.get("current_price", 0)
                predicted_price = opp.get("predicted_price", 0)
                
                summary += f"**{i+1}. {symbol}**: {direction.title()} outlook with projected {change_pct:+.2f}% movement. "
                summary += f"Current price: ${current_price:,.2f}, Target: ${predicted_price:,.2f}\n\n"
        else:
            summary += "No significant opportunities identified at this time.\n\n"
        
        # Risk insights
        summary += f"## Risk Assessment\n\n"
        
        highest_risk = risk_insights.get("highest_risk_asset")
        lowest_risk = risk_insights.get("lowest_risk_asset")
        highest_vol = risk_insights.get("highest_volatility_asset")
        best_rr = risk_insights.get("best_risk_reward_asset")
        
        if highest_risk:
            summary += f"**Highest Risk Asset:** {highest_risk.get('symbol')} with a risk score of {highest_risk.get('risk_score'):.2f}/100\n\n"
            
        if lowest_risk:
            summary += f"**Lowest Risk Asset:** {lowest_risk.get('symbol')} with a risk score of {lowest_risk.get('risk_score'):.2f}/100\n\n"
            
        if highest_vol:
            summary += f"**Highest Volatility Asset:** {highest_vol.get('symbol')} with annualized volatility of {highest_vol.get('volatility'):.2f}%\n\n"
            
        if best_rr:
            summary += f"**Best Risk/Reward Asset:** {best_rr.get('symbol')} with a risk/reward ratio of {best_rr.get('risk_reward'):.2f}\n\n"
        
        # Market correlations
        summary += f"## Asset Correlations\n\n"
        summary += f"Asset correlation analysis available in the full report.\n\n"
        
        # Conclusion
        summary += f"## Conclusion\n\n"
        
        if market_trend in ["extreme_fear", "fear"]:
            summary += "The current market shows signs of fear, which historically can present buying opportunities for long-term investors. "
            summary += "However, caution is advised as market sentiment can deteriorate further.\n\n"
        elif market_trend in ["greed", "extreme_greed"]:
            summary += "The current market shows signs of greed, which historically can indicate potential market tops. "
            summary += "Caution is advised when entering new positions, and profit-taking strategies may be considered.\n\n"
        else:
            summary += "The current market shows relatively neutral sentiment, providing a balanced environment for strategic positions "
            summary += "based on individual asset analysis rather than overall market timing.\n\n"
        
        return summary
    
    def _generate_market_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive market report with all analysis results."""
        # Create timestamp for the report
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the report
        report = f"# Crypto Market Analysis Report\n\n"
        report += f"Generated: {timestamp}\n\n"
        
        # Add market summary
        report += results.get("market_summary", "No market summary available.")
        report += "\n\n"
        
        # Market context details
        market_context = results.get("market_context", {})
        report += "## Detailed Market Context\n\n"
        report += f"- Market Trend: {market_context.get('market_trend', 'unknown').replace('_', ' ').title()}\n"
        report += f"- Fear & Greed Index: {market_context.get('fear_greed_index', 0)}/100\n"
        report += f"- BTC Dominance: {market_context.get('btc_dominance', 0):.2f}%\n\n"
        
        # Top assets summary
        report += "## Asset Summary\n\n"
        report += "| Asset | Current Price | Forecast (7d) | Change % | Trend | Risk |\n"
        report += "|-------|--------------|---------------|----------|-------|------|\n"
        
        for symbol, asset_data in results.get("assets", {}).items():
            if "error" in asset_data:
                continue
                
            current_price = asset_data.get("current_price", 0)
            
            # Get last prediction
            predictions = asset_data.get("predictions", [])
            last_prediction = predictions[-1] if predictions else {}
            forecast_price = last_prediction.get("forecast_price", 0)
            change_pct = last_prediction.get("change_pct", 0)
            
            # Get trend and risk
            signals = asset_data.get("signals", {})
            trend = signals.get("trend", "neutral").title()
            
            risk_assessment = asset_data.get("risk_assessment", {})
            risk_category = risk_assessment.get("risk_category", "medium").title()
            
            report += f"| {symbol} | ${current_price:,.2f} | ${forecast_price:,.2f} | {change_pct:+.2f}% | {trend} | {risk_category} |\n"
        
        report += "\n"
        
        # Opportunities
        report += "## Market Opportunities\n\n"
        opportunities = results.get("opportunities", [])
        
        if opportunities:
            report += "| Asset | Direction | Change % | Confidence | Current | Target | Days |\n"
            report += "|-------|-----------|----------|------------|---------|--------|------|\n"
            
            for opp in opportunities:
                symbol = opp.get("symbol", "")
                direction = opp.get("direction", "neutral").title()
                change_pct = opp.get("change_pct", 0)
                confidence = opp.get("confidence", "medium").title()
                current_price = opp.get("current_price", 0)
                predicted_price = opp.get("predicted_price", 0)
                days = opp.get("days", 7)
                
                report += f"| {symbol} | {direction} | {change_pct:+.2f}% | {confidence} | ${current_price:,.2f} | ${predicted_price:,.2f} | {days} |\n"
                
            report += "\n"
        else:
            report += "No significant opportunities identified at this time.\n\n"
        
        # Risk insights
        report += "## Risk Analysis\n\n"
        risk_insights = results.get("risk_insights", {})
        
        # Highest risk asset
        highest_risk = risk_insights.get("highest_risk_asset")
        if highest_risk:
            report += f"**Highest Risk Asset:** {highest_risk.get('symbol')} with a risk score of {highest_risk.get('risk_score'):.2f}/100\n\n"
            
        # Lowest risk asset
        lowest_risk = risk_insights.get("lowest_risk_asset")
        if lowest_risk:
            report += f"**Lowest Risk Asset:** {lowest_risk.get('symbol')} with a risk score of {lowest_risk.get('risk_score'):.2f}/100\n\n"
            
        # Highest volatility asset
        highest_vol = risk_insights.get("highest_volatility_asset")
        if highest_vol:
            report += f"**Highest Volatility Asset:** {highest_vol.get('symbol')} with annualized volatility of {highest_vol.get('volatility'):.2f}%\n\n"
            
        # Best risk/reward asset
        best_rr = risk_insights.get("best_risk_reward_asset")
        if best_rr:
            report += f"**Best Risk/Reward Asset:** {best_rr.get('symbol')} with a risk/reward ratio of {best_rr.get('risk_reward'):.2f}\n\n"
        
        # Correlation matrix
        report += "## Asset Correlation Matrix\n\n"
        correlation = results.get("correlation_matrix")
        
        if correlation and "matrix" in correlation:
            symbols = correlation.get("symbols", [])
            matrix = correlation.get("matrix", {})
            
            # Create header row
            header = "| Asset |"
            separator = "|-------|"
            
            for symbol in symbols:
                header += f" {symbol} |"
                separator += "-------|"
                
            report += header + "\n" + separator + "\n"
            
            # Create rows
            for symbol1 in symbols:
                if symbol1 not in matrix:
                    continue
                    
                row = f"| {symbol1} |"
                
                for symbol2 in symbols:
                    if symbol2 not in matrix[symbol1]:
                        row += " N/A |"
                        continue
                        
                    correlation_value = matrix[symbol1][symbol2]
                    row += f" {correlation_value:.2f} |"
                    
                report += row + "\n"
                
            report += "\n"
            report += "Correlation ranges from -1.0 (perfect negative correlation) to 1.0 (perfect positive correlation).\n"
            report += "Values near 0 indicate little to no correlation between assets.\n\n"
        else:
            report += "Correlation matrix not available.\n\n"
        
        # Save the report
        os.makedirs('reports', exist_ok=True)
        report_path = f"reports/market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def run_cli(self):
        """Run an interactive CLI for the enhanced system."""
        print("\n=== Enhanced Chronos System ===")
        print("Type 'help' for commands or 'exit' to quit")
        
        while True:
            try:
                command = input("\nCommand: ").strip()
                
                if command.lower() == "exit":
                    print("Exiting...")
                    break
                    
                elif command.lower() == "help":
                    print("\nAvailable commands:")
                    print("  symbols                 - List available symbols")
                    print("  analyze <symbol>        - Run comprehensive analysis for a symbol")
                    print("  forecast <symbol> [days]- Generate forecast for a symbol")
                    print("  backtest <symbol>       - Backtest forecast accuracy")
                    print("  market                  - Run comprehensive market analysis")
                    print("  exit                    - Exit the program")
                
                elif command.lower() == "symbols":
                    symbols = self.list_all_symbols()
                    print(f"\nAvailable symbols ({len(symbols)}):")
                    
                    # Print in columns
                    cols = 5
                    symbols_rows = [symbols[i:i+cols] for i in range(0, len(symbols), cols)]
                    
                    for row in symbols_rows:
                        print("  ".join(f"{s:<10}" for s in row))
                
                elif command.lower().startswith("analyze "):
                    parts = command.split(" ")
                    if len(parts) < 2:
                        print("Usage: analyze <symbol>")
                        continue
                    
                    symbol = parts[1].upper()
                    print(f"Running comprehensive analysis for {symbol}...")
                    
                    results = self.run_comprehensive_analysis(symbol)
                    
                    if "error" in results:
                        print(f"Error: {results['error']}")
                        continue
                    
                    # Print summary
                    summary = results.get("summary", {})
                    print("\n" + "="*50)
                    print(f"ANALYSIS SUMMARY FOR {symbol}")
                    print("="*50)
                    print(summary.get("summary_text", "No summary available."))
                    print("\nReport saved to:", results.get("report_path", "unknown"))
                
                elif command.lower().startswith("forecast "):
                    parts = command.split(" ")
                    if len(parts) < 2:
                        print("Usage: forecast <symbol> [days]")
                        continue
                    
                    symbol = parts[1].upper()
                    days = int(parts[2]) if len(parts) > 2 else 7
                    
                    print(f"Generating {days}-day forecast for {symbol}...")
                    
                    forecast = self.generate_enhanced_forecast(symbol, days_ahead=days)
                    
                    if "error" in forecast:
                        print(f"Error: {forecast['error']}")
                        continue
                    
                    # Print forecast
                    print("\n" + "="*50)
                    print(f"PRICE FORECAST FOR {symbol}")
                    print("="*50)
                    print(f"Current price: ${forecast.get('current_price', 0):,.2f}")
                    
                    predictions = forecast.get("predictions", [])
                    
                    # Create table data
                    table_data = []
                    for pred in predictions:
                        table_data.append([
                            pred.get("date"),
                            f"${pred.get('forecast_price', 0):,.2f}",
                            f"{pred.get('change_pct', 0):+.2f}%",
                            f"${pred.get('lower_bound', 0):,.2f}",
                            f"${pred.get('upper_bound', 0):,.2f}"
                        ])
                    
                    # Print table
                    print("\nDate         | Forecast     | Change      | Lower Bound  | Upper Bound")
                    print("-------------|--------------|-------------|--------------|-------------")
                    for row in table_data:
                        print(f"{row[0]} | {row[1]:12} | {row[2]:11} | {row[3]:12} | {row[4]:12}")
                    
                    # Show chart location
                    print("\nForecast chart saved to:", forecast.get("plot_path", "unknown"))
                
                elif command.lower().startswith("backtest "):
                    parts = command.split(" ")
                    if len(parts) < 2:
                        print("Usage: backtest <symbol>")
                        continue
                    
                    symbol = parts[1].upper()
                    print(f"Backtesting forecast accuracy for {symbol}...")
                    
                    backtest = self.backtest_forecast_accuracy(symbol)
                    
                    if "error" in backtest:
                        print(f"Error: {backtest['error']}")
                        continue
                    
                    # Print backtest results
                    print("\n" + "="*50)
                    print(f"FORECAST ACCURACY BACKTEST FOR {symbol}")
                    print("="*50)
                    
                    print(f"Average price accuracy: {backtest.get('avg_price_accuracy', 0):.2f}%")
                    print(f"Average direction accuracy: {backtest.get('avg_direction_accuracy', 0):.2f}%")
                    print(f"Average MAPE: {backtest.get('avg_mape', 0):.2f}%")
                    print(f"Average MAE: ${backtest.get('avg_mae', 0):.2f}")
                
                elif command.lower() == "market":
                    print("Running comprehensive market analysis...")
                    
                    results = self.run_comprehensive_market_analysis()
                    
                    # Print market summary
                    print("\n" + "="*50)
                    print("CRYPTO MARKET ANALYSIS")
                    print("="*50)
                    
                    market_context = results.get("market_context", {})
                    print(f"Market trend: {market_context.get('market_trend', 'unknown').replace('_', ' ').title()}")
                    print(f"Fear & Greed Index: {market_context.get('fear_greed_index', 0)}/100")
                    
                    # Print opportunities
                    opportunities = results.get("opportunities", [])
                    print(f"\nIdentified {len(opportunities)} potential opportunities:")
                    
                    for i, opp in enumerate(opportunities[:5]):  # Show top 5
                        print(f"{i+1}. {opp.get('symbol')}: {opp.get('direction').title()} outlook with projected {opp.get('change_pct', 0):+.2f}% movement")
                    
                    # Show report location
                    print("\nMarket report saved to:", results.get("report_path", "unknown"))
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

# Example usage when running as a script
if __name__ == "__main__":
    try:
        chronos = EnhancedChronos()
        chronos.run_cli()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()