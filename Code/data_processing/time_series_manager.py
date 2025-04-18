
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class TimeSeriesManager:
    def __init__(self, csv_loader=None, storage_manager=None):
        if csv_loader is None:
            from Code.data_processing.csv_loader import CryptoCSVLoader
            self.csv_loader = CryptoCSVLoader()
        else:
            self.csv_loader = csv_loader

        if storage_manager is None:
            from Sample_Data.vector_store.storage_manager import StorageManager
            self.storage = StorageManager()
            self.storage.connect()
        else:
            self.storage = storage_manager

        self.processed_symbols = set()
        logger.info("Time Series Manager initialized")

    def get_recent_data(self, symbol, limit=100):
        try:
            data = self.storage.retrieve_time_series(symbol, limit=limit)
            if not data or len(data) == 0:
                logger.warning(f"No data found for {symbol}")
                return None

            logger.info(f"Retrieved {len(data)} data points for {symbol}")
            df = pd.DataFrame(data)

            required_columns = ['timestamp', 'price', 'volume_24h']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'price' and 'close' in df.columns:
                        df['price'] = df['close']
                    elif col == 'volume_24h' and 'volume' in df.columns:
                        df['volume_24h'] = df['volume']
                    else:
                        logger.warning(f"Missing column {col} for {symbol}")
                        df[col] = 0

            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            df = df.sort_values('timestamp')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return None

    def get_sentiment_data(self, symbol):
        import pandas as pd
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=30),
            'sentiment_score': [0.5] * 30
        })

    def forecast_prices(self, symbol, days_ahead=7):
        try:
            logger.info(f"Forecasting prices for {symbol} ({days_ahead} days ahead)")
            df = self.get_recent_data(symbol, limit=100)

            if df is None or len(df) < 30:
                logger.warning(f"Not enough data for {symbol} forecasting")
                return None

            sentiment_data = self.get_sentiment_data(symbol)
            logger.info(f"Using placeholder sentiment data for {symbol}")

            forecast_df = df.copy()
            window_size = min(30, len(df) // 3)

            if len(df) > window_size:
                forecast_df['MA'] = df['price'].rolling(window=window_size).mean()
            else:
                forecast_df['MA'] = df['price']
            forecast_df['MA'] = forecast_df['MA'].bfill()

            last_ma = forecast_df['MA'].iloc[-1]
            last_close = forecast_df['price'].iloc[-1]

            days_for_trend = min(7, len(df) - 1)
            trend_factor = (df['price'].iloc[-1] / df['price'].iloc[-days_for_trend-1]) - 1 if days_for_trend > 0 else 0

            last_date = pd.to_datetime(df.index[-1])
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]

            forecast = pd.DataFrame(index=future_dates, columns=['forecast_price', 'upper_bound', 'lower_bound'])
            for i, date in enumerate(future_dates):
                day_forecast = last_close * (1 + trend_factor * (i+1)/days_for_trend)
                noise_factor = 0.005
                forecast.loc[date, 'forecast_price'] = day_forecast
                forecast.loc[date, 'upper_bound'] = day_forecast * (1 + noise_factor * (i+1))
                forecast.loc[date, 'lower_bound'] = day_forecast * (1 - noise_factor * (i+1))

            logger.info(f"Forecast generated for {symbol}")
            return forecast

        except Exception as e:
            logger.error(f"Error forecasting prices for {symbol}: {str(e)}")
            last_close = 0
            try:
                if df is not None and len(df) > 0:
                    last_close = df['price'].iloc[-1]
            except:
                pass

            last_date = pd.Timestamp.now()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]

            forecast = pd.DataFrame(index=future_dates, columns=['forecast_price', 'upper_bound', 'lower_bound'])
            for date in future_dates:
                forecast.loc[date, 'forecast_price'] = last_close
                forecast.loc[date, 'upper_bound'] = last_close * 1.05
                forecast.loc[date, 'lower_bound'] = last_close * 0.95

            logger.warning(f"Using fallback forecast for {symbol}")
            return forecast

    def load_and_store_all(self, force: bool = False) -> Dict[str, int]:
        """
        Load and store all available time series data.

        Args:
            force: Force update even if data exists

        Returns:
            Dict with statistics on processed data
        """
        logger.info("Loading and storing all time series data")

        # Get available symbols from CSV loader
        symbols = self.csv_loader.get_available_symbols()
        logger.info(f"Found {len(symbols)} available symbols")

        results = {
            "total_symbols": len(symbols),
            "success_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "total_data_points": 0
        }

        for symbol in symbols:
            try:
                if not force and symbol in self.processed_symbols:
                    logger.info(f"Symbol {symbol} already processed, skipping")
                    results["skipped_count"] += 1
                    continue

                existing_data = self.storage.retrieve_time_series(symbol, limit=1)
                if existing_data and not force:
                    logger.info(f"Data for {symbol} already exists in Weaviate, skipping")
                    self.processed_symbols.add(symbol)
                    results["skipped_count"] += 1
                    continue

                logger.info(f"Loading data for {symbol}")
                data = self.csv_loader.load_historical_data(symbol.replace("USDT", ""))

                if not data:
                    logger.warning(f"No data found for {symbol}")
                    results["failure_count"] += 1
                    continue

                for point in data:
                    if "symbol" in point and not point["symbol"].endswith("USDT"):
                        point["symbol"] = f"{point['symbol']}USDT"

                logger.info(f"Storing {len(data)} data points for {symbol}")
                success = self.storage.store_time_series(data)

                if success:
                    logger.info(f"Stored {len(data)} data points for {symbol}")
                    self.processed_symbols.add(symbol)
                    results["success_count"] += 1
                    results["total_data_points"] += len(data)
                else:
                    logger.error(f"Failed to store data for {symbol}")
                    results["failure_count"] += 1

                time.sleep(0.1)  # Avoid overwhelming the database

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results["failure_count"] += 1

        logger.info("Completed time series storage process")
        return results


    def get_processed_symbols(self) -> List[str]:
        return list(self.processed_symbols)

    def close(self):
        if hasattr(self.storage, 'close'):
            self.storage.close()

if __name__ == "__main__":
    manager = TimeSeriesManager()

    try:
        results = manager.load_and_store_all(force=True)
        print(f"Processed {results['success_count']} symbols successfully")
        print(f"Total data points stored: {results['total_data_points']}")
        print("Skipped:", manager.get_processed_symbols())
        print(f"Skipped symbols: {results['skipped_count']}")
    finally:
        manager.close()
