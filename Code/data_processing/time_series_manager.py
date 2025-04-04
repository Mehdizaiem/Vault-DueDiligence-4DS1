# File path: Code/data_processing/time_series_manager.py

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class TimeSeriesManager:
    """
    Manager for loading, processing, and storing time series data
    in the CryptoTimeSeries collection.
    """
    
    def __init__(self, csv_loader=None, storage_manager=None):
        """
        Initialize the time series manager.
        
        Args:
            csv_loader: CSV loader instance (will be created if None)
            storage_manager: Storage manager instance (will be created if None)
        """
        # Initialize CSV loader if not provided
        if csv_loader is None:
            from Code.data_processing.csv_loader import CryptoCSVLoader
            self.csv_loader = CryptoCSVLoader()
        else:
            self.csv_loader = csv_loader
        
        # Initialize storage manager if not provided
        if storage_manager is None:
            from Sample_Data.vector_store.storage_manager import StorageManager
            self.storage = StorageManager()
            self.storage.connect()
        else:
            self.storage = storage_manager
        
        # Track processed symbols
        self.processed_symbols = set()
        
        logger.info("Time Series Manager initialized")
        
    def get_recent_data(self, symbol, limit=100):
        """
        Get most recent price data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            limit: Maximum number of data points to retrieve
            
        Returns:
            DataFrame with price data or None if error
        """
        try:
            # Retrieve time series data from storage
            data = self.storage.retrieve_time_series(symbol, limit=limit)
            
            if not data or len(data) == 0:
                logger.warning(f"No data found for {symbol}")
                return None
                
            logger.info(f"Retrieved {len(data)} data points for {symbol}")
            
            # Convert to DataFrame
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Make sure important columns exist
            required_columns = ['timestamp', 'price', 'volume_24h']
            for col in required_columns:
                if col not in df.columns:
                    # For price and volume, try alternate column names
                    if col == 'price' and 'close' in df.columns:
                        df['price'] = df['close']
                    elif col == 'volume_24h' and 'volume' in df.columns:
                        df['volume_24h'] = df['volume']
                    else:
                        logger.warning(f"Missing required column {col} for {symbol}")
                        # Add empty column as placeholder
                        df[col] = 0
            
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return None

    def get_sentiment_data(self, symbol):
        """
        Get sentiment data for a symbol (placeholder implementation).
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Placeholder sentiment data
        """
        # This is a placeholder - in a real implementation, you would fetch sentiment
        # data from your news sentiment collection
        import pandas as pd
        
        # Create a placeholder DataFrame for now
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=30),
            'sentiment_score': [0.5] * 30
        })

    def forecast_prices(self, symbol, days_ahead=7):
        """
        Create price forecasts with improved error handling.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            days_ahead: Number of days to forecast
            
        Returns:
            DataFrame with forecasted prices or None if error
        """
        try:
            logger.info(f"Forecasting prices for {symbol} ({days_ahead} days ahead)")
            
            # Get historical data
            df = self.get_recent_data(symbol, limit=100)
            
            if df is None or len(df) < 30:  # Need sufficient data for forecasting
                logger.warning(f"Insufficient data for {symbol} forecasting (need at least 30 points)")
                return None
                
            # Get sentiment data (placeholder or actual implementation)
            sentiment_data = self.get_sentiment_data(symbol)
            logger.info(f"Getting sentiment data for {symbol} (placeholder)")
            
            # Choose forecast model
            logger.info(f"Getting forecast model for {symbol} (using simple MA)")
            
            # Make a copy to avoid SettingWithCopyWarning
            forecast_df = df.copy()
            
            # Calculate moving averages for prediction
            # Use shorter windows if we have limited data
            window_size = min(30, len(df) // 3)
            
            # Add moving average column safely
            if len(df) > window_size:
                forecast_df['MA'] = df['price'].rolling(window=window_size).mean()
            else:
                # Fallback if we don't have enough data
                forecast_df['MA'] = df['price']
            
            # Fill NaN values that come from the rolling window
            forecast_df['MA'] = forecast_df['MA'].bfill()
            
            # Get last available value for prediction
            last_ma = forecast_df['MA'].iloc[-1] if len(forecast_df) > 0 else df['price'].iloc[-1]
            last_close = forecast_df['price'].iloc[-1] if len(forecast_df) > 0 else 0
            
            # Calculate trend factor over last 7 days (if available)
            days_for_trend = min(7, len(df) - 1)
            if days_for_trend > 0 and len(df) > days_for_trend:
                trend_factor = (df['price'].iloc[-1] / df['price'].iloc[-days_for_trend-1]) - 1
            else:
                trend_factor = 0
                
            # Generate future dates
            import pandas as pd
            last_date = df.index[-1] if len(df) > 0 else pd.Timestamp.now()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]
            
            # Create forecast dataframe
            forecast = pd.DataFrame(index=future_dates, columns=['forecast_price', 'upper_bound', 'lower_bound'])
            
            # Generate simple forecast with trend adjustment
            for i, date in enumerate(future_dates):
                # Apply trend to create a simple forecast
                day_forecast = last_close * (1 + trend_factor * (i+1)/days_for_trend)
                
                # Add slight noise for variation
                noise_factor = 0.005  # 0.5% noise
                forecast.loc[date, 'forecast_price'] = day_forecast
                forecast.loc[date, 'upper_bound'] = day_forecast * (1 + noise_factor * (i+1))
                forecast.loc[date, 'lower_bound'] = day_forecast * (1 - noise_factor * (i+1))
            
            logger.info(f"Forecast generated for {symbol} ({days_ahead} days ahead)")
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting prices for {symbol}: {str(e)}")
            # Rather than just logging the error, return a default forecast
            last_close = 0
            try:
                if df is not None and len(df) > 0:
                    last_close = df['price'].iloc[-1]
            except:
                pass
                
            # Create a fallback forecast
            import pandas as pd
            last_date = pd.Timestamp.now()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]
            
            # Create a flat forecast as fallback
            forecast = pd.DataFrame(index=future_dates, columns=['forecast_price', 'upper_bound', 'lower_bound'])
            for date in future_dates:
                forecast.loc[date, 'forecast_price'] = last_close
                forecast.loc[date, 'upper_bound'] = last_close * 1.05  # 5% upper bound
                forecast.loc[date, 'lower_bound'] = last_close * 0.95  # 5% lower bound
                
            logger.warning(f"Using fallback forecast for {symbol} due to error")
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
        
        # Get available symbols
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
                # Check if we need to process this symbol
                if not force and symbol in self.processed_symbols:
                    logger.info(f"Symbol {symbol} already processed, skipping")
                    results["skipped_count"] += 1
                    continue
                
                # Check if data already exists in Weaviate
                existing_data = self.storage.retrieve_time_series(symbol, limit=1)
                if existing_data and not force:
                    logger.info(f"Data for {symbol} already exists in Weaviate, skipping")
                    self.processed_symbols.add(symbol)
                    results["skipped_count"] += 1
                    continue
                
                # Load data from CSV
                logger.info(f"Loading data for {symbol}")
                data = self.csv_loader.load_historical_data(symbol)
                
                if not data:
                    logger.warning(f"No data found for {symbol}")
                    results["failure_count"] += 1
                    continue
                
                # Store in Weaviate
                logger.info(f"Storing {len(data)} data points for {symbol}")
                success = self.storage.store_time_series(data)
                
                if success:
                    logger.info(f"Successfully stored data for {symbol}")
                    self.processed_symbols.add(symbol)
                    results["success_count"] += 1
                    results["total_data_points"] += len(data)
                else:
                    logger.error(f"Failed to store data for {symbol}")
                    results["failure_count"] += 1
                
                # Small delay to avoid overwhelming Weaviate
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results["failure_count"] += 1
        
        logger.info(f"Completed processing {results['total_symbols']} symbols")
        return results
    
    def load_and_store_symbol(self, symbol: str, force: bool = False) -> bool:
        """
        Load and store time series data for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            force: Force update even if data exists
            
        Returns:
            True if successful
        """
        logger.info(f"Loading and storing data for {symbol}")
        
        try:
            # Check if symbol already processed
            if not force and symbol in self.processed_symbols:
                logger.info(f"Symbol {symbol} already processed, skipping")
                return True
            
            # Normalize symbol format (strip USDT if present)
            csv_symbol = symbol.replace("USDT", "")
            
            # Check if data already exists in Weaviate
            existing_data = self.storage.retrieve_time_series(symbol, limit=1)
            if existing_data and not force:
                logger.info(f"Data for {symbol} already exists in Weaviate, skipping")
                self.processed_symbols.add(symbol)
                return True
            
            # Load data from CSV using the normalized symbol
            data = self.csv_loader.load_historical_data(csv_symbol)
            
            if not data:
                logger.warning(f"No data found for {csv_symbol}")
                return False
            
            # Ensure all data points have the correct symbol format (with USDT)
            for point in data:
                # Make sure the symbol is consistently formatted with USDT
                if "symbol" in point and not point["symbol"].endswith("USDT"):
                    point["symbol"] = f"{point['symbol']}USDT"
            
            # Store in Weaviate
            success = self.storage.store_time_series(data)
            
            if success:
                logger.info(f"Successfully stored {len(data)} data points for {symbol}")
                self.processed_symbols.add(symbol)
                return True
            else:
                logger.error(f"Failed to store data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False
        
    def get_processed_symbols(self) -> List[str]:
        """Get list of processed symbols"""
        return list(self.processed_symbols)
    
    
    
    def close(self):
        """Close the storage connection"""
        if hasattr(self.storage, 'close'):
            self.storage.close()

# Example usage
if __name__ == "__main__":
    manager = TimeSeriesManager()
    
    try:
        # Either process all symbols
        results = manager.load_and_store_all()
        print(f"Processed {results['success_count']} symbols successfully")
        print(f"Total data points stored: {results['total_data_points']}")
        
        # Or process a specific symbol
        # success = manager.load_and_store_symbol("BTCUSD")
        # print(f"Successfully processed BTCUSD: {success}")
    finally:
        manager.close()