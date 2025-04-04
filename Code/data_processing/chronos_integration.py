import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import TimeSeriesManager and ChronosForecaster
from Code.data_processing.time_series_manager import TimeSeriesManager
from Code.data_processing.chronos_forecaster import ChronosForecaster

class ChronosIntegration:
    """
    Integration layer between TimeSeriesManager and ChronosForecaster
    to enable advanced forecasting capabilities while maintaining
    compatibility with existing systems.
    """
    
    def __init__(self, storage_manager=None):
        """
        Initialize the integration layer.
        
        Args:
            storage_manager: Storage manager instance (will be passed to TimeSeriesManager)
        """
        # Initialize TimeSeriesManager
        self.time_series_manager = TimeSeriesManager(storage_manager=storage_manager)
        
        # Initialize ChronosForecaster
        self.forecaster = ChronosForecaster()
        
        # Track available symbols
        self.available_symbols = set()
        
        logger.info("ChronosIntegration initialized")
    
    def list_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from the time series manager.
        
        Returns:
            List of available symbols
        """
        try:
            symbols = self.time_series_manager.get_processed_symbols()
            if not symbols:
                # If no processed symbols, try to retrieve from CSV loader
                symbols = self.time_series_manager.csv_loader.get_available_symbols()
                
            self.available_symbols = set(symbols)
            return list(self.available_symbols)
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_historical_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data to retrieve
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Retrieve data using TimeSeriesManager's storage connection
            data = self.time_series_manager.storage.retrieve_time_series(symbol, limit=days)
            
            if not data or len(data) == 0:
                logger.warning(f"No historical data retrieved for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure timestamp is properly handled
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            
            # Create price column if it doesn't exist (use close as price)
            if 'price' not in df.columns and 'close' in df.columns:
                df['price'] = df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get news sentiment data for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of sentiment data to retrieve
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            # Clean symbol to get base asset name
            base_asset = symbol.replace("USDT", "").replace("USD", "").lower()
            
            # Get sentiment stats from storage
            sentiment_data = self.time_series_manager.storage.get_sentiment_stats(base_asset, days)
            
            if not sentiment_data or 'sentiment_distribution' not in sentiment_data:
                logger.warning(f"No sentiment data available for {base_asset}")
                return None
                
            # Since get_sentiment_stats doesn't return daily data directly,
            # we'll create a synthetic daily sentiment series based on overall statistics
            avg_sentiment = sentiment_data.get('avg_sentiment', 0.5)
            sentiment_trend = sentiment_data.get('trend', 'stable')
            
            # Create an array of dates for the past 'days' days
            dates = [datetime.now() - timedelta(days=i) for i in range(days)]
            dates.reverse()  # Put in chronological order
            
            # Create a synthetic sentiment trend based on overall statistics
            sentiment_scores = []
            
            if sentiment_trend == 'improving':
                # Create an improving trend
                base_sentiment = max(0.3, avg_sentiment - 0.1)
                for i in range(len(dates)):
                    progress = i / len(dates)
                    sentiment_scores.append(base_sentiment + progress * 0.2)
            elif sentiment_trend == 'declining':
                # Create a declining trend
                base_sentiment = min(0.7, avg_sentiment + 0.1)
                for i in range(len(dates)):
                    progress = i / len(dates)
                    sentiment_scores.append(base_sentiment - progress * 0.2)
            else:
                # Create a stable trend with minor fluctuations
                for _ in range(len(dates)):
                    sentiment_scores.append(avg_sentiment + (np.random.random() - 0.5) * 0.1)
            
            # Create DataFrame
            sentiment_df = pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiment_scores,
                'article_count': 1  # Placeholder
            })
            
            if len(sentiment_df) > 0:
                sentiment_df = sentiment_df.set_index('date').sort_index()
                
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment data for {symbol}: {e}")
            return None
    
    def get_onchain_metrics(self, symbol: str) -> pd.DataFrame:
        """
        Get on-chain metrics for a symbol if available.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with on-chain metrics or None
        """
        # Placeholder for future implementation
        # This would retrieve on-chain data from the appropriate storage collection
        return None
    
    def forecast_price(self, symbol: str, horizon: str = 'short', days_ahead: int = 7, 
                     with_sentiment: bool = True, with_anomalies: bool = True) -> Dict[str, Any]:
        """
        Generate price forecast for a symbol.
        
        Args:
            symbol: Trading symbol
            horizon: 'short' (1-7 days) or 'long' (7-30 days)
            days_ahead: Number of days to forecast
            with_sentiment: Whether to include sentiment data
            with_anomalies: Whether to detect anomalies
            
        Returns:
            Dict with forecast results
        """
        logger.info(f"Generating {horizon}-term {days_ahead}-day forecast for {symbol}")
        
        # Normalize symbol
        if not symbol.endswith(("USDT", "USD")):
            symbol = f"{symbol}USDT"
        
        # Get historical data
        market_df = self.get_historical_data(symbol)
        if market_df is None or len(market_df) < 30:
            logger.error(f"Insufficient historical data for {symbol}")
            return {
                "error": f"Insufficient historical data for {symbol}",
                "symbol": symbol,
                "forecast_date": datetime.now().isoformat()
            }
        
        # Get sentiment data if requested
        sentiment_df = None
        if with_sentiment:
            sentiment_df = self.get_news_sentiment(symbol)
        
        # Get on-chain data if available
        onchain_df = self.get_onchain_metrics(symbol)
        
        # Generate forecast
        forecast_df = self.forecaster.forecast(
            symbol=symbol,
            market_df=market_df,
            horizon=horizon,
            days_ahead=days_ahead,
            sentiment_df=sentiment_df,
            onchain_df=onchain_df
        )
        
        if forecast_df is None or len(forecast_df) == 0:
            logger.error(f"Failed to generate forecast for {symbol}")
            return {
                "error": f"Failed to generate forecast for {symbol}",
                "symbol": symbol,
                "forecast_date": datetime.now().isoformat()
            }
        
        # Detect recent anomalies if requested
        anomalies = None
        if with_anomalies:
            try:
                anomalies_df = self.forecaster.detect_latest_anomalies(
                    symbol=symbol, 
                    market_df=market_df, 
                    lookback_days=30
                )
                
                if anomalies_df is not None and len(anomalies_df) > 0:
                    # Convert anomalies to list of dictionaries
                    anomalies = anomalies_df.reset_index().to_dict('records')
                    
                    # Format timestamps
                    for anomaly in anomalies:
                        if 'timestamp' in anomaly or 'index' in anomaly:
                            key = 'timestamp' if 'timestamp' in anomaly else 'index'
                            anomaly['date'] = pd.to_datetime(anomaly[key]).isoformat()
                            del anomaly[key]
            except Exception as e:
                logger.warning(f"Error detecting anomalies for {symbol}: {e}")
        
        # Create plot
        plot_path = None
        try:
            # Create plots directory
            plots_dir = os.path.join(project_root, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate plot filename
            filename = f"{symbol}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = os.path.join(plots_dir, filename)
            
            # Generate plot
            self.forecaster.plot_forecast(
                symbol=symbol,
                forecast_df=forecast_df,
                market_df=market_df,
                output_path=plot_path
            )
            
            # Make path relative for easier display
            plot_path = os.path.relpath(plot_path, project_root)
            
        except Exception as e:
            logger.warning(f"Error generating plot for {symbol}: {e}")
        
        # Format forecast results
        predictions = []
        for _, row in forecast_df.iterrows():
            prediction = {
                "date": row['date'].isoformat() if isinstance(row['date'], datetime) else row['date'],
                "predicted_price": float(row['forecast_price']),
                "change_pct": float(row['change_pct']),
                "lower_bound": float(row['lower_bound']) if 'lower_bound' in row else None,
                "upper_bound": float(row['upper_bound']) if 'upper_bound' in row else None
            }
            predictions.append(prediction)
        
        # Create response
        response = {
            "symbol": symbol,
            "forecast_date": datetime.now().isoformat(),
            "horizon": horizon,
            "days_ahead": days_ahead,
            "predictions": predictions,
            "latest_price": float(market_df['price'].iloc[-1]) if 'price' in market_df.columns else None,
            "plot_path": plot_path,
            "with_sentiment": with_sentiment
        }
        
        # Add anomalies if detected
        if anomalies:
            response["anomalies"] = anomalies
            response["anomaly_detected"] = True
        else:
            response["anomaly_detected"] = False
        
        # Add sentiment summary if used
        if with_sentiment and sentiment_df is not None and len(sentiment_df) > 0:
            response["sentiment_summary"] = {
                "avg_sentiment": float(sentiment_df['sentiment_score'].mean()),
                "sentiment_count": int(sentiment_df['article_count'].sum() if 'article_count' in sentiment_df.columns else len(sentiment_df)),
                "sentiment_trend": "positive" if sentiment_df['sentiment_score'].iloc[-1] > sentiment_df['sentiment_score'].mean() else "negative"
            }
        
        return response
    
    def bulk_forecast(self, symbols: List[str] = None, horizon: str = 'short', 
                    days_ahead: int = 7) -> Dict[str, Any]:
        """
        Generate forecasts for multiple symbols.
        
        Args:
            symbols: List of symbols to forecast (None for all available)
            horizon: 'short' or 'long'
            days_ahead: Number of days to forecast
            
        Returns:
            Dict with forecast results for each symbol
        """
        if symbols is None:
            symbols = self.list_available_symbols()
            
        if not symbols:
            logger.warning("No symbols available for forecasting")
            return {"error": "No symbols available for forecasting"}
        
        logger.info(f"Generating {horizon}-term {days_ahead}-day forecasts for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                forecast = self.forecast_price(
                    symbol=symbol,
                    horizon=horizon,
                    days_ahead=days_ahead,
                    with_sentiment=True,
                    with_anomalies=True
                )
                
                results[symbol] = forecast
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {e}")
                results[symbol] = {
                    "error": str(e),
                    "symbol": symbol,
                    "forecast_date": datetime.now().isoformat()
                }
        
        return {
            "forecasts": results,
            "timestamp": datetime.now().isoformat(),
            "symbols_count": len(symbols),
            "successful_forecasts": sum(1 for result in results.values() if "error" not in result)
        }
    
    def identify_market_opportunities(self, min_change_pct: float = 5.0, horizon: str = 'short', 
                                    days_ahead: int = 7) -> Dict[str, Any]:
        """
        Identify potential market opportunities based on forecasts.
        
        Args:
            min_change_pct: Minimum forecasted price change percentage
            horizon: 'short' or 'long'
            days_ahead: Number of days to forecast
            
        Returns:
            Dict with identified opportunities
        """
        # Get all available symbols
        symbols = self.list_available_symbols()
        
        if not symbols:
            logger.warning("No symbols available for opportunity identification")
            return {"error": "No symbols available"}
        
        logger.info(f"Identifying market opportunities across {len(symbols)} symbols")
        
        opportunities = []
        for symbol in symbols:
            try:
                # Generate forecast
                forecast = self.forecast_price(
                    symbol=symbol,
                    horizon=horizon,
                    days_ahead=days_ahead,
                    with_sentiment=True,
                    with_anomalies=True
                )
                
                if "error" in forecast:
                    continue
                
                # Check if any prediction exceeds threshold
                predictions = forecast.get("predictions", [])
                if not predictions:
                    continue
                
                # Check final prediction
                final_prediction = predictions[-1]
                change_pct = final_prediction.get("change_pct", 0)
                
                if abs(change_pct) >= min_change_pct:
                    # Determine direction and confidence
                    direction = "bullish" if change_pct > 0 else "bearish"
                    
                    # Calculate confidence level based on interval width
                    confidence = "medium"  # Default
                    if "lower_bound" in final_prediction and "upper_bound" in final_prediction:
                        range_width = final_prediction["upper_bound"] - final_prediction["lower_bound"]
                        predicted_price = final_prediction["predicted_price"]
                        
                        # Relative width as percentage of predicted price
                        relative_width = (range_width / predicted_price) * 100
                        
                        if relative_width < 10:
                            confidence = "high"
                        elif relative_width > 30:
                            confidence = "low"
                    
                    # Use sentiment if available
                    sentiment_aligned = None
                    if "sentiment_summary" in forecast:
                        sentiment_score = forecast["sentiment_summary"].get("avg_sentiment", 0.5)
                        sentiment_trend = forecast["sentiment_summary"].get("sentiment_trend")
                        
                        # Check if sentiment aligns with prediction direction
                        sentiment_aligned = (
                            (direction == "bullish" and sentiment_score > 0.55) or
                            (direction == "bearish" and sentiment_score < 0.45)
                        )
                    
                    # Create opportunity entry
                    opportunity = {
                        "symbol": symbol,
                        "direction": direction,
                        "confidence": confidence,
                        "change_pct": change_pct,
                        "days": days_ahead,
                        "current_price": forecast.get("latest_price"),
                        "predicted_price": final_prediction.get("predicted_price"),
                        "prediction_date": final_prediction.get("date"),
                        "sentiment_aligned": sentiment_aligned,
                        "anomaly_detected": forecast.get("anomaly_detected", False),
                        "plot_path": forecast.get("plot_path")
                    }
                    
                    opportunities.append(opportunity)
            
            except Exception as e:
                logger.error(f"Error processing {symbol} for opportunities: {e}")
        
        # Sort opportunities by absolute change percentage
        opportunities.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon,
            "days_ahead": days_ahead,
            "min_change_pct": min_change_pct,
            "total_symbols_analyzed": len(symbols),
            "opportunities_count": len(opportunities),
            "opportunities": opportunities
        }
    
    def retrain_models(self, symbols: List[str] = None, force: bool = False) -> Dict[str, Any]:
        """
        Retrain forecasting models for specified symbols.
        
        Args:
            symbols: List of symbols to retrain (None for all available)
            force: Force retraining regardless of last training date
            
        Returns:
            Dict with retraining results
        """
        if symbols is None:
            symbols = self.list_available_symbols()
            
        if not symbols:
            logger.warning("No symbols available for retraining")
            return {"error": "No symbols available for retraining"}
        
        logger.info(f"Retraining models for {len(symbols)} symbols")
        
        # Define function to get market data for a symbol
        def get_market_data(symbol):
            return self.get_historical_data(symbol)
        
        # Define function to get sentiment data for a symbol
        def get_sentiment_data(symbol):
            return self.get_news_sentiment(symbol)
        
        # Call auto_retrain in the forecaster
        results = self.forecaster.auto_retrain(
            symbols=symbols,
            market_data_func=get_market_data,
            sentiment_data_func=get_sentiment_data,
            force=force
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols_count": len(symbols),
            "successful_retrains": results.get("success", 0),
            "failed_retrains": results.get("failed", 0),
            "skipped_retrains": results.get("skipped", 0),
            "details": results.get("details", {})
        }
    
    def close(self):
        """Close any open connections."""
        self.time_series_manager.close()

# Example usage
if __name__ == "__main__":
    # Example standalone usage
    chronos = ChronosIntegration()
    available_symbols = chronos.list_available_symbols()
    print(f"Available symbols: {available_symbols}")
    
    if available_symbols:
        # Try forecasting for the first available symbol
        symbol = available_symbols[0]
        forecast = chronos.forecast_price(symbol)
        print(f"Forecast for {symbol}: {forecast}")
    
    chronos.close()