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
    that enables advanced forecasting capabilities while maintaining
    compatibility with existing systems in the agentic RAG pipeline.
    
    Provides:
    - Market data retrieval from vector store
    - Sentiment data integration
    - On-chain metrics integration
    - Automated model management (training, evaluation, selection)
    - Price forecasting with multiple models
    - Market insights generation
    - Anomaly detection
    - Trading opportunity identification
    """
    
    def __init__(self, storage_manager=None):
        """
        Initialize the integration layer.
        
        Args:
            storage_manager: Storage manager instance (will be passed to TimeSeriesManager)
        """
        # Initialize TimeSeriesManager for data access
        self.time_series_manager = TimeSeriesManager(storage_manager=storage_manager)
        
        # Initialize ChronosForecaster with improved model paths
        forecast_model_dir = os.path.join(project_root, "models", "chronos")
        os.makedirs(forecast_model_dir, exist_ok=True)
        self.forecaster = ChronosForecaster(model_dir=forecast_model_dir)
        
        # Track available symbols
        self.available_symbols = set()
        
        # Cache settings
        self.cache_enabled = True
        self.cache_duration = {'market': 15, 'sentiment': 60, 'forecast': 60}  # minutes
        self.cache = {'market': {}, 'sentiment': {}, 'forecasts': {}}
        
        # Create plot directory
        self.plots_dir = os.path.join(project_root, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        logger.info(f"ChronosIntegration initialized with {len(self.forecaster.available_models)} model types")
    
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
    
    def get_historical_data(self, symbol: str, days: int = 365, use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical data for a symbol with caching.
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data to retrieve
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical data
        """
        # Check cache if enabled
        cache_key = f"{symbol}_{days}"
        if use_cache and self.cache_enabled and cache_key in self.cache['market']:
            cache_entry = self.cache['market'][cache_key]
            cache_time = cache_entry.get('timestamp')
            cache_age = (datetime.now() - cache_time).total_seconds() / 60  # in minutes
            
            if cache_age < self.cache_duration['market']:
                logger.info(f"Using cached market data for {symbol} (age: {cache_age:.1f} min)")
                return cache_entry.get('data')
        
        try:
            # Standardize symbol format
            if not symbol.upper().endswith(("USDT", "USD")):
                symbol = f"{symbol.upper()}USDT"
            
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
            
            # Update cache
            if self.cache_enabled:
                self.cache['market'][cache_key] = {
                    'data': df,
                    'timestamp': datetime.now()
                }
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return None
    
    def get_news_sentiment(self, symbol: str, days: int = 30, use_cache: bool = True) -> pd.DataFrame:
        """
        Get news sentiment data for a symbol with caching.
        
        Args:
            symbol: Trading symbol
            days: Number of days of sentiment data to retrieve
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with sentiment data
        """
        # Check cache if enabled
        cache_key = f"{symbol}_{days}"
        if use_cache and self.cache_enabled and cache_key in self.cache['sentiment']:
            cache_entry = self.cache['sentiment'][cache_key]
            cache_time = cache_entry.get('timestamp')
            cache_age = (datetime.now() - cache_time).total_seconds() / 60  # in minutes
            
            if cache_age < self.cache_duration['sentiment']:
                logger.info(f"Using cached sentiment data for {symbol} (age: {cache_age:.1f} min)")
                return cache_entry.get('data')
        
        try:
            # Clean symbol to get base asset name
            base_asset = symbol.replace("USDT", "").replace("USD", "").lower()
            
            # Get sentiment stats from storage
            sentiment_data = self.time_series_manager.storage.get_sentiment_stats(base_asset, days)
            
            if not sentiment_data or 'sentiment_distribution' not in sentiment_data:
                logger.warning(f"No sentiment data available for {base_asset}")
                return self._generate_synthetic_sentiment(symbol, days)
                
            # Since get_sentiment_stats doesn't return daily data directly,
            # We'll extract actual sentiment articles if available
            sentiment_articles = self.time_series_manager.storage.get_sentiment_articles(base_asset, limit=100)
            
            if sentiment_articles and len(sentiment_articles) > 0:
                # Create DataFrame from articles
                sentiment_df = pd.DataFrame(sentiment_articles)
                
                # Ensure date column exists and is datetime
                if 'date' in sentiment_df.columns:
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                elif 'timestamp' in sentiment_df.columns:
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['timestamp'])
                    sentiment_df = sentiment_df.drop('timestamp', axis=1)
                else:
                    # Create date from analyzed_at if available
                    if 'analyzed_at' in sentiment_df.columns:
                        sentiment_df['date'] = pd.to_datetime(sentiment_df['analyzed_at'])
                    else:
                        # Use current date as fallback
                        sentiment_df['date'] = datetime.now()
                
                # Filter to requested time period
                start_date = datetime.now() - timedelta(days=days)
                sentiment_df = sentiment_df[sentiment_df['date'] >= start_date]
                
                # Ensure sentiment_score exists
                if 'sentiment_score' not in sentiment_df.columns and 'sentiment' in sentiment_df.columns:
                    sentiment_df['sentiment_score'] = sentiment_df['sentiment']
                elif 'sentiment_score' not in sentiment_df.columns:
                    # Create from sentiment_label if available
                    if 'sentiment_label' in sentiment_df.columns:
                        label_map = {'POSITIVE': 0.75, 'NEUTRAL': 0.5, 'NEGATIVE': 0.25}
                        sentiment_df['sentiment_score'] = sentiment_df['sentiment_label'].map(
                            lambda x: label_map.get(x, 0.5) if isinstance(x, str) else 0.5
                        )
                    else:
                        # Use a neutral score as fallback
                        sentiment_df['sentiment_score'] = 0.5
                
                # Sort by date
                sentiment_df = sentiment_df.sort_values('date')
                
                # Set date as index for easy alignment with market data
                sentiment_df = sentiment_df.set_index('date')
                
                # Add article count column if not present
                if 'article_count' not in sentiment_df.columns:
                    sentiment_df['article_count'] = 1
                
                # Update cache
                if self.cache_enabled:
                    self.cache['sentiment'][cache_key] = {
                        'data': sentiment_df,
                        'timestamp': datetime.now()
                    }
                
                return sentiment_df
            else:
                # If no articles available, generate synthetic sentiment from stats
                return self._generate_synthetic_sentiment(symbol, days, sentiment_data)
                
        except Exception as e:
            logger.error(f"Error retrieving sentiment data for {symbol}: {e}")
            return self._generate_synthetic_sentiment(symbol, days)
    
    def _generate_synthetic_sentiment(self, symbol: str, days: int, sentiment_stats: Dict = None) -> pd.DataFrame:
        """
        Generate synthetic sentiment data based on overall statistics or defaults.
        
        Args:
            symbol: Trading symbol
            days: Number of days of sentiment data to generate
            sentiment_stats: Statistics from sentiment analysis if available
            
        Returns:
            DataFrame with synthetic sentiment data
        """
        logger.info(f"Generating synthetic sentiment data for {symbol} ({days} days)")
        
        # Get average sentiment from stats or use neutral default
        avg_sentiment = 0.5
        sentiment_trend = "stable"
        
        if sentiment_stats:
            avg_sentiment = sentiment_stats.get('avg_sentiment', 0.5)
            sentiment_trend = sentiment_stats.get('trend', 'stable')
        
        # Create an array of dates for the past 'days' days
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Put in chronological order
        
        # Generate a synthetic sentiment trend based on overall statistics
        sentiment_scores = []
        np.random.seed(hash(symbol) % 10000)  # Deterministic but unique per symbol
        
        if sentiment_trend == 'improving':
            # Create an improving trend
            base_sentiment = max(0.3, avg_sentiment - 0.1)
            for i in range(len(dates)):
                progress = i / len(dates)
                # Add some noise to the trend
                noise = (np.random.random() - 0.5) * 0.1
                sentiment_scores.append(min(0.9, base_sentiment + progress * 0.2 + noise))
        elif sentiment_trend == 'declining':
            # Create a declining trend
            base_sentiment = min(0.7, avg_sentiment + 0.1)
            for i in range(len(dates)):
                progress = i / len(dates)
                # Add some noise to the trend
                noise = (np.random.random() - 0.5) * 0.1
                sentiment_scores.append(max(0.1, base_sentiment - progress * 0.2 + noise))
        else:
            # Create a stable trend with minor fluctuations
            for _ in range(len(dates)):
                sentiment_scores.append(
                    min(0.9, max(0.1, avg_sentiment + (np.random.random() - 0.5) * 0.2))
                )
        
        # Create article counts (more articles on days with extreme sentiment)
        article_counts = []
        for score in sentiment_scores:
            # Sentiment far from neutral (0.5) tends to have more articles
            sentiment_extremity = abs(score - 0.5) * 2  # 0 for neutral, 1 for extreme
            # More articles for extreme sentiment (1-5 articles per day)
            count = max(1, int(1 + sentiment_extremity * 4 + np.random.random() * 2))
            article_counts.append(count)
        
        # Create DataFrame
        sentiment_df = pd.DataFrame({
            'sentiment_score': sentiment_scores,
            'article_count': article_counts
        }, index=dates)
        
        # Update cache if enabled
        cache_key = f"{symbol}_{days}"
        if self.cache_enabled:
            self.cache['sentiment'][cache_key] = {
                'data': sentiment_df,
                'timestamp': datetime.now()
            }
        
        return sentiment_df
    
    def get_onchain_metrics(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get on-chain metrics for a symbol if available.
        
        Args:
            symbol: Trading symbol
            days: Number of days of on-chain data to retrieve
            
        Returns:
            DataFrame with on-chain metrics or None
        """
        try:
            # Clean symbol to get base asset name
            base_asset = symbol.replace("USDT", "").replace("USD", "").lower()
            
            # Try to retrieve on-chain data from storage
            onchain_data = self.time_series_manager.storage.get_onchain_metrics(base_asset, days)
            
            if not onchain_data or not isinstance(onchain_data, list) or len(onchain_data) == 0:
                logger.warning(f"No on-chain data available for {base_asset}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(onchain_data)
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.set_index('timestamp').sort_index()
                df = df.drop('date', axis=1)
            else:
                # No timestamp column, return None
                logger.warning(f"On-chain data for {base_asset} lacks timestamp information")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving on-chain data for {symbol}: {e}")
            return None
    
    def forecast_price(self, symbol: str, model_type: str = 'ensemble', days_ahead: int = 7, 
                     with_sentiment: bool = True, with_onchain: bool = True,
                     with_anomalies: bool = True, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate price forecast for a symbol using selected model type.
        
        Args:
            symbol: Trading symbol
            model_type: Model type to use ('short', 'long', 'lstm', 'transformer', 'ensemble')
            days_ahead: Number of days to forecast
            with_sentiment: Whether to include sentiment data
            with_onchain: Whether to include on-chain data
            with_anomalies: Whether to detect anomalies
            force_retrain: Whether to force model retraining
            
        Returns:
            Dict with forecast results
        """
        logger.info(f"Generating {model_type} {days_ahead}-day forecast for {symbol}")
        
        # Check cache if not forcing retrain
        cache_key = f"{symbol}_{model_type}_{days_ahead}"
        if not force_retrain and self.cache_enabled and cache_key in self.cache['forecasts']:
            cache_entry = self.cache['forecasts'][cache_key]
            cache_time = cache_entry.get('timestamp')
            cache_age = (datetime.now() - cache_time).total_seconds() / 60  # in minutes
            
            if cache_age < self.cache_duration['forecast']:
                logger.info(f"Using cached forecast for {symbol} (age: {cache_age:.1f} min)")
                return cache_entry.get('data')
        
        # Normalize symbol
        if not symbol.upper().endswith(("USDT", "USD")):
            symbol = f"{symbol.upper()}USDT"
        
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
        
        # Get on-chain data if requested and available
        onchain_df = None
        if with_onchain:
            onchain_df = self.get_onchain_metrics(symbol)
        
        # Map model type to forecaster horizon
        model_map = {
            'short': 'short',
            'long': 'long',
            'lstm': 'lstm',
            'transformer': 'transformer',
            'ensemble': 'ensemble'
        }
        
        # Default to ensemble if invalid model type
        horizon = model_map.get(model_type.lower(), 'ensemble')
        
        # Generate forecast
        forecast_df = self.forecaster.forecast(
            symbol=symbol,
            market_df=market_df,
            horizon=horizon,
            days_ahead=days_ahead,
            sentiment_df=sentiment_df,
            onchain_df=onchain_df,
            retrain=force_retrain
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
        
        # Generate market insights
        insights = self.forecaster.generate_market_insights(
            symbol=symbol,
            market_df=market_df,
            forecast_df=forecast_df,
            sentiment_df=sentiment_df,
            days=30
        )
        
        # Create plot
        plot_path = None
        try:
            # Generate plot filename
            filename = f"{symbol}_{model_type}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_path = os.path.join(self.plots_dir, filename)
            
            # Generate plot
            self.forecaster.plot_forecast(
                symbol=symbol,
                forecast_df=forecast_df,
                market_df=market_df,
                output_path=plot_path,
                show_anomalies=with_anomalies
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
            "model_type": model_type,
            "days_ahead": days_ahead,
            "predictions": predictions,
            "latest_price": float(market_df['price'].iloc[-1]) if 'price' in market_df.columns else None,
            "plot_path": plot_path,
            "insights": insights
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
        
        # Update cache
        if self.cache_enabled:
            self.cache['forecasts'][cache_key] = {
                'data': response,
                'timestamp': datetime.now()
            }
        
        return response
    
    def bulk_forecast(self, symbols: List[str] = None, model_type: str = 'ensemble', 
                    days_ahead: int = 7, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Generate forecasts for multiple symbols.
        
        Args:
            symbols: List of symbols to forecast (None for all available)
            model_type: Model type to use
            days_ahead: Number of days to forecast
            force_retrain: Whether to force model retraining
            
        Returns:
            Dict with forecast results for each symbol
        """
        if symbols is None:
            symbols = self.list_available_symbols()
            
        if not symbols:
            logger.warning("No symbols available for forecasting")
            return {"error": "No symbols available for forecasting"}
        
        logger.info(f"Generating {model_type} {days_ahead}-day forecasts for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                forecast = self.forecast_price(
                    symbol=symbol,
                    model_type=model_type,
                    days_ahead=days_ahead,
                    with_sentiment=True,
                    with_onchain=True,
                    with_anomalies=True,
                    force_retrain=force_retrain
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
    
    def identify_market_opportunities(self, min_change_pct: float = 5.0, model_type: str = 'ensemble', 
                                    days_ahead: int = 7, confidence_threshold: str = 'medium') -> Dict[str, Any]:
        """
        Identify potential market opportunities based on forecasts.
        
        Args:
            min_change_pct: Minimum forecasted price change percentage
            model_type: Model type to use for forecasts
            days_ahead: Number of days to forecast
            confidence_threshold: Minimum confidence level ('low', 'medium', 'high')
            
        Returns:
            Dict with identified opportunities
        """
        # Get all available symbols
        symbols = self.list_available_symbols()
        
        if not symbols:
            logger.warning("No symbols available for opportunity identification")
            return {"error": "No symbols available"}
        
        logger.info(f"Identifying market opportunities across {len(symbols)} symbols")
        
        # Map confidence thresholds to numerical values
        confidence_map = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        
        min_confidence = confidence_map.get(confidence_threshold.lower(), 2)
        
        opportunities = []
        for symbol in symbols:
            try:
                # Generate forecast
                forecast = self.forecast_price(
                    symbol=symbol,
                    model_type=model_type,
                    days_ahead=days_ahead,
                    with_sentiment=True,
                    with_onchain=True,
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
                    confidence_score = 2
                    
                    if "lower_bound" in final_prediction and "upper_bound" in final_prediction:
                        range_width = final_prediction["upper_bound"] - final_prediction["lower_bound"]
                        predicted_price = final_prediction["predicted_price"]
                        
                        # Relative width as percentage of predicted price
                        relative_width = (range_width / predicted_price) * 100
                        
                        if relative_width < 10:
                            confidence = "high"
                            confidence_score = 3
                        elif relative_width > 30:
                            confidence = "low"
                            confidence_score = 1
                    
                    # Skip if confidence is below threshold
                    if confidence_score < min_confidence:
                        continue
                    
                    # Get RSI data for additional confirmation
                    rsi = None
                    if "insights" in forecast and "rsi" in forecast["insights"]:
                        rsi = forecast["insights"]["rsi"]
                    
                    # Check if RSI confirms trend direction
                    rsi_confirms = None
                    if rsi is not None:
                        if direction == "bullish" and rsi < 70:  # Not overbought
                            rsi_confirms = True
                        elif direction == "bearish" and rsi > 30:  # Not oversold
                            rsi_confirms = True
                        else:
                            rsi_confirms = False
                    
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
                    
                    # Check if there are anomalies
                    anomalies = forecast.get("anomaly_detected", False)
                    
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
                        "rsi": rsi,
                        "rsi_confirms": rsi_confirms,
                        "sentiment_aligned": sentiment_aligned,
                        "anomaly_detected": anomalies,
                        "plot_path": forecast.get("plot_path")
                    }
                    
                    # Add insights if available
                    if "insights" in forecast:
                        insights = forecast["insights"]
                        
                        # Extract key insights
                        opportunity["market_summary"] = insights.get("market_summary")
                        
                        # Add support/resistance levels
                        opportunity["support_levels"] = insights.get("support_levels", [])
                        opportunity["resistance_levels"] = insights.get("resistance_levels", [])
                        
                        # Add trend data
                        opportunity["trend"] = insights.get("trend")
                        opportunity["volatility"] = insights.get("volatility")
                    
                    opportunities.append(opportunity)
            
            except Exception as e:
                logger.error(f"Error processing {symbol} for opportunities: {e}")
        
        # Sort opportunities by confidence and change percentage
        opportunities.sort(key=lambda x: (
            confidence_map.get(x["confidence"], 0), 
            abs(x["change_pct"])
        ), reverse=True)
        
        # Separate bullish and bearish opportunities
        bullish = [op for op in opportunities if op["direction"] == "bullish"]
        bearish = [op for op in opportunities if op["direction"] == "bearish"]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "days_ahead": days_ahead,
            "min_change_pct": min_change_pct,
            "confidence_threshold": confidence_threshold,
            "total_symbols_analyzed": len(symbols),
            "opportunities_count": len(opportunities),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "bullish_opportunities": bullish,
            "bearish_opportunities": bearish
        }
    
    def evaluate_models(self, symbol: str, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate forecast models for a specific symbol.
        
        Args:
            symbol: Trading symbol to evaluate
            model_types: List of model types to evaluate (None for all available)
            
        Returns:
            Dict with evaluation results
        """
        if model_types is None:
            model_types = ['short', 'long', 'lstm', 'transformer', 'ensemble']
        
        # Filter to available models
        model_types = [mt for mt in model_types if mt in self.forecaster.available_models]
        
        logger.info(f"Evaluating {len(model_types)} model types for {symbol}")
        
        # Get historical data
        market_df = self.get_historical_data(symbol)
        if market_df is None or len(market_df) < 30:
            return {"error": f"Insufficient historical data for {symbol}"}
        
        results = {}
        for model_type in model_types:
            try:
                evaluation = self.forecaster.evaluate_model(symbol, market_df, model_type)
                if evaluation:
                    results[model_type] = evaluation
                else:
                    results[model_type] = {"error": f"Evaluation failed for {model_type}"}
            except Exception as e:
                logger.error(f"Error evaluating {model_type} model: {e}")
                results[model_type] = {"error": str(e)}
        
        # Generate summary metrics
        summary = {}
        for model_type, result in results.items():
            if "metrics" in result:
                metrics = result["metrics"]
                
                # Extract key metrics for summary
                model_summary = {}
                
                # Handle different metric structures
                if isinstance(metrics, dict):
                    # Combine metrics from different components
                    mae_metrics = [v for k, v in metrics.items() if 'mae' in k.lower()]
                    rmse_metrics = [v for k, v in metrics.items() if 'rmse' in k.lower()]
                    mape_metrics = [v for k, v in metrics.items() if 'mape' in k.lower()]
                    
                    if mae_metrics:
                        model_summary['mae'] = sum(mae_metrics) / len(mae_metrics)
                    if rmse_metrics:
                        model_summary['rmse'] = sum(rmse_metrics) / len(rmse_metrics)
                    if mape_metrics:
                        model_summary['mape'] = sum(mape_metrics) / len(mape_metrics)
                
                summary[model_type] = model_summary
        
        # Calculate best model by mean absolute percentage error (MAPE)
        best_model = None
        best_mape = float('inf')
        
        for model_type, metrics in summary.items():
            if 'mape' in metrics and metrics['mape'] < best_mape:
                best_mape = metrics['mape']
                best_model = model_type
        
        return {
            "symbol": symbol,
            "evaluation_date": datetime.now().isoformat(),
            "model_evaluations": results,
            "summary": summary,
            "best_model": best_model
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
            return self.get_historical_data(symbol, use_cache=False)
        
        # Define function to get sentiment data for a symbol
        def get_sentiment_data(symbol):
            return self.get_news_sentiment(symbol, use_cache=False)
        
        # Call auto_retrain in the forecaster
        results = self.forecaster.auto_retrain(
            symbols=symbols,
            market_data_func=get_market_data,
            sentiment_data_func=get_sentiment_data,
            force=force
        )
        
        # Clear forecast cache after retraining
        if self.cache_enabled:
            self.cache['forecasts'] = {}
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols_count": len(symbols),
            "successful_retrains": results.get("success", 0),
            "failed_retrains": results.get("failed", 0),
            "skipped_retrains": results.get("skipped", 0),
            "details": results.get("details", {})
        }
    
    def generate_market_report(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive market report for multiple symbols.
        
        Args:
            symbols: List of symbols to include (None for all available)
            
        Returns:
            Dict with market report data
        """
        if symbols is None:
            symbols = self.list_available_symbols()[:10]  # Limit to top 10 for performance
            
        if not symbols:
            logger.warning("No symbols available for market report")
            return {"error": "No symbols available"}
        
        logger.info(f"Generating market report for {len(symbols)} symbols")
        
        # Gather insights for each symbol
        symbol_insights = {}
        anomalies = []
        opportunities = []
        
        for symbol in symbols:
            try:
                # Get market data
                market_df = self.get_historical_data(symbol)
                if market_df is None or len(market_df) < 30:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                # Get sentiment data
                sentiment_df = self.get_news_sentiment(symbol)
                
                # Generate forecast
                forecast_df = self.forecaster.forecast(
                    symbol=symbol,
                    market_df=market_df,
                    horizon='ensemble',
                    days_ahead=7
                )
                
                if forecast_df is None:
                    logger.warning(f"Failed to generate forecast for {symbol}")
                    continue
                
                # Get market insights
                insights = self.forecaster.generate_market_insights(
                    symbol=symbol,
                    market_df=market_df,
                    forecast_df=forecast_df,
                    sentiment_df=sentiment_df
                )
                
                # Store insights
                symbol_insights[symbol] = insights
                
                # Check for anomalies
                anomalies_df = self.forecaster.detect_latest_anomalies(symbol, market_df)
                if anomalies_df is not None and len(anomalies_df) > 0:
                    # Add symbol to anomalies
                    for idx, row in anomalies_df.iterrows():
                        anomaly = row.to_dict()
                        anomaly['symbol'] = symbol
                        anomaly['date'] = idx.isoformat() if isinstance(idx, datetime) else str(idx)
                        anomalies.append(anomaly)
                
                # Check for opportunities
                final_forecast = forecast_df.iloc[-1]
                change_pct = final_forecast['change_pct']
                
                if abs(change_pct) >= 5.0:  # At least 5% change
                    opportunity = {
                        'symbol': symbol,
                        'direction': 'bullish' if change_pct > 0 else 'bearish',
                        'change_pct': change_pct,
                        'current_price': market_df['price'].iloc[-1],
                        'forecast_price': final_forecast['forecast_price'],
                        'confidence': 'medium'
                    }
                    opportunities.append(opportunity)
                
            except Exception as e:
                logger.error(f"Error processing {symbol} for market report: {e}")
        
        # Calculate market overview
        overview = {
            'total_symbols': len(symbol_insights),
            'bullish_count': len([s for s, i in symbol_insights.items() if i.get('trend') == 'bullish']),
            'bearish_count': len([s for s, i in symbol_insights.items() if i.get('trend') == 'bearish']),
            'anomalies_count': len(anomalies),
            'opportunities_count': len(opportunities)
        }
        
        # Calculate average sentiment
        sentiment_scores = []
        for symbol, insight in symbol_insights.items():
            if 'sentiment' in insight and 'avg_sentiment' in insight['sentiment']:
                sentiment_scores.append(insight['sentiment']['avg_sentiment'])
        
        if sentiment_scores:
            overview['avg_sentiment'] = sum(sentiment_scores) / len(sentiment_scores)
            
            if overview['avg_sentiment'] > 0.6:
                overview['market_sentiment'] = 'strongly positive'
            elif overview['avg_sentiment'] > 0.55:
                overview['market_sentiment'] = 'positive'
            elif overview['avg_sentiment'] < 0.4:
                overview['market_sentiment'] = 'negative'
            elif overview['avg_sentiment'] < 0.45:
                overview['market_sentiment'] = 'somewhat negative'
            else:
                overview['market_sentiment'] = 'neutral'
        
        # Generate market summary
        market_summary = []
        
        # Add overview of trend
        if overview['bullish_count'] > overview['bearish_count']:
            market_summary.append(f"Overall market trend is bullish with {overview['bullish_count']} out of {overview['total_symbols']} symbols showing upward momentum.")
        else:
            market_summary.append(f"Overall market trend is bearish with {overview['bearish_count']} out of {overview['total_symbols']} symbols showing downward pressure.")
        
        # Add sentiment
        if 'market_sentiment' in overview:
            market_summary.append(f"Market sentiment is {overview['market_sentiment']}.")
        
        # Add anomalies
        if overview['anomalies_count'] > 0:
            market_summary.append(f"Detected {overview['anomalies_count']} market anomalies across {len(set(a['symbol'] for a in anomalies))} symbols, indicating potential market events.")
        
        # Add opportunities
        if opportunities:
            bullish_ops = [o for o in opportunities if o['direction'] == 'bullish']
            bearish_ops = [o for o in opportunities if o['direction'] == 'bearish']
            
            if bullish_ops:
                symbols_str = ", ".join([o['symbol'] for o in bullish_ops[:3]])
                if len(bullish_ops) > 3:
                    symbols_str += f", and {len(bullish_ops) - 3} more"
                market_summary.append(f"Identified {len(bullish_ops)} bullish opportunities, including {symbols_str}.")
            
            if bearish_ops:
                symbols_str = ", ".join([o['symbol'] for o in bearish_ops[:3]])
                if len(bearish_ops) > 3:
                    symbols_str += f", and {len(bearish_ops) - 3} more"
                market_summary.append(f"Identified {len(bearish_ops)} bearish opportunities, including {symbols_str}.")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overview": overview,
            "market_summary": " ".join(market_summary),
            "insights": symbol_insights,
            "anomalies": anomalies,
            "opportunities": opportunities
        }
    
    def clear_cache(self):
        """Clear all cache data"""
        self.cache = {'market': {}, 'sentiment': {}, 'forecasts': {}}
        logger.info("Cache cleared")
    
    def close(self):
        """Close any open connections."""
        self.time_series_manager.close()
        logger.info("Connections closed")

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