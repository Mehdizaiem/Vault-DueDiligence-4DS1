#!/usr/bin/env python
"""
Core functionality for storing Chronos cryptocurrency forecasts in Weaviate
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import Weaviate client functions
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from weaviate.classes.config import DataType, Configure

def create_forecast_schema(client):
    """
    Create the Forecast collection if it doesn't exist.
    
    Args:
        client: Weaviate client
        
    Returns:
        The created or existing collection
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("Forecast")
        logger.info("Forecast collection already exists")
        return collection
    except Exception:
        logger.info("Creating Forecast collection")
        
        try:
            # Create the collection without vectorizer since we don't need embeddings
            collection = client.collections.create(
                name="Forecast",
                description="Collection for cryptocurrency price forecasts",
                vectorizer_config=Configure.Vectorizer.none(),  # No vectorizer needed
                properties=[
                    # Basic forecast metadata
                    {
                        "name": "symbol",
                        "data_type": DataType.TEXT,
                        "description": "Cryptocurrency symbol (e.g., BTCUSDT)"
                    },
                    {
                        "name": "forecast_timestamp",
                        "data_type": DataType.DATE,
                        "description": "When the forecast was generated"
                    },
                    {
                        "name": "model_name",
                        "data_type": DataType.TEXT,
                        "description": "Name of the model used for forecasting"
                    },
                    {
                        "name": "model_type",
                        "data_type": DataType.TEXT,
                        "description": "Type of forecasting model (e.g., chronos, ensemble, lstm)"
                    },
                    {
                        "name": "days_ahead",
                        "data_type": DataType.INT,
                        "description": "Number of days in the forecast horizon"
                    },
                    
                    # Current market state
                    {
                        "name": "current_price",
                        "data_type": DataType.NUMBER,
                        "description": "Current price at time of forecast"
                    },
                    
                    # Forecast data
                    {
                        "name": "forecast_dates",
                        "data_type": DataType.DATE_ARRAY,
                        "description": "Array of forecast dates"
                    },
                    {
                        "name": "forecast_values",
                        "data_type": DataType.NUMBER_ARRAY,
                        "description": "Array of forecasted price values (typically median forecast)"
                    },
                    {
                        "name": "lower_bounds",
                        "data_type": DataType.NUMBER_ARRAY,
                        "description": "Array of lower confidence interval bounds"
                    },
                    {
                        "name": "upper_bounds",
                        "data_type": DataType.NUMBER_ARRAY,
                        "description": "Array of upper confidence interval bounds"
                    },
                    
                    # Forecast statistics
                    {
                        "name": "final_forecast",
                        "data_type": DataType.NUMBER,
                        "description": "Final forecasted price value"
                    },
                    {
                        "name": "change_pct",
                        "data_type": DataType.NUMBER,
                        "description": "Forecasted percentage change from current price"
                    },
                    {
                        "name": "trend",
                        "data_type": DataType.TEXT,
                        "description": "Overall trend description (e.g., bullish, bearish, neutral)"
                    },
                    {
                        "name": "probability_increase",
                        "data_type": DataType.NUMBER,
                        "description": "Probability of price increase (0-100)"
                    },
                    {
                        "name": "average_uncertainty",
                        "data_type": DataType.NUMBER,
                        "description": "Average uncertainty in the forecast (%)"
                    },
                    {
                        "name": "insight",
                        "data_type": DataType.TEXT,
                        "description": "Text description of forecast insights"
                    },
                    
                    # Additional metadata
                    {
                        "name": "plot_path",
                        "data_type": DataType.TEXT,
                        "description": "Path to the forecast plot image"
                    }
                ]
            )
            
            logger.info("Successfully created Forecast collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create Forecast collection: {e}")
            raise

def store_chronos_forecast(forecast_results: Dict, market_insights: Dict, 
                         symbol: str, model_name: str, days_ahead: int, 
                         plot_path: Optional[str] = None) -> bool:
    """
    Store Chronos forecast results in the Forecast collection.
    
    Args:
        forecast_results: Results from Chronos forecaster.forecast() method
        market_insights: Results from Chronos forecaster.generate_market_insights() method
        symbol: Cryptocurrency symbol
        model_name: Name of the Chronos model used
        days_ahead: Number of days in forecast horizon
        plot_path: Path to the saved forecast plot
        
    Returns:
        bool: Success status
    """
    client = get_weaviate_client()
    
    try:
        # Ensure the Forecast schema exists
        collection = create_forecast_schema(client)
        
        # Format dates to ISO format strings for Weaviate
        forecast_dates = [date.isoformat() for date in forecast_results.get('dates', [])]
        
        # Extract forecast values
        # Get median forecast from quantiles
        quantiles = forecast_results.get('quantiles', [None])[0]
        quantile_levels = forecast_results.get('quantile_levels', [0.1, 0.5, 0.9])
        
        if quantiles is not None:
            # Find index of median or closest to median
            if 0.5 in quantile_levels:
                median_idx = quantile_levels.index(0.5)
            else:
                median_idx = len(quantile_levels) // 2
                
            # Get lower, median, and upper bounds
            low_idx = 0  # Lowest quantile
            high_idx = len(quantile_levels) - 1  # Highest quantile
            
            # Extract forecast values arrays
            forecast_values = quantiles[:, median_idx].tolist()
            lower_bounds = quantiles[:, low_idx].tolist()
            upper_bounds = quantiles[:, high_idx].tolist()
        else:
            # Fallback to mean if quantiles not available
            mean = forecast_results.get('mean', [None])[0]
            forecast_values = mean.tolist() if mean is not None else []
            lower_bounds = []
            upper_bounds = []
        
        # Create properties dictionary
        properties = {
            "symbol": symbol,
            "forecast_timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "model_type": "chronos",
            "days_ahead": days_ahead,
            
            # Current market state
            "current_price": market_insights.get('current_price', 0.0),
            
            # Forecast data
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            
            # Forecast statistics
            "final_forecast": market_insights.get('final_forecast', 0.0),
            "change_pct": market_insights.get('change_pct', 0.0),
            "trend": market_insights.get('trend', "unknown"),
            "probability_increase": market_insights.get('probability_increase', 0.0),
            "average_uncertainty": market_insights.get('average_uncertainty', 0.0),
            "insight": market_insights.get('insight', ""),
        }
        
        # Add plot path if provided
        if plot_path:
            properties["plot_path"] = plot_path
        
        # Store in Weaviate
        collection.data.insert(properties=properties)
        
        logger.info(f"Successfully stored forecast for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        if client:
            client.close()

def retrieve_latest_forecast(symbol: str, limit: int = 1) -> List[Dict[str, Any]]:
    """
    Retrieve the latest forecast(s) for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        limit: Maximum number of forecasts to retrieve
        
    Returns:
        List of forecast objects
    """
    client = get_weaviate_client()
    
    try:
        collection = client.collections.get("Forecast")
        
        # Query for forecasts for this symbol, ordered by timestamp
        from weaviate.classes.query import Filter, Sort
        
        response = collection.query.fetch_objects(
            filters=Filter.by_property("symbol").equal(symbol),
            sort=Sort.by_property("forecast_timestamp", ascending=False),
            limit=limit
        )
        
        # Format results
        results = []
        for obj in response.objects:
            result = {
                "id": str(obj.uuid),
                **obj.properties
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving forecasts: {e}")
        return []
    finally:
        if client:
            client.close()

def compare_forecasts(symbol: str, date_range: Optional[int] = 30) -> Dict[str, Any]:
    """
    Compare forecasts over time for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        date_range: Number of days to look back for forecasts
        
    Returns:
        Dict with forecast comparison analysis
    """
    client = get_weaviate_client()
    
    try:
        collection = client.collections.get("Forecast")
        
        # Get forecasts within date range
        from weaviate.classes.query import Filter, Sort
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range)
        start_date_str = start_date.isoformat()
        
        # Query for forecasts
        response = collection.query.fetch_objects(
            filters=(Filter.by_property("symbol").equal(symbol) & 
                    Filter.by_property("forecast_timestamp").greater_than(start_date_str)),
            return_properties=[
                "forecast_timestamp", "final_forecast", "current_price", 
                "change_pct", "trend", "model_type"
            ],
            sort=Sort.by_property("forecast_timestamp", ascending=True)
        )
        
        if not response.objects:
            return {
                "symbol": symbol,
                "forecasts_count": 0,
                "message": f"No forecasts found for {symbol} in the last {date_range} days"
            }
        
        # Extract forecast trends over time
        forecasts = []
        for obj in response.objects:
            forecasts.append({
                "timestamp": obj.properties.get("forecast_timestamp"),
                "current_price": obj.properties.get("current_price"),
                "final_forecast": obj.properties.get("final_forecast"),
                "change_pct": obj.properties.get("change_pct"),
                "trend": obj.properties.get("trend"),
                "model_type": obj.properties.get("model_type")
            })
        
        # Calculate trend consistency
        trend_counts = {}
        for forecast in forecasts:
            trend = forecast.get("trend")
            if trend:
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
        
        # Find most common trend
        most_common_trend = max(trend_counts.items(), key=lambda x: x[1])[0] if trend_counts else "unknown"
        trend_consistency = (trend_counts.get(most_common_trend, 0) / len(forecasts)) * 100 if forecasts else 0
        
        # Calculate average forecasted change
        avg_change = sum(f.get("change_pct", 0) for f in forecasts) / len(forecasts) if forecasts else 0
        
        # Check forecast direction changes
        direction_changes = 0
        prev_direction = None
        for forecast in forecasts:
            current_direction = "up" if forecast.get("change_pct", 0) > 0 else "down"
            if prev_direction is not None and current_direction != prev_direction:
                direction_changes += 1
            prev_direction = current_direction
        
        return {
            "symbol": symbol,
            "forecasts_count": len(forecasts),
            "date_range": date_range,
            "trend_distribution": trend_counts,
            "most_common_trend": most_common_trend,
            "trend_consistency": trend_consistency,
            "avg_forecasted_change": avg_change,
            "direction_changes": direction_changes,
            "forecasts": forecasts
        }
        
    except Exception as e:
        logger.error(f"Error comparing forecasts: {e}")
        return {"error": str(e)}
    finally:
        if client:
            client.close()

def run_and_store_chronos_forecast(symbol: str, days_ahead: int = 14, 
                                model_name: str = "amazon/chronos-t5-small",
                                use_gpu: bool = True) -> Dict[str, Any]:
    """
    Run a Chronos forecast and store the results in Weaviate.
    
    Args:
        symbol: Cryptocurrency symbol
        days_ahead: Number of days to forecast
        model_name: Name of the Chronos model to use
        use_gpu: Whether to use GPU for inference
        
    Returns:
        Dict with forecast results and storage status
    """
    try:
        # Import the ChronosForecaster class
        from models.chronos.chronos_crypto_forecaster import ChronosForecaster, prepare_crypto_data
        
        # Load cryptocurrency data
        data = prepare_crypto_data(symbol)
        
        if data is None or len(data) == 0:
            return {"error": f"Failed to load data for {symbol}"}
        
        # Initialize forecaster
        forecaster = ChronosForecaster(
            model_name=model_name,
            use_gpu=use_gpu
        )
        
        # Generate forecast
        forecast_results = forecaster.forecast(
            data,
            prediction_length=days_ahead,
            num_samples=100,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Plot forecast
        plot_path = forecaster.plot_forecast(
            data,
            forecast_results,
            symbol
        )
        
        # Generate insights
        insights = forecaster.generate_market_insights(
            data,
            forecast_results,
            symbol
        )
        
        # Store in Weaviate
        storage_success = store_chronos_forecast(
            forecast_results=forecast_results,
            market_insights=insights,
            symbol=symbol,
            model_name=model_name,
            days_ahead=days_ahead,
            plot_path=plot_path
        )
        
        # Return combined results
        return {
            "symbol": symbol,
            "forecast_timestamp": datetime.now().isoformat(),
            "insights": insights,
            "storage_success": storage_success,
            "plot_path": plot_path
        }
        
    except Exception as e:
        logger.error(f"Error in run_and_store_chronos_forecast: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}