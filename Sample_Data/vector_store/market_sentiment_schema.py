import logging
import os
import sys
from weaviate.exceptions import WeaviateBaseError
from datetime import datetime, timedelta
import random
import numpy as np
from weaviate.classes.config import DataType

# Configure paths
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sentiment_schema(client):
    """
    Create the CryptoNewsSentiment collection if it doesn't exist.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoNewsSentiment")
        logger.info("CryptoNewsSentiment collection already exists")
        return collection
    except WeaviateBaseError:
        logger.info("Creating CryptoNewsSentiment collection")
        
        try:
            # Create the collection with null vectorizer
            collection = client.collections.create(
                name="CryptoNewsSentiment",
                description="Collection for crypto news articles with sentiment analysis",
                vectorizer_config=None,  # Use None instead of text2vec-none
                properties=[
                    {
                        "name": "source",
                        "data_type": DataType.TEXT,
                        "description": "Source of the news article"
                    },
                    {
                        "name": "title",
                        "data_type": DataType.TEXT,
                        "description": "Title of the news article"
                    },
                    {
                        "name": "url",
                        "data_type": DataType.TEXT,
                        "description": "URL of the news article"
                    },
                    {
                        "name": "content",
                        "data_type": DataType.TEXT,
                        "description": "Content of the news article"
                    },
                    {
                        "name": "date",
                        "data_type": DataType.DATE,
                        "description": "Publication date of the article"
                    },
                    {
                        "name": "authors",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Authors of the article"
                    },
                    {
                        "name": "sentiment_label",
                        "data_type": DataType.TEXT,
                        "description": "Sentiment label (POSITIVE, NEUTRAL, NEGATIVE)"
                    },
                    {
                        "name": "sentiment_score",
                        "data_type": DataType.NUMBER,
                        "description": "Sentiment score (0-1)"
                    },
                    {
                        "name": "analyzed_at",
                        "data_type": DataType.DATE,
                        "description": "Timestamp when sentiment analysis was performed"
                    },
                    {
                        "name": "image_url",
                        "data_type": DataType.TEXT,
                        "description": "URL of the article's featured image"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoNewsSentiment collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoNewsSentiment collection: {e}")
            raise

def create_forecast_schema(client):
    """
    Create the CryptoForecasts collection if it doesn't exist.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoForecasts")
        logger.info("CryptoForecasts collection already exists")
        return collection
    except WeaviateBaseError:
        logger.info("Creating CryptoForecasts collection")
        
        try:
            # Create the collection
            collection = client.collections.create(
                name="CryptoForecasts",
                description="Collection for cryptocurrency price forecasts",
                vectorizer_config=None,  # Use None
                properties=[
                    {
                        "name": "symbol",
                        "data_type": DataType.TEXT,
                        "description": "Cryptocurrency symbol (e.g., BTCUSDT)"
                    },
                    {
                        "name": "forecast_date",
                        "data_type": DataType.DATE,
                        "description": "Date when the forecast was generated"
                    },
                    {
                        "name": "prediction_dates",
                        "data_type": DataType.DATE_ARRAY,
                        "description": "Array of dates for which prices are predicted"
                    },
                    {
                        "name": "predicted_prices",
                        "data_type": DataType.NUMBER_ARRAY,
                        "description": "Array of predicted prices"
                    },
                    {
                        "name": "change_percentages",
                        "data_type": DataType.NUMBER_ARRAY,
                        "description": "Array of percentage changes"
                    },
                    {
                        "name": "model_version",
                        "data_type": DataType.TEXT,
                        "description": "Version of the model used for prediction"
                    },
                    {
                        "name": "model_mae",
                        "data_type": DataType.NUMBER,
                        "description": "Mean Absolute Error of the model"
                    },
                    {
                        "name": "model_rmse",
                        "data_type": DataType.NUMBER,
                        "description": "Root Mean Squared Error of the model"
                    },
                    {
                        "name": "sentiment_incorporated",
                        "data_type": DataType.BOOLEAN,
                        "description": "Whether sentiment data was incorporated"
                    },
                    {
                        "name": "avg_sentiment",
                        "data_type": DataType.NUMBER,
                        "description": "Average sentiment score incorporated"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoForecasts collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoForecasts collection: {e}")
            raise

def create_market_metrics_schema(client):
    """
    Create the MarketMetrics collection if it doesn't exist.
    """
    try:
        # First, try to delete the collection if it exists
        try:
            client.collections.delete("MarketMetrics")
            logger.info("Deleted existing MarketMetrics collection")
        except Exception:
            pass
            
        logger.info("Creating MarketMetrics collection")
        
        # Create the collection
        collection = client.collections.create(
            name="MarketMetrics",
            description="Collection for cryptocurrency market metrics",
            vectorizer_config=None,  # Use None instead of text2vec-none
            properties=[
                {
                    "name": "symbol",
                    "data_type": DataType.TEXT,
                    "description": "Cryptocurrency symbol (e.g., BTCUSDT)"
                },
                {
                    "name": "source",
                    "data_type": DataType.TEXT,
                    "description": "Data source (e.g., binance, coingecko)"
                },
                {
                    "name": "price",
                    "data_type": DataType.NUMBER,
                    "description": "Current price in USD"
                },
                {
                    "name": "market_cap",
                    "data_type": DataType.NUMBER,
                    "description": "Market capitalization"
                },
                {
                    "name": "volume_24h",
                    "data_type": DataType.NUMBER,
                    "description": "24h trading volume"
                },
                {
                    "name": "price_change_24h",
                    "data_type": DataType.NUMBER,
                    "description": "24h price change percentage"
                },
                {
                    "name": "timestamp",
                    "data_type": DataType.DATE,
                    "description": "Data timestamp"
                }
            ]
        )
        
        logger.info("Successfully created MarketMetrics collection")
        
        # Add sample data
        add_sample_market_data(client, collection)
        
        return collection
    except Exception as e:
        logger.error(f"Failed to create MarketMetrics collection: {e}")
        raise

def add_sample_market_data(client, collection=None):
    """
    Add sample market data for testing.
    """
    try:
        if collection is None:
            collection = client.collections.get("MarketMetrics")
            
        # Generate sample data for major cryptocurrencies
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        
        # Base prices for different cryptocurrencies
        base_prices = {
            "BTCUSDT": 85000.0,
            "ETHUSDT": 3000.0,
            "SOLUSDT": 150.0,
            "BNBUSDT": 600.0,
            "ADAUSDT": 0.5
        }
        
        # Generate 30 days of sample data
        sample_data = []
        
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100.0)
            
            for i in range(30):
                # Add some randomness to price
                random_factor = 1 + (random.random() - 0.5) * 0.1  # ±5% variation
                price = base_price * random_factor
                
                # Adjust base price for trend (slightly upward)
                base_price *= (1 + random.random() * 0.01)  # Up to 1% daily increase
                
                # Generate other metrics
                market_cap = price * (10**9 if symbol in ["BTCUSDT", "ETHUSDT"] else 10**8)
                volume_24h = market_cap * random.uniform(0.05, 0.15)  # 5-15% of market cap
                price_change_24h = (random.random() - 0.5) * 10  # -5% to +5% daily change
                
                # Create date (days in the past) - ensure RFC3339 format
                date = datetime.now() - timedelta(days=29-i)
                formatted_date = date.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Add to sample data
                sample_data.append({
                    "symbol": symbol,
                    "source": "sample_data",  # IMPORTANT: this is the source property
                    "price": price,
                    "market_cap": market_cap,
                    "volume_24h": volume_24h,
                    "price_change_24h": price_change_24h,
                    "timestamp": formatted_date
                })
        
        # Insert data in batches
        batch_size = 50
        for i in range(0, len(sample_data), batch_size):
            batch = sample_data[i:i+batch_size]
            collection.data.insert_many(batch)
            
        logger.info(f"Added {len(sample_data)} sample market data points")
        
    except Exception as e:
        logger.error(f"Error adding sample market data: {e}")

def create_due_diligence_schema(client):
    """
    Create the CryptoDueDiligenceDocuments collection if it doesn't exist.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoDueDiligenceDocuments")
        logger.info("CryptoDueDiligenceDocuments collection already exists")
        return collection
    except WeaviateBaseError:
        logger.info("Creating CryptoDueDiligenceDocuments collection")
        
        try:
            # Create the collection
            collection = client.collections.create(
                name="CryptoDueDiligenceDocuments",
                description="Collection for all documents related to crypto fund due diligence",
                vectorizer_config=None,  # Use None
                properties=[
                    # Basic document properties
                    {
                        "name": "content",
                        "data_type": DataType.TEXT,
                        "description": "Extracted text from the document"
                    },
                    {
                        "name": "source",
                        "data_type": DataType.TEXT,
                        "description": "Original file name or source"
                    },
                    {
                        "name": "document_type",
                        "data_type": DataType.TEXT,
                        "description": "Type of document"
                    },
                    {
                        "name": "title",
                        "data_type": DataType.TEXT,
                        "description": "Title or name of the document"
                    },
                    {
                        "name": "date",
                        "data_type": DataType.DATE,
                        "description": "Creation or publication date"
                    },
                    # Additional metadata fields
                    {
                        "name": "author_issuer",
                        "data_type": DataType.TEXT,
                        "description": "Author, issuer, or organization responsible"
                    },
                    {
                        "name": "category",
                        "data_type": DataType.TEXT,
                        "description": "Category (e.g., technical, legal, compliance, business)"
                    },
                    {
                        "name": "risk_score",
                        "data_type": DataType.NUMBER,
                        "description": "Risk assessment score (0-100)"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoDueDiligenceDocuments collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoDueDiligenceDocuments collection: {e}")
            raise

def store_forecast(client, symbol, forecast_df, model_metrics=None, sentiment_incorporated=True, avg_sentiment=0.5):
    """
    Store a forecast in the CryptoForecasts collection.
    """
    try:
        # Ensure the schema exists
        collection = create_forecast_schema(client)
        
        # Prepare forecast data
        from datetime import datetime
        
        # Format forecast date in RFC3339
        forecast_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Convert dates to ISO format (RFC3339)
        prediction_dates = []
        for d in forecast_df['date']:
            if hasattr(d, 'strftime'):
                prediction_dates.append(d.strftime("%Y-%m-%dT%H:%M:%SZ"))
            else:
                # If it's already a string, ensure it's in the correct format
                try:
                    parsed_date = datetime.fromisoformat(str(d).replace('Z', '+00:00'))
                    prediction_dates.append(parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ"))
                except:
                    # Fallback to current date
                    prediction_dates.append(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        properties = {
            "symbol": symbol,
            "forecast_date": forecast_date,
            "prediction_dates": prediction_dates,
            "predicted_prices": forecast_df['predicted_price'].tolist(),
            "change_percentages": forecast_df['change_pct'].tolist(),
            "model_version": "RandomForest_v1",
            "sentiment_incorporated": sentiment_incorporated,
            "avg_sentiment": float(avg_sentiment)
        }
        
        # Add model metrics if provided
        if model_metrics:
            if 'mae' in model_metrics:
                properties["model_mae"] = float(model_metrics['mae']) if model_metrics['mae'] is not None else 0.0
            if 'rmse' in model_metrics:
                properties["model_rmse"] = float(model_metrics['rmse']) if model_metrics['rmse'] is not None else 0.0
        
        # Store the forecast
        collection.data.insert(properties=properties)
        
        logger.info(f"Successfully stored forecast for {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"Error storing forecast: {e}")
        return False

# Example usage
if __name__ == "__main__":
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    
    client = get_weaviate_client()
    try:
        # Create schemas
        create_sentiment_schema(client)
        create_forecast_schema(client)
        create_market_metrics_schema(client)
        create_due_diligence_schema(client)
        
        print("Schemas created successfully")
    finally:
        client.close()