# File path: Sample_Data/vector_store/schema_manager.py
import logging
import os
import sys
from weaviate.classes.config import DataType, Configure
from datetime import datetime, timedelta
import random


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_crypto_due_diligence_schema(client):
    """
    Create the CryptoDueDiligenceDocuments collection with all-MPNet embeddings.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoDueDiligenceDocuments")
        logger.info("CryptoDueDiligenceDocuments collection already exists")
        return collection
    except Exception:
        logger.info("Creating CryptoDueDiligenceDocuments collection")
        
        try:
            # Create the collection with all-MPNet vectorizer
            collection = client.collections.create(
                name="CryptoDueDiligenceDocuments",
                description="Collection for all documents related to crypto fund due diligence",
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),  # Use all-MPNet as default
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=Configure.VectorIndex.Distance.cosine
                ),
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
                    },
                    {
                        "name": "keywords",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Key terms extracted from the document"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoDueDiligenceDocuments collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoDueDiligenceDocuments collection: {e}")
            raise

def create_crypto_news_sentiment_schema(client):
    """
    Create the CryptoNewsSentiment collection with FinBERT embeddings.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoNewsSentiment")
        logger.info("CryptoNewsSentiment collection already exists")
        return collection
    except Exception:
        logger.info("Creating CryptoNewsSentiment collection")
        
        try:
            # Create the collection with custom vectorizer (we'll provide vectors from FinBERT)
            collection = client.collections.create(
                name="CryptoNewsSentiment",
                description="Collection for crypto news articles with sentiment analysis",
                vectorizer_config=Configure.Vectorizer.none(),  # We'll provide FinBERT vectors directly
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=Configure.VectorIndex.Distance.cosine
                ),
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
                    },
                    {
                        "name": "related_assets",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Cryptocurrency assets mentioned in the article"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoNewsSentiment collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoNewsSentiment collection: {e}")
            raise

def create_market_metrics_schema(client):
    """
    Create the MarketMetrics collection without embeddings.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("MarketMetrics")
        logger.info("MarketMetrics collection already exists")
        return collection
    except Exception:
        logger.info("Creating MarketMetrics collection")
        
        try:
            # Create the collection without vectorizer
            collection = client.collections.create(
                name="MarketMetrics",
                description="Collection for cryptocurrency market metrics",
                vectorizer_config=Configure.Vectorizer.none(),  # No vectorizer needed
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
            return collection
        except Exception as e:
            logger.error(f"Failed to create MarketMetrics collection: {e}")
            raise

def create_crypto_time_series_schema(client):
    """
    Create the CryptoTimeSeries collection without embeddings.
    Fixed to use correct data types.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("CryptoTimeSeries")
        logger.info("CryptoTimeSeries collection already exists")
        return collection
    except Exception:
        logger.info("Creating CryptoTimeSeries collection")
        
        try:
            # Create the collection without vectorizer
            collection = client.collections.create(
                name="CryptoTimeSeries",
                description="Historical price data from various exchanges",
                vectorizer_config=Configure.Vectorizer.none(),  # No vectorizer needed
                properties=[
                    {
                        "name": "symbol",
                        "dataType": ["text"],
                        "description": "Cryptocurrency symbol (e.g., BTCUSDT)"
                    },
                    {
                        "name": "exchange",
                        "dataType": ["text"],
                        "description": "Exchange (e.g., Binance, Coinbase)"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Data timestamp"
                    },
                    {
                        "name": "open",
                        "dataType": ["number"],  # Using number type for price values
                        "description": "Opening price"
                    },
                    {
                        "name": "high",
                        "dataType": ["number"],  # Using number type for price values
                        "description": "Highest price"
                    },
                    {
                        "name": "low",
                        "dataType": ["number"],  # Using number type for price values
                        "description": "Lowest price"
                    },
                    {
                        "name": "close",
                        "dataType": ["number"],  # Using number type for price values
                        "description": "Closing price"
                    },
                    {
                        "name": "volume",
                        "dataType": ["number"],  # Using number type for volume (FIXED)
                        "description": "Trading volume"
                    },
                    {
                        "name": "interval",
                        "dataType": ["text"],
                        "description": "Time interval (e.g., 1d, 1h, 15m)"
                    }
                ]
            )
            
            logger.info("Successfully created CryptoTimeSeries collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create CryptoTimeSeries collection: {str(e)}")
            raise

def create_onchain_analytics_schema(client):
    """
    Create the OnChainAnalytics collection without embeddings.
    """
    try:
        # Check if collection already exists
        collection = client.collections.get("OnChainAnalytics")
        logger.info("OnChainAnalytics collection already exists")
        return collection
    except Exception:
        logger.info("Creating OnChainAnalytics collection")
        
        try:
            # Create the collection without vectorizer
            collection = client.collections.create(
                name="OnChainAnalytics",
                description="Blockchain wallet analytics and transaction data",
                vectorizer_config=Configure.Vectorizer.none(),  # No vectorizer needed
                properties=[
                    {
                        "name": "address",
                        "data_type": DataType.TEXT,
                        "description": "Wallet or contract address"
                    },
                    {
                        "name": "blockchain",
                        "data_type": DataType.TEXT,
                        "description": "Blockchain network (ethereum, binance, etc.)"
                    },
                    {
                        "name": "entity_type",
                        "data_type": DataType.TEXT,
                        "description": "Type of entity (wallet, contract, token)"
                    },
                    {
                        "name": "transaction_count",
                        "data_type": DataType.INT,
                        "description": "Total number of transactions"
                    },
                    {
                        "name": "token_transaction_count",
                        "data_type": DataType.INT,
                        "description": "Total number of token transactions"
                    },
                    {
                        "name": "total_received",
                        "data_type": DataType.NUMBER,
                        "description": "Total value received in native currency"
                    },
                    {
                        "name": "total_sent",
                        "data_type": DataType.NUMBER,
                        "description": "Total value sent in native currency"
                    },
                    {
                        "name": "balance",
                        "data_type": DataType.NUMBER,
                        "description": "Current balance in native currency"
                    },
                    {
                        "name": "first_activity",
                        "data_type": DataType.DATE,
                        "description": "Timestamp of first activity"
                    },
                    {
                        "name": "last_activity",
                        "data_type": DataType.DATE,
                        "description": "Timestamp of most recent activity"
                    },
                    {
                        "name": "active_days",
                        "data_type": DataType.INT,
                        "description": "Number of days between first and last activity"
                    },
                    {
                        "name": "unique_interactions",
                        "data_type": DataType.INT,
                        "description": "Number of unique addresses interacted with"
                    },
                    {
                        "name": "contract_interactions",
                        "data_type": DataType.INT,
                        "description": "Number of contract interactions"
                    },
                    {
                        "name": "tokens",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Token symbols held by this address"
                    },
                    {
                        "name": "risk_score",
                        "data_type": DataType.NUMBER,
                        "description": "Risk assessment score (0-100)"
                    },
                    {
                        "name": "risk_level",
                        "data_type": DataType.TEXT,
                        "description": "Risk level category"
                    },
                    {
                        "name": "risk_factors",
                        "data_type": DataType.TEXT_ARRAY,
                        "description": "Identified risk factors"
                    },
                    {
                        "name": "related_fund",
                        "data_type": DataType.TEXT,
                        "description": "Related crypto fund name if applicable"
                    },
                    {
                        "name": "analysis_timestamp",
                        "data_type": DataType.DATE,
                        "description": "When this analysis was performed"
                    }
                ]
            )
            
            logger.info("Successfully created OnChainAnalytics collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create OnChainAnalytics collection: {e}")
            raise

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
                    
                    # Image storage
                    {
                        "name": "plot_path",
                        "data_type": DataType.TEXT,
                        "description": "Path to the forecast plot image"
                    },
                    {
                        "name": "plot_image",
                        "data_type": DataType.BLOB,
                        "description": "Base64 encoded forecast plot image"
                    }
                ]
            )
            
            logger.info("Successfully created Forecast collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create Forecast collection: {e}")
            raise

def setup_all_schemas(client):
    """Setup all required schemas"""
    try:
        create_crypto_due_diligence_schema(client)
        create_crypto_news_sentiment_schema(client)
        create_market_metrics_schema(client)
        create_crypto_time_series_schema(client)
        create_onchain_analytics_schema(client)
        create_forecast_schema(client)  # Added forecast schema
        logger.info("✅ All collections created successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up schemas: {e}")
        return False

if __name__ == "__main__":
    # 🔧 Add the root of your project to the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(project_root)

    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    
    client = get_weaviate_client()
    try:
        success = setup_all_schemas(client)
        if success:
            print("All schemas set up successfully!")
        else:
            print("Failed to set up schemas")
    finally:
        client.close()