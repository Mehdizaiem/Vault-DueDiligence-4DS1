import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import Weaviate client and types
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
import weaviate.classes.config as wc

def fix_market_metrics():
    """Fix the CryptoNewsSentiment collection to handle embeddings properly"""
    client = get_weaviate_client()
    
    try:
        # First try to delete the existing collections
        try:
            client.collections.delete("CryptoNewsSentiment")
            logger.info("Deleted existing CryptoNewsSentiment collection")
        except Exception as e:
            logger.info(f"No existing CryptoNewsSentiment collection to delete: {e}")
        
        try:
            client.collections.delete("CryptoForecasts")
            logger.info("Deleted existing CryptoForecasts collection")
        except Exception as e:
            logger.info(f"No existing CryptoForecasts collection to delete: {e}")
        
        # Create the CryptoNewsSentiment collection with correct property configuration
        logger.info("Creating CryptoNewsSentiment collection")
        client.collections.create(
            name="CryptoNewsSentiment",
            description="Collection for crypto news articles with sentiment analysis",
            properties=[
                wc.Property(
                    name="source",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="title",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="url",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="content",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="date",
                    data_type=wc.DataType.DATE
                ),
                wc.Property(
                    name="authors",
                    data_type=wc.DataType.TEXT_ARRAY
                ),
                wc.Property(
                    name="sentiment_label",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="sentiment_score",
                    data_type=wc.DataType.NUMBER
                ),
                wc.Property(
                    name="analyzed_at",
                    data_type=wc.DataType.DATE
                ),
                wc.Property(
                    name="image_url",
                    data_type=wc.DataType.TEXT
                )
            ],
            vectorizer_config=None
        )
        
        # Create the CryptoForecasts collection with corrected data type
        logger.info("Creating CryptoForecasts collection")
        client.collections.create(
            name="CryptoForecasts",
            description="Collection for cryptocurrency price forecasts",
            properties=[
                wc.Property(
                    name="symbol",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="forecast_date",
                    data_type=wc.DataType.DATE
                ),
                wc.Property(
                    name="prediction_dates",
                    data_type=wc.DataType.DATE_ARRAY
                ),
                wc.Property(
                    name="predicted_prices",
                    data_type=wc.DataType.NUMBER_ARRAY
                ),
                wc.Property(
                    name="change_percentages",
                    data_type=wc.DataType.NUMBER_ARRAY
                ),
                wc.Property(
                    name="model_version",
                    data_type=wc.DataType.TEXT
                ),
                wc.Property(
                    name="model_mae",
                    data_type=wc.DataType.NUMBER
                ),
                wc.Property(
                    name="model_rmse",
                    data_type=wc.DataType.NUMBER
                ),
                wc.Property(
                    name="sentiment_incorporated",
                    data_type=wc.DataType.NUMBER  # Changed from BOOLEAN to NUMBER
                ),
                wc.Property(
                    name="avg_sentiment",
                    data_type=wc.DataType.NUMBER
                )
            ],
            vectorizer_config=None
        )
        
        logger.info("Successfully recreated CryptoNewsSentiment and CryptoForecasts collections")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        client.close()

if __name__ == "__main__":
    if fix_market_metrics():
        print("✅ Successfully fixed sentiment and forecast collections")
    else:
        print("❌ Failed to fix collections")