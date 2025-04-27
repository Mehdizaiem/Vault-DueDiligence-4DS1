import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import only Weaviate client
from Sample_Data.vector_store.weaviate_client import get_weaviate_client

def fetch_latest_btc_sentiments():
    """Fetch the latest 10 BTC sentiment articles"""
    logger.info("Fetching latest 10 BTC sentiment articles")
    
    try:
        # Get Weaviate client
        client = get_weaviate_client()
        logger.info("Connected to Weaviate")
        
        # Get collection
        collection = client.collections.get("CryptoNewsSentiment")
        
        # Query latest 10 articles mentioning BTC
        response = collection.query.fetch_objects(
            limit=10,
            return_properties=["title", "date", "sentiment_label", "sentiment_score", "content"]
        )
        
        if not response.objects:
            logger.warning("No articles found")
            return
        
        # Process and display articles
        print("\n=== Latest 10 BTC Sentiment Articles ===")
        for i, article in enumerate(response.objects, 1):
            props = article.properties
            title = props.get("title", "No title")
            date = pd.to_datetime(props.get("date")).strftime("%Y-%m-%d")
            sentiment = props.get("sentiment_label", "NEUTRAL")
            score = float(props.get("sentiment_score", 0.5))
            
            print(f"\n{i}. [{date}] {title}")
            print(f"   Sentiment: {sentiment} (Score: {score:.3f})")
            
            # Show first 100 characters of content
            content = props.get("content", "")
            if content:
                print(f"   Preview: {content[:100]}...")
        
        print("\n=== End of Results ===")
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("Closed Weaviate connection")

if __name__ == "__main__":
    fetch_latest_btc_sentiments() 