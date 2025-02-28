import os
import sys
import logging
from datetime import datetime
import pandas as pd
from typing import Optional, Dict, List, Union, Any
from weaviate.exceptions import WeaviateBaseError
import json

# Configure paths
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Import project modules
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from Sample_Data.vector_store.market_sentiment_schema import create_sentiment_schema
from Sample_Data.vector_store.embed import generate_embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def store_sentiment_data(df: pd.DataFrame) -> bool:
    """
    Store sentiment data in Weaviate.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        bool: True if successful, False otherwise
    """
    client = get_weaviate_client()
    
    try:
        # Ensure the schema exists
        create_sentiment_schema(client)
        
        # Get the collection
        collection = client.collections.get("CryptoNewsSentiment")
        
        # Store data in batches
        batch_size = 50
        total_stored = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            objects_to_add = []
            
            for _, row in batch.iterrows():
                try:
                    # Generate embedding for the content
                    content = row.get('content', '')
                    vector = None
                    if content:
                        try:
                            vector = generate_embedding(content)
                        except Exception as e:
                            logger.warning(f"Error generating embedding: {e}")
                    
                    # Prepare properties
                    properties = {
                        "source": row.get('source', 'unknown'),
                        "title": row.get('title', ''),
                        "url": row.get('url', ''),
                        "content": content,
                        "sentiment_label": row.get('sentiment_label', 'NEUTRAL'),
                        "sentiment_score": float(row.get('sentiment_score', 0.5)),
                        "analyzed_at": row.get('analyzed_at', datetime.now().isoformat())
                    }
                    
                    # Handle optional properties
                    if 'date' in row and row['date'] is not None:
                        if isinstance(row['date'], str):
                            properties['date'] = row['date']
                        else:
                            properties['date'] = row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date'])
                    
                    if 'authors' in row and row['authors'] is not None:
                        if isinstance(row['authors'], list):
                            properties['authors'] = row['authors']
                        elif isinstance(row['authors'], str) and row['authors'].startswith('[') and row['authors'].endswith(']'):
                            # Parse string representation of list
                            try:
                                authors = json.loads(row['authors'].replace("'", "\""))
                                properties['authors'] = authors
                            except:
                                properties['authors'] = [row['authors']]
                        else:
                            properties['authors'] = [row['authors']]
                    
                    if 'image_url' in row and row['image_url'] is not None:
                        properties['image_url'] = row['image_url']
                    
                    # Add object
                    if vector:
                        objects_to_add.append({"properties": properties, "vector": vector})
                    else:
                        objects_to_add.append({"properties": properties})
                    
                except Exception as e:
                    logger.error(f"Error preparing sentiment data: {e}")
            
            # Insert batch
            if objects_to_add:
                try:
                    for obj in objects_to_add:
                        if "vector" in obj and obj["vector"] is not None:
                            collection.data.insert(
                                properties=obj["properties"],
                                vector=obj["vector"]
                            )
                        else:
                            # Skip items without vectors as we need them for searching
                            logger.warning(f"Skipping item without vector: {obj['properties'].get('title', 'unknown')}")
                            continue
                    
                    total_stored += len(objects_to_add)
                    logger.info(f"Stored batch of {len(objects_to_add)} sentiment records")
                except Exception as e:
                    logger.error(f"Error storing sentiment batch: {e}")
        
        logger.info(f"Successfully stored {total_stored} sentiment records")
        return True
    
    except Exception as e:
        logger.error(f"Error storing sentiment data: {e}")
        return False
    
    finally:
        if client:
            client.close()

def get_sentiment_stats(symbol: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
    """
    Get sentiment statistics from the database.
    
    Args:
        symbol (str, optional): Crypto symbol to filter by
        days (int): Number of recent days to include
        
    Returns:
        dict: Sentiment statistics
    """
    client = get_weaviate_client()
    
    try:
        # Get the collection
        try:
            collection = client.collections.get("CryptoNewsSentiment")
        except WeaviateBaseError:
            logger.warning("CryptoNewsSentiment collection not found")
            return {
                "total_articles": 0,
                "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
                "avg_sentiment": 0.5
            }
        
        # Build query
        if symbol:
            # Format symbol (remove USDT if present)
            clean_symbol = symbol.replace("USDT", "").lower()
            
            # Get content containing the symbol
            from weaviate.classes.query import Filter
            query_result = collection.query.fetch_objects(
                filters=Filter.by_property("content").contains_all([clean_symbol]),
                return_properties=["source", "title", "sentiment_label", "sentiment_score", "date", "analyzed_at"],
                limit=100
            )
        else:
            # Get all recent sentiment data
            query_result = collection.query.fetch_objects(
                return_properties=["source", "title", "sentiment_label", "sentiment_score", "date", "analyzed_at"],
                limit=100
            )
        
        if not query_result.objects:
            return {
                "total_articles": 0,
                "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
                "avg_sentiment": 0.5
            }
        
        # Process results
        sentiment_distribution = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        sentiment_scores = []
        
        for obj in query_result.objects:
            label = obj.properties.get("sentiment_label", "NEUTRAL")
            score = obj.properties.get("sentiment_score", 0.5)
            
            sentiment_distribution[label] = sentiment_distribution.get(label, 0) + 1
            sentiment_scores.append(score)
        
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        return {
            "total_articles": len(query_result.objects),
            "sentiment_distribution": sentiment_distribution,
            "avg_sentiment": avg_sentiment
        }
    
    except Exception as e:
        logger.error(f"Error getting sentiment stats: {e}")
        return {
            "total_articles": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0},
            "avg_sentiment": 0.5,
            "error": str(e)
        }
    
    finally:
        if client:
            client.close()

def cleanup_sentiment_data():
    """Delete the CryptoNewsSentiment collection to start fresh"""
    client = get_weaviate_client()
    
    try:
        client.collections.delete("CryptoNewsSentiment")
        logger.info("Successfully deleted CryptoNewsSentiment collection")
        return True
    except Exception as e:
        logger.info(f"Collection doesn't exist or already deleted: {e}")
        return False
    finally:
        if client:
            client.close()

# Example usage
if __name__ == "__main__":
    # Test sentiment stats
    stats = get_sentiment_stats()
    print("Sentiment Statistics:")
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Average Sentiment: {stats['avg_sentiment']:.2f}")
    print("Sentiment Distribution:")
    for label, count in stats['sentiment_distribution'].items():
        print(f"  {label}: {count}")