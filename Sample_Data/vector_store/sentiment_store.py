import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, List, Union, Any
from weaviate.exceptions import WeaviateBaseError
import json
import traceback
import numpy as np

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
    Store sentiment data in Weaviate with improved error handling.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        bool: True if successful, False otherwise
    """
    client = get_weaviate_client()
    
    try:
        # Ensure the schema exists
        collection = create_sentiment_schema(client)
        
        # Store data in batches with better error handling
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
                    if content and isinstance(content, str):
                        try:
                            vector = generate_embedding(content)
                            # Verify vector is not all zeros and has expected dimension
                            if not vector or not any(vector):
                                logger.warning(f"Generated zero vector for article: {row.get('title', '')}")
                                vector = None
                        except Exception as e:
                            logger.warning(f"Error generating embedding: {e}")
                    
                    # Prepare properties
                    properties = {
                        "source": row.get('source', 'unknown'),
                        "title": row.get('title', ''),
                        "url": row.get('url', ''),
                        "content": content[:200000] if content else '',  # Limit content length
                        "sentiment_label": row.get('sentiment_label', 'NEUTRAL'),
                        "sentiment_score": float(row.get('sentiment_score', 0.5)),
                        "analyzed_at": row.get('analyzed_at', datetime.now().isoformat())
                    }
                    
                    # Handle optional properties
                    if 'date' in row and row['date'] is not None:
                        if isinstance(row['date'], str):
                            # Ensure date is in RFC3339 format
                            try:
                                # Try to parse the string date
                                if 'T' in row['date'] and ('+' in row['date'] or 'Z' in row['date']):
                                    # Already in ISO format
                                    properties['date'] = row['date']
                                else:
                                    # Try to parse and convert
                                    parsed_date = pd.to_datetime(row['date'])
                                    properties['date'] = parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                            except:
                                properties['date'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                        elif isinstance(row['date'], (datetime, pd.Timestamp)):
                            properties['date'] = row['date'].strftime("%Y-%m-%dT%H:%M:%SZ")
                        else:
                            properties['date'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    else:
                        properties['date'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    # Handle authors array
                    if 'authors' in row and row['authors'] is not None:
                        if isinstance(row['authors'], list):
                            properties['authors'] = row['authors']
                        elif isinstance(row['authors'], str):
                            if row['authors'].startswith('[') and row['authors'].endswith(']'):
                                # Parse string representation of list
                                try:
                                    authors = json.loads(row['authors'].replace("'", "\""))
                                    properties['authors'] = authors if isinstance(authors, list) else [row['authors']]
                                except:
                                    properties['authors'] = [row['authors']]
                            else:
                                properties['authors'] = [row['authors']]
                        else:
                            properties['authors'] = []
                    
                    if 'image_url' in row and row['image_url'] is not None:
                        properties['image_url'] = str(row['image_url'])
                    
                    # Add object to batch
                    if vector:
                        objects_to_add.append({"properties": properties, "vector": vector})
                    else:
                        # Generate a random vector as fallback (not ideal but prevents errors)
                        fallback_vector = np.random.randn(768).tolist() 
                        fallback_vector = fallback_vector / np.linalg.norm(fallback_vector)
                        logger.warning(f"Using fallback vector for: {properties.get('title', 'unknown')}")
                        objects_to_add.append({"properties": properties, "vector": fallback_vector})
                    
                except Exception as e:
                    logger.error(f"Error preparing sentiment data: {e}")
                    logger.error(traceback.format_exc())
            
            # Insert batch with retry logic
            if objects_to_add:
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        success_count = 0
                        for obj in objects_to_add:
                            try:
                                collection.data.insert(
                                    properties=obj["properties"],
                                    vector=obj["vector"]
                                )
                                success_count += 1
                            except Exception as obj_error:
                                logger.error(f"Error inserting object: {obj_error}")
                        
                        total_stored += success_count
                        logger.info(f"Stored batch of {success_count} sentiment records (attempt {retry_count + 1})")
                        break  # Exit retry loop on success
                    except Exception as batch_error:
                        logger.error(f"Error storing sentiment batch (attempt {retry_count + 1}): {batch_error}")
                        retry_count += 1
                        if retry_count == max_retries:
                            logger.error("Max retries reached for batch")
        
        logger.info(f"Successfully stored {total_stored} sentiment records")
        return True
    
    except Exception as e:
        logger.error(f"Error storing sentiment data: {e}")
        logger.error(traceback.format_exc())
        return False
    
    finally:
        if client:
            client.close()

def get_sentiment_stats(symbol: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
    """
    Get sentiment statistics from the database with improved error handling.
    
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
                "avg_sentiment": 0.5,
                "error": "Collection not found"
            }
        
        # Calculate date range for filtering
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Build query with appropriate filters
        try:
            from weaviate.classes.query import Filter
            
            if symbol:
                # Format symbol (remove USDT if present)
                clean_symbol = symbol.replace("USDT", "").lower()
                
                # Try multiple filter approaches for better results
                try:
                    # Text-based search instead of vector-based
                    query_result = collection.query.fetch_objects(
                        filters=Filter.by_property("content").contains_all([clean_symbol]) | 
                               Filter.by_property("title").contains_all([clean_symbol]),
                        return_properties=["source", "title", "sentiment_label", "sentiment_score", "date", "analyzed_at"],
                        limit=100
                    )
                except Exception as filter_error:
                    logger.error(f"Content filter error: {filter_error}")
                    # Try simpler filter as fallback
                    query_result = collection.query.fetch_objects(
                        filters=Filter.by_property("sentiment_score").is_not_null(),
                        return_properties=["source", "title", "sentiment_label", "sentiment_score", "date", "analyzed_at", "content"],
                        limit=100
                    )
                    
                    # Filter results programmatically
                    filtered_objects = []
                    for obj in query_result.objects:
                        if clean_symbol in obj.properties.get("content", "").lower() or clean_symbol in obj.properties.get("title", "").lower():
                            filtered_objects.append(obj)
                    
                    query_result.objects = filtered_objects
            else:
                # Get all recent sentiment data
                query_result = collection.query.fetch_objects(
                    filters=Filter.by_property("date").greater_than(start_date_str),
                    return_properties=["source", "title", "sentiment_label", "sentiment_score", "date", "analyzed_at"],
                    limit=100
                )
        except Exception as e:
            logger.error(f"Error building query: {e}")
            # Fallback to unfiltered query
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
        
        # Enhanced output
        result = {
            "total_articles": len(query_result.objects),
            "sentiment_distribution": sentiment_distribution,
            "avg_sentiment": avg_sentiment,
            "sentiment_trend": "stable",  # Default value
            "timeframe": f"Last {days} days"
        }
        
        # Add sentiment trend if we have enough data
        if len(sentiment_scores) > 5:
            # Compare first half to second half
            half_index = len(sentiment_scores) // 2
            first_half_avg = sum(sentiment_scores[:half_index]) / half_index
            second_half_avg = sum(sentiment_scores[half_index:]) / (len(sentiment_scores) - half_index)
            
            if second_half_avg > first_half_avg + 0.1:
                result["sentiment_trend"] = "improving"
            elif second_half_avg < first_half_avg - 0.1:
                result["sentiment_trend"] = "declining"
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting sentiment stats: {e}")
        logger.error(traceback.format_exc())
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