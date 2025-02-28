import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.embed import generate_embedding
from Sample_Data.vector_store.weaviate_client import get_weaviate_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def search_documents(client, query_text, top_k=3):
    """
    Search for documents using vector similarity.
    
    Args:
        client (weaviate.WeaviateClient): The active Weaviate client
        query_text (str): The query text to search for
        top_k (int): Number of results to return
        
    Returns:
        list: List of matching documents
    """
    # Generate embedding for the query
    query_vector = generate_embedding(query_text)
    
    try:
        # Try searching CryptoDueDiligenceDocuments first
        try:
            # Get the collection
            collection = client.collections.get("CryptoDueDiligenceDocuments")
            
            # Perform the search
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"]
            )
            
            # Format the results
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "source": obj.properties.get("source", "Unknown"),
                    "distance": obj.metadata.distance if obj.metadata else "N/A"
                })
            
            if results:  # If we found results, return them
                return results
                
        except Exception as e:
            logger.warning(f"CryptoDueDiligenceDocuments search failed: {e}")
        
        # Fall back to searching CryptoNewsSentiment
        try:
            # Get the CryptoNewsSentiment collection
            collection = client.collections.get("CryptoNewsSentiment")
            
            # Perform the vector search
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_properties=["source", "title", "content", "sentiment_label", "sentiment_score"],
                return_metadata=["distance"]
            )
            
            # Format the results
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "source": f"{obj.properties.get('source', 'Unknown')} - {obj.properties.get('title', '')}",
                    "distance": obj.metadata.distance if obj.metadata else "N/A",
                    "sentiment": f"{obj.properties.get('sentiment_label', 'NEUTRAL')} ({obj.properties.get('sentiment_score', 0.5):.2f})"
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"CryptoNewsSentiment search failed: {e}")
            
        # Last resort - try to find content containing keywords
        try:
            collection = client.collections.get("CryptoNewsSentiment")
            
            # Extract keywords from query (simple approach)
            keywords = [word.lower() for word in query_text.split() if len(word) > 3]
            
            if not keywords:
                logger.warning("No substantial keywords found in query")
                return []
            
            # Search by keyword
            from weaviate.classes.query import Filter
            query_filter = Filter.by_property("content").contains_any(keywords)
            
            response = collection.query.fetch_objects(
                filters=query_filter,
                limit=top_k,
                return_properties=["source", "title", "content", "sentiment_label", "sentiment_score"]
            )
            
            # Format the results
            results = []
            for obj in response.objects:
                results.append({
                    "content": obj.properties.get("content", ""),
                    "source": f"{obj.properties.get('source', 'Unknown')} - {obj.properties.get('title', '')}",
                    "distance": "Keyword match",
                    "sentiment": f"{obj.properties.get('sentiment_label', 'NEUTRAL')} ({obj.properties.get('sentiment_score', 0.5):.2f})"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

if __name__ == "__main__":
    client = get_weaviate_client()
    try:
        query = "cryptocurrency market trends"
        results = search_documents(client, query)
        
        if results:
            for idx, doc in enumerate(results):
                print(f"Result {idx + 1}: {doc['source']}")
                print(f"Content: {doc['content'][:300]}...\n")
        else:
            print("No results found. Make sure you've ingested documents first.")
    finally:
        client.close()