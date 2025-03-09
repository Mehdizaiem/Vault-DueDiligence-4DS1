import os
import sys
import logging
import traceback
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

def search_documents(client, query_text, top_k=3, similarity_threshold=0.6):
    """
    Search for documents using improved vector similarity with hybrid search.
    
    Args:
        client (weaviate.WeaviateClient): The active Weaviate client
        query_text (str): The query text to search for
        top_k (int): Number of results to return
        similarity_threshold (float): Threshold for similarity cutoff
        
    Returns:
        list: List of matching documents
    """
    if not query_text or not isinstance(query_text, str):
        logger.warning("Invalid query text provided")
        return []
        
    # Generate embedding for the query with error handling
    try:
        query_vector = generate_embedding(query_text)
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    # Track collections that have been searched
    searched_collections = set()
    
    try:
        # First try searching CryptoDueDiligenceDocuments
        results = _search_collection(client, "CryptoDueDiligenceDocuments", query_text, query_vector, top_k, similarity_threshold)
        searched_collections.add("CryptoDueDiligenceDocuments")
        
        if results:
            return results
            
        # Then try CryptoNewsSentiment
        results = _search_collection(client, "CryptoNewsSentiment", query_text, query_vector, top_k, similarity_threshold)
        searched_collections.add("CryptoNewsSentiment")
        
        if results:
            return results
        
        # If no results, try keyword search across all collections
        logger.info("No vector search results found, trying keyword search")
        
        # Generate keywords from query (minimum 3 characters)
        keywords = [word.lower() for word in query_text.split() if len(word) >= 3]
        
        # Try all relevant collections that we haven't searched yet
        collection_names = [
            "CryptoDueDiligenceDocuments", 
            "CryptoNewsSentiment"
        ]
        
        combined_results = []
        
        for collection_name in collection_names:
            if collection_name in searched_collections:
                continue  # Skip collections we've already searched
                
            keyword_results = _keyword_search(client, collection_name, keywords, top_k)
            combined_results.extend(keyword_results)
            
        # Sort all results by relevance if we have any
        if combined_results:
            combined_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            return combined_results[:top_k]
            
        # If still no results, return empty list with message
        logger.warning(f"No results found for query: {query_text}")
        return []
            
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def _search_collection(client, collection_name, query_text, query_vector, top_k, similarity_threshold):
    """Helper function to search a specific collection"""
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
        
        return results
    except weaviate.exceptions.WeaviateQueryError as e:
        if "collection not found" in str(e).lower():
            print("No documents found in collection. Please ingest documents first.")
            return []
        else:
            raise
    except Exception as e:
        logger.error(f"Keyword search failed for {collection_name}: {e}")
        return []

if __name__ == "__main__":
    from vector_store.weaviate_client import get_weaviate_client
    client = get_weaviate_client()
    try:
        query = "fraudulent crypto transactions"
        results = search_documents(client, query)
        
        if results:
            for idx, doc in enumerate(results):
                print(f"Result {idx + 1}: {doc['source']} (Distance: {doc['distance']:.3f})")
                print(f"Content: {doc['content'][:500]}...\n")
        else:
            print("No results found. Make sure you've ingested documents first.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            print("Weaviate client connection closed.")