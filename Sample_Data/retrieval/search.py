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
        collection = client.collections.get(collection_name)
        
        # Try hybrid search (vector + BM25 text search) for better results
        try:
            from weaviate.classes.query import QueryNearTextArgument
            
            # Hybrid search (alpha=0.5 means 50% vector search, 50% BM25 text search)
            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=0.5,  # Balance between vector and keyword search
                limit=top_k,
                return_metadata=["distance", "score"]
            )
            
            # Format the results
            results = []
            for obj in response.objects:
                # Calculate hybrid score
                vector_dist = obj.metadata.distance if obj.metadata and hasattr(obj.metadata, 'distance') else 1.0
                bm25_score = obj.metadata.score if obj.metadata and hasattr(obj.metadata, 'score') else 0.0
                
                # Convert distance to similarity (1 - distance)
                similarity = 1.0 - vector_dist
                
                # Check if similarity is above threshold
                if similarity >= similarity_threshold:
                    # Get content and source from properties
                    content = obj.properties.get("content", "")
                    source = obj.properties.get("source", "Unknown")
                    
                    # Add collection-specific formatting
                    if collection_name == "CryptoNewsSentiment":
                        title = obj.properties.get("title", "")
                        sentiment = obj.properties.get("sentiment_label", "NEUTRAL")
                        sentiment_score = obj.properties.get("sentiment_score", 0.5)
                        source = f"{source} - {title}"
                        source_type = "news"
                    else:
                        sentiment = None
                        sentiment_score = None
                        source_type = "document"
                    
                    results.append({
                        "content": content[:1000] + "..." if len(content) > 1000 else content,
                        "source": source,
                        "source_type": source_type,
                        "distance": vector_dist,
                        "similarity": similarity,
                        "relevance": (similarity + bm25_score) / 2,  # Combined relevance score
                        "sentiment": f"{sentiment} ({sentiment_score:.2f})" if sentiment and sentiment_score else None
                    })
            
            # Sort by relevance score (descending)
            results.sort(key=lambda x: x["relevance"], reverse=True)
            return results
            
        except Exception as hybrid_error:
            logger.error(f"Hybrid search error: {hybrid_error}")
            
            # Fall back to vector-only search
            logger.info("Falling back to vector-only search")
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=["distance"]
            )
            
            # Format the results
            results = []
            for obj in response.objects:
                distance = obj.metadata.distance if obj.metadata else 0.9
                similarity = 1.0 - distance
                
                if similarity >= similarity_threshold:
                    content = obj.properties.get("content", "")
                    source = obj.properties.get("source", "Unknown")
                    
                    if collection_name == "CryptoNewsSentiment":
                        title = obj.properties.get("title", "")
                        sentiment = obj.properties.get("sentiment_label", "NEUTRAL")
                        sentiment_score = obj.properties.get("sentiment_score", 0.5)
                        source = f"{source} - {title}"
                        source_type = "news"
                    else:
                        sentiment = None
                        sentiment_score = None
                        source_type = "document"
                    
                    results.append({
                        "content": content[:1000] + "..." if len(content) > 1000 else content,
                        "source": source,
                        "source_type": source_type,
                        "distance": distance,
                        "similarity": similarity,
                        "relevance": similarity,
                        "sentiment": f"{sentiment} ({sentiment_score:.2f})" if sentiment and sentiment_score else None
                    })
            
            return results
        
    except Exception as e:
        logger.warning(f"{collection_name} search failed: {e}")
        return []

def _keyword_search(client, collection_name, keywords, top_k):
    """Perform keyword search on a collection"""
    if not keywords:
        return []
        
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        
        # Build property filter based on collection
        from weaviate.classes.query import Filter
        
        if collection_name == "CryptoDueDiligenceDocuments":
            # Search content field for documents
            query_filter = Filter.by_property("content").contains_any(keywords)
        else:
            # For news, search both content and title
            query_filter = (
                Filter.by_property("content").contains_any(keywords) | 
                Filter.by_property("title").contains_any(keywords)
            )
        
        # Execute query
        response = collection.query.fetch_objects(
            filters=query_filter,
            limit=top_k
        )
        
        # Format results
        results = []
        for obj in response.objects:
            content = obj.properties.get("content", "")
            source = obj.properties.get("source", "Unknown")
            
            # Collection-specific processing
            if collection_name == "CryptoNewsSentiment":
                title = obj.properties.get("title", "")
                sentiment = obj.properties.get("sentiment_label", "NEUTRAL")
                sentiment_score = obj.properties.get("sentiment_score", 0.5)
                source = f"{source} - {title}"
                source_type = "news"
            else:
                sentiment = None
                sentiment_score = None
                source_type = "document"
            
            # Calculate basic relevance score based on keyword matches
            relevance = 0.0
            for keyword in keywords:
                if keyword in content.lower():
                    relevance += 0.1  # Add 0.1 for each matching keyword
            
            results.append({
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "source": source,
                "source_type": source_type,
                "distance": "Keyword match",
                "similarity": relevance,
                "relevance": min(0.8, relevance),  # Cap at 0.8 for keyword matches
                "sentiment": f"{sentiment} ({sentiment_score:.2f})" if sentiment and sentiment_score else None
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Keyword search failed for {collection_name}: {e}")
        return []

if __name__ == "__main__":
    client = get_weaviate_client()
    try:
        query = "cryptocurrency market trends"
        results = search_documents(client, query)
        
        if results:
            for idx, doc in enumerate(results):
                print(f"Result {idx + 1}: {doc['source']} (Relevance: {doc.get('relevance', 'N/A'):.3f})")
                print(f"Content: {doc['content'][:300]}...\n")
        else:
            print("No results found. Make sure you've ingested documents first.")
    finally:
        client.close()