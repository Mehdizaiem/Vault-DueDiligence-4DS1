import weaviate
import os
from dotenv import load_dotenv
from vector_store.embed import generate_embedding

# Load environment variables
load_dotenv()

# Get Weaviate connection details
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Ensure URL has proper scheme
if not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"

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
        # Get the correct collection
        collection = client.collections.get("CryptoDueDiligenceDocuments")
        
        # Perform the vector search
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
                "distance": obj.metadata.distance if obj.metadata else float('inf'),
                "title": obj.properties.get("title", "Untitled"),
                "document_type": obj.properties.get("document_type", "Unknown")
            })
        
        return results
    
    except weaviate.exceptions.WeaviateBaseError as e:
        if "collection not found" in str(e).lower():
            print("Collection 'CryptoDueDiligenceDocuments' not found. Please ingest documents first.")
            return []
        else:
            print(f"Weaviate error during search: {str(e)}")
            raise
    except Exception as e:
        print(f"Unexpected error during search: {str(e)}")
        raise

if __name__ == "__main__":
    from vector_store.weaviate_client import get_weaviate_client
    
    try:
        # Initialize and connect client
        client = get_weaviate_client()
        
        # Test query
        query = "fraudulent crypto transactions"
        results = search_documents(client, query)
        
        if results:
            for idx, doc in enumerate(results):
                print(f"Result {idx + 1}: {doc['source']} (Distance: {doc['distance']:.3f})")
                print(f"Title: {doc['title']}")
                print(f"Type: {doc['document_type']}")
                print(f"Content: {doc['content'][:500]}...\n")
        else:
            print("No results found. Make sure you've ingested documents first.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            print("Weaviate client connection closed.")