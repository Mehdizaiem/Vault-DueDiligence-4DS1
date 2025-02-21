import weaviate
import os
from dotenv import load_dotenv
from vector_store.embed import generate_embedding

# Load environment variables
load_dotenv()

# Get Weaviate connection details
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Ensure URL has http:// or https:// prefix
if not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"http://{WEAVIATE_URL}"

# Initialize Weaviate client with v4 syntax
client = weaviate.WeaviateClient(
    connection_params=weaviate.connect.ConnectionParams.from_url(
        url=WEAVIATE_URL,
        grpc_port=WEAVIATE_GRPC_PORT
    )
)

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
        # Get the collection
        collection = client.collections.get("LegalDocument")
        
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
        print(f"Unexpected error during search: {str(e)}")
        raise

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
    finally:
        client.close()