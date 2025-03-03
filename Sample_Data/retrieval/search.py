import weaviate
from vector_store.embed import generate_embedding
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_documents(client, query_text, top_k=5):
    """Search Weaviate for similar documents."""
    query_vector = generate_embedding(query_text)
    collection = client.collections.get("CryptoDueDiligenceDocuments")

    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=["distance"]
    )

    results = []
    for obj in response.objects:
        results.append({
            "source": obj.properties.get("source", "Unknown"),  # ✅ Prevents KeyError
            "title": obj.properties.get("filename", "Unknown"),
            "category": obj.properties.get("category", "Unknown"),
            "content": obj.properties.get("content", "")[:500] + "...",
            "distance": obj.metadata.distance if obj.metadata else "N/A"
        })
    
    return results
