import weaviate
from vector_store.embed import generate_embedding

def search_documents(client, query_text, top_k=5):
    """
    Search Weaviate for similar documents using vector search.
    """
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
            "title": obj.properties.get("title", "Unknown"),
            "source": obj.properties.get("source", "Unknown"),
            "content": obj.properties.get("content", "")[:500] + "...",
            "document_type": obj.properties.get("document_type", "Unknown"),
            "category": obj.properties.get("category", "Unknown"),
            "distance": obj.metadata.distance if obj.metadata else "N/A"
        })
    
    return results

if __name__ == "__main__":
    from vector_store.weaviate_client import get_weaviate_client
    client = get_weaviate_client()
    
    query = "crypto fraud regulations"
    results = search_documents(client, query)

    for idx, doc in enumerate(results):
        print(f"Result {idx + 1}: {doc['title']} (Type: {doc['document_type']}, Distance: {doc['distance']:.3f})")
        print(f"Category: {doc['category']}")
        print(f"Content: {doc['content']}\n")

    client.close()
