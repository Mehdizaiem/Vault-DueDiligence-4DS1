import logging
from vector_store.weaviate_client import get_weaviate_client
from weaviate.classes.query import Filter

# Configure logging
logging.basicConfig(level=logging.INFO)

# Connect to Weaviate
client = get_weaviate_client()

try:
    # Get collection
    collection = client.collections.get("CryptoDueDiligenceDocuments")
    
    # 1. Check if documents contain specific keywords
    print("\n=== KEYWORD SEARCH TEST ===")
    keywords = ["crypto", "fraud", "regulation"]
    
    for keyword in keywords:
        query_filter = Filter.by_property("content").contains_any([keyword])
        response = collection.query.fetch_objects(
            filters=query_filter,
            limit=3
        )
        print(f"Found {len(response.objects)} documents containing '{keyword}'")
        
        if response.objects:
            for obj in response.objects:
                print(f"- Source: {obj.properties.get('source')}")
    
    # 2. Test with lower threshold
    print("\n=== HYBRID SEARCH WITH LOWER THRESHOLD ===")
    from Sample_Data.vector_store.embed import generate_mpnet_embedding
    
    query_text = "crypto fraud regulations"
    query_vector = generate_mpnet_embedding(query_text)
    
    response = collection.query.hybrid(
        query=query_text,
        vector=query_vector,
        alpha=0.25,  # Favor keyword search more
        limit=5,
        return_metadata=["distance", "score"]
    )
    
    print(f"Found {len(response.objects)} documents with hybrid search")
    for obj in response.objects:
        distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
        similarity = 1.0 - distance
        print(f"- Source: {obj.properties.get('source')} (Similarity: {similarity:.3f})")
    
    # 3. Check if any documents contain regulations-related terms
    print("\n=== REGULATIONS TERMINOLOGY CHECK ===")
    reg_terms = ["regulation", "compliance", "legal", "regulatory", "law", "policy"]
    
    for term in reg_terms:
        query_filter = Filter.by_property("content").contains_any([term])
        response = collection.query.fetch_objects(
            filters=query_filter,
            limit=3
        )
        print(f"Found {len(response.objects)} documents containing '{term}'")
        
        if response.objects:
            print(f"Example: {response.objects[0].properties.get('source')}")

finally:
    client.close()