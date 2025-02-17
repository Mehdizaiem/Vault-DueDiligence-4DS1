from vector_store.weaviate_client import get_weaviate_client
from data_ingestion.ingest import ingest_documents
from retrieval.search import search_documents
import os

def run_pipeline():
    """Run the complete document processing pipeline."""
    print("Step 1: Connecting to Weaviate...")
    try:
        client = get_weaviate_client()
        
        if not client.is_live():
            print("Weaviate connection failed. Exiting.")
            return

        print("Step 2: Loading and ingesting documents...")
        documents = ingest_documents(client)

        if not documents:
            print("No documents found. Exiting pipeline.")
            return
        
        print(f"Loaded and ingested {len(documents)} documents.")

        print("\nStep 3: Searching for 'crypto fraud regulations'...")
        results = search_documents("crypto fraud regulations")

        if results:
            for idx, doc in enumerate(results):
                print(f"Result {idx + 1}: {doc['source']} (Score: {doc.get('score', 'N/A'):.3f})")
                print(f"Content: {doc['content'][:300]}...\n")
        else:
            print("No search results found.")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    run_pipeline()