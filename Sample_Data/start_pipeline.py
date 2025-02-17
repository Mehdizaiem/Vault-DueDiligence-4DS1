from data_ingestion.ingest import load_documents, ingest_documents
from retrieval.search import search_documents
import os

def run_pipeline():
    """Run the complete document processing pipeline."""
    
    # Step 1: Load and ingest documents
    print("Step 1: Loading and ingesting documents...")
    documents = ingest_documents()
    
    if not documents:
        print("No documents were found or loaded. Check your DATA_PATH environment variable.")
        print(f"Current working directory: {os.getcwd()}")
        print("Exiting the pipeline.")
        return
    
    print(f"Loaded and ingested {len(documents)} documents.")
    
    # Step 2: Demonstrate search
    query = "crypto fraud regulations"
    print(f"\nStep 2: Searching for '{query}'...")
    results = search_documents(query)
    
    if results:
        print(f"Found {len(results)} results:")
        for idx, doc in enumerate(results):
            print(f"Result {idx + 1}: {doc['source']} (Score: {doc.get('score', 'N/A'):.3f})")
            print(f"Content: {doc['content'][:300]}...\n")
    else:
        print("No search results found. This may be normal if your documents don't match the query,")
        print("or it could indicate an issue with the search functionality.")

if __name__ == "__main__":
    run_pipeline()