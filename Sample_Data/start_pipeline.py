# Sample_Data/start_pipeline.py
from vector_store.weaviate_client import get_weaviate_client
from data_ingestion.ingest import ingest_documents
from retrieval.search import search_documents
from src.feature_selection import feature_selection_pipeline

def run_pipeline():
    client = None
    print("📌 Step 1: Connecting to Weaviate...")
    try:
        client = get_weaviate_client()
        if not client.is_live():
            print("❌ Weaviate connection failed. Exiting.")
            return
        print("📥 Step 2: Loading and ingesting documents...")
        ingest_documents(client)
        print("🔍 Step 3: Searching for 'crypto fraud regulations'...")
        results = search_documents(client, "crypto fraud regulations")
        if results:
            for idx, doc in enumerate(results):
                print(f"✅ Result {idx + 1}: {doc['source']} (Distance: {doc.get('distance', 'N/A'):.3f})")
                print(f"📝 Content: {doc['content'][:300]}...\n")
        else:
            print("⚠️ No search results found.")
        print("📊 Step 4: Running Feature Selection...")
        feature_selection_pipeline(client)
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
    finally:
        if client is not None:
            client.close()
            print("🔌 Weaviate client connection closed.")

if __name__ == "__main__":
    run_pipeline()