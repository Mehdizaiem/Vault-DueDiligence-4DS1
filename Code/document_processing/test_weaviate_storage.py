# Save as Code/document_processing/test_weaviate_storage.py
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.storage_manager import StorageManager

# Initialize storage manager
storage_manager = StorageManager()

if not storage_manager.connect():
    print("Failed to connect to Weaviate")
    sys.exit(1)

try:
    # Get CryptoDueDiligenceDocuments collection
    collection = storage_manager.client.collections.get("CryptoDueDiligenceDocuments")
    
    # Get document count
    count_result = collection.aggregate.over_all(total_count=True)
    total_documents = count_result.total_count
    
    print(f"Total documents in CryptoDueDiligenceDocuments: {total_documents}")
    
    # Retrieve a few documents
    response = collection.query.fetch_objects(limit=3)
    
    print("\nSample documents:")
    for i, obj in enumerate(response.objects, 1):
        print(f"\nDocument {i}:")
        print(f"  UUID: {obj.uuid}")
        print(f"  Title: {obj.properties.get('title', 'Untitled')}")
        print(f"  Document Type: {obj.properties.get('document_type', 'Unknown')}")
        
        # Check if TF-IDF keywords are stored
        keywords = obj.properties.get('keywords', [])
        print(f"  Keywords: {keywords[:5]}")
        
        # Show content preview
        content = obj.properties.get('content', '')
        if content:
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"  Content preview: {preview}")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    storage_manager.close()