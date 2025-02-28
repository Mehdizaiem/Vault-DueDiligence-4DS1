import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
import os
import sys
from dotenv import load_dotenv

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def get_weaviate_client():
    """Create and return a Weaviate client with improved error handling and connection options"""
    load_dotenv()
    
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:9090")
    WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    
    try:
        client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=WEAVIATE_URL,
                grpc_port=WEAVIATE_GRPC_PORT
            ),
            additional_config=AdditionalConfig(
                timeout=Timeout(connect=60, query=120, init=60)
            ),
            skip_init_checks=False  # Keep false unless gRPC issues persist
        )
        
        # Explicitly connect to Weaviate
        client.connect()
        
        if not client.is_live():
            raise ConnectionError("Failed to connect to Weaviate. Check server status.")
        
        print("Weaviate client is connected.")
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        print("Please ensure Weaviate is running, accessible, and your network connection is stable.")
        raise
    
def check_weaviate_connection(client):
    """Utility function to test Weaviate connection"""
    try:
        # v4 syntax for checking connection
        client.connect()
        print("Successfully connected to Weaviate!")
        return True
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Test connection
    client = get_weaviate_client()
    try:
        if client.is_live():
            print("✅ Connection test successful!")
            
            # Show collections
            collections = client.collections.list()
            print(f"\nAvailable collections:")
            for collection in collections:
                print(f"- {collection.name}")
        else:
            print("❌ Connection test failed: Weaviate is not responding")
    finally:
        client.close()