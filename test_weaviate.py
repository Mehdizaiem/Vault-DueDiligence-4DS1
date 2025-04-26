# test_weaviate.py
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Try to import numpy first to verify installation
    import numpy as np
    logger.info("Successfully imported numpy")
    
    # Try to import other required dependencies
    import pandas as pd
    logger.info("Successfully imported pandas")
    
    # Try to import weaviate
    import weaviate
    logger.info("Successfully imported weaviate")
    
    # Try to connect to Weaviate
    client = weaviate.Client("http://localhost:8080")  # Adjust URL as needed
    
    # Check if Weaviate is ready
    if client.is_ready():
        # Get schema
        schema = client.schema.get()
        
        # Print collection names
        collection_names = [c['class'] for c in schema['classes']] if 'classes' in schema else []
        
        result = {
            "status": "connected",
            "weaviate_version": client.get_meta()['version'] if 'version' in client.get_meta() else "unknown",
            "collections": collection_names,
            "document_count": {}
        }
        
        # Count documents in key collections
        for collection in collection_names:
            try:
                count = client.query.aggregate(collection).with_meta_count().do()
                result["document_count"][collection] = count["data"]["Aggregate"][collection][0]["meta"]["count"]
            except Exception as e:
                result["document_count"][collection] = f"Error: {str(e)}"
        
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"status": "error", "message": "Weaviate is not ready"}))
        
except ImportError as e:
    print(json.dumps({"status": "error", "message": f"Missing dependencies: {str(e)}"}))
except Exception as e:
    print(json.dumps({"status": "error", "message": str(e)}))