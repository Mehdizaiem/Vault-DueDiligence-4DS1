import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from weaviate.exceptions import WeaviateBaseError
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_schema_properties(client, collection_name="CryptoDueDiligenceDocuments"):
    """Get available properties from the schema"""
    try:
        # Get the collection
        collection = client.collections.get(collection_name)
        # Get schema configuration
        config = collection.config.get()
        # Extract property names
        properties = [prop.name for prop in config.properties]
        
        logger.info(f"Found {len(properties)} properties in schema: {properties}")
        return properties
    except Exception as e:
        logger.error(f"Error getting schema properties: {str(e)}")
        return ["title", "document_type", "source", "content"]  # Default fallback

def get_documents(client, properties, document_type=None, limit=100):
    """
    Retrieve documents with specified properties from Weaviate
    
    Args:
        client: Weaviate client
        properties: List of properties to retrieve
        document_type: Optional filter by document type
        limit: Maximum number of documents to retrieve
        
    Returns:
        List of documents
    """
    try:
        collection = client.collections.get("CryptoDueDiligenceDocuments")
        
        # Build query kwargs - Weaviate v4 syntax
        kwargs = {
            "limit": limit,
            "return_properties": properties  # Correct parameter name
        }
        
        # Add filter if document_type specified and it's in the properties
        if document_type and "document_type" in properties:
            from weaviate.classes.query import Filter
            kwargs["filters"] = Filter.by_property("document_type").equal(document_type)
        
        # Execute query
        results = collection.query.fetch_objects(**kwargs)
        
        return results.objects
    
    except WeaviateBaseError as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return []

def visualize_document_types(documents, output_dir):
    """Visualize distribution of document types"""
    if not documents:
        logger.warning("No documents to visualize document types")
        return
    
    # Check if document_type is available in the documents
    first_doc = documents[0]
    if not hasattr(first_doc, 'properties') or not first_doc.properties.get('document_type'):
        logger.warning("document_type property not available in documents")
        return
    
    # Count document types
    doc_types = {}
    for doc in documents:
        props = doc.properties
        if not props:
            continue
            
        doc_type = props.get('document_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    # Create DataFrame
    df = pd.DataFrame({'document_type': doc_types.keys(), 'count': doc_types.values()})
    df = df.sort_values('count', ascending=False)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Plot pie chart
    plt.pie(df['count'], labels=df['document_type'], autopct='%1.1f%%', 
            startangle=90, shadow=True, explode=[0.05] * len(df))
    plt.axis('equal')
    plt.title('Document Types Distribution')
    
    # Save figure
    file_name = "document_types.png"
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()
    
    logger.info(f"Created document types visualization: {file_name}")

def export_documents(documents, output_dir):
    """Export document data as JSON for further analysis"""
    if not documents:
        logger.warning("No documents to export")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for export
    export_data = []
    for doc in documents:
        if not hasattr(doc, 'properties') or not doc.properties:
            continue
        
        # Extract all available properties
        doc_data = {
            'id': str(doc.uuid),  # Convert UUID to string
        }
        
        # Add all properties
        for key, value in doc.properties.items():
            doc_data[key] = value
        
        export_data.append(doc_data)
    
    # Export as JSON
    with open(os.path.join(output_dir, 'documents_export.json'), 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info("Exported document data to documents_export.json")

def main():
    parser = argparse.ArgumentParser(description='Visualize crypto document features')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                      help='Directory to save visualizations')
    parser.add_argument('--document_type', type=str, help='Filter by document type')
    parser.add_argument('--limit', type=int, default=1000, 
                      help='Maximum number of documents to process')
    
    args = parser.parse_args()
    
    # Import needed here to avoid circular imports
    from vector_store.weaviate_client import get_weaviate_client
    
    try:
        # Initialize Weaviate client
        client = get_weaviate_client()
        
        # Get available properties from schema
        available_properties = get_schema_properties(client)
        
        # Filter essential properties that exist in schema
        essential_properties = [p for p in ["title", "document_type", "source", "content"] if p in available_properties]
        
        if not essential_properties:
            logger.warning("No essential properties found in schema")
            essential_properties = ["title"]  # Minimum fallback
        
        # Retrieve documents with available properties
        logger.info(f"Retrieving documents with properties: {essential_properties}")
        documents = get_documents(client, essential_properties, args.document_type, args.limit)
        
        if not documents:
            logger.warning("No documents found")
            return
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Generate visualizations if possible
        if "document_type" in essential_properties:
            visualize_document_types(documents, args.output_dir)
        
        # Export document data
        export_documents(documents, args.output_dir)
        
        logger.info(f"All visualizations saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    main()