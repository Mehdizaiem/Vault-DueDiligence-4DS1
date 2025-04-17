import os
import json
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from feature_extraction import CryptoFeatureExtractor
from vector_store.weaviate_client import get_weaviate_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction_weaviate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_weaviate_documents(output_dir, collection_name="CryptoDueDiligenceDocuments", limit=1000, summary_file=None):
    """
    Process documents already stored in Weaviate and extract features
    
    Args:
        output_dir: Directory to save extracted features
        collection_name: Name of the Weaviate collection
        limit: Maximum number of documents to process
        summary_file: Optional file path to save summary of extracted features
    """
    # Initialize feature extractor
    extractor = CryptoFeatureExtractor()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to Weaviate
    logger.info(f"Connecting to Weaviate...")
    client = get_weaviate_client()
    
    # Get the collection
    try:
        collection = client.collections.get(collection_name)
        logger.info(f"Connected to collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error connecting to Weaviate collection {collection_name}: {str(e)}")
        return []
    
    # Query documents
    try:
        logger.info(f"Querying up to {limit} documents from Weaviate...")
        query_result = collection.query.fetch_objects(
            limit=limit,
            return_properties=["content", "document_type", "source", "title"]  # Specify properties explicitly
        )
        
        documents = query_result.objects
        logger.info(f"Retrieved {len(documents)} documents from Weaviate")
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return []
    
    # Initialize results storage
    all_results = []
    
    # Process each document with progress bar
    for doc in tqdm(documents, desc="Processing documents"):
        try:
            # Extract document properties
            props = doc.properties
            if not props or not props.get("content"):
                logger.warning(f"Document {doc.uuid} has no content, skipping")
                continue
            
            document_text = props.get("content", "")
            doc_type = props.get("document_type", "unknown")
            source = props.get("source", "unknown")
            
            logger.info(f"Processing document: {source} (type: {doc_type})")
            
            # Extract features
            features = extractor.extract_features(document_text, doc_type)
            
            # Create metadata
            metadata = {
                "uuid": str(doc.uuid),  # Convert UUID to string for JSON serialization
                "source": source,
                "document_type": doc_type,
                "title": props.get("title", "Unknown"),
                "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Combine metadata and features
            result = {
                "metadata": metadata,
                "features": features
            }
            
            # Generate a filename from source or uuid
            file_name = os.path.basename(source) if source != "unknown" else str(doc.uuid)
            output_file = os.path.join(output_dir, f"{file_name}_features.json")
            
            # Save features to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            # Store for summary
            all_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing document {doc.uuid if hasattr(doc, 'uuid') else 'unknown'}: {str(e)}")
    
    # Create summary if requested
    if summary_file and all_results:
        try:
            # Flatten results for a DataFrame
            flat_data = []
            
            for result in all_results:
                # Start with metadata
                entry = {k: v for k, v in result["metadata"].items()}
                
                # Add top-level features
                for key, value in result["features"].items():
                    # Handle primitive types directly
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        entry[f"feature_{key}"] = value
                    # Handle lists by counting items
                    elif isinstance(value, list):
                        entry[f"feature_{key}_count"] = len(value)
                    # Handle dictionaries by storing as string
                    elif isinstance(value, dict):
                        # Try to extract some common useful metrics if they exist
                        if key == "crypto_mentions" and value:
                            entry[f"feature_top_crypto"] = max(value.items(), key=lambda x: x[1])[0] if value else None
                        elif key == "risk_factors" and isinstance(value, list) and value:
                            entry[f"feature_top_risk"] = value[0]["category"] if value else None
                        else:
                            # Just count keys for other dicts
                            entry[f"feature_{key}_keys"] = len(value)
                
                flat_data.append(entry)
            
            # Convert to DataFrame
            df = pd.DataFrame(flat_data)
            
            # Save as CSV
            df.to_csv(summary_file, index=False)
            logger.info(f"Summary saved to {summary_file}")
            
            # Basic statistics
            logger.info(f"Processed {len(df)} documents successfully")
            logger.info(f"Document types: {df['document_type'].value_counts().to_dict()}")
            
            # Find common features
            feature_cols = [col for col in df.columns if col.startswith("feature_")]
            if feature_cols:
                logger.info(f"Extracted {len(feature_cols)} different features")
                
                # Find most common features
                non_null_counts = df[feature_cols].count()
                common_features = non_null_counts.sort_values(ascending=False).head(10)
                logger.info(f"Most common features:\n{common_features}")
                
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
    
    # Close the connection
    client.close()
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Extract features from documents stored in Weaviate")
    parser.add_argument("--collection", default="CryptoDueDiligenceDocuments", help="Weaviate collection name")
    parser.add_argument("--output_dir", default="extracted_features", help="Directory to save extracted features")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of documents to process")
    parser.add_argument("--summary", default="extraction_summary.csv", help="File to save summary of extracted features")
    
    args = parser.parse_args()
    
    logger.info(f"Starting feature extraction for documents in Weaviate collection: {args.collection}")
    process_weaviate_documents(args.output_dir, args.collection, args.limit, args.summary)
    logger.info("Feature extraction complete")

if __name__ == "__main__":
    main()