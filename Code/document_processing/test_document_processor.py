import os
import logging
import sys

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create processor
    processor = CryptoDocumentProcessor()
    
    # Test with a sample document
    sample_text = """
    This is a sample cryptocurrency agreement. 
    Bitcoin and Ethereum are mentioned as payment methods.
    The SEC has regulatory oversight over this transaction.
    """
    
    # Process the sample
    result = processor.process_document(sample_text, filename="test.txt")
    
    # Print results
    print("\n----- DOCUMENT PROCESSING TEST RESULTS -----")
    print(f"Document Type: {result.get('document_type')}")
    print(f"Entities Found: {len(result.get('entities', {}).get('persons', [])) + len(result.get('entities', {}).get('organizations', []))}")
    print(f"Cryptocurrencies: {result.get('entities', {}).get('cryptocurrencies', [])}")
    print(f"Regulatory Mentions: {result.get('regulatory_mentions', {}).get('regulatory_bodies', [])}")
    print(f"Key Sentences: {result.get('summary', {}).get('key_sentences', [])[:1]}")
    print("---------------------------------------------\n")
    
    logger.info("Document processor test complete")

if __name__ == "__main__":
    main()