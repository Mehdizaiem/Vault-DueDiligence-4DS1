# File path: Code/document_processing/test_tfidf_enhancements.py
import logging
import os
import sys
from pathlib import Path

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Code.document_processing.tf_idf_processor import TFIDFProcessor
from Code.document_processing.crypto_document_processor import CryptoDocumentProcessor
from Code.document_processing.document_similarity import DocumentSimilarityService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tfidf_keyword_extraction():
    """Test TF-IDF keyword extraction"""
    # Sample crypto document
    sample_text = """
    Bitcoin is a decentralized digital currency, without a central bank or single administrator, 
    that can be sent from user to user on the peer-to-peer bitcoin network without the need for 
    intermediaries. Transactions are verified by network nodes through cryptography and recorded 
    in a public distributed ledger called a blockchain. The cryptocurrency was invented in 2008 
    by an unknown person or group of people using the name Satoshi Nakamoto. The currency began 
    use in 2009 when its implementation was released as open-source software.
    
    Bitcoins are created as a reward for a process known as mining. They can be exchanged for 
    other currencies, products, and services. Bitcoin has been criticized for its use in illegal 
    transactions, the large amount of electricity used by miners, price volatility, and thefts 
    from exchanges.
    """
    
    # Initialize TF-IDF processor
    tfidf_processor = TFIDFProcessor()
    
    # Update corpus with sample text
    tfidf_processor.update_corpus(sample_text)
    
    # Extract keywords
    keywords = tfidf_processor.extract_keywords(sample_text, max_keywords=10)
    
    # Print results
    print("\n----- TF-IDF Keyword Extraction -----")
    print(f"Extracted keywords: {keywords}")
    print("--------------------------------------\n")
    
    return keywords

def test_document_similarity():
    """Test document similarity calculation"""
    # Sample documents
    doc1 = """
    Ethereum is a decentralized, open-source blockchain with smart contract functionality. 
    Ether is the native cryptocurrency of the platform. After Bitcoin, it is the second-largest 
    cryptocurrency by market capitalization. The Ethereum network hosts many tokens including stablecoins.
    """
    
    doc2 = """
    Ethereum is an open-source platform that uses blockchain technology to create and run 
    decentralized digital applications enabling users to make agreements and conduct transactions 
    directly with each other to buy, sell and trade goods and services without a middle man.
    """
    
    doc3 = """
    The Securities and Exchange Commission (SEC) has issued new guidelines for cryptocurrency 
    tokens, outlining when they might be considered securities. This regulatory move has 
    significant implications for blockchain-based startups raising funds through token sales.
    """
    
    # Initialize document similarity service
    similarity_service = DocumentSimilarityService()
    
    # Compute similarities
    sim_1_2 = similarity_service.compute_document_similarity(doc1, doc2)
    sim_1_3 = similarity_service.compute_document_similarity(doc1, doc3)
    sim_2_3 = similarity_service.compute_document_similarity(doc2, doc3)
    
    # Find similar documents
    documents = [
        {"id": 1, "content": doc1, "title": "Ethereum Overview"},
        {"id": 2, "content": doc2, "title": "Ethereum Platform"},
        {"id": 3, "content": doc3, "title": "SEC Cryptocurrency Regulation"}
    ]
    
    similar_to_ethereum = similarity_service.find_similar_documents(doc1, documents)
    
    # Print results
    print("\n----- Document Similarity -----")
    print(f"Similarity between doc1 and doc2: {sim_1_2:.4f}")
    print(f"Similarity between doc1 and doc3: {sim_1_3:.4f}")
    print(f"Similarity between doc2 and doc3: {sim_2_3:.4f}")
    print("\nSimilar documents to Ethereum overview:")
    for doc in similar_to_ethereum:
        print(f"  - {doc['title']} (similarity: {doc['similarity_score']:.4f})")
    print("-------------------------------\n")
    
    return sim_1_2, sim_1_3, sim_2_3, similar_to_ethereum

def test_enhanced_document_processor():
    """Test enhanced document processor with TF-IDF"""
    # Sample document text
    sample_text = """
    Regulatory Compliance in DeFi Protocols
    
    Decentralized Finance (DeFi) protocols face unique regulatory challenges due to their borderless 
    nature. Various jurisdictions have begun implementing compliance frameworks that DeFi projects 
    must navigate. The Financial Action Task Force (FATF) has issued guidelines specifically targeting 
    virtual asset service providers (VASPs) which may apply to certain DeFi protocols.
    
    Key compliance considerations include:
    1. Anti-Money Laundering (AML) procedures
    2. Know Your Customer (KYC) requirements
    3. Securities regulations for tokens
    4. Tax reporting obligations
    
    Projects implementing strong compliance measures may gain competitive advantages through 
    increased institutional adoption. However, balancing regulatory compliance with the 
    decentralized ethos of DeFi remains a significant challenge for the industry.
    
    The SEC has specifically highlighted concerns regarding unregistered securities offerings 
    in the form of governance tokens. Meanwhile, the CFTC has focused on derivatives trading 
    within DeFi ecosystems.
    """
    
    # Initialize processor
    processor = CryptoDocumentProcessor()
    
    # Process document
    result = processor.process_document(sample_text, filename="regulatory_compliance.txt")
    
    # Print results
    print("\n----- Enhanced Document Processor -----")
    print(f"Document type: {result['document_type']}")
    print(f"TF-IDF Enhanced Keywords (main topics): {result['summary']['main_topics']}")
    print("\nKey insights:")
    for insight in result['summary']['key_insights']:
        print(f"  - {insight}")
    print("--------------------------------------\n")
    
    return result

def main():
    print("\n===== Testing TF-IDF Enhancements =====\n")
    
    # Test TF-IDF keyword extraction
    keywords = test_tfidf_keyword_extraction()
    
    # Test document similarity
    similarities = test_document_similarity()
    
    # Test enhanced document processor
    result = test_enhanced_document_processor()
    
    print("\n===== All Tests Completed =====\n")

if __name__ == "__main__":
    main()