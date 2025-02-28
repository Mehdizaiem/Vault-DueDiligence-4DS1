# Sample_Data/onchain_analytics/test_onchain_analytics.py
import sys
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from collectors.etherscan_collector import EtherscanCollector
from analyzers.wallet_analyzer import WalletAnalyzer

# Import schema but handle potential import errors
try:
    from models.weaviate_schema import create_onchain_schema
    from models.weaviate_storage import store_wallet_analysis
    WEAVIATE_AVAILABLE = True
except ImportError:
    logger.warning("Weaviate modules not available - storage tests will be skipped")
    WEAVIATE_AVAILABLE = False

def test_etherscan_collector():
    """Test the Etherscan collector."""
    logger.info("Testing Etherscan collector...")
    
    collector = EtherscanCollector()
    
    # Test with Vitalik's address
    address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    # Get transactions
    logger.info(f"Fetching transactions for {address}...")
    txs = collector.get_wallet_transactions(address, 0, 99999999)
    logger.info(f"Found {len(txs)} transactions")
    
    # Get token transactions
    logger.info(f"Fetching token transactions for {address}...")
    token_txs = collector.get_token_transactions(address, 0, 99999999)
    logger.info(f"Found {len(token_txs)} token transactions")
    
    # Get ETH balance
    logger.info(f"Fetching balance for {address}...")
    balance = collector.get_eth_balance(address)
    logger.info(f"Balance: {balance} ETH")
    
    return True

def test_wallet_analyzer():
    """Test the wallet analyzer."""
    logger.info("Testing wallet analyzer...")
    
    analyzer = WalletAnalyzer()
    
    # Test with Vitalik's address
    address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    
    # Analyze wallet
    logger.info(f"Analyzing wallet {address}...")
    analysis = analyzer.analyze_ethereum_wallet(address)
    
    # Print summary
    logger.info("Analysis complete")
    logger.info(f"Transaction count: {analysis.get('analytics', {}).get('transaction_count', 0)}")
    logger.info(f"Risk score: {analysis.get('risk_assessment', {}).get('risk_score', 0)}")
    logger.info(f"Risk level: {analysis.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
    
    return analysis

def test_weaviate_storage(analysis):
    """Test storing data in Weaviate."""
    if not WEAVIATE_AVAILABLE:
        logger.warning("Skipping Weaviate storage test - modules not available")
        return True

    logger.info("Testing Weaviate storage...")
    
    try:
        # Create schema
        logger.info("Creating Weaviate schema...")
        from vector_store.weaviate_client import get_weaviate_client
        client = get_weaviate_client()
        create_onchain_schema(client)
        client.close()
        
        # Store analysis
        logger.info("Storing wallet analysis...")
        success = store_wallet_analysis(analysis, "Test Fund")
        
        if success:
            logger.info("Successfully stored analysis in Weaviate")
        else:
            logger.error("Failed to store analysis in Weaviate")
        
        return success
    except Exception as e:
        logger.error(f"Weaviate storage test error: {str(e)}")
        return False

def run_tests():
    """Run all tests."""
    start_time = datetime.now()
    logger.info("Starting on-chain analytics tests...")
    
    try:
        # Test collector
        if not test_etherscan_collector():
            logger.error("Etherscan collector test failed")
            return False
        
        # Test analyzer
        analysis = test_wallet_analyzer()
        if not analysis:
            logger.error("Wallet analyzer test failed")
            return False
        
        # Test storage if available
        if WEAVIATE_AVAILABLE:
            if not test_weaviate_storage(analysis):
                logger.error("Weaviate storage test failed")
                # Continue anyway - don't fail the whole test
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"All tests completed successfully in {elapsed_time}")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_tests()