import sys
import os
import logging
import time
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("onchain_analytics.log")
    ]
)
logger = logging.getLogger(__name__)

# Add Sample_Data to path (similar to etl_pipeline.py)
SAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(SAMPLE_DATA_PATH)

# Import collectors and analyzers
from Sample_Data.onchain_analytics.collectors.etherscan_collector import EtherscanCollector
from Sample_Data.onchain_analytics.collectors.santiment_collector import SantimentCollector
from Sample_Data.onchain_analytics.analyzers.wallet_analyzer import WalletAnalyzer
from Sample_Data.onchain_analytics.analyzers.transaction_analyzer import TransactionAnalyzer
from Sample_Data.onchain_analytics.analyzers.token_analyzer import TokenAnalyzer

# Import Weaviate modules
try:
    from Sample_Data.onchain_analytics.models.weaviate_schema import create_onchain_schema
    from Sample_Data.onchain_analytics.models.weaviate_storage import store_wallet_analysis
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client  # Explicit import
    WEAVIATE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Weaviate modules not available - storage tests will be skipped: {str(e)}")
    WEAVIATE_AVAILABLE = False


def test_etherscan_collector():
    """Test the Etherscan collector with a known working address."""
    logger.info("\n===== Testing Etherscan Collector =====")
    
    collector = EtherscanCollector()
    
    # Use the known working Binance address that we've confirmed works
    address = "0x28c6c06298d514db089934071355e5743bf21d60"
    
    try:
        # Get normal transactions
        logger.info(f"Fetching transactions for {address}...")
        txs = collector.get_wallet_transactions(address, page=1, offset=5)
        logger.info(f"Found {len(txs)} transactions")
        
        if txs:
            # Print details of first transaction
            tx = txs[0]
            logger.info(f"Sample transaction:")
            logger.info(f"  Hash: {tx.get('hash')}")
            logger.info(f"  From: {tx.get('from')}")
            logger.info(f"  To: {tx.get('to')}")
            logger.info(f"  Value: {float(tx.get('value', 0)) / 1e18} ETH")
            
            # Save example transaction
            with open("sample_tx.json", "w") as f:
                json.dump(tx, f, indent=2)
        
        # Pause to avoid rate limiting
        time.sleep(1)
        
        # Get token transactions
        logger.info(f"Fetching token transactions for {address}...")
        token_txs = collector.get_token_transactions(address, page=1, offset=5)
        logger.info(f"Found {len(token_txs)} token transactions")
        
        if token_txs:
            # Print details of first token transaction
            token_tx = token_txs[0]
            logger.info(f"Sample token transaction:")
            logger.info(f"  Token: {token_tx.get('tokenSymbol')} ({token_tx.get('tokenName')})")
            logger.info(f"  Value: {float(token_tx.get('value', 0)) / (10 ** int(token_tx.get('tokenDecimal', 18)))} {token_tx.get('tokenSymbol')}")
        
        # Pause to avoid rate limiting
        time.sleep(1)
        
        # Get ETH balance
        logger.info(f"Fetching balance for {address}...")
        balance = collector.get_eth_balance(address)
        logger.info(f"Balance: {balance} ETH")
        
        return True, address, txs, token_txs
        
    except Exception as e:
        logger.error(f"Etherscan collector test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, address, [], []

def test_santiment_collector():
    """Test the Santiment collector if API key is available."""
    logger.info("\n===== Testing Santiment Collector =====")
    
    try:
        collector = SantimentCollector()
        
        if not collector.api_key:
            logger.warning("No Santiment API key found. Skipping test.")
            return False
        
        # Test with Ethereum
        logger.info("Fetching on-chain metrics for Ethereum...")
        metrics = collector.get_token_metrics("ethereum")
        
        if metrics:
            logger.info("Successfully retrieved Santiment metrics")
            logger.info(f"Data: {metrics}")
            return True
        else:
            logger.warning("No metrics retrieved from Santiment")
            return False
            
    except Exception as e:
        logger.error(f"Santiment collector test failed: {str(e)}")
        return False

def test_transaction_analyzer(transactions):
    """Test the transaction analyzer with real transaction data."""
    logger.info("\n===== Testing Transaction Analyzer =====")
    
    if not transactions:
        logger.warning("No transactions available for analysis. Skipping test.")
        return False
    
    try:
        analyzer = TransactionAnalyzer()
        
        # Limit to 20 transactions for testing
        test_txs = transactions[:20] if len(transactions) > 20 else transactions
        
        logger.info(f"Analyzing {len(test_txs)} transactions...")
        analysis = analyzer.analyze_transactions(test_txs)
        
        if analysis:
            logger.info("Transaction analysis successful")
            logger.info(f"Transaction stats: {analysis.get('transaction_stats', {})}")
            
            if 'patterns' in analysis:
                logger.info(f"Transaction patterns: {analysis.get('patterns', {})}")
            
            if 'risk_indicators' in analysis and analysis['risk_indicators']:
                logger.info(f"Risk indicators: {len(analysis['risk_indicators'])}")
                for indicator in analysis['risk_indicators']:
                    logger.info(f"  {indicator.get('type')}: {indicator.get('description')} (Severity: {indicator.get('severity')})")
            
            return True
        else:
            logger.warning("Transaction analysis failed or returned empty result")
            return False
            
    except Exception as e:
        logger.error(f"Transaction analyzer test failed: {str(e)}")
        return False

def test_token_analyzer(token_transactions):
    """Test the token analyzer with real token transaction data."""
    logger.info("\n===== Testing Token Analyzer =====")
    
    if not token_transactions:
        logger.warning("No token transactions available for analysis. Skipping test.")
        return False
    
    try:
        analyzer = TokenAnalyzer()
        
        # Limit to 20 token transactions for testing
        test_token_txs = token_transactions[:20] if len(token_transactions) > 20 else token_transactions
        
        logger.info(f"Analyzing {len(test_token_txs)} token transactions...")
        analysis = analyzer.analyze_token_transactions(test_token_txs)
        
        if analysis and 'tokens' in analysis:
            logger.info("Token analysis successful")
            logger.info(f"Token count: {analysis.get('token_count', 0)}")
            
            if 'most_active_tokens' in analysis:
                logger.info(f"Most active tokens: {analysis.get('most_active_tokens', [])}")
            
            if 'token_categories' in analysis:
                logger.info(f"Token categories: {analysis.get('token_categories', {})}")
            
            return True
        else:
            logger.warning("Token analysis failed or returned empty result")
            return False
            
    except Exception as e:
        logger.error(f"Token analyzer test failed: {str(e)}")
        return False

def test_wallet_analyzer(address, transactions, token_transactions):
    """Test the wallet analyzer with a real address and transaction data."""
    logger.info("\n===== Testing Wallet Analyzer =====")
    
    if not address:
        logger.warning("No address available for analysis. Skipping test.")
        return False
    
    try:
        analyzer = WalletAnalyzer()
        
        # Test the analyzer in two ways:
        # 1. With the full API-based analysis
        logger.info(f"Performing full analysis of wallet {address}...")
        start_time = time.time()
        analysis = analyzer.analyze_ethereum_wallet(address)
        elapsed = time.time() - start_time
        
        if "error" in analysis:
            logger.warning(f"Full wallet analysis encountered an error: {analysis.get('error')}")
            
            # 2. If full analysis fails, try with just the transactions we already have
            logger.info("Testing with pre-fetched transaction data...")
            
            # If we have Etherscan data but the analyzer couldn't fetch it directly
            # This could happen due to rate limiting during testing
            if transactions or token_transactions:
                # We would need to modify the analyzer to accept pre-fetched transactions
                # This is a placeholder for that functionality
                logger.info("Wallet analyzer doesn't currently support pre-fetched transactions.")
                
            return False
        else:
            logger.info(f"Wallet analysis completed in {elapsed:.2f} seconds")
            
            # Log the analysis results
            risk_assessment = analysis.get('risk_assessment', {})
            analytics = analysis.get('analytics', {})
            
            logger.info(f"Risk score: {risk_assessment.get('risk_score', 0)}")
            logger.info(f"Risk level: {risk_assessment.get('risk_level', 'Unknown')}")
            
            if 'risk_factors' in risk_assessment and risk_assessment['risk_factors']:
                logger.info(f"Risk factors: {risk_assessment['risk_factors']}")
            
            logger.info(f"Transaction count: {analytics.get('transaction_count', 0)}")
            logger.info(f"Token transaction count: {analytics.get('token_transaction_count', 0)}")
            logger.info(f"Balance (ETH): {analytics.get('balance_eth', 0)}")
            
            if 'tokens' in analytics and analytics['tokens']:
                logger.info(f"Tokens: {list(analytics['tokens'].keys())}")
            
            # Save the analysis to a file for review
            with open("wallet_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
                
            logger.info("Wallet analysis saved to wallet_analysis.json")
            
            return True, analysis
            
    except Exception as e:
        logger.error(f"Wallet analyzer test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, None

def test_weaviate_storage(analysis):
    """Test storing analysis data in Weaviate with detailed logging."""
    logger.info("\n===== Testing Weaviate Storage =====")
    
    if not WEAVIATE_AVAILABLE:
        logger.warning("Weaviate modules not available - storage test skipped")
        logger.info("To enable Weaviate storage:")
        logger.info("1. Make sure Weaviate Docker container is running")
        logger.info("2. Check that the vector_store module is accessible")
        return False
    
    if not analysis or "error" in analysis:
        logger.warning("No valid analysis data to store in Weaviate")
        return False
    
    try:
        # Add parent directories to path to find vector_store
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        logger.info(f"Adding to path: {parent_path}")
        sys.path.append(parent_path)
        
        # Check what modules are available in the path
        logger.info("Checking for vector_store module...")
        try:
            import Sample_Data.vector_store
            logger.info("✅ Successfully imported Sample_Data.vector_store")
        except ImportError as e:
            logger.error(f"❌ Failed to import Sample_Data.vector_store: {str(e)}")
            logger.info("Available modules in Sample_Data:")
            
            # List available modules in Sample_Data
            sample_data_path = os.path.join(parent_path, "Sample_Data")
            if os.path.exists(sample_data_path):
                modules = [d for d in os.listdir(sample_data_path) 
                          if os.path.isdir(os.path.join(sample_data_path, d)) and not d.startswith('__')]
                logger.info(f"Modules found: {modules}")
            else:
                logger.error(f"Sample_Data directory not found at {sample_data_path}")
        
        # Try to import the weaviate client
        logger.info("Importing weaviate_client...")
        try:
            from Sample_Data.vector_store.weaviate_client import get_weaviate_client
            logger.info("✅ Successfully imported get_weaviate_client")
        except ImportError as e:
            logger.error(f"❌ Failed to import weaviate_client: {str(e)}")
            # Show the vector_store directory contents
            vector_store_path = os.path.join(parent_path, "Sample_Data", "vector_store")
            if os.path.exists(vector_store_path):
                files = os.listdir(vector_store_path)
                logger.info(f"Files in vector_store: {files}")
            return False
        
        # Try to create schema
        logger.info("Creating Weaviate schema...")
        try:
            # Get the Weaviate client
            logger.info("Initializing Weaviate client...")
            client = get_weaviate_client()
            
            # Check if the client is connected
            logger.info("Testing Weaviate connection...")
            if hasattr(client, "is_live") and callable(client.is_live):
                is_live = client.is_live()
                logger.info(f"Weaviate connection status: {'Live' if is_live else 'Not Live'}")
                
                if not is_live:
                    logger.error("Weaviate is not running or not accessible")
                    logger.info("Make sure Docker container is running with: docker-compose up -d")
                    client.close()
                    return False
            else:
                logger.warning("Client does not have is_live method, skipping connection check")
            
            # Create the schema
            logger.info("Importing create_onchain_schema...")
            from models.weaviate_schema import create_onchain_schema
            
            logger.info("Creating OnChainAnalytics schema...")
            schema_created = create_onchain_schema(client)
            
            if schema_created:
                logger.info("✅ Schema created successfully")
            else:
                logger.error("❌ Failed to create schema")
                client.close()
                return False
                
            # Close the client after creating schema
            client.close()
            logger.info("Weaviate client closed after schema creation")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        # Store analysis
        logger.info("Storing wallet analysis in Weaviate...")
        try:
            # Print analysis size for debugging
            analysis_size = len(json.dumps(analysis))
            logger.info(f"Analysis size: {analysis_size} bytes")
            
            # Store the analysis
            success = store_wallet_analysis(analysis, "Test Fund")
            
            if success:
                logger.info("✅ Successfully stored analysis in Weaviate")
                return True
            else:
                logger.error("❌ Failed to store analysis in Weaviate")
                return False
        except Exception as e:
            logger.error(f"Error in store_wallet_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Weaviate storage test failed with unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_tests():
    """Run all onchain analytics tests."""
    start_time = datetime.now()
    logger.info("Starting onchain analytics test suite...")
    
    results = {
        "etherscan": False,
        "santiment": False,
        "transaction_analyzer": False,
        "token_analyzer": False,
        "wallet_analyzer": False,
        "weaviate_storage": False
    }
    
    try:
        # Test Etherscan collector
        etherscan_result, address, transactions, token_transactions = test_etherscan_collector()
        results["etherscan"] = etherscan_result
        
        # Pause to avoid rate limiting
        time.sleep(1)
        
        # Test Santiment collector (if available)
        results["santiment"] = test_santiment_collector()
        
        # Test Transaction analyzer (if we have transactions)
        if transactions:
            results["transaction_analyzer"] = test_transaction_analyzer(transactions)
            
            # Test Token analyzer (if we have token transactions)
            if token_transactions:
                results["token_analyzer"] = test_token_analyzer(token_transactions)
            
            # Pause to avoid rate limiting
            time.sleep(1)
            
            # Test Wallet analyzer with the address and transactions
            wallet_result, analysis = test_wallet_analyzer(address, transactions, token_transactions)
            results["wallet_analyzer"] = wallet_result
            
            # Test Weaviate storage if wallet analysis succeeded
            if wallet_result and analysis:
                results["weaviate_storage"] = test_weaviate_storage(analysis)
        
        # Print summary
        logger.info("\n===== Test Results Summary =====")
        for test, result in results.items():
            logger.info(f"{test}: {'✅ PASSED' if result else '❌ FAILED'}")
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"All tests completed in {elapsed_time}")
        
        overall_success = any(results.values())  # Consider success if at least one test passed
        return overall_success
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_tests()