# Sample_Data/onchain_analytics/analyze_wallet.py
import sys
import os
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("onchain_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Add path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyzers.wallet_analyzer import WalletAnalyzer
from models.weaviate_storage import store_wallet_analysis

def analyze_wallet(address: str, related_fund: str = None, store: bool = True):
    """Analyze a wallet address and optionally store results."""
    start_time = datetime.now()
    logger.info(f"Starting analysis of wallet: {address}")
    
    try:
        # Create analyzer
        analyzer = WalletAnalyzer()
        
        # Analyze wallet
        analysis = analyzer.analyze_ethereum_wallet(address)
        
        if "error" in analysis:
            logger.warning(f"Analysis error: {analysis['error']}")
            return analysis
        
        # Print summary
        analytics = analysis.get("analytics", {})
        risk = analysis.get("risk_assessment", {})
        
        logger.info(f"Analysis complete for {address}")
        logger.info(f"Transaction count: {analytics.get('transaction_count', 0)}")
        logger.info(f"Token transaction count: {analytics.get('token_transaction_count', 0)}")
        logger.info(f"Balance (ETH): {analytics.get('balance_eth', 0)}")
        logger.info(f"Risk score: {risk.get('risk_score', 0)}")
        logger.info(f"Risk level: {risk.get('risk_level', 'Unknown')}")
        
        # Store in Weaviate if requested
        if store:
            success = store_wallet_analysis(analysis, related_fund)
            if success:
                logger.info(f"Successfully stored analysis for {address} in Weaviate")
            else:
                logger.error(f"Failed to store analysis for {address}")
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"Total analysis time: {elapsed_time}")
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing wallet {address}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"address": address, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Analyze a blockchain wallet address")
    parser.add_argument("address", help="Wallet address to analyze")
    parser.add_argument("--fund", help="Related crypto fund name")
    parser.add_argument("--no-store", action="store_true", help="Skip storing in Weaviate")
    
    args = parser.parse_args()
    
    analyze_wallet(args.address, args.fund, not args.no_store)

if __name__ == "__main__":
    main()