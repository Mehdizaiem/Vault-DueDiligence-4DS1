# Sample_Data/onchain_analytics/analyze_fund.py
import sys
import os
import argparse
import logging
import json
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fund_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Add path to allow imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyzers.wallet_analyzer import WalletAnalyzer
from models.weaviate_storage import store_wallet_analysis

def analyze_fund_wallets(fund_name: str, addresses: List[str], output_file: str = None, store: bool = True):
    """Analyze multiple wallet addresses associated with a fund."""
    start_time = datetime.now()
    logger.info(f"Starting analysis for fund: {fund_name} with {len(addresses)} addresses")
    
    analyzer = WalletAnalyzer()
    results = []
    
    for i, address in enumerate(addresses):
        logger.info(f"Analyzing address {i+1}/{len(addresses)}: {address}")
        
        # Analyze wallet
        analysis = analyzer.analyze_ethereum_wallet(address)
        
        # Store in Weaviate if requested
        if store and "error" not in analysis:
            success = store_wallet_analysis(analysis, fund_name)
            if success:
                logger.info(f"Stored analysis for {address} in Weaviate")
            else:
                logger.warning(f"Failed to store analysis for {address}")
        
        # Add to results
        results.append(analysis)
    
    # Generate fund-level summary
    summary = generate_fund_summary(fund_name, results)
    logger.info(f"Analysis summary for {fund_name}:")
    logger.info(f"Total addresses analyzed: {len(results)}")
    logger.info(f"Average risk score: {summary.get('average_risk_score', 0)}")
    logger.info(f"Fund risk level: {summary.get('fund_risk_level', 'Unknown')}")
    
    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "fund_name": fund_name,
                "summary": summary,
                "wallet_analyses": results
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    elapsed_time = datetime.now() - start_time
    logger.info(f"Total analysis time: {elapsed_time}")
    
    return summary, results

def generate_fund_summary(fund_name: str, analyses: List[Dict]) -> Dict[str, Any]:
    """Generate a summary of fund-level analytics from multiple wallet analyses."""
    valid_analyses = [a for a in analyses if "error" not in a]
    
    if not valid_analyses:
        return {
            "fund_name": fund_name,
            "error": "No valid wallet analyses available"
        }
    
    # Calculate aggregate metrics
    total_tx_count = sum(a.get("analytics", {}).get("transaction_count", 0) for a in valid_analyses)
    total_token_tx_count = sum(a.get("analytics", {}).get("token_transaction_count", 0) for a in valid_analyses)
    total_balance = sum(a.get("analytics", {}).get("balance_eth", 0) for a in valid_analyses)
    
    # Calculate average risk score
    risk_scores = [a.get("risk_assessment", {}).get("risk_score", 50) for a in valid_analyses]
    avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 50
    
    # Determine fund risk level
    if avg_risk_score < 20:
        fund_risk_level = "Very Low"
    elif avg_risk_score < 40:
        fund_risk_level = "Low"
    elif avg_risk_score < 60:
        fund_risk_level = "Medium"
    elif avg_risk_score < 80:
        fund_risk_level = "High"
    else:
        fund_risk_level = "Very High"
    
    # Collect all tokens from all wallets
    all_tokens = {}
    for analysis in valid_analyses:
        analytics = analysis.get("analytics", {})
        tokens = analytics.get("tokens", {})
        
        for symbol, data in tokens.items():
            if symbol not in all_tokens:
                all_tokens[symbol] = {
                    "name": data.get("name", "Unknown"),
                    "address": data.get("address", ""),
                    "count": 0
                }
            all_tokens[symbol]["count"] += 1
    
    # Find most common tokens
    common_tokens = sorted(all_tokens.items(), key=lambda x: x[1]["count"], reverse=True)
    top_tokens = [token[0] for token in common_tokens[:5]] if common_tokens else []
    
    return {
        "fund_name": fund_name,
        "wallet_count": len(valid_analyses),
        "total_transaction_count": total_tx_count,
        "total_token_transaction_count": total_token_tx_count,
        "total_balance_eth": total_balance,
        "average_risk_score": avg_risk_score,
        "fund_risk_level": fund_risk_level,
        "top_tokens": top_tokens,
        "token_count": len(all_tokens),
        "analysis_timestamp": datetime.now().isoformat()
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze multiple wallet addresses for a crypto fund")
    parser.add_argument("fund", help="Fund name")
    parser.add_argument("--addresses", "-a", nargs="+", help="List of wallet addresses")
    parser.add_argument("--file", "-f", help="File containing wallet addresses (one per line)")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--no-store", action="store_true", help="Skip storing in Weaviate")
    
    args = parser.parse_args()
    
    # Get addresses from arguments or file
    addresses = []
    if args.addresses:
        addresses = args.addresses
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                addresses = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading addresses file: {str(e)}")
            sys.exit(1)
    else:
        logger.error("Either --addresses or --file must be provided")
        sys.exit(1)
    
    # Run analysis
    analyze_fund_wallets(args.fund, addresses, args.output, not args.no_store)

if __name__ == "__main__":
    main()