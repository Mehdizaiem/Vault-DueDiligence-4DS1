# Sample_Data/onchain_analytics/utils.py
import os
import sys
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def format_blockchain_address(address: str) -> str:
    """Format a blockchain address (lowercase and remove any spaces)."""
    if not address:
        return ""
    return address.lower().strip()

def wei_to_eth(wei_value: int) -> float:
    """Convert wei to ETH."""
    try:
        return float(wei_value) / 1e18
    except (ValueError, TypeError):
        return 0.0

def timestamp_to_date(timestamp: int) -> str:
    """Convert a Unix timestamp to a human-readable date string."""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return "Unknown date"

def get_token_metadata(symbol: str) -> Dict[str, Any]:
    """Get metadata for a token symbol."""
    # This could be expanded with a more comprehensive database
    # For now, just return some basic information for common tokens
    tokens = {
        "ETH": {"name": "Ethereum", "type": "native", "category": "layer1"},
        "USDT": {"name": "Tether", "type": "erc20", "category": "stablecoin"},
        "USDC": {"name": "USD Coin", "type": "erc20", "category": "stablecoin"},
        "DAI": {"name": "Dai", "type": "erc20", "category": "stablecoin"},
        "UNI": {"name": "Uniswap", "type": "erc20", "category": "defi"},
        "AAVE": {"name": "Aave", "type": "erc20", "category": "defi"},
        "LINK": {"name": "Chainlink", "type": "erc20", "category": "oracle"},
        "WBTC": {"name": "Wrapped Bitcoin", "type": "erc20", "category": "wrapped"},
    }
    
    return tokens.get(symbol.upper(), {"name": f"Unknown ({symbol})", "type": "unknown", "category": "other"})

def categorize_transaction(tx: Dict) -> str:
    """Categorize a transaction based on its properties."""
    # Simple categorization logic
    if tx.get("input", "0x") == "0x":
        return "transfer"
    
    # Some common function signatures
    if tx.get("input", "").startswith("0xa9059cbb"):
        return "erc20_transfer"
    elif tx.get("input", "").startswith("0x095ea7b3"):
        return "erc20_approve"
    elif tx.get("input", "").startswith("0x2e1a7d4d"):
        return "withdraw"
    elif tx.get("input", "").startswith("0x3593564c"):
        return "execute_trade"
    
    return "contract_interaction"

def calculate_risk_factors(data: Dict) -> List[Dict]:
    """Calculate risk factors from analysis data."""
    risk_factors = []
    
    # Check for large transactions
    if data.get("max_value_eth", 0) > 100:
        risk_factors.append({
            "factor": "large_transaction",
            "description": f"Large transaction of {data.get('max_value_eth')} ETH detected",
            "severity": "medium"
        })
    
    # Check for high token diversity
    if data.get("token_count", 0) > 50:
        risk_factors.append({
            "factor": "high_token_diversity",
            "description": f"Unusually diverse token portfolio with {data.get('token_count')} tokens",
            "severity": "low"
        })
    
    # Check for burst activity
    if data.get("burst_activity", False):
        risk_factors.append({
            "factor": "burst_activity",
            "description": "Periods of unusually high transaction frequency detected",
            "severity": "medium"
        })
    
    return risk_factors