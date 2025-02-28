# Sample_Data/onchain_analytics/analyzers/token_analyzer.py
import sys
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class TokenAnalyzer:
    """Analyzer for token holdings and transfers."""
    
    def analyze_token_transactions(self, token_transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze token transactions for patterns and portfolio composition."""
        if not token_transactions:
            return {"error": "No token transactions provided for analysis"}
        
        # Extract token data
        tokens = {}
        for tx in token_transactions:
            token_symbol = tx.get("tokenSymbol", "UNKNOWN")
            token_name = tx.get("tokenName", "Unknown Token")
            token_address = tx.get("contractAddress", "")
            token_decimals = int(tx.get("tokenDecimal", 18))
            
            # Extract transfer details
            tx_from = tx.get("from", "").lower()
            tx_to = tx.get("to", "").lower()
            value = float(tx.get("value", 0)) / (10 ** token_decimals)
            timestamp = int(tx.get("timeStamp", 0))
            
            # Initialize token data if not seen before
            if token_symbol not in tokens:
                tokens[token_symbol] = {
                    "name": token_name,
                    "address": token_address,
                    "decimals": token_decimals,
                    "transfers_in": 0,
                    "transfers_out": 0,
                    "volume_in": 0,
                    "volume_out": 0,
                    "first_seen": timestamp,
                    "last_seen": timestamp
                }
            
            # Update transfer counts and volumes
            token_data = tokens[token_symbol]
            
            # Update timestamps
            if timestamp < token_data["first_seen"]:
                token_data["first_seen"] = timestamp
            if timestamp > token_data["last_seen"]:
                token_data["last_seen"] = timestamp
            
            # Determine if this was an incoming or outgoing transfer
            # based on the address being analyzed (would need to be passed in)
            # For this example, we'll use a placeholder approach
            is_incoming = True  # Assume all are incoming for simplicity
            
            if is_incoming:
                token_data["transfers_in"] += 1
                token_data["volume_in"] += value
            else:
                token_data["transfers_out"] += 1
                token_data["volume_out"] += value
        
        # Generate token statistics
        token_count = len(tokens)
        tokens_by_transfers = sorted(tokens.items(), key=lambda x: x[1]["transfers_in"] + x[1]["transfers_out"], reverse=True)
        
        # Calculate holding period for each token
        for symbol, data in tokens.items():
            holding_time = data["last_seen"] - data["first_seen"]
            data["holding_days"] = holding_time / 86400  # Convert seconds to days
        
        # Identify token categories (if possible)
        token_categories = self._categorize_tokens(tokens)
        
        return {
            "token_count": token_count,
            "tokens": tokens,
            "most_active_tokens": [t[0] for t in tokens_by_transfers[:5]],
            "token_categories": token_categories,
            "transaction_count": len(token_transactions)
        }
    
    def _categorize_tokens(self, tokens: Dict) -> Dict[str, List[str]]:
        """Attempt to categorize tokens based on known symbols."""
        categories = {
            "stablecoins": [],
            "defi": [],
            "gaming": [],
            "exchange_tokens": [],
            "privacy": [],
            "other": []
        }
        
        # Known token categories (simplified)
        stablecoins = ["USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP"]
        defi_tokens = ["AAVE", "COMP", "UNI", "SUSHI", "YFI", "MKR", "SNX", "CRV"]
        gaming_tokens = ["MANA", "AXS", "SAND", "ENJ", "GALA"]
        exchange_tokens = ["BNB", "CRO", "FTT", "LEO", "HT", "KCS"]
        privacy_tokens = ["ZEC", "XMR", "DASH", "SCRT"]
        
        for symbol in tokens.keys():
            if symbol in stablecoins:
                categories["stablecoins"].append(symbol)
            elif symbol in defi_tokens:
                categories["defi"].append(symbol)
            elif symbol in gaming_tokens:
                categories["gaming"].append(symbol)
            elif symbol in exchange_tokens:
                categories["exchange_tokens"].append(symbol)
            elif symbol in privacy_tokens:
                categories["privacy"].append(symbol)
            else:
                categories["other"].append(symbol)
        
        return categories