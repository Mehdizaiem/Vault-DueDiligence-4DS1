# Sample_Data/onchain_analytics/analyzers/wallet_analyzer.py
import sys
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

# Add path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from collectors.etherscan_collector import EtherscanCollector

logger = logging.getLogger(__name__)

class WalletAnalyzer:
    """Analyzer for wallet addresses."""
    
    def __init__(self):
        self.etherscan = EtherscanCollector()
    
    def analyze_ethereum_wallet(self, address: str) -> Dict[str, Any]:
        """Perform comprehensive analysis on an Ethereum wallet."""
        logger.info(f"Analyzing Ethereum wallet: {address}")
        
        # Get transaction data
        transactions = self.etherscan.get_wallet_transactions(address)
        token_transactions = self.etherscan.get_token_transactions(address)
        internal_transactions = self.etherscan.get_internal_transactions(address)
        eth_balance = self.etherscan.get_eth_balance(address)
        
        # Calculate basic metrics
        tx_count = len(transactions)
        token_tx_count = len(token_transactions)
        internal_tx_count = len(internal_transactions)
        
        if tx_count == 0:
            logger.warning(f"No transactions found for address: {address}")
            return {
                "address": address,
                "blockchain": "ethereum",
                "error": "No transactions found"
            }
        
        # Calculate value metrics (in ETH)
        total_received = 0
        total_sent = 0
        
        for tx in transactions:
            value = float(tx.get("value", 0)) / 1e18  # Convert wei to ETH
            if tx.get("to", "").lower() == address.lower():
                total_received += value
            elif tx.get("from", "").lower() == address.lower():
                total_sent += value
        
        # Calculate time-based metrics
        timestamps = [int(tx.get("timeStamp", 0)) for tx in transactions if tx.get("timeStamp")]
        first_tx_time = min(timestamps) if timestamps else 0
        latest_tx_time = max(timestamps) if timestamps else 0
        
        account_age_days = 0
        if first_tx_time > 0 and latest_tx_time > 0:
            account_age_days = (latest_tx_time - first_tx_time) / 86400  # seconds to days
        
        # Identify unique interactions
        interacted_addresses = set()
        for tx in transactions:
            if tx.get("from", "").lower() != address.lower():
                interacted_addresses.add(tx.get("from", "").lower())
            if tx.get("to", "").lower() != address.lower():
                interacted_addresses.add(tx.get("to", "").lower())
        
        # Contract interactions
        contract_interactions = []
        for tx in transactions:
            if tx.get("input", "0x") != "0x" and tx.get("from", "").lower() == address.lower():
                contract_interactions.append(tx.get("to", "").lower())
        
        unique_contracts = list(set(contract_interactions))
        
        # Token analysis
        tokens = {}
        for tx in token_transactions:
            token_symbol = tx.get("tokenSymbol", "UNKNOWN")
            token_name = tx.get("tokenName", "Unknown Token")
            token_address = tx.get("contractAddress", "")
            
            if token_symbol not in tokens:
                tokens[token_symbol] = {
                    "name": token_name,
                    "address": token_address,
                    "transfers": 0
                }
            
            tokens[token_symbol]["transfers"] += 1
        
        # Calculate risk score
        metrics = {
            "tx_count": tx_count,
            "account_age_days": account_age_days,
            "contract_interactions": len(unique_contracts),
            "unique_interactions": len(interacted_addresses),
            "token_variety": len(tokens)
        }
        risk_score = self._calculate_risk_score(metrics)
        risk_level = self._get_risk_level(risk_score)
        risk_factors = self._identify_risk_factors(metrics)
        
        logger.info(f"Analysis complete for {address} - Risk Score: {risk_score}, Level: {risk_level}")
        
        return {
            "address": address,
            "blockchain": "ethereum",
            "analysis_timestamp": datetime.now().isoformat(),
            "analytics": {
                "transaction_count": tx_count,
                "token_transaction_count": token_tx_count,
                "internal_transaction_count": internal_tx_count,
                "total_received_eth": total_received,
                "total_sent_eth": total_sent,
                "balance_eth": eth_balance,
                "first_transaction_timestamp": first_tx_time,
                "latest_transaction_timestamp": latest_tx_time,
                "account_age_days": account_age_days,
                "unique_interactions_count": len(interacted_addresses),
                "contract_interaction_count": len(contract_interactions),
                "unique_contracts_count": len(unique_contracts),
                "token_types_count": len(tokens),
                "tokens": tokens
            },
            "risk_assessment": {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors
            }
        }
    
    def _calculate_risk_score(self, metrics: Dict) -> float:
        """Calculate risk score (0-100) based on wallet metrics."""
        score = 50  # Start at medium risk
        
        # Age factor (newer = higher risk)
        age_days = metrics.get("account_age_days", 0)
        if age_days < 30:
            score += 20
        elif age_days < 180:
            score += 10
        elif age_days > 365:
            score -= 10
        
        # Transaction count (very low or very high can be risk signals)
        tx_count = metrics.get("tx_count", 0)
        if tx_count < 5:
            score += 15  # Very new account
        elif tx_count > 1000:
            score += 5   # Unusually high activity
        
        # Contract interaction diversity
        contract_count = metrics.get("contract_interactions", 0)
        if contract_count == 0:
            score -= 5   # No contract interactions
        elif contract_count > 20:
            score -= 10  # Diverse usage is generally lower risk
        
        # Network diversity
        unique_interactions = metrics.get("unique_interactions", 0)
        if unique_interactions < 3:
            score += 15  # Limited network suggests potential for isolation
        elif unique_interactions > 50:
            score -= 10  # Wide network is generally lower risk
        
        # Token diversity
        token_variety = metrics.get("token_variety", 0)
        if token_variety > 10:
            score -= 5   # More token types suggests established user
        
        # Cap the score between 0-100
        return max(0, min(100, score))
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to categorical risk level."""
        if risk_score < 20:
            return "Very Low"
        elif risk_score < 40:
            return "Low"
        elif risk_score < 60:
            return "Medium"
        elif risk_score < 80:
            return "High"
        else:
            return "Very High"
    
    def _identify_risk_factors(self, metrics: Dict) -> List[str]:
        """Identify specific risk factors based on metrics."""
        factors = []
        
        if metrics.get("account_age_days", 0) < 30:
            factors.append("Recently created wallet")
            
        if metrics.get("tx_count", 0) < 5:
            factors.append("Very few transactions")
            
        if metrics.get("unique_interactions", 0) < 3:
            factors.append("Limited interaction network")
            
        if metrics.get("contract_interactions", 0) == 0:
            factors.append("No interaction with smart contracts")
            
        return factors