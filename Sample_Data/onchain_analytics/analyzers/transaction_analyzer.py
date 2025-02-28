# Sample_Data/onchain_analytics/analyzers/transaction_analyzer.py
import sys
import os
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class TransactionAnalyzer:
    """Analyzer for blockchain transactions."""
    
    def analyze_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze a list of transactions to identify patterns and risks."""
        if not transactions:
            return {"error": "No transactions provided for analysis"}
        
        # Transaction volume analysis
        transaction_count = len(transactions)
        transaction_values = [float(tx.get("value", 0)) / 1e18 for tx in transactions]
        total_value = sum(transaction_values)
        avg_value = total_value / transaction_count if transaction_count > 0 else 0
        max_value = max(transaction_values) if transaction_values else 0
        
        # Temporal analysis
        timestamps = [int(tx.get("timeStamp", 0)) for tx in transactions if tx.get("timeStamp")]
        timestamps.sort()
        
        # Calculate transaction frequency
        if len(timestamps) > 1:
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_time_between_txs = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            max_inactive_period = max(time_diffs) if time_diffs else 0
        else:
            avg_time_between_txs = 0
            max_inactive_period = 0
        
        # Gas analysis
        gas_prices = [int(tx.get("gasPrice", 0)) for tx in transactions if tx.get("gasPrice")]
        avg_gas_price = sum(gas_prices) / len(gas_prices) if gas_prices else 0
        
        # Transaction success rate
        success_count = sum(1 for tx in transactions if tx.get("isError") == "0")
        error_count = transaction_count - success_count
        success_rate = (success_count / transaction_count) * 100 if transaction_count > 0 else 0
        
        # Transaction pattern detection
        patterns = self._detect_transaction_patterns(transactions)
        
        # Risk indicators
        risk_indicators = self._identify_risk_indicators(transactions)
        
        return {
            "transaction_stats": {
                "count": transaction_count,
                "total_value_eth": total_value,
                "average_value_eth": avg_value,
                "max_value_eth": max_value,
                "success_rate": success_rate,
                "error_count": error_count
            },
            "temporal_analysis": {
                "first_transaction": min(timestamps) if timestamps else None,
                "last_transaction": max(timestamps) if timestamps else None,
                "avg_time_between_txs_seconds": avg_time_between_txs,
                "max_inactive_period_days": max_inactive_period / 86400 if max_inactive_period else 0
            },
            "gas_analysis": {
                "avg_gas_price_wei": avg_gas_price
            },
            "patterns": patterns,
            "risk_indicators": risk_indicators
        }
    
    def _detect_transaction_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Detect common transaction patterns."""
        patterns = {}
        
        # Detect regular transfers (simple ETH sends)
        regular_transfers = [tx for tx in transactions if tx.get("input") == "0x"]
        patterns["regular_transfers_pct"] = (len(regular_transfers) / len(transactions)) * 100 if transactions else 0
        
        # Detect contract interactions
        contract_interactions = [tx for tx in transactions if tx.get("input") != "0x"]
        patterns["contract_interactions_pct"] = (len(contract_interactions) / len(transactions)) * 100 if transactions else 0
        
        # Detect repeated interactions with same address
        to_addresses = [tx.get("to", "").lower() for tx in transactions]
        address_counts = {}
        for addr in to_addresses:
            if addr:
                address_counts[addr] = address_counts.get(addr, 0) + 1
        
        # Find most common interaction
        most_common_address = max(address_counts.items(), key=lambda x: x[1]) if address_counts else (None, 0)
        patterns["most_common_recipient"] = most_common_address[0]
        patterns["most_common_recipient_count"] = most_common_address[1]
        
        return patterns
    
    def _identify_risk_indicators(self, transactions: List[Dict]) -> List[Dict]:
        """Identify potential risk indicators in transaction patterns."""
        risk_indicators = []
        
        # Check for large single transactions
        large_txs = [tx for tx in transactions if float(tx.get("value", 0)) / 1e18 > 10]  # >10 ETH
        if large_txs:
            risk_indicators.append({
                "type": "large_transactions",
                "description": f"Found {len(large_txs)} large transactions (>10 ETH)",
                "severity": "medium"
            })
        
        # Check for failed transactions
        failed_txs = [tx for tx in transactions if tx.get("isError") == "1"]
        failed_rate = (len(failed_txs) / len(transactions)) * 100 if transactions else 0
        if failed_rate > 20:
            risk_indicators.append({
                "type": "high_failure_rate",
                "description": f"High transaction failure rate: {failed_rate:.1f}%",
                "severity": "medium"
            })
        
        # Check for burst activity (many transactions in short period)
        timestamps = [int(tx.get("timeStamp", 0)) for tx in transactions if tx.get("timeStamp")]
        timestamps.sort()
        
        if len(timestamps) > 5:
            # Look for cases where 5+ transactions happen within an hour
            for i in range(len(timestamps) - 4):
                time_window = timestamps[i+4] - timestamps[i]
                if time_window < 3600:  # 1 hour in seconds
                    risk_indicators.append({
                        "type": "burst_activity",
                        "description": "5+ transactions within a 1-hour period",
                        "severity": "low"
                    })
                    break
        
        return risk_indicators