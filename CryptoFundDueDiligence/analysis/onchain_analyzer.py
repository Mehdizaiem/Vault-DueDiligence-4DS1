"""
OnChain Analyzer Module

This module analyzes on-chain data for Ethereum wallets mentioned in fund documents.
It retrieves transaction history, balance information, and risk metrics from OnChainAnalytics.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnChainAnalyzer:
    """
    Analyzes on-chain data for Ethereum wallets mentioned in fund documents.
    """
    
    def __init__(self, data_retriever):
        """
        Initialize the on-chain analyzer.
        
        Args:
            data_retriever: Object that retrieves data from collections
        """
        self.retriever = data_retriever
        
    def analyze_wallets(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Analyze on-chain data for specified wallet addresses.
        
        Args:
            wallet_addresses: List of Ethereum addresses
            
        Returns:
            Dict with wallet analysis data
        """
        logger.info(f"Analyzing {len(wallet_addresses)} wallet addresses")
        
        # Initialize result structure
        wallet_analysis = {
            "wallets": {},
            "aggregate_stats": {
                "total_balance_eth": 0.0,
                "total_transaction_count": 0,
                "average_risk_score": 0.0,
                "high_risk_wallets": 0,
                "medium_risk_wallets": 0,
                "low_risk_wallets": 0
            },
            "risk_assessment": {
                "overall_risk_score": 0.0,
                "risk_factors": [],
                "risk_level": "Unknown"
            },
            "analysis_date": datetime.now().isoformat()
        }
        
        total_risk_score = 0.0
        analyzed_count = 0
        
        # Analyze each wallet address
        for address in wallet_addresses:
            # Get on-chain data
            wallet_data = self._get_wallet_data(address)
            
            if wallet_data:
                # Add to wallets dictionary
                wallet_analysis["wallets"][address] = wallet_data
                
                # Update aggregate stats
                wallet_analysis["aggregate_stats"]["total_balance_eth"] += wallet_data.get("balance", 0.0)
                wallet_analysis["aggregate_stats"]["total_transaction_count"] += wallet_data.get("transaction_count", 0)
                
                # Update risk counts
                risk_level = wallet_data.get("risk_level", "Unknown").lower()
                if risk_level == "high":
                    wallet_analysis["aggregate_stats"]["high_risk_wallets"] += 1
                elif risk_level == "medium":
                    wallet_analysis["aggregate_stats"]["medium_risk_wallets"] += 1
                elif risk_level == "low":
                    wallet_analysis["aggregate_stats"]["low_risk_wallets"] += 1
                
                # Update total risk score for average calculation
                risk_score = wallet_data.get("risk_score", 0.0)
                total_risk_score += risk_score
                analyzed_count += 1
                
                # Collect risk factors
                if wallet_data.get("risk_factors"):
                    for factor in wallet_data.get("risk_factors", []):
                        if factor not in wallet_analysis["risk_assessment"]["risk_factors"]:
                            wallet_analysis["risk_assessment"]["risk_factors"].append(factor)
        
        # Calculate average risk score if we analyzed any wallets
        if analyzed_count > 0:
            wallet_analysis["aggregate_stats"]["average_risk_score"] = total_risk_score / analyzed_count
            
            # Set overall risk score and level
            wallet_analysis["risk_assessment"]["overall_risk_score"] = wallet_analysis["aggregate_stats"]["average_risk_score"]
            wallet_analysis["risk_assessment"]["risk_level"] = self._determine_risk_level(wallet_analysis["risk_assessment"]["overall_risk_score"])
        
        # Add assessment of wallet diversification
        wallet_analysis["risk_assessment"]["wallet_diversification"] = self._assess_wallet_diversification(wallet_analysis["wallets"])
        
        # Add assessment of transaction patterns
        wallet_analysis["risk_assessment"]["transaction_patterns"] = self._assess_transaction_patterns(wallet_analysis["wallets"])
        
        return wallet_analysis
    
    def _get_wallet_data(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Get on-chain data for a specific wallet address.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            Dict with wallet data or None if not found
        """
        try:
            # Get wallet data from on-chain analytics
            on_chain_data = self.retriever.get_onchain_analytics(address)
            
            if not on_chain_data:
                logger.warning(f"No on-chain data found for address: {address}")
                return None
            
            # Process and structure the data
            wallet_data = {
                "address": address,
                "blockchain": on_chain_data.get("blockchain", "ethereum"),
                "entity_type": on_chain_data.get("entity_type", "wallet"),
                "balance": on_chain_data.get("balance", 0.0),
                "transaction_count": on_chain_data.get("transaction_count", 0),
                "token_transaction_count": on_chain_data.get("token_transaction_count", 0),
                "total_received": on_chain_data.get("total_received", 0.0),
                "total_sent": on_chain_data.get("total_sent", 0.0),
                "first_activity": on_chain_data.get("first_activity"),
                "last_activity": on_chain_data.get("last_activity"),
                "active_days": on_chain_data.get("active_days", 0),
                "unique_interactions": on_chain_data.get("unique_interactions", 0),
                "contract_interactions": on_chain_data.get("contract_interactions", 0),
                "tokens": on_chain_data.get("tokens", []),
                "risk_score": on_chain_data.get("risk_score", 50.0),
                "risk_level": on_chain_data.get("risk_level", "Medium"),
                "risk_factors": on_chain_data.get("risk_factors", [])
            }
            
            # Calculate additional metrics
            # Wallet age in days
            if wallet_data.get("first_activity") and wallet_data.get("last_activity"):
                try:
                    first_date = datetime.fromisoformat(wallet_data["first_activity"].replace('Z', '+00:00'))
                    last_date = datetime.fromisoformat(wallet_data["last_activity"].replace('Z', '+00:00'))
                    wallet_data["age_days"] = (last_date - first_date).days
                except (ValueError, TypeError):
                    wallet_data["age_days"] = wallet_data.get("active_days", 0)
            else:
                wallet_data["age_days"] = wallet_data.get("active_days", 0)
            
            # Activity frequency (transactions per active day)
            if wallet_data["active_days"] > 0:
                wallet_data["activity_frequency"] = wallet_data["transaction_count"] / wallet_data["active_days"]
            else:
                wallet_data["activity_frequency"] = 0
            
            # Token diversity (number of different tokens held)
            wallet_data["token_diversity"] = len(wallet_data["tokens"])
            
            # Enriched activity classification
            wallet_data["activity_level"] = self._classify_activity_level(wallet_data)
            
            logger.info(f"Retrieved and processed on-chain data for {address}")
            return wallet_data
            
        except Exception as e:
            logger.error(f"Error retrieving on-chain data for {address}: {e}")
            return None
    
    def _classify_activity_level(self, wallet_data: Dict[str, Any]) -> str:
        """
        Classify the activity level of a wallet.
        
        Args:
            wallet_data: Wallet data dictionary
            
        Returns:
            Activity level classification (high, medium, low, dormant)
        """
        # Default to low activity
        activity_level = "Low"
        
        # Check if wallet is dormant (no activity in the last 90 days)
        if wallet_data.get("last_activity"):
            try:
                last_activity = datetime.fromisoformat(wallet_data["last_activity"].replace('Z', '+00:00'))
                days_since_activity = (datetime.now() - last_activity).days
                
                if days_since_activity > 90:
                    return "Dormant"
            except (ValueError, TypeError):
                pass
        
        # Classify based on activity frequency
        activity_freq = wallet_data.get("activity_frequency", 0)
        
        if activity_freq > 5:  # More than 5 transactions per active day
            activity_level = "High"
        elif activity_freq > 1:  # 1-5 transactions per active day
            activity_level = "Medium"
        
        return activity_level
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level category (Low, Medium, High)
        """
        if risk_score < 30:
            return "Low"
        elif risk_score < 70:
            return "Medium"
        else:
            return "High"
    
    def _assess_wallet_diversification(self, wallets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess wallet diversification based on balance distribution.
        
        Args:
            wallets: Dictionary of wallets and their data
            
        Returns:
            Dict with diversification assessment
        """
        # Initialize assessment
        assessment = {
            "diversification_score": 0.0,
            "concentration_risk": "Unknown",
            "largest_wallet_pct": 0.0,
            "wallet_count": len(wallets)
        }
        
        if not wallets:
            return assessment
        
        # Calculate total balance across all wallets
        total_balance = sum(wallet.get("balance", 0.0) for wallet in wallets.values())
        
        if total_balance <= 0:
            return assessment
        
        # Calculate balance percentages and find the largest wallet
        wallet_percentages = []
        
        for address, wallet in wallets.items():
            balance = wallet.get("balance", 0.0)
            percentage = (balance / total_balance) * 100 if total_balance > 0 else 0
            wallet_percentages.append(percentage)
        
        # Sort percentages in descending order
        wallet_percentages.sort(reverse=True)
        
        # Set largest wallet percentage
        assessment["largest_wallet_pct"] = wallet_percentages[0] if wallet_percentages else 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI) for diversification
        # HHI is the sum of squared percentages (as decimals), higher means more concentration
        hhi = sum((pct/100)**2 for pct in wallet_percentages)
        
        # Convert HHI to a diversification score (0-100, higher is more diversified)
        # Perfect diversification would be 1/n, worst is 1.0
        perfect_hhi = 1.0 / len(wallets) if len(wallets) > 0 else 0
        worst_hhi = 1.0
        
        # Normalize score to 0-100 range
        if worst_hhi > perfect_hhi:  # Avoid division by zero or negative
            normalized_hhi = (hhi - perfect_hhi) / (worst_hhi - perfect_hhi)
            assessment["diversification_score"] = 100 * (1 - normalized_hhi)
        else:
            assessment["diversification_score"] = 50  # Default middle value
        
        # Determine concentration risk
        if assessment["largest_wallet_pct"] > 80:
            assessment["concentration_risk"] = "High"
        elif assessment["largest_wallet_pct"] > 50:
            assessment["concentration_risk"] = "Medium"
        else:
            assessment["concentration_risk"] = "Low"
        
        return assessment
    
    def _assess_transaction_patterns(self, wallets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess transaction patterns across wallets.
        
        Args:
            wallets: Dictionary of wallets and their data
            
        Returns:
            Dict with transaction pattern assessment
        """
        # Initialize assessment
        assessment = {
            "pattern_risk_score": 50.0,
            "pattern_risk_level": "Medium",
            "transaction_frequency": "Normal",
            "unusual_patterns": []
        }
        
        if not wallets:
            return assessment
        
        # Collect transaction counts and frequencies
        tx_counts = []
        tx_frequencies = []
        inactive_wallets = 0
        high_activity_wallets = 0
        contract_interaction_pct = []
        
        for address, wallet in wallets.items():
            # Transaction count
            tx_count = wallet.get("transaction_count", 0)
            tx_counts.append(tx_count)
            
            # Transaction frequency
            tx_freq = wallet.get("activity_frequency", 0)
            tx_frequencies.append(tx_freq)
            
            # Activity level
            activity_level = wallet.get("activity_level", "Low")
            if activity_level == "Dormant":
                inactive_wallets += 1
            elif activity_level == "High":
                high_activity_wallets += 1
            
            # Contract interaction percentage
            contract_int = wallet.get("contract_interactions", 0)
            total_tx = wallet.get("transaction_count", 0)
            if total_tx > 0:
                contract_pct = (contract_int / total_tx) * 100
                contract_interaction_pct.append(contract_pct)
        
        # Calculate averages
        avg_tx_count = sum(tx_counts) / len(tx_counts) if tx_counts else 0
        avg_tx_freq = sum(tx_frequencies) / len(tx_frequencies) if tx_frequencies else 0
        avg_contract_pct = sum(contract_interaction_pct) / len(contract_interaction_pct) if contract_interaction_pct else 0
        
        # Determine transaction frequency category
        if avg_tx_freq > 3:
            assessment["transaction_frequency"] = "High"
        elif avg_tx_freq < 0.5:
            assessment["transaction_frequency"] = "Low"
        
        # Check for unusual patterns
        if inactive_wallets / len(wallets) > 0.3:
            assessment["unusual_patterns"].append("High percentage of inactive wallets")
        
        if high_activity_wallets / len(wallets) > 0.3:
            assessment["unusual_patterns"].append("High percentage of very active wallets")
        
        if avg_contract_pct > 70:
            assessment["unusual_patterns"].append("High percentage of contract interactions")
        
        # Calculate pattern risk score
        risk_score = 50.0  # Start at medium risk
        
        # Adjust based on unusual patterns
        risk_score += len(assessment["unusual_patterns"]) * 10
        
        # Cap at 0-100 range
        risk_score = max(0, min(100, risk_score))
        
        assessment["pattern_risk_score"] = risk_score
        assessment["pattern_risk_level"] = self._determine_risk_level(risk_score)
        
        return assessment