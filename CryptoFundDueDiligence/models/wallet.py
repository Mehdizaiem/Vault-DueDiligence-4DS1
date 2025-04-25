"""
Wallet Model Module

This module defines the Wallet class and related functionality for representing
and analyzing crypto wallet data for due diligence reports.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

class Wallet:
    """
    Represents a cryptocurrency wallet with its address, balance, transactions,
    and other relevant on-chain information for due diligence analysis.
    """
    
    def __init__(self, 
                 address: str,
                 wallet_type: Optional[str] = None,
                 balance: Optional[float] = None,
                 blockchain: str = "ethereum",
                 security_features: Optional[List[str]] = None):
        """
        Initialize a wallet object.
        
        Args:
            address: The blockchain address
            wallet_type: Type of wallet (e.g., "Cold Storage", "Hot Wallet", "Staking")
            balance: Current balance in native currency units
            blockchain: Blockchain network (default: "ethereum")
            security_features: List of security features implemented
        """
        # Validate Ethereum address format
        if blockchain.lower() == "ethereum" and not self._is_valid_eth_address(address):
            raise ValueError(f"Invalid Ethereum address format: {address}")
            
        self.address = address
        self.wallet_type = wallet_type or self._infer_wallet_type()
        self.balance = balance or 0.0
        self.blockchain = blockchain
        self.security_features = security_features or []
        
        # Advanced metrics
        self.transaction_count = 0
        self.token_transaction_count = 0
        self.last_activity = None
        self.first_activity = None
        self.active_days = 0
        self.tokens = []
        self.risk_score = 0
        self.risk_level = "Unknown"
        self.risk_factors = []
        self.transaction_history = []
        self.unique_interactions = 0
        self.contract_interactions = 0
        self.activity_level = "Unknown"
        self.age_days = 0
        self.activity_frequency = 0
        
    def _is_valid_eth_address(self, address: str) -> bool:
        """
        Validate Ethereum address format.
        
        Args:
            address: The Ethereum address to validate
            
        Returns:
            bool: True if address format is valid
        """
        return bool(re.match(r'^0x[a-fA-F0-9]{40}$', address))
    
    def _infer_wallet_type(self) -> str:
        """
        Infer wallet type based on address and other properties.
        
        Returns:
            str: Inferred wallet type
        """
        # This is a placeholder. In a real implementation, 
        # you might use address patterns, transaction history,
        # or other heuristics to infer the wallet type.
        return "General"

    def calculate_risk_score(self) -> float:
        """
        Calculate a risk score for this wallet based on various factors.
        
        Returns:
            float: Risk score (0-100, higher means higher risk)
        """
        base_score = 50.0  # Start at medium risk
        
        # Adjust based on activity
        if self.age_days > 365:  # Established wallet (> 1 year)
            base_score -= 10
        elif self.age_days < 30:  # New wallet (< 1 month)
            base_score += 15
            
        # Adjust based on transaction count
        if self.transaction_count > 1000:  # Very active
            base_score -= 5
        elif self.transaction_count < 10:  # Almost inactive
            base_score += 10
            
        # Adjust based on security features
        security_score = min(20, len(self.security_features) * 5)
        base_score -= security_score
            
        # Adjust based on wallet type
        if self.wallet_type == "Cold Storage":
            base_score -= 15
        elif self.wallet_type == "Hot Wallet":
            base_score += 5
            
        # Adjust based on risk factors
        base_score += len(self.risk_factors) * 5
        
        # Ensure score stays within 0-100 range
        return max(0, min(100, base_score))

    def determine_risk_level(self) -> str:
        """
        Determine risk level category based on risk score.
        
        Returns:
            str: Risk level ("Low", "Medium", "High")
        """
        if self.risk_score < 30:
            return "Low"
        elif self.risk_score < 70:
            return "Medium"
        else:
            return "High"

    def update_from_onchain_data(self, onchain_data: Dict[str, Any]) -> None:
        """
        Update wallet with data retrieved from on-chain analysis.
        
        Args:
            onchain_data: Dictionary with on-chain analytics data
        """
        # Update basic properties
        if "balance" in onchain_data:
            self.balance = onchain_data["balance"]
            
        if "entity_type" in onchain_data and not self.wallet_type:
            self.wallet_type = onchain_data["entity_type"]
            
        # Update transaction data
        self.transaction_count = onchain_data.get("transaction_count", self.transaction_count)
        self.token_transaction_count = onchain_data.get("token_transaction_count", self.token_transaction_count)
        self.unique_interactions = onchain_data.get("unique_interactions", self.unique_interactions)
        self.contract_interactions = onchain_data.get("contract_interactions", self.contract_interactions)
        
        # Update activity dates
        if "first_activity" in onchain_data:
            self.first_activity = onchain_data["first_activity"]
        
        if "last_activity" in onchain_data:
            self.last_activity = onchain_data["last_activity"]
        
        # Update tokens
        if "tokens" in onchain_data:
            self.tokens = onchain_data["tokens"]
        
        # Update risk assessment
        if "risk_score" in onchain_data:
            self.risk_score = onchain_data["risk_score"]
        else:
            self.risk_score = self.calculate_risk_score()
            
        if "risk_level" in onchain_data:
            self.risk_level = onchain_data["risk_level"]
        else:
            self.risk_level = self.determine_risk_level()
            
        if "risk_factors" in onchain_data:
            self.risk_factors = onchain_data["risk_factors"]
            
        # Calculate additional metrics
        self.active_days = onchain_data.get("active_days", self.active_days)
        
        # Calculate age in days
        if self.first_activity and self.last_activity:
            try:
                first_date = datetime.fromisoformat(self.first_activity.replace('Z', '+00:00'))
                last_date = datetime.fromisoformat(self.last_activity.replace('Z', '+00:00'))
                self.age_days = (last_date - first_date).days
            except (ValueError, TypeError):
                self.age_days = self.active_days
        
        # Calculate activity frequency
        if self.active_days > 0:
            self.activity_frequency = self.transaction_count / self.active_days
        
        # Determine activity level
        self.activity_level = self._determine_activity_level()

    def _determine_activity_level(self) -> str:
        """
        Determine the activity level of the wallet.
        
        Returns:
            str: Activity level classification
        """
        # Check if wallet is dormant (no activity in the last 90 days)
        if self.last_activity:
            try:
                last_activity = datetime.fromisoformat(self.last_activity.replace('Z', '+00:00'))
                days_since_activity = (datetime.now() - last_activity).days
                
                if days_since_activity > 90:
                    return "Dormant"
            except (ValueError, TypeError):
                pass
        
        # Classify based on activity frequency
        if self.activity_frequency > 5:  # More than 5 transactions per active day
            return "High"
        elif self.activity_frequency > 1:  # 1-5 transactions per active day
            return "Medium"
        else:
            return "Low"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert wallet object to dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation of the wallet
        """
        return {
            "address": self.address,
            "wallet_type": self.wallet_type,
            "balance": self.balance,
            "blockchain": self.blockchain,
            "security_features": self.security_features,
            "transaction_count": self.transaction_count,
            "token_transaction_count": self.token_transaction_count,
            "first_activity": self.first_activity,
            "last_activity": self.last_activity,
            "active_days": self.active_days,
            "age_days": self.age_days,
            "tokens": self.tokens,
            "unique_interactions": self.unique_interactions,
            "contract_interactions": self.contract_interactions,
            "activity_frequency": self.activity_frequency,
            "activity_level": self.activity_level,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Wallet':
        """
        Create a Wallet object from a dictionary.
        
        Args:
            data: Dictionary with wallet data
            
        Returns:
            Wallet: New Wallet instance
        """
        wallet = cls(
            address=data["address"],
            wallet_type=data.get("wallet_type"),
            balance=data.get("balance"),
            blockchain=data.get("blockchain", "ethereum"),
            security_features=data.get("security_features")
        )
        
        # Set additional properties
        wallet.transaction_count = data.get("transaction_count", 0)
        wallet.token_transaction_count = data.get("token_transaction_count", 0)
        wallet.first_activity = data.get("first_activity")
        wallet.last_activity = data.get("last_activity")
        wallet.active_days = data.get("active_days", 0)
        wallet.age_days = data.get("age_days", 0)
        wallet.tokens = data.get("tokens", [])
        wallet.unique_interactions = data.get("unique_interactions", 0)
        wallet.contract_interactions = data.get("contract_interactions", 0)
        wallet.activity_frequency = data.get("activity_frequency", 0)
        wallet.activity_level = data.get("activity_level", "Unknown")
        wallet.risk_score = data.get("risk_score", 0)
        wallet.risk_level = data.get("risk_level", "Unknown")
        wallet.risk_factors = data.get("risk_factors", [])
        
        return wallet
    
    def __str__(self) -> str:
        """String representation of the wallet."""
        return f"Wallet({self.address}, type={self.wallet_type}, balance={self.balance} ETH, risk={self.risk_level})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the wallet."""
        return f"Wallet(address='{self.address}', wallet_type='{self.wallet_type}', balance={self.balance}, blockchain='{self.blockchain}')"


class WalletCollection:
    """
    Represents a collection of wallets belonging to a crypto fund,
    with methods for aggregate analysis and risk assessment.
    """
    
    def __init__(self, wallets: Optional[List[Wallet]] = None):
        """
        Initialize a wallet collection.
        
        Args:
            wallets: List of Wallet objects
        """
        self.wallets = wallets or []
        
    def add_wallet(self, wallet: Wallet) -> None:
        """
        Add a wallet to the collection.
        
        Args:
            wallet: Wallet object to add
        """
        self.wallets.append(wallet)
        
    def get_wallet(self, address: str) -> Optional[Wallet]:
        """
        Get a wallet by address.
        
        Args:
            address: The wallet address to find
            
        Returns:
            Wallet: The matching wallet or None if not found
        """
        for wallet in self.wallets:
            if wallet.address.lower() == address.lower():
                return wallet
        return None
    
    def get_total_balance(self) -> float:
        """
        Get the total balance across all wallets.
        
        Returns:
            float: Total balance in native currency
        """
        return sum(wallet.balance for wallet in self.wallets)
    
    def calculate_aggregate_risk_score(self) -> float:
        """
        Calculate the aggregate risk score for the entire wallet collection.
        
        Returns:
            float: Aggregate risk score (0-100)
        """
        if not self.wallets:
            return 50.0  # Default medium risk
            
        # Calculate weighted average risk score based on balance
        total_balance = self.get_total_balance()
        
        if total_balance <= 0:
            # If no balance or negative balance, use simple average
            return sum(wallet.risk_score for wallet in self.wallets) / len(self.wallets)
        
        # Calculate weighted average
        weighted_score = 0.0
        for wallet in self.wallets:
            weight = wallet.balance / total_balance
            weighted_score += wallet.risk_score * weight
            
        return weighted_score
    
    def analyze_wallet_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of funds across wallets.
        
        Returns:
            Dict: Analysis results including diversification metrics
        """
        if not self.wallets:
            return {
                "wallet_count": 0,
                "diversification_score": 0.0,
                "concentration_risk": "Unknown",
                "largest_wallet_pct": 0.0
            }
        
        # Calculate total balance
        total_balance = self.get_total_balance()
        
        if total_balance <= 0:
            return {
                "wallet_count": len(self.wallets),
                "diversification_score": 0.0,
                "concentration_risk": "Unknown",
                "largest_wallet_pct": 0.0
            }
        
        # Calculate balance percentages
        wallet_percentages = []
        for wallet in self.wallets:
            percentage = (wallet.balance / total_balance) * 100
            wallet_percentages.append({
                "address": wallet.address,
                "percentage": percentage,
                "wallet_type": wallet.wallet_type
            })
        
        # Sort by percentage in descending order
        wallet_percentages.sort(key=lambda x: x["percentage"], reverse=True)
        
        # Calculate largest wallet percentage
        largest_wallet_pct = wallet_percentages[0]["percentage"] if wallet_percentages else 0.0
        
        # Calculate Herfindahl-Hirschman Index (HHI) for diversification
        hhi = sum((pct["percentage"]/100)**2 for pct in wallet_percentages)
        
        # Convert HHI to a diversification score (0-100, higher is more diversified)
        perfect_hhi = 1.0 / len(self.wallets)
        worst_hhi = 1.0
        
        # Normalize score to 0-100 range
        if worst_hhi > perfect_hhi:  # Avoid division by zero or negative
            normalized_hhi = (hhi - perfect_hhi) / (worst_hhi - perfect_hhi)
            diversification_score = 100 * (1 - normalized_hhi)
        else:
            diversification_score = 50  # Default middle value
        
        # Determine concentration risk
        if largest_wallet_pct > 80:
            concentration_risk = "High"
        elif largest_wallet_pct > 50:
            concentration_risk = "Medium"
        else:
            concentration_risk = "Low"
        
        return {
            "wallet_count": len(self.wallets),
            "diversification_score": diversification_score,
            "concentration_risk": concentration_risk,
            "largest_wallet_pct": largest_wallet_pct,
            "wallet_distribution": wallet_percentages
        }
    
    def analyze_wallet_types(self) -> Dict[str, Any]:
        """
        Analyze the distribution of wallet types.
        
        Returns:
            Dict: Analysis of wallet types and their balances
        """
        if not self.wallets:
            return {
                "wallet_types": {},
                "type_distribution": []
            }
        
        # Count wallets by type
        wallet_types = {}
        type_balances = {}
        
        for wallet in self.wallets:
            wallet_type = wallet.wallet_type or "Unknown"
            
            # Increment count
            if wallet_type in wallet_types:
                wallet_types[wallet_type] += 1
            else:
                wallet_types[wallet_type] = 1
                
            # Add balance
            if wallet_type in type_balances:
                type_balances[wallet_type] += wallet.balance
            else:
                type_balances[wallet_type] = wallet.balance
        
        # Calculate total balance
        total_balance = sum(type_balances.values())
        
        # Create distribution items
        type_distribution = []
        for wallet_type, count in wallet_types.items():
            balance = type_balances.get(wallet_type, 0)
            percentage = (balance / total_balance) * 100 if total_balance > 0 else 0
            
            type_distribution.append({
                "type": wallet_type,
                "count": count,
                "balance": balance,
                "percentage": percentage
            })
        
        # Sort by balance percentage
        type_distribution.sort(key=lambda x: x["percentage"], reverse=True)
        
        return {
            "wallet_types": wallet_types,
            "type_distribution": type_distribution
        }
    
    def analyze_security_features(self) -> Dict[str, Any]:
        """
        Analyze security features across wallets.
        
        Returns:
            Dict: Analysis of security features
        """
        if not self.wallets:
            return {
                "security_score": 0,
                "features": {},
                "recommendations": []
            }
        
        # Count wallets with each security feature
        feature_counts = {}
        
        for wallet in self.wallets:
            for feature in wallet.security_features:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1
        
        # Calculate security coverage
        feature_coverage = {}
        for feature, count in feature_counts.items():
            coverage_pct = (count / len(self.wallets)) * 100
            feature_coverage[feature] = coverage_pct
        
        # Calculate security score
        security_score = 0
        
        # Check for important security features
        if feature_counts.get("Multi-sig", 0) > 0:
            security_score += 40  # Multi-sig is very important
            
        if feature_counts.get("Hardware Security", 0) > 0:
            security_score += 30  # Hardware security is important
            
        if feature_counts.get("Time-locked", 0) > 0:
            security_score += 15  # Time-locks add security
            
        if feature_counts.get("Daily limits", 0) > 0:
            security_score += 15  # Daily limits help contain potential damage
        
        # Adjust based on coverage
        for feature, coverage in feature_coverage.items():
            if coverage < 50:  # Less than 50% of wallets have this feature
                security_score -= 10
        
        # Ensure score stays within 0-100 range
        security_score = max(0, min(100, security_score))
        
        # Generate recommendations
        recommendations = []
        
        if "Multi-sig" not in feature_counts:
            recommendations.append("Implement multi-signature security for all wallets")
            
        if "Hardware Security" not in feature_counts:
            recommendations.append("Use hardware wallets for cold storage")
            
        if "Time-locked" not in feature_counts:
            recommendations.append("Consider implementing time-locks for large withdrawals")
            
        if "Daily limits" not in feature_counts:
            recommendations.append("Set daily transaction limits to minimize potential losses")
        
        return {
            "security_score": security_score,
            "features": feature_coverage,
            "recommendations": recommendations
        }
    
    def get_aggregated_risk_factors(self) -> List[Dict[str, Any]]:
        """
        Get aggregated risk factors across all wallets with frequency counts.
        
        Returns:
            List: Risk factors with count and severity
        """
        risk_factor_counts = {}
        
        for wallet in self.wallets:
            for factor in wallet.risk_factors:
                if factor in risk_factor_counts:
                    risk_factor_counts[factor] += 1
                else:
                    risk_factor_counts[factor] = 1
        
        # Convert to list of dictionaries with severity
        risk_factors = []
        for factor, count in risk_factor_counts.items():
            # Determine severity based on prevalence
            prevalence = count / len(self.wallets) if self.wallets else 0
            
            if prevalence > 0.7:  # Present in >70% of wallets
                severity = "High"
            elif prevalence > 0.3:  # Present in 30-70% of wallets
                severity = "Medium"
            else:  # Present in <30% of wallets
                severity = "Low"
                
            risk_factors.append({
                "factor": factor,
                "count": count,
                "prevalence": prevalence * 100,  # Convert to percentage
                "severity": severity
            })
        
        # Sort by count in descending order
        risk_factors.sort(key=lambda x: x["count"], reverse=True)
        
        return risk_factors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the wallet collection to a dictionary.
        
        Returns:
            Dict: Dictionary representation of the wallet collection
        """
        wallets_data = [wallet.to_dict() for wallet in self.wallets]
        
        # Calculate aggregate metrics
        aggregate_risk_score = self.calculate_aggregate_risk_score()
        wallet_distribution = self.analyze_wallet_distribution()
        wallet_types = self.analyze_wallet_types()
        security_analysis = self.analyze_security_features()
        risk_factors = self.get_aggregated_risk_factors()
        
        return {
            "wallets": wallets_data,
            "total_balance": self.get_total_balance(),
            "wallet_count": len(self.wallets),
            "aggregate_risk_score": aggregate_risk_score,
            "wallet_distribution": wallet_distribution,
            "wallet_types": wallet_types,
            "security_analysis": security_analysis,
            "risk_factors": risk_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalletCollection':
        """
        Create a WalletCollection from a dictionary.
        
        Args:
            data: Dictionary with wallet collection data
            
        Returns:
            WalletCollection: New WalletCollection instance
        """
        collection = cls()
        
        # Create wallets from wallet data
        for wallet_data in data.get("wallets", []):
            wallet = Wallet.from_dict(wallet_data)
            collection.add_wallet(wallet)
            
        return collection
    
    def __len__(self) -> int:
        """Get the number of wallets in the collection."""
        return len(self.wallets)
    
    def __iter__(self):
        """Iterate through wallets in the collection."""
        return iter(self.wallets)