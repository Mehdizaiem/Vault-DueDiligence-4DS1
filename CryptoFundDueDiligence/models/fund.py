"""
Fund Model Module

This module defines the model for a cryptocurrency fund, including its attributes,
portfolio, wallet infrastructure, and risk metrics. It serves as the central data
structure for fund due diligence analysis.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import json

@dataclass
class TeamMember:
    """Represents a team member at the fund"""
    name: str
    title: str
    background: List[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    role: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "title": self.title,
            "background": self.background,
            "years_experience": self.years_experience,
            "role": self.role
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamMember':
        """Create TeamMember from dictionary"""
        return cls(
            name=data.get("name", ""),
            title=data.get("title", ""),
            background=data.get("background", []),
            years_experience=data.get("years_experience"),
            role=data.get("role")
        )

@dataclass
class Wallet:
    """Represents a cryptocurrency wallet"""
    address: str
    type: str
    balance: float
    security_features: List[str] = field(default_factory=list)
    blockchain: str = "ethereum"
    tokens: List[str] = field(default_factory=list)
    transaction_count: int = 0
    token_transaction_count: int = 0
    first_activity: Optional[str] = None
    last_activity: Optional[str] = None
    risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "address": self.address,
            "type": self.type,
            "balance": self.balance,
            "security_features": self.security_features,
            "blockchain": self.blockchain,
            "tokens": self.tokens,
            "transaction_count": self.transaction_count,
            "token_transaction_count": self.token_transaction_count,
            "first_activity": self.first_activity,
            "last_activity": self.last_activity,
            "risk_score": self.risk_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Wallet':
        """Create Wallet from dictionary"""
        return cls(
            address=data.get("address", ""),
            type=data.get("type", ""),
            balance=data.get("balance", 0.0),
            security_features=data.get("security_features", []),
            blockchain=data.get("blockchain", "ethereum"),
            tokens=data.get("tokens", []),
            transaction_count=data.get("transaction_count", 0),
            token_transaction_count=data.get("token_transaction_count", 0),
            first_activity=data.get("first_activity"),
            last_activity=data.get("last_activity"),
            risk_score=data.get("risk_score", 0.0)
        )

@dataclass
class ComplianceFramework:
    """Represents a fund's compliance framework"""
    kyc_aml_procedures: List[str] = field(default_factory=list)
    regulatory_status: List[str] = field(default_factory=list)
    jurisdictions: List[str] = field(default_factory=list)
    tax_considerations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "kyc_aml_procedures": self.kyc_aml_procedures,
            "regulatory_status": self.regulatory_status,
            "jurisdictions": self.jurisdictions,
            "tax_considerations": self.tax_considerations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceFramework':
        """Create ComplianceFramework from dictionary"""
        return cls(
            kyc_aml_procedures=data.get("kyc_aml_procedures", []),
            regulatory_status=data.get("regulatory_status", []),
            jurisdictions=data.get("jurisdictions", []),
            tax_considerations=data.get("tax_considerations", [])
        )

@dataclass
class RiskAssessment:
    """Represents a comprehensive risk assessment of the fund"""
    overall_risk_score: float = 0.0
    risk_level: str = "Unknown"
    risk_components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    suggested_mitigations: List[str] = field(default_factory=list)
    analysis_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level,
            "risk_components": self.risk_components,
            "risk_factors": self.risk_factors,
            "suggested_mitigations": self.suggested_mitigations,
            "analysis_date": self.analysis_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskAssessment':
        """Create RiskAssessment from dictionary"""
        return cls(
            overall_risk_score=data.get("overall_risk_score", 0.0),
            risk_level=data.get("risk_level", "Unknown"),
            risk_components=data.get("risk_components", {}),
            risk_factors=data.get("risk_factors", []),
            suggested_mitigations=data.get("suggested_mitigations", []),
            analysis_date=data.get("analysis_date")
        )

@dataclass
class Performance:
    """Represents the fund's performance metrics"""
    annual_returns: Dict[str, float] = field(default_factory=dict)
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "annual_returns": self.annual_returns,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "benchmark_comparison": self.benchmark_comparison
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Performance':
        """Create Performance from dictionary"""
        return cls(
            annual_returns=data.get("annual_returns", {}),
            sharpe_ratio=data.get("sharpe_ratio"),
            max_drawdown=data.get("max_drawdown"),
            benchmark_comparison=data.get("benchmark_comparison", {})
        )

class CryptoFund:
    """
    Main class representing a cryptocurrency fund with all its attributes
    and analytical data.
    """
    
    def __init__(self, name: str, document_id: Optional[str] = None):
        """
        Initialize a CryptoFund instance.
        
        Args:
            name: The fund name
            document_id: The ID of the source document
        """
        self.name = name
        self.document_id = document_id
        self.document_source = None
        
        # Basic fund information
        self.launch_date = None
        self.aum = 0.0
        self.strategy = None
        self.management_fee = 0.0
        self.performance_fee = 0.0
        self.minimum_investment = None
        self.lock_up_period = None
        
        # Portfolio structure
        self.portfolio_allocations = {}  # Asset: percentage
        self.protocol_exposures = {}  # Protocol: percentage
        
        # Wallet infrastructure
        self.wallets = []  # List of Wallet objects
        self.security_protocols = []  # Security measures
        self.wallets_aggregate_stats = {}  # Aggregate wallet statistics
        
        # Team information
        self.team_members = []  # List of TeamMember objects
        self.security_team = []  # Security team details
        
        # Compliance
        self.compliance = ComplianceFramework()
        
        # Technical information
        self.tech_infrastructure = {}  # Technical infrastructure details
        
        # Performance
        self.performance = Performance()
        
        # Risk assessment
        self.risk_assessment = RiskAssessment()
        
        # Analysis metadata
        self.analysis_date = datetime.now().isoformat()
        self.analysis_confidence = 0.0  # Confidence score (0-100)
        self.data_sources = []  # Sources used for analysis
        
    def add_wallet(self, wallet: Wallet) -> None:
        """
        Add a wallet to the fund.
        
        Args:
            wallet: Wallet object to add
        """
        self.wallets.append(wallet)
    
    def add_team_member(self, member: TeamMember) -> None:
        """
        Add a team member to the fund.
        
        Args:
            member: TeamMember object to add
        """
        self.team_members.append(member)
    
    def update_portfolio_allocation(self, asset: str, percentage: float) -> None:
        """
        Update portfolio allocation for an asset.
        
        Args:
            asset: Asset name
            percentage: Allocation percentage (0-1)
        """
        self.portfolio_allocations[asset] = percentage
    
    def update_risk_assessment(self, risk_assessment: RiskAssessment) -> None:
        """
        Update the fund's risk assessment.
        
        Args:
            risk_assessment: New risk assessment
        """
        self.risk_assessment = risk_assessment
    
    def calculate_aggregate_wallet_stats(self) -> None:
        """Calculate aggregate statistics for all wallets"""
        if not self.wallets:
            return
        
        total_balance = sum(wallet.balance for wallet in self.wallets)
        total_tx_count = sum(wallet.transaction_count for wallet in self.wallets)
        avg_risk_score = sum(wallet.risk_score for wallet in self.wallets) / len(self.wallets)
        
        self.wallets_aggregate_stats = {
            "total_balance_eth": total_balance,
            "total_transaction_count": total_tx_count,
            "average_risk_score": avg_risk_score,
            "wallet_count": len(self.wallets)
        }
    
    def get_crypto_exposure(self) -> List[str]:
        """
        Get the list of cryptocurrencies the fund is exposed to.
        
        Returns:
            List of cryptocurrency names/symbols
        """
        cryptos = set()
        
        # Extract from portfolio allocations
        for asset in self.portfolio_allocations.keys():
            asset_lower = asset.lower()
            
            # Extract cryptocurrency names from asset descriptions
            if "bitcoin" in asset_lower or "btc" in asset_lower:
                cryptos.add("bitcoin")
            elif "ethereum" in asset_lower or "eth" in asset_lower:
                cryptos.add("ethereum")
            elif "solana" in asset_lower or "sol" in asset_lower:
                cryptos.add("solana")
            elif "binance" in asset_lower or "bnb" in asset_lower:
                cryptos.add("binance")
            elif "cardano" in asset_lower or "ada" in asset_lower:
                cryptos.add("cardano")
            elif "ripple" in asset_lower or "xrp" in asset_lower:
                cryptos.add("ripple")
            # Add more crypto detection patterns as needed
        
        # Extract from wallet tokens
        for wallet in self.wallets:
            for token in wallet.tokens:
                token_lower = token.lower()
                if token_lower == "btc" or token_lower == "wbtc":
                    cryptos.add("bitcoin")
                elif token_lower == "eth" or token_lower == "weth":
                    cryptos.add("ethereum")
                elif token_lower == "sol":
                    cryptos.add("solana")
                elif token_lower == "bnb":
                    cryptos.add("binance")
                elif token_lower == "ada":
                    cryptos.add("cardano")
                elif token_lower == "xrp":
                    cryptos.add("ripple")
                # Handle other tokens
        
        return list(cryptos)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert fund to dictionary representation.
        
        Returns:
            Dictionary representation of the fund
        """
        # Calculate wallet stats before serializing
        self.calculate_aggregate_wallet_stats()
        
        return {
            "name": self.name,
            "document_id": self.document_id,
            "document_source": self.document_source,
            "basic_info": {
                "launch_date": self.launch_date,
                "aum": self.aum,
                "strategy": self.strategy,
                "management_fee": self.management_fee,
                "performance_fee": self.performance_fee,
                "minimum_investment": self.minimum_investment,
                "lock_up_period": self.lock_up_period
            },
            "portfolio": {
                "allocations": self.portfolio_allocations,
                "protocol_exposures": self.protocol_exposures
            },
            "wallet_infrastructure": {
                "wallets": [wallet.to_dict() for wallet in self.wallets],
                "security_protocols": self.security_protocols,
                "aggregate_stats": self.wallets_aggregate_stats
            },
            "team": {
                "members": [member.to_dict() for member in self.team_members],
                "security_team": self.security_team
            },
            "compliance": self.compliance.to_dict(),
            "tech_infrastructure": self.tech_infrastructure,
            "performance": self.performance.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "analysis_metadata": {
                "analysis_date": self.analysis_date,
                "analysis_confidence": self.analysis_confidence,
                "data_sources": self.data_sources,
                "crypto_exposure": self.get_crypto_exposure()
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert fund to JSON string.
        
        Args:
            indent: JSON indentation
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoFund':
        """
        Create a CryptoFund instance from a dictionary.
        
        Args:
            data: Dictionary representation of a fund
            
        Returns:
            CryptoFund instance
        """
        fund = cls(
            name=data.get("name", ""),
            document_id=data.get("document_id")
        )
        
        fund.document_source = data.get("document_source")
        
        # Basic info
        basic_info = data.get("basic_info", {})
        fund.launch_date = basic_info.get("launch_date")
        fund.aum = basic_info.get("aum", 0.0)
        fund.strategy = basic_info.get("strategy")
        fund.management_fee = basic_info.get("management_fee", 0.0)
        fund.performance_fee = basic_info.get("performance_fee", 0.0)
        fund.minimum_investment = basic_info.get("minimum_investment")
        fund.lock_up_period = basic_info.get("lock_up_period")
        
        # Portfolio
        portfolio = data.get("portfolio", {})
        fund.portfolio_allocations = portfolio.get("allocations", {})
        fund.protocol_exposures = portfolio.get("protocol_exposures", {})
        
        # Wallets
        wallet_infra = data.get("wallet_infrastructure", {})
        fund.wallets = [Wallet.from_dict(w) for w in wallet_infra.get("wallets", [])]
        fund.security_protocols = wallet_infra.get("security_protocols", [])
        fund.wallets_aggregate_stats = wallet_infra.get("aggregate_stats", {})
        
        # Team
        team_data = data.get("team", {})
        fund.team_members = [TeamMember.from_dict(m) for m in team_data.get("members", [])]
        fund.security_team = team_data.get("security_team", [])
        
        # Compliance
        compliance_data = data.get("compliance", {})
        fund.compliance = ComplianceFramework.from_dict(compliance_data)
        
        # Technical infrastructure
        fund.tech_infrastructure = data.get("tech_infrastructure", {})
        
        # Performance
        performance_data = data.get("performance", {})
        fund.performance = Performance.from_dict(performance_data)
        
        # Risk assessment
        risk_data = data.get("risk_assessment", {})
        fund.risk_assessment = RiskAssessment.from_dict(risk_data)
        
        # Analysis metadata
        metadata = data.get("analysis_metadata", {})
        fund.analysis_date = metadata.get("analysis_date")
        fund.analysis_confidence = metadata.get("analysis_confidence", 0.0)
        fund.data_sources = metadata.get("data_sources", [])
        
        return fund
    
    @classmethod
    def from_json(cls, json_string: str) -> 'CryptoFund':
        """
        Create a CryptoFund instance from a JSON string.
        
        Args:
            json_string: JSON string representation
            
        Returns:
            CryptoFund instance
        """
        data = json.loads(json_string)
        return cls.from_dict(data)

# Example usage
if __name__ == "__main__":
    # Create a fund
    fund = CryptoFund(name="Example Crypto Fund")
    
    # Add basic info
    fund.aum = 157.4  # in millions
    fund.launch_date = "2023-01-15"
    fund.strategy = "Long-term Ethereum ecosystem investment with selective DeFi protocol exposure"
    fund.management_fee = 0.02  # 2%
    fund.performance_fee = 0.20  # 20%
    fund.minimum_investment = "5 ETH"
    fund.lock_up_period = "6 months with quarterly redemptions thereafter"
    
    # Add portfolio allocations
    fund.update_portfolio_allocation("Ethereum Direct Holdings", 0.625)  # 62.5%
    fund.update_portfolio_allocation("DeFi Protocol Liquidity", 0.183)   # 18.3%
    fund.update_portfolio_allocation("ETH Staking", 0.142)               # 14.2%
    fund.update_portfolio_allocation("NFT Blue Chips", 0.025)            # 2.5%
    fund.update_portfolio_allocation("Layer 2 Projects", 0.020)          # 2.0%
    fund.update_portfolio_allocation("Cash/Stablecoin Reserve", 0.005)   # 0.5%
    
    # Add wallets
    cold_storage = Wallet(
        address="0x7b2e78D4dfB9a4E2A789B752c48c3ecD5F03F66E",
        type="Cold Storage",
        balance=1245.0,
        security_features=["Multi-sig (4-of-7)", "Hardware Security"]
    )
    
    hot_wallet = Wallet(
        address="0x9A87e5F1cBD73D2e96C97087172cc478dCAE7d42",
        type="Hot Wallet",
        balance=125.0,
        security_features=["Multi-sig (3-of-5)", "Daily limits"]
    )
    
    staking_wallet = Wallet(
        address="0x3Dc6E84771F5D5B56f63A2eAb18c909B5Ba9dD54",
        type="Staking Wallet",
        balance=980.0,
        security_features=["Validator nodes with distributed key shards"]
    )
    
    fund.add_wallet(cold_storage)
    fund.add_wallet(hot_wallet)
    fund.add_wallet(staking_wallet)
    
    # Add security protocols
    fund.security_protocols = [
        "Hardware wallet implementation using Ledger Enterprise solutions",
        "24/7 transaction monitoring with anomaly detection",
        "Multi-signature authorization required for transactions >5 ETH",
        "Time-locked transactions for withdrawals exceeding 50 ETH",
        "Regular security audits conducted by CipherBlade (Last: March 2025)"
    ]
    
    # Add team members
    cio = TeamMember(
        name="Alex Chen",
        title="CIO",
        background=[
            "Former Goldman Sachs Digital Assets",
            "8+ years in crypto, since 2017",
            "ETH developer since Homestead"
        ]
    )
    
    risk_officer = TeamMember(
        name="Sarah Matthews",
        title="Risk Officer",
        background=[
            "Previously at BlackRock Risk Management",
            "CFA, CAIA certifications",
            "Implemented proprietary crypto risk models"
        ]
    )
    
    trading_head = TeamMember(
        name="Jayden Williams",
        title="Head of Trading",
        background=[
            "Ex-Citadel quantitative trader",
            "Specializes in MEV protection strategies",
            "Developed custom execution algorithms"
        ]
    )
    
    fund.add_team_member(cio)
    fund.add_team_member(risk_officer)
    fund.add_team_member(trading_head)
    
    # Add security team
    fund.security_team = [
        "In-house security team with 5 members",
        "24/7 on-call rotation",
        "Quarterly penetration testing",
        "Bug bounty program (up to 50 ETH rewards)"
    ]
    
    # Add compliance framework
    fund.compliance = ComplianceFramework(
        kyc_aml_procedures=[
            "Enhanced due diligence for all investors",
            "Chainalysis integration for transaction monitoring",
            "Travel rule compliance for transfers >$3,000",
            "Quarterly compliance reviews by Elliptic"
        ],
        regulatory_status=[
            "Registered with FinCEN as MSB (United States)",
            "VASP registration in EU (under MiCA framework)",
            "Cayman Islands Monetary Authority (CIMA) registered",
            "Singapore MAS exemption holder"
        ],
        jurisdictions=["US", "EU", "Cayman", "Singapore"],
        tax_considerations=[
            "K-1 reporting for US investors",
            "Automated gain/loss calculation for all transactions",
            "Cross-border tax optimization strategy",
            "Quarterly tax accrual estimates provided"
        ]
    )
    
    # Add technical infrastructure
    fund.tech_infrastructure = {
        "trading_systems": [
            "Custom order routing across 8 exchanges",
            "Redundant custody solutions",
            "MEV-resistant transaction routing",
            "Slippage optimization algorithms"
        ],
        "monitoring_tools": [
            "Dune Analytics custom dashboards",
            "Nansen portfolio tracking",
            "DefiLlama TVL monitoring",
            "Tenderly for smart contract alerts"
        ]
    }
    
    # Add performance data
    fund.performance = Performance(
        annual_returns={
            "2023": 1.87,  # 187%
            "2024": 0.42,  # 42%
            "2025_ytd": 0.15  # 15% YTD
        },
        sharpe_ratio=1.8,
        max_drawdown=0.42,  # 42%
        benchmark_comparison={
            "vs_eth_2023": 0.22,  # Outperformed ETH by 22%
            "vs_eth_2024": 0.04,  # Outperformed ETH by 4%
            "vs_eth_2025_ytd": 0.03  # Outperformed ETH by 3% YTD
        }
    )
    
    # Print fund as JSON
    print(fund.to_json())