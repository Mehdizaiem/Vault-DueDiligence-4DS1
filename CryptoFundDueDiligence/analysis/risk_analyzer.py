"""
Risk Analyzer Module

This module evaluates various risk factors for crypto funds based on all available data.
It combines information from document analysis, market data, and on-chain analytics
to generate a comprehensive risk assessment.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """
    Evaluates comprehensive risk factors for crypto funds.
    """
    
    def __init__(self, data_retriever):
        """
        Initialize the risk analyzer.
        
        Args:
            data_retriever: Object that retrieves data from collections
        """
        self.retriever = data_retriever
        
        # Define risk weights for different categories
        self.risk_weights = {
            "market_risk": 0.25,
            "smart_contract_risk": 0.20,
            "regulatory_risk": 0.15,
            "liquidity_risk": 0.15,
            "operational_risk": 0.15,
            "concentration_risk": 0.10
        }
        
        # Define regulatory status risk by jurisdiction
        self.regulatory_risk_by_jurisdiction = {
            "US": {
                "SEC": 0.7,       # Securities and Exchange Commission
                "FinCEN": 0.6,    # Financial Crimes Enforcement Network
                "CFTC": 0.7,      # Commodity Futures Trading Commission
                "None": 0.9       # No registration
            },
            "EU": {
                "MiCA": 0.5,      # Markets in Crypto-Assets regulation
                "VASP": 0.6,      # Virtual Asset Service Provider
                "None": 0.8       # No registration
            },
            "UK": {
                "FCA": 0.6,       # Financial Conduct Authority
                "None": 0.8       # No registration
            },
            "Singapore": {
                "MAS": 0.5,       # Monetary Authority of Singapore
                "None": 0.7       # No registration
            },
            "Cayman": {
                "CIMA": 0.6,      # Cayman Islands Monetary Authority
                "None": 0.7       # No registration
            },
            "BVI": {
                "FSC": 0.6,       # Financial Services Commission
                "None": 0.8       # No registration
            },
            "Switzerland": {
                "FINMA": 0.5,     # Financial Market Supervisory Authority
                "None": 0.7       # No registration
            },
            "Other": {
                "Registered": 0.7,  # Registered with local regulator
                "None": 0.9         # No registration
            }
        }
    
    def analyze_fund_risks(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any], 
                           market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on a crypto fund.
        
        Args:
            fund_data: Fund information from document analysis
            wallet_analysis: On-chain wallet analysis results
            market_analysis: Market data analysis results
            
        Returns:
            Dict with risk assessment results
        """
        logger.info("Starting comprehensive risk analysis")
        
        # Initialize risk assessment object
        risk_assessment = {
            "overall_risk_score": 0.0,
            "risk_level": "Unknown",
            "risk_components": {},
            "risk_factors": [],
            "analysis_date": datetime.now().isoformat()
        }
        
        # Calculate individual risk components
        market_risk = self._evaluate_market_risk(market_analysis)
        smart_contract_risk = self._evaluate_smart_contract_risk(fund_data, wallet_analysis)
        regulatory_risk = self._evaluate_regulatory_risk(fund_data)
        liquidity_risk = self._evaluate_liquidity_risk(fund_data, wallet_analysis, market_analysis)
        operational_risk = self._evaluate_operational_risk(fund_data, wallet_analysis)
        concentration_risk = self._evaluate_concentration_risk(fund_data, wallet_analysis, market_analysis)
        
        # Store risk components
        risk_assessment["risk_components"] = {
            "market_risk": market_risk,
            "smart_contract_risk": smart_contract_risk,
            "regulatory_risk": regulatory_risk,
            "liquidity_risk": liquidity_risk,
            "operational_risk": operational_risk,
            "concentration_risk": concentration_risk
        }
        
        # Calculate overall weighted risk score
        overall_score = (
            (market_risk["score"] * self.risk_weights["market_risk"]) +
            (smart_contract_risk["score"] * self.risk_weights["smart_contract_risk"]) +
            (regulatory_risk["score"] * self.risk_weights["regulatory_risk"]) +
            (liquidity_risk["score"] * self.risk_weights["liquidity_risk"]) +
            (operational_risk["score"] * self.risk_weights["operational_risk"]) +
            (concentration_risk["score"] * self.risk_weights["concentration_risk"])
        )
        
        risk_assessment["overall_risk_score"] = overall_score
        risk_assessment["risk_level"] = self._determine_risk_level(overall_score)
        
        # Combine all risk factors
        for component in risk_assessment["risk_components"].values():
            for factor in component.get("factors", []):
                if factor not in risk_assessment["risk_factors"]:
                    risk_assessment["risk_factors"].append(factor)
        
        # Add suggested mitigations
        risk_assessment["suggested_mitigations"] = self._suggest_mitigations(risk_assessment)
        
        logger.info(f"Risk analysis complete. Overall risk score: {overall_score:.2f}, Risk level: {risk_assessment['risk_level']}")
        return risk_assessment
    
    def _evaluate_market_risk(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate market risk based on volatility, correlation, and market conditions.
        
        Args:
            market_analysis: Market data analysis results
            
        Returns:
            Dict with market risk assessment
        """
        # Initialize market risk assessment
        market_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Check if we have volatility data
        volatility_data = market_analysis.get("volatility", {})
        if not volatility_data:
            market_risk["factors"].append("Insufficient volatility data for proper risk assessment")
            return market_risk
        
        # Calculate average volatility across assets
        asset_volatilities = []
        for symbol, vol_metrics in volatility_data.items():
            if "annual_volatility" in vol_metrics:
                asset_volatilities.append(vol_metrics["annual_volatility"])
        
        if asset_volatilities:
            avg_volatility = sum(asset_volatilities) / len(asset_volatilities)
            
            # Scale volatility to risk score (0-10)
            # Typical annualized crypto volatility ranges from 30% to 120%
            volatility_score = min(10, max(0, avg_volatility / 12))
            
            market_risk["score"] = volatility_score
            
            # Add risk factors based on volatility
            if avg_volatility > 100:
                market_risk["factors"].append("Extremely high market volatility")
            elif avg_volatility > 70:
                market_risk["factors"].append("High market volatility")
            
            # Check for max drawdowns
            performance_data = market_analysis.get("historical_performance", {})
            for symbol, perf in performance_data.items():
                if "metrics" in perf and "current_drawdown_pct" in perf["metrics"]:
                    drawdown = abs(perf["metrics"]["current_drawdown_pct"])
                    if drawdown > 40:
                        market_risk["factors"].append(f"Severe drawdown for {symbol} ({drawdown:.1f}%)")
                        market_risk["score"] += 1  # Increase risk score
        else:
            market_risk["factors"].append("No volatility data available")
        
        # Check correlation data for diversification
        correlation_data = market_analysis.get("correlations", {})
        if correlation_data:
            high_correlations = 0
            correlation_pairs = 0
            
            for symbol1, correlations in correlation_data.items():
                for symbol2, corr_value in correlations.items():
                    correlation_pairs += 1
                    if corr_value > 0.8:  # High correlation
                        high_correlations += 1
            
            if correlation_pairs > 0:
                high_corr_percentage = (high_correlations / correlation_pairs) * 100
                
                if high_corr_percentage > 75:
                    market_risk["factors"].append("Very high correlation between assets (poor diversification)")
                    market_risk["score"] += 1.5
                elif high_corr_percentage > 50:
                    market_risk["factors"].append("High correlation between assets (limited diversification)")
                    market_risk["score"] += 1
        
        # Cap risk score at 10
        market_risk["score"] = min(10, market_risk["score"])
        market_risk["level"] = self._determine_risk_level(market_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return market_risk
    
    def _evaluate_smart_contract_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate smart contract risk based on protocol exposure and on-chain interactions.
        
        Args:
            fund_data: Fund information from document analysis
            wallet_analysis: On-chain wallet analysis results
            
        Returns:
            Dict with smart contract risk assessment
        """
        # Initialize smart contract risk assessment
        smart_contract_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Risk scores for different contract types (0-10 scale)
        contract_type_risk = {
            "lending_protocol": 6.5,
            "dex": 6.0,
            "yield_farming": 7.5,
            "aggregator": 7.0,
            "bridge": 8.0,
            "derivatives": 7.5,
            "staking": 5.0,
            "dao": 6.0,
            "nft_marketplace": 5.5
        }
        
        # Check portfolio data for DeFi protocol exposure
        portfolio_data = fund_data.get("portfolio_data", {})
        defi_exposure = 0.0
        high_risk_protocol_exposure = 0.0
        protocol_risks = {}
        
        # Extract DeFi protocol exposure
        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            
            # Identify DeFi protocols
            if any(protocol in asset_lower for protocol in ["aave", "compound", "uniswap", "sushiswap", "curve", 
                                                            "yearn", "maker", "balancer", "synthetix", "1inch",
                                                            "pancakeswap", "dydx"]):
                defi_exposure += allocation
                
                # Categorize the protocol
                protocol_type = "unknown"
                if any(term in asset_lower for term in ["aave", "compound", "maker"]):
                    protocol_type = "lending_protocol"
                elif any(term in asset_lower for term in ["uniswap", "sushiswap", "pancakeswap", "balancer"]):
                    protocol_type = "dex"
                elif any(term in asset_lower for term in ["yearn", "harvest"]):
                    protocol_type = "yield_farming"
                elif any(term in asset_lower for term in ["1inch", "paraswap"]):
                    protocol_type = "aggregator"
                elif any(term in asset_lower for term in ["wormhole", "multichain", "bridge"]):
                    protocol_type = "bridge"
                elif any(term in asset_lower for term in ["dydx", "perpetual", "futures"]):
                    protocol_type = "derivatives"
                
                # Get risk score for this protocol type
                risk_score = contract_type_risk.get(protocol_type, 7.0)
                protocol_risks[asset] = risk_score
                
                # Track high risk protocol exposure
                if risk_score > 7.0:
                    high_risk_protocol_exposure += allocation
        
        # Analyze on-chain contract interactions
        contract_interaction_pct = 0.0
        unique_contracts = set()
        wallet_count = 0
        
        for address, wallet in wallet_analysis.get("wallets", {}).items():
            contract_int = wallet.get("contract_interactions", 0)
            total_tx = wallet.get("transaction_count", 0)
            
            if total_tx > 0:
                wallet_count += 1
                contract_interaction_pct += (contract_int / total_tx)
            
            # Check for interactions with known risky contracts
            # This is a simplified approach - in a real system, you'd have a database of known risky contracts
        
        # Calculate average contract interaction percentage
        if wallet_count > 0:
            contract_interaction_pct /= wallet_count
        
        # Set risk score based on findings
        base_score = 5.0  # Default medium risk
        
        # Adjust for DeFi exposure
        if defi_exposure > 0.5:  # Over 50% in DeFi
            base_score += 2.0
            smart_contract_risk["factors"].append("High exposure to DeFi protocols")
        elif defi_exposure > 0.25:  # Over 25% in DeFi
            base_score += 1.0
            smart_contract_risk["factors"].append("Significant exposure to DeFi protocols")
        
        # Adjust for high-risk protocol exposure
        if high_risk_protocol_exposure > 0.2:  # Over 20% in high-risk protocols
            base_score += 1.5
            smart_contract_risk["factors"].append("Significant exposure to high-risk protocols")
        
        # Adjust for contract interaction frequency
        if contract_interaction_pct > 0.7:  # Over 70% of transactions are contract interactions
            base_score += 1.0
            smart_contract_risk["factors"].append("Very high frequency of contract interactions")
        
        # Check if specific high-risk protocols are mentioned
        if any(protocol in protocol_risks for protocol in ["yearn", "curve", "synthetix"]):
            smart_contract_risk["factors"].append("Exposure to complex DeFi protocols with potential smart contract risks")
        
        # Cap risk score at 10
        smart_contract_risk["score"] = min(10, base_score)
        smart_contract_risk["level"] = self._determine_risk_level(smart_contract_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return smart_contract_risk
    
    def _evaluate_regulatory_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate regulatory risk based on compliance information and jurisdictions.
        
        Args:
            fund_data: Fund information from document analysis
            
        Returns:
            Dict with regulatory risk assessment
        """
        # Initialize regulatory risk assessment
        regulatory_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Extract compliance information
        compliance_data = fund_data.get("compliance_data", {})
        kyc_aml = compliance_data.get("kyc_aml", [])
        regulatory_status = compliance_data.get("regulatory_status", [])
        tax_considerations = compliance_data.get("tax_considerations", [])
        
        # Identify jurisdictions
        jurisdictions = []
        for status in regulatory_status:
            status_lower = status.lower()
            
            # Look for common regulatory bodies and extract jurisdictions
            if "sec" in status_lower or "fincen" in status_lower or "cftc" in status_lower:
                jurisdictions.append("US")
            elif "mica" in status_lower or "eu" in status_lower:
                jurisdictions.append("EU")
            elif "fca" in status_lower or "uk" in status_lower:
                jurisdictions.append("UK")
            elif "mas" in status_lower or "singapore" in status_lower:
                jurisdictions.append("Singapore")
            elif "cima" in status_lower or "cayman" in status_lower:
                jurisdictions.append("Cayman")
            elif "bvi" in status_lower:
                jurisdictions.append("BVI")
            elif "finma" in status_lower or "switzerland" in status_lower:
                jurisdictions.append("Switzerland")
        
        # If no jurisdictions identified, use "Other"
        if not jurisdictions:
            jurisdictions = ["Other"]
        
        # Calculate base risk score from regulatory status
        jurisdiction_risk_scores = []
        
        for jurisdiction in jurisdictions:
            jurisdiction_risk = self.regulatory_risk_by_jurisdiction.get(jurisdiction, self.regulatory_risk_by_jurisdiction["Other"])
            
            # Check if registered with any regulator in this jurisdiction
            is_registered = False
            for status in regulatory_status:
                status_lower = status.lower()
                if (jurisdiction == "US" and any(reg in status_lower for reg in ["sec", "fincen", "cftc"])) or \
                   (jurisdiction == "EU" and any(reg in status_lower for reg in ["mica", "vasp"])) or \
                   (jurisdiction == "UK" and "fca" in status_lower) or \
                   (jurisdiction == "Singapore" and "mas" in status_lower) or \
                   (jurisdiction == "Cayman" and "cima" in status_lower) or \
                   (jurisdiction == "BVI" and "fsc" in status_lower) or \
                   (jurisdiction == "Switzerland" and "finma" in status_lower) or \
                   (jurisdiction == "Other" and "regist" in status_lower):
                    is_registered = True
                    break
            
            # Get risk score based on registration status
            if is_registered:
                risk_key = next((reg for reg in jurisdiction_risk.keys() if reg != "None"), "Registered")
                jurisdiction_risk_scores.append(jurisdiction_risk[risk_key])
            else:
                jurisdiction_risk_scores.append(jurisdiction_risk["None"])
        
        # Use the average jurisdiction risk score
        if jurisdiction_risk_scores:
            avg_jurisdiction_risk = sum(jurisdiction_risk_scores) / len(jurisdiction_risk_scores)
            base_score = avg_jurisdiction_risk * 10  # Convert 0-1 to 0-10 scale
        else:
            base_score = 7.0  # Default to slightly higher risk if no jurisdiction info
            regulatory_risk["factors"].append("No clear regulatory jurisdiction identified")
        
        # Adjust for KYC/AML procedures
        if kyc_aml:
            has_kyc = any("kyc" in item.lower() for item in kyc_aml)
            has_aml = any("aml" in item.lower() for item in kyc_aml)
            
            if has_kyc and has_aml:
                base_score -= 1.0  # Reduce risk
            elif has_kyc or has_aml:
                base_score -= 0.5  # Slightly reduce risk
        else:
            base_score += 1.0  # Increase risk
            regulatory_risk["factors"].append("No KYC/AML procedures mentioned")
        
        # Adjust for tax reporting
        if tax_considerations:
            has_tax_reporting = any("report" in item.lower() for item in tax_considerations)
            if has_tax_reporting:
                base_score -= 0.5  # Slightly reduce risk
        
        # Add appropriate risk factors
        if "US" in jurisdictions:
            if base_score > 6.0:
                regulatory_risk["factors"].append("Potential regulatory exposure in the US market")
            if "crypto" in str(fund_data).lower() and "security" in str(fund_data).lower():
                regulatory_risk["factors"].append("Potential securities law considerations")
        
        if base_score > 7.0:
            regulatory_risk["factors"].append("Limited regulatory compliance framework")
        
        if "None" in [risk_key for j_risk in [self.regulatory_risk_by_jurisdiction[j] for j in jurisdictions] for risk_key in j_risk]:
            regulatory_risk["factors"].append("Operating without full regulatory registration in some jurisdictions")
        
        # Cap risk score at 10
        regulatory_risk["score"] = min(10, max(0, base_score))
        regulatory_risk["level"] = self._determine_risk_level(regulatory_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return regulatory_risk
    
    def _evaluate_liquidity_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any], 
                                market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate liquidity risk based on holdings, market depth, and redemption terms.
        
        Args:
            fund_data: Fund information from document analysis
            wallet_analysis: On-chain wallet analysis results
            market_analysis: Market data analysis results
            
        Returns:
            Dict with liquidity risk assessment
        """
        # Initialize liquidity risk assessment
        liquidity_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Extract relevant information
        portfolio_data = fund_data.get("portfolio_data", {})
        fund_info = fund_data.get("fund_info", {})
        
        # Check for lock-up period
        lock_up_str = fund_info.get("lock_up", "").lower()
        has_lockup = "lock" in lock_up_str or "lockup" in lock_up_str
        lockup_period_months = 0
        
        if has_lockup:
            # Try to extract the lockup period in months
            import re
            
            # Look for periods like "6 months" or "1 year"
            months_match = re.search(r'(\d+)\s*month', lock_up_str)
            years_match = re.search(r'(\d+)\s*year', lock_up_str)
            
            if months_match:
                lockup_period_months = int(months_match.group(1))
            elif years_match:
                lockup_period_months = int(years_match.group(1)) * 12
        
        # Check for redemption terms
        redemption_terms = "quarterly" if "quarter" in lock_up_str else \
                          "monthly" if "month" in lock_up_str else \
                          "weekly" if "week" in lock_up_str else \
                          "daily" if "day" in lock_up_str else \
                          "unknown"
        
        # Check stablecoin reserves
        stablecoin_reserves = 0.0
        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            if any(stable in asset_lower for stable in ["usdt", "usdc", "dai", "busd", "tusd", "usdp", "stablecoin", "cash"]):
                stablecoin_reserves += allocation
        
        # Base liquidity risk on fund structure
        base_score = 5.0
        
        # Adjust for lock-up period
        if lockup_period_months > 12:
            base_score += 1.0  # Higher liquidity risk with long lockup
            liquidity_risk["factors"].append("Long lock-up period (>12 months)")
        elif lockup_period_months > 6:
            base_score += 0.5  # Slightly higher risk with moderate lockup
        elif lockup_period_months == 0:
            base_score -= 0.5  # Lower risk if no lockup (easier to exit)
        
        # Adjust for redemption terms
        if redemption_terms == "quarterly":
            base_score += 0.5
        elif redemption_terms == "monthly":
            base_score += 0.0  # Neutral
        elif redemption_terms == "weekly":
            base_score -= 0.5
        elif redemption_terms == "daily":
            base_score -= 1.0
        else:  # unknown
            base_score += 0.5
            liquidity_risk["factors"].append("Unclear redemption terms")
        
        # Adjust for stablecoin reserves
        if stablecoin_reserves < 0.01:  # Less than 1%
            base_score += 1.5
            liquidity_risk["factors"].append("Very low stablecoin reserves")
        elif stablecoin_reserves < 0.05:  # Less than 5%
            base_score += 1.0
            liquidity_risk["factors"].append("Low stablecoin reserves")
        elif stablecoin_reserves > 0.2:  # More than 20%
            base_score -= 1.0  # Reduced liquidity risk with high reserves
        
        # Check for high exposure to illiquid assets
        illiquid_assets_exposure = 0.0
        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            if any(term in asset_lower for term in ["nft", "locked", "illiquid", "private", "stake"]):
                illiquid_assets_exposure += allocation
        
        if illiquid_assets_exposure > 0.3:  # More than 30%
            base_score += 1.5
            liquidity_risk["factors"].append("High exposure to illiquid assets")
        elif illiquid_assets_exposure > 0.15:  # More than 15%
            base_score += 1.0
            liquidity_risk["factors"].append("Moderate exposure to illiquid assets")
        
        # Check for market trading volume (relative to fund size)
        market_data = market_analysis.get("current_data", {})
        fund_aum = float(fund_info.get("aum", 0))
        
        if fund_aum > 0:
            for symbol, data in market_data.items():
                # Get asset allocation for this symbol
                symbol_base = symbol.replace("USDT", "").lower()
                allocation = 0.0
                
                # Try to match the symbol to portfolio assets
                for asset, alloc in portfolio_data.items():
                    if symbol_base in asset.lower() or symbol_base in asset.lower().replace(" ", ""):
                        allocation = alloc
                        break
                
                if allocation > 0.1:  # Only check for assets with >10% allocation
                    volume_24h = float(data.get("volume_24h", 0))
                    asset_value = fund_aum * allocation
                    
                    # If fund's holding is more than 10% of daily volume, that's a liquidity risk
                    if volume_24h > 0 and (asset_value / volume_24h) > 0.1:
                        base_score += 1.0
                        liquidity_risk["factors"].append(f"Fund holdings may impact market liquidity for {symbol}")
        
        # Cap risk score at 10
        liquidity_risk["score"] = min(10, max(0, base_score))
        liquidity_risk["level"] = self._determine_risk_level(liquidity_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return liquidity_risk
    
    def _evaluate_operational_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate operational risk based on team experience, security measures, and controls.
        
        Args:
            fund_data: Fund information from document analysis
            wallet_analysis: On-chain wallet analysis results
            
        Returns:
            Dict with operational risk assessment
        """
        # Initialize operational risk assessment
        operational_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Extract relevant information
        team_data = fund_data.get("team_data", {})
        key_personnel = team_data.get("key_personnel", [])
        security_team = team_data.get("security_team", [])
        
        # Extract security features from wallets
        wallet_security_features = []
        for address, wallet in wallet_analysis.get("wallets", {}).items():
            features = wallet.get("security_features", [])
            wallet_security_features.extend([f.lower() for f in features])
        
        # Evaluate team experience
        has_experienced_cio = False
        has_risk_officer = False
        has_tech_expert = False
        
        for person in key_personnel:
            title = person.get("title", "").lower()
            background = person.get("background", [])
            
            # Evaluate CIO/investment lead experience
            if any(role in title for role in ["cio", "chief investment", "head of invest", "investment", "portfolio"]):
                has_experienced_cio = True
                
                # Check for red flags
                if not any(term in str(background).lower() for term in ["year", "experience", "background"]):
                    operational_risk["factors"].append("Limited information about investment team experience")
            
            # Check for risk management roles
            if any(role in title for role in ["risk", "compliance"]):
                has_risk_officer = True
            
            # Check for technical expertise
            if any(role in title for role in ["cto", "tech", "develop", "engineer"]):
                has_tech_expert = True
        
        # Evaluate wallet security features
        has_multisig = any("multi" in feature for feature in wallet_security_features)
        has_hardware = any("hardware" in feature for feature in wallet_security_features)
        has_timelock = any("time" in feature and "lock" in feature for feature in wallet_security_features)
        
        # Base operational risk assessment
        base_score = 5.0
        
        # Adjust for team composition
        if not has_experienced_cio:
            base_score += 1.5
            operational_risk["factors"].append("No experienced investment lead identified")
        
        if not has_risk_officer:
            base_score += 1.0
            operational_risk["factors"].append("No dedicated risk management personnel")
        
        if not has_tech_expert:
            base_score += 1.0
            operational_risk["factors"].append("Limited technical expertise in the team")
        
        # Adjust for security team
        if not security_team:
            base_score += 1.0
            operational_risk["factors"].append("No dedicated security team mentioned")
        
        # Adjust for wallet security features
        if not has_multisig:
            base_score += 1.5
            operational_risk["factors"].append("No multi-signature wallet security")
        
        if not has_hardware:
            base_score += 1.0
            operational_risk["factors"].append("No hardware wallet security mentioned")
        
        if not has_timelock:
            base_score += 0.5
            operational_risk["factors"].append("No time-locked transactions for large withdrawals")
        
        # Cap risk score at 10
        operational_risk["score"] = min(10, max(0, base_score))
        operational_risk["level"] = self._determine_risk_level(operational_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return operational_risk
    
    def _evaluate_concentration_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any], 
                                     market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate concentration risk based on asset allocation and diversification.
        
        Args:
            fund_data: Fund information from document analysis
            wallet_analysis: On-chain wallet analysis results
            market_analysis: Market data analysis results
            
        Returns:
            Dict with concentration risk assessment
        """
        # Initialize concentration risk assessment
        concentration_risk = {
            "score": 5.0,  # Default medium risk
            "level": "Medium",
            "factors": []
        }
        
        # Extract portfolio allocation
        portfolio_data = fund_data.get("portfolio_data", {})
        
        # Calculate Herfindahl-Hirschman Index (HHI) for portfolio concentration
        # HHI is the sum of squared percentages (as decimals), higher means more concentration
        hhi = sum((allocation)**2 for allocation in portfolio_data.values()) if portfolio_data else 1.0
        
        # Convert HHI to a concentration score (0-10, higher is more concentrated)
        # Perfect diversification would be 1/n, worst is 1.0
        n = len(portfolio_data) if portfolio_data else 1
        perfect_hhi = 1.0 / n if n > 0 else 0
        worst_hhi = 1.0
        
        # Normalize score to 0-10 range
        if worst_hhi > perfect_hhi:  # Avoid division by zero or negative
            normalized_hhi = (hhi - perfect_hhi) / (worst_hhi - perfect_hhi)
            concentration_score = normalized_hhi * 10
        else:
            concentration_score = 5.0  # Default middle value
        
        # Check for single asset dominance
        if portfolio_data:
            largest_allocation = max(portfolio_data.values())
            largest_asset = max(portfolio_data.items(), key=lambda x: x[1])[0]
            
            if largest_allocation > 0.7:  # More than 70%
                concentration_risk["factors"].append(f"Extreme concentration in {largest_asset} ({largest_allocation*100:.1f}%)")
                concentration_score += 2.0
            elif largest_allocation > 0.5:  # More than 50%
                concentration_risk["factors"].append(f"High concentration in {largest_asset} ({largest_allocation*100:.1f}%)")
                concentration_score += 1.0
        
        # Check for wallet diversification
        wallet_diversification = wallet_analysis.get("risk_assessment", {}).get("wallet_diversification", {})
        
        if wallet_diversification:
            diversification_score = wallet_diversification.get("diversification_score", 50)
            concentration_risk_level = wallet_diversification.get("concentration_risk", "Medium")
            
            # Convert diversification score (0-100, higher is better) to concentration risk (0-10, higher is worse)
            wallet_concentration_score = (100 - diversification_score) / 10
            
            # Blend portfolio concentration and wallet concentration
            concentration_score = (concentration_score * 0.7) + (wallet_concentration_score * 0.3)
            
            if concentration_risk_level == "High":
                concentration_risk["factors"].append("High concentration of funds in a few wallets")
        
        # Check for ecosystem concentration
        ethereum_tokens = 0.0
        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            if "eth" in asset_lower or any(token in asset_lower for token in ["aave", "uniswap", "compound", "maker"]):
                ethereum_tokens += allocation
        
        if ethereum_tokens > 0.8:  # More than 80% in Ethereum ecosystem
            concentration_risk["factors"].append("Very high concentration in Ethereum ecosystem")
            concentration_score += 1.0
        
        # Set final score
        concentration_risk["score"] = min(10, max(0, concentration_score))
        concentration_risk["level"] = self._determine_risk_level(concentration_risk["score"] * 10)  # Convert 0-10 to 0-100
        
        return concentration_risk
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on risk score.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level category (Low, Medium-Low, Medium, Medium-High, High)
        """
        if risk_score < 20:
            return "Low"
        elif risk_score < 40:
            return "Medium-Low"
        elif risk_score < 60:
            return "Medium"
        elif risk_score < 80:
            return "Medium-High"
        else:
            return "High"
    
    def _suggest_mitigations(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """
        Suggest risk mitigation strategies based on risk assessment.
        
        Args:
            risk_assessment: Risk assessment results
            
        Returns:
            List of suggested mitigations
        """
        mitigations = []
        
        # Get risk components and factors
        risk_components = risk_assessment.get("risk_components", {})
        risk_factors = risk_assessment.get("risk_factors", [])
        
        # Suggest mitigations for market risk
        market_risk = risk_components.get("market_risk", {})
        if market_risk.get("score", 0) > 6:
            mitigations.append("Implement hedging strategies to reduce exposure to market volatility")
            mitigations.append("Consider option-based downside protection for large positions")
        
        # Suggest mitigations for smart contract risk
        smart_contract_risk = risk_components.get("smart_contract_risk", {})
        if smart_contract_risk.get("score", 0) > 6:
            mitigations.append("Limit exposure to complex or new DeFi protocols")
            mitigations.append("Ensure all protocols used have undergone thorough security audits")
            mitigations.append("Consider smart contract insurance for significant DeFi positions")
        
        # Suggest mitigations for regulatory risk
        regulatory_risk = risk_components.get("regulatory_risk", {})
        if regulatory_risk.get("score", 0) > 6:
            mitigations.append("Strengthen KYC/AML procedures and documentation")
            mitigations.append("Consult with legal experts on regulatory compliance in all operating jurisdictions")
            mitigations.append("Establish a compliance monitoring system for regulatory changes")
        
        # Suggest mitigations for liquidity risk
        liquidity_risk = risk_components.get("liquidity_risk", {})
        if liquidity_risk.get("score", 0) > 6:
            mitigations.append("Increase stablecoin/cash reserves for redemption management")
            mitigations.append("Implement tiered redemption process to handle large withdrawals")
            mitigations.append("Set position limits relative to asset trading volumes")
        
        # Suggest mitigations for operational risk
        operational_risk = risk_components.get("operational_risk", {})
        if operational_risk.get("score", 0) > 6:
            mitigations.append("Implement multi-signature wallet security for all major holdings")
            mitigations.append("Establish a dedicated security team or engage external security consultants")
            mitigations.append("Conduct regular security audits and penetration testing")
        
        # Suggest mitigations for concentration risk
        concentration_risk = risk_components.get("concentration_risk", {})
        if concentration_risk.get("score", 0) > 6:
            mitigations.append("Diversify holdings across more assets and ecosystems")
            mitigations.append("Set maximum allocation limits for individual assets")
            mitigations.append("Distribute funds across multiple wallet infrastructures")
        
        # Check for specific risk factors and suggest targeted mitigations
        for factor in risk_factors:
            factor_lower = factor.lower()
            
            if "multi-signature" in factor_lower or "multisig" in factor_lower:
                if "no " in factor_lower:
                    mitigations.append("Implement multi-signature security for all cold storage wallets")
            
            if "correlation" in factor_lower and "high" in factor_lower:
                mitigations.append("Diversify into assets with lower correlation to reduce portfolio volatility")
            
            if "stablecoin" in factor_lower and "low" in factor_lower:
                mitigations.append("Increase stablecoin reserves to handle redemption requests")
            
            if "security team" in factor_lower and "no " in factor_lower:
                mitigations.append("Establish a dedicated security team with cryptocurrency expertise")
        
        # Remove duplicates and return
        return list(dict.fromkeys(mitigations))
