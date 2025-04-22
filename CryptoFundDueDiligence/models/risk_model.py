"""
Risk Model Module

This module implements comprehensive risk assessment models for crypto funds.
It analyzes wallet security, portfolio concentration, market risks, 
smart contract risks, and regulatory compliance risks.
"""

import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskModel:
    """
    Comprehensive risk assessment model for crypto funds.
    """
    
    def __init__(self, data_retriever=None):
        """
        Initialize the risk model with optional data retriever.
        
        Args:
            data_retriever: Object that can fetch additional data from collections
        """
        self.retriever = data_retriever
        
        # Define risk categories and weights
        self.risk_categories = {
            "wallet_security": 0.20,  # 20% weight
            "portfolio_concentration": 0.15,  # 15% weight
            "market_volatility": 0.15,  # 15% weight
            "smart_contract": 0.15,  # 15% weight
            "regulatory_compliance": 0.15,  # 15% weight
            "operational": 0.10,  # 10% weight
            "liquidity": 0.10   # 10% weight
        }
        
        # Initialize risk factor scorers
        self.wallet_risk_scorer = WalletRiskScorer()
        self.portfolio_risk_scorer = PortfolioRiskScorer()
        self.market_risk_scorer = MarketRiskScorer()
        self.contract_risk_scorer = SmartContractRiskScorer()
        self.compliance_risk_scorer = ComplianceRiskScorer()
        self.operational_risk_scorer = OperationalRiskScorer()
        self.liquidity_risk_scorer = LiquidityRiskScorer()
        
    def assess_overall_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment on fund data.
        
        Args:
            fund_data: Dictionary containing fund information
            
        Returns:
            Dict with risk assessment results
        """
        logger.info(f"Performing risk assessment for fund: {fund_data.get('fund_info', {}).get('fund_name', 'Unknown')}")
        
        # Get individual risk scores
        wallet_risk = self.wallet_risk_scorer.calculate_risk(fund_data)
        portfolio_risk = self.portfolio_risk_scorer.calculate_risk(fund_data)
        market_risk = self.market_risk_scorer.calculate_risk(fund_data)
        contract_risk = self.contract_risk_scorer.calculate_risk(fund_data)
        compliance_risk = self.compliance_risk_scorer.calculate_risk(fund_data)
        operational_risk = self.operational_risk_scorer.calculate_risk(fund_data)
        liquidity_risk = self.liquidity_risk_scorer.calculate_risk(fund_data)
        
        # Calculate weighted risk score
        overall_score = (
            wallet_risk.get("score", 5) * self.risk_categories["wallet_security"] +
            portfolio_risk.get("score", 5) * self.risk_categories["portfolio_concentration"] +
            market_risk.get("score", 5) * self.risk_categories["market_volatility"] +
            contract_risk.get("score", 5) * self.risk_categories["smart_contract"] +
            compliance_risk.get("score", 5) * self.risk_categories["regulatory_compliance"] +
            operational_risk.get("score", 5) * self.risk_categories["operational"] +
            liquidity_risk.get("score", 5) * self.risk_categories["liquidity"]
        )
        
        # Scale to 0-100
        overall_risk_score = min(100, max(0, overall_score * 10))
        
        # Map to risk categories
        risk_category = self._map_score_to_category(overall_risk_score)
        
        # Generate risk assessment report
        risk_assessment = {
            "overall_risk_score": overall_risk_score,
            "risk_category": risk_category,
            "component_scores": {
                "wallet_security": wallet_risk,
                "portfolio_concentration": portfolio_risk,
                "market_volatility": market_risk,
                "smart_contract": contract_risk,
                "regulatory_compliance": compliance_risk,
                "operational": operational_risk,
                "liquidity": liquidity_risk
            },
            "risk_factors": self._extract_risk_factors([
                wallet_risk, portfolio_risk, market_risk, contract_risk, 
                compliance_risk, operational_risk, liquidity_risk
            ]),
            "mitigation_strategies": self._generate_mitigation_strategies([
                wallet_risk, portfolio_risk, market_risk, contract_risk, 
                compliance_risk, operational_risk, liquidity_risk
            ]),
            "assessment_date": datetime.now().isoformat()
        }
        
        return risk_assessment
    
    def _map_score_to_category(self, score: float) -> str:
        """
        Map numerical risk score to risk category.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            Risk category string
        """
        if score < 20:
            return "Very Low"
        elif score < 40:
            return "Low"
        elif score < 60:
            return "Medium"
        elif score < 80:
            return "High"
        else:
            return "Very High"
    
    def _extract_risk_factors(self, component_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key risk factors from component risk assessments.
        
        Args:
            component_risks: List of risk component dictionaries
            
        Returns:
            List of key risk factors
        """
        all_factors = []
        
        for component in component_risks:
            factors = component.get("factors", [])
            if factors:
                all_factors.extend(factors)
        
        # Sort by risk level (highest first)
        all_factors.sort(key=lambda x: x.get("risk_level", 0), reverse=True)
        
        # Return top risk factors (up to 10)
        return all_factors[:10]
    
    def _generate_mitigation_strategies(self, component_risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate mitigation strategies for identified risks.
        
        Args:
            component_risks: List of risk component dictionaries
            
        Returns:
            List of mitigation strategies
        """
        all_strategies = []
        
        for component in component_risks:
            strategies = component.get("mitigation_strategies", [])
            if strategies:
                all_strategies.extend(strategies)
        
        # Remove duplicates while preserving order
        unique_strategies = []
        strategy_texts = set()
        
        for strategy in all_strategies:
            strategy_text = strategy.get("description", "")
            if strategy_text and strategy_text not in strategy_texts:
                strategy_texts.add(strategy_text)
                unique_strategies.append(strategy)
        
        return unique_strategies


class WalletRiskScorer:
    """Assesses security risks related to wallet infrastructure"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate wallet security risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        wallet_data = fund_data.get("wallet_data", [])
        
        if not wallet_data:
            return {
                "score": 8.0,  # High risk if no wallet data available
                "category": "High",
                "factors": [{
                    "name": "Missing Wallet Data",
                    "description": "No wallet information available for assessment",
                    "risk_level": 8.0
                }],
                "mitigation_strategies": [{
                    "description": "Request detailed wallet infrastructure documentation from the fund",
                    "priority": "High"
                }]
            }
        
        # Initialize score calculation
        risk_factors = []
        mitigation_strategies = []
        
        # Check for minimum wallet diversification
        wallet_count = len(wallet_data)
        if wallet_count < 3:
            risk_factors.append({
                "name": "Low Wallet Diversification",
                "description": f"Only {wallet_count} wallets identified, increasing single-point-of-failure risk",
                "risk_level": 7.0
            })
            
            mitigation_strategies.append({
                "description": "Implement additional wallet segregation to reduce concentration risk",
                "priority": "High"
            })
        
        # Check for multi-sig security
        multisig_wallets = sum(1 for wallet in wallet_data if any("multi-sig" in feature.lower() for feature in wallet.get("security_features", [])))
        multisig_percentage = multisig_wallets / wallet_count if wallet_count > 0 else 0
        
        if multisig_percentage < 0.5:
            risk_factors.append({
                "name": "Insufficient Multi-signature Security",
                "description": f"Only {int(multisig_percentage * 100)}% of wallets use multi-signature security",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": "Implement multi-signature security for all primary wallets",
                "priority": "High"
            })
        
        # Check for hardware security
        hardware_wallets = sum(1 for wallet in wallet_data if any("hardware" in feature.lower() for feature in wallet.get("security_features", [])))
        hardware_percentage = hardware_wallets / wallet_count if wallet_count > 0 else 0
        
        if hardware_percentage < 0.5:
            risk_factors.append({
                "name": "Limited Hardware Security",
                "description": f"Only {int(hardware_percentage * 100)}% of wallets use hardware security",
                "risk_level": 6.0
            })
            
            mitigation_strategies.append({
                "description": "Implement hardware security solutions for all critical wallets",
                "priority": "Medium"
            })
        
        # Check for hot wallet exposure
        hot_wallets = sum(1 for wallet in wallet_data if wallet.get("type", "").lower() == "hot wallet")
        hot_wallet_balance = sum(wallet.get("balance", 0) for wallet in wallet_data if wallet.get("type", "").lower() == "hot wallet")
        total_balance = sum(wallet.get("balance", 0) for wallet in wallet_data if wallet.get("balance") is not None)
        
        hot_wallet_percentage = hot_wallet_balance / total_balance if total_balance > 0 else 0
        
        if hot_wallet_percentage > 0.15:
            risk_factors.append({
                "name": "High Hot Wallet Exposure",
                "description": f"{int(hot_wallet_percentage * 100)}% of funds in hot wallets, exceeding recommended 15% maximum",
                "risk_level": 7.5
            })
            
            mitigation_strategies.append({
                "description": "Reduce hot wallet holdings to below 15% of total assets",
                "priority": "High"
            })
        
        # Average security score based on features
        security_scores = []
        for wallet in wallet_data:
            features = wallet.get("security_features", [])
            score = 0
            
            # Score based on security features
            if any("multi-sig" in feature.lower() for feature in features):
                score += 3  # Good security
                
                # Extra points for higher thresholds
                for feature in features:
                    if "multi-sig" in feature.lower():
                        threshold_match = feature.lower().find("of-")
                        if threshold_match > 0:
                            # Extract numerator and denominator
                            try:
                                numerator = int(feature[threshold_match-1:threshold_match])
                                denominator = int(feature[threshold_match+3:threshold_match+4])
                                
                                # Better ratios get better scores
                                if numerator > denominator/2:
                                    score += 1  # Higher security threshold
                            except:
                                pass
            
            if any("hardware" in feature.lower() for feature in features):
                score += 2  # Good security
                
            if any("time-lock" in feature.lower() for feature in features):
                score += 1  # Good security
                
            if any("daily limit" in feature.lower() for feature in features):
                score += 1  # Good security
            
            # Adjust for wallet type
            wallet_type = wallet.get("type", "").lower()
            if wallet_type == "cold storage":
                score += 2  # Lower risk
            elif wallet_type == "hot wallet":
                score -= 2  # Higher risk
            
            # Normalize score to 0-10 range
            normalized_score = max(0, min(10, score))
            security_scores.append(10 - normalized_score)  # Convert to risk score (10 - security score)
        
        # Calculate overall wallet security risk score
        avg_risk_score = sum(security_scores) / len(security_scores) if security_scores else 5.0
        
        # Add additional factors if needed
        if len(risk_factors) == 0:
            risk_factors.append({
                "name": "General Wallet Security",
                "description": "Standard wallet security practices observed",
                "risk_level": avg_risk_score
            })
        
        if len(mitigation_strategies) == 0:
            mitigation_strategies.append({
                "description": "Maintain current wallet security practices and regularly audit",
                "priority": "Medium"
            })
        
        # Determine risk category
        if avg_risk_score < 3:
            risk_category = "Very Low"
        elif avg_risk_score < 5:
            risk_category = "Low"
        elif avg_risk_score < 7:
            risk_category = "Medium"
        elif avg_risk_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": avg_risk_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class PortfolioRiskScorer:
    """Assesses risks related to portfolio concentration and diversification"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate portfolio concentration risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        portfolio_data = fund_data.get("portfolio_data", {})
        
        if not portfolio_data:
            return {
                "score": 7.0,  # Medium-high risk if no portfolio data available
                "category": "High",
                "factors": [{
                    "name": "Missing Portfolio Data",
                    "description": "No portfolio allocation information available for assessment",
                    "risk_level": 7.0
                }],
                "mitigation_strategies": [{
                    "description": "Request detailed portfolio allocation data from the fund",
                    "priority": "High"
                }]
            }
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        
        # Calculate the Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum(allocation ** 2 for allocation in portfolio_data.values())
        
        # Normalize HHI to 0-10 scale (higher is more concentrated/risky)
        # A perfectly diversified portfolio with infinite assets would have HHI close to 0
        # A single-asset portfolio would have HHI of 1
        hhi_risk_score = min(10, hhi * 10)
        
        if hhi_risk_score > 5:
            risk_factors.append({
                "name": "High Portfolio Concentration",
                "description": f"Portfolio concentration (HHI: {hhi:.2f}) indicates excessive concentration in few assets",
                "risk_level": hhi_risk_score
            })
            
            mitigation_strategies.append({
                "description": "Diversify portfolio across additional assets to reduce concentration risk",
                "priority": "High"
            })
        
        # Check for over-exposure to a single asset
        max_allocation = max(portfolio_data.values()) if portfolio_data else 1.0
        max_asset = next((asset for asset, allocation in portfolio_data.items() if allocation == max_allocation), "Unknown")
        
        if max_allocation > 0.5:
            risk_factors.append({
                "name": "Single Asset Over-exposure",
                "description": f"Over-concentration in {max_asset} ({max_allocation:.1%}) exceeds recommended maximum of 50%",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": f"Reduce exposure to {max_asset} to below 50% of portfolio",
                "priority": "High"
            })
        
        # Check for stablecoin reserves
        stablecoin_allocations = {
            asset: allocation for asset, allocation in portfolio_data.items() 
            if "usd" in asset.lower() or "usdt" in asset.lower() or "usdc" in asset.lower() or 
               "dai" in asset.lower() or "stablecoin" in asset.lower() or "stable" in asset.lower()
        }
        stablecoin_percentage = sum(stablecoin_allocations.values()) if stablecoin_allocations else 0
        
        if stablecoin_percentage < 0.05:
            risk_factors.append({
                "name": "Low Stablecoin Reserves",
                "description": f"Stablecoin reserves ({stablecoin_percentage:.1%}) below recommended 5% minimum",
                "risk_level": 6.0
            })
            
            mitigation_strategies.append({
                "description": "Increase stablecoin reserves to at least 5% to ensure liquidity during volatile markets",
                "priority": "Medium"
            })
        
        # Calculate overall portfolio concentration risk score using HHI as base
        # but adjusting for other factors
        concentration_score = hhi_risk_score
        
        # Adjust for stablecoin reserves
        if stablecoin_percentage < 0.03:
            concentration_score += 1  # Higher risk for very low reserves
        elif stablecoin_percentage >= 0.1:
            concentration_score -= 0.5  # Lower risk for higher reserves
        
        # Adjust for extreme allocation
        if max_allocation > 0.7:
            concentration_score += 1.5  # Much higher risk
        elif max_allocation > 0.5:
            concentration_score += 1  # Higher risk
        
        # Ensure score is in valid range
        concentration_score = max(0, min(10, concentration_score))
        
        # Determine risk category
        if concentration_score < 3:
            risk_category = "Very Low"
        elif concentration_score < 5:
            risk_category = "Low"
        elif concentration_score < 7:
            risk_category = "Medium"
        elif concentration_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": concentration_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class MarketRiskScorer:
    """Assesses market-related risks such as volatility and correlation"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate market volatility risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        # For simplicity, we'll use the portfolio data to identify assets
        # and hardcode some market risk metrics
        # In a real implementation, this would pull data from market metrics collection
        
        portfolio_data = fund_data.get("portfolio_data", {})
        market_data = fund_data.get("market_data", {})
        
        if not portfolio_data:
            return {
                "score": 6.0,  # Medium risk if no portfolio data available
                "category": "Medium",
                "factors": [{
                    "name": "Missing Market Data",
                    "description": "No portfolio or market data available for volatility assessment",
                    "risk_level": 6.0
                }],
                "mitigation_strategies": [{
                    "description": "Request detailed portfolio and market data for proper risk assessment",
                    "priority": "Medium"
                }]
            }
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        
        # Simplified volatility assessment based on asset classes
        # In real implementation, this would calculate from actual price data
        volatility_by_asset = {
            # Higher number = higher volatility
            "Bitcoin": 0.8,
            "BTC": 0.8,
            "Ethereum": 0.85,
            "ETH": 0.85,
            "Solana": 0.9,
            "SOL": 0.9,
            "Binance Coin": 0.8,
            "BNB": 0.8,
            "Cardano": 0.85,
            "ADA": 0.85,
            "Ripple": 0.8,
            "XRP": 0.8,
            "Polkadot": 0.85,
            "DOT": 0.85,
            "Dogecoin": 0.95,
            "DOGE": 0.95,
            "Stablecoin": 0.1,
            "USDT": 0.1,
            "USDC": 0.1,
            "DeFi Protocol": 0.9,
            "NFT": 0.95,
            "Layer 2": 0.85,
            "Cash": 0.0
        }
        
        # Calculate portfolio volatility (simplified)
        weighted_volatility = 0
        total_allocated = sum(portfolio_data.values())
        
        for asset, allocation in portfolio_data.items():
            # Find best matching volatility
            asset_volatility = 0.8  # Default for unknown assets
            
            for key, vol in volatility_by_asset.items():
                if key.lower() in asset.lower():
                    asset_volatility = vol
                    break
            
            # Weight by allocation percentage
            normalized_allocation = allocation / total_allocated if total_allocated > 0 else 0
            weighted_volatility += asset_volatility * normalized_allocation
        
        # Convert to 0-10 scale
        volatility_score = weighted_volatility * 10
        
        if volatility_score > 7:
            risk_factors.append({
                "name": "High Portfolio Volatility",
                "description": f"Portfolio composition indicates high volatility risk ({volatility_score:.1f}/10)",
                "risk_level": volatility_score
            })
            
            mitigation_strategies.append({
                "description": "Increase allocation to less volatile assets to reduce overall portfolio volatility",
                "priority": "High"
            })
        
        # Check for market cycle positioning
        # This would ideally use real market data, but we'll simplify for demonstration
        is_bull_market = True  # Placeholder - would be determined from actual data
        high_beta_allocation = sum(allocation for asset, allocation in portfolio_data.items() 
                                   if any(key in asset.lower() for key in ["sol", "avax", "dot", "meme", "defi", "nft"]))
        
        if not is_bull_market and high_beta_allocation > 0.3:
            risk_factors.append({
                "name": "Bear Market High-Beta Exposure",
                "description": f"High allocation ({high_beta_allocation:.1%}) to high-beta assets during potential bear market",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": "Reduce exposure to high-beta assets and increase stablecoin reserves during market downturns",
                "priority": "High"
            })
        
        # Determine risk category based on volatility score
        if volatility_score < 3:
            risk_category = "Very Low"
        elif volatility_score < 5:
            risk_category = "Low"
        elif volatility_score < 7:
            risk_category = "Medium"
        elif volatility_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": volatility_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class SmartContractRiskScorer:
    """Assesses risks related to smart contract exposure"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate smart contract risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        portfolio_data = fund_data.get("portfolio_data", {})
        
        if not portfolio_data:
            return {
                "score": 5.0,  # Medium risk if no portfolio data available
                "category": "Medium",
                "factors": [{
                    "name": "Unknown Smart Contract Exposure",
                    "description": "No portfolio data available to assess smart contract risk",
                    "risk_level": 5.0
                }],
                "mitigation_strategies": [{
                    "description": "Request detailed smart contract exposure information from the fund",
                    "priority": "Medium"
                }]
            }
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        
        # Identify DeFi and smart contract exposures
        defi_allocations = {
            asset: allocation for asset, allocation in portfolio_data.items() 
            if any(term in asset.lower() for term in ["defi", "protocol", "compound", "aave", "uniswap", "curve", "swap", "yield", "lending"])
        }
        
        defi_percentage = sum(defi_allocations.values()) if defi_allocations else 0
        
        # High DeFi exposure increases smart contract risk
        if defi_percentage > 0.25:
            risk_factors.append({
                "name": "High DeFi Exposure",
                "description": f"Significant exposure ({defi_percentage:.1%}) to DeFi protocols increases smart contract risk",
                "risk_level": 7.5
            })
            
            mitigation_strategies.append({
                "description": "Limit DeFi protocol exposure to 25% of portfolio and diversify across multiple protocols",
                "priority": "High"
            })
        
        # Calculate risk score based on exposures
        # Base score - increases with higher DeFi allocation
        contract_risk_score = 3 + (defi_percentage * 10)
        
        # Additional checks for specific allocations
        # Check for exposure to newer protocols (would be more sophisticated in real implementation)
        new_protocol_keywords = ["v3", "beta", "new", "flux", "sushi", "pancake"]
        new_protocol_allocation = sum(allocation for asset, allocation in portfolio_data.items() 
                                     if any(keyword in asset.lower() for keyword in new_protocol_keywords))
        
        if new_protocol_allocation > 0.1:
            risk_factors.append({
                "name": "Exposure to Newer Protocols",
                "description": f"Allocation to newer, less audited protocols ({new_protocol_allocation:.1%}) increases smart contract risk",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": "Limit exposure to newer protocols and prioritize well-audited, established DeFi platforms",
                "priority": "High"
            })
            
            # Increase risk score
            contract_risk_score += 1.5
        
        # Check if fund mentions smart contract audits
        mentions_audits = False  # This would be determined from document analysis
        
        if defi_percentage > 0.1 and not mentions_audits:
            risk_factors.append({
                "name": "No Audit Verification",
                "description": "No evidence of smart contract audit verification despite DeFi exposure",
                "risk_level": 7.0
            })
            
            mitigation_strategies.append({
                "description": "Implement protocol audit verification process before allocating capital",
                "priority": "High"
            })
            
            # Increase risk score
            contract_risk_score += 1
        
        # Ensure score is in valid range
        contract_risk_score = max(0, min(10, contract_risk_score))
        
        # Determine risk category
        if contract_risk_score < 3:
            risk_category = "Very Low"
        elif contract_risk_score < 5:
            risk_category = "Low"
        elif contract_risk_score < 7:
            risk_category = "Medium"
        elif contract_risk_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": contract_risk_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class ComplianceRiskScorer:
    """Assesses regulatory and compliance risks"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate regulatory compliance risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        compliance_data = fund_data.get("compliance_data", {})
        
        if not compliance_data:
            return {
                "score": 7.0,  # Medium-high risk if no compliance data available
                "category": "High",
                "factors": [{
                    "name": "Missing Compliance Information",
                    "description": "No compliance framework information available for assessment",
                    "risk_level": 7.0
                }],
                "mitigation_strategies": [{
                    "description": "Request detailed compliance documentation from the fund",
                    "priority": "High"
                }]
            }
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        base_risk_score = 5.0  # Start with medium risk
        
        # Check KYC/AML procedures
        kyc_aml = compliance_data.get("kyc_aml", [])
        
        if not kyc_aml:
            risk_factors.append({
                "name": "Inadequate KYC/AML Procedures",
                "description": "No clear KYC/AML procedures documented",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": "Implement comprehensive KYC/AML procedures and document them clearly",
                "priority": "High"
            })
            
            # Increase risk score
            base_risk_score += 2
        else:
            # Check for comprehensive KYC/AML procedures
            comprehensive_kyc = any("enhanced" in item.lower() or "comprehensive" in item.lower() for item in kyc_aml)
            has_monitoring = any("monitoring" in item.lower() or "screening" in item.lower() for item in kyc_aml)
            
            if not comprehensive_kyc:
                risk_factors.append({
                    "name": "Basic KYC/AML Procedures",
                    "description": "Standard but not enhanced KYC/AML procedures in place",
                    "risk_level": 5.0
                })
                
                mitigation_strategies.append({
                    "description": "Upgrade to enhanced due diligence procedures for investors",
                    "priority": "Medium"
                })
            else:
                # Reduce risk score for good KYC/AML
                base_risk_score -= 1
                
            if not has_monitoring:
                risk_factors.append({
                    "name": "Lacking Continuous Monitoring",
                    "description": "No evidence of ongoing transaction monitoring or screening",
                    "risk_level": 6.0
                })
                
                mitigation_strategies.append({
                    "description": "Implement continuous transaction monitoring and regular rescreening",
                    "priority": "Medium"
                })
            else:
                # Reduce risk score for good monitoring
                base_risk_score -= 0.5
                
        # Check regulatory status
        regulatory_status = compliance_data.get("regulatory_status", [])
        
        if not regulatory_status:
            risk_factors.append({
                "name": "Unclear Regulatory Status",
                "description": "No clear regulatory registrations or licenses documented",
                "risk_level": 8.5
            })
            
            mitigation_strategies.append({
                "description": "Obtain appropriate regulatory registrations in key jurisdictions",
                "priority": "High"
            })
            
            # Increase risk score
            base_risk_score += 2.5
        else:
            # Check for key regulatory bodies
            regulated_in_us = any("fincen" in item.lower() or "sec" in item.lower() or "cftc" in item.lower() or "united states" in item.lower() for item in regulatory_status)
            regulated_in_eu = any("mica" in item.lower() or "eu" in item.lower() or "europe" in item.lower() for item in regulatory_status)
            
            if not regulated_in_us and not regulated_in_eu:
                risk_factors.append({
                    "name": "Major Jurisdiction Gap",
                    "description": "No regulatory coverage in major jurisdictions (US or EU)",
                    "risk_level": 7.0
                })
                
                mitigation_strategies.append({
                    "description": "Secure regulatory registration in either US or EU jurisdictions",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1.5
            else:
                # Reduce risk score for good regulatory coverage
                base_risk_score -= 1
        
        # Check tax considerations
        tax_considerations = compliance_data.get("tax_considerations", [])
        
        if not tax_considerations:
            risk_factors.append({
                "name": "Inadequate Tax Documentation",
                "description": "No clear tax reporting or compliance procedures documented",
                "risk_level": 6.0
            })
            
            mitigation_strategies.append({
                "description": "Implement comprehensive tax reporting procedures for all jurisdictions",
                "priority": "Medium"
            })
            
            # Increase risk score
            base_risk_score += 1
        else:
            has_reporting = any("reporting" in item.lower() or "k-1" in item.lower() for item in tax_considerations)
            has_calculation = any("calculation" in item.lower() or "estimates" in item.lower() for item in tax_considerations)
            
            if not has_reporting or not has_calculation:
                risk_factors.append({
                    "name": "Incomplete Tax Framework",
                    "description": "Tax framework missing key components for investor reporting",
                    "risk_level": 5.0
                })
                
                mitigation_strategies.append({
                    "description": "Enhance tax reporting framework to include all necessary components",
                    "priority": "Medium"
                })
                
                # Increase risk score slightly
                base_risk_score += 0.5
        
        # Ensure score is in valid range
        compliance_score = max(0, min(10, base_risk_score))
        
        # Determine risk category
        if compliance_score < 3:
            risk_category = "Very Low"
        elif compliance_score < 5:
            risk_category = "Low"
        elif compliance_score < 7:
            risk_category = "Medium"
        elif compliance_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": compliance_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class OperationalRiskScorer:
    """Assesses operational risks including team, processes, and infrastructure"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate operational risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        team_data = fund_data.get("team_data", {})
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        base_risk_score = 5.0  # Start with medium risk
        
        # Check team information
        key_personnel = team_data.get("key_personnel", [])
        security_team = team_data.get("security_team", [])
        
        if not key_personnel:
            risk_factors.append({
                "name": "Undocumented Team",
                "description": "No key team members or backgrounds documented",
                "risk_level": 8.0
            })
            
            mitigation_strategies.append({
                "description": "Provide detailed information about key team members and their qualifications",
                "priority": "High"
            })
            
            # Increase risk score
            base_risk_score += 2
        else:
            # Check for team experience
            has_technical_expertise = any("developer" in str(person).lower() or 
                                        "engineer" in str(person).lower() or 
                                        "cto" in str(person).lower() or
                                        "technical" in str(person).lower()
                                        for person in key_personnel)
            
            has_financial_expertise = any("cfo" in str(person).lower() or
                                         "financial" in str(person).lower() or
                                         "trader" in str(person).lower() or
                                         "investment" in str(person).lower()
                                         for person in key_personnel)
            
            has_compliance_expertise = any("compliance" in str(person).lower() or
                                          "legal" in str(person).lower() or
                                          "regulatory" in str(person).lower()
                                          for person in key_personnel)
            
            # Check for gaps in expertise
            if not has_technical_expertise:
                risk_factors.append({
                    "name": "Technical Expertise Gap",
                    "description": "No documented technical expertise on the team",
                    "risk_level": 7.0
                })
                
                mitigation_strategies.append({
                    "description": "Add technical expertise to the team or board of advisors",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1.5
            
            if not has_financial_expertise:
                risk_factors.append({
                    "name": "Financial Expertise Gap",
                    "description": "No documented financial expertise on the team",
                    "risk_level": 7.0
                })
                
                mitigation_strategies.append({
                    "description": "Add investment management expertise to the team",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1.5
            
            if not has_compliance_expertise:
                risk_factors.append({
                    "name": "Compliance Expertise Gap",
                    "description": "No documented compliance or legal expertise on the team",
                    "risk_level": 6.0
                })
                
                mitigation_strategies.append({
                    "description": "Add compliance expertise or engage external compliance advisors",
                    "priority": "Medium"
                })
                
                # Increase risk score
                base_risk_score += 1
        
        # Check security team
        if not security_team:
            risk_factors.append({
                "name": "Security Team Gap",
                "description": "No dedicated security team or procedures documented",
                "risk_level": 7.5
            })
            
            mitigation_strategies.append({
                "description": "Establish dedicated security team or engage security consultants",
                "priority": "High"
            })
            
            # Increase risk score
            base_risk_score += 1.5
        else:
            # Check for key security practices
            has_monitoring = any("monitor" in item.lower() for item in security_team)
            has_testing = any("test" in item.lower() or "audit" in item.lower() for item in security_team)
            has_incident_response = any("incident" in item.lower() or "response" in item.lower() for item in security_team)
            
            if not has_monitoring:
                risk_factors.append({
                    "name": "Security Monitoring Gap",
                    "description": "No 24/7 security monitoring procedures documented",
                    "risk_level": 6.5
                })
                
                mitigation_strategies.append({
                    "description": "Implement 24/7 security monitoring",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1
            
            if not has_testing:
                risk_factors.append({
                    "name": "Security Testing Gap",
                    "description": "No regular security testing or auditing procedures documented",
                    "risk_level": 6.0
                })
                
                mitigation_strategies.append({
                    "description": "Implement regular security testing and audits",
                    "priority": "Medium"
                })
                
                # Increase risk score
                base_risk_score += 0.8
            
            if not has_incident_response:
                risk_factors.append({
                    "name": "Incident Response Gap",
                    "description": "No incident response procedures documented",
                    "risk_level": 6.0
                })
                
                mitigation_strategies.append({
                    "description": "Develop and document incident response procedures",
                    "priority": "Medium"
                })
                
                # Increase risk score
                base_risk_score += 0.8
        
        # Ensure score is in valid range
        operational_score = max(0, min(10, base_risk_score))
        
        # Determine risk category
        if operational_score < 3:
            risk_category = "Very Low"
        elif operational_score < 5:
            risk_category = "Low"
        elif operational_score < 7:
            risk_category = "Medium"
        elif operational_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": operational_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }


class LiquidityRiskScorer:
    """Assesses liquidity risks related to investments and redemption capabilities"""
    
    def calculate_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate liquidity risk score.
        
        Args:
            fund_data: Fund information dictionary
            
        Returns:
            Dict with risk assessment
        """
        portfolio_data = fund_data.get("portfolio_data", {})
        fund_info = fund_data.get("fund_info", {})
        
        # Initialize risk calculation
        risk_factors = []
        mitigation_strategies = []
        base_risk_score = 5.0  # Start with medium risk
        
        # Check portfolio liquidity
        if portfolio_data:
            # Categorize assets by liquidity
            high_liquidity_assets = sum(allocation for asset, allocation in portfolio_data.items() 
                                      if any(term in asset.lower() for term in ["cash", "usd", "usdt", "usdc", "dai", "stablecoin"]))
            
            medium_liquidity_assets = sum(allocation for asset, allocation in portfolio_data.items() 
                                        if any(term in asset.lower() for term in ["bitcoin", "btc", "ethereum", "eth", "bnb", "sol"]))
            
            low_liquidity_assets = sum(allocation for asset, allocation in portfolio_data.items() 
                                     if any(term in asset.lower() for term in ["nft", "locked", "staking", "illiquid", "altcoin"]))
            
            # Remaining assets are considered medium-low liquidity
            medium_low_liquidity = 1.0 - (high_liquidity_assets + medium_liquidity_assets + low_liquidity_assets)
            
            # Check for liquidity concerns
            if high_liquidity_assets < 0.1:
                risk_factors.append({
                    "name": "Low Liquid Reserves",
                    "description": f"Insufficient highly liquid assets ({high_liquidity_assets:.1%}) for potential redemptions",
                    "risk_level": 7.5
                })
                
                mitigation_strategies.append({
                    "description": "Increase allocation to highly liquid assets to at least 10%",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1.5
            
            if low_liquidity_assets > 0.3:
                risk_factors.append({
                    "name": "High Illiquid Allocation",
                    "description": f"Excessive allocation to illiquid assets ({low_liquidity_assets:.1%})",
                    "risk_level": 8.0
                })
                
                mitigation_strategies.append({
                    "description": "Reduce allocation to illiquid assets below 30%",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 2
            
            # Calculate weighted liquidity score
            # 0 = most liquid, 1 = least liquid
            liquidity_score = (high_liquidity_assets * 0 + 
                              medium_liquidity_assets * 0.3 + 
                              medium_low_liquidity * 0.6 + 
                              low_liquidity_assets * 1.0)
            
            # Convert to risk score (0-10)
            portfolio_liquidity_risk = liquidity_score * 10
            
            # Adjust base risk score based on portfolio liquidity
            base_risk_score = (base_risk_score * 0.5) + (portfolio_liquidity_risk * 0.5)
        
        # Check redemption terms
        lock_up_period = fund_info.get("lock_up", "")
        
        if lock_up_period:
            # Extract lock-up duration in months (approximate)
            lock_up_months = 0
            
            if "day" in lock_up_period.lower():
                days_match = re.search(r'(\d+)\s*day', lock_up_period.lower())
                if days_match:
                    lock_up_months = int(days_match.group(1)) / 30
            elif "month" in lock_up_period.lower():
                months_match = re.search(r'(\d+)\s*month', lock_up_period.lower())
                if months_match:
                    lock_up_months = int(months_match.group(1))
            elif "year" in lock_up_period.lower():
                years_match = re.search(r'(\d+)\s*year', lock_up_period.lower())
                if years_match:
                    lock_up_months = int(years_match.group(1)) * 12
            
            # Check for liquidity mismatches
            if lock_up_months < 1 and (low_liquidity_assets + medium_low_liquidity) > 0.4:
                risk_factors.append({
                    "name": "Liquidity Mismatch",
                    "description": f"Short lock-up period ({lock_up_period}) with high allocation to less liquid assets ({(low_liquidity_assets + medium_low_liquidity):.1%})",
                    "risk_level": 8.5
                })
                
                mitigation_strategies.append({
                    "description": "Extend lock-up period or increase allocation to liquid assets",
                    "priority": "Critical"
                })
                
                # Increase risk score
                base_risk_score += 2.5
            elif lock_up_months < 3 and low_liquidity_assets > 0.2:
                risk_factors.append({
                    "name": "Moderate Liquidity Mismatch",
                    "description": f"Relatively short lock-up period ({lock_up_period}) with significant allocation to illiquid assets ({low_liquidity_assets:.1%})",
                    "risk_level": 7.0
                })
                
                mitigation_strategies.append({
                    "description": "Adjust lock-up period or reduce allocation to illiquid assets",
                    "priority": "High"
                })
                
                # Increase risk score
                base_risk_score += 1.5
        else:
            # No lock-up information
            risk_factors.append({
                "name": "Undocumented Redemption Terms",
                "description": "No clear lock-up or redemption terms documented",
                "risk_level": 6.0
            })
            
            mitigation_strategies.append({
                "description": "Document and implement appropriate redemption terms based on portfolio liquidity",
                "priority": "Medium"
            })
            
            # Increase risk score
            base_risk_score += 1
        
        # Ensure score is in valid range
        liquidity_score = max(0, min(10, base_risk_score))
        
        # Determine risk category
        if liquidity_score < 3:
            risk_category = "Very Low"
        elif liquidity_score < 5:
            risk_category = "Low"
        elif liquidity_score < 7:
            risk_category = "Medium"
        elif liquidity_score < 9:
            risk_category = "High"
        else:
            risk_category = "Very High"
        
        return {
            "score": liquidity_score,
            "category": risk_category,
            "factors": risk_factors,
            "mitigation_strategies": mitigation_strategies
        }