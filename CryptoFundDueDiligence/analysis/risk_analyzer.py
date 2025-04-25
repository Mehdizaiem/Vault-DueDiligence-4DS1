# Path: analysis/risk_analyzer.py
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import math
import re

from CryptoFundDueDiligence.utils.report_utils import format_currency # Import re for lock-up parsing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """
    Evaluates comprehensive risk factors for crypto funds, ensuring factors are structured dictionaries.
    """

    def __init__(self, data_retriever=None):
        """
        Initialize the risk analyzer.
        """
        self.retriever = data_retriever # Keep retriever if needed for future enhancements

        # Define risk weights for different categories
        self.risk_weights = {
            "market_risk": 0.25,
            "smart_contract_risk": 0.20,
            "regulatory_risk": 0.15,
            "liquidity_risk": 0.15,
            "operational_risk": 0.15,
            "concentration_risk": 0.10
        }

        # Define regulatory status risk levels (simplified, 0-10 scale, higher is riskier)
        self.regulatory_risk_map = {
            "Registered": 3.0,
            "Pending": 5.0,
            "Mentioned": 6.5,
            "No Registration Found": 8.0,
            "Unknown": 7.0
        }
        logger.info("RiskAnalyzer initialized.")


    def analyze_fund_risks(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any],
                           market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on a crypto fund.
        """
        fund_name = fund_data.get("fund_info", {}).get("fund_name", "Unknown Fund")
        logger.info(f"Starting comprehensive risk analysis for fund: {fund_name}")

        risk_assessment = {
            "overall_risk_score": 0.0,
            "risk_level": "Unknown",
            "risk_components": {},
            "risk_factors": [], # This will now store the dictionaries
            "suggested_mitigations": [], # Keep this separate
            "analysis_date": datetime.now().isoformat()
        }

        # Calculate individual risk components
        # These methods now return dicts with structured factors
        market_risk = self._evaluate_market_risk(market_analysis)
        smart_contract_risk = self._evaluate_smart_contract_risk(fund_data, wallet_analysis)
        regulatory_risk = self._evaluate_regulatory_risk(fund_data)
        liquidity_risk = self._evaluate_liquidity_risk(fund_data, market_analysis) # Removed wallet_analysis dependency here if not strictly needed for liquidity
        operational_risk = self._evaluate_operational_risk(fund_data, wallet_analysis)
        concentration_risk = self._evaluate_concentration_risk(fund_data, wallet_analysis)

        risk_assessment["risk_components"] = {
            "market_risk": market_risk,
            "smart_contract_risk": smart_contract_risk,
            "regulatory_risk": regulatory_risk,
            "liquidity_risk": liquidity_risk,
            "operational_risk": operational_risk,
            "concentration_risk": concentration_risk
        }

        # Calculate overall weighted risk score (using 0-10 scale scores from components)
        overall_score_10 = sum(
            comp.get("score", 5.0) * self.risk_weights.get(comp_name, 0)
            for comp_name, comp in risk_assessment["risk_components"].items()
        )
        # Normalize the weighted score (max possible is 10 if all weights sum to 1)
        total_weight = sum(self.risk_weights.values())
        overall_score_10_normalized = overall_score_10 / total_weight if total_weight > 0 else overall_score_10

        # Convert to 0-100 scale
        overall_risk_score_100 = min(100, max(0, overall_score_10_normalized * 10))

        risk_assessment["overall_risk_score"] = overall_risk_score_100
        risk_assessment["risk_level"] = self._determine_risk_level(overall_risk_score_100)

        # Combine all structured risk factors from components
        all_factors = []
        for component_data in risk_assessment["risk_components"].values():
            all_factors.extend(component_data.get("factors", []))

        # Sort factors by risk level (descending) and store
        risk_assessment["risk_factors"] = sorted(all_factors, key=lambda x: x.get("risk_level", 0), reverse=True)

        # Generate suggested mitigations based on factors
        risk_assessment["suggested_mitigations"] = self._suggest_mitigations(risk_assessment["risk_factors"])

        logger.info(f"Risk analysis complete for {fund_name}. Overall risk score: {overall_risk_score_100:.2f}, Risk level: {risk_assessment['risk_level']}")
        return risk_assessment

    def _evaluate_market_risk(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        market_risk = {"score": 5.0, "level": "Medium", "factors": []}
        volatility_data = market_analysis.get("volatility", {})
        performance_data = market_analysis.get("historical_performance", {})
        correlation_data = market_analysis.get("correlations", {})

        if not volatility_data:
            market_risk["factors"].append({
                "factor": "Missing Volatility Data",
                "description": "Insufficient volatility data prevented market risk assessment.",
                "risk_level": 6.0 # Default higher risk if no data
            })
            market_risk["score"] = 6.0
            market_risk["level"] = self._determine_risk_level(60)
            return market_risk

        asset_volatilities = [v.get("annual_volatility", 0) for v in volatility_data.values() if v.get("annual_volatility") is not None]

        if asset_volatilities:
            avg_volatility = sum(asset_volatilities) / len(asset_volatilities)
            # Scale volatility (0-150% range typical) to 0-10 risk score
            volatility_score = min(10, max(0, avg_volatility / 15.0))
            market_risk["score"] = volatility_score # Base score on volatility

            if avg_volatility > 100:
                market_risk["factors"].append({
                    "factor": "Extremely High Market Volatility",
                    "description": f"Average annualized volatility ({avg_volatility:.1f}%) exceeds 100%, indicating extreme price swings.",
                    "risk_level": 9.0
                })
            elif avg_volatility > 70:
                 market_risk["factors"].append({
                    "factor": "High Market Volatility",
                    "description": f"Average annualized volatility ({avg_volatility:.1f}%) is high (over 70%).",
                    "risk_level": 7.5
                 })

            # Check drawdowns (adjust score based on severity)
            max_drawdown = 0
            for symbol, perf in performance_data.items():
                 if "metrics" in perf and "current_drawdown_pct" in perf["metrics"]:
                     drawdown = abs(perf["metrics"]["current_drawdown_pct"])
                     max_drawdown = max(max_drawdown, drawdown)
                     if drawdown > 50:
                         market_risk["factors"].append({
                            "factor": f"Severe Drawdown Risk ({symbol})",
                            "description": f"Asset {symbol} experienced a drawdown of {drawdown:.1f}% from its recent high.",
                            "risk_level": 8.5
                         })
                         market_risk["score"] = max(market_risk["score"], 8.5) # Elevate score if severe drawdown exists
                     elif drawdown > 30:
                          market_risk["factors"].append({
                            "factor": f"Significant Drawdown Risk ({symbol})",
                            "description": f"Asset {symbol} experienced a drawdown of {drawdown:.1f}% from its recent high.",
                            "risk_level": 7.0
                          })
        else:
            market_risk["factors"].append({
                "factor": "No Volatility Data Calculated",
                "description": "Could not calculate average volatility from provided data.",
                "risk_level": 6.0
            })
            market_risk["score"] = 6.0 # Increase risk score if volatility missing

        # Check correlations
        if correlation_data:
            high_correlations = 0
            correlation_pairs = 0
            total_corr = 0
            for symbol1, correlations in correlation_data.items():
                for symbol2, corr_value in correlations.items():
                    # Ensure we process each pair once and avoid self-correlation if structure includes it
                    if symbol1 < symbol2:
                         correlation_pairs += 1
                         total_corr += abs(corr_value) # Use absolute correlation for diversification check
                         if abs(corr_value) > 0.8:
                             high_correlations += 1

            if correlation_pairs > 0:
                avg_abs_correlation = total_corr / correlation_pairs
                high_corr_percentage = (high_correlations / correlation_pairs) * 100

                if avg_abs_correlation > 0.6 or high_corr_percentage > 50:
                    factor_name = "High Asset Correlation"
                    factor_desc = f"Assets show high positive correlation (Avg Abs Corr: {avg_abs_correlation:.2f}, >80% Corr Pairs: {high_corr_percentage:.0f}%), limiting diversification benefits."
                    factor_level = 7.0 + (avg_abs_correlation - 0.6) * 5 # Scale risk based on avg correlation
                    market_risk["factors"].append({"factor": factor_name, "description": factor_desc, "risk_level": min(9.0, factor_level)})
                    market_risk["score"] = max(market_risk["score"], min(9.0, factor_level)) # Adjust overall score

        market_risk["score"] = min(10, max(0, market_risk["score"])) # Clamp score
        market_risk["level"] = self._determine_risk_level(market_risk["score"] * 10)
        return market_risk

    def _evaluate_smart_contract_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any]) -> Dict[str, Any]:
        smart_contract_risk = {"score": 5.0, "level": "Medium", "factors": []}
        portfolio_data = fund_data.get("portfolio_data", {})
        onchain_analysis = wallet_analysis # Assuming wallet_analysis contains the onchain data

        # Simplified risk scores for protocol types (higher is riskier)
        protocol_type_risk = {"lending": 6.5, "dex": 6.0, "yield_farming": 7.5, "aggregator": 7.0, "bridge": 8.0, "derivatives": 7.5, "staking": 5.0, "nft": 5.5, "unknown": 7.0}
        defi_keywords = ["aave", "compound", "uniswap", "sushiswap", "curve", "yearn", "maker", "balancer", "synthetix", "1inch", "pancakeswap", "dydx", "lido", "rocket pool", "yield", "farm", "liquid", "stake", "pool"]

        defi_exposure = 0.0
        weighted_protocol_risk = 0
        high_risk_exposure = 0.0 # Exposure to protocols with risk > 7.0

        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            is_defi = any(keyword in asset_lower for keyword in defi_keywords)

            if is_defi:
                defi_exposure += allocation
                # Basic protocol type inference (can be improved)
                ptype = "unknown"
                if any(k in asset_lower for k in ["lending", "aave", "compound", "maker"]): ptype = "lending"
                elif any(k in asset_lower for k in ["dex", "uniswap", "sushiswap", "curve", "balancer", "swap"]): ptype = "dex"
                elif any(k in asset_lower for k in ["yield", "farm", "harvest", "yearn"]): ptype = "yield_farming"
                elif any(k in asset_lower for k in ["bridge", "wormhole", "multichain"]): ptype = "bridge"
                elif any(k in asset_lower for k in ["stake", "lido", "rocket"]): ptype = "staking"
                elif any(k in asset_lower for k in ["nft", "collectible"]): ptype = "nft"

                protocol_risk_score = protocol_type_risk.get(ptype, 7.0)
                weighted_protocol_risk += allocation * protocol_risk_score
                if protocol_risk_score > 7.0:
                    high_risk_exposure += allocation

        # Calculate score based on exposure and weighted risk
        base_score = 5.0
        if defi_exposure > 0:
            avg_protocol_risk = weighted_protocol_risk / defi_exposure
            # Score increases significantly with DeFi exposure and average risk of protocols used
            base_score = 4.0 + (defi_exposure * 4.0) + (avg_protocol_risk - 5.0) * 0.5 # Adjusted formula

        if defi_exposure > 0.5:
            smart_contract_risk["factors"].append({"factor": "High DeFi Protocol Exposure", "description": f"Over {defi_exposure*100:.0f}% of portfolio allocated to DeFi, increasing smart contract risk.", "risk_level": 7.0 + defi_exposure * 2})
        elif defi_exposure > 0.2:
             smart_contract_risk["factors"].append({"factor": "Significant DeFi Exposure", "description": f"More than {defi_exposure*100:.0f}% of portfolio allocated to DeFi.", "risk_level": 6.0 + defi_exposure * 2})

        if high_risk_exposure > 0.2:
             smart_contract_risk["factors"].append({"factor": "Exposure to High-Risk Protocols", "description": f"Over {high_risk_exposure*100:.0f}% allocated to potentially higher-risk protocol types (e.g., bridges, yield farming).", "risk_level": 7.5 + high_risk_exposure * 3})
             base_score += 1.5 # Add penalty for high-risk exposure

        # Consider on-chain interaction data if available
        avg_contract_interaction_pct = onchain_analysis.get("risk_assessment", {}).get("transaction_patterns", {}).get("avg_contract_interaction_pct", 0) / 100 if onchain_analysis else 0
        if avg_contract_interaction_pct > 0.7:
             smart_contract_risk["factors"].append({"factor": "High On-Chain Contract Interaction", "description": f"Wallets show a high average rate ({avg_contract_interaction_pct*100:.0f}%) of contract interactions.", "risk_level": 6.5})
             base_score += 0.5 # Slightly increase score

        # Audit check (example, should come from document analysis ideally)
        # mentions_audits = "audit" in str(fund_data).lower()
        # if defi_exposure > 0.1 and not mentions_audits:
        #     smart_contract_risk["factors"].append({"factor": "Lack of Verified Audits", "description": "No mention of security audits for utilized DeFi protocols was found.", "risk_level": 7.0})
        #     base_score += 1.0

        smart_contract_risk["score"] = min(10, max(0, base_score))
        smart_contract_risk["level"] = self._determine_risk_level(smart_contract_risk["score"] * 10)
        return smart_contract_risk

    def _evaluate_regulatory_risk(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        regulatory_risk = {"score": 5.0, "level": "Medium", "factors": []}
        compliance_analysis = fund_data.get("compliance_analysis", {})
        fund_info = fund_data.get("fund_info", {})

        base_score = 5.0 # Initialize default base_score

        if compliance_analysis and "error" not in compliance_analysis:
            overall_compliance_score = compliance_analysis.get("overall_compliance_score", 50)
            base_score = (100 - overall_compliance_score) / 10.0 # Overwrite if compliance data is good
            regulatory_risk["score"] = base_score

            compliance_gaps = compliance_analysis.get("compliance_gaps", [])
            if compliance_gaps:
                for gap in compliance_gaps[:3]:
                     regulatory_risk["factors"].append({
                         "factor": "Identified Compliance Gap", "description": gap, "risk_level": 7.0
                     })
                base_score = min(10, base_score + len(compliance_gaps) * 0.5) # Adjust base_score

            kyc_aml_score = compliance_analysis.get("kyc_aml_assessment", {}).get("coverage_score", 50)
            if kyc_aml_score < 50:
                 regulatory_risk["factors"].append({
                     "factor": "Weak KYC/AML Coverage", "description": f"KYC/AML procedure coverage estimated at {kyc_aml_score:.1f}%.", "risk_level": 7.5
                 })
                 base_score = min(10, base_score + 1.0) # Adjust base_score

            reg_status = compliance_analysis.get("regulatory_status", {})
            unregistered_major = False
            for jur, status_info in reg_status.items():
                 if jur in ["US", "EU", "UK", "Singapore"] and status_info.get("registration_status") == "No Registration Found":
                     unregistered_major = True
                     regulatory_risk["factors"].append({
                        "factor": f"No Registration Found ({jur})", "description": f"Fund does not appear registered in {jur}.", "risk_level": 8.0
                     })
            if unregistered_major:
                 base_score = min(10, base_score + 1.5) # Adjust base_score

        else:
            # --- Fallback Logic ---
            logger.warning("No detailed compliance analysis found or analysis failed, performing basic regulatory risk assessment.")
            # ****** FIX: Initialize base_score here for the fallback path ******
            base_score = 7.0 # Start with a higher default risk in fallback
            # ********************************************************************
            compliance_data = fund_data.get("compliance_data", {}) # Use raw data
            regulatory_status_list = compliance_data.get("regulatory_status", [])
            kyc_aml_list = compliance_data.get("kyc_aml", [])

            if not regulatory_status_list and not kyc_aml_list:
                 regulatory_risk["factors"].append({
                    "factor": "Missing Compliance Information", "description": "Limited or no information found regarding compliance framework.", "risk_level": 8.0
                 })
                 base_score = 8.0 # Override base score
            else:
                if not kyc_aml_list:
                     base_score = min(10, base_score + 1.5) # Adjust base_score
                     regulatory_risk["factors"].append({"factor": "KYC/AML Procedures Undocumented", "description": "No documentation regarding KYC/AML procedures found.", "risk_level": 7.5})
                if not regulatory_status_list:
                     base_score = min(10, base_score + 1.5) # Adjust base_score
                     regulatory_risk["factors"].append({"factor": "Regulatory Status Unclear", "description": "No regulatory registrations or licenses mentioned.", "risk_level": 8.0})
                elif not any(jur in str(regulatory_status_list).lower() for jur in ["us", "eu", "uk", "sec", "fca", "mica"]):
                      base_score = min(10, base_score + 1.0) # Adjust base_score
                      regulatory_risk["factors"].append({"factor": "Limited Major Jurisdiction Coverage", "description": "No clear registration mentioned in major jurisdictions like US, EU, UK.", "risk_level": 7.0})
            # --- End Fallback Logic ---

        # Final score assignment after adjustments
        regulatory_risk["score"] = min(10, max(0, base_score))
        regulatory_risk["level"] = self._determine_risk_level(regulatory_risk["score"] * 10)
        return regulatory_risk

    def _evaluate_liquidity_risk(self, fund_data: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        liquidity_risk = {"score": 5.0, "level": "Medium", "factors": []}
        portfolio_data = fund_data.get("portfolio_data", {})
        fund_info = fund_data.get("fund_info", {})

        # 1. Assess Portfolio Liquidity Profile
        high_liq, med_liq, low_liq = 0.0, 0.0, 0.0
        illiquid_terms = ["nft", "locked", "stake", "illiquid", "private", "seed", "vesting"]
        high_liq_terms = ["cash", "stablecoin", "usdc", "usdt", "dai", "busd"]
        med_liq_terms = ["bitcoin", "btc", "ethereum", "eth", "bnb", "sol", "ada", "xrp"] # Major caps

        for asset, allocation in portfolio_data.items():
            asset_lower = asset.lower()
            if any(term in asset_lower for term in illiquid_terms): low_liq += allocation
            elif any(term in asset_lower for term in high_liq_terms): high_liq += allocation
            elif any(term in asset_lower for term in med_liq_terms): med_liq += allocation
            # Assume remaining is medium-low liquidity

        # Base score on liquidity breakdown (lower score = more liquid = less risk)
        # Weighted score (0=high liq, 0.5=med liq, 1=low liq) -> converted to 0-10 risk
        portfolio_liq_score = (high_liq * 0 + med_liq * 3.0 + (1.0 - high_liq - med_liq - low_liq) * 6.0 + low_liq * 10.0)
        base_score = portfolio_liq_score # Start score based purely on portfolio mix

        if high_liq < 0.05:
            liquidity_risk["factors"].append({"factor": "Low Liquid Reserves", "description": f"Liquid reserves (cash/stablecoins) constitute only {high_liq*100:.1f}% of the portfolio.", "risk_level": 7.0})
            base_score += 1.0
        if low_liq > 0.3:
             liquidity_risk["factors"].append({"factor": "High Illiquid Allocation", "description": f"Significant allocation ({low_liq*100:.1f}%) to potentially illiquid or locked assets.", "risk_level": 8.0})
             base_score += 2.0

        # 2. Assess Redemption Terms / Lock-up
        lock_up_str = fund_info.get("lock_up", "").lower()
        lock_up_months = 0
        if lock_up_str:
             months_match = re.search(r'(\d+)\s*month', lock_up_str)
             years_match = re.search(r'(\d+)\s*year', lock_up_str)
             if months_match: lock_up_months = int(months_match.group(1))
             elif years_match: lock_up_months = int(years_match.group(1)) * 12

        redemption_freq = "unknown"
        if "daily" in lock_up_str: redemption_freq = "daily"
        elif "weekly" in lock_up_str: redemption_freq = "weekly"
        elif "monthly" in lock_up_str: redemption_freq = "monthly"
        elif "quarterly" in lock_up_str: redemption_freq = "quarterly"
        elif "annually" in lock_up_str or "yearly" in lock_up_str: redemption_freq = "annually"

        # Add risk based on lock-up and redemption frequency relative to portfolio liquidity
        if lock_up_months > 12 or redemption_freq in ["annually"]:
             liquidity_risk["factors"].append({"factor": "Long Lock-up/Redemption", "description": f"Extended lock-up or infrequent redemption ({lock_up_str}) limits investor liquidity.", "risk_level": 6.5})
             base_score += 1.0
        elif lock_up_months == 0 and redemption_freq == "daily":
             base_score -= 1.0 # Lower risk for high liquidity terms
        elif redemption_freq == "unknown" and lock_up_months == 0:
             liquidity_risk["factors"].append({"factor": "Unclear Redemption Terms", "description": "Lock-up and redemption frequency are not specified.", "risk_level": 6.0})
             base_score += 1.0

        # Check for mismatch (e.g., short lockup but illiquid assets)
        if lock_up_months < 3 and low_liq > 0.2:
             liquidity_risk["factors"].append({"factor": "Potential Liquidity Mismatch", "description": f"Short lock-up ({lock_up_months} months) conflicts with significant illiquid asset allocation ({low_liq*100:.1f}%).", "risk_level": 8.5})
             base_score += 2.0

        # 3. Consider Market Depth (Simplified - using 24h volume vs estimated holding size)
        fund_aum_millions = fund_info.get("aum", 0) # Assume AUM is in millions
        fund_aum_usd = fund_aum_millions * 1_000_000 if fund_aum_millions else 0
        market_data = market_analysis.get("current_data", {})

        if fund_aum_usd > 0 and market_data:
             for asset, allocation in portfolio_data.items():
                 if allocation < 0.05: continue # Ignore small holdings
                 asset_holding_usd = fund_aum_usd * allocation
                 # Find corresponding market data
                 symbol_found = None
                 asset_lower = asset.lower()
                 for symbol, data in market_data.items():
                     symbol_base = symbol.replace('USDT','').replace('USD','').lower()
                     if symbol_base == asset_lower or asset_lower in symbol.lower():
                         symbol_found = symbol
                         break
                 if symbol_found:
                     volume_24h_usd = market_data[symbol_found].get("volume_24h", 0)
                     if volume_24h_usd > 0 and (asset_holding_usd / volume_24h_usd) > 0.05: # If holding is > 5% of daily volume
                         liquidity_risk["factors"].append({"factor": f"Market Impact Risk ({symbol_found})", "description": f"Fund's holding size in {symbol_found} ({format_currency(asset_holding_usd)}) is significant relative to its 24h volume ({format_currency(volume_24h_usd)}).", "risk_level": 7.0})
                         base_score += 0.5 # Add small risk increment per impactful asset


        liquidity_risk["score"] = min(10, max(0, base_score))
        liquidity_risk["level"] = self._determine_risk_level(liquidity_risk["score"] * 10)
        return liquidity_risk

    def _evaluate_operational_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any]) -> Dict[str, Any]:
        operational_risk = {"score": 5.0, "level": "Medium", "factors": []}
        team_data = fund_data.get("team_data", {})
        wallet_data = wallet_analysis.get("wallets", {}) # Use analyzed wallet data

        key_personnel = team_data.get("key_personnel", [])
        security_team_info = team_data.get("security_team", []) # This is likely just text bullets

        base_score = 5.0

        # Team assessment
        if not key_personnel:
            operational_risk["factors"].append({"factor": "Undocumented Team", "description": "Lack of information on key management personnel.", "risk_level": 8.0})
            base_score += 2.0
        else:
            # Check for key roles (simplified check)
            has_investment_lead = any("invest" in p.get("title", "").lower() or "cio" in p.get("title", "").lower() for p in key_personnel)
            has_risk_compliance = any("risk" in p.get("title", "").lower() or "compli" in p.get("title", "").lower() for p in key_personnel)
            has_tech_lead = any("tech" in p.get("title", "").lower() or "cto" in p.get("title", "").lower() or "engineer" in p.get("title", "").lower() for p in key_personnel)

            if not has_investment_lead:
                 base_score += 1.5; operational_risk["factors"].append({"factor": "Investment Lead Unclear", "description": "No clear investment lead (CIO/Portfolio Manager) identified.", "risk_level": 7.0})
            if not has_risk_compliance:
                 base_score += 1.0; operational_risk["factors"].append({"factor": "Risk/Compliance Role Missing", "description": "No dedicated risk or compliance personnel identified.", "risk_level": 6.5})
            if not has_tech_lead:
                 base_score += 1.0; operational_risk["factors"].append({"factor": "Technical Lead Missing", "description": "No clear technical lead (CTO/Lead Engineer) identified.", "risk_level": 6.0})

        # Security team / processes assessment
        if not security_team_info:
            base_score += 1.5
            operational_risk["factors"].append({"factor": "Security Team Undocumented", "description": "No information regarding a dedicated security team or security processes.", "risk_level": 7.5})
        else:
            # Check for keywords in security team description
            sec_text = " ".join(security_team_info).lower()
            if "audit" not in sec_text and "test" not in sec_text:
                base_score += 0.5; operational_risk["factors"].append({"factor": "Security Audits Unmentioned", "description": "No mention of regular security audits or penetration testing.", "risk_level": 6.0})
            if "monitor" not in sec_text:
                 base_score += 0.5; operational_risk["factors"].append({"factor": "Security Monitoring Unmentioned", "description": "No mention of continuous security monitoring practices.", "risk_level": 6.0})

        # Wallet security features assessment (from wallet_analysis)
        num_wallets = len(wallet_data)
        if num_wallets > 0:
             multisig_count = sum(1 for w in wallet_data.values() if any("multi-sig" in f.lower() for f in w.get("security_features",[])))
             hardware_count = sum(1 for w in wallet_data.values() if any("hardware" in f.lower() for f in w.get("security_features",[])))

             if multisig_count / num_wallets < 0.5: # Less than 50% have multisig
                 base_score += 1.5; operational_risk["factors"].append({"factor": "Limited Multi-Sig Usage", "description": "Multi-signature security is not implemented across a majority of wallets.", "risk_level": 7.5})
             if hardware_count / num_wallets < 0.3: # Less than 30% use hardware security (esp. for cold storage)
                  base_score += 1.0; operational_risk["factors"].append({"factor": "Limited Hardware Security Usage", "description": "Hardware security solutions appear underutilized for fund wallets.", "risk_level": 7.0})
        else:
             base_score += 1.0 # Penalty if no wallet data to assess security features
             operational_risk["factors"].append({"factor": "Wallet Security Unverifiable", "description": "No wallet data available to assess operational security features like multi-sig.", "risk_level": 7.0})

        operational_risk["score"] = min(10, max(0, base_score))
        operational_risk["level"] = self._determine_risk_level(operational_risk["score"] * 10)
        return operational_risk

    def _evaluate_concentration_risk(self, fund_data: Dict[str, Any], wallet_analysis: Dict[str, Any]) -> Dict[str, Any]:
        concentration_risk = {"score": 5.0, "level": "Medium", "factors": []}
        portfolio_data = fund_data.get("portfolio_data", {})
        onchain_analysis = wallet_analysis # Contains wallet distribution info

        # 1. Portfolio Asset Concentration (HHI)
        hhi = sum(allocation ** 2 for allocation in portfolio_data.values()) if portfolio_data else 1.0
        # Normalize HHI (0-1) to risk score (0-10)
        hhi_risk_score = min(10, max(0, hhi * 10))
        base_score = hhi_risk_score # Start score based on HHI

        if hhi_risk_score > 7: # HHI > 0.7 indicates high concentration
            concentration_risk["factors"].append({"factor": "High Portfolio Asset Concentration", "description": f"Portfolio Herfindahl-Hirschman Index (HHI) of {hhi:.2f} indicates high concentration in a few assets.", "risk_level": hhi_risk_score})
        elif hhi_risk_score > 4: # HHI > 0.4 indicates moderate concentration
             concentration_risk["factors"].append({"factor": "Moderate Portfolio Asset Concentration", "description": f"Portfolio HHI ({hhi:.2f}) suggests moderate asset concentration.", "risk_level": hhi_risk_score})

        # Check single asset dominance
        if portfolio_data:
            largest_allocation = max(portfolio_data.values())
            largest_asset = max(portfolio_data.items(), key=lambda x: x[1])[0]
            if largest_allocation > 0.5:
                 concentration_risk["factors"].append({"factor": f"Dominant Holding ({largest_asset})", "description": f"Largest holding '{largest_asset}' accounts for {largest_allocation*100:.1f}% of the portfolio.", "risk_level": 7.0 + (largest_allocation - 0.5)*5})
                 base_score = max(base_score, 7.0 + (largest_allocation - 0.5)*5) # Increase score based on dominance

        # 2. Wallet Concentration (Funds Distribution)
        wallet_dist_info = onchain_analysis.get("risk_assessment", {}).get("wallet_diversification", {})
        if wallet_dist_info:
             wallet_conc_risk_level = wallet_dist_info.get("concentration_risk", "Medium")
             wallet_largest_pct = wallet_dist_info.get("largest_wallet_pct", 0)
             # Convert wallet concentration level to score impact
             wallet_conc_impact = 0
             if wallet_conc_risk_level == "High":
                 wallet_conc_impact = 2.0
                 concentration_risk["factors"].append({"factor": "High Wallet Concentration", "description": f"Funds are highly concentrated, with the largest wallet holding {wallet_largest_pct:.1f}%.", "risk_level": 8.0})
             elif wallet_conc_risk_level == "Medium":
                 wallet_conc_impact = 1.0
                 concentration_risk["factors"].append({"factor": "Moderate Wallet Concentration", "description": f"Funds show moderate concentration across wallets (Largest: {wallet_largest_pct:.1f}%).", "risk_level": 6.5})
             # Adjust base score slightly for wallet concentration
             base_score = (base_score * 0.8) + (wallet_conc_impact * 1.0) # Blend scores, giving asset concentration more weight
        else:
             # If wallet concentration data is missing, add a small risk factor
             base_score += 0.5
             concentration_risk["factors"].append({"factor": "Wallet Concentration Unknown", "description": "Data unavailable to assess fund distribution across wallets.", "risk_level": 6.0})


        # 3. Ecosystem Concentration (Example: Ethereum)
        ethereum_exposure = sum(allocation for asset, allocation in portfolio_data.items() if "eth" in asset.lower() or "erc" in asset.lower())
        if ethereum_exposure > 0.8:
             concentration_risk["factors"].append({"factor": "High Ecosystem Concentration (Ethereum)", "description": f"Over {ethereum_exposure*100:.0f}% of portfolio concentrated within the Ethereum ecosystem.", "risk_level": 6.5})
             base_score = max(base_score, 6.5) # Update score if this risk is higher

        concentration_risk["score"] = min(10, max(0, base_score))
        concentration_risk["level"] = self._determine_risk_level(concentration_risk["score"] * 10)
        return concentration_risk

    def _determine_risk_level(self, risk_score_100: float) -> str:
        """Maps 0-100 score to category."""
        if risk_score_100 < 20: return "Very Low"
        elif risk_score_100 < 40: return "Low"
        elif risk_score_100 < 60: return "Medium"
        elif risk_score_100 < 80: return "High" # Changed thresholds slightly
        else: return "Very High"

    def _suggest_mitigations(self, risk_factors: List[Dict[str, Any]]) -> List[str]:
        """Suggest mitigations based on identified risk factor dictionaries."""
        mitigations = set() # Use set to avoid duplicates
        high_risk_factors = sorted([f for f in risk_factors if f.get("risk_level", 0) >= 7.0], key=lambda x: x.get("risk_level", 0), reverse=True)

        for factor_data in high_risk_factors[:5]: # Focus on top 5 high-risk factors
            factor_name = factor_data.get("factor", "").lower()
            # Suggest specific mitigations based on factor names
            if "volatility" in factor_name: mitigations.add("Implement hedging strategies or diversification into lower volatility assets.")
            elif "drawdown" in factor_name: mitigations.add("Review stop-loss strategy and position sizing.")
            elif "correlation" in factor_name: mitigations.add("Diversify into assets with low or negative correlation.")
            elif "defi exposure" in factor_name: mitigations.add("Limit overall DeFi exposure and vet protocols for audits.")
            elif "high-risk protocols" in factor_name: mitigations.add("Reduce exposure to unaudited/experimental protocols; consider insurance.")
            elif "compliance gap" in factor_name or "regulatory status" in factor_name or "no registration" in factor_name: mitigations.add("Consult legal counsel to address compliance gaps and ensure proper registration.")
            elif "kyc/aml" in factor_name: mitigations.add("Strengthen KYC/AML procedures and potentially engage third-party verification.")
            elif "liquidity mismatch" in factor_name or "illiquid allocation" in factor_name: mitigations.add("Align redemption terms with portfolio liquidity or reduce illiquid holdings.")
            elif "low liquid reserves" in factor_name: mitigations.add("Increase allocation to cash or stablecoins to meet potential redemptions.")
            elif "team undocumented" in factor_name or "expertise gap" in factor_name: mitigations.add("Strengthen team with required expertise or improve documentation.")
            elif "security team" in factor_name or "audits unmentioned" in factor_name: mitigations.add("Establish formal security processes, team/consultant, and conduct regular audits.")
            elif "multi-sig" in factor_name: mitigations.add("Implement multi-signature wallets for all significant fund holdings.")
            elif "hardware security" in factor_name: mitigations.add("Utilize hardware security modules (HSMs) or hardware wallets for cold storage.")
            elif "asset concentration" in factor_name or "dominant holding" in factor_name: mitigations.add("Diversify portfolio holdings to reduce single-asset concentration risk.")
            elif "wallet concentration" in factor_name: mitigations.add("Distribute funds across a larger number of secure wallets.")
            elif "ecosystem concentration" in factor_name: mitigations.add("Consider diversification across multiple blockchain ecosystems.")

        # Add general suggestions if few specific ones were added
        if len(mitigations) < 3:
             mitigations.add("Conduct regular risk reviews and update mitigation strategies.")
             mitigations.add("Maintain transparent communication with investors regarding risks.")

        return list(mitigations)