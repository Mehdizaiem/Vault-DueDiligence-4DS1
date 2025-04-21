"""
Compliance Analyzer Module

This module evaluates regulatory compliance for crypto funds based on document analysis.
It assess KYC/AML procedures, regulatory registrations, and tax considerations across jurisdictions.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplianceAnalyzer:
    """
    Analyzes regulatory compliance aspects of crypto funds.
    """
    
    def __init__(self, data_retriever):
        """
        Initialize the compliance analyzer.
        
        Args:
            data_retriever: Object that retrieves data from collections
        """
        self.retriever = data_retriever
        
        # Define major regulatory jurisdictions and their requirements
        self.jurisdictions = {
            "US": {
                "regulators": ["SEC", "FinCEN", "CFTC", "IRS", "OCC", "FINRA"],
                "key_regulations": ["Securities Act", "Bank Secrecy Act", "Commodity Exchange Act", "FinCEN MSB"]
            },
            "EU": {
                "regulators": ["ESMA", "EBA", "MiCA", "AMLD5"],
                "key_regulations": ["MiCA", "AMLD5", "eIDAS", "VASP registration"]
            },
            "UK": {
                "regulators": ["FCA", "HMRC", "PRA"],
                "key_regulations": ["FCA registration", "MLRs", "PSRs"]
            },
            "Singapore": {
                "regulators": ["MAS"],
                "key_regulations": ["PS Act", "SFA"]
            },
            "Switzerland": {
                "regulators": ["FINMA"],
                "key_regulations": ["FMIA", "AMLA", "FinSA"]
            },
            "Cayman": {
                "regulators": ["CIMA"],
                "key_regulations": ["VASP Law", "MAL"]
            },
            "BVI": {
                "regulators": ["FSC"],
                "key_regulations": ["SIBA", "AML Code"]
            },
            "Hong Kong": {
                "regulators": ["SFC"],
                "key_regulations": ["AMLO", "SFO"]
            }
        }
        
        # Common KYC/AML requirements
        self.kyc_aml_requirements = [
            "customer identification",
            "identity verification",
            "beneficial ownership",
            "source of funds",
            "risk assessment",
            "ongoing monitoring",
            "suspicious transaction reporting",
            "record keeping",
            "screening against sanctions lists",
            "travel rule compliance"
        ]
    
    def analyze_compliance(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze compliance information for a crypto fund.
        
        Args:
            fund_data: Fund information from document analysis
            
        Returns:
            Dict with compliance analysis results
        """
        logger.info("Starting compliance analysis")
        
        # Extract compliance information
        compliance_data = fund_data.get("compliance_data", {})
        kyc_aml = compliance_data.get("kyc_aml", [])
        regulatory_status = compliance_data.get("regulatory_status", [])
        tax_considerations = compliance_data.get("tax_considerations", [])
        
        # Initialize compliance analysis results
        compliance_analysis = {
            "jurisdictions": [],
            "registrations": [],
            "kyc_aml_score": 0,
            "regulatory_score": 0,
            "tax_compliance_score": 0,
            "overall_compliance_score": 0,
            "compliance_gaps": [],
            "recommendations": [],
            "analysis_date": datetime.now().isoformat()
        }
        
        # Identify operating jurisdictions
        identified_jurisdictions = self._identify_jurisdictions(fund_data)
        compliance_analysis["jurisdictions"] = identified_jurisdictions
        
        # Identify regulatory registrations
        registrations = self._identify_registrations(regulatory_status)
        compliance_analysis["registrations"] = registrations
        
        # Score KYC/AML procedures
        kyc_aml_assessment = self._assess_kyc_aml(kyc_aml)
        compliance_analysis.update(kyc_aml_assessment)
        
        # Score regulatory compliance
        regulatory_assessment = self._assess_regulatory_compliance(
            identified_jurisdictions, 
            registrations,
            fund_data
        )
        compliance_analysis.update(regulatory_assessment)
        
        # Score tax compliance
        tax_assessment = self._assess_tax_compliance(tax_considerations, identified_jurisdictions)
        compliance_analysis.update(tax_assessment)
        
        # Calculate overall compliance score
        kyc_weight = 0.3
        regulatory_weight = 0.5
        tax_weight = 0.2
        
        overall_score = (
            (compliance_analysis["kyc_aml_score"] * kyc_weight) +
            (compliance_analysis["regulatory_score"] * regulatory_weight) +
            (compliance_analysis["tax_compliance_score"] * tax_weight)
        )
        
        compliance_analysis["overall_compliance_score"] = overall_score
        compliance_analysis["compliance_level"] = self._determine_compliance_level(overall_score)
        
        # Generate recommendations
        compliance_analysis["recommendations"] = self._generate_recommendations(compliance_analysis)
        
        logger.info(f"Compliance analysis complete with score: {overall_score:.1f}")
        return compliance_analysis
    
    def _identify_jurisdictions(self, fund_data: Dict[str, Any]) -> List[str]:
        """
        Identify jurisdictions where the fund operates or is registered.
        
        Args:
            fund_data: Fund information from document analysis
            
        Returns:
            List of identified jurisdictions
        """
        identified_jurisdictions = []
        
        # Check all text content for jurisdiction mentions
        all_text = str(fund_data).lower()
        
        # Check for each jurisdiction and its regulators
        for jurisdiction, info in self.jurisdictions.items():
            jurisdiction_lower = jurisdiction.lower()
            
            # Check if jurisdiction name is mentioned
            if jurisdiction_lower in all_text:
                identified_jurisdictions.append(jurisdiction)
                continue
                
            # Check if any regulators from this jurisdiction are mentioned
            for regulator in info["regulators"]:
                if regulator.lower() in all_text:
                    identified_jurisdictions.append(jurisdiction)
                    break
        
        # If no jurisdictions identified, check for common terms
        if not identified_jurisdictions:
            if "cayman" in all_text or "offshore" in all_text:
                identified_jurisdictions.append("Cayman")
            if "bvi" in all_text or "british virgin" in all_text:
                identified_jurisdictions.append("BVI")
            if "delaware" in all_text or "llc" in all_text:
                identified_jurisdictions.append("US")
        
        # If still no jurisdictions, default to most common
        if not identified_jurisdictions:
            identified_jurisdictions = ["Cayman", "US"]  # Most common for crypto funds
        
        return identified_jurisdictions
    
    def _identify_registrations(self, regulatory_status: List[str]) -> List[Dict[str, str]]:
        """
        Identify specific regulatory registrations from the regulatory status information.
        
        Args:
            regulatory_status: List of regulatory status statements
            
        Returns:
            List of identified registrations with regulator and jurisdiction
        """
        registrations = []
        
        for status in regulatory_status:
            status_lower = status.lower()
            
            # Check for US regulators
            if "sec" in status_lower:
                registrations.append({"regulator": "SEC", "jurisdiction": "US"})
            if "fincen" in status_lower:
                registrations.append({"regulator": "FinCEN", "jurisdiction": "US"})
            if "cftc" in status_lower:
                registrations.append({"regulator": "CFTC", "jurisdiction": "US"})
            if "finra" in status_lower:
                registrations.append({"regulator": "FINRA", "jurisdiction": "US"})
            
            # Check for EU regulators
            if "mica" in status_lower:
                registrations.append({"regulator": "MiCA", "jurisdiction": "EU"})
            if "vasp" in status_lower and "eu" in status_lower:
                registrations.append({"regulator": "VASP", "jurisdiction": "EU"})
            
            # Check for UK regulators
            if "fca" in status_lower:
                registrations.append({"regulator": "FCA", "jurisdiction": "UK"})
            
            # Check for Singapore regulator
            if "mas" in status_lower:
                registrations.append({"regulator": "MAS", "jurisdiction": "Singapore"})
            
            # Check for Cayman regulator
            if "cima" in status_lower:
                registrations.append({"regulator": "CIMA", "jurisdiction": "Cayman"})
            
            # Check for BVI regulator
            if "fsc" in status_lower and ("bvi" in status_lower or "virgin" in status_lower):
                registrations.append({"regulator": "FSC", "jurisdiction": "BVI"})
            
            # Check for Swiss regulator
            if "finma" in status_lower:
                registrations.append({"regulator": "FINMA", "jurisdiction": "Switzerland"})
            
            # Check for Hong Kong regulator
            if "sfc" in status_lower and ("hong" in status_lower or "hk" in status_lower):
                registrations.append({"regulator": "SFC", "jurisdiction": "Hong Kong"})
        
        return registrations
    
    def _assess_kyc_aml(self, kyc_aml: List[str]) -> Dict[str, Any]:
        """
        Assess KYC/AML procedures against common requirements.
        
        Args:
            kyc_aml: List of KYC/AML procedure statements
            
        Returns:
            Dict with KYC/AML assessment
        """
        assessment = {
            "kyc_aml_score": 0,
            "kyc_aml_details": {
                "strengths": [],
                "gaps": []
            }
        }
        
        if not kyc_aml:
            assessment["kyc_aml_details"]["gaps"].append("No KYC/AML procedures documented")
            return assessment
        
        # Convert all statements to a single lowercase string for easier matching
        kyc_aml_text = " ".join(kyc_aml).lower()
        
        # Check for each KYC/AML requirement
        fulfilled_requirements = []
        missing_requirements = []
        
        for requirement in self.kyc_aml_requirements:
            requirement_lower = requirement.lower()
            
            # Check if the requirement is mentioned
            if requirement_lower in kyc_aml_text:
                fulfilled_requirements.append(requirement)
            else:
                # Check for alternative terminology
                if requirement == "customer identification" and any(term in kyc_aml_text for term in ["kyc", "know your customer", "identify client"]):
                    fulfilled_requirements.append(requirement)
                elif requirement == "identity verification" and any(term in kyc_aml_text for term in ["id verification", "verify identity", "identify verification"]):
                    fulfilled_requirements.append(requirement)
                elif requirement == "beneficial ownership" and any(term in kyc_aml_text for term in ["ubo", "ultimate beneficial", "real owner"]):
                    fulfilled_requirements.append(requirement)
                elif requirement == "source of funds" and any(term in kyc_aml_text for term in ["sof", "fund source", "money source"]):
                    fulfilled_requirements.append(requirement)
                elif requirement == "suspicious transaction reporting" and any(term in kyc_aml_text for term in ["str", "suspicious activity", "sar"]):
                    fulfilled_requirements.append(requirement)
                else:
                    missing_requirements.append(requirement)
        
        # Add fulfilled requirements as strengths
        assessment["kyc_aml_details"]["strengths"] = [f"Implements {req}" for req in fulfilled_requirements]
        
        # Add missing requirements as gaps
        assessment["kyc_aml_details"]["gaps"] = [f"No mention of {req}" for req in missing_requirements]
        
        # Calculate KYC/AML score (0-100)
        if self.kyc_aml_requirements:
            kyc_aml_score = (len(fulfilled_requirements) / len(self.kyc_aml_requirements)) * 100
        else:
            kyc_aml_score = 0
            
        assessment["kyc_aml_score"] = kyc_aml_score
        
        return assessment
    
    def _assess_regulatory_compliance(self, jurisdictions: List[str], registrations: List[Dict[str, str]], 
                                     fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess regulatory compliance across identified jurisdictions.
        
        Args:
            jurisdictions: List of operating jurisdictions
            registrations: List of regulatory registrations
            fund_data: Fund information from document analysis
            
        Returns:
            Dict with regulatory compliance assessment
        """
        assessment = {
            "regulatory_score": 0,
            "regulatory_details": {
                "compliant_jurisdictions": [],
                "non_compliant_jurisdictions": [],
                "regulatory_gaps": []
            }
        }
        
        if not jurisdictions:
            assessment["regulatory_details"]["regulatory_gaps"].append("No clear jurisdictions identified")
            return assessment
        
        compliant_jurisdictions = []
        non_compliant_jurisdictions = []
        regulatory_gaps = []
        
        # Check each jurisdiction for compliance
        for jurisdiction in jurisdictions:
            # Get the registrations for this jurisdiction
            jurisdiction_registrations = [reg for reg in registrations if reg["jurisdiction"] == jurisdiction]
            
            # If US jurisdiction, check for relevant registrations
            if jurisdiction == "US":
                if not any(reg["regulator"] in ["SEC", "FinCEN", "CFTC"] for reg in jurisdiction_registrations):
                    non_compliant_jurisdictions.append(jurisdiction)
                    regulatory_gaps.append(f"Missing required US registrations (SEC, FinCEN, or CFTC)")
                else:
                    compliant_jurisdictions.append(jurisdiction)
            
            # If EU jurisdiction, check for MiCA or VASP registration
            elif jurisdiction == "EU":
                if not any(reg["regulator"] in ["MiCA", "VASP"] for reg in jurisdiction_registrations):
                    non_compliant_jurisdictions.append(jurisdiction)
                    regulatory_gaps.append(f"Missing required EU registration under MiCA or as VASP")
                else:
                    compliant_jurisdictions.append(jurisdiction)
            
            # For other jurisdictions, check if at least one registration exists
            else:
                jurisdiction_info = self.jurisdictions.get(jurisdiction, {"regulators": []})
                expected_regulators = jurisdiction_info["regulators"]
                
                if not any(reg["regulator"] in expected_regulators for reg in jurisdiction_registrations):
                    non_compliant_jurisdictions.append(jurisdiction)
                    regulatory_gaps.append(f"No registrations found for {jurisdiction}")
                else:
                    compliant_jurisdictions.append(jurisdiction)
        
        # Check fund activities against regulatory requirements
        fund_info = fund_data.get("fund_info", {})
        portfolio_data = fund_data.get("portfolio_data", {})
        
        # Extract fund activities
        activities = []
        if portfolio_data:
            for asset, allocation in portfolio_data.items():
                asset_lower = asset.lower()
                
                if any(term in asset_lower for term in ["lending", "borrow", "interest"]):
                    activities.append("lending")
                if any(term in asset_lower for term in ["stake", "validator", "staking"]):
                    activities.append("staking")
                if any(term in asset_lower for term in ["trade", "swap", "dex"]):
                    activities.append("trading")
                if any(term in asset_lower for term in ["derivative", "futures", "option"]):
                    activities.append("derivatives")
        
        # Check if regulatory registrations match activities
        if "US" in jurisdictions:
            if "lending" in activities and not any(reg["regulator"] == "SEC" for reg in registrations):
                regulatory_gaps.append("Lending activities may require SEC registration")
            
            if "derivatives" in activities and not any(reg["regulator"] == "CFTC" for reg in registrations):
                regulatory_gaps.append("Derivatives trading may require CFTC registration")
        
        # Store results in assessment
        assessment["regulatory_details"]["compliant_jurisdictions"] = compliant_jurisdictions
        assessment["regulatory_details"]["non_compliant_jurisdictions"] = non_compliant_jurisdictions
        assessment["regulatory_details"]["regulatory_gaps"] = regulatory_gaps
        
        # Calculate regulatory score
        if jurisdictions:
            regulatory_score = (len(compliant_jurisdictions) / len(jurisdictions)) * 100
            
            # Adjust score based on regulatory gaps
            regulatory_score = max(0, regulatory_score - (len(regulatory_gaps) * 10))
        else:
            regulatory_score = 0
            
        assessment["regulatory_score"] = regulatory_score
        
        return assessment
    
    def _assess_tax_compliance(self, tax_considerations: List[str], jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Assess tax compliance considerations.
        
        Args:
            tax_considerations: List of tax consideration statements
            jurisdictions: List of operating jurisdictions
            
        Returns:
            Dict with tax compliance assessment
        """
        assessment = {
            "tax_compliance_score": 0,
            "tax_details": {
                "strengths": [],
                "gaps": []
            }
        }
        
        if not tax_considerations:
            assessment["tax_details"]["gaps"].append("No tax considerations documented")
            return assessment
        
        # Convert all statements to a single lowercase string for easier matching
        tax_text = " ".join(tax_considerations).lower()
        
        # Key tax compliance aspects to check for
        tax_aspects = [
            "tax reporting",
            "gain/loss calculation",
            "cross-border considerations",
            "investor reporting",
            "tax optimization"
        ]
        
        # Check for each tax aspect
        fulfilled_aspects = []
        missing_aspects = []
        
        for aspect in tax_aspects:
            aspect_lower = aspect.lower()
            
            # Check if the aspect is mentioned
            if aspect_lower in tax_text:
                fulfilled_aspects.append(aspect)
            else:
                # Check for alternative terminology
                if aspect == "tax reporting" and any(term in tax_text for term in ["file tax", "tax return", "tax filing"]):
                    fulfilled_aspects.append(aspect)
                elif aspect == "gain/loss calculation" and any(term in tax_text for term in ["capital gain", "profit calculation", "tax basis"]):
                    fulfilled_aspects.append(aspect)
                elif aspect == "cross-border considerations" and any(term in tax_text for term in ["international tax", "foreign tax", "offshore"]):
                    fulfilled_aspects.append(aspect)
                elif aspect == "investor reporting" and any(term in tax_text for term in ["k-1", "investor report", "tax statement"]):
                    fulfilled_aspects.append(aspect)
                else:
                    missing_aspects.append(aspect)
        
        # Check for jurisdiction-specific tax considerations
        jurisdiction_tax_mentions = 0
        
        for jurisdiction in jurisdictions:
            if jurisdiction.lower() in tax_text:
                jurisdiction_tax_mentions += 1
                
            # Check for US-specific tax terms
            if jurisdiction == "US" and any(term in tax_text for term in ["k-1", "irs", "fatca", "federal tax"]):
                jurisdiction_tax_mentions += 1
        
        # Add fulfilled aspects as strengths
        assessment["tax_details"]["strengths"] = [f"Addresses {aspect}" for aspect in fulfilled_aspects]
        
        # Add missing aspects as gaps
        assessment["tax_details"]["gaps"] = [f"No mention of {aspect}" for aspect in missing_aspects]
        
        if jurisdiction_tax_mentions < len(jurisdictions):
            assessment["tax_details"]["gaps"].append("Tax considerations don't address all operating jurisdictions")
        
        # Calculate tax compliance score (0-100)
        tax_score = 0
        if tax_aspects:
            # 60% of score from fulfilled aspects
            aspect_score = (len(fulfilled_aspects) / len(tax_aspects)) * 60
            
            # 40% of score from jurisdiction coverage
            jurisdiction_score = min(40, (jurisdiction_tax_mentions / max(1, len(jurisdictions))) * 40)
            
            tax_score = aspect_score + jurisdiction_score
            
        assessment["tax_compliance_score"] = tax_score
        
        return assessment
    
    def _determine_compliance_level(self, compliance_score: float) -> str:
        """
        Determine compliance level based on overall score.
        
        Args:
            compliance_score: Overall compliance score (0-100)
            
        Returns:
            Compliance level category
        """
        if compliance_score >= 85:
            return "High Compliance"
        elif compliance_score >= 70:
            return "Substantial Compliance"
        elif compliance_score >= 50:
            return "Partial Compliance"
        elif compliance_score >= 30:
            return "Minimal Compliance"
        else:
            return "Non-Compliant"
    
     #def _generate_recommendations(self, compliance_analysis: Dict[str, Any]) -> List[str]:
        # """
        # Generate recommendations to improve compliance.
        
       