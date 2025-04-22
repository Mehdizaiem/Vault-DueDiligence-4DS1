"""
Compliance Analyzer Module

This module analyzes compliance-related information for crypto funds.
It evaluates regulatory status, KYC/AML procedures, and tax considerations
across different jurisdictions.
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
    Analyzes compliance aspects of crypto funds across different regulatory frameworks.
    """
    
    def __init__(self, data_retriever):
        """
        Initialize the compliance analyzer.
        
        Args:
            data_retriever: Object that retrieves data from collections
        """
        self.retriever = data_retriever
        
        # Define regulatory bodies by jurisdiction
        self.regulatory_bodies = {
            "US": ["SEC", "FinCEN", "CFTC", "FINRA", "OCC", "FDIC", "IRS", "MSB"],
            "EU": ["ESMA", "EBA", "MiCA", "AMLD5", "VASP", "NCA"],
            "UK": ["FCA", "PRA", "HMRC", "MLR"],
            "Singapore": ["MAS", "PSA"],
            "Hong Kong": ["SFC", "HKMA"],
            "Japan": ["FSA", "JFSA"],
            "Switzerland": ["FINMA", "SBA", "VQF"],
            "Cayman Islands": ["CIMA", "CIR"],
            "BVI": ["FSC", "SIBA"],
            "Bermuda": ["BMA", "DABA"],
            "Australia": ["ASIC", "AUSTRAC"],
            "Canada": ["CSA", "OSC", "FINTRAC"],
            "Dubai": ["DFSA", "VARA"]
        }
        
        # Define compliance requirements by jurisdiction
        self.compliance_requirements = {
            "US": {
                "kyc_aml": ["Customer identification", "KYC verification", "BSA compliance", "SAR filing", "AML program"],
                "licensing": ["MSB registration", "Broker-dealer license", "Money transmitter", "Trust charter"],
                "reporting": ["1099 forms", "FinCEN Form 114", "Form 8949", "Schedule D"]
            },
            "EU": {
                "kyc_aml": ["KYC/CDD procedures", "AMLD5 compliance", "UBO registry", "PEP screening"],
                "licensing": ["VASP registration", "MiCA license", "E-money license"],
                "reporting": ["Tax information exchange", "Cross-border reporting", "DAC8 compliance"]
            },
            "UK": {
                "kyc_aml": ["MLR 2017 procedures", "AML risk assessment", "Customer due diligence"],
                "licensing": ["FCA registration", "EMI authorization"],
                "reporting": ["HMRC reporting", "CTF compliance"]
            },
            "Singapore": {
                "kyc_aml": ["MAS Notice PSN02", "Risk-based approach", "CDD measures"],
                "licensing": ["PSA license", "CMS license", "MAS exemption"],
                "reporting": ["GST reporting", "Annual returns"]
            },
            "Switzerland": {
                "kyc_aml": ["FINMA circular 2016/7", "AMLA compliance", "Risk categorization"],
                "licensing": ["FINMA authorization", "SRO membership", "Banking license"],
                "reporting": ["AEOI reporting", "Tax compliance"]
            },
            "Cayman Islands": {
                "kyc_aml": ["AML regulations", "Cayman KYC standards"],
                "licensing": ["CIMA registration", "Virtual Asset License"],
                "reporting": ["Economic substance", "FATCA/CRS reporting"]
            }
        }
        
        # Define regulatory frameworks specific to crypto assets
        self.crypto_frameworks = {
            "US": {
                "securities": {
                    "body": "SEC",
                    "tests": ["Howey Test", "Investment contract", "Securities offering"],
                    "exemptions": ["Regulation D", "Regulation A", "Regulation S", "Rule 144A"]
                },
                "commodities": {
                    "body": "CFTC",
                    "tests": ["Commodity definition", "Futures contract"],
                    "exemptions": ["Retail commodity exception", "Actual delivery"]
                },
                "money_transmission": {
                    "body": "FinCEN & State regulators",
                    "tests": ["BSA definition", "Money transmission"],
                    "exemptions": ["Payment processor exemption", "Software provider"]
                }
            },
            "EU": {
                "mica": {
                    "body": "ESMA & NCAs",
                    "categories": ["Asset-referenced tokens", "E-money tokens", "Other crypto-assets"],
                    "requirements": ["White paper", "Authorization", "Ongoing obligations"]
                },
                "aml": {
                    "body": "National FIUs",
                    "requirements": ["VASP registration", "Travel rule", "Risk assessment"]
                }
            },
            "Singapore": {
                "ps_act": {
                    "body": "MAS",
                    "categories": ["Digital payment token service", "E-money issuance"],
                    "requirements": ["License", "Capital requirements", "AML/CFT"]
                },
                "securities": {
                    "body": "MAS",
                    "tests": ["Capital markets products", "Collective investment scheme"],
                    "exemptions": ["Private placement", "Small offers"]
                }
            }
        }
    
    def analyze_compliance(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive compliance analysis on a crypto fund.
        
        Args:
            fund_data: Fund information from document analysis
            
        Returns:
            Dict with compliance assessment results
        """
        logger.info("Starting comprehensive compliance analysis")
        
        # Extract compliance information
        compliance_data = fund_data.get("compliance_data", {})
        fund_info = fund_data.get("fund_info", {})
        
        # Initialize compliance assessment object
        compliance_assessment = {
            "overall_compliance_score": 0.0,
            "compliance_level": "Unknown",
            "jurisdictions": [],
            "regulatory_status": {},
            "kyc_aml_assessment": {},
            "tax_compliance": {},
            "compliance_gaps": [],
            "recommended_actions": [],
            "analysis_date": datetime.now().isoformat()
        }
        
        # Identify relevant jurisdictions
        jurisdictions = self._identify_jurisdictions(compliance_data, fund_info)
        compliance_assessment["jurisdictions"] = jurisdictions
        
        # Assess regulatory status in each jurisdiction
        regulatory_status = self._assess_regulatory_status(compliance_data, fund_info, jurisdictions)
        compliance_assessment["regulatory_status"] = regulatory_status
        
        # Assess KYC/AML procedures
        kyc_aml_assessment = self._assess_kyc_aml(compliance_data, jurisdictions)
        compliance_assessment["kyc_aml_assessment"] = kyc_aml_assessment
        
        # Assess tax compliance
        tax_compliance = self._assess_tax_compliance(compliance_data, jurisdictions)
        compliance_assessment["tax_compliance"] = tax_compliance
        
        # Calculate overall compliance score
        compliance_assessment["overall_compliance_score"] = self._calculate_overall_score(
            regulatory_status, kyc_aml_assessment, tax_compliance
        )
        
        # Determine compliance level
        compliance_assessment["compliance_level"] = self._determine_compliance_level(
            compliance_assessment["overall_compliance_score"]
        )
        
        # Identify compliance gaps
        compliance_assessment["compliance_gaps"] = self._identify_compliance_gaps(
            regulatory_status, kyc_aml_assessment, tax_compliance, jurisdictions
        )
        
        # Recommend actions
        compliance_assessment["recommended_actions"] = self._recommend_actions(
            compliance_assessment["compliance_gaps"], jurisdictions
        )
        
        logger.info(f"Compliance analysis complete. Overall score: {compliance_assessment['overall_compliance_score']}, " + 
                   f"Level: {compliance_assessment['compliance_level']}")
        
        return compliance_assessment
    
    def _identify_jurisdictions(self, compliance_data: Dict[str, Any], fund_info: Dict[str, Any]) -> List[str]:
        """
        Identify relevant jurisdictions for the fund.
        
        Args:
            compliance_data: Compliance information from document analysis
            fund_info: Basic fund information
            
        Returns:
            List of identified jurisdictions
        """
        jurisdictions = set()
        
        # Extract from regulatory status mentions
        regulatory_status = compliance_data.get("regulatory_status", [])
        for status in regulatory_status:
            status_lower = status.lower()
            
            # Check for mentions of jurisdictions and their regulatory bodies
            for jurisdiction, bodies in self.regulatory_bodies.items():
                # Direct mention of jurisdiction
                if jurisdiction.lower() in status_lower:
                    jurisdictions.add(jurisdiction)
                    
                # Mention of regulatory body
                for body in bodies:
                    if body.lower() in status_lower:
                        jurisdictions.add(jurisdiction)
        
        # Check other fund information for jurisdiction mentions
        fund_text = str(fund_info)
        for jurisdiction in self.regulatory_bodies.keys():
            if jurisdiction.lower() in fund_text.lower():
                jurisdictions.add(jurisdiction)
        
        # If no jurisdictions identified, use default important ones
        if not jurisdictions:
            jurisdictions = {"US", "EU", "Cayman Islands"}
            logger.warning("No specific jurisdictions identified, using default set")
        
        return list(jurisdictions)
    
    def _assess_regulatory_status(self, compliance_data: Dict[str, Any], fund_info: Dict[str, Any], 
                                 jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Assess regulatory status in each identified jurisdiction.
        
        Args:
            compliance_data: Compliance information from document analysis
            fund_info: Basic fund information
            jurisdictions: List of relevant jurisdictions
            
        Returns:
            Dict with regulatory status assessment
        """
        regulatory_status = {}
        
        # Extract regulatory status mentions
        status_mentions = compliance_data.get("regulatory_status", [])
        
        # Assess each jurisdiction
        for jurisdiction in jurisdictions:
            jurisdiction_bodies = self.regulatory_bodies.get(jurisdiction, [])
            
            # Initialize status for this jurisdiction
            regulatory_status[jurisdiction] = {
                "registered_with": [],
                "registration_confirmed": False,
                "registration_status": "Unknown",
                "compliance_score": 0.0
            }
            
            # Check for mentions of regulatory bodies
            for body in jurisdiction_bodies:
                body_lower = body.lower()
                
                for status in status_mentions:
                    status_lower = status.lower()
                    
                    if body_lower in status_lower:
                        regulatory_status[jurisdiction]["registered_with"].append(body)
                        
                        # Check if registration is confirmed
                        if any(term in status_lower for term in ["registered", "authorized", "licensed", "approved"]):
                            regulatory_status[jurisdiction]["registration_confirmed"] = True
                            if "pending" in status_lower:
                                regulatory_status[jurisdiction]["registration_status"] = "Pending"
                            else:
                                regulatory_status[jurisdiction]["registration_status"] = "Registered"
            
            # Set registration status if not already set
            if regulatory_status[jurisdiction]["registered_with"] and not regulatory_status[jurisdiction]["registration_confirmed"]:
                regulatory_status[jurisdiction]["registration_status"] = "Mentioned"
            elif not regulatory_status[jurisdiction]["registered_with"]:
                regulatory_status[jurisdiction]["registration_status"] = "No Registration Found"
            
            # Calculate compliance score for this jurisdiction
            if regulatory_status[jurisdiction]["registration_status"] == "Registered":
                regulatory_status[jurisdiction]["compliance_score"] = 85.0 # Starting point for registered entities
                
                # Additional score for more comprehensive registrations
                if len(regulatory_status[jurisdiction]["registered_with"]) > 1:
                    regulatory_status[jurisdiction]["compliance_score"] += 5.0 * (len(regulatory_status[jurisdiction]["registered_with"]) - 1)
                    
            elif regulatory_status[jurisdiction]["registration_status"] == "Pending":
                regulatory_status[jurisdiction]["compliance_score"] = 60.0
            elif regulatory_status[jurisdiction]["registration_status"] == "Mentioned":
                regulatory_status[jurisdiction]["compliance_score"] = 40.0
            else:
                regulatory_status[jurisdiction]["compliance_score"] = 20.0
                
            # Cap at 100
            regulatory_status[jurisdiction]["compliance_score"] = min(100.0, regulatory_status[jurisdiction]["compliance_score"])
        
        return regulatory_status
    
    def _assess_kyc_aml(self, compliance_data: Dict[str, Any], jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Assess KYC/AML procedures against requirements in relevant jurisdictions.
        
        Args:
            compliance_data: Compliance information from document analysis
            jurisdictions: List of relevant jurisdictions
            
        Returns:
            Dict with KYC/AML assessment
        """
        kyc_aml_assessment = {
            "procedures_mentioned": [],
            "coverage_score": 0.0,
            "jurisdiction_coverage": {}
        }
        
        # Extract KYC/AML procedures
        kyc_aml_procedures = compliance_data.get("kyc_aml", [])
        kyc_aml_text = " ".join(kyc_aml_procedures).lower()
        
        # Identify mentioned procedures
        all_kyc_terms = set()
        for terms in [req for jur in self.compliance_requirements.values() for req in jur.get("kyc_aml", [])]:
            all_kyc_terms.update(terms.lower() for term in terms)
        
        for procedure in kyc_aml_procedures:
            kyc_aml_assessment["procedures_mentioned"].append(procedure)
        
        # Check coverage by jurisdiction
        total_score = 0.0
        applicable_jurisdictions = 0
        
        for jurisdiction in jurisdictions:
            if jurisdiction not in self.compliance_requirements:
                continue
                
            applicable_jurisdictions += 1
            jurisdiction_requirements = self.compliance_requirements[jurisdiction].get("kyc_aml", [])
            
            if not jurisdiction_requirements:
                continue
                
            # Check each requirement
            met_requirements = []
            for requirement in jurisdiction_requirements:
                requirement_lower = requirement.lower()
                
                # Check if the requirement is mentioned
                if any(term in kyc_aml_text for term in requirement_lower.split()):
                    met_requirements.append(requirement)
            
            # Calculate coverage for this jurisdiction
            if jurisdiction_requirements:
                coverage_pct = (len(met_requirements) / len(jurisdiction_requirements)) * 100
            else:
                coverage_pct = 0.0
                
            kyc_aml_assessment["jurisdiction_coverage"][jurisdiction] = {
                "requirements": jurisdiction_requirements,
                "met_requirements": met_requirements,
                "coverage_percentage": coverage_pct
            }
            
            total_score += coverage_pct
        
        # Calculate overall coverage score
        if applicable_jurisdictions > 0:
            kyc_aml_assessment["coverage_score"] = total_score / applicable_jurisdictions
        
        return kyc_aml_assessment
    
    def _assess_tax_compliance(self, compliance_data: Dict[str, Any], jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Assess tax compliance measures against requirements in relevant jurisdictions.
        
        Args:
            compliance_data: Compliance information from document analysis
            jurisdictions: List of relevant jurisdictions
            
        Returns:
            Dict with tax compliance assessment
        """
        tax_compliance = {
            "measures_mentioned": [],
            "coverage_score": 0.0,
            "jurisdiction_coverage": {}
        }
        
        # Extract tax considerations
        tax_considerations = compliance_data.get("tax_considerations", [])
        tax_text = " ".join(tax_considerations).lower()
        
        # Identify mentioned tax measures
        for measure in tax_considerations:
            tax_compliance["measures_mentioned"].append(measure)
        
        # Check coverage by jurisdiction
        total_score = 0.0
        applicable_jurisdictions = 0
        
        for jurisdiction in jurisdictions:
            if jurisdiction not in self.compliance_requirements:
                continue
                
            applicable_jurisdictions += 1
            jurisdiction_requirements = self.compliance_requirements[jurisdiction].get("reporting", [])
            
            if not jurisdiction_requirements:
                continue
                
            # Check each requirement
            met_requirements = []
            for requirement in jurisdiction_requirements:
                requirement_lower = requirement.lower()
                
                # Check if the requirement is mentioned
                if any(term in tax_text for term in requirement_lower.split()):
                    met_requirements.append(requirement)
            
            # Calculate coverage for this jurisdiction
            if jurisdiction_requirements:
                coverage_pct = (len(met_requirements) / len(jurisdiction_requirements)) * 100
            else:
                coverage_pct = 0.0
                
            tax_compliance["jurisdiction_coverage"][jurisdiction] = {
                "requirements": jurisdiction_requirements,
                "met_requirements": met_requirements,
                "coverage_percentage": coverage_pct
            }
            
            total_score += coverage_pct
        
        # Calculate overall coverage score
        if applicable_jurisdictions > 0:
            tax_compliance["coverage_score"] = total_score / applicable_jurisdictions
        
        return tax_compliance
    
    def _calculate_overall_score(self, regulatory_status: Dict[str, Any], 
                               kyc_aml_assessment: Dict[str, Any],
                               tax_compliance: Dict[str, Any]) -> float:
        """
        Calculate overall compliance score.
        
        Args:
            regulatory_status: Regulatory status assessment
            kyc_aml_assessment: KYC/AML assessment
            tax_compliance: Tax compliance assessment
            
        Returns:
            Overall compliance score (0-100)
        """
        # Weights for different components
        weights = {
            "regulatory": 0.5,
            "kyc_aml": 0.3,
            "tax": 0.2
        }
        
        # Calculate average regulatory score across jurisdictions
        regulatory_scores = [status["compliance_score"] for status in regulatory_status.values()]
        avg_regulatory_score = sum(regulatory_scores) / len(regulatory_scores) if regulatory_scores else 0.0
        
        # Get KYC/AML score
        kyc_aml_score = kyc_aml_assessment.get("coverage_score", 0.0)
        
        # Get tax compliance score
        tax_score = tax_compliance.get("coverage_score", 0.0)
        
        # Calculate weighted overall score
        overall_score = (
            (avg_regulatory_score * weights["regulatory"]) +
            (kyc_aml_score * weights["kyc_aml"]) +
            (tax_score * weights["tax"])
        )
        
        return overall_score
    
    def _determine_compliance_level(self, score: float) -> str:
        """
        Determine compliance level based on overall score.
        
        Args:
            score: Overall compliance score (0-100)
            
        Returns:
            Compliance level category
        """
        if score >= 85:
            return "High"
        elif score >= 70:
            return "Substantial"
        elif score >= 50:
            return "Moderate"
        elif score >= 30:
            return "Partial"
        else:
            return "Low"
    
    def _identify_compliance_gaps(self, regulatory_status: Dict[str, Any],
                                kyc_aml_assessment: Dict[str, Any],
                                tax_compliance: Dict[str, Any],
                                jurisdictions: List[str]) -> List[str]:
        """
        Identify specific compliance gaps based on assessment.
        
        Args:
            regulatory_status: Regulatory status assessment
            kyc_aml_assessment: KYC/AML assessment
            tax_compliance: Tax compliance assessment
            jurisdictions: List of relevant jurisdictions
            
        Returns:
            List of identified compliance gaps
        """
        gaps = []
        
        # Check regulatory gaps
        for jurisdiction, status in regulatory_status.items():
            if status["registration_status"] == "No Registration Found":
                gaps.append(f"No regulatory registration found for {jurisdiction}")
            elif status["registration_status"] == "Mentioned" or status["registration_status"] == "Pending":
                gaps.append(f"Incomplete regulatory registration in {jurisdiction}")
        
        # Check KYC/AML gaps
        for jurisdiction, coverage in kyc_aml_assessment.get("jurisdiction_coverage", {}).items():
            if coverage["coverage_percentage"] < 50:
                missing = set(coverage["requirements"]) - set(coverage["met_requirements"])
                if missing:
                    gaps.append(f"Missing KYC/AML procedures for {jurisdiction}: {', '.join(list(missing)[:3])}")
                else:
                    gaps.append(f"Inadequate KYC/AML procedures for {jurisdiction}")
        
        # Check tax compliance gaps
        for jurisdiction, coverage in tax_compliance.get("jurisdiction_coverage", {}).items():
            if coverage["coverage_percentage"] < 50:
                missing = set(coverage["requirements"]) - set(coverage["met_requirements"])
                if missing:
                    gaps.append(f"Missing tax compliance measures for {jurisdiction}: {', '.join(list(missing)[:3])}")
                else:
                    gaps.append(f"Inadequate tax compliance measures for {jurisdiction}")
        
        # Check for specific crypto-related requirements
        for jurisdiction in jurisdictions:
            if jurisdiction in self.crypto_frameworks:
                if jurisdiction == "US":
                    if "securities" in self.crypto_frameworks[jurisdiction]:
                        # Check for securities compliance if dealing with tokens
                        if "token" in str(kyc_aml_assessment).lower() or "security" in str(kyc_aml_assessment).lower():
                            if not any("sec" in str(status).lower() for status in regulatory_status.values()):
                                gaps.append("Potential securities compliance gap for token offerings in US market")
                
                elif jurisdiction == "EU":
                    if "mica" in self.crypto_frameworks[jurisdiction]:
                        # Check for MiCA compliance
                        if not any("mica" in str(status).lower() for status in regulatory_status.values()):
                            gaps.append("Missing MiCA compliance measures for EU operations")
        
        return gaps
    
    def _recommend_actions(self, gaps: List[str], jurisdictions: List[str]) -> List[str]:
        """
        Recommend compliance actions based on identified gaps.
        
        Args:
            gaps: List of identified compliance gaps
            jurisdictions: List of relevant jurisdictions
            
        Returns:
            List of recommended actions
        """
        actions = []
        
        # Process each gap and recommend specific actions
        for gap in gaps:
            if "No regulatory registration" in gap:
                jurisdiction = re.search(r"for (\w+(?:\s+\w+)*)", gap)
                if jurisdiction:
                    jur = jurisdiction.group(1)
                    if jur in self.regulatory_bodies:
                        bodies = ", ".join(self.regulatory_bodies[jur][:3])
                        actions.append(f"Initiate registration process with relevant authorities in {jur} ({bodies})")
                    else:
                        actions.append(f"Consult with legal counsel regarding registration requirements in {jur}")
            
            elif "Incomplete regulatory registration" in gap:
                jurisdiction = re.search(r"in (\w+(?:\s+\w+)*)", gap)
                if jurisdiction:
                    actions.append(f"Complete pending registration process in {jurisdiction.group(1)}")
            
            elif "Missing KYC/AML procedures" in gap:
                jurisdiction = re.search(r"for (\w+(?:\s+\w+)*)", gap)
                if jurisdiction:
                    jur = jurisdiction.group(1)
                    actions.append(f"Implement comprehensive KYC/AML procedures compliant with {jur} requirements")
            
            elif "Missing tax compliance" in gap:
                jurisdiction = re.search(r"for (\w+(?:\s+\w+)*)", gap)
                if jurisdiction:
                    jur = jurisdiction.group(1)
                    actions.append(f"Consult with tax advisors regarding {jur} reporting requirements")
            
            elif "securities compliance gap" in gap:
                actions.append("Conduct Howey Test analysis to determine if tokens constitute securities")
                actions.append("Consider SEC registration or applicable exemptions for token offerings")
            
            elif "MiCA compliance" in gap:
                actions.append("Prepare for MiCA compliance including white paper requirements and operational procedures")
        
        # Add generic recommendations if no specific gaps or limited actions
        if len(actions) < 2:
            high_priority_jurisdictions = [j for j in jurisdictions if j in ["US", "EU", "UK", "Singapore"]]
            
            if high_priority_jurisdictions:
                for jur in high_priority_jurisdictions[:2]:
                    if jur == "US":
                        actions.append("Conduct comprehensive US regulatory review covering SEC, FinCEN and CFTC requirements")
                    elif jur == "EU":
                        actions.append("Prepare for MiCA implementation with updated compliance procedures")
                    elif jur == "UK":
                        actions.append("Review FCA registration requirements for crypto asset businesses")
                    elif jur == "Singapore":
                        actions.append("Assess licensing requirements under Payment Services Act for digital token services")
        
        # Add general best practice recommendations
        if len(actions) < 3:
            actions.append("Implement ongoing compliance monitoring system for regulatory changes")
            actions.append("Conduct periodic compliance audits with specialized crypto compliance experts")
        
        return actions

    def get_regulatory_frameworks(self, jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Get detailed information about regulatory frameworks in specified jurisdictions.
        
        Args:
            jurisdictions: List of jurisdictions to get information for
            
        Returns:
            Dict with regulatory framework information
        """
        frameworks = {}
        
        for jurisdiction in jurisdictions:
            if jurisdiction in self.crypto_frameworks:
                frameworks[jurisdiction] = self.crypto_frameworks[jurisdiction]
        
        return frameworks
    
    def get_compliance_requirements(self, jurisdictions: List[str]) -> Dict[str, Any]:
        """
        Get detailed compliance requirements for specified jurisdictions.
        
        Args:
            jurisdictions: List of jurisdictions to get requirements for
            
        Returns:
            Dict with compliance requirements by jurisdiction
        """
        requirements = {}
        
        for jurisdiction in jurisdictions:
            if jurisdiction in self.compliance_requirements:
                requirements[jurisdiction] = self.compliance_requirements[jurisdiction]
        
        return requirements
    
    def analyze_regulatory_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze regulatory documents from the document collections.
        
        Args:
            documents: List of regulatory documents
            
        Returns:
            Dict with regulatory analysis
        """
        regulatory_analysis = {
            "relevant_regulations": [],
            "jurisdiction_coverage": {},
            "key_requirements": {},
            "recent_developments": []
        }
        
        # Process each document
        for document in documents:
            document_type = document.get("document_type", "").lower()
            title = document.get("title", "").lower()
            content = document.get("content", "").lower()
            
            # Check if it's a regulatory document
            if "regulation" in document_type or "compliance" in document_type or "regulatory" in title:
                # Identify jurisdiction
                jurisdiction = None
                for jur in self.regulatory_bodies.keys():
                    if jur.lower() in title or jur.lower() in content[:500]:
                        jurisdiction = jur
                        break
                
                if not jurisdiction:
                    jurisdiction = "Global"
                
                # Add to relevant regulations
                regulatory_analysis["relevant_regulations"].append({
                    "title": document.get("title", "Untitled"),
                    "jurisdiction": jurisdiction,
                    "date": document.get("date", "Unknown"),
                    "id": document.get("id", "")
                })
                
                # Update jurisdiction coverage
                if jurisdiction not in regulatory_analysis["jurisdiction_coverage"]:
                    regulatory_analysis["jurisdiction_coverage"][jurisdiction] = []
                
                regulatory_analysis["jurisdiction_coverage"][jurisdiction].append(document.get("title", "Untitled"))
                
                # Extract key requirements
                requirement_patterns = [
                    r"(must|shall|required to|obligation to) ([^\.]+)",
                    r"requirement[s]? (?:to|for) ([^\.]+)",
                    r"mandatory ([^\.]+)"
                ]
                
                for pattern in requirement_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        for match in matches[:3]:  # Limit to first 3 matches
                            requirement = match[1] if isinstance(match, tuple) else match
                            
                            if jurisdiction not in regulatory_analysis["key_requirements"]:
                                regulatory_analysis["key_requirements"][jurisdiction] = []
                                
                            if requirement not in regulatory_analysis["key_requirements"][jurisdiction]:
                                regulatory_analysis["key_requirements"][jurisdiction].append(requirement)
                
                # Check for recent developments
                if "new" in content[:500] or "update" in content[:500] or "recent" in content[:500]:
                    regulatory_analysis["recent_developments"].append({
                        "title": document.get("title", "Untitled"),
                        "jurisdiction": jurisdiction,
                        "date": document.get("date", "Unknown")
                    })
        
        return regulatory_analysis