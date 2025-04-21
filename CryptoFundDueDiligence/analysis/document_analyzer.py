"""
Document Analyzer Module

This module extracts key information from uploaded documents about crypto funds.
It uses NLP techniques to identify fund details, wallets, and other critical information.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Advanced document analyzer for crypto fund due diligence documents.
    Extracts structured information from documents about crypto funds.
    """
    
    def __init__(self):
        """Initialize the document analyzer with patterns for data extraction"""
        # Common regex patterns
        self.patterns = {
            "ethereum_address": r"0x[a-fA-F0-9]{40}\b",
            "fund_name": r"(?:Fund Name|Name)[:\s]+([^\n]+)",
            "aum": r"(?:AUM|Assets Under Management)[:\s]*[\$£€]?([0-9.,]+\s*[BMK]?illion|\d+(?:\.\d+)?)",
            "management_fee": r"(?:Management Fee)[:\s]+([0-9.]+%)",
            "performance_fee": r"(?:Performance Fee)[:\s]+([0-9.]+%\s*(?:\(.*?\))?)",
            "strategy": r"(?:Strategy|Investment Strategy)[:\s]+([^\n.]+)",
            "launch_date": r"(?:Launch Date|Founded|Established)[:\s]+([^\n]+)",
            "min_investment": r"(?:Minimum Investment)[:\s]+([^\n]+)",
            "lock_up": r"(?:Lock-up|Lock up|Lockup)[:\s]+([^\n]+)",
            "risk_score": r"(?:Risk Score|Risk Rating)[:\s]+([0-9.]+\s*%?|[0-9.]+\s*\/\s*[0-9.]+)",
            "security_features": r"(?:Security Features|Security)[:\s]+([^\n]+)",
            "portfolio_allocation": r"(?:Allocation|Portfolio|Holdings)[:\s]+([^\n.]+(?:\n(?:[ \t]+[^\n]+))*)",
        }
        
        # Initialize specialized extractors
        self.wallet_extractor = WalletExtractor()
        self.portfolio_extractor = PortfolioExtractor()
        self.risk_extractor = RiskExtractor()
        self.compliance_extractor = ComplianceExtractor()
        self.team_extractor = TeamExtractor()
        
    def analyze_document(self, content: str) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on document content.
        
        Args:
            content (str): Document text content
            
        Returns:
            Dict[str, Any]: Extracted structured information
        """
        logger.info("Starting document analysis")
        
        # Extract basic fund information
        fund_info = self._extract_fund_info(content)
        
        # Extract wallets information
        wallet_data = self.wallet_extractor.extract_wallets(content)
        
        # Extract portfolio information
        portfolio_data = self.portfolio_extractor.extract_portfolio(content)
        
        # Extract risk assessment
        risk_data = self.risk_extractor.extract_risk_factors(content)
        
        # Extract compliance information
        compliance_data = self.compliance_extractor.extract_compliance_info(content)
        
        # Extract team information
        team_data = self.team_extractor.extract_team_info(content)
        
        # Combine all extracted information
        analysis_results = {
            "fund_info": fund_info,
            "wallet_data": wallet_data,
            "portfolio_data": portfolio_data,
            "risk_data": risk_data,
            "compliance_data": compliance_data,
            "team_data": team_data,
            "analysis_confidence": self._calculate_confidence(fund_info, wallet_data, portfolio_data)
        }
        
        logger.info(f"Document analysis completed with {analysis_results['analysis_confidence']}% confidence")
        return analysis_results
    
    def _extract_fund_info(self, content: str) -> Dict[str, Any]:
        """
        Extract basic fund information from the document.
        
        Args:
            content (str): Document content
            
        Returns:
            Dict[str, Any]: Fund information
        """
        fund_info = {}
        
        # Extract each field using regex patterns
        for field, pattern in self.patterns.items():
            if field in ["ethereum_address", "portfolio_allocation"]:
                continue  # Skip complex fields handled by specialized extractors
                
            matches = re.search(pattern, content, re.IGNORECASE)
            if matches:
                fund_info[field] = matches.group(1).strip()
        
        # Clean and normalize fields
        if "aum" in fund_info:
            fund_info["aum"] = self._normalize_aum(fund_info["aum"])
            
        if "management_fee" in fund_info:
            fund_info["management_fee"] = self._normalize_percentage(fund_info["management_fee"])
            
        if "performance_fee" in fund_info:
            fund_info["performance_fee"] = self._normalize_percentage(fund_info["performance_fee"])
        
        return fund_info
    
    def _normalize_aum(self, aum_str: str) -> float:
        """Convert AUM string to a float value in millions"""
        # Remove currency symbols and commas
        aum_str = re.sub(r'[$£€,]', '', aum_str)
        
        # Handle "Million", "Billion" suffixes
        if "million" in aum_str.lower() or "m" in aum_str.lower():
            multiplier = 1
        elif "billion" in aum_str.lower() or "b" in aum_str.lower():
            multiplier = 1000
        else:
            multiplier = 0.000001  # Assume raw number is in currency units
        
        # Extract the numeric part
        value = re.search(r'\d+(?:\.\d+)?', aum_str)
        if value:
            return float(value.group(0)) * multiplier
        return 0.0
    
    def _normalize_percentage(self, pct_str: str) -> float:
        """Convert percentage string to a float value"""
        value = re.search(r'\d+(?:\.\d+)?', pct_str)
        if value:
            return float(value.group(0))
        return 0.0
    
    def _calculate_confidence(self, fund_info, wallet_data, portfolio_data) -> int:
        """
        Calculate confidence score for the analysis.
        
        Returns:
            int: Confidence percentage (0-100)
        """
        score = 0
        max_score = 100
        
        # Check essential fund information
        if fund_info.get("fund_name"):
            score += 15
        if fund_info.get("aum"):
            score += 10
        if fund_info.get("strategy"):
            score += 10
            
        # Check wallet information
        if wallet_data and len(wallet_data) > 0:
            score += 20
            
        # Check portfolio information
        if portfolio_data and len(portfolio_data) > 0:
            score += 20
            
        # Ensure score doesn't exceed 100
        return min(score, max_score)


class WalletExtractor:
    """Extract wallet information from documents"""
    
    def extract_wallets(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract wallet addresses and related information.
        
        Args:
            content (str): Document content
            
        Returns:
            List[Dict[str, Any]]: List of wallet data
        """
        wallet_data = []
        
        # Find Ethereum wallet addresses
        eth_addresses = re.findall(r'0x[a-fA-F0-9]{40}\b', content)
        
        # For each found address, try to find related information
        for address in eth_addresses:
            wallet_info = {
                "address": address,
                "type": self._determine_wallet_type(address, content),
                "balance": self._extract_wallet_balance(address, content),
                "security_features": self._extract_security_features(address, content)
            }
            wallet_data.append(wallet_info)
            
        return wallet_data
    
    def _determine_wallet_type(self, address: str, content: str) -> str:
        """Determine the wallet type based on surrounding context"""
        # Define search windows before and after the address
        address_index = content.find(address)
        if address_index == -1:
            return "Unknown"
            
        # Look at text before the address (within reasonable window)
        context_before = content[max(0, address_index - 200):address_index]
        
        # Check for wallet type indicators
        if any(term in context_before.lower() for term in ["cold", "storage", "hardware", "offline"]):
            return "Cold Storage"
        elif any(term in context_before.lower() for term in ["hot", "operational", "trading"]):
            return "Hot Wallet"
        elif any(term in context_before.lower() for term in ["staking", "validator", "stake"]):
            return "Staking Wallet"
        elif any(term in context_before.lower() for term in ["treasury", "reserve"]):
            return "Treasury"
        else:
            return "General"
    
    def _extract_wallet_balance(self, address: str, content: str) -> Optional[float]:
        """Extract the wallet balance from context"""
        # Find the balance associated with this address
        address_index = content.find(address)
        if address_index == -1:
            return None
            
        # Look at text after the address (within reasonable window)
        context_after = content[address_index:min(len(content), address_index + 200)]
        
        # Match balance patterns
        balance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ETH|BTC|SOL|ADA)', context_after)
        if balance_match:
            return float(balance_match.group(1))
            
        return None
    
    def _extract_security_features(self, address: str, content: str) -> List[str]:
        """Extract security features for this wallet from context"""
        features = []
        
        # Find context around the address
        address_index = content.find(address)
        if address_index == -1:
            return features
            
        # Look at surrounding text
        context = content[max(0, address_index - 200):min(len(content), address_index + 200)]
        
        # Check for common security features
        if re.search(r'multi[\-\s]sig|multisig', context, re.IGNORECASE):
            # Try to extract the signature threshold (e.g., 2-of-3, 3/5)
            threshold_match = re.search(r'(\d+)[\-\s]of[\-\s](\d+)|(\d+)/(\d+)', context, re.IGNORECASE)
            if threshold_match:
                groups = threshold_match.groups()
                if groups[0] and groups[1]:  # Format: "X-of-Y"
                    features.append(f"Multi-sig ({groups[0]}-of-{groups[1]})")
                elif groups[2] and groups[3]:  # Format: "X/Y"
                    features.append(f"Multi-sig ({groups[2]}-of-{groups[3]})")
            else:
                features.append("Multi-sig")
                
        if re.search(r'hardware|ledger|trezor', context, re.IGNORECASE):
            features.append("Hardware Security")
            
        if re.search(r'time[\-\s]lock|timelock', context, re.IGNORECASE):
            features.append("Time-locked")
            
        if re.search(r'daily\s+limit|transaction\s+limit', context, re.IGNORECASE):
            limit_match = re.search(r'limit\D*(\d+(?:\.\d+)?)', context, re.IGNORECASE)
            if limit_match:
                features.append(f"Daily limit ({limit_match.group(1)} ETH)")
            else:
                features.append("Daily limits")
        
        if not features:
            # If no specific features found, add a generic placeholder
            features.append("Standard security")
            
        return features


class PortfolioExtractor:
    """Extract portfolio allocation information from documents"""
    
    def extract_portfolio(self, content: str) -> Dict[str, float]:
        """
        Extract portfolio allocation data from the document.
        
        Args:
            content (str): Document content
            
        Returns:
            Dict[str, float]: Portfolio allocations (asset: percentage)
        """
        portfolio = {}
        
        # Look for portfolio allocation section
        allocation_sections = [
            "Asset Allocation",
            "Portfolio Allocation", 
            "Current Allocations",
            "Holdings",
            "Investment Portfolio"
        ]
        
        # Find section with allocations
        section_text = None
        for section_name in allocation_sections:
            pattern = f"{section_name}[:\\s]*([\\s\\S]*?)(?:(?:\\n\\s*\\n)|$)"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return portfolio
        
        # Extract allocation entries (typically in format "Asset: Percentage")
        # Handle both percentage notation (50%) and decimal notation (0.5)
        allocations = re.findall(r'([^:]+?):\s*([\d.]+%?|[\d.]+)', section_text)
        
        # If we didn't find entries with colon format, try bullet/list format
        if not allocations:
            # Look for patterns like "• Asset Name: XX%" or "- Asset Name: XX%"
            allocations = re.findall(r'[•\-\*]\s*([^:]+?):\s*([\d.]+%?|[\d.]+)', section_text)
        
        # If still no entries, try a more flexible pattern
        if not allocations:
            # This handles "Asset Name XX%" format without colon
            allocations = re.findall(r'([A-Za-z][A-Za-z\s]+)(?:\s+|:)([\d.]+%?|[\d.]+)', section_text)
        
        # Process all found allocations
        for asset, percentage in allocations:
            asset_name = asset.strip()
            
            # Convert percentage to float
            if '%' in percentage:
                value = float(percentage.replace('%', '').strip()) / 100
            else:
                value = float(percentage.strip())
                
            # Only add if it looks like a reasonable percentage
            if 0 <= value <= 1.0 or (value > 1.0 and value <= 100.0):
                if value > 1.0:  # Convert to decimal if given as percentage without % sign
                    value /= 100.0
                portfolio[asset_name] = value
        
        return portfolio


class RiskExtractor:
    """Extract risk assessment information from documents"""
    
    def extract_risk_factors(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract risk factors and ratings from the document.
        
        Args:
            content (str): Document content
            
        Returns:
            List[Dict[str, Any]]: List of risk factors with ratings and mitigations
        """
        risk_factors = []
        
        # Look for risk assessment section
        risk_sections = [
            "Risk Assessment",
            "Risk Factors",
            "Market Risk",
            "Risk Analysis"
        ]
        
        # Find section with risk information
        section_text = None
        for section_name in risk_sections:
            pattern = f"{section_name}[:\\s]*([\\s\\S]*?)(?:(?:\\n\\s*\\n)|$)"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return risk_factors
        
        # Try to find a risk table (Risk Factor | Rating | Mitigation)
        # First try to find table-formatted risks
        table_risks = re.findall(r'([A-Za-z][A-Za-z\s]+)(?:\s+|:)(?:rating:)?\s*(\d+(?:[.-]\d+)?\s*(?:/\s*\d+)?|\d+%)(?:\s+|:)(.*?)(?=\n|$)', section_text, re.IGNORECASE)
        
        if table_risks:
            for factor, rating, mitigation in table_risks:
                # Normalize rating to a value between 0-10
                normalized_rating = self._normalize_rating(rating)
                
                risk_factors.append({
                    "factor": factor.strip(),
                    "rating": normalized_rating,
                    "mitigation": mitigation.strip() if mitigation.strip() else None
                })
                
        # If no table found, try to extract individual risk mentions
        if not risk_factors:
            # Look for mentions of common risk types
            risk_types = [
                "Price Volatility", "Market Risk", "Smart Contract Risk", "Regulatory Risk",
                "Compliance Risk", "Liquidity Risk", "Security Risk", "Technical Risk",
                "Counterparty Risk", "Concentration Risk", "Operational Risk"
            ]
            
            for risk_type in risk_types:
                pattern = f"{risk_type}[:\\s]+([^\\n.]+)"
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    
                    # Try to find a rating in the description
                    rating_match = re.search(r'(\d+(?:[.-]\d+)?\s*(?:/\s*\d+)?|\d+%)', description)
                    rating = self._normalize_rating(rating_match.group(0)) if rating_match else 5  # Default rating
                    
                    risk_factors.append({
                        "factor": risk_type,
                        "rating": rating,
                        "mitigation": None
                    })
        
        return risk_factors
    
    def _normalize_rating(self, rating_str: str) -> float:
        """
        Normalize various rating formats to a scale of 0-10.
        
        Args:
            rating_str (str): Rating as a string, e.g., "7/10", "70%", "7.5"
            
        Returns:
            float: Normalized rating on scale 0-10
        """
        # Handle percentage format
        if '%' in rating_str:
            percentage = float(rating_str.replace('%', '').strip())
            return percentage / 10
            
        # Handle X/Y format
        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)', rating_str)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            return (numerator / denominator) * 10
            
        # Handle simple number
        try:
            value = float(rating_str.strip())
            # If value appears to be on a 0-5 scale
            if value <= 5.5:
                return value * 2
            # If value appears to be on a 0-100 scale
            elif value > 10:
                return value / 10
            return value
        except ValueError:
            return 5.0  # Default value if parsing fails


class ComplianceExtractor:
    """Extract compliance information from documents"""
    
    def extract_compliance_info(self, content: str) -> Dict[str, Any]:
        """
        Extract compliance framework information from the document.
        
        Args:
            content (str): Document content
            
        Returns:
            Dict[str, Any]: Compliance information
        """
        compliance_data = {
            "kyc_aml": [],
            "regulatory_status": [],
            "tax_considerations": []
        }
        
        # Look for compliance section
        compliance_sections = [
            "Compliance Framework",
            "Regulatory Status",
            "Compliance",
            "Regulatory Compliance",
            "KYC/AML"
        ]
        
        # Find section with compliance information
        section_text = None
        for section_name in compliance_sections:
            pattern = f"{section_name}[:\\s]*([\\s\\S]*?)(?:(?:\\n\\s*\\n)|$)"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return compliance_data
        
        # Extract KYC/AML procedures
        kyc_section = re.search(r'KYC/AML\s+(?:Procedures|Process|Framework)[:\s]*([^.]*(?:\.[^.]*)*)', content, re.IGNORECASE)
        if kyc_section:
            # Split into bullet points or sentences
            kyc_text = kyc_section.group(1)
            kyc_items = [item.strip() for item in re.split(r'[•\-\*\.\n]', kyc_text) if item.strip()]
            compliance_data["kyc_aml"] = kyc_items
        
        # Extract regulatory status
        reg_status_pattern = r'(?:Registered|Registration|Licensed|License|Authorized|Regulated)[:\s]+([^\n.]+)'
        reg_matches = re.findall(reg_status_pattern, section_text, re.IGNORECASE)
        if reg_matches:
            compliance_data["regulatory_status"] = [match.strip() for match in reg_matches]
        
        # Extract tax considerations
        tax_section = re.search(r'Tax\s+(?:Considerations|Status|Reporting)[:\s]*([^.]*(?:\.[^.]*)*)', content, re.IGNORECASE)
        if tax_section:
            # Split into bullet points or sentences
            tax_text = tax_section.group(1)
            tax_items = [item.strip() for item in re.split(r'[•\-\*\.\n]', tax_text) if item.strip()]
            compliance_data["tax_considerations"] = tax_items
        
        return compliance_data


class TeamExtractor:
    """Extract team information from documents"""
    
    def extract_team_info(self, content: str) -> Dict[str, Any]:
        """
        Extract team information from the document.
        
        Args:
            content (str): Document content
            
        Returns:
            Dict[str, Any]: Team information
        """
        team_data = {
            "key_personnel": [],
            "security_team": []
        }
        
        # Look for team section
        team_sections = [
            "Team Assessment",
            "Key Personnel",
            "Management Team", 
            "Team",
            "Personnel"
        ]
        
        # Find section with team information
        section_text = None
        for section_name in team_sections:
            pattern = f"{section_name}[:\\s]*([\\s\\S]*?)(?:(?:\\n\\s*\\n)|$)"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return team_data
        
        # Extract key personnel
        # Look for patterns like "Name, Title" or "Name - Title"
        personnel_matches = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s*([A-Za-z\s]+)(?:\n|$)', section_text)
        
        if personnel_matches:
            for name, title in personnel_matches:
                person_data = {
                    "name": name.strip(),
                    "title": title.strip()
                }
                
                # Look for additional details about this person
                person_pattern = f"{name}[\\s\\S]*?(?:(?:\\n\\n)|$)"
                person_match = re.search(person_pattern, section_text)
                if person_match:
                    person_text = person_match.group(0)
                    
                    # Extract background points
                    background = []
                    for line in person_text.split('\n'):
                        if line.strip() and line.strip() != name and not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', line.strip()):
                            # Remove bullet points and other markers
                            clean_line = re.sub(r'^[•\-\*]\s*', '', line.strip())
                            if clean_line:
                                background.append(clean_line)
                    
                    if background:
                        person_data["background"] = background
                
                # Add to key personnel list
                team_data["key_personnel"].append(person_data)
        
        # Extract security team information if present
        security_section = re.search(r'Security\s+Team[:\s]*([^.]*(?:\.[^.]*)*)', content, re.IGNORECASE)
        if security_section:
            # Split into bullet points or sentences
            security_text = security_section.group(1)
            security_items = [item.strip() for item in re.split(r'[•\-\*\.\n]', security_text) if item.strip()]
            team_data["security_team"] = security_items
        
        return team_data