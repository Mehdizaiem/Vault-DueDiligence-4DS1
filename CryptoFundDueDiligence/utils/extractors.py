"""
Extractors Module

This module provides specialized extractors for pulling various data points
from crypto fund documents including addresses, metrics, dates, and more.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseExtractor:
    """Base class for all extractors with common utility methods"""
    
    def extract_numeric(self, text: str) -> Optional[float]:
        """Extract a numeric value from text"""
        if not text:
            return None
        
        # Find all numbers in the text
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches:
            return float(matches[0])
        return None
    
    def extract_percentage(self, text: str) -> Optional[float]:
        """Extract a percentage value from text and convert to decimal (0-1)"""
        if not text:
            return None
        
        # Find percentage pattern
        match = re.search(r'(\d+\.?\d*)%', text)
        if match:
            return float(match.group(1)) / 100
        
        # If no percentage symbol, try to find a reasonable numeric value
        numeric = self.extract_numeric(text)
        if numeric is not None:
            # Assume it's already decimal if < 1, otherwise convert from percentage
            if numeric <= 1:
                return numeric
            else:
                return numeric / 100
        
        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing excess whitespace and standardizing format"""
        if not text:
            return ""
        
        # Replace multiple whitespace with single space
        normalized = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        normalized = normalized.strip()
        return normalized


class AddressExtractor(BaseExtractor):
    """Extracts blockchain addresses from text"""
    
    def extract_ethereum_addresses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract Ethereum addresses and associated metadata.
        
        Args:
            text: Document text to search
            
        Returns:
            List of dictionaries with address and metadata
        """
        results = []
        
        # Pattern for Ethereum addresses
        eth_pattern = r'0x[a-fA-F0-9]{40}\b'
        matches = re.finditer(eth_pattern, text)
        
        for match in matches:
            address = match.group(0)
            
            # Extract context around address (200 chars before and after)
            start_pos = max(0, match.start() - 200)
            end_pos = min(len(text), match.end() + 200)
            context = text[start_pos:end_pos]
            
            # Try to determine wallet type
            wallet_type = self.determine_wallet_type(address, context)
            
            # Try to extract balance
            balance = self.extract_balance(context)
            
            # Try to extract security features
            security = self.extract_security_features(context)
            
            results.append({
                "address": address,
                "type": wallet_type,
                "balance": balance,
                "security_features": security
            })
        
        return results
    
    def determine_wallet_type(self, address: str, context: str) -> str:
        """
        Determine the wallet type based on surrounding context.
        
        Args:
            address: The Ethereum address
            context: Text surrounding the address
            
        Returns:
            String indicating wallet type
        """
        context_lower = context.lower()
        
        # Check for type indicators in context
        if any(term in context_lower for term in ["cold storage", "cold wallet", "hardware", "offline"]):
            return "Cold Storage"
        elif any(term in context_lower for term in ["hot wallet", "operational", "trading"]):
            return "Hot Wallet"
        elif any(term in context_lower for term in ["staking", "validator", "stake"]):
            return "Staking Wallet"
        elif any(term in context_lower for term in ["treasury", "reserve"]):
            return "Treasury"
        else:
            return "Unknown"
    
    def extract_balance(self, context: str) -> Optional[Dict[str, Any]]:
        """
        Extract balance information from context.
        
        Args:
            context: Text surrounding an address
            
        Returns:
            Dictionary with balance amount and currency
        """
        # Look for balance patterns
        balance_match = re.search(r'(\d+(?:\.\d+)?)\s*(ETH|BTC|SOL|USDT|USDC)', context)
        if balance_match:
            return {
                "amount": float(balance_match.group(1)),
                "currency": balance_match.group(2)
            }
        return None
    
    def extract_security_features(self, context: str) -> List[str]:
        """
        Extract security features from context.
        
        Args:
            context: Text surrounding an address
            
        Returns:
            List of security features
        """
        features = []
        context_lower = context.lower()
        
        # Check for common security features
        if re.search(r'multi[\-\s]sig|multisig', context_lower):
            # Try to extract the signature threshold (e.g., 2-of-3, 3/5)
            threshold_match = re.search(r'(\d+)[\-\s]of[\-\s](\d+)|(\d+)/(\d+)', context)
            if threshold_match:
                groups = threshold_match.groups()
                if groups[0] and groups[1]:  # Format: "X-of-Y"
                    features.append(f"Multi-sig ({groups[0]}-of-{groups[1]})")
                elif groups[2] and groups[3]:  # Format: "X/Y"
                    features.append(f"Multi-sig ({groups[2]}-of-{groups[3]})")
            else:
                features.append("Multi-sig")
                
        if re.search(r'hardware|ledger|trezor', context_lower):
            features.append("Hardware Security")
            
        if re.search(r'time[\-\s]lock|timelock', context_lower):
            features.append("Time-locked")
            
        if re.search(r'daily\s+limit|transaction\s+limit', context_lower):
            limit_match = re.search(r'limit\D*(\d+(?:\.\d+)?)', context_lower)
            if limit_match:
                features.append(f"Daily limit ({limit_match.group(1)} ETH)")
            else:
                features.append("Daily limits")
        
        return features


class MetricsExtractor(BaseExtractor):
    """Extracts financial metrics from text"""
    
    def extract_aum(self, text: str) -> Optional[float]:
        """
        Extract Assets Under Management (AUM) value in millions.
        
        Args:
            text: Text to search
            
        Returns:
            AUM value in millions or None if not found
        """
        if not text:
            return None
        
        # Look for AUM patterns
        aum_pattern = r'(?:AUM|Assets Under Management|Total Assets)[:\s]*[\$£€]?([0-9.,]+\s*[BMK]?illion|\d+(?:\.\d+)?)'
        match = re.search(aum_pattern, text, re.IGNORECASE)
        
        if not match:
            return None
        
        aum_str = match.group(1)
        
        # Remove currency symbols and commas
        aum_str = re.sub(r'[$£€,]', '', aum_str)
        
        # Handle "Million", "Billion" suffixes
        if "million" in aum_str.lower() or "m" in aum_str.lower():
            multiplier = 1
        elif "billion" in aum_str.lower() or "b" in aum_str.lower():
            multiplier = 1000
        elif "thousand" in aum_str.lower() or "k" in aum_str.lower():
            multiplier = 0.001
        else:
            multiplier = 0.000001  # Assume raw number is in currency units
        
        # Extract the numeric part
        value = re.search(r'\d+(?:\.\d+)?', aum_str)
        if value:
            return float(value.group(0)) * multiplier
        return None
    
    def extract_fee_structure(self, text: str) -> Dict[str, float]:
        """
        Extract fee structure (management fee, performance fee, etc.).
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary with fee types and values
        """
        fees = {}
        
        # Management fee
        mgmt_fee_pattern = r'(?:Management Fee)[:\s]+([0-9.]+%)'
        mgmt_match = re.search(mgmt_fee_pattern, text, re.IGNORECASE)
        if mgmt_match:
            fees["management_fee"] = self.extract_percentage(mgmt_match.group(1))
        
        # Performance fee
        perf_fee_pattern = r'(?:Performance Fee)[:\s]+([0-9.]+%\s*(?:\(.*?\))?)'
        perf_match = re.search(perf_fee_pattern, text, re.IGNORECASE)
        if perf_match:
            fees["performance_fee"] = self.extract_percentage(perf_match.group(1))
        
        # Entry fee
        entry_fee_pattern = r'(?:Entry Fee|Subscription Fee)[:\s]+([0-9.]+%)'
        entry_match = re.search(entry_fee_pattern, text, re.IGNORECASE)
        if entry_match:
            fees["entry_fee"] = self.extract_percentage(entry_match.group(1))
        
        # Exit fee
        exit_fee_pattern = r'(?:Exit Fee|Redemption Fee)[:\s]+([0-9.]+%)'
        exit_match = re.search(exit_fee_pattern, text, re.IGNORECASE)
        if exit_match:
            fees["exit_fee"] = self.extract_percentage(exit_match.group(1))
        
        return fees
    
    def extract_performance_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract performance metrics (returns, Sharpe ratio, etc.).
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Historical returns
        yearly_returns = {}
        years = ["2020", "2021", "2022", "2023", "2024", "2025"]
        
        for year in years:
            pattern = fr'{year}[:\s]+([+-]?\d+(?:\.\d+)?%)'
            match = re.search(pattern, text)
            if match:
                yearly_returns[year] = self.extract_percentage(match.group(1))
        
        if yearly_returns:
            metrics["yearly_returns"] = yearly_returns
        
        # Sharpe ratio
        sharpe_pattern = r'Sharpe(?:\s+Ratio)?[:\s]+([0-9.]+)'
        sharpe_match = re.search(sharpe_pattern, text, re.IGNORECASE)
        if sharpe_match:
            metrics["sharpe_ratio"] = float(sharpe_match.group(1))
        
        # Maximum drawdown
        drawdown_pattern = r'(?:Maximum Drawdown|Max Drawdown)[:\s]+([0-9.]+%)'
        drawdown_match = re.search(drawdown_pattern, text, re.IGNORECASE)
        if drawdown_match:
            metrics["max_drawdown"] = self.extract_percentage(drawdown_match.group(1))
        
        # Volatility
        volatility_pattern = r'(?:Volatility|Standard Deviation)[:\s]+([0-9.]+%)'
        volatility_match = re.search(volatility_pattern, text, re.IGNORECASE)
        if volatility_match:
            metrics["volatility"] = self.extract_percentage(volatility_match.group(1))
        
        return metrics
    
    def extract_risk_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract risk metrics and ratings.
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary with risk metrics
        """
        metrics = {}
        
        # Overall risk score
        risk_score_pattern = r'(?:Risk Score|Risk Rating)[:\s]+([0-9.]+\s*%?|[0-9.]+\s*\/\s*[0-9.]+)'
        risk_match = re.search(risk_score_pattern, text, re.IGNORECASE)
        if risk_match:
            score_text = risk_match.group(1)
            
            # Handle different formats
            if '/' in score_text:  # Format: X/Y
                parts = score_text.split('/')
                if len(parts) == 2:
                    try:
                        metrics["risk_score"] = float(parts[0].strip()) / float(parts[1].strip())
                    except ValueError:
                        pass
            elif '%' in score_text:  # Percentage
                metrics["risk_score"] = self.extract_percentage(score_text)
            else:  # Simple number
                try:
                    metrics["risk_score"] = float(score_text.strip())
                except ValueError:
                    pass
        
        # Individual risk factors
        risk_factors = []
        risk_factor_pattern = r'([A-Za-z][A-Za-z\s]+Risk)[:\s]+([0-9.]+)'
        for match in re.finditer(risk_factor_pattern, text, re.IGNORECASE):
            factor = match.group(1).strip()
            rating = float(match.group(2))
            risk_factors.append({"factor": factor, "rating": rating})
        
        if risk_factors:
            metrics["risk_factors"] = risk_factors
        
        return metrics


class PortfolioExtractor(BaseExtractor):
    """Extracts portfolio allocation information"""
    
    def extract_allocations(self, text: str) -> Dict[str, float]:
        """
        Extract portfolio allocations from text.
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary mapping asset names to allocation percentages (0-1)
        """
        allocations = {}
        
        # Find a portfolio or allocation section
        section_headers = [
            "Portfolio Allocation", 
            "Asset Allocation",
            "Current Allocations",
            "Holdings",
            "Investment Portfolio"
        ]
        
        # Try to find the relevant section
        section_text = None
        for header in section_headers:
            pattern = fr"{header}[:\s]*([^.]*(?:\.[^.]*)*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            # Look for allocation patterns in the entire text
            section_text = text
        
        # Look for allocation entries
        alloc_patterns = [
            r'([A-Za-z][A-Za-z\s]+):\s*([\d.]+%)',  # Pattern: "Asset: XX%"
            r'([A-Za-z][A-Za-z\s]+)\s+([\d.]+%)',   # Pattern: "Asset XX%"
            r'[•\-\*]\s*([A-Za-z][A-Za-z\s]+):\s*([\d.]+%)',  # Pattern: "• Asset: XX%"
            r'[•\-\*]\s*([A-Za-z][A-Za-z\s]+)\s+([\d.]+%)'    # Pattern: "• Asset XX%"
        ]
        
        for pattern in alloc_patterns:
            for match in re.finditer(pattern, section_text):
                asset = match.group(1).strip()
                percentage = match.group(2)
                
                # Convert percentage to float (0-1)
                value = self.extract_percentage(percentage)
                if value is not None:
                    allocations[asset] = value
        
        return allocations
    
    def extract_top_holdings(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract information about top holdings.
        
        Args:
            text: Text to search
            
        Returns:
            List of dictionaries with holding information
        """
        holdings = []
        
        # Find a section about top holdings
        section_headers = [
            "Top Holdings",
            "Key Holdings",
            "Major Positions",
            "Top Positions"
        ]
        
        # Try to find the relevant section
        section_text = None
        for header in section_headers:
            pattern = fr"{header}[:\s]*([^.]*(?:\.[^.]*)*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return holdings
        
        # Extract individual holdings
        # Various patterns to cover different formatting styles
        holding_patterns = [
            r'([A-Za-z0-9][A-Za-z0-9\s.]+):\s*\$?([\d.]+[BMK]?)',  # Asset: $amount
            r'([A-Za-z0-9][A-Za-z0-9\s.]+)\s+\$?([\d.]+[BMK]?)',   # Asset $amount
            r'[•\-\*]\s*([A-Za-z0-9][A-Za-z0-9\s.]+):\s*\$?([\d.]+[BMK]?)',  # • Asset: $amount
            r'[•\-\*]\s*([A-Za-z0-9][A-Za-z0-9\s.]+)\s+\$?([\d.]+[BMK]?)',   # • Asset $amount
            r'([A-Za-z0-9][A-Za-z0-9\s.]+):\s*\$?([\d.]+)\s*\(([^)]+)\)',    # Asset: $amount (description)
            r'([A-Za-z0-9][A-Za-z0-9\s.]+)\s+\$?([\d.]+)\s*\(([^)]+)\)'      # Asset $amount (description)
        ]
        
        for pattern in holding_patterns:
            if "(" in pattern:  # Pattern with description in parentheses
                for match in re.finditer(pattern, section_text):
                    asset = match.group(1).strip()
                    amount_str = match.group(2)
                    description = match.group(3).strip() if match.group(3) else None
                    
                    # Normalize amount (convert K, M, B to actual numbers)
                    amount = self._normalize_amount(amount_str)
                    
                    holdings.append({
                        "asset": asset,
                        "amount": amount,
                        "description": description
                    })
            else:  # Pattern without description
                for match in re.finditer(pattern, section_text):
                    asset = match.group(1).strip()
                    amount_str = match.group(2)
                    
                    # Normalize amount (convert K, M, B to actual numbers)
                    amount = self._normalize_amount(amount_str)
                    
                    holdings.append({
                        "asset": asset,
                        "amount": amount
                    })
        
        return holdings
    
    def _normalize_amount(self, amount_str: str) -> float:
        """
        Normalize amount strings with K, M, B suffixes to actual numbers.
        
        Args:
            amount_str: Amount as string, potentially with suffix
            
        Returns:
            Normalized float value
        """
        # Remove any currency symbols
        amount_str = re.sub(r'[$£€,]', '', amount_str)
        
        # Check for suffixes
        if 'B' in amount_str or 'b' in amount_str:
            multiplier = 1_000_000_000
            amount_str = amount_str.replace('B', '').replace('b', '')
        elif 'M' in amount_str or 'm' in amount_str:
            multiplier = 1_000_000
            amount_str = amount_str.replace('M', '').replace('m', '')
        elif 'K' in amount_str or 'k' in amount_str:
            multiplier = 1_000
            amount_str = amount_str.replace('K', '').replace('k', '')
        else:
            multiplier = 1
        
        try:
            return float(amount_str) * multiplier
        except ValueError:
            return 0.0


class DateExtractor(BaseExtractor):
    """Extracts and normalizes dates from text"""
    
    def extract_dates(self, text: str) -> Dict[str, Optional[datetime]]:
        """
        Extract key dates from text.
        
        Args:
            text: Text to search
            
        Returns:
            Dictionary with date types and datetime objects
        """
        dates = {}
        
        # Launch date / inception date
        launch_pattern = r'(?:Launch Date|Inception Date|Founded|Established)[:\s]+([^\n]+)'
        launch_match = re.search(launch_pattern, text, re.IGNORECASE)
        if launch_match:
            date_str = launch_match.group(1).strip()
            dates["launch_date"] = self._parse_date(date_str)
        
        # Report date / as of date
        report_pattern = r'(?:Report Date|As of|Data as of)[:\s]+([^\n]+)'
        report_match = re.search(report_pattern, text, re.IGNORECASE)
        if report_match:
            date_str = report_match.group(1).strip()
            dates["report_date"] = self._parse_date(date_str)
        
        # Audit date
        audit_pattern = r'(?:Last Audit|Audit Date|Audited on)[:\s]+([^\n]+)'
        audit_match = re.search(audit_pattern, text, re.IGNORECASE)
        if audit_match:
            date_str = audit_match.group(1).strip()
            dates["audit_date"] = self._parse_date(date_str)
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Try to parse a date string into a datetime object.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_str:
            return None
        
        # Clean up the date string
        date_str = date_str.strip()
        
        # Try various date formats
        formats = [
            '%Y-%m-%d',              # 2023-01-15
            '%B %d, %Y',             # January 15, 2023
            '%b %d, %Y',             # Jan 15, 2023
            '%d %B %Y',              # 15 January 2023
            '%d %b %Y',              # 15 Jan 2023
            '%m/%d/%Y',              # 01/15/2023
            '%d/%m/%Y',              # 15/01/2023
            '%Y/%m/%d',              # 2023/01/15
            '%B %Y',                 # January 2023
            '%b %Y',                 # Jan 2023
            '%m-%Y',                 # 01-2023
            '%Y'                     # 2023
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If all format attempts fail, try to extract just the year
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            try:
                return datetime(int(year_match.group(1)), 1, 1)  # Use January 1st of the year
            except ValueError:
                pass
        
        return None


class TeamExtractor(BaseExtractor):
    """Extracts information about team members"""
    
    def extract_team_members(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract information about team members.
        
        Args:
            text: Text to search
            
        Returns:
            List of dictionaries with team member information
        """
        team_members = []
        
        # Find team section
        section_headers = [
            "Team Assessment",
            "Key Personnel",
            "Management Team",
            "Team",
            "Personnel"
        ]
        
        # Try to find the relevant section
        section_text = None
        for header in section_headers:
            pattern = fr"{header}[:\s]*([^.]*(?:\.[^.]*)*)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                section_text = match.group(1)
                break
        
        if not section_text:
            return team_members
        
        # Extract individual team members
        # Look for patterns like "Name, Title" or "Name - Title"
        member_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s*([A-Za-z\s]+)',     # Name, Title
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[-–]\s*([A-Za-z\s]+)', # Name - Title
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\n|\r\n)(?:[-•*]\s*)?([A-Za-z\s]+)'  # Name\nTitle or Name\n- Title
        ]
        
        for pattern in member_patterns:
            for match in re.finditer(pattern, section_text):
                name = match.group(1).strip()
                title = match.group(2).strip()
                
                # Look for more information about this person
                person_info = self._extract_person_details(name, section_text)
                
                team_members.append({
                    "name": name,
                    "title": title,
                    **person_info
                })
        
        return team_members
    
    def _extract_person_details(self, name: str, text: str) -> Dict[str, Any]:
        """
        Extract additional details about a team member.
        
        Args:
            name: Person's name
            text: Text to search
            
        Returns:
            Dictionary with additional details
        """
        details = {}
        
        # Find context around the person's name
        name_parts = name.split()
        if len(name_parts) < 2:
            return details
        
        # Look for paragraphs that mention the person
        first_name = name_parts[0]
        last_name = name_parts[1]
        
        # Try to find a section about this person
        person_section_pattern = fr"({name}[^.]*(?:\.[^.]*)+)"
        section_match = re.search(person_section_pattern, text)
        
        if section_match:
            person_text = section_match.group(1)
            
            # Extract background information
            background = []
            for line in person_text.split('\n'):
                if line.strip() and line.strip() != name and not re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+', line.strip()):
                    # Remove bullet points and other markers
                    clean_line = re.sub(r'^[•\-\*]\s*', '', line.strip())
                    if clean_line:
                        background.append(clean_line)
            
            if background:
                details["background"] = background
            
            # Look for experience duration
            experience_pattern = r'(\d+)(?:\+)?\s*(?:years|yr)(?:\s+of)?\s+experience'
            exp_match = re.search(experience_pattern, person_text, re.IGNORECASE)
            if exp_match:
                details["years_experience"] = int(exp_match.group(1))
            
            # Look for previous companies
            company_patterns = [
                r'(?:former|previously|ex)[^.]*?\b(at|with)\s+([A-Z][A-Za-z\s]+)',
                r'(?:worked|employed)[^.]*?\b(at|with)\s+([A-Z][A-Za-z\s]+)'
            ]
            
            previous_companies = []
            for pattern in company_patterns:
                for match in re.finditer(pattern, person_text, re.IGNORECASE):
                    company = match.group(2).strip()
                    if company and company not in previous_companies:
                        previous_companies.append(company)
            
            if previous_companies:
                details["previous_companies"] = previous_companies
        
        return details