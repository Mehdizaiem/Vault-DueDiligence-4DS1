"""
Validators Module

This module contains validation functions for ensuring extracted data is correct,
complete, and properly formatted before analysis and reporting.
"""

import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates extracted data for consistency, completeness, and correctness.
    """
    
    @staticmethod
    def validate_ethereum_address(address: str) -> bool:
        """
        Validate if a string is a proper Ethereum address.
        
        Args:
            address (str): The address to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check basic format (0x followed by 40 hex characters)
        if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
            return False
        
        # Optional: Add additional checksum validation if needed
        # (EIP-55 checksum validation would go here)
        
        return True
    
    @staticmethod
    def validate_fund_info(fund_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean fund information.
        
        Args:
            fund_info (Dict): The fund information to validate
            
        Returns:
            Dict: Validated and cleaned fund information
        """
        validated = {}
        
        # Check required fields
        required_fields = ["fund_name"]
        for field in required_fields:
            if field not in fund_info or not fund_info[field]:
                logger.warning(f"Required field '{field}' is missing in fund information")
        
        # Validate and clean fund name
        if "fund_name" in fund_info and fund_info["fund_name"]:
            validated["fund_name"] = fund_info["fund_name"].strip()
        else:
            # Try to infer fund name if missing
            validated["fund_name"] = "Unnamed Fund"
            
        # Validate AUM (Assets Under Management)
        if "aum" in fund_info:
            aum_value = DataValidator.validate_aum(fund_info["aum"])
            if aum_value is not None:
                validated["aum"] = aum_value
        
        # Validate management fee
        if "management_fee" in fund_info:
            mgt_fee = DataValidator.validate_percentage(fund_info["management_fee"])
            if mgt_fee is not None:
                validated["management_fee"] = mgt_fee
        
        # Validate performance fee
        if "performance_fee" in fund_info:
            perf_fee = DataValidator.validate_percentage(fund_info["performance_fee"])
            if perf_fee is not None:
                validated["performance_fee"] = perf_fee
        
        # Validate launch date
        if "launch_date" in fund_info:
            launch_date = DataValidator.validate_date(fund_info["launch_date"])
            if launch_date:
                validated["launch_date"] = launch_date
        
        # Add all other fields with minimal validation
        for key, value in fund_info.items():
            if key not in validated and value:
                validated[key] = value
        
        return validated
    
    @staticmethod
    def validate_aum(aum) -> Optional[float]:
        """
        Validate AUM (Assets Under Management) value.
        
        Args:
            aum: AUM value (can be string, float, or other formats)
            
        Returns:
            float: Validated AUM in millions, or None if invalid
        """
        if isinstance(aum, float):
            return aum
        
        if isinstance(aum, int):
            return float(aum)
        
        if isinstance(aum, str):
            # Remove currency symbols and commas
            aum_str = re.sub(r'[$£€,]', '', aum)
            
            # Extract numeric part
            match = re.search(r'(\d+(?:\.\d+)?)', aum_str)
            if not match:
                return None
                
            value = float(match.group(1))
            
            # Apply multiplier based on suffix
            if 'million' in aum_str.lower() or 'm' in aum_str.lower():
                return value
            elif 'billion' in aum_str.lower() or 'b' in aum_str.lower():
                return value * 1000
            elif 'trillion' in aum_str.lower() or 't' in aum_str.lower():
                return value * 1000000
            elif 'k' in aum_str.lower() or 'thousand' in aum_str.lower():
                return value / 1000
            else:
                # Assume raw value, convert to millions
                return value / 1000000
        
        return None
    
    @staticmethod
    def validate_percentage(percentage) -> Optional[float]:
        """
        Validate a percentage value.
        
        Args:
            percentage: Percentage value (can be string, float, etc.)
            
        Returns:
            float: Percentage as a decimal (0-1), or None if invalid
        """
        if isinstance(percentage, float):
            # If already a float, ensure it's in the right range
            if 0 <= percentage <= 1:
                return percentage
            elif 1 < percentage <= 100:
                return percentage / 100
        
        if isinstance(percentage, int):
            if 0 <= percentage <= 100:
                return percentage / 100
        
        if isinstance(percentage, str):
            # Remove % sign and spaces
            pct_str = percentage.replace('%', '').strip()
            
            # Extract numeric part
            match = re.search(r'(\d+(?:\.\d+)?)', pct_str)
            if not match:
                return None
                
            value = float(match.group(1))
            
            # Convert to decimal based on range
            if 0 <= value <= 1:
                return value
            elif 1 < value <= 100:
                return value / 100
        
        return None
    
    @staticmethod
    def validate_date(date_str: str) -> Optional[str]:
        """
        Validate and standardize a date string.
        
        Args:
            date_str: Date in various formats
            
        Returns:
            str: ISO formatted date (YYYY-MM-DD), or None if invalid
        """
        if not date_str:
            return None
            
        # Try various date formats
        formats = [
            "%Y-%m-%d",            # 2023-01-15
            "%d-%m-%Y",            # 15-01-2023
            "%d/%m/%Y",            # 15/01/2023
            "%m/%d/%Y",            # 01/15/2023
            "%B %d, %Y",           # January 15, 2023
            "%d %B %Y",            # 15 January 2023
            "%b %d, %Y",           # Jan 15, 2023
            "%Y"                   # 2023 (just year)
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If no exact format works, try extracting year
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            year = year_match.group(1)
            return f"{year}-01-01"  # Default to January 1st of the year
        
        return None
    
    @staticmethod
    def validate_wallet_data(wallet_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate wallet data.
        
        Args:
            wallet_data: List of wallet information
            
        Returns:
            List[Dict]: Validated wallet data
        """
        validated_wallets = []
        
        for wallet in wallet_data:
            if not wallet.get("address"):
                continue
                
            # Validate Ethereum address
            if not DataValidator.validate_ethereum_address(wallet["address"]):
                logger.warning(f"Invalid Ethereum address: {wallet['address']}")
                continue
            
            validated_wallet = {
                "address": wallet["address"],
                "type": wallet.get("type", "Unknown")
            }
            
            # Validate balance
            if "balance" in wallet and wallet["balance"] is not None:
                if isinstance(wallet["balance"], (int, float)) and wallet["balance"] >= 0:
                    validated_wallet["balance"] = wallet["balance"]
            
            # Validate security features
            if "security_features" in wallet and isinstance(wallet["security_features"], list):
                validated_wallet["security_features"] = wallet["security_features"]
            
            validated_wallets.append(validated_wallet)
        
        return validated_wallets
    
    @staticmethod
    def validate_portfolio_data(portfolio_data: Dict[str, float]) -> Dict[str, float]:
        """
        Validate portfolio allocation data.
        
        Args:
            portfolio_data: Asset allocations as percentages
            
        Returns:
            Dict: Validated portfolio data
        """
        validated_portfolio = {}
        
        # Check total allocation
        total_allocation = sum(portfolio_data.values())
        
        # If total is very close to 1.0 (or 100%), assume decimals
        is_decimal = (0.99 <= total_allocation <= 1.01)
        
        # If total is very close to 100 (or 100%), assume percentages
        is_percentage = (99 <= total_allocation <= 101)
        
        # Validate each asset allocation
        for asset, allocation in portfolio_data.items():
            # Skip invalid allocations
            if allocation < 0:
                continue
                
            # Normalize to decimal (0-1)
            if is_percentage and allocation > 1:
                normalized = allocation / 100
            else:
                normalized = allocation
                
            # Further validation: ensure allocation is reasonable
            if 0 <= normalized <= 1:
                validated_portfolio[asset] = normalized
        
        # Renormalize if sum is not very close to 1.0
        total = sum(validated_portfolio.values())
        if total > 0 and (total < 0.99 or total > 1.01):
            for asset in validated_portfolio:
                validated_portfolio[asset] /= total
        
        return validated_portfolio
    
    @staticmethod
    def validate_risk_data(risk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate risk assessment data.
        
        Args:
            risk_data: List of risk factors with ratings
            
        Returns:
            List[Dict]: Validated risk data
        """
        validated_risks = []
        
        for risk in risk_data:
            if not risk.get("factor"):
                continue
                
            validated_risk = {
                "factor": risk["factor"]
            }
            
            # Validate rating
            if "rating" in risk:
                # Ensure rating is within 0-10 scale
                if isinstance(risk["rating"], (int, float)):
                    rating = float(risk["rating"])
                    if 0 <= rating <= 10:
                        validated_risk["rating"] = rating
                    elif 0 <= rating <= 100:
                        validated_risk["rating"] = rating / 10
                    else:
                        validated_risk["rating"] = min(max(rating, 0), 10)
                else:
                    validated_risk["rating"] = 5.0  # Default middle rating
            
            # Include mitigation if present
            if "mitigation" in risk and risk["mitigation"]:
                validated_risk["mitigation"] = risk["mitigation"]
            
            validated_risks.append(validated_risk)
        
        return validated_risks
    
    @staticmethod
    def validate_compliance_data(compliance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate compliance information.
        
        Args:
            compliance_data: Compliance information
            
        Returns:
            Dict: Validated compliance data
        """
        validated_compliance = {}
        
        # Validate KYC/AML procedures
        if "kyc_aml" in compliance_data and isinstance(compliance_data["kyc_aml"], list):
            validated_compliance["kyc_aml"] = compliance_data["kyc_aml"]
        
        # Validate regulatory status
        if "regulatory_status" in compliance_data and isinstance(compliance_data["regulatory_status"], list):
            validated_compliance["regulatory_status"] = compliance_data["regulatory_status"]
        
        # Validate tax considerations
        if "tax_considerations" in compliance_data and isinstance(compliance_data["tax_considerations"], list):
            validated_compliance["tax_considerations"] = compliance_data["tax_considerations"]
            
        return validated_compliance
    
    @staticmethod
    def validate_team_data(team_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate team information.
        
        Args:
            team_data: Team information
            
        Returns:
            Dict: Validated team data
        """
        validated_team = {}
        
        # Validate key personnel
        if "key_personnel" in team_data and isinstance(team_data["key_personnel"], list):
            validated_personnel = []
            
            for person in team_data["key_personnel"]:
                if not person.get("name"):
                    continue
                    
                validated_person = {
                    "name": person["name"]
                }
                
                if "title" in person and person["title"]:
                    validated_person["title"] = person["title"]
                
                if "background" in person and isinstance(person["background"], list):
                    validated_person["background"] = person["background"]
                
                validated_personnel.append(validated_person)
            
            validated_team["key_personnel"] = validated_personnel
        
        # Validate security team
        if "security_team" in team_data and isinstance(team_data["security_team"], list):
            validated_team["security_team"] = team_data["security_team"]
            
        return validated_team
    
    @staticmethod
    def validate_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate market data.
        
        Args:
            market_data: Market data for cryptocurrencies
            
        Returns:
            Dict: Validated market data
        """
        validated_market = {}
        
        # Validate current data
        if "current_data" in market_data and isinstance(market_data["current_data"], dict):
            validated_current = {}
            
            for symbol, data in market_data["current_data"].items():
                if not isinstance(data, dict):
                    continue
                    
                validated_symbol_data = {}
                
                # Validate price
                if "price" in data and isinstance(data["price"], (int, float)) and data["price"] >= 0:
                    validated_symbol_data["price"] = data["price"]
                
                # Validate other common fields
                for field in ["market_cap", "volume_24h", "price_change_24h"]:
                    if field in data and isinstance(data[field], (int, float)):
                        validated_symbol_data[field] = data[field]
                
                if validated_symbol_data:
                    validated_current[symbol] = validated_symbol_data
            
            if validated_current:
                validated_market["current_data"] = validated_current
        
        # Validate historical performance
        if "historical_performance" in market_data and isinstance(market_data["historical_performance"], dict):
            validated_historical = {}
            
            for symbol, performance in market_data["historical_performance"].items():
                if not isinstance(performance, dict):
                    continue
                    
                validated_performance = {}
                
                # Validate yearly returns
                if "yearly_returns" in performance and isinstance(performance["yearly_returns"], dict):
                    validated_yearly = {}
                    
                    for year, ret in performance["yearly_returns"].items():
                        if isinstance(ret, (int, float)):
                            validated_yearly[year] = ret
                    
                    if validated_yearly:
                        validated_performance["yearly_returns"] = validated_yearly
                
                # Validate metrics
                metrics = ["max_drawdown", "sharpe_ratio", "volatility"]
                for metric in metrics:
                    if metric in performance and isinstance(performance[metric], (int, float)):
                        validated_performance[metric] = performance[metric]
                
                if validated_performance:
                    validated_historical[symbol] = validated_performance
            
            if validated_historical:
                validated_market["historical_performance"] = validated_historical
        
        # Include other sections with minimal validation
        for section in ["volatility", "correlations", "forecasts"]:
            if section in market_data and market_data[section]:
                validated_market[section] = market_data[section]
        
        # Validate analysis date
        if "analysis_date" in market_data:
            try:
                # Try to parse date to ensure it's valid
                datetime.fromisoformat(market_data["analysis_date"])
                validated_market["analysis_date"] = market_data["analysis_date"]
            except (ValueError, TypeError):
                validated_market["analysis_date"] = datetime.now().isoformat()
                
        return validated_market 