"""
Slide Templates Module

This module defines templates for different slide types used in due diligence reports.
It provides a consistent approach to creating various slide layouts while maintaining
the overall design language of the presentation.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from sklearn.metrics import recall_score

from reporting.design_elements import DesignElements

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlideTemplates:
    """
    Provides templates for various slide types used in crypto fund due diligence reports.
    Each template defines the layout, content placeholders, and styling for a specific
    type of slide.
    """
    
    def __init__(self):
        """Initialize slide templates with design elements."""
        self.design = DesignElements()
    
    def get_cover_slide_spec(self, title: str, subtitle: str, date: str) -> Dict[str, Any]:
        """
        Get the specification for a cover slide.
        
        Args:
            title: Report title
            subtitle: Report subtitle
            date: Report date
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_LAYOUT_IDX,
            "background": self.design.COVER_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.TITLE_SIZE,
                    "color": (255, 255, 255),  # White
                    "bold": True,
                    "position": {
                        "top": 2.2,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": recall_score,
                    "title": "Compliance Score",
                    "position": {
                        "top": 2.8,
                        "left": 0.5,
                        "width": 3.0,
                        "height": 3.0
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": kyc_aml_coverage, # type: ignore
                    "title": "KYC/AML Coverage",
                    "position": {
                        "top": 2.8,
                        "left": 4.0,
                        "width": 3.0,
                        "height": 3.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "Compliance Assessment Overview",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 2.8,
                        "left": 7.0,
                        "width": 3.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "text_block",
                    "text": "This assessment evaluates the fund's compliance with regulatory requirements across relevant jurisdictions, with particular focus on KYC/AML procedures, registration status, and tax compliance.",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 3.4,
                        "left": 7.0,
                        "width": 3.0,
                        "height": 2.0
                    }
                }
            ]
        }
    
    def get_table_slide_spec(self, title: str, table_data: List[List[str]]) -> Dict[str, Any]:
        """
        Get the specification for a slide with a table.
        
        Args:
            title: Slide title
            table_data: Table data with headers as first row
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "table",
                    "data": table_data,
                    "header": True,  # First row is headers
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 5.0
                    }
                }
            ]
        }
    
    def get_text_slide_spec(self, title: str, content: str, text_size: int = 14) -> Dict[str, Any]:
        """
        Get the specification for a slide with text content.
        
        Args:
            title: Slide title
            content: Text content
            text_size: Font size for the content
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": content,
                    "font": self.design.BODY_FONT,
                    "size": text_size,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 5.0
                    }
                }
            ]
        }
    
    def get_conclusion_slide_spec(self, title: str, fund_name: str, risk_level: str,
                               risk_score: float, compliance_level: str, compliance_score: float,
                               strengths: List[str], concerns: List[str]) -> Dict[str, Any]:
        """
        Get the specification for a conclusion slide.
        
        Args:
            title: Slide title
            fund_name: Fund name
            risk_level: Risk level category
            risk_score: Risk score (0-100)
            compliance_level: Compliance level category
            compliance_score: Compliance score (0-100)
            strengths: List of strengths
            concerns: List of concerns
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"{fund_name} Due Diligence Summary",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE + 2,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Risk Level: {risk_level} ({risk_score:.1f}/100)\nCompliance Level: {compliance_level} ({compliance_score:.1f}/100)",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 2.1,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.6
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Strengths:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 2.8,
                        "left": 0.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": strengths,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (0, 153, 51),  # Green for strengths
                    "position": {
                        "top": 3.3,
                        "left": 0.8,
                        "width": 4.2,
                        "height": 3.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Concerns:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 2.8,
                        "left": 5.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": concerns,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (204, 0, 0),  # Red for concerns
                    "position": {
                        "top": 3.3,
                        "left": 5.8,
                        "width": 4.2,
                        "height": 3.0
                    }
                }
            ]
        }
    
    def get_executive_summary_spec(self, title: str, fund_name: str, aum: str, 
                                 strategy: str, risk_score: float, risk_level: str,
                                 risk_color: Tuple[int, int, int],
                                 key_strengths: List[str], key_concerns: List[str]) -> Dict[str, Any]:
        """
        Get the specification for an executive summary slide.
        
        Args:
            title: Slide title
            fund_name: Fund name
            aum: AUM formatted as string
            strategy: Strategy description
            risk_score: Overall risk score (0-100)
            risk_level: Risk level category
            risk_color: RGB tuple for risk level
            key_strengths: List of key strengths
            key_concerns: List of key concerns
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Fund: {fund_name}\nAUM: {aum}\nStrategy: {strategy}",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 4.5,
                        "height": 1.0
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": risk_score,
                    "title": f"Risk Level: {risk_level}",
                    "color": risk_color,
                    "position": {
                        "top": 1.5,
                        "left": 5.5,
                        "width": 3.0,
                        "height": 2.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Strengths:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 3.3,
                        "left": 0.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": key_strengths,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (0, 153, 51),  # Green for strengths
                    "position": {
                        "top": 3.8,
                        "left": 0.8,
                        "width": 4.2,
                        "height": 2.5
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Concerns:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 3.3,
                        "left": 5.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": key_concerns,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (204, 0, 0),  # Red for concerns
                    "position": {
                        "top": 3.8,
                        "left": 5.8,
                        "width": 4.2,
                        "height": 2.5
                    }
                }
            ]
        }
    
    def get_fund_overview_spec(self, title: str, fund_data: List[List[str]], 
                             strategy_description: str) -> Dict[str, Any]:
        """
        Get the specification for a fund overview slide.
        
        Args:
            title: Slide title
            fund_data: List of [label, value] pairs for fund information
            strategy_description: Detailed strategy description
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TWO_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "table",
                    "data": fund_data,
                    "header": False,  # No header row
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 4.5,
                        "height": 4.5
                    }
                },
                {
                    "type": "text_block",
                    "text": "Investment Strategy",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 5.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "text_block",
                    "text": strategy_description,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 2.0,
                        "left": 5.5,
                        "width": 4.5,
                        "height": 4.0
                    }
                }
            ]
        }
    
    def get_team_analysis_spec(self, title: str, team_profiles: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Get the specification for a team analysis slide.
        
        Args:
            title: Slide title
            team_profiles: List of team member profiles with name, title, background
            
        Returns:
            Dictionary with slide specification
        """
        elements = [
            {
                "type": "title",
                "text": title,
                "font": self.design.TITLE_FONT,
                "size": self.design.HEADING_SIZE,
                "color": self.design.PRIMARY_COLOR,
                "bold": True,
                "position": {
                    "top": 0.5,
                    "left": 0.5,
                    "width": 9.0,
                    "height": 0.8
                }
            }
        ]
        
        # Add team profiles
        # Layout depends on the number of team members
        if len(team_profiles) <= 3:
            # Single row layout
            width = 3.0
            for i, profile in enumerate(team_profiles):
                left = 0.5 + (i * (width + 0.25))
                elements.append({
                    "type": "team_profile",
                    "name": profile["name"],
                    "title": profile["title"],
                    "background": profile["background"],
                    "position": {
                        "top": 1.5,
                        "left": left,
                        "width": width,
                        "height": 3.5
                    }
                })
        else:
            # Two-row layout
            width = 3.0
            profiles_per_row = 3
            for i, profile in enumerate(team_profiles):
                row = i // profiles_per_row
                col = i % profiles_per_row
                top = 1.5 + (row * 2.5)
                left = 0.5 + (col * (width + 0.25))
                elements.append({
                    "type": "team_profile",
                    "name": profile["name"],
                    "title": profile["title"],
                    "background": profile["background"],
                    "position": {
                        "top": top,
                        "left": left,
                        "width": width,
                        "height": 2.2
                    }
                })
        
        return {
            "layout_idx": self.design.TITLE_ONLY_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": elements
        }
    
    def get_portfolio_allocation_spec(self, title: str, chart_data: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Get the specification for a portfolio allocation slide with pie chart.
        
        Args:
            title: Slide title
            chart_data: List of (asset, percentage) tuples
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.CHART_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "pie_chart",
                    "data": chart_data,
                    "position": {
                        "top": 1.5,
                        "left": 1.0,
                        "width": 5.0,
                        "height": 5.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "Portfolio Composition",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 6.5,
                        "width": 3.0,
                        "height": 0.4
                    }
                },
                {
                    "type": "table",
                    "data": [[asset, f"{percentage:.1f}%"] for asset, percentage in chart_data],
                    "header": ["Asset", "Allocation"],
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 2.0,
                        "left": 6.5,
                        "width": 3.0,
                        "height": 4.5
                    }
                }
            ]
        }
    
    def get_market_analysis_spec(self, title: str, market_data: List[List[str]]) -> Dict[str, Any]:
        """
        Get the specification for a market analysis slide with table.
        
        Args:
            title: Slide title
            market_data: Table data with headers as first row
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": "Current Market Data for Assets in Portfolio",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.4
                    }
                },
                {
                    "type": "table",
                    "data": market_data,
                    "header": True,  # First row is headers
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 2.0,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 4.0
                    }
                }
            ]
        }
    
    def get_wallet_overview_spec(self, title: str, total_balance: str, wallet_count: int,
                              avg_risk_score: float, wallet_chart_data: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Get the specification for a wallet overview slide.
        
        Args:
            title: Slide title
            total_balance: Total wallet balance as formatted string
            wallet_count: Number of wallets
            avg_risk_score: Average risk score
            wallet_chart_data: List of (wallet_type, balance) tuples
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.CHART_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Total Balance: {total_balance}\nNumber of Wallets: {wallet_count}",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE + 2,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "pie_chart",
                    "data": wallet_chart_data,
                    "title": "Wallet Distribution by Type",
                    "position": {
                        "top": 2.3,
                        "left": 0.5,
                        "width": 5.0,
                        "height": 4.0
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": 100 - avg_risk_score,  # Convert risk to security score
                    "title": "Wallet Security Score",
                    "position": {
                        "top": 2.3,
                        "left": 6.0,
                        "width": 3.0,
                        "height": 3.0
                    }
                }
            ]
        }
    
    def get_wallet_security_spec(self, title: str, wallet_data: List[List[str]], 
                              security_score: float, wallet_diversification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the specification for a wallet security analysis slide.
        
        Args:
            title: Slide title
            wallet_data: Table data with headers as first row
            security_score: Security score (0-100)
            wallet_diversification: Wallet diversification data
            
        Returns:
            Dictionary with slide specification
        """
        # Extract wallet diversification metrics
        diversification_score = wallet_diversification.get("diversification_score", 50)
        concentration_risk = wallet_diversification.get("concentration_risk", "Medium")
        largest_wallet_pct = wallet_diversification.get("largest_wallet_pct", 0)
        
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "table",
                    "data": wallet_data,
                    "header": True,  # First row is headers
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 2.5
                    }
                },
                {
                    "type": "text_block",
                    "text": "Security Assessment",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 4.1,
                        "left": 0.5,
                        "width": 4.0,
                        "height": 0.4
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Overall Security Score: {security_score:.1f}/100\n" + 
                           f"Wallet Diversification: {diversification_score:.1f}/100\n" +
                           f"Concentration Risk: {concentration_risk}\n" +
                           f"Largest Wallet: {largest_wallet_pct:.1f}%",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 4.5,
                        "left": 0.5,
                        "width": 4.0,
                        "height": 1.5
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": security_score,
                    "title": "Security Score",
                    "position": {
                        "top": 4.1,
                        "left": 6.0,
                        "width": 3.0,
                        "height": 2.5
                    }
                }
            ]
        }
    
    def get_risk_overview_spec(self, title: str, overall_risk_score: float, risk_level: str,
                            risk_color: Tuple[int, int, int], radar_labels: List[str],
                            radar_values: List[float]) -> Dict[str, Any]:
        """
        Get the specification for a risk overview slide with radar chart.
        
        Args:
            title: Slide title
            overall_risk_score: Overall risk score (0-100)
            risk_level: Risk level category
            risk_color: RGB tuple for risk level
            radar_labels: Labels for radar chart axes
            radar_values: Values for radar chart (0-10 scale)
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.CHART_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Overall Risk Assessment: {risk_level} ({overall_risk_score:.1f}/100)",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE + 2,
                    "color": risk_color,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "radar_chart",
                    "labels": radar_labels,
                    "values": radar_values,
                    "position": {
                        "top": 2.0,
                        "left": 0.5,
                        "width": 5.0,
                        "height": 4.0
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": overall_risk_score,
                    "title": f"Risk Score: {risk_level}",
                    "color": risk_color,
                    "invert": True,  # Higher is worse for risk
                    "position": {
                        "top": 2.0,
                        "left": 6.0,
                        "width": 3.0,
                        "height": 3.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "This risk assessment evaluates multiple risk dimensions to provide a comprehensive view of the fund's risk profile.",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 5.2,
                        "left": 6.0,
                        "width": 3.0,
                        "height": 1.5
                    }
                }
            ]
        }
    
    def get_risk_factors_spec(self, title: str, risk_factors: List[str]) -> Dict[str, Any]:
        """
        Get the specification for a risk factors slide.
        
        Args:
            title: Slide title
            risk_factors: List of risk factor descriptions
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": "The following risk factors have been identified through our comprehensive due diligence process:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.6
                    }
                },
                {
                    "type": "bullet_list",
                    "items": risk_factors,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 2.2,
                        "left": 0.8,
                        "width": 8.7,
                        "height": 4.0
                    }
                }
            ]
        }
    
    def get_compliance_overview_spec(self, title: str, overall_score: float, compliance_level: str,
                                 jurisdictions: str, kyc_aml_coverage: float) -> Dict[str, Any]:
        """
        Get the specification for a compliance overview slide.
        
        Args:
            title: Slide title
            overall_score: Overall compliance score (0-100)
            compliance_level: Compliance level category
            jurisdictions: Comma-separated list of jurisdictions
            kyc_aml_coverage: KYC/AML coverage score (0-100)
            
        Returns:
            Dictionary with slide specification
        """
        # Determine color based on compliance score
        compliance_color = self.design.get_risk_scale_color(overall_score, invert=False)
        
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Overall Compliance Level: {compliance_level} ({overall_score:.1f}/100)",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE + 2,
                    "color": compliance_color,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Relevant Jurisdictions: {jurisdictions}",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 2.1,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": overall_score,
                    "title": "Compliance Score",
                    "color": compliance_color,
                    "invert": False,  # Higher is better for compliance
                    "position": {
                        "top": 2.7,
                        "left": 0.5,
                        "width": 4.0,
                        "height": 3.0
                    }
                },
                {
                    "type": "gauge_chart",
                    "value": kyc_aml_coverage,
                    "title": "KYC/AML Coverage",
                    "color": self.design.get_risk_scale_color(kyc_aml_coverage, invert=False),
                    "invert": False,  # Higher is better for compliance
                    "position": {
                        "top": 2.7,
                        "left": 5.5,
                        "width": 4.0,
                        "height": 3.0
                    }
                }
            ]
        }

    def get_table_slide_spec(self, title: str, table_data: List[List[str]]) -> Dict[str, Any]:
        """
        Get the specification for a slide with a table.
        
        Args:
            title: Slide title
            table_data: Table data with headers as first row
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "table",
                    "data": table_data,
                    "header": True,  # First row is headers
                    "stripes": True,  # Use alternating row colors
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 5.0
                    }
                }
            ]
        }
        
    def get_text_slide_spec(self, title: str, content: str, text_size: int = 14) -> Dict[str, Any]:
        """
        Get the specification for a slide with text content.
        
        Args:
            title: Slide title
            content: Text content
            text_size: Font size for the content
            
        Returns:
            Dictionary with slide specification
        """
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": content,
                    "font": self.design.BODY_FONT,
                    "size": text_size,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 5.0
                    }
                }
            ]
        }
    
    def get_conclusion_slide_spec(self, title: str, fund_name: str, risk_level: str, risk_score: float,
                              compliance_level: str, compliance_score: float, 
                              strengths: List[str], concerns: List[str]) -> Dict[str, Any]:
        """
        Get the specification for a conclusion slide.
        
        Args:
            title: Slide title
            fund_name: Fund name
            risk_level: Risk level category
            risk_score: Risk score (0-100)
            compliance_level: Compliance level category
            compliance_score: Compliance score (0-100)
            strengths: List of key strengths
            concerns: List of key concerns
            
        Returns:
            Dictionary with slide specification
        """
        # Determine colors based on scores
        risk_color = self.design.get_risk_scale_color(risk_score, invert=True)
        compliance_color = self.design.get_risk_scale_color(compliance_score, invert=False)
        
        return {
            "layout_idx": self.design.TITLE_CONTENT_LAYOUT_IDX,
            "background": self.design.SLIDE_BACKGROUND,
            "elements": [
                {
                    "type": "title",
                    "text": title,
                    "font": self.design.TITLE_FONT,
                    "size": self.design.HEADING_SIZE,
                    "color": self.design.PRIMARY_COLOR,
                    "bold": True,
                    "position": {
                        "top": 0.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Due Diligence Summary for {fund_name}",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE + 2,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 1.5,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.5
                    }
                },
                {
                    "type": "text_block",
                    "text": f"Risk Level: {risk_level} ({risk_score:.1f}/100)\nCompliance Level: {compliance_level} ({compliance_score:.1f}/100)",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "position": {
                        "top": 2.1,
                        "left": 0.5,
                        "width": 9.0,
                        "height": 0.8
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Strengths:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 3.0,
                        "left": 0.5,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": strengths,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (0, 153, 51),  # Green for strengths
                    "position": {
                        "top": 3.5,
                        "left": 0.8,
                        "width": 4.2,
                        "height": 3.0
                    }
                },
                {
                    "type": "text_block",
                    "text": "Key Concerns:",
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": self.design.DARK_COLOR,
                    "bold": True,
                    "position": {
                        "top": 3.0,
                        "left": 5.0,
                        "width": 4.5,
                        "height": 0.4
                    }
                },
                {
                    "type": "bullet_list",
                    "items": concerns,
                    "font": self.design.BODY_FONT,
                    "size": self.design.BODY_SIZE,
                    "color": (204, 0, 0),  # Red for concerns
                    "position": {
                        "top": 3.5,
                        "left": 5.3,
                        "width": 4.2,
                        "height": 3.0
                    }
                }
            ]
        }
       