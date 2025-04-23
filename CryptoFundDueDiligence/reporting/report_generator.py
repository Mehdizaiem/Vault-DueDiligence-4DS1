"""
Report Generator Module

This module orchestrates the generation of comprehensive PowerPoint reports for
crypto fund due diligence analysis. It combines information from document analysis,
market data, on-chain analytics, and other sources into a polished presentation.
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import json

from pptx import Presentation

# Import internal modules
from reporting.pptx_builder import PresentationBuilder
from reporting.chart_factory import ChartFactory
from reporting.slide_templates import SlideTemplates
from reporting.design_elements import DesignElements
from utils.report_utils import format_currency, calculate_change_text, summarize_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Orchestrates the generation of comprehensive crypto fund due diligence reports
    in PowerPoint format with professional design and data visualizations.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the report generator.
        
        Args:
            template_path: Path to custom PowerPoint template (optional)
        """
        self.template_path = template_path or os.path.join(
            os.path.dirname(__file__), 'templates', 'base_template.pptx'
        )
        
        # Initialize components
        self.design = DesignElements()
        self.slide_templates = SlideTemplates()
        self.chart_factory = ChartFactory()
        
        # Verify template exists
        if not os.path.exists(self.template_path):
            logger.warning(f"Template not found at {self.template_path}, using default template")
            self.template_path = None
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive PowerPoint report based on analysis results.
        
        Args:
            analysis_results: The complete due diligence analysis results
            output_path: Path where to save the PowerPoint file (optional)
            
        Returns:
            str: Path to the generated PowerPoint file
        """
        logger.info("Starting PowerPoint report generation")
        start_time = time.time()
        
        try:
            # Create presentation builder
            builder = PresentationBuilder(self.template_path)
            
            # Extract key data from analysis results
            fund_info = analysis_results.get("fund_info", {})
            portfolio_data = analysis_results.get("portfolio_data", {})
            wallet_analysis = analysis_results.get("wallet_analysis", {})
            market_analysis = analysis_results.get("market_analysis", {})
            risk_assessment = analysis_results.get("risk_assessment", {})
            compliance_analysis = analysis_results.get("compliance_analysis", {})
            team_data = analysis_results.get("team_data", {})
            
            # Generate report sections
            self._create_cover_slide(builder, fund_info)
            self._create_executive_summary(builder, analysis_results)
            self._create_fund_overview(builder, fund_info)
            self._create_team_analysis(builder, team_data)
            self._create_portfolio_analysis(builder, portfolio_data, market_analysis)
            self._create_wallet_security_analysis(builder, wallet_analysis)
            self._create_risk_assessment(builder, risk_assessment)
            self._create_compliance_analysis(builder, compliance_analysis)
            self._create_conclusion_slide(builder, risk_assessment, analysis_results)
            
            # Save the presentation
            if not output_path:
                # Create a default filename with timestamp
                fund_name = fund_info.get("fund_name", "CryptoFund")
                safe_fund_name = "".join(c if c.isalnum() else "_" for c in fund_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(os.getcwd(), "reports")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{safe_fund_name}_DueDiligence_{timestamp}.pptx")
            
            # Save the presentation
            builder.save(output_path)
            
            logger.info(f"Report generated successfully in {time.time() - start_time:.2f} seconds")
            logger.info(f"Report saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
    
    def _create_cover_slide(self, builder: PresentationBuilder, fund_info: Dict[str, Any]) -> None:
        """
        Create the cover slide for the report.
        
        Args:
            builder: The presentation builder
            fund_info: Fund information dictionary
        """
        fund_name = fund_info.get("fund_name", "Crypto Fund")
        builder.add_cover_slide(
            title=f"{fund_name} Due Diligence Report",
            subtitle=f"Comprehensive Analysis & Risk Assessment",
            date=datetime.now().strftime("%B %d, %Y"),
            background_color=self.design.COVER_BACKGROUND,
            logo_path=None  # Add a logo path if desired
        )
    
    def _create_executive_summary(self, builder: PresentationBuilder, 
                                analysis_results: Dict[str, Any]) -> None:
        """
        Create the executive summary slide.
        
        Args:
            builder: The presentation builder
            analysis_results: The complete analysis results
        """
        fund_info = analysis_results.get("fund_info", {})
        risk_assessment = analysis_results.get("risk_assessment", {})
        
        fund_name = fund_info.get("fund_name", "Crypto Fund")
        aum = fund_info.get("aum", 0)
        fund_strategy = fund_info.get("strategy", "Not specified")
        
        # Format risk score
        risk_score = risk_assessment.get("overall_risk_score", 50)
        risk_level = risk_assessment.get("risk_level", "Medium")
        risk_color = self._get_risk_color(risk_level)
        
        # Get top strengths and concerns
        strengths = self._extract_strengths(analysis_results)
        concerns = self._extract_concerns(analysis_results)
        
        builder.add_executive_summary_slide(
            title="Executive Summary",
            fund_name=fund_name,
            aum=format_currency(aum, in_millions=True),
            strategy=summarize_text(fund_strategy, max_length=100),
            risk_score=risk_score,
            risk_level=risk_level,
            risk_color=risk_color,
            key_strengths=strengths[:3],  # Top 3 strengths
            key_concerns=concerns[:3]     # Top 3 concerns
        )
    
    def _create_fund_overview(self, builder: PresentationBuilder, 
                             fund_info: Dict[str, Any]) -> None:
        """
        Create the fund overview slide.
        
        Args:
            builder: The presentation builder
            fund_info: Fund information dictionary
        """
        # Extract fund information
        fund_name = fund_info.get("fund_name", "Crypto Fund")
        aum = fund_info.get("aum", 0)
        strategy = fund_info.get("strategy", "Not specified")
        mgmt_fee = fund_info.get("management_fee", 0) * 100  # Convert to percentage
        perf_fee = fund_info.get("performance_fee", 0) * 100  # Convert to percentage
        launch_date = fund_info.get("launch_date", "Unknown")
        min_investment = fund_info.get("min_investment", "Unknown")
        lock_up = fund_info.get("lock_up", "Not specified")
        
        # Create fund info table data
        fund_data = [
            ["Fund Name", fund_name],
            ["AUM", format_currency(aum, in_millions=True)],
            ["Strategy", summarize_text(strategy, max_length=100)],
            ["Management Fee", f"{mgmt_fee:.2f}%"],
            ["Performance Fee", f"{perf_fee:.2f}%"],
            ["Launch Date", launch_date],
            ["Minimum Investment", min_investment],
            ["Lock-up Period", lock_up]
        ]
        
        builder.add_fund_overview_slide(
            title="Fund Overview",
            fund_data=fund_data,
            strategy_description=strategy
        )
    
    def _create_team_analysis(self, builder: PresentationBuilder,
                             team_data: Dict[str, Any]) -> None:
        """
        Create slide(s) for team analysis.
        
        Args:
            builder: The presentation builder
            team_data: Team information dictionary
        """
        key_personnel = team_data.get("key_personnel", [])
        security_team = team_data.get("security_team", [])
        
        # If we have team members, create a team analysis slide
        if key_personnel:
            # Format team data for the slide
            team_profiles = []
            for person in key_personnel[:6]:  # Limit to 6 people per slide
                name = person.get("name", "")
                title = person.get("title", "")
                background = person.get("background", [])
                
                # Format background points
                background_text = "\n".join([f"• {point}" for point in background[:3]])
                
                team_profiles.append({
                    "name": name,
                    "title": title,
                    "background": background_text
                })
            
            builder.add_team_analysis_slide(
                title="Team Analysis",
                team_profiles=team_profiles
            )
        
        # If we have security team information, add it to a separate slide or section
        if security_team:
            security_points = [f"• {point}" for point in security_team]
            builder.add_text_slide(
                title="Security Team",
                content="\n".join(security_points)
            )
    
    def _create_portfolio_analysis(self, builder: PresentationBuilder,
                                 portfolio_data: Dict[str, float],
                                 market_analysis: Dict[str, Any]) -> None:
        """
        Create slides for portfolio analysis including allocation charts.
        
        Args:
            builder: The presentation builder
            portfolio_data: Portfolio allocation dictionary
            market_analysis: Market data and analysis
        """
        # Only create this slide if we have portfolio data
        if not portfolio_data:
            return
        
        # Prepare data for pie chart
        chart_data = []
        for asset, allocation in portfolio_data.items():
            # Convert allocation to percentage
            percentage = allocation * 100
            chart_data.append((asset, percentage))
        
        # Sort by allocation (descending)
        chart_data.sort(key=lambda x: x[1], reverse=True)
        
        # If too many items, group smaller ones into "Other"
        if len(chart_data) > 7:
            top_assets = chart_data[:6]
            other_allocation = sum(alloc for _, alloc in chart_data[6:])
            chart_data = top_assets + [("Other", other_allocation)]
        
        # Create portfolio allocation slide with pie chart
        builder.add_portfolio_allocation_slide(
            title="Portfolio Allocation",
            chart_data=chart_data
        )
        
        # If we have market data, create market analysis slide
        if market_analysis:
            current_data = market_analysis.get("current_data", {})
            performance = market_analysis.get("historical_performance", {})
            volatility = market_analysis.get("volatility", {})
            
            # Create market data table
            market_table_data = []
            
            # Add headers
            market_table_data.append(["Asset", "Price", "24h Change", "7d Change", "Volatility"])
            
            # Add data for each asset
            for symbol, data in current_data.items():
                price = data.get("price", 0)
                change_24h = data.get("price_change_24h", 0)
                
                # Get 7d change from performance data if available
                change_7d = 0
                if symbol in performance and "1w" in performance[symbol]:
                    change_7d = performance[symbol]["1w"].get("change_pct", 0)
                
                # Get volatility if available
                asset_volatility = 0
                if symbol in volatility:
                    asset_volatility = volatility[symbol].get("volatility_7d", 0)
                
                market_table_data.append([
                    symbol,
                    format_currency(price),
                    calculate_change_text(change_24h),
                    calculate_change_text(change_7d),
                    f"{asset_volatility:.1f}%"
                ])
            
            # Only add the slide if we have data
            if len(market_table_data) > 1:
                builder.add_market_analysis_slide(
                    title="Market Analysis",
                    market_data=market_table_data
                )
    
    def _create_wallet_security_analysis(self, builder: PresentationBuilder,
                                       wallet_analysis: Dict[str, Any]) -> None:
        """
        Create slides for wallet security analysis.
        
        Args:
            builder: The presentation builder
            wallet_analysis: Wallet analysis data
        """
        # Skip if no wallet data
        if not wallet_analysis or "wallets" not in wallet_analysis:
            return
        
        wallets = wallet_analysis.get("wallets", {})
        aggregate_stats = wallet_analysis.get("aggregate_stats", {})
        risk_assessment = wallet_analysis.get("risk_assessment", {})
        
        if not wallets:
            return
        
        # Create wallet overview slide
        total_balance = aggregate_stats.get("total_balance_eth", 0)
        wallet_count = len(wallets)
        avg_risk_score = aggregate_stats.get("average_risk_score", 50)
        
        # Prepare wallet distribution data for chart
        wallet_types = {}
        for address, wallet in wallets.items():
            wallet_type = wallet.get("type", "Unknown")
            if wallet_type not in wallet_types:
                wallet_types[wallet_type] = 0
            wallet_types[wallet_type] += wallet.get("balance", 0)
        
        # Convert to chart data format
        wallet_chart_data = [(wtype, balance) for wtype, balance in wallet_types.items()]
        
        # Create the wallet overview slide
        builder.add_wallet_overview_slide(
            title="Wallet Infrastructure",
            total_balance=f"{total_balance:.2f} ETH",
            wallet_count=wallet_count,
            avg_risk_score=avg_risk_score,
            wallet_chart_data=wallet_chart_data
        )
        
        # Create wallet security analysis slide
        wallet_table_data = [["Address", "Type", "Balance", "Risk Level", "Security Features"]]
        
        # Add data for each wallet (limit to top 5 by balance)
        wallet_items = list(wallets.items())
        wallet_items.sort(key=lambda x: x[1].get("balance", 0), reverse=True)
        
        for address, wallet in wallet_items[:5]:
            # Format address (truncate for display)
            display_address = f"{address[:6]}...{address[-4:]}"
            wallet_type = wallet.get("type", "Unknown")
            balance = wallet.get("balance", 0)
            risk_level = wallet.get("risk_level", "Medium")
            
            # Format security features
            security_features = wallet.get("security_features", [])
            security_text = ", ".join(security_features[:2])
            if len(security_features) > 2:
                security_text += "..."
            
            wallet_table_data.append([
                display_address,
                wallet_type,
                f"{balance:.2f} ETH",
                risk_level,
                security_text
            ])
        
        # Only add the slide if we have data
        if len(wallet_table_data) > 1:
            builder.add_wallet_security_slide(
                title="Wallet Security Analysis",
                wallet_data=wallet_table_data,
                security_score=100 - avg_risk_score,  # Convert risk score to security score
                wallet_diversification=risk_assessment.get("wallet_diversification", {})
            )
    
    def _create_risk_assessment(self, builder: PresentationBuilder,
                              risk_assessment: Dict[str, Any]) -> None:
        """
        Create slides for risk assessment.
        
        Args:
            builder: The presentation builder
            risk_assessment: Risk assessment data
        """
        # Skip if no risk assessment data
        if not risk_assessment:
            return
        
        # Extract risk components and overall risk
        risk_components = risk_assessment.get("risk_components", {})
        overall_risk_score = risk_assessment.get("overall_risk_score", 50)
        risk_level = risk_assessment.get("risk_level", "Medium")
        risk_factors = risk_assessment.get("risk_factors", [])
        mitigations = risk_assessment.get("suggested_mitigations", [])
        
        # Create data for the radar chart
        radar_labels = []
        radar_values = []
        
        for risk_type, risk_data in risk_components.items():
            # Format the label (convert snake_case to Title Case)
            label = " ".join(word.capitalize() for word in risk_type.split("_"))
            
            # Get the risk score (0-10)
            score = risk_data.get("score", 5)
            
            radar_labels.append(label)
            radar_values.append(score)
        
        # Create risk overview slide with radar chart
        builder.add_risk_overview_slide(
            title="Risk Assessment",
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            risk_color=self._get_risk_color(risk_level),
            radar_labels=radar_labels,
            radar_values=radar_values
        )
        
        # Create detailed risk factors slide
        if risk_factors:
            risk_factors_formatted = [f"• {factor}" for factor in risk_factors[:8]]
            
            builder.add_risk_factors_slide(
                title="Key Risk Factors",
                risk_factors=risk_factors_formatted
            )
        
        # Create risk mitigation recommendations slide
        if mitigations:
            mitigations_formatted = [f"• {mitigation}" for mitigation in mitigations[:8]]
            
            builder.add_text_slide(
                title="Risk Mitigation Recommendations",
                content="\n".join(mitigations_formatted),
                text_size=14
            )
    
    def _create_compliance_analysis(self, builder: PresentationBuilder,
                                  compliance_analysis: Dict[str, Any]) -> None:
        """
        Create slides for compliance analysis.
        
        Args:
            builder: The presentation builder
            compliance_analysis: Compliance analysis data
        """
        # Skip if no compliance data
        if not compliance_analysis:
            return
        
        # Extract compliance information
        overall_score = compliance_analysis.get("overall_compliance_score", 50)
        compliance_level = compliance_analysis.get("compliance_level", "Medium")
        jurisdictions = compliance_analysis.get("jurisdictions", [])
        regulatory_status = compliance_analysis.get("regulatory_status", {})
        kyc_aml = compliance_analysis.get("kyc_aml_assessment", {})
        compliance_gaps = compliance_analysis.get("compliance_gaps", [])
        
        # Create compliance overview slide
        builder.add_compliance_overview_slide(
            title="Compliance Overview",
            overall_score=overall_score,
            compliance_level=compliance_level,
            jurisdictions=", ".join(jurisdictions),
            kyc_aml_coverage=kyc_aml.get("coverage_score", 0)
        )
        
        # Create regulatory status slide
        if regulatory_status:
            regulatory_table = [["Jurisdiction", "Registration Status", "Compliance Score"]]
            
            for jurisdiction, status in regulatory_status.items():
                registration = status.get("registration_status", "Unknown")
                score = status.get("compliance_score", 0)
                
                regulatory_table.append([
                    jurisdiction,
                    registration,
                    f"{score:.1f}%"
                ])
            
            builder.add_table_slide(
                title="Regulatory Status by Jurisdiction",
                table_data=regulatory_table
            )
        
        # Create compliance gaps and recommendations slide
        if compliance_gaps:
            gaps_formatted = [f"• {gap}" for gap in compliance_gaps[:8]]
            
            builder.add_text_slide(
                title="Compliance Gaps and Recommendations",
                content="\n".join(gaps_formatted),
                text_size=14
            )
    
    def _create_conclusion_slide(self, builder: PresentationBuilder,
                               risk_assessment: Dict[str, Any],
                               analysis_results: Dict[str, Any]) -> None:
        """
        Create the conclusion slide.
        
        Args:
            builder: The presentation builder
            risk_assessment: Risk assessment data
            analysis_results: The complete analysis results
        """
        # Extract fund info
        fund_info = analysis_results.get("fund_info", {})
        fund_name = fund_info.get("fund_name", "Crypto Fund")
        
        # Extract risk and compliance info
        risk_level = risk_assessment.get("risk_level", "Medium")
        overall_risk_score = risk_assessment.get("overall_risk_score", 50)
        
        compliance_analysis = analysis_results.get("compliance_analysis", {})
        compliance_level = compliance_analysis.get("compliance_level", "Medium")
        compliance_score = compliance_analysis.get("overall_compliance_score", 50)
        
        # Get strengths and concerns
        strengths = self._extract_strengths(analysis_results)
        concerns = self._extract_concerns(analysis_results)
        
        builder.add_conclusion_slide(
            title="Conclusion",
            fund_name=fund_name,
            risk_level=risk_level,
            risk_score=overall_risk_score,
            compliance_level=compliance_level,
            compliance_score=compliance_score,
            strengths=strengths[:5],
            concerns=concerns[:5]
        )
    
    def _extract_strengths(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Extract key strengths from the analysis results.
        
        Args:
            analysis_results: The complete analysis results
            
        Returns:
            List of strength points
        """
        strengths = []
        
        # Check portfolio diversification
        portfolio_data = analysis_results.get("portfolio_data", {})
        if portfolio_data:
            max_allocation = max(portfolio_data.values()) if portfolio_data else 1.0
            if max_allocation < 0.3:
                strengths.append("Well-diversified portfolio with no single asset exceeding 30%")
        
        # Check wallet security
        wallet_analysis = analysis_results.get("wallet_analysis", {})
        wallets = wallet_analysis.get("wallets", {})
        
        if wallets:
            # Check for multisig security
            has_multisig = False
            for _, wallet in wallets.items():
                security_features = wallet.get("security_features", [])
                if any("multi-sig" in feature.lower() for feature in security_features):
                    has_multisig = True
                    break
            
            if has_multisig:
                strengths.append("Enhanced security through multi-signature wallet infrastructure")
            
            # Check for wallet diversification
            wallet_count = len(wallets)
            if wallet_count >= 3:
                strengths.append(f"Risk distributed across {wallet_count} wallets")
        
        # Check compliance
        compliance_analysis = analysis_results.get("compliance_analysis", {})
        compliance_score = compliance_analysis.get("overall_compliance_score", 0)
        if compliance_score > 75:
            strengths.append("Strong regulatory compliance framework")
        
        kyc_aml = compliance_analysis.get("kyc_aml_assessment", {})
        kyc_score = kyc_aml.get("coverage_score", 0)
        if kyc_score > 70:
            strengths.append("Comprehensive KYC/AML procedures in place")
        
        # Check team experience
        team_data = analysis_results.get("team_data", {})
        key_personnel = team_data.get("key_personnel", [])
        
        if key_personnel and len(key_personnel) >= 3:
            strengths.append("Experienced management team with relevant expertise")
        
        # Add generic strengths if needed
        if len(strengths) < 3:
            generic_strengths = [
                "Transparent documentation and disclosures",
                "Clear investment strategy with defined objectives",
                "Established operational procedures",
                "Regular performance reporting"
            ]
            strengths.extend(generic_strengths)
        
        return strengths[:5]  # Return top 5 strengths
    
    def _extract_concerns(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Extract key concerns from the analysis results.
        
        Args:
            analysis_results: The complete analysis results
            
        Returns:
            List of concern points
        """
        concerns = []
        
        # Check portfolio concentration
        portfolio_data = analysis_results.get("portfolio_data", {})
        if portfolio_data:
            max_allocation = max(portfolio_data.values()) if portfolio_data else 0
            max_asset = max(portfolio_data.items(), key=lambda x: x[1])[0] if portfolio_data else "Unknown"
            
            if max_allocation > 0.5:
                concerns.append(f"High concentration risk with {max_asset} at {max_allocation*100:.1f}%")
        
        # Check risk assessment
        risk_assessment = analysis_results.get("risk_assessment", {})
        risk_factors = risk_assessment.get("risk_factors", [])
        
        if risk_factors:
            # Add top risk factors as concerns
            for factor in risk_factors[:2]:
                concerns.append(factor)
        
        # Check compliance gaps
        compliance_analysis = analysis_results.get("compliance_analysis", {})
        compliance_gaps = compliance_analysis.get("compliance_gaps", [])
        
        if compliance_gaps:
            # Add top compliance gaps as concerns
            for gap in compliance_gaps[:2]:
                concerns.append(gap)
        
        # Check wallet security
        wallet_analysis = analysis_results.get("wallet_analysis", {})
        aggregate_stats = wallet_analysis.get("aggregate_stats", {})
        
        avg_risk_score = aggregate_stats.get("average_risk_score", 0)
        if avg_risk_score > 70:
            concerns.append("High wallet security risk score")
        
        # Add generic concerns if needed
        if len(concerns) < 3:
            generic_concerns = [
                "Market volatility exposure",
                "Regulatory uncertainty in key jurisdictions",
                "Limited operational history",
                "Smart contract vulnerability exposure"
            ]
            concerns.extend(generic_concerns)
        
        return concerns[:5]  # Return top 5 concerns
    
    def _get_risk_color(self, risk_level: str) -> str:
        """
        Get appropriate color for risk level visualization.
        
        Args:
            risk_level: Risk level string
            
        Returns:
            Hex color code
        """
        risk_colors = {
            "Very Low": self.design.RISK_VERY_LOW,
            "Low": self.design.RISK_LOW,
            "Medium-Low": self.design.RISK_MEDIUM_LOW,
            "Medium": self.design.RISK_MEDIUM,
            "Medium-High": self.design.RISK_MEDIUM_HIGH,
            "High": self.design.RISK_HIGH,
            "Very High": self.design.RISK_VERY_HIGH
        }
        
        return risk_colors.get(risk_level, self.design.RISK_MEDIUM)