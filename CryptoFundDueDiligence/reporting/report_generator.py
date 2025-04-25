# Path: reporting/report_generator.py
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import traceback
import numpy as np

from reporting.pptx_builder import PresentationBuilder
from reporting.chart_factory import ChartFactory
from reporting.design_elements import DesignElements
# --- Ensure rgb_to_hex is imported ---
from utils.report_utils import format_currency, calculate_change_text, summarize_text, rgb_to_hex
# --------------------------------------
from llm.groq_client import GroqClient
from llm.prompt_builder import ReportPromptBuilder
from llm.narrative_generator import NarrativeGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, template_path: Optional[str] = None):
        self.design = DesignElements()
        self.chart_factory = ChartFactory()
        
        # If template_path is not provided, try to find the default template
        if not template_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_path = os.path.join(project_root, "reporting", "templates", "base_template.pptx")
            if os.path.exists(template_path):
                logger.info(f"Using default template from: {template_path}")
            else:
                logger.warning(f"Default template not found at {template_path}")
                template_path = None
        
        self.builder = PresentationBuilder(template_path)
        
        # Initialize narrative generator (existing code)
        try:
            groq_client = GroqClient()
            if not groq_client.api_key: raise ValueError("Groq API Key missing.")
            prompt_builder = ReportPromptBuilder()
            self.narrative_generator = NarrativeGenerator(groq_client, prompt_builder)
            logger.info("NarrativeGenerator initialized successfully.")
        except ValueError as ve: logger.error(f"Failed to initialize GroqClient: {ve}")
        except Exception as e: logger.error(f"Failed to initialize NarrativeGenerator: {e}", exc_info=True)

    def generate_report(self, analysis_results: Dict[str, Any],
                       output_path: Optional[str] = None) -> str:
        logger.info("Starting PowerPoint report generation process...")
        start_time = time.time(); builder = self.builder
        try:
            # --- 1. Prepare Data ---
            fund_info=analysis_results.get("fund_info",{}); portfolio_data=analysis_results.get("portfolio_data",{}); wallet_analysis=analysis_results.get("wallet_analysis",{}); market_analysis=analysis_results.get("market_analysis",{})
            risk_assessment=analysis_results.get("risk_assessment",{}); compliance_analysis=analysis_results.get("compliance_analysis",{}); team_data=analysis_results.get("team_data",{}); llm_prompt_data=analysis_results.get("llm_prompt_data",{})
            fund_info_for_prompt = llm_prompt_data.get("fund_info", fund_info)
            risk_summary_for_prompt = llm_prompt_data.get("risk_summary", {"overall_risk_score": risk_assessment.get("overall_risk_score", 50),"risk_level": risk_assessment.get("risk_level", "Medium"),"top_risk_factors": [f.get("factor", "Unknown") for f in risk_assessment.get("risk_factors", [])[:3] if isinstance(f, dict) and f.get("factor")]})
            compliance_summary_for_prompt = llm_prompt_data.get("compliance_summary", {"overall_compliance_score": compliance_analysis.get("overall_compliance_score", 50),"compliance_level": compliance_analysis.get("compliance_level", "Medium"),"top_compliance_gaps": compliance_analysis.get("compliance_gaps", [])[:3]})
            key_points_for_prompt = llm_prompt_data.get("key_points", {"strengths": [], "concerns": []})

            # --- 2. Generate Narratives ---
            executive_summary_text = "[Executive Summary generation unavailable.]"; risk_narrative_text = "[Risk Narrative generation unavailable.]"; compliance_narrative_text = "[Compliance Narrative generation unavailable.]"; conclusion_text = "[Conclusion generation unavailable.]"
            overall_assessment_points = {"strengths": ["N/A - Check Logs"], "concerns": ["N/A - Check Logs"]}
            if self.narrative_generator:
                logger.info("Attempting to generate narratives using LLM...")
                try:
                    overall_context_data = {"fund_info_summary": {"name": fund_info.get("fund_name"), "aum": fund_info.get("aum")},"portfolio_highlights": {"num_assets": len(portfolio_data), "top_holding_pct": max(portfolio_data.values(), default=0)*100 if portfolio_data else 0},"wallet_security_score": 100 - wallet_analysis.get("aggregate_stats", {}).get("average_risk_score", 50),"risk_summary": risk_summary_for_prompt,"compliance_summary": compliance_summary_for_prompt}
                    overall_assessment_points = self.narrative_generator.generate_overall_assessment_points(overall_context_data); logger.info(f"Generated overall assessment points: {overall_assessment_points}")
                    key_points_for_prompt = overall_assessment_points
                    exec_summary_data = {"fund_info": fund_info_for_prompt, "risk_summary": risk_summary_for_prompt,"compliance_summary": compliance_summary_for_prompt, "key_points": key_points_for_prompt}
                    executive_summary_text = self.narrative_generator.generate_executive_summary(exec_summary_data)
                    risk_narrative_text = self.narrative_generator.generate_risk_narrative({"risk_assessment": risk_assessment}) if "error" not in risk_assessment else "[Risk narrative unavailable due to analysis error.]"
                    compliance_narrative_text = self.narrative_generator.generate_compliance_narrative({"compliance_analysis": compliance_analysis}) if "error" not in compliance_analysis else "[Compliance narrative unavailable due to analysis error.]"
                    conclusion_data = {"fund_info": fund_info_for_prompt, "risk_summary": risk_summary_for_prompt, "compliance_summary": compliance_summary_for_prompt, "key_points": key_points_for_prompt}
                    conclusion_text = self.narrative_generator.generate_conclusion(conclusion_data)
                    logger.info("LLM narrative generation attempt completed.")
                except Exception as llm_e:
                    logger.error(f"LLM narrative generation failed: {llm_e}", exc_info=True)
                    overall_assessment_points = {"strengths": self._extract_strengths(analysis_results), "concerns": self._extract_concerns(analysis_results)}; executive_summary_text = self._generate_fallback_summary(analysis_results, overall_assessment_points)
                    risk_narrative_text = "[Risk narrative generation failed.]"; compliance_narrative_text = "[Compliance narrative generation failed.]"; conclusion_text = "[Conclusion generation failed.]"
            else: logger.warning("NarrativeGenerator not available."); overall_assessment_points = {"strengths": self._extract_strengths(analysis_results), "concerns": self._extract_concerns(analysis_results)}; executive_summary_text = self._generate_fallback_summary(analysis_results, overall_assessment_points)

            # --- 3. Build Slides ---
            logger.info("Building presentation slides...")
            self._call_builder(builder.add_cover_slide, fund_info=fund_info)
            self._call_builder(builder.add_executive_summary_slide, analysis_results=analysis_results, summary_text=executive_summary_text, assessment_points=overall_assessment_points)
            self._call_builder(builder.add_fund_overview_slide, fund_info=fund_info)
            self._call_builder(builder.add_team_analysis_slide, team_data=team_data)
            self._call_builder(builder.add_portfolio_allocation_slide, portfolio_data=portfolio_data, market_analysis=market_analysis)
            self._call_builder(builder.add_wallet_security_analysis, wallet_analysis=wallet_analysis)
            self._call_builder(builder.add_risk_assessment_slides, risk_assessment=risk_assessment, risk_narrative=risk_narrative_text)
            self._call_builder(builder.add_compliance_analysis_slides, compliance_analysis=compliance_analysis, compliance_narrative=compliance_narrative_text)
            self._call_builder(builder.add_conclusion_slide, analysis_results=analysis_results, conclusion_text=conclusion_text, assessment_points=overall_assessment_points)

            # --- 4. Save Presentation ---
            # (Save logic remains the same)
            if not output_path:
                fund_name = fund_info.get("fund_name", "CryptoFund"); safe_fund_name = "".join(c if c.isalnum() else "_" for c in fund_name)[:50]; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir_config = getattr(builder, 'config', {}).get("output_dir", "reports") # Safer access
                project_root_from_report_gen = os.path.dirname(os.path.dirname(__file__)); output_dir = os.path.join(project_root_from_report_gen, output_dir_config)
                os.makedirs(output_dir, exist_ok=True); output_path = os.path.join(output_dir, f"{safe_fund_name}_DueDiligence_{timestamp}.pptx")
            builder.save(output_path)
            logger.info(f"Report generated successfully in {time.time() - start_time:.2f} seconds"); logger.info(f"Report saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating report: {e}", exc_info=True)
            try:
                if 'builder' in locals():
                    partial_path = output_path.replace(".pptx", "_error.pptx") if output_path else os.path.join(os.getcwd(), "reports", f"PARTIAL_REPORT_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx")
                    os.makedirs(os.path.dirname(partial_path), exist_ok=True); builder.save(partial_path)
                    logger.info(f"Partial report saved to {partial_path} due to errors."); return partial_path
            except Exception as save_e: logger.error(f"Failed to save partial report after error: {save_e}")
            raise

    # ==========================================================================
    # Internal Slide Creation Wrapper Methods (Calling the Builder safely)
    # ==========================================================================
    def _call_builder(self, builder_method, **kwargs):
        """Wrapper to safely call builder methods and log errors."""
        method_name = builder_method.__name__
        try:
            logger.debug(f"Calling builder method: {method_name}")
            # Clean kwargs for specific methods if necessary (e.g., remove analysis_results)
            data_for_builder = self._prepare_data_for_builder(method_name, kwargs)
            builder_method(**data_for_builder)
        except Exception as e:
            logger.error(f"Error calling builder method '{method_name}': {e}", exc_info=True)
            # Optionally add an error slide?
            # self.builder.add_text_slide(title=f"Error: {method_name}", content=f"Could not generate slide content:\n{traceback.format_exc()}")

    # Fix for ReportGenerator._prepare_data_for_builder method

    # Update to _prepare_data_for_builder method to properly handle colors

    def _prepare_data_for_builder(self, method_name: str, kwargs: Dict) -> Dict:
        """Prepares kwargs specifically for the target builder method."""
        # Extract analysis_results if present, but keep it for later use if needed
        analysis_results = kwargs.pop('analysis_results', None)
        
        if method_name == 'add_fund_overview_slide':
            # Simply pass through the fund_info
            return {"fund_info": kwargs.get("fund_info", {})}
            
        elif method_name == 'add_team_analysis_slide':
            # Simply pass through the team_data
            return {"team_data": kwargs.get("team_data", {})}
            
        elif method_name == 'add_portfolio_allocation_slide':
            # Pass through portfolio_data and market_analysis
            return {
                "portfolio_data": kwargs.get("portfolio_data", {}),
                "market_analysis": kwargs.get("market_analysis", {})
            }
            
        elif method_name == 'add_wallet_security_analysis':
            # Pass through wallet_analysis
            return {"wallet_analysis": kwargs.get("wallet_analysis", {})}
            
        elif method_name == 'add_risk_assessment_slides':
            # Prepare risk data for the new method
            risk_assessment = kwargs.get("risk_assessment", {})
            risk_narrative = kwargs.get("risk_narrative", "")

        if method_name == 'add_cover_slide':
            fund_info = kwargs.get("fund_info", {})
            return {
                "title": f"{fund_info.get('fund_name', 'Fund')} Due Diligence Report",
                "subtitle": "Comprehensive Analysis & Risk Assessment",
                "date": datetime.now().strftime("%B %d, %Y")
            }
            
        elif method_name == 'add_executive_summary_slide':
            # Ensure we return a dictionary even when analysis_results is None
            fund_info = kwargs.get("fund_info", {})
            risk_assessment = kwargs.get("risk_assessment", {})
            compliance_analysis = kwargs.get("compliance_analysis", {})
            summary_text = kwargs.get("summary_text", "")
            assessment_points = kwargs.get("assessment_points", {"strengths": [], "concerns": []})
            
            # Always return a dictionary with required parameters
            return {
                "title": "Executive Summary",
                "fund_name": fund_info.get("fund_name", "N/A"),
                "aum": format_currency(fund_info.get("aum", 0), in_millions=True),
                "strategy": summarize_text(fund_info.get("strategy", "N/A"), max_length=60),
                "risk_score": risk_assessment.get("overall_risk_score", 50),
                "risk_level": risk_assessment.get("risk_level", "Medium"),
                "risk_color": self._get_risk_color(risk_assessment.get("risk_level", "Medium")),
                "compliance_score": compliance_analysis.get("overall_compliance_score", 50),
                "compliance_level": compliance_analysis.get("compliance_level", "Medium"),
                "compliance_color": self._get_risk_color(compliance_analysis.get("compliance_level", "Medium")),
                "summary_text": summary_text,
                "key_strengths": assessment_points.get("strengths", []),
                "key_concerns": assessment_points.get("concerns", [])
            }
        elif method_name == 'add_risk_assessment_slides':
                risk_assessment = kwargs.get("risk_assessment", {})
                risk_narrative = kwargs.get("risk_narrative", "")
                if not risk_assessment or "error" in risk_assessment:
                    return {"title": "Risk Assessment", "content": "Risk assessment could not be generated."}
                    
                components = risk_assessment.get("risk_components", {})
                score = risk_assessment.get("overall_risk_score", 50)
                level = risk_assessment.get("risk_level", "Medium")
                factors = risk_assessment.get("risk_factors", [])
                mitigations = risk_assessment.get("suggested_mitigations", [])
                labels, values = [], []
                
                for rt, rd in components.items():
                    if isinstance(rd, dict):
                        labels.append(" ".join(w.capitalize() for w in rt.split("_")))
                        values.append(rd.get("score", 5.0))
                        
                risk_color = self._get_risk_color(level)
                risk_color_hex = self.design.rgb_to_hex(risk_color)  # Use the design object's method
                
                return {
                    "overview_data": {
                        "title": "Risk Assessment Overview",
                        "overall_risk_score": score,
                        "risk_level": level,
                        "risk_color": risk_color_hex,
                        "radar_labels": labels,
                        "radar_values": values,
                        "risk_narrative": risk_narrative
                    },
                    "factors_data": {
                        "title": "Key Identified Risk Factors",
                        "risk_factors": [f"• {f.get('factor','?')}" for f in factors if isinstance(f, dict)][:8]
                    } if factors else None,
                    "mitigation_data": {
                        "title": "Suggested Risk Mitigations",
                        "content": "\n".join([f"• {m}" for m in mitigations[:8]]), 
                        "text_size": 14
                    } if mitigations else None
                }
                
        elif method_name == 'add_compliance_analysis_slides':
                compliance_analysis = kwargs.get("compliance_analysis", {})
                compliance_narrative = kwargs.get("compliance_narrative", "")
                if not compliance_analysis or "error" in compliance_analysis:
                    return {"title": "Compliance Analysis", "content": "Compliance analysis could not be generated."}
                    
                score = compliance_analysis.get("overall_compliance_score", 50)
                level = compliance_analysis.get("compliance_level", "Medium")
                juris = compliance_analysis.get("jurisdictions", [])
                kyc = compliance_analysis.get("kyc_aml_assessment", {})
                reg_status = compliance_analysis.get("regulatory_status", {})
                gaps = compliance_analysis.get("compliance_gaps", [])
                
                reg_table_data = None
                if reg_status:
                    reg_table_data = [["Jurisdiction", "Status", "Score"]]
                    for j, s in reg_status.items():
                        if isinstance(s, dict):
                            reg_table_data.append([
                                j, 
                                s.get("registration_status", "?"), 
                                f"{s.get('compliance_score', 0):.1f}%"
                            ])
                
                return {
                    "overview_data": {
                        "title": "Compliance Overview", 
                        "overall_score": score, 
                        "compliance_level": level, 
                        "jurisdictions": ", ".join(juris) if juris else "N/A", 
                        "kyc_aml_coverage": kyc.get("coverage_score", 0), 
                        "compliance_narrative": compliance_narrative
                    },
                    "reg_status_data": {
                        "title": "Regulatory Status by Jurisdiction", 
                        "table_data": reg_table_data
                    } if reg_table_data else None,
                    "gaps_data": {
                        "title": "Identified Compliance Gaps", 
                        "content": "\n".join([f"• {g}" for g in gaps[:8]]), 
                        "text_size": 14
                    } if gaps else None
                }
                
        elif method_name == 'add_conclusion_slide':
                risk_score = kwargs.get("risk_assessment", {}).get("overall_risk_score", 50)
                risk_level = kwargs.get("risk_assessment", {}).get("risk_level", "Medium")
                compliance_score = kwargs.get("compliance_analysis", {}).get("overall_compliance_score", 50)
                compliance_level = kwargs.get("compliance_analysis", {}).get("compliance_level", "Medium")
                return {
                    "title": "Conclusion & Overall Assessment",
                    "fund_name": kwargs.get("fund_info", {}).get("fund_name", "N/A"),
                    "risk_level": risk_level, 
                    "risk_score": risk_score,
                    "compliance_level": compliance_level, 
                    "compliance_score": compliance_score,
                    "conclusion_summary": kwargs.get("conclusion_text", ""),
                    "strengths": kwargs.get("assessment_points", {}).get("strengths", [])[:3],
                    "concerns": kwargs.get("assessment_points", {}).get("concerns", [])[:3]
                }

            # Default: return original kwargs if no specific prep needed
        logger.warning(f"No specific data preparation defined for builder method '{method_name}'. Passing raw relevant args.")
            # Filter kwargs to only pass what might be relevant
        relevant_keys = ["title", "fund_info", "team_data", "portfolio_data", "market_analysis", 
                            "wallet_analysis", "risk_assessment", "compliance_analysis"]
        return {k: v for k, v in kwargs.items() if k in relevant_keys or 
                    k in ["summary_text", "assessment_points", "risk_narrative", "compliance_narrative", "conclusion_text"]}
    # --- Fallback Methods ---
    def _generate_fallback_summary(self, analysis_results: Dict[str, Any], assessment_points: Dict[str, List[str]]) -> str:
        # (Keep existing fallback logic)
        fund_name=analysis_results.get("fund_info", {}).get("fund_name", "The fund"); risk_level=analysis_results.get("risk_assessment", {}).get("risk_level", "Medium"); compliance_level=analysis_results.get("compliance_analysis", {}).get("compliance_level", "Medium")
        strengths=assessment_points.get("strengths", ["standard procedures"]); concerns=assessment_points.get("concerns", ["market risks"])
        return f"{fund_name} shows a {risk_level.lower()} risk and {compliance_level.lower()} compliance profile. Strengths include {strengths[0].lower().replace('.','')}. Concerns involve {concerns[0].lower().replace('.','')}.";

    def _extract_strengths(self, analysis_results: Dict[str, Any]) -> List[str]:
        # (Keep existing fallback logic)
        strengths = []; portfolio_data = analysis_results.get("portfolio_data", {}); wallet_analysis = analysis_results.get("wallet_analysis", {}); compliance_analysis = analysis_results.get("compliance_analysis", {}); team_data = analysis_results.get("team_data", {}); portfolio_values = list(portfolio_data.values()) if portfolio_data else [1.1];
        if portfolio_data and np.max(portfolio_values) <= 0.3: strengths.append("Good portfolio diversification."); wallet_score = wallet_analysis.get("aggregate_stats", {}).get("average_risk_score", 100); 
        if wallet_score < 40: strengths.append("Strong wallet security posture."); compliance_score = compliance_analysis.get("overall_compliance_score", 0); 
        if compliance_score > 75: strengths.append("High level of regulatory compliance."); 
        if len(team_data.get("key_personnel", [])) >= 3: strengths.append("Experienced management team."); 
        if not strengths: strengths.append("Standard operational setup observed."); return strengths[:3]

    def _extract_concerns(self, analysis_results: Dict[str, Any]) -> List[str]:
        # (Keep existing fallback logic)
        concerns = []; portfolio_data = analysis_results.get("portfolio_data", {}); risk_assessment = analysis_results.get("risk_assessment", {}); compliance_analysis = analysis_results.get("compliance_analysis", {}); wallet_analysis = analysis_results.get("wallet_analysis", {}); portfolio_values = list(portfolio_data.values()) if portfolio_data else [0]; 
        if portfolio_data and np.max(portfolio_values) > 0.5: max_asset = max(portfolio_data.items(), key=lambda x: x[1])[0]; concerns.append(f"High portfolio concentration ({max_asset})."); risk_score = risk_assessment.get("overall_risk_score", 0); 
        if risk_score > 70: concerns.append("Elevated overall risk profile."); compliance_gaps = compliance_analysis.get("compliance_gaps", []); 
        if compliance_gaps: concerns.append(f"Identified compliance gaps ({len(compliance_gaps)} areas)."); wallet_score = wallet_analysis.get("aggregate_stats", {}).get("average_risk_score", 0); 
        if wallet_score > 60: concerns.append("Potential wallet security weaknesses."); 
        if not concerns: concerns.append("Standard market and operational risks apply."); return concerns[:3]

    def _get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        risk_level_map = { "Very Low": self.design.RISK_VERY_LOW, "Low": self.design.RISK_LOW,"Medium-Low": self.design.RISK_MEDIUM_LOW, "Medium": self.design.RISK_MEDIUM,"Medium-High": self.design.RISK_MEDIUM_HIGH, "High": self.design.RISK_HIGH,"Very High": self.design.RISK_VERY_HIGH }
        return risk_level_map.get(risk_level, self.design.RISK_MEDIUM)