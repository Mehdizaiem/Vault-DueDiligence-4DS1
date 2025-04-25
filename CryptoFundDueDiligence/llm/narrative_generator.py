# Path: llm/narrative_generator.py
import logging
import re
from typing import Dict, List, Any
from llm.groq_client import GroqClient
from llm.prompt_builder import ReportPromptBuilder

logger = logging.getLogger(__name__)

class NarrativeGenerator:
    def __init__(self, groq_client: GroqClient, prompt_builder: ReportPromptBuilder):
        self.client = groq_client
        self.prompter = prompt_builder

    def _generate_narrative(self, prompt: str, section_name: str) -> str:
        logger.info(f"Generating narrative for section: {section_name}")
        try:
            response = self.client.generate(prompt)
            if response.startswith("Error:"):
                logger.error(f"LLM generation failed for {section_name}: {response}")
                return f"[{section_name} Narrative Generation Failed]"
            logger.info(f"Successfully generated narrative for {section_name}")
            return response.strip()
        except Exception as e:
            logger.error(f"Exception during {section_name} narrative generation: {e}", exc_info=True)
            return f"[{section_name} Narrative Generation Exception]"

    def generate_executive_summary(self, structured_data: Dict) -> str:
        prompt = self.prompter.build_executive_summary_prompt(
            fund_info=structured_data.get("fund_info", {}),
            risk_summary=structured_data.get("risk_summary", {}),
            compliance_summary=structured_data.get("compliance_summary", {}),
            key_points=structured_data.get("key_points", {})
        )
        return self._generate_narrative(prompt, "Executive Summary")

    def generate_risk_narrative(self, structured_data: Dict) -> str:
        prompt = self.prompter.build_risk_narrative_prompt(
            risk_assessment=structured_data.get("risk_assessment", {})
        )
        return self._generate_narrative(prompt, "Risk Narrative")

    def generate_compliance_narrative(self, structured_data: Dict) -> str:
        prompt = self.prompter.build_compliance_narrative_prompt(
            compliance_analysis=structured_data.get("compliance_analysis", {})
        )
        return self._generate_narrative(prompt, "Compliance Narrative")

    def generate_conclusion(self, structured_data: Dict) -> str:
        prompt = self.prompter.build_conclusion_prompt(
            fund_info=structured_data.get("fund_info", {}),
            risk_summary=structured_data.get("risk_summary", {}),
            compliance_summary=structured_data.get("compliance_summary", {}),
            key_points=structured_data.get("key_points", {})
        )
        return self._generate_narrative(prompt, "Conclusion")

    def generate_overall_assessment_points(self, structured_data: Dict) -> Dict[str, List[str]]:
        prompt = self.prompter.build_overall_assessment_points_prompt(structured_data)
        raw_response = self._generate_narrative(prompt, "Overall Assessment Points")

        strengths = []
        concerns = []

        if raw_response.startswith("Error:") or "[" in raw_response: # Handle generation failure
            logger.warning(f"Could not generate overall assessment points: {raw_response}")
            return {"strengths": ["Assessment generation failed."], "concerns": ["Assessment generation failed."]}

        lines = raw_response.split('\n')
        for line in lines:
            line_clean = line.strip()
            if line_clean.startswith("STRENGTH:"):
                strengths.append(line_clean.replace("STRENGTH:", "").strip())
            elif line_clean.startswith("CONCERN:"):
                concerns.append(line_clean.replace("CONCERN:", "").strip())

        # Fallback if parsing fails or returns empty
        if not strengths:
            strengths = ["Could not automatically determine key strengths."]
        if not concerns:
            concerns = ["Could not automatically determine key concerns."]

        return {"strengths": strengths[:3], "concerns": concerns[:3]} # Limit to top 3