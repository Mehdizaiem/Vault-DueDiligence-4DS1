# Path: llm/prompt_builder.py
from typing import Dict, List, Any

class ReportPromptBuilder:

    def _format_data_as_context(self, data: Dict, title: str) -> str:
        context = f"--- {title.upper()} DATA ---\n"
        for key, value in data.items():
            if isinstance(value, dict):
                context += f"{key.replace('_', ' ').title()}:\n"
                for sub_key, sub_value in value.items():
                    context += f"  - {sub_key.replace('_', ' ').title()}: {sub_value}\n"
            elif isinstance(value, list):
                 if value: # Only include if list is not empty
                    context += f"{key.replace('_', ' ').title()}:\n"
                    for item in value[:5]: # Limit list items
                        context += f"  - {item}\n"
                    if len(value) > 5:
                        context += "  - ... (more items exist)\n"
            else:
                context += f"{key.replace('_', ' ').title()}: {value}\n"
        context += "---\n"
        return context

    def build_executive_summary_prompt(self, fund_info: Dict, risk_summary: Dict, compliance_summary: Dict, key_points: Dict) -> str:
        context = ""
        context += self._format_data_as_context(fund_info, "Fund Information")
        context += self._format_data_as_context(risk_summary, "Risk Summary")
        context += self._format_data_as_context(compliance_summary, "Compliance Summary")
        context += self._format_data_as_context(key_points, "Key Strengths and Concerns") # Assumes key_points = {"strengths": [...], "concerns": [...]}

        prompt = f"""
        You are a Senior Due Diligence Analyst summarizing findings for an executive audience.
        Based *only* on the structured data provided below, write a concise (around 150 words) and professional Executive Summary for a cryptocurrency fund due diligence report.

        Instructions:
        1. Start with the fund's name and a brief mention of its AUM and core strategy.
        2. State the overall risk level and score clearly.
        3. State the overall compliance level and score clearly.
        4. Briefly mention the top 1-2 key strengths identified.
        5. Briefly mention the top 1-2 key concerns identified.
        6. Maintain an objective and formal tone. Do NOT add any information not present in the context.

        CONTEXT:
        {context}

        Write the Executive Summary:
        """
        return prompt.strip()

    def build_risk_narrative_prompt(self, risk_assessment: Dict) -> str:
        context = self._format_data_as_context(risk_assessment, "Risk Assessment Data")

        prompt = f"""
        You are a Risk Analyst explaining the risk profile of a crypto fund.
        Based *only* on the provided risk assessment data (component scores, overall score, level, and key factors), write a narrative paragraph (approx. 100-120 words) summarizing the fund's main risk exposures.

        Instructions:
        1. Start by stating the overall risk level.
        2. Identify the 2-3 risk components with the highest scores (most significant risks).
        3. Briefly explain *why* these might be the primary risks, referencing the key risk factors provided.
        4. Do NOT suggest mitigations or add information not present in the context.
        5. Maintain an analytical and objective tone.

        CONTEXT:
        {context}

        Write the Risk Narrative Summary:
        """
        return prompt.strip()

    def build_compliance_narrative_prompt(self, compliance_analysis: Dict) -> str:
        context = self._format_data_as_context(compliance_analysis, "Compliance Analysis Data")

        prompt = f"""
        You are a Compliance Officer summarizing the compliance status of a crypto fund.
        Based *only* on the provided compliance analysis data (overall score, level, jurisdictions, KYC/AML score, gaps), write a narrative paragraph (approx. 100-120 words) summarizing the fund's compliance posture.

        Instructions:
        1. Start by stating the overall compliance level and score.
        2. Mention the adequacy of the KYC/AML framework based on its score/coverage.
        3. Briefly touch upon the regulatory status/coverage in key jurisdictions mentioned.
        4. Reference 1-2 of the most significant compliance gaps, if any are listed.
        5. Maintain a factual and neutral tone. Do NOT add external regulatory knowledge or opinions.

        CONTEXT:
        {context}

        Write the Compliance Narrative Summary:
        """
        return prompt.strip()

    def build_conclusion_prompt(self, fund_info: Dict, risk_summary: Dict, compliance_summary: Dict, key_points: Dict) -> str:
        context = ""
        context += self._format_data_as_context(fund_info, "Fund Information")
        context += self._format_data_as_context(risk_summary, "Risk Summary")
        context += self._format_data_as_context(compliance_summary, "Compliance Summary")
        context += self._format_data_as_context(key_points, "Key Strengths and Concerns")

        prompt = f"""
        You are a Senior Due Diligence Analyst writing the conclusion for a report.
        Based *only* on the structured data provided below, write a concluding paragraph (approx. 100 words) for the cryptocurrency fund due diligence report.

        Instructions:
        1. Briefly restate the fund's name.
        2. Synthesize the overall findings regarding the risk level and compliance level.
        3. Reiterate the single most significant strength and the single most significant concern identified from the key points provided.
        4. Provide a final concluding sentence summarizing the overall assessment perspective (e.g., favorable profile, requires caution, significant issues).
        5. Maintain an objective tone. Do NOT introduce new information or recommendations.

        CONTEXT:
        {context}

        Write the Conclusion:
        """
        return prompt.strip()

    def build_overall_assessment_points_prompt(self, all_structured_data: Dict) -> str:
        context = "---\nKEY FINDINGS SUMMARY:\n"
        # Simplified context generation for this specific prompt
        for section, data in all_structured_data.items():
             context += f"\n# {section.replace('_', ' ').title()} Highlights:\n"
             if isinstance(data, dict):
                 for key, value in list(data.items())[:4]: # Limit items per section
                     context += f"- {key.replace('_', ' ').title()}: {str(value)[:100]}\n" # Truncate long values
             elif isinstance(data, list):
                 for item in data[:3]: # Limit list items
                     context += f"- {str(item)[:100]}\n"
             else:
                 context += f"- {str(data)[:150]}\n" # General data
        context += "---\n"

        prompt = f"""
        You are an AI synthesizing findings from a crypto fund due diligence process.
        Based *strictly* on the Key Findings Summary provided below, identify:
        1. The top 3 most significant OVERALL strengths of the fund.
        2. The top 3 most significant OVERALL concerns or weaknesses of the fund.

        Consider the interplay between different findings (e.g., strong team but weak compliance).

        Instructions:
        - List strengths starting with "STRENGTH:"
        - List concerns starting with "CONCERN:"
        - Be concise for each point.
        - Base your points *only* on the information given in the context. Do not infer or add external knowledge.
        - If you cannot confidently identify 3 strengths or 3 concerns from the data, list fewer.

        CONTEXT:
        {context}

        Identify the Top 3 Strengths and Top 3 Concerns:
        STRENGTH:
        STRENGTH:
        STRENGTH:
        CONCERN:
        CONCERN:
        CONCERN:
        """
        return prompt.strip()