"""
Configuration for report generation
"""
from pptx.dml.color import RGBColor
from pptx.util import Pt, Inches

# Report topics
REPORT_TOPICS = [
    "legal_regulatory",
    "team_background",
    "technical",
    "financial",
    "market_price",
    "governance",
    "risk",
    "on_chain"
]

# Template settings
TEMPLATE_SETTINGS = {
    "font_family": "Arial",
    "title_font_size": Pt(44),
    "subtitle_font_size": Pt(32),
    "body_font_size": Pt(18),
    "caption_font_size": Pt(14),
    "primary_color": RGBColor(0, 84, 164),  # Navy blue
    "secondary_color": RGBColor(127, 127, 127),  # Gray
    "accent_color": RGBColor(255, 165, 0),  # Orange
    "background_color": RGBColor(255, 255, 255),  # White
    "text_color": RGBColor(0, 0, 0),  # Black
    "margin_top": Inches(0.5),
    "margin_bottom": Inches(0.5),
    "margin_left": Inches(0.75),
    "margin_right": Inches(0.75)
}

# Chart settings
CHART_SETTINGS = {
    "style": "seaborn-v0_8-whitegrid",
    "figure_size": (10, 6),
    "colors": ["#0054A4", "#7F7F7F", "#FFA500", "#2ECC71", "#E74C3C"],
    "font_size": 12,
    "title_size": 16,
    "line_width": 2,
    "grid_alpha": 0.3
}

# Table settings
TABLE_SETTINGS = {
    "header_fill": RGBColor(0, 84, 164),
    "header_text_color": RGBColor(255, 255, 255),
    "row_fill_1": RGBColor(255, 255, 255),
    "row_fill_2": RGBColor(242, 242, 242),
    "border_color": RGBColor(127, 127, 127),
    "border_width": Pt(0.5)
}

# Report sections
REPORT_SECTIONS = {
    "executive_summary": {
        "title": "Executive Summary",
        "description": "High-level overview of the cryptocurrency"
    },
    "market_analysis": {
        "title": "Market Analysis",
        "description": "Current market position and trends"
    },
    "technical_assessment": {
        "title": "Technical Assessment",
        "description": "Blockchain technology and architecture"
    },
    "financial_analysis": {
        "title": "Financial Analysis",
        "description": "Tokenomics and financial metrics"
    },
    "risk_assessment": {
        "title": "Risk Assessment",
        "description": "Comprehensive risk analysis"
    },
    "conclusion": {
        "title": "Investment Conclusion",
        "description": "Final recommendations and outlook"
    }
}