"""
Report Utilities Module

This module provides helper functions for formatting and transforming data
in the report generation process. It handles number formatting, text manipulation,
and other common utilities needed for PowerPoint report generation.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_currency(value: Union[int, float], in_millions: bool = False, 
                   currency_symbol: str = "$", decimal_places: int = 2) -> str:
    """
    Format a numeric value as currency.
    
    Args:
        value: The numeric value to format
        in_millions: Whether the value is already in millions
        currency_symbol: Currency symbol to use
        decimal_places: Number of decimal places to display
        
    Returns:
        Formatted currency string
    """
    try:
        # Convert to float to ensure proper formatting
        float_value = float(value)
        
        # Format for millions if specified
        if in_millions:
            # Value is already in millions, just format it
            formatted = f"{currency_symbol}{float_value:,.{decimal_places}f}M"
        else:
            # Check for appropriate scale
            if abs(float_value) >= 1_000_000_000:
                # Billions
                formatted = f"{currency_symbol}{float_value / 1_000_000_000:,.{decimal_places}f}B"
            elif abs(float_value) >= 1_000_000:
                # Millions
                formatted = f"{currency_symbol}{float_value / 1_000_000:,.{decimal_places}f}M"
            elif abs(float_value) >= 1_000:
                # Thousands
                formatted = f"{currency_symbol}{float_value / 1_000:,.{decimal_places}f}K"
            else:
                # Regular format
                formatted = f"{currency_symbol}{float_value:,.{decimal_places}f}"
        
        return formatted
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting currency value {value}: {e}")
        return f"{currency_symbol}0.00"

def calculate_change_text(percent_change: float, include_symbol: bool = True,
                        positive_prefix: str = "+", negative_prefix: str = "-",
                        with_color_tag: bool = False) -> str:
    """
    Format a percentage change with appropriate sign and optional color tag.
    
    Args:
        percent_change: The percentage change value
        include_symbol: Whether to include the percentage symbol
        positive_prefix: Prefix for positive changes (default: "+")
        negative_prefix: Prefix for negative changes (default: "-")
        with_color_tag: Whether to include HTML-style color tags
        
    Returns:
        Formatted change text
    """
    try:
        float_value = float(percent_change)
        
        # Determine sign and color
        if float_value > 0:
            prefix = positive_prefix
            color = "green"
        elif float_value < 0:
            prefix = negative_prefix
            # Make sure we don't add a double negative
            float_value = abs(float_value)
            color = "red"
        else:
            prefix = ""
            color = "gray"
        
        # Format the value
        if include_symbol:
            formatted = f"{prefix}{float_value:.2f}%"
        else:
            formatted = f"{prefix}{float_value:.2f}"
        
        # Add color tags if requested
        if with_color_tag:
            return f"<font color='{color}'>{formatted}</font>"
        else:
            return formatted
            
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting percent change {percent_change}: {e}")
        return "0.00%"

def summarize_text(text: str, max_length: int = 100, 
                 end_with_ellipsis: bool = True) -> str:
    """
    Summarize text to a maximum length, optionally ending with ellipsis.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        end_with_ellipsis: Whether to add ellipsis to truncated text
        
    Returns:
        Summarized text
    """
    if not text:
        return ""
    
    # If text is already shorter than max_length, return it as is
    if len(text) <= max_length:
        return text
    
    # Find the last space before max_length to avoid cutting words
    last_space = text[:max_length].rfind(" ")
    
    if last_space == -1:
        # No space found, just cut at max_length
        summary = text[:max_length]
    else:
        # Cut at the last space
        summary = text[:last_space]
    
    # Add ellipsis if requested
    if end_with_ellipsis:
        summary += "..."
    
    return summary

def format_date(date_str: Optional[str] = None, 
              date_format: str = "%B %d, %Y") -> str:
    """
    Format a date string or current date.
    
    Args:
        date_str: Date string to format (default: current date)
        date_format: Format string for strftime
        
    Returns:
        Formatted date string
    """
    try:
        if date_str:
            # Try to parse the date string
            if isinstance(date_str, datetime):
                date_obj = date_str
            else:
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched, raise error
                    raise ValueError(f"Could not parse date: {date_str}")
        else:
            # Use current date
            date_obj = datetime.now()
        
        # Format the date
        return date_obj.strftime(date_format)
        
    except Exception as e:
        logger.warning(f"Error formatting date {date_str}: {e}")
        return datetime.now().strftime(date_format)

def clean_text_for_slide(text: str) -> str:
    """
    Clean and prepare text for insertion into a PowerPoint slide.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace problematic characters
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\u2019": "'",  # Smart single quote
        "\u201C": "\"",  # Smart left double quote
        "\u201D": "\"",  # Smart right double quote
        "\u2013": "-",   # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "..."  # Ellipsis
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_key_points(text: str, max_points: int = 5, 
                     min_length: int = 40) -> List[str]:
    """
    Extract key bullet points from longer text.
    
    Args:
        text: Text to extract from
        max_points: Maximum number of bullet points to extract
        min_length: Minimum length of each bullet point
        
    Returns:
        List of extracted bullet points
    """
    if not text:
        return []
    
    # Try to find existing bullet points
    bullet_pattern = r'[•\-\*]\s*(.*?)(?=\n[•\-\*]|\n\n|$)'
    bullets = re.findall(bullet_pattern, text)
    
    if bullets:
        # Filter out short bullet points
        bullets = [b.strip() for b in bullets if len(b.strip()) >= min_length]
        # Limit to max_points
        return bullets[:max_points]
    
    # If no bullet points found, extract sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
    
    # Limit to max_points
    return sentences[:max_points]

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color code string (e.g., "FF0000").
    """
    if not isinstance(rgb, tuple) or len(rgb) != 3:
        logger.warning(f"Invalid RGB tuple provided to rgb_to_hex: {rgb}. Returning default black '000000'.")
        return "000000"
    try:
        r, g, b = [max(0, min(int(val), 255)) for val in rgb]
        return f"{r:02X}{g:02X}{b:02X}"
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting RGB {rgb} to hex: {e}")
        return "000000"

def create_risk_label(score: float, with_emoji: bool = True) -> str:
    """
    Create a risk label based on a score (0-100).
    
    Args:
        score: Risk score (0-100)
        with_emoji: Whether to include emoji indicators
        
    Returns:
        Risk label string
    """
    if score < 20:
        return "🔵 Very Low" if with_emoji else "Very Low"
    elif score < 40:
        return "🟢 Low" if with_emoji else "Low"
    elif score < 60:
        return "🟡 Medium" if with_emoji else "Medium"
    elif score < 80:
        return "🟠 High" if with_emoji else "High"
    else:
        return "🔴 Very High" if with_emoji else "Very High"

def truncate_address(address: str, chars: int = 6) -> str:
    """
    Truncate a blockchain address for display.
    
    Args:
        address: Full blockchain address
        chars: Number of characters to keep on each end
        
    Returns:
        Truncated address string
    """
    if not address or len(address) <= (chars * 2 + 3):
        return address
    
    return f"{address[:chars]}...{address[-chars:]}"

def parse_percentage(value: Union[str, int, float]) -> float:
    """
    Parse a percentage value from various formats.
    
    Args:
        value: Percentage value as string or number
        
    Returns:
        Parsed percentage as decimal (0-1)
    """
    try:
        if isinstance(value, (int, float)):
            # If it's already a number, check if it needs conversion
            if value > 1.0:
                return value / 100.0
            return value
        
        # Parse string
        if not value:
            return 0.0
            
        # Remove percentage sign and spaces
        cleaned = value.replace('%', '').strip()
        
        # Convert to float
        result = float(cleaned)
        
        # Convert to decimal if needed
        if result > 1.0:
            return result / 100.0
            
        return result
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing percentage value {value}: {e}")
        return 0.0