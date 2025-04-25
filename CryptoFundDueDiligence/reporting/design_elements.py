"""
Design Elements Module

This module defines the design constants, color schemes, and visual styling elements
for creating consistent, professional PowerPoint presentations. It centralizes
all design decisions to maintain a cohesive visual identity across reports.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from venv import logger

class DesignElements:
    """
    Defines the design elements used throughout the presentation to ensure
    consistency in colors, fonts, spacing, and overall visual styling.
    """
    
    def __init__(self):
        """Initialize design constants."""
        # Brand colors (RGB tuples)
        self.PRIMARY_COLOR = (0, 102, 204)      # Blue
        self.SECONDARY_COLOR = (102, 45, 145)   # Purple
        self.ACCENT_COLOR = (255, 153, 0)       # Orange
        self.DARK_COLOR = (51, 51, 51)          # Dark gray
        self.LIGHT_COLOR = (245, 245, 245)      # Light gray
        self.ACCENT_1 = (255, 153, 0)           # Same as ACCENT_COLOR 
        self.ACCENT_2 = (102, 45, 145)    
        # Background colors
        self.COVER_BACKGROUND = (25, 33, 51)    # Dark blue-gray for cover
        self.SLIDE_BACKGROUND = (255, 255, 255) # White for content slides
        
        # Risk level colors
        self.RISK_VERY_LOW = (0, 153, 51)       # Green
        self.RISK_LOW = (102, 204, 51)          # Light green
        self.RISK_MEDIUM_LOW = (153, 204, 0)    # Yellow-green
        self.RISK_MEDIUM = (255, 204, 0)        # Yellow
        self.RISK_MEDIUM_HIGH = (255, 153, 51)  # Orange
        self.RISK_HIGH = (255, 51, 0)           # Red-orange
        self.RISK_VERY_HIGH = (204, 0, 0)       # Red
        
        # Chart colors
        self.CHART_COLORS = [
            (0, 102, 204),   # Blue
            (255, 153, 0),   # Orange
            (102, 45, 145),  # Purple
            (0, 153, 51),    # Green
            (204, 0, 0),     # Red
            (51, 153, 255),  # Light blue
            (255, 102, 0),   # Dark orange
            (153, 51, 204),  # Light purple
            (51, 204, 51),   # Light green
            (153, 0, 0),     # Dark red
            (0, 204, 255),   # Cyan
            (204, 102, 0)    # Brown
        ]
        
        # Font settings
        self.TITLE_FONT = "Segoe UI"
        self.BODY_FONT = "Segoe UI"
        self.TITLE_SIZE = 32
        self.SUBTITLE_SIZE = 20
        self.HEADING_SIZE = 18
        self.BODY_SIZE = 14
        self.SMALL_SIZE = 10
        
        # Spacing (in inches)
        self.MARGIN = 0.5
        self.TITLE_MARGIN_TOP = 0.5
        self.SECTION_SPACING = 0.2
        self.PARAGRAPH_SPACING = 0.1
        
        # Slide layouts
        self.TITLE_LAYOUT_IDX = 0
        self.TITLE_CONTENT_LAYOUT_IDX = 1
        self.SECTION_LAYOUT_IDX = 2
        self.TWO_CONTENT_LAYOUT_IDX = 3
        self.COMPARISON_LAYOUT_IDX = 4
        self.TITLE_ONLY_LAYOUT_IDX = 5
        self.BLANK_LAYOUT_IDX = 6
        self.CHART_LAYOUT_IDX = 7
        
        # Chart settings
        self.CHART_WIDTH = 6.0      # Default chart width in inches
        self.CHART_HEIGHT = 4.0     # Default chart height in inches
        self.CHART_LEFT = 1.5       # Default chart left position
        self.CHART_TOP = 2.0        # Default chart top position
        
        # Table settings
        self.TABLE_HEADER_BG = (230, 230, 230)  # Light gray for table headers
        self.TABLE_STRIPE_BG = (245, 245, 250)  # Very light blue for alternating rows
        
        # Icon paths (if using custom icons)
        self.ICON_PATH = "reporting/templates/icons"
        
        # Special element settings
        self.RISK_GAUGE_SIZE = 3.0   # Size for risk gauge charts
        self.RISK_GAUGE_LEFT = 3.0   # Left position for risk gauge
        self.RISK_GAUGE_TOP = 2.5    # Top position for risk gauge

        self.POSITIVE = self.RISK_LOW # Example: Use light green for positive
        self.NEGATIVE = self.RISK_HIGH 
        self.ACCENT_1 = (255, 153, 0) 
        
    TEXT_LIGHT = (255, 255, 255)  # White
    TEXT_DARK = (0, 0, 0)        # Black    
    def get_risk_scale_color(self, value: float, invert: bool = False) -> Tuple[int, int, int]:
        """
        Get color on a risk scale based on value (0-100).
        
        Args:
            value: Value on a 0-100 scale
            invert: If True, low values are red and high values are green
            
        Returns:
            RGB color tuple
        """
        # Normalize value to 0-100 range
        normalized = max(0, min(value, 100))
        
        # Apply inversion if needed
        if invert:
            normalized = 100 - normalized
        
        # Determine color based on thresholds
        if normalized < 20:
            return self.RISK_VERY_HIGH
        elif normalized < 40:
            return self.RISK_HIGH
        elif normalized < 50:
            return self.RISK_MEDIUM_HIGH
        elif normalized < 60:
            return self.RISK_MEDIUM
        elif normalized < 70: 
            return self.RISK_MEDIUM_LOW
        elif normalized < 85:
            return self.RISK_LOW
        else:
            return self.RISK_VERY_LOW
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
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
    def get_change_color(self, value: float, positive_is_good: bool = True) -> Tuple[int, int, int]:
        """
        Get color for percentage changes.
        
        Args:
            value: Percent change value
            positive_is_good: If True, positive values are green (default for returns)
            
        Returns:
            RGB color tuple
        """
        if positive_is_good:
            # For returns, profit, etc.: positive is good (green)
            if value > 0:
                return self.RISK_LOW  # Green
            elif value < 0:
                return self.RISK_HIGH  # Red
            else:
                return self.DARK_COLOR  # Neutral
        else:
            # For costs, risk, etc.: positive is bad (red)
            if value > 0:
                return self.RISK_HIGH  # Red
            elif value < 0:
                return self.RISK_LOW  # Green
            else:
                return self.DARK_COLOR  # Neutral