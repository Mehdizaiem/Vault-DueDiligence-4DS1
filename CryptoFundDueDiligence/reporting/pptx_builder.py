"""
PowerPoint Builder Module

This module handles the creation and manipulation of PowerPoint presentations,
providing a high-level API for adding professionally designed slides with
various layouts, charts, and visual elements.
"""

import os
import logging
import math
from typing import Dict, List, Any, Optional, Union, Tuple

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.dml.color import RGBColor
from pptx.text.text import _Paragraph  # For type hinting

# Assuming these exist in the specified locations
from reporting.chart_factory import ChartFactory
from reporting.design_elements import DesignElements

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PresentationBuilder:
    """
    Provides high-level methods for building sophisticated PowerPoint presentations
    with consistent styling, professional layouts, and data visualizations.

    Attributes:
        design (DesignElements): Instance for accessing design constants (colors, etc.).
        chart_factory (ChartFactory): Instance for creating various chart types.
        prs (Presentation): The python-pptx Presentation object being built.
        slide_width (Emu): Width of the slides in EMUs.
        slide_height (Emu): Height of the slides in EMUs.
        slide_count (int): Counter for the number of slides added.
    """

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize the presentation builder.

        Args:
            template_path: Path to a PowerPoint template (.pptx file, optional).
                           If provided and exists, it's used as the base.
                           Otherwise, a default presentation is created.
        """
        self.design = DesignElements()
        self.chart_factory = ChartFactory(self.design) # Pass design elements if needed by chart factory

        # Initialize presentation
        try:
            if template_path and os.path.exists(template_path):
                self.prs = Presentation(template_path)
                logger.info(f"Using template: {template_path}")
            else:
                self.prs = Presentation()
                logger.info("Using default presentation template")
                # Set default slide size if needed (widescreen 16:9)
                self.prs.slide_width = Inches(13.33)
                self.prs.slide_height = Inches(7.5)
        except Exception as e:
            logger.error(f"Error initializing Presentation object: {e}", exc_info=True)
            logger.warning("Falling back to a new default presentation.")
            self.prs = Presentation()
            self.prs.slide_width = Inches(13.33)
            self.prs.slide_height = Inches(7.5)

        # Store slide dimensions for reference
        self.slide_width = self.prs.slide_width
        self.slide_height = self.prs.slide_height

        # Track metrics
        self.slide_count = len(self.prs.slides) # Start count based on template slides
        logger.info(f"Initialized PresentationBuilder with {self.slide_count} existing slides.")

    def save(self, output_path: str) -> None:
        """
        Save the presentation to a file.

        Creates the output directory if it doesn't exist.

        Args:
            output_path: Path where to save the .pptx file.
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir: # Ensure it's not an empty string (saving in current dir)
                os.makedirs(output_dir, exist_ok=True)

            # Save the presentation
            self.prs.save(output_path)
            logger.info(f"Presentation saved to {output_path} with a total of {len(self.prs.slides)} slides ({self.slide_count} added by builder).")
        except Exception as e:
            logger.error(f"Failed to save presentation to {output_path}: {e}", exc_info=True)
            raise # Re-raise the exception after logging

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    def _get_slide_layout(self, layout_index: int):
        """Safely retrieves a slide layout by index."""
        try:
            return self.prs.slide_layouts[layout_index]
        except IndexError:
            logger.warning(f"Slide layout index {layout_index} out of bounds. Using layout 0 (Title Slide).")
            return self.prs.slide_layouts[0] # Fallback to a common layout

    def _style_text_frame(self, text_frame,
                          font_size: Optional[Pt] = None,
                          font_color: Optional[str] = None, # Hex string
                          bold: Optional[bool] = None,
                          italic: Optional[bool] = None,
                          alignment: Optional[PP_ALIGN] = None,
                          vertical_anchor: Optional[MSO_ANCHOR] = MSO_ANCHOR.TOP,
                          word_wrap: Optional[bool] = True,
                          auto_size: Optional[MSO_AUTO_SIZE] = None,
                          apply_to_all_paragraphs: bool = False) -> None:
        """
        Apply common styling to a text frame and its first paragraph (or all).

        Args:
            text_frame: The TextFrame object to style.
            font_size: Font size (e.g., Pt(12)).
            font_color: Font color hex string (e.g., "FF0000").
            bold: True for bold text.
            italic: True for italic text.
            alignment: Text alignment (e.g., PP_ALIGN.LEFT).
            vertical_anchor: Vertical alignment within the text frame.
            word_wrap: Enable or disable word wrap.
            auto_size: Text frame auto-sizing behavior.
            apply_to_all_paragraphs: If True, apply font styles to all paragraphs,
                                     otherwise only the first. Alignment and anchor
                                     always apply to the frame.
        """
        if vertical_anchor is not None:
            text_frame.vertical_anchor = vertical_anchor
        if word_wrap is not None:
            text_frame.word_wrap = word_wrap
        if auto_size is not None:
            text_frame.auto_size = auto_size

        paragraphs = text_frame.paragraphs
        if not paragraphs: # Add a paragraph if the frame is empty
             paragraphs.append(text_frame.add_paragraph())

        target_paragraphs = paragraphs if apply_to_all_paragraphs else [paragraphs[0]]

        for p in target_paragraphs:
            # Apply alignment to the paragraph itself
            if alignment is not None:
                p.alignment = alignment

            # Style the first run (or create one if needed)
            run = p.font # Get the default font for the paragraph
            if p.runs:
                run = p.runs[0].font # Prefer styling the first run if it exists

            if font_size is not None:
                run.size = font_size
            if font_color is not None:
                try:
                    run.color.rgb = RGBColor.from_string(font_color)
                except ValueError:
                     logger.warning(f"Invalid RGB color string: {font_color}. Skipping color.")
            if bold is not None:
                run.bold = bold
            if italic is not None:
                run.italic = italic

    def _get_color_from_score(self, score: float, invert: bool = False) -> str:
        """
        Determine a color based on a score (0-100).

        Args:
            score: The score (0-100).
            invert: If True, high scores are considered "bad" (e.g., risk).

        Returns:
            Hex color string from DesignElements.
        """
        if invert:
            if score < 40:
                return self.design.POSITIVE # Low risk is good
            elif score < 70:
                return self.design.WARNING
            else:
                return self.design.NEGATIVE # High risk is bad
        else:
            if score < 40:
                return self.design.NEGATIVE # Low score is bad
            elif score < 70:
                return self.design.WARNING
            else:
                return self.design.POSITIVE # High score is good

    def _add_risk_indicator(self, slide, left: Inches, top: Inches, width: Inches, height: Inches,
                            score: float, level: str, color_hex: str) -> None:
        """
        Adds a visual block indicating a risk score and level.

        Args:
            slide: The slide object to add the indicator to.
            left, top, width, height: Position and dimensions using Inches.
            score: The numerical risk score (0-100).
            level: The qualitative risk level (e.g., "High", "Medium", "Low").
            color_hex: The background color hex string for the indicator.
        """
        # Background shape
        rect = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        rect.fill.solid()
        try:
            rect.fill.fore_color.rgb = RGBColor.from_string(color_hex)
        except ValueError:
            logger.warning(f"Invalid risk color {color_hex}, using default.")
            rect.fill.fore_color.rgb = RGBColor.from_string(self.design.TEXT_DARK) # Fallback
        rect.line.fill.background() # No border

        # Text Box inside the shape
        text_box = slide.shapes.add_textbox(
            left + Inches(0.1), top + Inches(0.1), width - Inches(0.2), height - Inches(0.2)
        )
        tf = text_box.text_frame
        tf.clear() # Remove default paragraph
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

        # Add title "Overall Risk"
        p_title = tf.add_paragraph()
        p_title.text = "Overall Risk"
        p_title.font.size = Pt(16)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT) # White text on colored bg
        p_title.alignment = PP_ALIGN.CENTER

        # Add Score
        p_score = tf.add_paragraph()
        p_score.text = f"{score:.1f} / 100"
        p_score.font.size = Pt(28)
        p_score.font.bold = True
        p_score.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT)
        p_score.alignment = PP_ALIGN.CENTER

        # Add Level
        p_level = tf.add_paragraph()
        p_level.text = level
        p_level.font.size = Pt(18)
        p_level.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT)
        p_level.alignment = PP_ALIGN.CENTER

    def _add_gauge_chart(self, slide, left: Inches, top: Inches, width: Inches, height: Inches,
                         score: float, level: str, color_hex: str, title: str) -> None:
        """
        Adds a simple visual block representing a gauge or score.
        (Note: This is a simplified visual, not a true gauge chart).

        Args:
            slide: The slide object.
            left, top, width, height: Position and dimensions using Inches.
            score: The numerical score (0-100).
            level: Optional qualitative level (e.g., "High", "Adequate").
            color_hex: Background color hex string based on score severity.
            title: Title for the gauge (e.g., "Compliance Score").
        """
         # Background shape
        rect = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )
        rect.fill.solid()
        try:
            rect.fill.fore_color.rgb = RGBColor.from_string(color_hex)
        except ValueError:
            logger.warning(f"Invalid gauge color {color_hex}, using default.")
            rect.fill.fore_color.rgb = RGBColor.from_string(self.design.TEXT_DARK) # Fallback
        rect.line.fill.background() # No border

        # Text Box inside the shape
        text_box = slide.shapes.add_textbox(
            left + Inches(0.1), top + Inches(0.1), width - Inches(0.2), height - Inches(0.2)
        )
        tf = text_box.text_frame
        tf.clear()
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = False
        tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

        # Add title
        p_title = tf.add_paragraph()
        p_title.text = title
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT)
        p_title.alignment = PP_ALIGN.CENTER

        # Add Score
        p_score = tf.add_paragraph()
        p_score.text = f"{score:.1f}"
        p_score.font.size = Pt(24)
        p_score.font.bold = True
        p_score.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT)
        p_score.alignment = PP_ALIGN.CENTER

        # Add Level (optional)
        if level:
            p_level = tf.add_paragraph()
            p_level.text = level
            p_level.font.size = Pt(14)
            p_level.font.color.rgb = RGBColor.from_string(self.design.TEXT_LIGHT)
            p_level.alignment = PP_ALIGN.CENTER


    def _analyze_asset_classes(self, chart_data: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Simple analysis to group assets into broader classes.

        Args:
            chart_data: List of (asset_name, percentage) tuples.

        Returns:
            Dictionary of {asset_class: total_percentage}.
        """
        # Basic keyword-based classification (customize as needed)
        class_map = {
            "Equity": ["stock", "equity", "share"],
            "Fixed Income": ["bond", "fixed income", "debt"],
            "Crypto": ["bitcoin", "ethereum", "crypto", "token", "btc", "eth"],
            "Commodity": ["gold", "oil", "commodity"],
            "Real Estate": ["real estate", "property"],
            "Cash": ["cash", "money market"],
            "Alternative": ["hedge fund", "private equity", "venture capital", "alt"],
        }
        asset_classes = {"Other": 0.0}
        classified_total = 0.0

        for name, percentage in chart_data:
            found_class = False
            name_lower = name.lower()
            for class_name, keywords in class_map.items():
                if any(keyword in name_lower for keyword in keywords):
                    asset_classes[class_name] = asset_classes.get(class_name, 0.0) + percentage
                    found_class = True
                    break
            if not found_class:
                asset_classes["Other"] += percentage
            classified_total += percentage

        # Normalize percentages if needed (though they should sum to ~100)
        # Clean up zero entries and sort
        final_classes = {k: v for k, v in asset_classes.items() if v > 0.01} # Threshold for small values
        return dict(sorted(final_classes.items(), key=lambda item: item[1], reverse=True))

    # --------------------------------------------------------------------------
    # Public Slide Creation Methods
    # --------------------------------------------------------------------------

    def add_cover_slide(self, title: str, subtitle: str,
                      date: Optional[str] = None,
                      background_color: Optional[str] = None, # Hex string
                      logo_path: Optional[str] = None) -> None:
        """
        Add a professionally designed cover slide.

        Args:
            title: Main title text.
            subtitle: Subtitle text.
            date: Date to display (optional, bottom left).
            background_color: Background color hex code (optional).
            logo_path: Path to logo image (optional, bottom right).
        """
        slide_layout = self._get_slide_layout(0) # Typically Title slide layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Cover Slide: '{title}'")

        # Set background color if provided
        if background_color:
            try:
                background = slide.background
                fill = background.fill
                fill.solid()
                fill.fore_color.rgb = RGBColor.from_string(background_color)
            except ValueError:
                logger.warning(f"Invalid background color {background_color}. Using default.")
            except Exception as e:
                 logger.error(f"Error setting background color: {e}", exc_info=True)

        title_color = self.design.TEXT_LIGHT if background_color else self.design.TEXT_DARK

        # Set title (assuming placeholder exists)
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(
                title_shape.text_frame,
                font_size=Pt(44),
                font_color=title_color,
                bold=True,
                alignment=PP_ALIGN.CENTER # Usually centered on title slide
            )
        except AttributeError:
            logger.warning("Title placeholder not found on layout 0. Title not set.")

        # Set subtitle (assuming placeholder exists)
        try:
            subtitle_shape = slide.placeholders[1] # Often the subtitle placeholder
            subtitle_shape.text = subtitle
            self._style_text_frame(
                subtitle_shape.text_frame,
                font_size=Pt(28),
                font_color=title_color,
                alignment=PP_ALIGN.CENTER
            )
        except (AttributeError, IndexError):
             logger.warning("Subtitle placeholder not found or invalid index on layout 0. Subtitle not set.")


        # Add date if provided
        if date:
            date_left = Inches(0.5)
            date_top = self.slide_height - Inches(0.75)
            date_width = Inches(4)
            date_height = Inches(0.5)
            try:
                date_shape = slide.shapes.add_textbox(
                    date_left, date_top, date_width, date_height
                )
                date_shape.text = date
                self._style_text_frame(
                    date_shape.text_frame,
                    font_size=Pt(12),
                    font_color=title_color,
                    alignment=PP_ALIGN.LEFT
                )
            except Exception as e:
                logger.error(f"Error adding date to cover slide: {e}", exc_info=True)

        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            logo_width_max = Inches(2.0)
            logo_height_max = Inches(1.0)
            logo_right_margin = Inches(0.5)
            logo_bottom_margin = Inches(0.25)

            logo_left = self.slide_width - logo_width_max - logo_right_margin
            logo_top = self.slide_height - logo_height_max - logo_bottom_margin
            try:
                # Add picture, maintaining aspect ratio within bounds
                slide.shapes.add_picture(
                    logo_path, logo_left, logo_top, width=logo_width_max#, height=logo_height_max # Let pptx handle aspect ratio
                )
            except Exception as e:
                 logger.error(f"Error adding logo '{logo_path}' to cover slide: {e}", exc_info=True)
        elif logo_path:
            logger.warning(f"Logo path specified but not found: {logo_path}")

    def add_executive_summary_slide(self, title: str, fund_name: str, aum: str,
                                 strategy: str, risk_score: float, risk_level: str,
                                 risk_color: str, key_strengths: List[str],
                                 key_concerns: List[str]) -> None:
        """
        Add an executive summary slide with fund overview and key points.

        Args:
            title: Slide title.
            fund_name: Fund name.
            aum: Assets under management (formatted string).
            strategy: Fund strategy summary.
            risk_score: Overall risk score (0-100).
            risk_level: Risk level category.
            risk_color: Risk color hex code.
            key_strengths: List of key strengths (bullet points).
            key_concerns: List of key concerns (bullet points).
        """
        # Use a layout suitable for title and two content areas if possible, or blank
        # Let's try layout 2 (Section Header often works) or fallback to 6 (Blank)
        try:
            slide_layout = self._get_slide_layout(2) # Section Header layout
            slide = self.prs.slides.add_slide(slide_layout)
        except IndexError:
            slide_layout = self._get_slide_layout(6) # Fallback to Blank
            slide = self.prs.slides.add_slide(slide_layout)
            # Manually add title if using blank layout
            title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.8))
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)

        self.slide_count += 1
        logger.info(f"Added Executive Summary Slide: '{title}'")

        # Set title if using a layout with a title placeholder
        if slide_layout != self.prs.slide_layouts[6]: # If not blank layout
            try:
                title_shape = slide.shapes.title
                title_shape.text = title
                self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
            except AttributeError:
                 logger.warning("Title placeholder not found on executive summary slide layout. Title may be missing.")


        # Define content areas
        col1_left = Inches(0.5)
        col2_left = Inches(6.8)
        col_width = Inches(6.0)
        row1_top = Inches(1.5)
        row2_top = Inches(4.0)
        box_height = Inches(2.2)

        # Add overview section (Top Left)
        try:
            overview_box = slide.shapes.add_textbox(col1_left, row1_top, col_width, box_height)
            tf = overview_box.text_frame
            tf.clear()
            tf.word_wrap = True

            p = tf.add_paragraph()
            p.text = "Fund Overview"
            p.font.bold = True
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            p = tf.add_paragraph()
            p.text = f"Fund Name: {fund_name}"
            p.font.size = Pt(14)
            p.space_after = Pt(3)

            p = tf.add_paragraph()
            p.text = f"AUM: {aum}"
            p.font.size = Pt(14)
            p.space_after = Pt(3)

            p = tf.add_paragraph()
            p.text = f"Strategy: {strategy}"
            p.font.size = Pt(14)
        except Exception as e:
            logger.error(f"Error adding overview section: {e}", exc_info=True)

        # Add risk score indicator (Bottom Left)
        try:
            self._add_risk_indicator(
                slide, col1_left, row2_top, col_width, box_height, # Reuse height for alignment
                risk_score, risk_level, risk_color
            )
        except Exception as e:
            logger.error(f"Error adding risk indicator: {e}", exc_info=True)

        # Add key strengths (Top Right)
        try:
            strengths_box = slide.shapes.add_textbox(col2_left, row1_top, col_width, box_height)
            tf = strengths_box.text_frame
            tf.clear()
            tf.word_wrap = True

            p = tf.add_paragraph()
            p.text = "Key Strengths"
            p.font.bold = True
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor.from_string(self.design.POSITIVE)
            p.space_after = Pt(6)

            for strength in key_strengths:
                p = tf.add_paragraph()
                p.text = f"✓ {strength}" # Use checkmark symbol
                p.font.size = Pt(14)
                p.level = 0 # Use level 0 for primary bullets
                p.space_after = Pt(3)
            tf.margin_left = Inches(0.1) # Add slight indent for bullets
        except Exception as e:
            logger.error(f"Error adding strengths section: {e}", exc_info=True)


        # Add key concerns (Bottom Right)
        try:
            concerns_box = slide.shapes.add_textbox(col2_left, row2_top, col_width, box_height)
            tf = concerns_box.text_frame
            tf.clear()
            tf.word_wrap = True

            p = tf.add_paragraph()
            p.text = "Key Concerns"
            p.font.bold = True
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor.from_string(self.design.NEGATIVE)
            p.space_after = Pt(6)

            for concern in key_concerns:
                p = tf.add_paragraph()
                p.text = f"• {concern}" # Use standard bullet
                # Alternative: p.text = f"! {concern}" # Use exclamation mark symbol
                p.font.size = Pt(14)
                p.level = 0
                p.space_after = Pt(3)
            tf.margin_left = Inches(0.1)
        except Exception as e:
            logger.error(f"Error adding concerns section: {e}", exc_info=True)


    def add_fund_overview_slide(self, title: str, fund_data: List[List[str]],
                              strategy_description: Optional[str] = None) -> None:
        """
        Add a fund overview slide with fund details in a table and strategy text.

        Args:
            title: Slide title.
            fund_data: Table data as a list of rows (e.g., [["Category", "Value"], ...]).
                       Assumes first row is the header.
            strategy_description: Detailed strategy description (optional, placed to the right).
        """
        slide_layout = self._get_slide_layout(5) # Title and Content or similar
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Fund Overview Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on fund overview slide layout.")

        # Determine layout: Table only or Table + Description
        has_description = bool(strategy_description)
        table_width = Inches(6.5) if has_description else Inches(12.33)
        table_left = Inches(0.5)

        # Add fund information table
        try:
            if not fund_data or not fund_data[0]:
                 logger.warning("Fund data is empty. Skipping table.")
                 return # Or add placeholder text

            rows = len(fund_data)
            cols = len(fund_data[0])
            table_top = Inches(1.5)
            # Estimate height, but allow pptx to adjust
            table_est_height = Inches(rows * 0.4 + 0.2)

            shape = slide.shapes.add_table(
                rows, cols, table_left, table_top, table_width, table_est_height
            )
            table = shape.table

            # Set column widths (example: 40% / 60% for 2 columns)
            if cols == 2:
                table.columns[0].width = int(table_width * 0.4)
                table.columns[1].width = int(table_width * 0.6)
            else: # Distribute evenly for other column counts
                 col_w = int(table_width / cols)
                 for i in range(cols):
                     table.columns[i].width = col_w


            # Fill table with data and style
            for i, row_data in enumerate(fund_data):
                for j, cell_value in enumerate(row_data):
                    cell = table.cell(i, j)
                    cell.text = str(cell_value)
                    tf = cell.text_frame
                    tf.margin_left = Inches(0.08)
                    tf.margin_right = Inches(0.08)
                    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

                    is_header = (i == 0)
                    is_alt_row = (i > 0 and i % 2 == 1)
                    is_first_col = (j == 0 and i > 0) # Bold non-header first column

                    # Default style
                    self._style_text_frame(
                        tf,
                        font_size=Pt(12),
                        bold=is_first_col,
                        font_color=self.design.TEXT_DARK
                    )

                    # Header style
                    if is_header:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_HEADER)
                        self._style_text_frame(
                            tf,
                            font_size=Pt(12),
                            bold=True,
                            font_color=self.design.TEXT_LIGHT
                        )
                    # Alternate row style
                    elif is_alt_row:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_ALT_ROW)

        except Exception as e:
             logger.error(f"Error adding fund data table: {e}", exc_info=True)


        # Add strategy description if provided
        if has_description:
            strategy_left = table_left + table_width + Inches(0.3)
            strategy_top = Inches(1.5)
            strategy_width = self.slide_width - strategy_left - Inches(0.5)
            strategy_height = self.slide_height - strategy_top - Inches(1.0) # Leave footer space

            try:
                strategy_box = slide.shapes.add_textbox(
                    strategy_left, strategy_top, strategy_width, strategy_height
                )
                tf = strategy_box.text_frame
                tf.clear()
                tf.word_wrap = True
                tf.vertical_anchor = MSO_ANCHOR.TOP
                tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT # Let box grow vertically

                p = tf.add_paragraph()
                p.text = "Investment Strategy"
                p.font.bold = True
                p.font.size = Pt(18)
                p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
                p.space_after = Pt(6)

                # Add description content (handle newlines)
                lines = strategy_description.split('\n')
                for line in lines:
                    p = tf.add_paragraph()
                    p.text = line
                    p.font.size = Pt(12)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.space_after = Pt(4)
                    p.alignment = PP_ALIGN.LEFT

            except Exception as e:
                 logger.error(f"Error adding strategy description: {e}", exc_info=True)

    def add_team_analysis_slide(self, title: str, team_profiles: List[Dict[str, str]]) -> None:
        """
        Add a team analysis slide with team member profiles in styled boxes.

        Arranges profiles dynamically (up to 9) in rows.

        Args:
            title: Slide title.
            team_profiles: List of team member profiles. Each dict should have
                           'name', 'title', and optionally 'background'.
        """
        slide_layout = self._get_slide_layout(6) # Blank layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Team Analysis Slide: '{title}'")

        # Add title manually for blank layout
        title_left = Inches(0.5)
        title_top = Inches(0.3)
        title_width = self.slide_width - Inches(1.0)
        title_height = Inches(0.8)

        try:
            title_shape = slide.shapes.add_textbox(
                title_left, title_top, title_width, title_height
            )
            title_shape.text = title
            self._style_text_frame(
                title_shape.text_frame,
                font_size=Pt(36), # Slightly smaller title for content-heavy slide
                font_color=self.design.TEXT_DARK,
                bold=True,
                alignment=PP_ALIGN.LEFT
            )
        except Exception as e:
            logger.error(f"Error adding title to team slide: {e}", exc_info=True)


        # Calculate layout based on number of profiles
        num_profiles = len(team_profiles)
        if num_profiles == 0:
            logger.warning("No team profiles provided for team analysis slide.")
            # Add a placeholder text?
            no_data_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(10), Inches(1))
            no_data_box.text = "No team information available."
            self._style_text_frame(no_data_box.text_frame, font_size=Pt(16), italic=True, alignment=PP_ALIGN.CENTER)
            return

        max_profiles_to_show = 9
        if num_profiles > max_profiles_to_show:
            logger.warning(f"Too many team profiles ({num_profiles}), showing first {max_profiles_to_show}.")
            team_profiles = team_profiles[:max_profiles_to_show]
            num_profiles = max_profiles_to_show

        # Determine grid layout (prefer 3 columns)
        profiles_per_row = 3
        num_rows = math.ceil(num_profiles / profiles_per_row)

        # Calculate dimensions for profile boxes
        total_width_available = self.slide_width - Inches(1.0) # Margins L/R
        total_height_available = self.slide_height - Inches(1.5) - Inches(0.5) # Title area and bottom margin
        h_margin = Inches(0.3) # Horizontal margin between boxes
        v_margin = Inches(0.3) # Vertical margin between boxes

        box_width = (total_width_available - h_margin * (profiles_per_row - 1)) / profiles_per_row
        # Adjust height based on rows, ensure reasonable max height
        box_height = min(Inches(2.8), (total_height_available - v_margin * (num_rows - 1)) / num_rows)

        # Calculate starting position (center the grid)
        grid_width = box_width * profiles_per_row + h_margin * (profiles_per_row - 1)
        grid_height = box_height * num_rows + v_margin * (num_rows - 1)
        start_left = (self.slide_width - grid_width) / 2
        start_top = Inches(1.3) # Below title


        # Create profile boxes
        profile_index = 0
        for row in range(num_rows):
            for col in range(profiles_per_row):
                if profile_index >= len(team_profiles):
                    break

                profile = team_profiles[profile_index]
                profile_index += 1

                # Calculate position for this box
                left = start_left + col * (box_width + h_margin)
                top = start_top + row * (box_height + v_margin)

                try:
                    # Add profile rectangle (subtle background, thin border)
                    profile_rect = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE, left, top, box_width, box_height
                    )
                    profile_rect.fill.solid()
                    profile_rect.fill.fore_color.rgb = RGBColor.from_string(self.design.BOX_BACKGROUND)
                    profile_rect.line.color.rgb = RGBColor.from_string(self.design.BOX_BORDER)
                    profile_rect.line.width = Pt(0.75)

                    # Add text box slightly inset for profile content
                    text_margin = Inches(0.15)
                    text_left = left + text_margin
                    text_top = top + text_margin
                    text_width = box_width - 2 * text_margin
                    text_height = box_height - 2 * text_margin

                    text_box = slide.shapes.add_textbox(
                        text_left, text_top, text_width, text_height
                    )
                    tf = text_box.text_frame
                    tf.clear()
                    tf.word_wrap = True
                    tf.vertical_anchor = MSO_ANCHOR.TOP
                    # Optional: tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE (if content overflows)

                    # Add name
                    p = tf.add_paragraph()
                    p.text = profile.get("name", "N/A")
                    p.font.bold = True
                    p.font.size = Pt(16)
                    p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
                    p.space_after = Pt(2)

                    # Add title
                    p = tf.add_paragraph()
                    p.text = profile.get("title", "N/A")
                    p.font.italic = True
                    p.font.size = Pt(11)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.space_after = Pt(6)

                    # Add background points (split by newline if present)
                    background_text = profile.get("background", "")
                    if background_text:
                        lines = background_text.split('\n')
                        for line in lines:
                             if line.strip(): # Avoid adding empty paragraphs
                                p = tf.add_paragraph()
                                p.text = f"• {line.strip()}" # Add bullet point
                                p.font.size = Pt(10)
                                p.font.color.rgb = RGBColor.from_string(self.design.TEXT_MUTED) # Softer color
                                p.level = 0 # Ensure bullets align
                                p.space_after = Pt(2)
                        tf.margin_left = Inches(0.1) # Indent bullet points slightly

                except Exception as e:
                    logger.error(f"Error adding profile box for {profile.get('name', 'Unknown')}: {e}", exc_info=True)


    def add_portfolio_allocation_slide(self, title: str, chart_data: List[Tuple[str, float]]) -> None:
        """
        Add a portfolio allocation slide with a pie chart and key insights.

        Args:
            title: Slide title.
            chart_data: List of (asset_name, percentage) tuples for the pie chart.
                        Percentages should sum to ~100.
        """
        slide_layout = self._get_slide_layout(5) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Portfolio Allocation Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on allocation slide layout.")

        # Define layout areas
        chart_left = Inches(0.5)
        chart_top = Inches(1.5)
        chart_width = Inches(6.5) # Allow space for legend/labels
        chart_height = Inches(5.5)

        insights_left = chart_left + chart_width + Inches(0.3)
        insights_top = Inches(1.5)
        insights_width = self.slide_width - insights_left - Inches(0.5) # Right margin
        insights_height = chart_height # Align height with chart

        # Add pie chart
        if not chart_data:
             logger.warning("No chart data provided for allocation slide. Skipping chart.")
             # Add placeholder text maybe?
        else:
            try:
                chart = self.chart_factory.create_pie_chart(
                    slide, chart_left, chart_top, chart_width, chart_height,
                    chart_data,
                    chart_title="Asset Allocation",
                    show_legend=True,
                    show_data_labels=True # Show percentages on slices
                )
            except Exception as e:
                 logger.error(f"Error creating pie chart: {e}", exc_info=True)


        # Add text box with key insights
        try:
            insights_box = slide.shapes.add_textbox(
                insights_left, insights_top, insights_width, insights_height
            )
            tf = insights_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Portfolio Insights"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(8)

            if not chart_data:
                p = tf.add_paragraph()
                p.text = "Allocation data not available."
                p.font.size = Pt(12)
                p.font.italic = True
                return # Don't try to analyze empty data

            # --- Analysis ---
            # Sort data by allocation (descending)
            sorted_data = sorted(chart_data, key=lambda x: x[1], reverse=True)

            # Get top asset
            if sorted_data:
                top_asset, top_allocation = sorted_data[0]
                p = tf.add_paragraph()
                p.text = f"• Largest Holding: {top_asset} ({top_allocation:.1f}%)"
                p.font.size = Pt(12)
                p.space_after = Pt(4)

            # Calculate concentration
            if len(sorted_data) >= 3:
                top_3_allocation = sum(alloc for _, alloc in sorted_data[:3])
                p = tf.add_paragraph()
                p.text = f"• Top 3 Holdings: {top_3_allocation:.1f}% of portfolio"
                p.font.size = Pt(12)
                p.space_after = Pt(4)
            elif len(sorted_data) > 0:
                 top_n_allocation = sum(alloc for _, alloc in sorted_data)
                 p = tf.add_paragraph()
                 p.text = f"• Total Holdings: {top_n_allocation:.1f}% ({len(sorted_data)} assets)"
                 p.font.size = Pt(12)
                 p.space_after = Pt(4)


            # Add diversification comment
            p = tf.add_paragraph()
            div_comment = "• Portfolio appears well-diversified."
            if sorted_data:
                if top_allocation > 50:
                    div_comment = "• High concentration risk in the largest holding."
                elif top_allocation > 30:
                    div_comment = "• Moderate concentration in the largest holding."
                elif len(sorted_data) < 5:
                    div_comment = "• Limited number of assets, check diversification."
            p.text = div_comment
            p.font.size = Pt(12)
            p.space_after = Pt(8) # More space before next section


            # Add asset class commentary
            asset_classes = self._analyze_asset_classes(chart_data)
            if asset_classes:
                p = tf.add_paragraph()
                p.text = "Asset Class Exposure:"
                p.font.size = Pt(14)
                p.font.bold = True
                p.space_after = Pt(4)

                for asset_class, percentage in asset_classes.items():
                    p = tf.add_paragraph()
                    p.text = f"  - {asset_class}: {percentage:.1f}%" # Indent slightly
                    p.font.size = Pt(12)
                    p.space_after = Pt(2)
            # --- End Analysis ---

        except Exception as e:
            logger.error(f"Error adding portfolio insights: {e}", exc_info=True)

    def add_market_analysis_slide(self, title: str, market_data: List[List[str]]) -> None:
        """
        Add a market analysis slide with market data in a table and insights.

        Args:
            title: Slide title.
            market_data: Table data as a list of rows (e.g., [["Asset", "Price", "Change %", "Vol %"], ...]).
                         Assumes first row is the header.
        """
        slide_layout = self._get_slide_layout(1) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Market Analysis Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
             logger.warning("Title placeholder not found on market analysis slide layout.")


        # Add market data table
        table_left = Inches(0.5)
        table_top = Inches(1.5)
        table_width = Inches(12.33) # Use full width
        table_height_estimate = Inches(0.4) # Base height per row
        insights_height = Inches(1.5) # Space for insights below table

        try:
            if not market_data or not market_data[0]:
                logger.warning("Market data is empty. Skipping table.")
                # Add placeholder text
                no_data_box = slide.shapes.add_textbox(table_left, table_top, table_width, Inches(1))
                no_data_box.text = "Market data not available."
                self._style_text_frame(no_data_box.text_frame, font_size=Pt(16), italic=True, alignment=PP_ALIGN.CENTER)
                return

            rows = len(market_data)
            cols = len(market_data[0])
            table_actual_height = Inches(rows * 0.4 + 0.2) # Header + rows

            shape = slide.shapes.add_table(
                rows, cols, table_left, table_top, table_width, table_actual_height
            )
            table = shape.table

            # Set column widths - distribute evenly as default
            col_width = int(table_width / cols)
            for i in range(cols):
                table.columns[i].width = col_width
            # Optional: Customize widths based on expected content if needed
            # e.g., if cols >= 4: table.columns[0].width = int(table_width * 0.3) ...

            # Fill table with data and style
            for i, row_data in enumerate(market_data):
                for j, cell_value_raw in enumerate(row_data):
                    cell = table.cell(i, j)
                    cell_value = str(cell_value_raw) # Ensure string
                    cell.text = cell_value
                    tf = cell.text_frame
                    tf.margin_left = Inches(0.08)
                    tf.margin_right = Inches(0.08)
                    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

                    is_header = (i == 0)
                    is_alt_row = (i > 0 and i % 2 == 1)

                    # Default style
                    self._style_text_frame(tf, font_size=Pt(11), font_color=self.design.TEXT_DARK)

                    # Header style
                    if is_header:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_HEADER)
                        self._style_text_frame(tf, font_size=Pt(11), bold=True, font_color=self.design.TEXT_LIGHT)
                    # Alternate row style
                    elif is_alt_row:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_ALT_ROW)

                    # --- Add specific column styling (Example) ---
                    # Assuming columns 2 & 3 (index 1 & 2) might contain % changes
                    if not is_header and j in [1, 2]:
                         # Check if value contains '+' or '-' and is likely numeric change
                         is_change_value = False
                         try:
                             # Attempt to strip common symbols and check if numeric
                             cleaned_value = cell_value.replace('%','').replace('+','').replace('-','').strip()
                             if cleaned_value: # Avoid errors on empty strings
                                 float(cleaned_value)
                                 is_change_value = True
                         except ValueError:
                             pass # Not a numeric change value

                         if is_change_value:
                             if "+" in cell_value or (cell_value.strip() and '-' not in cell_value and float(cleaned_value) > 0):
                                 self._style_text_frame(tf, font_color=self.design.POSITIVE, bold=True)
                             elif "-" in cell_value:
                                 self._style_text_frame(tf, font_color=self.design.NEGATIVE, bold=True)
                    # --- End specific column styling ---

        except Exception as e:
             logger.error(f"Error adding market data table: {e}", exc_info=True)


        # Add market insights text box below table
        insights_left = table_left
        insights_top = table_top + table_actual_height + Inches(0.2) # Position below table
        insights_width = table_width

        try:
            insights_box = slide.shapes.add_textbox(
                insights_left, insights_top, insights_width, insights_height
            )
            tf = insights_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP

            p = tf.add_paragraph()
            p.text = "Market Commentary"
            p.font.bold = True
            p.font.size = Pt(16)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(4)

            p = tf.add_paragraph()
            p.text = "This table summarizes recent market performance for relevant assets or benchmarks. Key metrics include current pricing, short-term percentage changes, and volatility indicators, providing context for the fund's operating environment."
            p.font.size = Pt(11) # Smaller font for commentary
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_MUTED)
        except Exception as e:
             logger.error(f"Error adding market insights: {e}", exc_info=True)


    def add_wallet_overview_slide(self, title: str, total_balance: str, wallet_count: int,
                                avg_risk_score: float, wallet_chart_data: List[Tuple[str, float]]) -> None:
        """
        Add a wallet overview slide with key metrics and wallet distribution chart.

        Args:
            title: Slide title.
            total_balance: Total wallet balance (formatted string, e.g., "$10.5M USD").
            wallet_count: Number of wallets analyzed.
            avg_risk_score: Average risk score across wallets (0-100).
            wallet_chart_data: List of (wallet_type/category, balance/value) tuples for pie chart.
        """
        slide_layout = self._get_slide_layout(5) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Wallet Overview Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on wallet overview slide layout.")

        # Define layout areas
        col1_left = Inches(0.5)
        col2_left = Inches(5.0) # Start chart/second column further right
        col_width1 = Inches(4.0)
        col_width2 = Inches(8.0) # Wider chart/second area
        row1_top = Inches(1.5)
        row2_top = Inches(4.0) # Position for second element in left column
        box_height = Inches(2.2) # Height for metrics/security box
        chart_height = Inches(5.0) # Height for pie chart

        # Add key metrics (Top Left)
        try:
            metrics_box = slide.shapes.add_textbox(col1_left, row1_top, col_width1, box_height)
            tf = metrics_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP

            p = tf.add_paragraph()
            p.text = "Wallet Infrastructure"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            p = tf.add_paragraph()
            p.text = f"Total Balance: {total_balance}"
            p.font.size = Pt(14)
            p.font.bold = True # Make balance prominent
            p.space_after = Pt(4)

            p = tf.add_paragraph()
            p.text = f"Number of Wallets: {wallet_count}"
            p.font.size = Pt(14)
            p.space_after = Pt(4)

            p = tf.add_paragraph()
            p.text = f"Avg. Wallet Risk Score: {avg_risk_score:.1f}/100"
            p.font.size = Pt(14)
            risk_color_hex = self._get_color_from_score(avg_risk_score, invert=True) # Risk score: high is bad
            p.font.color.rgb = RGBColor.from_string(risk_color_hex)
            p.font.bold = True # Highlight risk score

        except Exception as e:
            logger.error(f"Error adding wallet metrics: {e}", exc_info=True)

        # Add wallet distribution chart (Right side)
        if not wallet_chart_data:
             logger.warning("No wallet chart data provided. Skipping chart.")
             # Add placeholder text maybe?
             no_chart_box = slide.shapes.add_textbox(col2_left, row1_top, col_width2, chart_height)
             no_chart_box.text = "Wallet distribution data not available."
             self._style_text_frame(no_chart_box.text_frame, font_size=Pt(14), italic=True, alignment=PP_ALIGN.CENTER, vertical_anchor=MSO_ANCHOR.MIDDLE)
        else:
            try:
                chart = self.chart_factory.create_pie_chart(
                    slide, col2_left, row1_top, col_width2, chart_height,
                    wallet_chart_data,
                    chart_title="Wallet Balance Distribution",
                    show_legend=True,
                    show_data_labels=True
                )
            except Exception as e:
                logger.error(f"Error creating wallet distribution chart: {e}", exc_info=True)

        # Add security assessment (Bottom Left)
        security_top = row1_top + box_height + Inches(0.3) # Position below metrics
        security_height = self.slide_height - security_top - Inches(0.5) # Fill remaining height

        try:
            security_box = slide.shapes.add_textbox(col1_left, security_top, col_width1, security_height)
            tf = security_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Security Assessment"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            # Determine security level based on risk score
            if avg_risk_score < 30:
                security_level = "High"
                security_color_hex = self.design.POSITIVE
                security_assessment = "Wallet infrastructure appears robust with low average risk. Likely indicates strong security practices (e.g., MPC, cold storage emphasis)."
            elif avg_risk_score < 60:
                security_level = "Medium"
                security_color_hex = self.design.WARNING
                security_assessment = "Moderate average wallet risk. Security practices may be adequate but warrant review, especially for high-value wallets. Potential mix of hot/cold or less secure types."
            else:
                security_level = "Low"
                security_color_hex = self.design.NEGATIVE
                security_assessment = "High average wallet risk suggests potential vulnerabilities. Further investigation into wallet types, key management, and security protocols is strongly recommended."

            p = tf.add_paragraph()
            p.text = f"Overall Security Posture: {security_level}"
            p.font.bold = True
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor.from_string(security_color_hex)
            p.space_after = Pt(4)

            p = tf.add_paragraph()
            p.text = security_assessment
            p.font.size = Pt(12)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_MUTED)
            p.space_after = Pt(4)

        except Exception as e:
            logger.error(f"Error adding security assessment: {e}", exc_info=True)


    def add_wallet_security_slide(self, title: str, wallet_data: List[List[str]],
                                security_score: float, wallet_diversification: Dict[str, Any]) -> None:
        """
        Add a wallet security analysis slide with detailed wallet table and metrics.

        Args:
            title: Slide title.
            wallet_data: Table data for wallets (e.g., [["Address", "Type", "Balance", "Risk", "Features"], ...]).
                         Assumes first row is header.
            security_score: Overall wallet security score (0-100, higher is better).
            wallet_diversification: Dict containing diversification metrics like:
                                    'diversification_score' (0-100),
                                    'concentration_risk' ("Low", "Medium", "High"),
                                    'largest_wallet_pct' (float %).
        """
        slide_layout = self._get_slide_layout(1) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Wallet Security Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on wallet security slide layout.")


        # Define layout areas
        table_top = Inches(1.5)
        table_width = Inches(12.33)
        metrics_top_margin = Inches(0.3) # Space between table and metrics
        metrics_height = Inches(2.0)

        # Add wallet data table
        table_actual_height = Inches(0.2) # Min height if no data
        try:
            if not wallet_data or not wallet_data[0]:
                 logger.warning("Wallet data is empty. Skipping table.")
                 # Add placeholder text
                 no_data_box = slide.shapes.add_textbox(Inches(0.5), table_top, table_width, Inches(1))
                 no_data_box.text = "Detailed wallet data not available."
                 self._style_text_frame(no_data_box.text_frame, font_size=Pt(16), italic=True, alignment=PP_ALIGN.CENTER)
                 metrics_top = table_top + Inches(1) + metrics_top_margin # Adjust metrics position
            else:
                rows = len(wallet_data)
                cols = len(wallet_data[0])
                table_actual_height = Inches(rows * 0.4 + 0.2) # Estimate height
                max_table_height = self.slide_height - table_top - metrics_height - Inches(1.0) # Max height before hitting bottom/metrics
                table_actual_height = min(table_actual_height, max_table_height) # Prevent overflow

                shape = slide.shapes.add_table(
                    rows, cols, Inches(0.5), table_top, table_width, table_actual_height
                )
                table = shape.table

                # Set column widths (example, adjust as needed)
                if cols >= 5:
                    table.columns[0].width = int(table_width * 0.25) # Address
                    table.columns[1].width = int(table_width * 0.15) # Type
                    table.columns[2].width = int(table_width * 0.15) # Balance
                    table.columns[3].width = int(table_width * 0.10) # Risk
                    table.columns[4].width = int(table_width * 0.35) # Features
                else: # Fallback to even distribution
                    col_w = int(table_width / cols)
                    for i in range(cols): table.columns[i].width = col_w

                # Fill table with data and style
                risk_col_index = 3 # Assuming 'Risk' is the 4th column (index 3)

                for i, row_data in enumerate(wallet_data):
                    for j, cell_value_raw in enumerate(row_data):
                        cell = table.cell(i, j)
                        cell_value = str(cell_value_raw)
                        cell.text = cell_value
                        tf = cell.text_frame
                        tf.margin_left = Inches(0.08)
                        tf.margin_right = Inches(0.08)
                        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

                        is_header = (i == 0)
                        is_alt_row = (i > 0 and i % 2 == 1)

                        # Default style
                        self._style_text_frame(tf, font_size=Pt(10), font_color=self.design.TEXT_DARK) # Smaller font for dense table

                        # Header style
                        if is_header:
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_HEADER)
                            self._style_text_frame(tf, font_size=Pt(10), bold=True, font_color=self.design.TEXT_LIGHT)
                        # Alternate row style
                        elif is_alt_row:
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_ALT_ROW)

                        # Add coloring for risk level column
                        if not is_header and j == risk_col_index:
                            risk_text = cell_value.lower()
                            if "low" in risk_text:
                                self._style_text_frame(tf, font_color=self.design.POSITIVE, bold=True)
                            elif "medium" in risk_text or "moderate" in risk_text:
                                self._style_text_frame(tf, font_color=self.design.WARNING, bold=True)
                            elif "high" in risk_text or "critical" in risk_text:
                                self._style_text_frame(tf, font_color=self.design.NEGATIVE, bold=True)

                metrics_top = table_top + table_actual_height + metrics_top_margin # Set position below actual table

        except Exception as e:
             logger.error(f"Error adding wallet data table: {e}", exc_info=True)
             metrics_top = table_top + Inches(1) + metrics_top_margin # Fallback position

        # Add Security Score Gauge (Bottom Left)
        gauge_width = Inches(4)
        gauge_left = Inches(0.5)
        try:
            score_color_hex = self._get_color_from_score(security_score) # High score is good
            self._add_gauge_chart(
                slide, gauge_left, metrics_top, gauge_width, metrics_height,
                security_score, "", score_color_hex, # No level text needed
                "Overall Wallet Security Score"
            )
        except Exception as e:
             logger.error(f"Error adding security score gauge: {e}", exc_info=True)


        # Add Diversification Metrics (Bottom Right)
        div_left = gauge_left + gauge_width + Inches(0.5)
        div_width = self.slide_width - div_left - Inches(0.5) # Fill remaining width
        try:
            div_box = slide.shapes.add_textbox(
                div_left, metrics_top, div_width, metrics_height
            )
            tf = div_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT # Allow growth if needed

            p = tf.add_paragraph()
            p.text = "Wallet Diversification & Concentration"
            p.font.bold = True
            p.font.size = Pt(16)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            if wallet_diversification:
                div_score = wallet_diversification.get("diversification_score", 0.0)
                concentration_risk = wallet_diversification.get("concentration_risk", "N/A")
                largest_wallet_pct = wallet_diversification.get("largest_wallet_pct", 0.0)

                # Diversification Score
                p = tf.add_paragraph()
                p.text = f"Diversification Score: {div_score:.1f} / 100"
                p.font.size = Pt(12)
                div_score_color = self._get_color_from_score(div_score) # High score is good
                p.font.color.rgb = RGBColor.from_string(div_score_color)
                p.font.bold = True
                p.space_after = Pt(4)

                # Concentration Risk Level
                p = tf.add_paragraph()
                p.text = f"Concentration Risk: {concentration_risk}"
                p.font.size = Pt(12)
                conc_risk_lower = concentration_risk.lower()
                if "low" in conc_risk_lower: conc_color = self.design.POSITIVE
                elif "medium" in conc_risk_lower: conc_color = self.design.WARNING
                elif "high" in conc_risk_lower: conc_color = self.design.NEGATIVE
                else: conc_color = self.design.TEXT_DARK # Default
                p.font.color.rgb = RGBColor.from_string(conc_color)
                p.font.bold = True
                p.space_after = Pt(4)

                # Largest Wallet %
                p = tf.add_paragraph()
                p.text = f"Largest Single Wallet: {largest_wallet_pct:.1f}% of Total Funds"
                p.font.size = Pt(12)
                p.space_after = Pt(4)

            else:
                p = tf.add_paragraph()
                p.text = "Wallet diversification metrics not available."
                p.font.size = Pt(12)
                p.font.italic = True

        except Exception as e:
             logger.error(f"Error adding diversification metrics: {e}", exc_info=True)

    def add_risk_overview_slide(self, title: str, overall_risk_score: float, risk_level: str,
                              risk_color: str, radar_labels: List[str], radar_values: List[float]) -> None:
        """
        Add a risk overview slide with overall score, radar chart, and summary.

        Args:
            title: Slide title.
            overall_risk_score: Overall risk score (0-100, higher is riskier).
            risk_level: Risk level category (e.g., "Medium").
            risk_color: Hex color corresponding to the risk level.
            radar_labels: Labels for radar chart categories (e.g., ["Market", "Credit", ...]).
            radar_values: Values for radar chart categories (scaled, e.g., 0-10 or 0-100).
                          ChartFactory needs to handle the scale.
        """
        slide_layout = self._get_slide_layout(5) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Risk Overview Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on risk overview slide layout.")


        # Define layout areas
        indicator_left = Inches(0.5)
        indicator_width = Inches(4.0)
        indicator_height = Inches(2.0)
        indicator_top = Inches(1.5)

        chart_left = indicator_left + indicator_width + Inches(0.5)
        chart_top = Inches(1.5)
        chart_width = self.slide_width - chart_left - Inches(0.5) # Fill remaining width
        chart_height = Inches(5.5) # Allow space below

        summary_left = indicator_left
        summary_top = indicator_top + indicator_height + Inches(0.3) # Below indicator
        summary_width = indicator_width
        summary_height = self.slide_height - summary_top - Inches(0.5) # Fill height

        # Add overall risk score indicator (Top Left)
        try:
            self._add_risk_indicator(
                slide, indicator_left, indicator_top, indicator_width, indicator_height,
                overall_risk_score, risk_level, risk_color
            )
        except Exception as e:
             logger.error(f"Error adding risk indicator: {e}", exc_info=True)


        # Add radar chart (Right side)
        if not radar_labels or not radar_values or len(radar_labels) != len(radar_values):
             logger.warning("Invalid radar chart data provided. Skipping chart.")
             # Add placeholder text maybe?
             no_chart_box = slide.shapes.add_textbox(chart_left, chart_top, chart_width, chart_height)
             no_chart_box.text = "Risk factor breakdown data not available."
             self._style_text_frame(no_chart_box.text_frame, font_size=Pt(14), italic=True, alignment=PP_ALIGN.CENTER, vertical_anchor=MSO_ANCHOR.MIDDLE)
        else:
            try:
                self.chart_factory.create_radar_chart(
                    slide, chart_left, chart_top, chart_width, chart_height,
                    radar_labels, radar_values,
                    chart_title="Risk Factor Exposure"
                )
            except Exception as e:
                 logger.error(f"Error creating radar chart: {e}", exc_info=True)


        # Add risk summary (Bottom Left)
        try:
            summary_box = slide.shapes.add_textbox(
                summary_left, summary_top, summary_width, summary_height
            )
            tf = summary_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Risk Assessment Summary"
            p.font.bold = True
            p.font.size = Pt(16)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            # Generate risk summary text based on risk level/score
            # Customize these summaries based on your risk categories
            risk_level_lower = risk_level.lower()
            summary = "Overall risk profile analysis indicates potential areas of concern. Refer to specific risk factor details." # Default
            if "low" in risk_level_lower:
                summary = "The fund exhibits a generally low risk profile across key areas. Risk management appears effective, with limited exposure to major negative factors identified during this assessment."
            elif "medium" in risk_level_lower or "moderate" in risk_level_lower:
                summary = "A moderate overall risk level is observed. While some risk controls seem adequate, specific areas (highlighted in the chart) show elevated exposure requiring monitoring or mitigation."
            elif "high" in risk_level_lower or "critical" in risk_level_lower:
                summary = "Significant risk factors contribute to a high overall risk assessment. Multiple areas show considerable exposure, indicating potential vulnerabilities that warrant immediate attention and strategic review."

            p = tf.add_paragraph()
            p.text = summary
            p.font.size = Pt(12)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
            p.space_after = Pt(4)

        except Exception as e:
             logger.error(f"Error adding risk summary: {e}", exc_info=True)

    def add_risk_factors_slide(self, title: str, risk_factors: List[str]) -> None:
        """
        Add a slide detailing specific risk factors as bullet points.

        Attempts a two-column layout if there are many factors.

        Args:
            title: Slide title.
            risk_factors: List of risk factor strings (each becomes a bullet point).
        """
        slide_layout = self._get_slide_layout(1) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Risk Factors Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on risk factors slide layout.")

        # Content area
        content_top = Inches(1.5)
        content_height = self.slide_height - content_top - Inches(0.5)
        content_left = Inches(0.5)
        content_width = self.slide_width - Inches(1.0)


        if not risk_factors:
            logger.warning("No risk factors provided for the slide.")
            # Add placeholder
            no_data_box = slide.shapes.add_textbox(content_left, content_top, content_width, Inches(1))
            no_data_box.text = "No specific risk factors identified or listed."
            self._style_text_frame(no_data_box.text_frame, font_size=Pt(16), italic=True, alignment=PP_ALIGN.CENTER)
            return

        # Add warning icon (optional visual cue)
        try:
            icon_size = Inches(0.6)
            icon_left = Inches(0.5)
            icon_top = content_top # Align with top of text
            # Using ISOSCELES_TRIANGLE pointing up
            warning_icon = slide.shapes.add_shape(
                MSO_SHAPE.ISOSCELES_TRIANGLE, icon_left, icon_top, icon_size, icon_size
            )
            warning_icon.rotation = 180.0 # Point down (like typical warning)
            warning_icon.fill.solid()
            warning_icon.fill.fore_color.rgb = RGBColor.from_string(self.design.NEGATIVE)
            warning_icon.line.fill.background() # No border
            text_start_left = icon_left + icon_size + Inches(0.2) # Start text after icon
            text_width = self.slide_width - text_start_left - Inches(0.5)
        except Exception as e:
            logger.error(f"Could not add warning icon: {e}", exc_info=True)
            # Fallback to standard layout if icon fails
            text_start_left = content_left
            text_width = content_width


        # Determine layout: Single or double column
        max_items_single_col = 6 # Adjust as needed based on font size/spacing
        use_two_columns = len(risk_factors) > max_items_single_col

        if not use_two_columns:
            # Single column layout
            try:
                box = slide.shapes.add_textbox(text_start_left, content_top, text_width, content_height)
                tf = box.text_frame
                tf.clear()
                tf.word_wrap = True
                tf.vertical_anchor = MSO_ANCHOR.TOP
                #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

                for factor in risk_factors:
                    p = tf.add_paragraph()
                    p.text = factor
                    p.font.size = Pt(16)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.level = 0 # Ensure bullets align if theme adds them
                    p.space_before = Pt(6)
                    p.space_after = Pt(6)
                tf.margin_left = Inches(0.1) # Indent if using bullets

            except Exception as e:
                 logger.error(f"Error adding single column risk factors: {e}", exc_info=True)

        else:
            # Two column layout
            col_width = (text_width - Inches(0.3)) / 2 # Width per column with gap
            col1_left = text_start_left
            col2_left = col1_left + col_width + Inches(0.3)
            mid_point = math.ceil(len(risk_factors) / 2)
            left_column_factors = risk_factors[:mid_point]
            right_column_factors = risk_factors[mid_point:]

            # Left column
            try:
                left_box = slide.shapes.add_textbox(col1_left, content_top, col_width, content_height)
                tf_left = left_box.text_frame
                tf_left.clear()
                tf_left.word_wrap = True
                tf_left.vertical_anchor = MSO_ANCHOR.TOP
                #tf_left.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

                for factor in left_column_factors:
                    p = tf_left.add_paragraph()
                    p.text = factor
                    p.font.size = Pt(14) # Slightly smaller for two columns
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.level = 0
                    p.space_before = Pt(4)
                    p.space_after = Pt(4)
                tf_left.margin_left = Inches(0.1)

            except Exception as e:
                logger.error(f"Error adding left column risk factors: {e}", exc_info=True)

            # Right column
            try:
                right_box = slide.shapes.add_textbox(col2_left, content_top, col_width, content_height)
                tf_right = right_box.text_frame
                tf_right.clear()
                tf_right.word_wrap = True
                tf_right.vertical_anchor = MSO_ANCHOR.TOP
                #tf_right.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

                for factor in right_column_factors:
                    p = tf_right.add_paragraph()
                    p.text = factor
                    p.font.size = Pt(14)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.level = 0
                    p.space_before = Pt(4)
                    p.space_after = Pt(4)
                tf_right.margin_left = Inches(0.1)

            except Exception as e:
                logger.error(f"Error adding right column risk factors: {e}", exc_info=True)

    def add_compliance_overview_slide(self, title: str, overall_score: float,
                                    compliance_level: str, jurisdictions: str,
                                    kyc_aml_coverage: float) -> None:
        """
        Add a compliance overview slide with scores, jurisdictions, and summary.

        Args:
            title: Slide title.
            overall_score: Overall compliance score (0-100, higher is better).
            compliance_level: Compliance level category (e.g., "High", "Moderate").
            jurisdictions: Comma-separated string or list of key jurisdictions.
            kyc_aml_coverage: KYC/AML coverage score (0-100, higher is better).
        """
        slide_layout = self._get_slide_layout(5) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Compliance Overview Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on compliance overview slide layout.")

        # Define layout areas
        gauge_width = Inches(3.5)
        gauge_height = Inches(2.0)
        col1_left = Inches(0.5)
        col2_left = col1_left + gauge_width + Inches(0.3)
        col3_left = col2_left + gauge_width + Inches(0.3)
        row1_top = Inches(1.5) # Top row for gauges/jurisdictions
        row2_top = row1_top + gauge_height + Inches(0.5) # Second row for summary
        summary_width = self.slide_width - Inches(1.0) # Full width summary

        # Add Overall Compliance Score Gauge (Top Left)
        try:
            compliance_color_hex = self._get_color_from_score(overall_score) # Higher score is better
            self._add_gauge_chart(
                slide, col1_left, row1_top, gauge_width, gauge_height,
                overall_score, compliance_level, compliance_color_hex,
                "Overall Compliance Score"
            )
        except Exception as e:
             logger.error(f"Error adding compliance score gauge: {e}", exc_info=True)


        # Add KYC/AML Score Gauge (Top Middle)
        try:
            kyc_color_hex = self._get_color_from_score(kyc_aml_coverage) # Higher score is better
            self._add_gauge_chart(
                slide, col2_left, row1_top, gauge_width, gauge_height,
                kyc_aml_coverage, "", kyc_color_hex, # No level text needed
                "KYC/AML Coverage Score"
            )
        except Exception as e:
             logger.error(f"Error adding KYC/AML score gauge: {e}", exc_info=True)


        # Add Jurisdictions Box (Top Right)
        jur_width = self.slide_width - col3_left - Inches(0.5) # Fill remaining width
        try:
            jur_box = slide.shapes.add_textbox(
                col3_left, row1_top, jur_width, gauge_height # Align height with gauges
            )
            tf = jur_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Key Jurisdictions"
            p.font.bold = True
            p.font.size = Pt(16)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            # Format jurisdictions string nicely
            if isinstance(jurisdictions, list):
                jur_text = ", ".join(jurisdictions)
            elif isinstance(jurisdictions, str):
                jur_text = jurisdictions.replace(",", ", ") # Ensure spacing
            else:
                jur_text = "N/A"

            p = tf.add_paragraph()
            p.text = jur_text if jur_text else "None Specified"
            p.font.size = Pt(12)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)

        except Exception as e:
            logger.error(f"Error adding jurisdictions box: {e}", exc_info=True)


        # Add Compliance Summary (Bottom Row)
        summary_height = self.slide_height - row2_top - Inches(0.5)
        try:
            summary_box = slide.shapes.add_textbox(
                col1_left, row2_top, summary_width, summary_height
            )
            tf = summary_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Compliance Assessment Summary"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.ACCENT_1)
            p.space_after = Pt(6)

            # Generate compliance summary text based on score/level
            # Customize these based on your compliance categories
            level_lower = compliance_level.lower()
            summary = "Compliance framework analysis indicates areas for review. Refer to detailed findings." # Default
            if overall_score >= 85 or "high" in level_lower or "substantial" in level_lower:
                summary = "The fund demonstrates a strong compliance posture with robust procedures observed across key areas. KYC/AML coverage is comprehensive, aligning well with regulatory expectations in relevant jurisdictions."
            elif overall_score >= 60 or "moderate" in level_lower or "adequate" in level_lower:
                summary = "Compliance measures appear adequate for basic requirements. However, opportunities for enhancement exist, particularly in [mention specific area if known, e.g., ongoing monitoring or documentation]. KYC/AML processes seem functional but could be strengthened."
            else: # Low score or "Partial"/"Low" level
                summary = "Significant gaps identified in the compliance framework pose potential regulatory risks. Key areas such as [mention specific area if known, e.g., policy enforcement or reporting] require immediate attention. Strengthening the overall compliance program is recommended."

            p = tf.add_paragraph()
            p.text = summary
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
            p.space_after = Pt(4)

        except Exception as e:
            logger.error(f"Error adding compliance summary: {e}", exc_info=True)


    def add_table_slide(self, title: str, table_data: List[List[str]],
                      notes: Optional[str] = None) -> None:
        """
        Add a generic slide containing a single, well-formatted table.

        Args:
            title: Slide title.
            table_data: Table data as list of lists. Assumes first row is header.
            notes: Optional notes or commentary to display below the table.
        """
        slide_layout = self._get_slide_layout(1) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Table Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on table slide layout.")


        # Define layout areas
        table_top = Inches(1.5)
        table_width = Inches(12.33)
        notes_top_margin = Inches(0.2)
        notes_height = Inches(1.0) # Reserve space for notes

        # Add table
        table_actual_height = Inches(0.2) # Min height
        try:
            if not table_data or not table_data[0]:
                logger.warning(f"Table data for slide '{title}' is empty. Skipping table.")
                no_data_box = slide.shapes.add_textbox(Inches(0.5), table_top, table_width, Inches(1))
                no_data_box.text = "Data not available for this table."
                self._style_text_frame(no_data_box.text_frame, font_size=Pt(16), italic=True, alignment=PP_ALIGN.CENTER)
                notes_top = table_top + Inches(1) + notes_top_margin # Adjust notes position
            else:
                rows = len(table_data)
                cols = len(table_data[0])
                table_actual_height = Inches(rows * 0.4 + 0.2)
                max_table_height = self.slide_height - table_top - (notes_height if notes else 0) - Inches(0.75) # Max height available
                table_actual_height = min(table_actual_height, max_table_height)

                shape = slide.shapes.add_table(
                    rows, cols, Inches(0.5), table_top, table_width, table_actual_height
                )
                table = shape.table

                # Set column widths (distribute evenly)
                col_width = int(table_width / cols)
                for i in range(cols):
                    table.columns[i].width = col_width

                # Fill table with data and style
                for i, row_data in enumerate(table_data):
                    for j, cell_value_raw in enumerate(row_data):
                        cell = table.cell(i, j)
                        cell_value = str(cell_value_raw)
                        cell.text = cell_value
                        tf = cell.text_frame
                        tf.margin_left = Inches(0.08)
                        tf.margin_right = Inches(0.08)
                        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

                        is_header = (i == 0)
                        is_alt_row = (i > 0 and i % 2 == 1)

                        # Default style
                        self._style_text_frame(tf, font_size=Pt(11), font_color=self.design.TEXT_DARK)

                        # Header style
                        if is_header:
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_HEADER)
                            self._style_text_frame(tf, font_size=Pt(11), bold=True, font_color=self.design.TEXT_LIGHT)
                        # Alternate row style
                        elif is_alt_row:
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = RGBColor.from_string(self.design.TABLE_ALT_ROW)

                notes_top = table_top + table_actual_height + notes_top_margin # Position below actual table

        except Exception as e:
            logger.error(f"Error adding table to slide '{title}': {e}", exc_info=True)
            notes_top = table_top + Inches(1) + notes_top_margin # Fallback position

        # Add notes if provided
        if notes:
            try:
                notes_box = slide.shapes.add_textbox(
                    Inches(0.5), notes_top, table_width, notes_height
                )
                tf = notes_box.text_frame
                tf.clear()
                tf.word_wrap = True
                tf.vertical_anchor = MSO_ANCHOR.TOP
                tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

                p = tf.add_paragraph()
                p.text = "Notes:"
                p.font.bold = True
                p.font.size = Pt(10)
                p.font.color.rgb = RGBColor.from_string(self.design.TEXT_MUTED)
                p.space_after = Pt(2)

                p = tf.add_paragraph()
                p.text = notes
                p.font.size = Pt(10)
                p.font.italic = True
                p.font.color.rgb = RGBColor.from_string(self.design.TEXT_MUTED)

            except Exception as e:
                logger.error(f"Error adding notes to table slide '{title}': {e}", exc_info=True)

    def add_text_slide(self, title: str, content: str, text_size: int = 16) -> None:
        """
        Add a generic slide with a title and block of text content.

        Args:
            title: Slide title.
            content: Text content (can include newlines).
            text_size: Font size in points for the main content.
        """
        slide_layout = self._get_slide_layout(1) # Title and Content layout
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Text Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on text slide layout.")


        # Add content text box (use main content placeholder if available, else add textbox)
        try:
            # Try accessing the main content placeholder (often index 1)
            content_placeholder = slide.placeholders[1]
            tf = content_placeholder.text_frame
            tf.clear() # Clear any default text
        except (AttributeError, IndexError):
            logger.info("Content placeholder not found or invalid index, adding new textbox.")
            text_left = Inches(0.5)
            text_top = Inches(1.5)
            text_width = Inches(12.33)
            text_height = Inches(5.5) # Allow ample space
            text_box = slide.shapes.add_textbox(text_left, text_top, text_width, text_height)
            tf = text_box.text_frame
            tf.clear()

        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT # Let it grow

        # Add content text, handling newlines
        lines = content.split('\n')
        first_line = True
        for line in lines:
            p = tf.add_paragraph()
            # Removed 'if first_line: p = tf.paragraphs[0]' because tf.clear() makes this unreliable
            # Always add a new paragraph for simplicity after clearing.
            p.text = line if line.strip() else " " # Use space for empty lines to maintain spacing
            p.font.size = Pt(text_size)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
            p.space_after = Pt(6) # Spacing between paragraphs
            p.level = 0 # Ensure no unintended indentation
            # first_line = False # Not needed anymore

    def add_conclusion_slide(self, title: str, fund_name: str, risk_level: str,
                           risk_score: float, compliance_level: str, compliance_score: float,
                           strengths: List[str], concerns: List[str],
                           recommendation: Optional[str] = None) -> None:
        """
        Add a conclusion slide summarizing key findings and assessment.

        Args:
            title: Slide title (e.g., "Conclusion", "Overall Assessment").
            fund_name: Fund name.
            risk_level: Final risk level category.
            risk_score: Final risk score (0-100, higher is riskier).
            compliance_level: Final compliance level category.
            compliance_score: Final compliance score (0-100, higher is better).
            strengths: List of key strengths (bullet points).
            concerns: List of key concerns (bullet points).
            recommendation: Optional overall recommendation text.
        """
        slide_layout = self._get_slide_layout(5) # Title and Content layout or similar
        slide = self.prs.slides.add_slide(slide_layout)
        self.slide_count += 1
        logger.info(f"Added Conclusion Slide: '{title}'")

        # Set title
        try:
            title_shape = slide.shapes.title
            title_shape.text = title
            self._style_text_frame(title_shape.text_frame, font_size=Pt(40), font_color=self.design.TEXT_DARK, bold=True)
        except AttributeError:
            logger.warning("Title placeholder not found on conclusion slide layout.")


        # Define layout areas
        row1_top = Inches(1.3)
        row2_top = Inches(3.0) # Start strengths/concerns lower
        row_height = Inches(3.5) # Height for strengths/concerns boxes
        col_width = Inches(6.0)
        col_gap = Inches(0.5)
        col1_left = Inches(0.5)
        col2_left = col1_left + col_width + col_gap

        # Add fund name and overall assessment / recommendation text (Row 1)
        try:
            summary_box = slide.shapes.add_textbox(
                col1_left, row1_top, self.slide_width - Inches(1.0), Inches(1.5) # Span width, limited height
            )
            tf = summary_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP

            p = tf.add_paragraph()
            p.text = f"Overall Assessment: {fund_name}"
            p.font.bold = True
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
            p.space_after = Pt(6)

            # Add recommendation or generate a summary assessment
            assessment_text = recommendation
            if not assessment_text:
                # Generate default assessment based on scores
                if risk_score < 40 and compliance_score > 70:
                    assessment_text = "Based on the due diligence findings, the fund presents a favorable profile with robust risk management and high compliance standards."
                elif risk_score < 60 and compliance_score > 50:
                    assessment_text = "The fund demonstrates a moderate risk and compliance profile. While generally sound, targeted improvements in identified areas of concern are advisable."
                else:
                    assessment_text = "Significant concerns regarding the fund's risk exposure and/or compliance framework were identified. Careful consideration and potential remediation are required."
            p = tf.add_paragraph()
            p.text = assessment_text
            p.font.size = Pt(14)
            p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
            p.space_after = Pt(8)


            # Add Key Scores Summary
            risk_color = self._get_color_from_score(risk_score, invert=True) # High risk = bad color
            compliance_color = self._get_color_from_score(compliance_score) # High compliance = good color

            p = tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            # Add risk score part
            run = p.add_run()
            run.text = f"Risk: {risk_level} ({risk_score:.1f})  "
            run.font.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor.from_string(risk_color)
            # Add compliance score part
            run = p.add_run()
            run.text = f"  Compliance: {compliance_level} ({compliance_score:.1f})"
            run.font.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor.from_string(compliance_color)

        except Exception as e:
            logger.error(f"Error adding conclusion summary text: {e}", exc_info=True)


        # Add Strengths column (Row 2 Left)
        try:
            strengths_box = slide.shapes.add_textbox(
                col1_left, row2_top, col_width, row_height
            )
            tf = strengths_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Key Strengths"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.POSITIVE)
            p.space_after = Pt(6)

            if not strengths:
                 p = tf.add_paragraph()
                 p.text = "(None explicitly listed)"
                 p.font.size = Pt(12)
                 p.font.italic = True
            else:
                for strength in strengths:
                    p = tf.add_paragraph()
                    p.text = f"✓ {strength}" # Checkmark symbol
                    p.font.size = Pt(12)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.level = 0
                    p.space_after = Pt(4)
                tf.margin_left = Inches(0.1) # Indent bullets

        except Exception as e:
             logger.error(f"Error adding strengths to conclusion: {e}", exc_info=True)


        # Add Concerns column (Row 2 Right)
        try:
            concerns_box = slide.shapes.add_textbox(
                col2_left, row2_top, col_width, row_height
            )
            tf = concerns_box.text_frame
            tf.clear()
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.TOP
            #tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            p = tf.add_paragraph()
            p.text = "Key Concerns / Areas for Improvement"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor.from_string(self.design.NEGATIVE)
            p.space_after = Pt(6)

            if not concerns:
                 p = tf.add_paragraph()
                 p.text = "(None explicitly listed)"
                 p.font.size = Pt(12)
                 p.font.italic = True
            else:
                for concern in concerns:
                    p = tf.add_paragraph()
                    p.text = f"• {concern}" # Bullet point symbol
                    # p.text = f"! {concern}" # Alternative: Exclamation symbol
                    p.font.size = Pt(12)
                    p.font.color.rgb = RGBColor.from_string(self.design.TEXT_DARK)
                    p.level = 0
                    p.space_after = Pt(4)
                tf.margin_left = Inches(0.1) # Indent bullets

        except Exception as e:
             logger.error(f"Error adding concerns to conclusion: {e}", exc_info=True)

