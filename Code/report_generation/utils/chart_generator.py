"""
Chart generation utilities for PowerPoint reports
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import logging
from typing import List, Dict
from matplotlib.dates import DateFormatter

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate charts for PowerPoint reports."""
    
    def __init__(self):
        # Set matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Chart settings
        self.colors = ['#0054A4', '#7F7F7F', '#FFA500', '#2ECC71', '#E74C3C']
        
    def generate_price_chart(self, price_data: List[Dict]) -> BytesIO:
        """Generate a price chart."""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(price_data)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Create chart
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['price'], linewidth=2, color=self.colors[0])
            
            # Format chart
            plt.title('Price History', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price (USD)', fontsize=12)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            date_formatter = DateFormatter('%Y-%m-%d')
            plt.gca().xaxis.set_major_formatter(date_formatter)
            
            # Format y-axis as currency
            def currency_formatter(x, p):
                return f'${x:,.2f}'
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error generating price chart: {str(e)}")
            # Return empty chart on error
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Price chart unavailable', ha='center', va='center')
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            return img_buffer
    
    def generate_sentiment_pie_chart(self, sentiment_data: Dict) -> BytesIO:
        """Generate a sentiment pie chart."""
        try:
            # Extract sentiment values
            labels = list(sentiment_data.keys())
            values = list(sentiment_data.values())
            
            # Create pie chart
            plt.figure(figsize=(8, 8))
            
            # Define colors for sentiment
            sentiment_colors = {
                'POSITIVE': '#2ECC71',  # Green
                'NEUTRAL': '#7F7F7F',   # Gray
                'NEGATIVE': '#E74C3C'   # Red
            }
            
            # Map colors to labels
            colors = [sentiment_colors.get(label.upper(), '#CCCCCC') for label in labels]
            
            # Create pie chart
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis('equal')
            
            plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error generating sentiment pie chart: {str(e)}")
            # Return empty chart on error
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, 'Sentiment chart unavailable', ha='center', va='center')
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            return img_buffer
    
    def generate_risk_radar_chart(self, risk_data: Dict) -> BytesIO:
        """Generate a risk radar chart."""
        try:
            # Extract risk categories and values
            categories = list(risk_data.keys())
            values = list(risk_data.values())
            
            # Number of variables
            num_vars = len(categories)
            
            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            
            # Make the plot circular
            values += values[:1]
            angles += angles[:1]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Draw the outline of our data
            ax.plot(angles, values, color=self.colors[0], linewidth=2)
            ax.fill(angles, values, color=self.colors[0], alpha=0.25)
            
            # Draw the category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=12)
            
            # Set value range (0-10 for risk scores)
            ax.set_ylim(0, 10)
            
            # Add title
            plt.title('Risk Assessment Radar', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error generating risk radar chart: {str(e)}")
            # Return empty chart on error
            plt.figure(figsize=(8, 8))
            plt.text(0.5, 0.5, 'Risk radar chart unavailable', ha='center', va='center')
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            return img_buffer
    
    def generate_comparison_bar_chart(self, comparison_data: Dict) -> BytesIO:
        """Generate a comparison bar chart."""
        try:
            # Extract data
            categories = list(comparison_data.keys())
            values1 = [v[0] for v in comparison_data.values()]
            values2 = [v[1] for v in comparison_data.values()]
            
            # Set up bar positions
            x = np.arange(len(categories))
            width = 0.35
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            rects1 = ax.bar(x - width/2, values1, width, label='Asset 1', color=self.colors[0])
            rects2 = ax.bar(x + width/2, values2, width, label='Asset 2', color=self.colors[1])
            
            # Add labels and title
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Comparative Analysis', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}',
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            plt.tight_layout()
            
            # Save to BytesIO
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error generating comparison bar chart: {str(e)}")
            # Return empty chart on error
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'Comparison chart unavailable', ha='center', va='center')
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            return img_buffer