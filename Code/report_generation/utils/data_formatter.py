"""
Data formatting utilities for PowerPoint reports
"""
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Union, Any

logger = logging.getLogger(__name__)

class DataFormatter:
    """Format data for presentation in PowerPoint reports."""
    
    def format_currency(self, value: Union[int, float], symbol: str = "$") -> str:
        """Format a numeric value as currency."""
        try:
            if value >= 1e9:
                return f"{symbol}{value/1e9:.2f}B"
            elif value >= 1e6:
                return f"{symbol}{value/1e6:.2f}M"
            elif value >= 1e3:
                return f"{symbol}{value/1e3:.2f}K"
            else:
                return f"{symbol}{value:.2f}"
        except Exception as e:
            logger.error(f"Error formatting currency: {str(e)}")
            return "N/A"
    
    def format_percentage(self, value: Union[int, float], decimal_places: int = 2) -> str:
        """Format a numeric value as percentage."""
        try:
            return f"{value:.{decimal_places}f}%"
        except Exception as e:
            logger.error(f"Error formatting percentage: {str(e)}")
            return "N/A"
    
    def format_datetime(self, value: Union[str, datetime], format_str: str = "%B %d, %Y") -> str:
        """Format a datetime value."""
        try:
            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return value.strftime(format_str)
        except Exception as e:
            logger.error(f"Error formatting datetime: {str(e)}")
            return str(value)
    
    def format_market_data(self, data: Dict) -> Dict:
        """Format market data for display."""
        try:
            formatted_data = {}
            
            # Format each field appropriately
            if 'price' in data:
                formatted_data['price'] = self.format_currency(data['price'])
            
            if 'market_cap' in data:
                formatted_data['market_cap'] = self.format_currency(data['market_cap'])
            
            if 'volume_24h' in data:
                formatted_data['volume_24h'] = self.format_currency(data['volume_24h'])
            
            if 'price_change_24h' in data:
                formatted_data['price_change_24h'] = self.format_percentage(data['price_change_24h'])
            
            if 'timestamp' in data:
                formatted_data['timestamp'] = self.format_datetime(data['timestamp'])
            
            return formatted_data
        except Exception as e:
            logger.error(f"Error formatting market data: {str(e)}")
            return data
    
    def format_table_data(self, data: List[Dict], columns: List[str] = None) -> List[List[str]]:
        """Format data for table display."""
        try:
            if not data:
                return []
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data)
            
            # Select columns if specified
            if columns:
                df = df[columns]
            
            # Format numeric columns
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if 'price' in col.lower() or 'value' in col.lower() or 'market_cap' in col.lower():
                    df[col] = df[col].apply(lambda x: self.format_currency(x))
                elif 'percent' in col.lower() or 'change' in col.lower():
                    df[col] = df[col].apply(lambda x: self.format_percentage(x))
            
            # Format datetime columns
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].apply(lambda x: self.format_datetime(x))
            
            # Convert to list of lists
            return [list(df.columns)] + df.values.tolist()
        except Exception as e:
            logger.error(f"Error formatting table data: {str(e)}")
            return []
    
    def format_text_for_slides(self, text: str, max_length: int = 500) -> str:
        """Format text for display on slides."""
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length-3] + "..."
            
            return text
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return text
    
    def format_bullet_points(self, items: List[str], max_items: int = 5) -> str:
        """Format items as bullet points."""
        try:
            # Limit number of items
            items = items[:max_items]
            
            # Format as bullet points
            return "\n".join([f"• {item}" for item in items])
        except Exception as e:
            logger.error(f"Error formatting bullet points: {str(e)}")
            return "• No data available"
    
    def format_key_metrics(self, metrics: Dict) -> Dict:
        """Format key metrics for display."""
        try:
            formatted_metrics = {}
            
            for key, value in metrics.items():
                # Format based on value type and key name
                if isinstance(value, (int, float)):
                    if 'price' in key.lower() or 'value' in key.lower() or 'cap' in key.lower():
                        formatted_metrics[key] = self.format_currency(value)
                    elif 'percent' in key.lower() or 'change' in key.lower() or 'rate' in key.lower():
                        formatted_metrics[key] = self.format_percentage(value)
                    elif value > 1000:
                        formatted_metrics[key] = f"{value:,.0f}"
                    else:
                        formatted_metrics[key] = f"{value:.2f}"
                elif isinstance(value, datetime):
                    formatted_metrics[key] = self.format_datetime(value)
                else:
                    formatted_metrics[key] = str(value)
            
            return formatted_metrics
        except Exception as e:
            logger.error(f"Error formatting key metrics: {str(e)}")
            return metrics
    
    def format_risk_factors(self, risk_factors: List[Dict]) -> List[Dict]:
        """Format risk factors for display."""
        try:
            formatted_risks = []
            
            for risk in risk_factors:
                formatted_risk = {}
                
                # Format risk level
                if 'level' in risk:
                    level = risk['level'].upper()
                    if level == 'HIGH':
                        formatted_risk['level'] = '⚠️ HIGH'
                    elif level == 'MEDIUM':
                        formatted_risk['level'] = '⚡ MEDIUM'
                    elif level == 'LOW':
                        formatted_risk['level'] = '✓ LOW'
                    else:
                        formatted_risk['level'] = risk['level']
                
                # Format other fields
                for key, value in risk.items():
                    if key != 'level':
                        formatted_risk[key] = str(value)
                
                formatted_risks.append(formatted_risk)
            
            return formatted_risks
        except Exception as e:
            logger.error(f"Error formatting risk factors: {str(e)}")
            return risk_factors
    
    def format_chart_data(self, data: List[Dict], x_field: str, y_field: str) -> Dict:
        """Format data for charts."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Ensure fields exist
            if x_field not in df.columns or y_field not in df.columns:
                raise ValueError(f"Required fields not found: {x_field}, {y_field}")
            
            # Format datetime if necessary
            if df[x_field].dtype == 'datetime64[ns]' or df[x_field].dtype == 'object':
                df[x_field] = pd.to_datetime(df[x_field])
            
            # Convert to numeric if necessary
            if df[y_field].dtype == 'object':
                df[y_field] = pd.to_numeric(df[y_field], errors='coerce')
            
            # Return formatted data
            return {
                'x': df[x_field].tolist(),
                'y': df[y_field].tolist(),
                'labels': [self.format_datetime(x) if isinstance(x, datetime) else str(x) for x in df[x_field]],
                'values': [self.format_currency(y) if isinstance(y, (int, float)) else str(y) for y in df[y_field]]
            }
        except Exception as e:
            logger.error(f"Error formatting chart data: {str(e)}")
            return {'x': [], 'y': [], 'labels': [], 'values': []}