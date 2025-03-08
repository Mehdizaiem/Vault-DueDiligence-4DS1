# File path: Code/data_processing/csv_loader.py

import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoCSVLoader:
    """Loader for cryptocurrency historical CSV data files"""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the CSV loader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        # Set default data directory if not provided
        if data_dir is None:
            # Default to project root/data/time series cryptos
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            self.data_dir = os.path.join(project_root, "data", "time series cryptos")
        elif not os.path.isabs(data_dir):
            # Relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            self.data_dir = os.path.join(project_root, data_dir)
        else:
            self.data_dir = data_dir
            
        logger.info(f"CSV loader initialized with data directory: {self.data_dir}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available crypto symbols from CSV files"""
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
            
        files = os.listdir(self.data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        # Extract symbol from filename (e.g., "BTC_USD Bitfinex Historical Data.csv" -> "BTCUSD")
        symbols = []
        for csv_file in csv_files:
            # Split by space or underscore to extract symbol info
            parts = csv_file.split()
            if len(parts) > 0:
                # Get the first part which should contain the symbol
                symbol_part = parts[0]
                
                # Handle common formats
                if '_' in symbol_part:
                    # Format like "BTC_USD"
                    symbol = symbol_part.replace('_', '')
                elif '/' in symbol_part:
                    # Format like "BTC/USD"
                    symbol = symbol_part.replace('/', '')
                else:
                    # Other format, just use as is
                    symbol = symbol_part
                
                symbols.append(symbol)
        
        logger.info(f"Found {len(symbols)} symbols in CSV files: {symbols}")
        return symbols
    
    def load_historical_data(self, symbol: str) -> List[Dict]:
        """
        Load historical data for a symbol from CSV file.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., "BTCUSD")
            
        Returns:
            List of historical data points
        """
        # Find the matching CSV file
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
            
        files = os.listdir(self.data_dir)
        
        # Create possible symbol formats to search for
        base, quote = self._split_symbol(symbol)
        symbol_formats = [
            f"{base}_{quote}",  # BTC_USD
            f"{base}/{quote}",  # BTC/USD
            f"{base}{quote}"    # BTCUSD
        ]
        
        # Find files that match any of the symbol formats
        matching_files = []
        for sym_format in symbol_formats:
            matching_files.extend([f for f in files if f.startswith(sym_format) and f.endswith('.csv')])
        
        if not matching_files:
            logger.warning(f"No CSV file found for symbol: {symbol}")
            return []
        
        csv_path = os.path.join(self.data_dir, matching_files[0])
        logger.info(f"Loading data from: {csv_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Examine columns to determine format
            columns = df.columns.tolist()
            logger.info(f"CSV columns: {columns}")
            
            # Process based on common CSV formats
            # Try to identify key columns automatically
            date_col = self._find_column(columns, ['Date', 'date', 'timestamp', 'time'])
            open_col = self._find_column(columns, ['Open', 'open', 'open_price'])
            high_col = self._find_column(columns, ['High', 'high', 'high_price'])
            low_col = self._find_column(columns, ['Low', 'low', 'low_price'])
            close_col = self._find_column(columns, ['Close', 'close', 'close_price', 'Price', 'price'])
            volume_col = self._find_column(columns, ['Volume', 'volume', 'Vol.', 'vol', 'volume_base', 'volume_quote'])
            
            # Check if we found the minimum required columns
            if not date_col or not close_col:
                logger.error(f"CSV format not recognized: missing required date or close columns")
                return []
            
            # Convert to list of dictionaries
            data_points = []
            
            # Extract exchange from filename if possible
            exchange_parts = matching_files[0].split()
            exchange = exchange_parts[1] if len(exchange_parts) > 1 else "unknown"
            
            for _, row in df.iterrows():
                try:
                    # Parse date
                    date_str = row[date_col]
                    try:
                        # Try different date formats
                        date_obj = pd.to_datetime(date_str)
                        date_iso = date_obj.isoformat()
                    except:
                        # If parsing fails, use current date
                        logger.warning(f"Failed to parse date: {date_str}")
                        date_iso = datetime.now().isoformat()
                    
                    # Get price values, with fallbacks
                    close_price = float(row[close_col])
                    open_price = float(row[open_col]) if open_col else close_price
                    high_price = float(row[high_col]) if high_col else close_price
                    low_price = float(row[low_col]) if low_col else close_price
                    volume = float(row[volume_col]) if volume_col else 0
                    
                    # Create standardized data point
                    data_point = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timestamp": date_iso,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "interval": "1d"  # Assuming daily data
                    }
                    
                    data_points.append(data_point)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
            
            # Sort by date
            data_points.sort(key=lambda x: x["timestamp"])
            
            logger.info(f"Loaded {len(data_points)} data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            return []
    
    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """Find a column from a list of possible column names"""
        for name in possible_names:
            # Case-insensitive search
            for col in columns:
                if col.lower() == name.lower():
                    return col
        return None
    
    def _split_symbol(self, symbol: str) -> tuple:
        """Split a symbol into base and quote parts"""
        # Try to intelligently split the symbol
        common_quotes = ["USD", "USDT", "BTC", "ETH"]
        
        for quote in common_quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        
        # Default fallback: assume last 3 characters are quote
        if len(symbol) > 3:
            return symbol[:-3], symbol[-3:]
        
        # Can't determine - return as is with empty quote
        return symbol, ""

# Example usage
if __name__ == "__main__":
    loader = CryptoCSVLoader()
    symbols = loader.get_available_symbols()
    print(f"Found symbols: {symbols}")
    
    if symbols:
        # Test loading data for the first symbol
        data = loader.load_historical_data(symbols[0])
        print(f"Loaded {len(data)} data points for {symbols[0]}")
        if data:
            print("First data point:", data[0])