# File path: Code/data_processing/csv_loader.py

import os
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import csv
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoCSVLoader:
    """Enhanced loader for cryptocurrency historical CSV data files with robust parsing"""
    
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
        
        # Define common patterns for parsing
        self.symbol_patterns = {
            # Format: regex pattern -> (base, quote)
            r'(.+)[-_/]([A-Z]{3,5})': lambda m: (m.group(1), m.group(2)),  # BTC-USD, BTC/USD, BTC_USD
            r'([A-Z]{3,5})[-_/](.+)': lambda m: (m.group(1), m.group(2)),  # Reverse order: USD-BTC
            r'([A-Z]+)([A-Z]{3,5})$': lambda m: (m.group(1), m.group(2))   # BTCUSD (no separator)
        }
        
        # Define common volume formats
        self.volume_multipliers = {
            'K': 1e3,   # Thousands
            'M': 1e6,   # Millions
            'B': 1e9,   # Billions
            'T': 1e12   # Trillions
        }
        
        # Exchange name mapping for standardization
        self.exchange_mapping = {
            "coinbase": "Coinbase",
            "binance": "Binance",
            "kraken": "Kraken",
            "bitfinex": "Bitfinex", 
            "bitstamp": "Bitstamp",
            "ftx": "FTX",
            "gemini": "Gemini",
            "huobi": "Huobi",
            "kucoin": "KuCoin",
            "okex": "OKEx"
        }
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available crypto symbols from CSV files with enhanced detection.
        
        Returns:
            List[str]: List of detected symbols
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
            
        files = os.listdir(self.data_dir)
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        
        symbols = []
        symbol_files = {}  # Track which files map to which symbols
        
        for csv_file in csv_files:
            # Try to extract symbol from filename
            symbol = self._extract_symbol_from_filename(csv_file)
            
            # If we couldn't extract from filename, try peeking at the file
            if not symbol:
                symbol = self._peek_symbol_from_file(os.path.join(self.data_dir, csv_file))
            
            if symbol:
                # Store with USDT suffix for consistency
                if not symbol.upper().endswith(('USD', 'USDT')):
                    symbol = f"{symbol}USD"
                
                # Format to uppercase
                symbol = symbol.upper()
                
                # Add to list if not already present
                if symbol not in symbols:
                    symbols.append(symbol)
                    symbol_files[symbol] = csv_file
                    logger.debug(f"Detected symbol {symbol} from file {csv_file}")
        
        logger.info(f"Found {len(symbols)} symbols in CSV files")
        for symbol, filename in symbol_files.items():
            logger.debug(f"Symbol {symbol} mapped to {filename}")
            
        return symbols
    
    def _extract_symbol_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract cryptocurrency symbol from filename using patterns.
        
        Args:
            filename: CSV filename
            
        Returns:
            str or None: Extracted symbol or None if not found
        """
        # Remove file extension
        basename = os.path.splitext(filename)[0]
        
        # Try each pattern
        for pattern, extractor in self.symbol_patterns.items():
            match = re.search(pattern, basename)
            if match:
                try:
                    base, quote = extractor(match)
                    # Clean up the base symbol
                    base = re.sub(r'[^A-Za-z]', '', base)
                    return base
                except Exception as e:
                    logger.debug(f"Error extracting from {basename} with pattern {pattern}: {e}")
                    continue
        
        # Fall back to splitting by common delimiters
        for delimiter in ['_', '-', ' ', '/']:
            if delimiter in basename:
                parts = basename.split(delimiter)
                if parts and len(parts[0]) <= 5:  # Assuming symbol is max 5 characters
                    return parts[0]
        
        return None
    
    def _peek_symbol_from_file(self, file_path: str) -> Optional[str]:
        """
        Peek into a file to try to detect the symbol by reading headers.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            str or None: Detected symbol or None
        """
        try:
            # Detect file encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read(4096)  # Read a sample
                encoding = chardet.detect(raw_data)['encoding']
            
            # Read first few lines
            with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                lines = [next(f) for _ in range(5) if f]  # Get up to 5 lines
            
            # Check if any line contains a symbol pattern
            for line in lines:
                for pattern, extractor in self.symbol_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        try:
                            base, quote = extractor(match)
                            # Clean up the base symbol
                            base = re.sub(r'[^A-Za-z]', '', base)
                            return base
                        except Exception:
                            continue
                            
            # If nothing found in lines, try pandas to read header
            df = pd.read_csv(file_path, nrows=0)  # Just get headers
            header_text = ' '.join(df.columns)
            
            # Look for common symbols in headers
            common_symbols = ['BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'SOL']
            for symbol in common_symbols:
                if symbol in header_text:
                    return symbol
                    
            return None
            
        except Exception as e:
            logger.debug(f"Error peeking into file {file_path}: {e}")
            return None
    
    def load_historical_data(self, symbol: str) -> List[Dict]:
        """
        Load historical data for a symbol from CSV file with enhanced robustness.
        
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
            f"{base}-{quote}",  # BTC-USD
            f"{base}{quote}",   # BTCUSD
            f"{base}"           # Just BTC
        ]
        
        # Find files that match any of the symbol formats (case insensitive)
        matching_files = []
        for sym_format in symbol_formats:
            matching_files.extend([
                f for f in files 
                if sym_format.lower() in f.lower() and f.lower().endswith('.csv')
            ])
        
        if not matching_files:
            logger.warning(f"No CSV file found for symbol: {symbol}")
            return []
        
        # Use the first matching file
        csv_path = os.path.join(self.data_dir, matching_files[0])
        logger.info(f"Loading data from: {csv_path}")
        
        try:
            # Detect file encoding first
            with open(csv_path, 'rb') as f:
                raw_data = f.read(min(100000, os.path.getsize(csv_path)))
                encoding_result = chardet.detect(raw_data)
                encoding = encoding_result['encoding']
                
            logger.debug(f"Detected encoding: {encoding} with confidence {encoding_result['confidence']}")
            
            # Try to identify CSV dialect first
            with open(csv_path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
                sample = f.read(4096)
                dialect = csv.Sniffer().sniff(sample)
                
            # First try to detect format by examining a few rows
            format_info = self._detect_csv_format(csv_path, encoding)
            
            if format_info:
                return self._load_with_format_info(csv_path, symbol, format_info, encoding)
            else:
                # Fallback: Try with pandas and column detection
                return self._load_with_pandas(csv_path, symbol, encoding)
                
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _detect_csv_format(self, file_path: str, encoding: str) -> Optional[Dict]:
        """
        Detect the format of a CSV file by examining its contents.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            
        Returns:
            Dict or None: Format information or None if format cannot be determined
        """
        try:
            # Read first few rows to detect format
            with open(file_path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
                # Skip potential comments or headers at the beginning
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                        
                    # If line looks like a header (contains Date and common column names)
                    if ('date' in line.lower() or 'time' in line.lower()) and \
                       ('open' in line.lower() or 'close' in line.lower() or 'price' in line.lower()):
                        break
                
                # Peek at header line
                header_line = line.strip()
                header_parts = [p.strip() for p in header_line.split(',')]
                
                # Check for standard formats
                date_idx = None
                open_idx = None
                high_idx = None
                low_idx = None
                close_idx = None
                volume_idx = None
                
                for i, part in enumerate(header_parts):
                    part_lower = part.lower()
                    if 'date' in part_lower or 'time' in part_lower:
                        date_idx = i
                    elif 'open' in part_lower:
                        open_idx = i
                    elif 'high' in part_lower:
                        high_idx = i
                    elif 'low' in part_lower:
                        low_idx = i
                    elif 'close' in part_lower or 'price' in part_lower:
                        close_idx = i
                    elif 'volume' in part_lower or 'vol' in part_lower:
                        volume_idx = i
                
                # Examine some rows to determine delimiter
                sample_rows = []
                for _ in range(3):
                    line = f.readline()
                    if line:
                        sample_rows.append(line.strip())
                
                # Determine the date format from sample rows
                date_format = None
                if date_idx is not None and sample_rows:
                    date_samples = [row.split(',')[date_idx] for row in sample_rows if len(row.split(',')) > date_idx]
                    date_format = self._detect_date_format(date_samples)
                
                # Return format info if we found necessary columns
                if date_idx is not None and close_idx is not None:
                    return {
                        'date_idx': date_idx,
                        'open_idx': open_idx,
                        'high_idx': high_idx,
                        'low_idx': low_idx,
                        'close_idx': close_idx, 
                        'volume_idx': volume_idx,
                        'date_format': date_format,
                        'headers': header_parts
                    }
            
            # Couldn't determine format
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting CSV format: {e}")
            return None
    
    def _detect_date_format(self, date_samples: List[str]) -> Optional[str]:
        """
        Detect the date format from sample date strings.
        
        Args:
            date_samples: Sample date strings
            
        Returns:
            str or None: Detected date format or None
        """
        # Common date formats to check
        formats = [
            '%Y-%m-%d',            # 2023-01-15
            '%Y-%m-%d %H:%M:%S',   # 2023-01-15 14:30:45
            '%Y/%m/%d',            # 2023/01/15
            '%m/%d/%Y',            # 01/15/2023
            '%d/%m/%Y',            # 15/01/2023
            '%b %d, %Y',           # Jan 15, 2023
            '%d %b %Y',            # 15 Jan 2023
            '%d-%m-%Y',            # 15-01-2023
            '%d.%m.%Y',            # 15.01.2023
            '%m-%d-%Y',            # 01-15-2023
            '%Y%m%d'               # 20230115
        ]
        
        # Try each format on the samples
        for fmt in formats:
            try:
                # Try to parse all samples
                if all(self._try_parse_date(sample, fmt) for sample in date_samples):
                    return fmt
            except Exception:
                continue
                
        # If no standard format found, check for Unix timestamp
        try:
            if all(sample.isdigit() for sample in date_samples):
                if all(1000000000 < int(sample) < 9999999999 for sample in date_samples):  # Unix timestamp range check
                    return 'unix'
                elif all(1000000000000 < int(sample) < 9999999999999 for sample in date_samples):  # Unix timestamp in ms
                    return 'unix_ms'
        except Exception:
            pass
            
        return None
    
    def _try_parse_date(self, date_string: str, date_format: str) -> bool:
        """
        Try to parse a date string using a specific format.
        
        Args:
            date_string: Date string to parse
            date_format: Format to try
            
        Returns:
            bool: True if parsing successful
        """
        try:
            datetime.strptime(date_string.strip('"\''), date_format)
            return True
        except ValueError:
            # Try additional date formats for specific cases
            try:
                # Handle formats with milliseconds or microseconds
                if '.' in date_string and date_format == '%Y-%m-%d %H:%M:%S':
                    datetime.strptime(date_string.split('.')[0].strip('"\''), date_format)
                    return True
                # Handle timezone info
                elif '+' in date_string or 'Z' in date_string:
                    from dateutil import parser
                    parser.parse(date_string.strip('"\''))
                    return True
                return False
            except Exception:
                return False
    
    def _load_with_format_info(self, file_path: str, symbol: str, format_info: Dict, encoding: str) -> List[Dict]:
        """
        Load data using detected format information.
        
        Args:
            file_path: Path to CSV file
            symbol: Cryptocurrency symbol
            format_info: Format information
            encoding: File encoding
            
        Returns:
            List of data points
        """
        data_points = []
        exchange = self._extract_exchange_from_path(file_path)
        
        try:
            # Read the file line by line for better error handling
            with open(file_path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
                reader = csv.reader(f)
                
                # Skip header
                next(reader)
                
                # Process each row
                for row_idx, row in enumerate(reader):
                    try:
                        if len(row) <= max(filter(None, [
                            format_info['date_idx'], 
                            format_info['open_idx'], 
                            format_info['high_idx'],
                            format_info['low_idx'],
                            format_info['close_idx'],
                            format_info['volume_idx']
                        ])):
                            continue  # Skip rows with insufficient columns
                        
                        # Parse date
                        date_str = row[format_info['date_idx']].strip('"\'')
                        timestamp_iso = self._parse_date_with_format(date_str, format_info['date_format'])
                        
                        # Get price values, using fallbacks if necessary
                        close_price = self._parse_numeric_value(row[format_info['close_idx']]) if format_info['close_idx'] is not None else 0.0
                        
                        if format_info['open_idx'] is not None:
                            open_price = self._parse_numeric_value(row[format_info['open_idx']])
                        else:
                            open_price = close_price
                            
                        if format_info['high_idx'] is not None:
                            high_price = self._parse_numeric_value(row[format_info['high_idx']])
                        else:
                            high_price = close_price
                            
                        if format_info['low_idx'] is not None:
                            low_price = self._parse_numeric_value(row[format_info['low_idx']])
                        else:
                            low_price = close_price
                            
                        if format_info['volume_idx'] is not None:
                            volume = self._parse_numeric_value(row[format_info['volume_idx']])
                        else:
                            volume = 0.0
                        
                        # Skip rows with invalid data
                        if close_price <= 0 or high_price <= 0 or low_price <= 0:
                            continue
                            
                        # Create data point
                        data_point = {
                            "symbol": symbol,
                            "exchange": exchange,
                            "timestamp": timestamp_iso,
                            "open": open_price,
                            "high": high_price,
                            "low": low_price,
                            "close": close_price,
                            "volume": volume,
                            "interval": "1d"  # Assuming daily data
                        }
                        
                        data_points.append(data_point)
                    except Exception as e:
                        logger.debug(f"Error processing row {row_idx + 2}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
        
        # Sort by date
        data_points.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Loaded {len(data_points)} data points for {symbol}")
        return data_points
    
    def _load_with_pandas(self, file_path: str, symbol: str, encoding: str) -> List[Dict]:
        """
        Load data using pandas with column auto-detection.
        
        Args:
            file_path: Path to CSV file
            symbol: Cryptocurrency symbol
            encoding: File encoding
            
        Returns:
            List of data points
        """
        try:
            # Read CSV file with pandas
            df = pd.read_csv(file_path, encoding=encoding or 'utf-8', error_bad_lines=False)
            
            # Clean up column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Try to identify key columns
            date_col = self._find_column(df.columns, ['date', 'time', 'timestamp', 'datetime'])
            open_col = self._find_column(df.columns, ['open', 'open price', 'opening price', 'first'])
            high_col = self._find_column(df.columns, ['high', 'high price', 'highest price', 'max'])
            low_col = self._find_column(df.columns, ['low', 'low price', 'lowest price', 'min'])
            close_col = self._find_column(df.columns, ['close', 'close price', 'closing price', 'last', 'price'])
            volume_col = self._find_column(df.columns, ['volume', 'vol', 'vol.', 'volume(btc)', 'volume(eth)', 'volume(base)', 'volume(quote)'])
            
            # Check if we found the minimum required columns
            if not date_col or not close_col:
                logger.error(f"CSV format not recognized: missing required date or close columns. Columns: {list(df.columns)}")
                return []
            
            # Extract exchange name from file path
            exchange = self._extract_exchange_from_path(file_path)
            
            # Process data
            data_points = []
            for idx, row in df.iterrows():
                try:
                    # Handle date parsing
                    date_str = str(row[date_col])
                    
                    try:
                        # Try pandas datetime parsing
                        date_obj = pd.to_datetime(date_str)
                        date_iso = date_obj.isoformat()
                    except Exception:
                        logger.debug(f"Failed to parse date: {date_str}")
                        # Try multiple date formats
                        parsed = False
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                date_iso = date_obj.isoformat()
                                parsed = True
                                break
                            except ValueError:
                                continue
                                
                        if not parsed:
                            # If all parsing fails, use current date as fallback
                            date_iso = datetime.now().isoformat()
                    
                    # Parse price values with robust error handling
                    close_price = self._parse_numeric_value(row[close_col])
                    open_price = self._parse_numeric_value(row[open_col]) if open_col else close_price
                    high_price = self._parse_numeric_value(row[high_col]) if high_col else close_price
                    low_price = self._parse_numeric_value(row[low_col]) if low_col else close_price
                    volume = self._parse_numeric_value(row[volume_col]) if volume_col else 0.0
                    
                    # Skip rows with invalid data
                    if close_price <= 0 or high_price <= 0 or low_price <= 0:
                        continue
                    
                    # Create data point
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
                    logger.debug(f"Error processing row {idx}: {e}")
                    continue
            
            # Sort by date
            data_points.sort(key=lambda x: x["timestamp"])
            
            # Validate and clean data
            valid_data_points = self._validate_and_clean_data(data_points, symbol)
            
            # Check data quality
            quality_issues = self._check_data_quality(valid_data_points, symbol)
            if quality_issues:
                for issue in quality_issues:
                    logger.warning(f"Data quality issue for {symbol}: {issue}")
            
            logger.info(f"Loaded {len(valid_data_points)} valid data points for {symbol}")
            return valid_data_points
            
        except Exception as e:
            logger.error(f"Error loading with pandas: {e}")
            return []
    
    def _parse_date_with_format(self, date_str: str, date_format: str) -> str:
        """
        Parse a date string using the provided format.
        
        Args:
            date_str: Date string
            date_format: Date format
            
        Returns:
            str: ISO format date string
        """
        try:
            if date_format == 'unix':
                # Unix timestamp in seconds
                timestamp = int(date_str)
                date_obj = datetime.fromtimestamp(timestamp)
            elif date_format == 'unix_ms':
                # Unix timestamp in milliseconds
                timestamp = int(date_str) / 1000
                date_obj = datetime.fromtimestamp(timestamp)
            else:
                # Standard format
                date_obj = datetime.strptime(date_str, date_format)
                
            return date_obj.isoformat()
        except Exception as e:
            logger.debug(f"Error parsing date {date_str} with format {date_format}: {e}")
            # Fall back to current date
            return datetime.now().isoformat()
    
    def _parse_numeric_value(self, value) -> float:
        """
        Parse a numeric value with robust handling of various formats.
        
        Args:
            value: Input value (string, float, etc.)
            
        Returns:
            float: Parsed value
        """
        if value is None:
            return 0.0
            
        # If already a number, return it
        if isinstance(value, (int, float)):
            return float(value)
            
        # Convert to string and strip whitespace
        value_str = str(value).strip()
        
        # Remove any commas, percentage signs, currency symbols
        value_str = value_str.replace(',', '')
        value_str = value_str.replace('%', '')
        value_str = re.sub(r'[$€£¥]', '', value_str)
        
        # Handle empty or invalid values
        if not value_str or value_str.lower() in ['na', 'n/a', 'nan', 'null', 'none', '-']:
            return 0.0
            
        try:
            # Try direct conversion
            return float(value_str)
        except ValueError:
            # Check for suffix multipliers (K, M, B, T)
            for suffix, multiplier in self.volume_multipliers.items():
                if value_str.upper().endswith(suffix):
                    try:
                        # Extract the numeric part and multiply
                        numeric_part = float(value_str[:-len(suffix)])
                        return numeric_part * multiplier
                    except ValueError:
                        # If parsing fails, continue to next check
                        pass
            
            # Try extracting numeric part using regex
            numeric_match = re.search(r'([-+]?\d*\.?\d+)', value_str)
            if numeric_match:
                try:
                    return float(numeric_match.group(1))
                except ValueError:
                    pass
            
            # If all else fails, return 0
            logger.debug(f"Could not parse numeric value: {value}")
            return 0.0
    
    def _find_column(self, columns: List[str], possible_names: List[str]) -> Optional[str]:
        """
        Find a column from a list of possible column names (case-insensitive).
        
        Args:
            columns: Available columns
            possible_names: Possible column names to look for
            
        Returns:
            str or None: Found column name or None
        """
        # First try exact matches (lowercased)
        columns_lower = [col.lower() for col in columns]
        for name in possible_names:
            if name.lower() in columns_lower:
                idx = columns_lower.index(name.lower())
                return columns[idx]
        
        # Then try partial matches
        for name in possible_names:
            for col in columns:
                if name.lower() in col.lower():
                    return col
        
        return None
    
    def _split_symbol(self, symbol: str) -> tuple:
        """
        Split a symbol into base and quote parts.
        
        Args:
            symbol: Cryptocurrency symbol
            
        Returns:
            tuple: (base, quote)
        """
        # Try to intelligently split the symbol
        common_quotes = ["USD", "USDT", "USDC", "BTC", "ETH"]
        
        # Clean up and standardize
        symbol = symbol.upper().strip()
        
        # Look for common quote currencies
        for quote in common_quotes:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base, quote
        
        # If no common quote found, try standard length
        if len(symbol) > 3:
            # Assume last 3 characters are the quote currency
            return symbol[:-3], symbol[-3:]
        
        # Fall back to just the symbol with USD as quote
        return symbol, "USD"
    
    def _validate_and_clean_data(self, data_points: List[Dict], symbol: str) -> List[Dict]:
        """
        Validate and clean loaded data points.
        
        Args:
            data_points: List of data points
            symbol: Cryptocurrency symbol
            
        Returns:
            List[Dict]: Validated and cleaned data points
        """
        if not data_points:
            return []
            
        valid_points = []
        duplicates = set()
        outliers_removed = 0
        
        # Get median values for outlier detection
        if len(data_points) > 5:
            closes = [point["close"] for point in data_points]
            volumes = [point["volume"] for point in data_points]
            
            median_close = np.median(closes)
            median_volume = np.median(volumes)
            
            # IQR for outlier detection
            close_iqr = np.percentile(closes, 75) - np.percentile(closes, 25)
            close_upper_bound = median_close + (close_iqr * 5)  # Allow higher outliers
            close_lower_bound = max(0, median_close - (close_iqr * 3))
            
            # More permissive for volume
            volume_iqr = np.percentile(volumes, 75) - np.percentile(volumes, 25)
            volume_upper_bound = median_volume + (volume_iqr * 10)
        else:
            # Not enough data for outlier detection
            close_upper_bound = float('inf')
            close_lower_bound = 0
            volume_upper_bound = float('inf')
        
        # Process each data point
        for point in data_points:
            # Skip duplicates
            timestamp = point["timestamp"]
            if timestamp in duplicates:
                continue
            duplicates.add(timestamp)
            
            # Check for price outliers (extremely high or negative values)
            if point["close"] > close_upper_bound or point["close"] < close_lower_bound:
                outliers_removed += 1
                continue
                
            # Check for volume outliers
            if point["volume"] > volume_upper_bound:
                # Keep the price data but cap the volume
                point["volume"] = volume_upper_bound
            
            # Ensure OHLC values are consistent
            # High should be the highest value
            point["high"] = max(point["open"], point["high"], point["low"], point["close"])
            # Low should be the lowest value
            point["low"] = min(point["open"], point["high"], point["low"], point["close"])
            # Volume should never be negative
            point["volume"] = max(0, point["volume"])
            
            # Ensure symbol is standardized
            if not point["symbol"].endswith(("USD", "USDT")):
                point["symbol"] = symbol
            
            valid_points.append(point)
        
        # Log validation results
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outliers from {symbol} data")
            
        return valid_points
    
    def _check_data_quality(self, data_points: List[Dict], symbol: str) -> List[str]:
        """
        Check data quality and return issues.
        
        Args:
            data_points: List of data points
            symbol: Cryptocurrency symbol
            
        Returns:
            List[str]: List of quality issues
        """
        issues = []
        
        if not data_points:
            issues.append("No valid data points found")
            return issues
            
        # Check for sufficient data
        if len(data_points) < 10:
            issues.append(f"Limited data: only {len(data_points)} points available")
            
        # Check date continuity (looking for gaps)
        if len(data_points) > 1:
            dates = [datetime.fromisoformat(point["timestamp"].replace('Z', '+00:00') if 'Z' in point["timestamp"] else point["timestamp"]) 
                    for point in data_points]
            
            # Sort dates
            dates.sort()
            
            # Look for large gaps
            gaps = []
            for i in range(1, len(dates)):
                delta = (dates[i] - dates[i-1]).days
                if delta > 7:  # Gap of more than 7 days
                    gaps.append(f"{dates[i-1].date()} to {dates[i].date()} ({delta} days)")
                    
            if len(gaps) > 0:
                if len(gaps) <= 3:
                    issues.append(f"Date gaps found: {', '.join(gaps)}")
                else:
                    issues.append(f"Multiple date gaps found: {len(gaps)} gaps")
                    
        # Check for time range
        if len(data_points) > 0:
            start_date = datetime.fromisoformat(data_points[0]["timestamp"].replace('Z', '+00:00') if 'Z' in data_points[0]["timestamp"] else data_points[0]["timestamp"])
            end_date = datetime.fromisoformat(data_points[-1]["timestamp"].replace('Z', '+00:00') if 'Z' in data_points[-1]["timestamp"] else data_points[-1]["timestamp"])
            
            # Calculate time range in days
            time_range_days = (end_date - start_date).days
            
            if time_range_days < 30:
                issues.append(f"Short time period: {time_range_days} days")
                
            # Check if data is recent
            days_since_last = (datetime.now() - end_date).days
            if days_since_last > 30:
                issues.append(f"Data not recent: last date is {days_since_last} days ago")
                
        # Check for unusual values
        price_zeros = sum(1 for point in data_points if point["close"] == 0)
        volume_zeros = sum(1 for point in data_points if point["volume"] == 0)
        
        if price_zeros > 0:
            issues.append(f"Zero prices found: {price_zeros} instances")
            
        if volume_zeros > len(data_points) * 0.5:
            issues.append(f"Many zero volumes: {volume_zeros} instances ({volume_zeros/len(data_points)*100:.1f}%)")
            
        return issues
    
    def _extract_exchange_from_path(self, file_path: str) -> str:
        """
        Extract exchange name from file path.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            str: Exchange name
        """
        filename = os.path.basename(file_path)
        
        # Try to find exchange name in the filename
        for exchange_key, exchange_name in self.exchange_mapping.items():
            if exchange_key.lower() in filename.lower():
                return exchange_name
        
        # Look for other exchange identifiers in the path
        for part in file_path.lower().split(os.sep):
            for exchange_key, exchange_name in self.exchange_mapping.items():
                if exchange_key.lower() in part.lower():
                    return exchange_name
        
        # Try to extract from file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(4096)  # Read a sample
                for exchange_key, exchange_name in self.exchange_mapping.items():
                    if exchange_key.lower() in content.lower():
                        return exchange_name
        except Exception:
            pass
            
                    # If no exchange found, extract from filename
        parts = filename.split('_')
        if len(parts) > 1:
            # Try to find exchange name in parts
            for part in parts:
                if len(part) > 2 and part.lower() not in ['btc', 'eth', 'usd', 'ltc']:
                    return part.capitalize()
                    
        # Fall back to "Unknown Exchange"
        return "Unknown Exchange"
    
    def load_multiple_symbols(self, symbols: List[str], max_retries: int = 2) -> Dict[str, List[Dict]]:
        """
        Load historical data for multiple symbols with retry logic.
        
        Args:
            symbols: List of cryptocurrency symbols
            max_retries: Maximum number of retry attempts per symbol
            
        Returns:
            Dict mapping symbols to their data points
        """
        results = {}
        failed_symbols = []
        
        logger.info(f"Loading data for {len(symbols)} symbols")
        
        for symbol in symbols:
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    data = self.load_historical_data(symbol)
                    
                    if data and len(data) > 0:
                        results[symbol] = data
                        logger.info(f"Successfully loaded {len(data)} points for {symbol}")
                        break
                    else:
                        retry_count += 1
                        logger.warning(f"Attempt {retry_count}/{max_retries + 1} failed for {symbol}")
                        
                        if retry_count <= max_retries:
                            # Try alternative symbol format
                            if symbol.endswith("USD"):
                                symbol = symbol.replace("USD", "USDT")
                            elif symbol.endswith("USDT"):
                                symbol = symbol.replace("USDT", "USD")
                        else:
                            failed_symbols.append(symbol)
                            
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error loading {symbol} (attempt {retry_count}/{max_retries + 1}): {e}")
                    
                    if retry_count > max_retries:
                        failed_symbols.append(symbol)
                        logger.error(f"Failed to load {symbol} after {max_retries + 1} attempts")
        
        # Log summary
        logger.info(f"Loaded data for {len(results)}/{len(symbols)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
            
        return results