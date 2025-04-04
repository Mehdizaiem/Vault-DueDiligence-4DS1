import os
import requests
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
project_root = os.path.abspath("C:/Users/asus/Documents/4DS1/Semester2/PIDS/VAULT_Project/Vault-DueDiligence-4DS1")
dotenv_path = os.path.join(project_root, ".env.local")

if os.path.exists(dotenv_path):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(dotenv_path))
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f"Environment file not found at {dotenv_path}")

class APICollector:
    """
    Collects cryptocurrency market data from multiple APIs
    with robust error handling and logging.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize API collector with configurable timeout.
        
        Args:
            timeout (int): Request timeout in seconds
        """
        self.timeout = timeout
        self.api_keys = {
            "binance": os.getenv("BINANCE_API_KEY"),
            "coingecko": os.getenv("COINGECKO_API_KEY"),
            "coinmarketcap": os.getenv("CMC_API_KEY"),
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
        }
        
        # Standard headers to mimic browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }

    def _make_request(self, url: str, additional_headers: Dict = None) -> Dict:
        """
        Make a robust API request with comprehensive error handling.
        
        Args:
            url (str): API endpoint URL
            additional_headers (Dict, optional): Additional request headers
        
        Returns:
            Dict: Parsed JSON response or empty dict
        """
        try:
            # Merge default and additional headers
            request_headers = {**self.headers, **(additional_headers or {})}
            
            response = requests.get(
                url, 
                headers=request_headers, 
                timeout=self.timeout
            )
            
            # Raise exception for bad status codes
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                logger.warning(f"Empty response received from {url}")
                return {}
            
            return data
        
        except requests.exceptions.Timeout:
            logger.error(f"Request to {url} timed out after {self.timeout} seconds")
        except requests.exceptions.TooManyRedirects:
            logger.error(f"Too many redirects when accessing {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
        except ValueError as ve:
            logger.error(f"Invalid JSON response from {url}: {ve}")
        
        return {}

    def _flatten_dict(self, 
                     d: Dict, 
                     parent_key: str = '', 
                     sep: str = '_', 
                     max_depth: int = 3) -> Dict:
        """
        Recursively flatten nested dictionaries with depth limit.
        
        Args:
            d (Dict): Dictionary to flatten
            parent_key (str): Parent key for nested items
            sep (str): Separator for nested keys
            max_depth (int): Maximum nesting depth
        
        Returns:
            Dict: Flattened dictionary
        """
        if max_depth <= 0:
            return d
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(
                        v, 
                        new_key, 
                        sep=sep, 
                        max_depth=max_depth-1
                    ).items()
                )
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    for i, sub_dict in enumerate(v):
                        items.extend(
                            self._flatten_dict(
                                sub_dict, 
                                f"{new_key}{sep}{i}", 
                                sep=sep, 
                                max_depth=max_depth-1
                            ).items()
                        )
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)

    def fetch_binance(self) -> List[Dict[str, Any]]:
        """
        Fetch comprehensive 24-hour ticker data from Binance.
        
        Returns:
            List[Dict]: Formatted Binance market data
        """
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = self._make_request(url)
        
        if not isinstance(data, list):
            logger.warning("Binance API did not return a list of tickers")
            return []
        
        formatted_data = []
        for item in data:
            try:
                # Filter out tokens with zero or near-zero volume
                if float(item.get('volume', 0)) > 0:
                    formatted_data.append({
                        "source": "binance",
                        "data": {
                            "symbol": item.get('symbol', 'UNKNOWN'),
                            "price_usd": float(item.get('lastPrice', 0)),
                            "market_cap_usd": float(item.get('quoteVolume', 0)),
                            "volume_24h_usd": float(item.get('volume', 0)),
                            "price_change_24h": float(item.get('priceChangePercent', 0))
                        }
                    })
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing Binance ticker {item.get('symbol')}: {e}")
        
        logger.info(f"Fetched {len(formatted_data)} valid Binance tickers")
        return formatted_data

    def fetch_coingecko(self) -> List[Dict[str, Any]]:
        """
        Fetch price data from CoinGecko with expanded coin list.
        
        Returns:
            List[Dict]: Formatted CoinGecko market data
        """
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,cardano,binancecoin,solana&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
        data = self._make_request(url)
        
        if not data:
            logger.warning("No data retrieved from CoinGecko")
            return []
        
        formatted_data = []
        for coin, details in data.items():
            try:
                formatted_data.append({
                    "source": "coingecko",
                    "data": {
                        "coin": coin,
                        "price_usd": details.get("usd"),
                        "market_cap_usd": details.get("usd_market_cap", 0),
                        "volume_24h_usd": details.get("usd_24h_vol", 0),
                        "price_change_24h": details.get("usd_24h_change", 0)
                    }
                })
            except Exception as e:
                logger.error(f"Error processing CoinGecko data for {coin}: {e}")
        
        logger.info(f"Fetched {len(formatted_data)} valid CoinGecko coins")
        return formatted_data

    def fetch_coinmarketcap(self) -> List[Dict[str, Any]]:
        """
        Fetch cryptocurrency listings from CoinMarketCap.
        
        Returns:
            List[Dict]: Formatted CoinMarketCap market data
        """
        api_key = self.api_keys["coinmarketcap"]
        if not api_key:
            logger.warning("CoinMarketCap API key not found. Skipping.")
            return []

        headers = {'X-CMC_PRO_API_KEY': api_key}
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?limit=100&convert=USD"
        response = self._make_request(url, additional_headers=headers)

        if not response or "data" not in response:
            logger.warning("No data retrieved from CoinMarketCap")
            return []

        formatted_data = []
        for item in response["data"]:
            try:
                flattened_data = self._flatten_dict(item)
                formatted_data.append({
                    "source": "coinmarketcap",
                    "data": {
                        "symbol": item.get('symbol', 'UNKNOWN'),
                        "price_usd": item.get('quote', {}).get('USD', {}).get('price', 0),
                        "market_cap_usd": item.get('quote', {}).get('USD', {}).get('market_cap', 0),
                        "volume_24h_usd": item.get('quote', {}).get('USD', {}).get('volume_24h', 0),
                        "price_change_24h": item.get('quote', {}).get('USD', {}).get('percent_change_24h', 0)
                    }
                })
            except Exception as e:
                logger.error(f"Error processing CoinMarketCap data: {e}")
        
        logger.info(f"Fetched {len(formatted_data)} valid CoinMarketCap listings")
        return formatted_data

    def fetch_cryptocompare(self) -> List[Dict[str, Any]]:
        """
        Fetch comprehensive price data from CryptoCompare.
        
        Returns:
            List[Dict]: Formatted CryptoCompare market data
        """
        api_key = self.api_keys["cryptocompare"]
        if not api_key:
            logger.warning("CryptoCompare API key not found. Skipping.")
            return []

        symbols = "BTC,ETH,ADA,BNB,SOL"
        url = f"https://min-api.cryptocompare.com/data/pricemulti?fsyms={symbols}&tsyms=USD&api_key={api_key}&extraParams=CryptoETLPipeline"
        
        # Fetch price data
        price_data = self._make_request(url)
        
        # Fetch additional market data
        market_url = f"https://min-api.cryptocompare.com/data/top/mktcap?limit=10&tsym=USD&api_key={api_key}"
        market_data = self._make_request(market_url)

        if not price_data or not market_data:
            logger.warning("Incomplete data retrieved from CryptoCompare")
            return []

        formatted_data = []
        for symbol, details in price_data.items():
            try:
                # Find corresponding market data
                market_info = next(
                    (item for item in market_data.get('Data', []) if item.get('CoinInfo', {}).get('Name') == symbol), 
                    {}
                )
                
                formatted_data.append({
                    "source": "cryptocompare",
                    "data": {
                        "symbol": symbol,
                        "price_usd": details.get('USD', 0),
                        "market_cap_usd": market_info.get('ConversionInfo', {}).get('TotalVolume24H', 0),
                        "volume_24h_usd": market_info.get('ConversionInfo', {}).get('Volume24HourTo', 0),
                        "price_change_24h": 0  # CryptoCompare API doesn't provide this directly
                    }
                })
            except Exception as e:
                logger.error(f"Error processing CryptoCompare data for {symbol}: {e}")
        
        logger.info(f"Fetched {len(formatted_data)} valid CryptoCompare listings")
        return formatted_data

def fetch_all() -> List[Dict[str, Any]]:
    """
    Fetch data from all configured cryptocurrency APIs.
    
    Returns:
        List[Dict]: Combined market data from multiple sources
    """
    collector = APICollector()
    collector_methods = [
        collector.fetch_binance,
        collector.fetch_coingecko,
        collector.fetch_coinmarketcap,
        collector.fetch_cryptocompare
    ]
    
    all_data = []
    for method in collector_methods:
        try:
            data = method()
            if data:
                all_data.extend(data)
        except Exception as e:
            logger.error(f"Error in {method.__name__}: {e}")
    
    logger.info(f"Total records fetched: {len(all_data)}")
    return all_data

if __name__ == "__main__":
    crypto_data = fetch_all()
    print(f"Fetched {len(crypto_data)} records from APIs.")