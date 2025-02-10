import os
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
dotenv_path = ".env.local" if os.path.exists(".env.local") else ".env"
load_dotenv(dotenv_path=dotenv_path)


class APICollector:
    def __init__(self):
        """Initialize API keys from environment variables."""
        self.api_keys = {
            "binance": os.getenv("BINANCE_API_KEY"),
            "coingecko": os.getenv("COINGECKO_API_KEY"),
            "coinmarketcap": os.getenv("CMC_API_KEY"),
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
        }

    def _make_request(self, url: str, headers: Dict = None) -> Dict:
        """Helper function to make API requests with error handling."""
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError("Empty response received")

            return data
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for {url}: {e}")
        except ValueError as ve:
            print(f"[ERROR] Invalid response from {url}: {ve}")

        return {}

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flattens nested dictionaries and handles lists."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):  # Handle list of dictionaries
                    for i, sub_dict in enumerate(v):
                        items.extend(self._flatten_dict(sub_dict, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((new_key, str(v)))  # Convert list to string for storage
            else:
                items.append((new_key, v))
        return dict(items)

    def fetch_binance(self) -> List[Dict[str, Any]]:
        """Fetches ticker price data from Binance."""
        url = "https://api.binance.com/api/v3/ticker/price"
        data = self._make_request(url)
        if isinstance(data, list):
            return [{"source": "binance", "data": item} for item in data]
        return []

    def fetch_coingecko(self) -> List[Dict[str, Any]]:
        """Fetches price data from CoinGecko."""
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"
        data = self._make_request(url)
        if data:
            return [{"source": "coingecko", "data": {"coin": coin, "price_usd": details.get("usd", None)}}
                    for coin, details in data.items()]
        return []

    def fetch_coinmarketcap(self) -> List[Dict[str, Any]]:
        """Fetches cryptocurrency listings from CoinMarketCap."""
        api_key = self.api_keys["coinmarketcap"]
        if not api_key:
            print("[WARNING] CoinMarketCap API key not found. Skipping.")
            return []

        headers = {'X-CMC_PRO_API_KEY': api_key}
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        response = self._make_request(url, headers)

        if "data" in response:
            formatted_data = []
            for item in response["data"]:
                flattened_data = self._flatten_dict(item)
                flattened_data["cmc_id"] = flattened_data.pop("id", None)  # Rename ID to avoid conflicts
                formatted_data.append({"source": "coinmarketcap", "data": flattened_data})
            return formatted_data
        return []

    def fetch_cryptocompare(self) -> List[Dict[str, Any]]:
        """Fetches price data from CryptoCompare."""
        api_key = self.api_keys["cryptocompare"]
        if not api_key:
            print("[WARNING] CryptoCompare API key not found. Skipping.")
            return []

        url = f"https://min-api.cryptocompare.com/data/pricemulti?fsyms=BTC,ETH&tsyms=USD&api_key={api_key}"
        data = self._make_request(url)

        if data:
            return [{"source": "cryptocompare", "data": {"symbol": symbol, **details}}
                    for symbol, details in data.items()]
        return []


def fetch_all() -> List[Dict[str, Any]]:
    """Fetches data from all sources and returns a combined list."""
    collector = APICollector()
    all_data = []

    collectors = [
        collector.fetch_binance,
        collector.fetch_coingecko,
        collector.fetch_coinmarketcap,
        collector.fetch_cryptocompare
    ]

    for fetch_func in collectors:
        data = fetch_func()
        if data:
            all_data.extend(data)

    return all_data


if __name__ == "__main__":
    crypto_data = fetch_all()
    print(f"Fetched {len(crypto_data)} records from APIs.")
