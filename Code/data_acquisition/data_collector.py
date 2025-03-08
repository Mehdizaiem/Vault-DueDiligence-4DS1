# File path: Code/data_acquisition/data_collector.py
import os
import sys
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataCollector:
    """
    Unified data collector for all crypto data sources including:
    - Market data from exchanges (Binance, CoinGecko, etc.)
    - News from crypto news sources
    - Blockchain data from on-chain analytics
    """
    
    def __init__(self):
        """Initialize the data collector with API keys from environment variables"""
        self.api_keys = {
            "binance": os.getenv("BINANCE_API_KEY"),
            "coingecko": os.getenv("COINGECKO_API_KEY"),
            "coinmarketcap": os.getenv("CMC_API_KEY"),
            "cryptocompare": os.getenv("CRYPTOCOMPARE_API_KEY"),
            "etherscan": os.getenv("ETHERSCAN_API_KEY"),
        }
        
        # Standard headers to mimic browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
        
        # Configure paths for data storage
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        os.makedirs(self.data_dir, exist_ok=True)

    def _make_request(self, url: str, params: Dict = None, headers: Dict = None, retries: int = 3) -> Dict:
        """
        Make API request with robust error handling and retry logic.
        
        Args:
            url (str): API endpoint URL
            params (Dict, optional): Request parameters
            headers (Dict, optional): Request headers
            retries (int): Number of retry attempts
            
        Returns:
            Dict: Response data or error object
        """
        if params is None:
            params = {}
            
        if headers is None:
            headers = self.headers
            
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                # Check HTTP status
                if response.status_code != 200:
                    logger.warning(f"HTTP error {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return {'error': f"HTTP error: {response.status_code}"}
                
                # Parse response
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}/{retries})")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return {'error': 'Request timeout'}
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return {'error': str(e)}
                    
            except ValueError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return {'error': f'JSON parsing error: {str(e)}'}

    def fetch_market_data(self, symbols: List[str] = None) -> List[Dict]:
        """
        Fetch current market data for specified crypto symbols.
        
        Args:
            symbols (List[str], optional): List of symbols to fetch data for
                                          Default: ["BTC", "ETH", "SOL", "BNB", "ADA"]
        
        Returns:
            List[Dict]: Market data for each symbol
        """
        if symbols is None:
            symbols = ["BTC", "ETH", "SOL", "BNB", "ADA"]
            
        logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        all_market_data = []
        
        # Try CoinGecko first (it has good free tier limits)
        try:
            # Convert symbols to coingecko IDs
            symbol_ids = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "BNB": "binancecoin",
                "ADA": "cardano"
            }
            
            # Get IDs for requested symbols
            requested_ids = [symbol_ids.get(symbol.upper(), symbol.lower()) for symbol in symbols]
            ids_param = ",".join(requested_ids)
            
            # Prepare CoinGecko API request
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": ids_param,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            # Add API key if available
            if self.api_keys["coingecko"]:
                params["x_cg_pro_api_key"] = self.api_keys["coingecko"]
                
            # Make request
            response = self._make_request(url, params)
            
            # Process response
            if not response or 'error' in response:
                logger.error(f"Error fetching data from CoinGecko: {response.get('error', 'Unknown error')}")
            else:
                for coin_id, data in response.items():
                    # Determine symbol from coin_id
                    symbol = next((sym for sym, id_val in symbol_ids.items() if id_val == coin_id), coin_id.upper())
                    
                    market_data = {
                        "symbol": f"{symbol}USDT",  # Format as BTCUSDT for consistency
                        "source": "coingecko",
                        "price": data.get("usd", 0),
                        "market_cap": data.get("usd_market_cap", 0),
                        "volume_24h": data.get("usd_24h_vol", 0),
                        "price_change_24h": data.get("usd_24h_change", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    all_market_data.append(market_data)
                    
                logger.info(f"Fetched {len(all_market_data)} records from CoinGecko")
                    
        except Exception as e:
            logger.error(f"Error in CoinGecko API: {str(e)}")
            
        # If CoinGecko failed or returned incomplete data, try Binance
        if len(all_market_data) < len(symbols):
            try:
                # First get ticker data
                url = "https://api.binance.com/api/v3/ticker/24hr"
                response = self._make_request(url)
                
                if not response or 'error' in response:
                    logger.error(f"Error fetching data from Binance: {response.get('error', 'Unknown error')}")
                else:
                    # Filter for requested symbols
                    symbol_tickers = [f"{s.upper()}USDT" for s in symbols]
                    
                    for ticker in response:
                        if ticker.get("symbol") in symbol_tickers:
                            market_data = {
                                "symbol": ticker.get("symbol"),
                                "source": "binance",
                                "price": float(ticker.get("lastPrice", 0)),
                                "market_cap": 0,  # Binance doesn't provide market cap
                                "volume_24h": float(ticker.get("volume", 0)) * float(ticker.get("lastPrice", 0)),
                                "price_change_24h": float(ticker.get("priceChangePercent", 0)),
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            # Check if we already have data for this symbol
                            existing_symbols = [item["symbol"] for item in all_market_data]
                            if market_data["symbol"] not in existing_symbols:
                                all_market_data.append(market_data)
                    
                    logger.info(f"Fetched {len(all_market_data)} total records after Binance")
                    
            except Exception as e:
                logger.error(f"Error in Binance API: {str(e)}")
                
        return all_market_data

    def fetch_historical_data(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict]:
        """
        Fetch historical price data for a specific symbol.
        
        Args:
            symbol (str): Symbol to fetch data for (e.g., "BTCUSDT")
            interval (str): Time interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit (int): Number of data points to fetch
            
        Returns:
            List[Dict]: Historical price data
        """
        logger.info(f"Fetching historical data for {symbol} at {interval} interval")
        
        # Format symbol
        symbol = symbol.upper()
        if not symbol.endswith("USDT") and not symbol.endswith("USD"):
            symbol = f"{symbol}USDT"
            
        # Use Binance for historical data
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = self._make_request(url, params)
        
        if 'error' in response:
            logger.error(f"Error fetching historical data: {response['error']}")
            return []
            
        # Binance returns klines as an array with specific positional values
        historical_data = []
        for kline in response:
            try:
                timestamp = datetime.fromtimestamp(kline[0] / 1000).isoformat()
                data_point = {
                    "symbol": symbol,
                    "exchange": "binance",
                    "timestamp": timestamp,
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "interval": interval
                }
                historical_data.append(data_point)
            except (IndexError, ValueError) as e:
                logger.error(f"Error parsing kline data: {e}")
                continue
                
        logger.info(f"Fetched {len(historical_data)} historical data points for {symbol}")
        return historical_data

    def fetch_news(self, limit_per_source: int = 10) -> List[Dict]:
        """
        Fetch news articles from crypto news sources.
        
        Args:
            limit_per_source (int): Number of articles to fetch per source
            
        Returns:
            List[Dict]: News articles with metadata
        """
        logger.info(f"Fetching crypto news with limit {limit_per_source} per source")
        
        # Import the news scraper
        try:
            from Code.data_acquisition.blockchain_collectors.news_scraper import CryptoNewsScraper
            scraper = CryptoNewsScraper()
            
            # Run the scraper
            news_df = scraper.run(limit_per_source=limit_per_source)
            
            # Convert to list of dictionaries
            if news_df is not None and not news_df.empty:
                news_list = news_df.to_dict('records')
                logger.info(f"Fetched {len(news_list)} news articles")
                return news_list
            else:
                logger.warning("No news articles found")
                return []
                
        except ImportError:
            logger.error("CryptoNewsScraper not found. Make sure news_scraper.py is available.")
            return []
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

    def fetch_onchain_data(self, address: str, blockchain: str = "ethereum") -> Dict:
        """
        Fetch on-chain data for a wallet or contract address.
        
        Args:
            address (str): Wallet or contract address
            blockchain (str): Blockchain name (ethereum, binance, etc.)
            
        Returns:
            Dict: On-chain analytics data
        """
        logger.info(f"Fetching on-chain data for {address} on {blockchain}")
        
        if blockchain.lower() == "ethereum":
            return self._fetch_ethereum_data(address)
        else:
            logger.warning(f"Blockchain {blockchain} not supported yet")
            return {"error": f"Blockchain {blockchain} not supported yet"}

    def _fetch_ethereum_data(self, address: str) -> Dict:
        """Fetch Ethereum on-chain data using Etherscan API"""
        if not self.api_keys["etherscan"]:
            logger.warning("No Etherscan API key found")
            return {"error": "No Etherscan API key available"}
            
        # Base parameters for all requests
        base_params = {
            "module": "account",
            "address": address,
            "apikey": self.api_keys["etherscan"]
        }
        
        # Fetch ETH balance
        balance_params = {**base_params, "action": "balance", "tag": "latest"}
        balance_response = self._make_request("https://api.etherscan.io/api", balance_params)
        
        # Fetch normal transactions
        tx_params = {
            **base_params, 
            "action": "txlist", 
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": "100",
            "sort": "desc"
        }
        tx_response = self._make_request("https://api.etherscan.io/api", tx_params)
        
        # Fetch token transactions
        token_tx_params = {
            **base_params, 
            "action": "tokentx", 
            "startblock": "0",
            "endblock": "99999999",
            "page": "1",
            "offset": "100",
            "sort": "desc"
        }
        token_tx_response = self._make_request("https://api.etherscan.io/api", token_tx_params)
        
        # Process and combine data
        balance_eth = 0
        tx_count = 0
        token_tx_count = 0
        first_tx_time = None
        last_tx_time = None
        unique_interactions = set()
        tokens = {}
        
        # Process balance
        if balance_response and balance_response.get("status") == "1":
            balance_wei = int(balance_response.get("result", "0"))
            balance_eth = balance_wei / 1e18
            
        # Process transactions
        if tx_response and tx_response.get("status") == "1":
            transactions = tx_response.get("result", [])
            tx_count = len(transactions)
            
            # Extract timestamps and interactions
            for tx in transactions:
                timestamp = int(tx.get("timeStamp", 0))
                
                if first_tx_time is None or timestamp < first_tx_time:
                    first_tx_time = timestamp
                    
                if last_tx_time is None or timestamp > last_tx_time:
                    last_tx_time = timestamp
                    
                # Add to unique interactions
                if tx.get("from", "").lower() != address.lower():
                    unique_interactions.add(tx.get("from", "").lower())
                    
                if tx.get("to", "").lower() != address.lower():
                    unique_interactions.add(tx.get("to", "").lower())
        
        # Process token transactions
        if token_tx_response and token_tx_response.get("status") == "1":
            token_transactions = token_tx_response.get("result", [])
            token_tx_count = len(token_transactions)
            
            # Extract token information
            for tx in token_transactions:
                symbol = tx.get("tokenSymbol", "UNKNOWN")
                name = tx.get("tokenName", "Unknown Token")
                token_address = tx.get("contractAddress", "")
                
                if symbol not in tokens:
                    tokens[symbol] = {
                        "name": name,
                        "address": token_address,
                        "transfers": 0
                    }
                
                tokens[symbol]["transfers"] += 1
        
        # Calculate account age in days
        account_age_days = 0
        if first_tx_time and last_tx_time:
            account_age_days = (last_tx_time - first_tx_time) / 86400  # seconds to days
        
        # Calculate risk score (example logic)
        risk_score = 50  # Default medium risk
        
        # Age factor (newer accounts have higher risk)
        if account_age_days < 30:
            risk_score += 20
        elif account_age_days < 180:
            risk_score += 10
        elif account_age_days > 365:
            risk_score -= 10
        
        # Transaction count (very low or very high can be risk signals)
        if tx_count < 5:
            risk_score += 15  # Very new account
        elif tx_count > 1000:
            risk_score += 5   # Unusually high activity
        
        # Network diversity (limited interactions suggest isolation)
        if len(unique_interactions) < 3:
            risk_score += 15
        elif len(unique_interactions) > 50:
            risk_score -= 10  # Wide network is generally lower risk
        
        # Token diversity
        if len(tokens) > 10:
            risk_score -= 5   # More token types suggests established user
        
        # Cap the score between 0-100
        risk_score = max(0, min(100, risk_score))
        
        # Determine risk level
        if risk_score < 20:
            risk_level = "Very Low"
        elif risk_score < 40:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        elif risk_score < 80:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Format first and last activity timestamps
        first_activity = datetime.fromtimestamp(first_tx_time).isoformat() if first_tx_time else None
        last_activity = datetime.fromtimestamp(last_tx_time).isoformat() if last_tx_time else None
        
        # Prepare final result
        result = {
            "address": address,
            "blockchain": "ethereum",
            "entity_type": "wallet",  # Simplified for now
            "transaction_count": tx_count,
            "token_transaction_count": token_tx_count,
            "balance": balance_eth,
            "first_activity": first_activity,
            "last_activity": last_activity,
            "active_days": account_age_days,
            "unique_interactions": len(unique_interactions),
            "tokens": list(tokens.keys()),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return result

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Test market data collection
    market_data = collector.fetch_market_data(["BTC", "ETH"])
    print(f"Fetched {len(market_data)} market data records")
    
    # Test historical data collection
    historical_data = collector.fetch_historical_data("BTCUSDT", interval="1d", limit=10)
    print(f"Fetched {len(historical_data)} historical data points")
    
    # Test on-chain data collection
    # Using a known Ethereum address (Binance hot wallet)
    onchain_data = collector.fetch_onchain_data("0x28c6c06298d514db089934071355e5743bf21d60")
    print(f"Risk score for wallet: {onchain_data.get('risk_score')}")