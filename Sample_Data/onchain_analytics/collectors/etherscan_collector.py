# Updated etherscan_collector.py
import os
import requests
import logging
import time
from typing import Dict, List, Any
from dotenv import load_dotenv
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class EtherscanCollector(BaseCollector):
    """Collector for Etherscan API data."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.etherscan.io/api"
        
        # If API key not provided, try loading from the correct .env.local path
        if not self.api_key:
            try:
                # Get path to root directory (two levels up from current file)
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                env_path = os.path.join(root_dir, '.env.local')
                
                if os.path.exists(env_path):
                    logger.info(f"Loading API key from {env_path}")
                    load_dotenv(dotenv_path=env_path)
                else:
                    logger.warning(f".env.local not found at {env_path}, trying default .env")
                    load_dotenv()
                
                self.api_key = os.getenv("ETHERSCAN_API_KEY")
            except Exception as e:
                logger.error(f"Error loading API key: {str(e)}")
            
        # Verify API key is available
        if not self.api_key:
            logger.warning("No Etherscan API key provided. API calls may fail.")
    
    def _make_request(self, endpoint: str, params: Dict = None, headers: Dict = None, retries: int = 2) -> Dict:
        """Make request to Etherscan API with error handling and retries."""
        if not params:
            params = {}
            
        # Add API key to parameters
        if self.api_key:
            params['apikey'] = self.api_key
            
        # Set default headers
        if headers is None:
            headers = {'Accept': 'application/json'}
            
        url = f"{self.base_url}{endpoint}"
        
        # Retry logic
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Making API request to {url} (attempt {attempt+1}/{retries+1})")
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                # Check HTTP status first
                if response.status_code != 200:
                    logger.error(f"HTTP error {response.status_code}: {response.text}")
                    if attempt < retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return {'status': '0', 'message': f"HTTP error: {response.status_code}", 'result': []}
                
                # Parse response
                data = response.json()
                
                # Handle Etherscan specific status
                if data.get('status') == '0':
                    error_msg = data.get('message', 'Unknown error')
                    logger.warning(f"Etherscan API returned status 0: {error_msg}")
                    
                    # Handle rate limiting
                    if "rate limit" in error_msg.lower():
                        if attempt < retries:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.info(f"Rate limit hit. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                    
                    # Handle max rate limit or other errors
                    return data
                
                # Successful response
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}/{retries+1})")
                if attempt < retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return {'status': '0', 'message': 'Request timeout', 'result': []}
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                if attempt < retries:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    return {'status': '0', 'message': str(e), 'result': []}
                    
            except ValueError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return {'status': '0', 'message': f'JSON parsing error: {str(e)}', 'result': []}
    
    def get_wallet_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999, page: int = 1, offset: int = 20) -> List[Dict]:
        """Get normal transactions for a wallet address with pagination."""
        logger.info(f"Getting transactions for address {address} (page {page}, offset {offset})")
        
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        # Debug response
        logger.debug(f"API response status: {response.get('status')}")
        logger.debug(f"API response message: {response.get('message', 'No message')}")
        
        if response.get('status') != '1':
            logger.warning(f"No transactions found for {address}: {response.get('message', 'Unknown reason')}")
            return []
            
        transactions = response.get('result', [])
        logger.info(f"Found {len(transactions)} transactions for {address}")
        
        return transactions
    
    def get_token_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999, page: int = 1, offset: int = 20) -> List[Dict]:
        """Get ERC20 token transactions with pagination."""
        logger.info(f"Getting token transactions for address {address} (page {page}, offset {offset})")
        
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        if response.get('status') != '1':
            logger.warning(f"No token transactions found for {address}: {response.get('message', 'Unknown reason')}")
            return []
            
        transactions = response.get('result', [])
        logger.info(f"Found {len(transactions)} token transactions for {address}")
        
        return transactions
    
    def get_internal_transactions(self, address: str, page: int = 1, offset: int = 20) -> List[Dict]:
        """Get internal transactions with pagination."""
        logger.info(f"Getting internal transactions for address {address} (page {page}, offset {offset})")
        
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'page': page,
            'offset': offset,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        if response.get('status') != '1':
            logger.warning(f"No internal transactions found for {address}: {response.get('message', 'Unknown reason')}")
            return []
            
        transactions = response.get('result', [])
        logger.info(f"Found {len(transactions)} internal transactions for {address}")
        
        return transactions
    
    def get_eth_balance(self, address: str) -> float:
        """Get ETH balance for a wallet address."""
        logger.info(f"Getting ETH balance for address {address}")
        
        params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest'
        }
        
        response = self._make_request("", params)
        
        if response.get('status') != '1':
            logger.warning(f"Could not get balance for {address}: {response.get('message', 'Unknown reason')}")
            return 0.0
            
        # Convert wei to ETH
        balance_wei = int(response.get('result', '0'))
        balance_eth = balance_wei / 1e18
        
        logger.info(f"Balance for {address}: {balance_eth} ETH")
        
        return balance_eth

    def get_abi(self, contract_address: str) -> str:
        """Get contract ABI for a verified contract."""
        logger.info(f"Getting ABI for contract {contract_address}")
        
        params = {
            'module': 'contract',
            'action': 'getabi',
            'address': contract_address
        }
        
        response = self._make_request("", params)
        
        if response.get('status') != '1':
            logger.warning(f"Could not get ABI for {contract_address}: {response.get('message', 'Unknown reason')}")
            return ""
            
        return response.get('result', "")
        
    def get_token_info(self, contract_address: str) -> Dict:
        """Get token information for an ERC20 token contract."""
        logger.info(f"Getting token info for {contract_address}")
        
        # Get token name
        name_params = {
            'module': 'account',
            'action': 'tokentx',
            'contractaddress': contract_address,
            'page': 1,
            'offset': 1,
            'sort': 'desc'
        }
        
        name_response = self._make_request("", name_params)
        
        if name_response.get('status') != '1':
            logger.warning(f"Could not get token info for {contract_address}")
            return {}
            
        token_txs = name_response.get('result', [])
        if not token_txs:
            return {}
            
        # Extract token info from first transaction
        token_tx = token_txs[0]
        
        return {
            'name': token_tx.get('tokenName', ''),
            'symbol': token_tx.get('tokenSymbol', ''),
            'decimals': int(token_tx.get('tokenDecimal', 18)),
            'address': contract_address
        }

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(level=logging.INFO)
    
    # Test the collector
    collector = EtherscanCollector()
    
    # Test address (Binance hot wallet)
    address = "0x28c6c06298d514db089934071355e5743bf21d60"
    
    # Test getting transactions
    txs = collector.get_wallet_transactions(address)
    print(f"Found {len(txs)} transactions")
    
    # Test getting token transactions
    token_txs = collector.get_token_transactions(address)
    print(f"Found {len(token_txs)} token transactions")
    
    # Test getting balance
    balance = collector.get_eth_balance(address)
    print(f"Balance: {balance} ETH")