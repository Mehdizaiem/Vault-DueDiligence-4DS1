# debug_etherscan.py
import os
import logging
import json
import requests
from dotenv import load_dotenv
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("etherscan_debug.log")
    ]
)
logger = logging.getLogger(__name__)

# Import directly from the collectors package since we're already in onchain_analytics
from collectors.etherscan_collector import EtherscanCollector

def load_api_key():
    """Load API key from .env file or environment variables"""
    # Path to the root directory where .env.local is located
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    env_path = os.path.join(root_dir, '.env.local')
    
    if os.path.exists(env_path):
        logger.info(f"Loading .env.local from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.warning(f".env.local not found at {env_path}, trying default .env")
        load_dotenv()
    
    api_key = os.getenv("ETHERSCAN_API_KEY")
    
    if not api_key:
        logger.error("ETHERSCAN_API_KEY not found in environment variables or .env file")
        return None
        
    return api_key

def test_direct_api_call(api_key, address):
    """Test direct API call to Etherscan"""
    logger.info(f"Testing direct API call to Etherscan for address: {address}")
    
    base_url = "https://api.etherscan.io/api"
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 10,
        'sort': 'desc',
        'apikey': api_key
    }
    
    try:
        logger.info(f"Sending request to {base_url} with params: {params}")
        response = requests.get(base_url, params=params)
        data = response.json()
        
        logger.info(f"API Response Status: {data.get('status')}")
        logger.info(f"API Response Message: {data.get('message')}")
        
        # Save the raw response for inspection
        with open("api_response.json", "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Raw API response saved to api_response.json")
        
        if data.get('status') == '1':
            result = data.get('result', [])
            logger.info(f"Found {len(result)} transactions")
            if result:
                logger.info(f"First transaction hash: {result[0].get('hash')}")
                return True
        else:
            logger.error(f"API Error: {data.get('message')}")
            return False
    
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return False

def test_valid_addresses():
    """Test a list of known valid Ethereum addresses"""
    api_key = load_api_key()
    if not api_key:
        return
    
    logger.info(f"Using API key: {api_key}")
    
    addresses = [
        # Popular exchanges
        "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
        "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",  # Coinbase
        # Ethereum Foundation
        "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae",
        # Random whale address
        "0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503"
    ]
    
    for address in addresses:
        logger.info(f"Testing address: {address}")
        success = test_direct_api_call(api_key, address)
        
        if success:
            logger.info(f"Successfully found transactions for {address}")
            return address
        
        # Wait between requests to avoid rate limiting
        time.sleep(1)
    
    logger.error("Could not find a working address")
    return None

def debug_collector_implementation(address):
    """Debug the EtherscanCollector implementation"""
    api_key = load_api_key()
    if not api_key or not address:
        return
        
    logger.info(f"Testing collector implementation with address: {address}")
    
    # Initialize collector with explicit API key
    collector = EtherscanCollector(api_key=api_key)
    
    # Test wallet transactions
    logger.info(f"Testing get_wallet_transactions...")
    txs = collector.get_wallet_transactions(address, 0, 99999999)
    
    logger.info(f"Collector returned {len(txs)} transactions")
    
    # Save collector response
    with open("collector_response.json", "w") as f:
        json.dump(txs, f, indent=2)
    
    # Test if there's a difference between direct API call and collector
    if len(txs) == 0:
        logger.error("Collector returned 0 transactions despite API returning results")
        
        # Check _make_request method
        logger.info("Testing _make_request method directly...")
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'desc'
        }
        
        raw_response = collector._make_request("", params)
        
        with open("make_request_response.json", "w") as f:
            json.dump(raw_response, f, indent=2)
            
        logger.info(f"_make_request response saved to make_request_response.json")
    else:
        logger.info("Collector implementation working correctly")

def main():
    """Main debug function"""
    logger.info("Starting Etherscan API debugging")
    
    # 1. Test if API key is valid
    api_key = load_api_key()
    if not api_key:
        return
        
    logger.info(f"Using API key: {api_key}")
    
    # 2. Test with known addresses
    valid_address = test_valid_addresses()
    
    # 3. Debug collector implementation
    if valid_address:
        debug_collector_implementation(valid_address)
    
    logger.info("Debugging complete")

if __name__ == "__main__":
    main()