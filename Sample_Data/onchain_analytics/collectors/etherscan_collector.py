# Sample_Data/onchain_analytics/collectors/etherscan_collector.py
import os
from typing import Dict, List, Any
from datetime import datetime
from .base_collector import BaseCollector

class EtherscanCollector(BaseCollector):
    """Collector for Etherscan API data."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.etherscan.io/api"
        # Use API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("ETHERSCAN_API_KEY")
    
    def get_wallet_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get normal transactions for a wallet address."""
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        if 'error' in response:
            return []
            
        if response.get('status') != '1':
            return []
            
        return response.get('result', [])
    
    def get_token_transactions(self, address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        """Get ERC20 token transactions for a wallet address."""
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        if 'error' in response:
            return []
            
        if response.get('status') != '1':
            return []
            
        return response.get('result', [])
    
    def get_internal_transactions(self, address: str) -> List[Dict]:
        """Get internal transactions for a wallet address."""
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'sort': 'desc'
        }
        
        response = self._make_request("", params)
        
        if 'error' in response:
            return []
            
        if response.get('status') != '1':
            return []
            
        return response.get('result', [])
    
    def get_eth_balance(self, address: str) -> float:
        """Get ETH balance for a wallet address."""
        params = {
            'module': 'account',
            'action': 'balance',
            'address': address,
            'tag': 'latest'
        }
        
        response = self._make_request("", params)
        
        if 'error' in response or response.get('status') != '1':
            return 0.0
            
        # Convert wei to ETH
        balance_wei = int(response.get('result', '0'))
        balance_eth = balance_wei / 1e18
        
        return balance_eth