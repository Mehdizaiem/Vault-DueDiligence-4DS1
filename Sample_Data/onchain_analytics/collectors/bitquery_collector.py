# Sample_Data/onchain_analytics/collectors/bitquery_collector.py
import os
import requests
from typing import Dict, List, Any
from datetime import datetime
from .base_collector import BaseCollector

class BitqueryCollector(BaseCollector):
    """Collector for Bitquery API data."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://graphql.bitquery.io"
        # Use API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("BITQUERY_API_KEY")
    
    def _make_request(self, query: str, variables: Dict = None) -> Dict:
        """Make GraphQL request to Bitquery API."""
        if not self.api_key:
            return {'error': 'API key not provided'}
            
        headers = {
            'Content-Type': 'application/json',
            'X-API-KEY': self.api_key
        }
        
        payload = {
            'query': query
        }
        
        if variables:
            payload['variables'] = variables
            
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def get_wallet_transactions(self, address: str, blockchain: str = 'ethereum', limit: int = 100) -> List[Dict]:
        """Get transactions for a wallet address across different blockchains."""
        query = """
        query ($network: EthereumNetwork!, $address: String!, $limit: Int!) {
          ethereum(network: $network) {
            transactions(
              options: {limit: $limit}
              txFrom: {is: $address}
            ) {
              hash
              timestamp {
                time(format: "%Y-%m-%d %H:%M:%S")
              }
              block {
                height
              }
              txFrom {
                address
              }
              txTo {
                address
              }
              gasValue
              gasPrice
              value
              success
            }
          }
        }
        """
        
        variables = {
            'network': blockchain.upper(),
            'address': address,
            'limit': limit
        }
        
        response = self._make_request(query, variables)
        
        if 'error' in response:
            return []
            
        try:
            transactions = response.get('data', {}).get('ethereum', {}).get('transactions', [])
            return transactions
        except:
            return []