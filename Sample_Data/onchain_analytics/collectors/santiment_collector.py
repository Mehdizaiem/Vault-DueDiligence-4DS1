# Sample_Data/onchain_analytics/collectors/santiment_collector.py
import os
import requests
from typing import Dict, List, Any
from datetime import datetime, timedelta
from .base_collector import BaseCollector

class SantimentCollector(BaseCollector):
    """Collector for Santiment API data."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://api.santiment.net/graphql"
        # Use API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("SANTIMENT_API_KEY")
    
    def _make_request(self, query: str, variables: Dict = None) -> Dict:
        """Make GraphQL request to Santiment API."""
        if not self.api_key:
            return {'error': 'API key not provided'}
            
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Apikey {self.api_key}'
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
    
    def get_token_metrics(self, slug: str, from_date: datetime = None, to_date: datetime = None) -> Dict:
        """Get on-chain metrics for a token."""
        if not from_date:
            from_date = datetime.now() - timedelta(days=30)
            
        if not to_date:
            to_date = datetime.now()
            
        from_str = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        query = """
        query ($slug: String!, $from: DateTime!, $to: DateTime!) {
          getMetric(metric: "daily_active_addresses") {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
          getMetric(metric: "transaction_volume") {
            timeseriesData(
              slug: $slug
              from: $from
              to: $to
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """
        
        variables = {
            'slug': slug,
            'from': from_str,
            'to': to_str
        }
        
        response = self._make_request(query, variables)
        
        if 'error' in response:
            return {}
            
        return response.get('data', {})