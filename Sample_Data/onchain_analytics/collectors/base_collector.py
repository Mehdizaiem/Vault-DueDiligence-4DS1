# Sample_Data/onchain_analytics/collectors/base_collector.py
import os
import requests
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseCollector:
    """Base class for blockchain data collectors."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = None
        
    def _make_request(self, endpoint: str, params: Dict = None, headers: Dict = None) -> Dict:
        """Make HTTP request with error handling and retry logic."""
        if not self.base_url:
            raise ValueError("Base URL not set")
            
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to parameters if provided
        if params is None:
            params = {}
            
        if self.api_key and 'apikey' not in params:
            params['apikey'] = self.api_key
            
        # Set default headers
        if headers is None:
            headers = {'Accept': 'application/json'}
            
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Handle rate limiting
            if e.response.status_code == 429:
                logger.warning("Rate limit hit. Please wait before making more requests.")
            logger.error(f"HTTP error: {e}")
            return {'error': str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {'error': str(e)}
        except ValueError as e:
            logger.error(f"JSON parsing error: {e}")
            return {'error': str(e)}