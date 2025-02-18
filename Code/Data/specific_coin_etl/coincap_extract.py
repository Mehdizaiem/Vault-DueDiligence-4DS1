import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_coincap_data(asset_id='bitcoin', days=365):
    """
    Fetch historical data from CoinCap API
    Args:
        asset_id (str): Asset ID (default: 'bitcoin')
        days (int): Number of days of historical data to fetch
    """
    # Calculate start and end timestamps
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Convert to millisecond timestamps
    start_timestamp = int(start.timestamp() * 1000)
    end_timestamp = int(end.timestamp() * 1000)
    
    url = f'https://api.coincap.io/v2/assets/{asset_id}/history'
    
    params = {
        'interval': 'd1',
        'start': start_timestamp,
        'end': end_timestamp
    }
    
    try:
        response = requests.get(url, params=params)
        print(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        data = response.json()
        
        if 'data' not in data:
            print("No data found in response")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        print(f"Successfully fetched {len(df)} days of {asset_id} data from CoinCap")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching CoinCap data: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Error details: {e.response.text}")
        return None