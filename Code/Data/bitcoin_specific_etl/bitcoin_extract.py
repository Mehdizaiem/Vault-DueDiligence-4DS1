import requests

def fetch_bitcoin_data(api_key=None):
    """
    Fetch Bitcoin historical data for the last year (365 days)
    Using public API endpoint since we're limited to 365 days
    """
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    
    params = {
        'vs_currency': 'usd',
        'days': 365,  # Maximum allowed for free tier
        'interval': 'daily'
    }
    
    try:
        # Use public API without authentication
        response = requests.get(url, params=params)
        
        # Print response status and headers for debugging
        print(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        print(f"Successfully fetched {params['days']} days of Bitcoin data")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Bitcoin data: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Error details: {e.response.text}")
        return None