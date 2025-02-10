import requests

def fetch_crypto_data(api_key):
    """Fetch cryptocurrency data from CoinGecko API"""
    headers = {
        'X-CG-API-KEY': api_key
    }
    
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None