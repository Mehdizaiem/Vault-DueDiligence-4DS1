import requests
import json

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
PARAMS = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10, "page": 1}

def fetch_coingecko_data():
    response = requests.get(COINGECKO_URL, params=PARAMS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

if __name__ == "__main__":
    data = fetch_coingecko_data()
    print(json.dumps(data, indent=2))
