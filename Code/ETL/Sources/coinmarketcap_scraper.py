#import requests
#import json

#API_KEY = "YOUR_API_KEY"
#HEADERS = {"X-CMC_PRO_API_KEY": API_KEY}
#URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

#def fetch_coinmarketcap_data():
#    response = requests.get(URL, headers=HEADERS)
#   if response.status_code == 200:
 #       return response.json()["data"]
#    else:
#        print(f"Error {response.status_code}: {response.text}")
#        return []

#if __name__ == "__main__":
#    data = fetch_coinmarketcap_data()
#    print(json.dumps(data, indent=2))
