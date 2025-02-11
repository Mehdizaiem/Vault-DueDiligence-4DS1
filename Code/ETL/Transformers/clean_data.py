def clean_crypto_data(raw_data):
    cleaned_data = []
    for coin in raw_data:
        cleaned_data.append({
            "name": coin["name"],
            "symbol": coin["symbol"],
            "price": coin.get("current_price", 0),
            "market_cap": coin.get("market_cap", 0),
            "volume_24h": coin.get("total_volume", 0),
        })
    return cleaned_data
