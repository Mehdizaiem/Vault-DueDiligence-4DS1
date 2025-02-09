# code/etl/config/settings.py
import os

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'crypto_vault_db',
    'user': 'postgres',
    'password': 'Spookes1234'  # Replace with your actual PostgreSQL password
}

def get_db_url():
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# CoinGecko API Key
COINGECKO_API_KEY = "CG-84A9nc2Jcq3Yk7hrX4JQ79gm"