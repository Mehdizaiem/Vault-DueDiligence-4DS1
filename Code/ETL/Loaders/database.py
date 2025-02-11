import psycopg2
from Config.settings import DB_CONFIG

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def create_table():
    query = """
    CREATE TABLE IF NOT EXISTS crypto_prices (
        id SERIAL PRIMARY KEY,
        name VARCHAR(50),
        symbol VARCHAR(10),
        price DECIMAL(18,8),
        market_cap BIGINT,
        volume_24h BIGINT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()

def insert_data(cleaned_data):
    query = """
    INSERT INTO crypto_prices (name, symbol, price, market_cap, volume_24h)
    VALUES (%s, %s, %s, %s, %s);
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            for coin in cleaned_data:
                cur.execute(query, (coin["name"], coin["symbol"], coin["price"], coin["market_cap"], coin["volume_24h"]))
            conn.commit()
