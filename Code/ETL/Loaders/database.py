import psycopg2
from Config.settings import DB_CONFIG

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

#coinGeko scraper 
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
#Tweeter scraper 
'''
def create_twitter_table():
    query = """
    CREATE TABLE IF NOT EXISTS twitter_data (
        id SERIAL PRIMARY KEY,
        tweet_id VARCHAR(50) UNIQUE,
        twitter_user VARCHAR(50),
        text TEXT,
        created_at TIMESTAMP
    );
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()

def insert_twitter_data(cleaned_tweets):
    query = """
    INSERT INTO twitter_data (tweet_id, user, text, created_at)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (tweet_id) DO NOTHING;
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            for tweet in cleaned_tweets:
                cur.execute(query, (tweet["tweet_id"], tweet["user"], tweet["text"], tweet["created_at"]))
            conn.commit()
'''

#Document scraper 

def create_document_data_table():
    """Ensure the document_data table exists in PostgreSQL."""
    query = """
    CREATE TABLE IF NOT EXISTS document_data (
        id SERIAL PRIMARY KEY,
        buyer_name VARCHAR(255),
        buyer_last_name VARCHAR(255),
        buyer_company VARCHAR(255),
        seller_name VARCHAR(255),
        seller_last_name VARCHAR(255),
        seller_company VARCHAR(255),
        agreement VARCHAR(255),
        date VARCHAR(255)
    );
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()

def insert_document_data(extracted_data):
    """Insert structured contract details into PostgreSQL."""
    query = """
    INSERT INTO document_data (
        buyer_name, buyer_last_name, buyer_company,
        seller_name, seller_last_name, seller_company,
        agreement, date
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (
                extracted_data.get("buyer_name", "Unknown"),
                extracted_data.get("buyer_last_name", "Unknown"),
                extracted_data.get("buyer_company", "Unknown"),
                extracted_data.get("seller_name", "Unknown"),
                extracted_data.get("seller_last_name", "Unknown"),
                extracted_data.get("seller_company", "Unknown"),
                extracted_data.get("agreement", "Unknown"),
                extracted_data.get("date", "Unknown")
            ))
            conn.commit()