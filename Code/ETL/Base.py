from Sources.coingecko_scraper import fetch_coingecko_data
from Transformers.clean_data import clean_crypto_data
from Loaders.database import create_table, insert_data
from Utils.logger import log_info

def run_etl():
    log_info("Starting ETL process...")

    raw_data = fetch_coingecko_data()
    cleaned_data = clean_crypto_data(raw_data)

    create_table()
    insert_data(cleaned_data)

    log_info("ETL process completed successfully!")

if __name__ == "__main__":
    run_etl()
