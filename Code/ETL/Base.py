from Sources.coingecko_scraper import fetch_coingecko_data
from Transformers.clean_data import clean_crypto_data
from Loaders.database import create_table, insert_data
from Loaders.database import create_document_data_table, insert_document_data

#from Sources.document_extractor import process_documents
from Utils.logger import log_info, log_error
import sys
import os
#from Sources.twitter_scraper import fetch_tweets
#from Transformers.clean_twitter_data import clean_twitter_data
#from Loaders.database import create_twitter_table, insert_twitter_data

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sources"))
from Sources.document_extractor import process_document

'''def run_twitter_etl(keyword=["crypto", "blockchain", "bitcoin","ethereum","crypto due diligence","due diligence"], count=200):
    log_info("Starting Twitter ETL process...")

    try:
        raw_tweets = fetch_tweets(keyword, count)
        cleaned_tweets = clean_twitter_data(raw_tweets)

        create_twitter_table()
        insert_twitter_data(cleaned_tweets)

        log_info("Twitter ETL process completed successfully!")
    
    except Exception as e:
        log_error(f"Error in Twitter ETL: {e}")
'''
def run_crypto_etl():
    log_info("Starting Crypto ETL process...")

    try:
        raw_data = fetch_coingecko_data()
        cleaned_data = clean_crypto_data(raw_data)

        create_table()
        insert_data(cleaned_data)

        log_info("Crypto ETL process completed successfully!")
    except Exception as e:
        log_error(f"Error in Crypto ETL: {e}")

def run_document_etl():
    """Run document ETL process."""
    log_info("Starting Document ETL process...")
    try:
        process_document(pdf_path=r"C:\Users\Nessrine\Downloads\agreement.pdf")
        log_info("Document ETL process completed successfully!")
    except Exception as e:
        log_error(f"Error in Document ETL: {e}")

if __name__ == "__main__":
    run_crypto_etl()
    #run_twitter_etl()
    run_document_etl()

