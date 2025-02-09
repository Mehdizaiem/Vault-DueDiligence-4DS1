# code/etl/test_coingecko.py
from base import CoinGeckoSource, DatabaseLoader, ETLPipeline
from config.settings import get_db_url

def test_coingecko_pipeline():
    try:
        # Initialize CoinGecko source with your API key
        source = CoinGeckoSource(api_key="CG-84A9nc2Jcq3Yk7hrX4JQ79gm")
        
        # Initialize database loader with your connection string
        loader = DatabaseLoader(get_db_url())
        
        # Create and run pipeline
        pipeline = ETLPipeline(source, loader)
        pipeline.run('coin_market_data')
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    test_coingecko_pipeline()