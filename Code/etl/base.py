# src/etl/base.py
from abc import ABC, abstractmethod
from datetime import datetime
import logging
from typing import Any, Dict, List
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data from source"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform extracted data"""
        pass

class DatabaseLoader:
    """Handles loading data into PostgreSQL database"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        
    def load(self, data: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
        """Load DataFrame into PostgreSQL"""
        try:
            data['loaded_at'] = datetime.utcnow()
            data.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False
            )
            logger.info(f"Successfully loaded {len(data)} rows into {table_name}")
        except Exception as e:
            logger.error(f"Error loading data into {table_name}: {str(e)}")
            raise

class ETLPipeline:
    """Coordinates the ETL process"""
    
    def __init__(self, data_source: DataSource, loader: DatabaseLoader):
        self.data_source = data_source
        self.loader = loader
        
    def run(self, table_name: str) -> None:
        """Execute the full ETL pipeline"""
        try:
            logger.info("Starting ETL pipeline")
            
            # Extract
            logger.info("Extracting data")
            raw_data = self.data_source.extract()
            
            # Transform
            logger.info("Transforming data")
            transformed_data = self.data_source.transform(raw_data)
            
            # Load
            logger.info("Loading data")
            self.loader.load(transformed_data, table_name)
            
            logger.info("ETL pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise

# Example implementation for CoinGecko API
class CoinGeckoSource(DataSource):
    """Implementation for CoinGecko API"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def extract(self) -> pd.DataFrame:
        """Extract market data from CoinGecko"""
        try:
            response = self._session.get(
                f"{self.BASE_URL}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 1,
                    "sparkline": False
                }
            )
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except Exception as e:
            logger.error(f"Error extracting data from CoinGecko: {str(e)}")
            raise
            
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform CoinGecko market data"""
        try:
            # Select and rename relevant columns
            columns = {
                'id': 'coin_id',
                'symbol': 'symbol',
                'name': 'name',
                'current_price': 'price_usd',
                'market_cap': 'market_cap_usd',
                'market_cap_rank': 'market_cap_rank',
                'total_volume': 'volume_24h_usd',
                'high_24h': 'high_24h_usd',
                'low_24h': 'low_24h_usd',
                'price_change_24h': 'price_change_24h_usd',
                'price_change_percentage_24h': 'price_change_percentage_24h',
                'circulating_supply': 'circulating_supply',
                'total_supply': 'total_supply',
                'max_supply': 'max_supply',
                'last_updated': 'last_updated'
            }
            
            transformed = data[columns.keys()].rename(columns=columns)
            
            # Convert timestamp
            transformed['last_updated'] = pd.to_datetime(transformed['last_updated'])
            
            # Add extraction timestamp
            transformed['extracted_at'] = datetime.utcnow()
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming CoinGecko data: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize components
    source = CoinGeckoSource()
    loader = DatabaseLoader('postgresql://user:password@localhost:5432/crypto_db')
    
    # Create and run pipeline
    pipeline = ETLPipeline(source, loader)
    pipeline.run('coin_market_data')