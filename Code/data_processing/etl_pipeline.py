import sys
import os
from datetime import datetime
import traceback
import logging
# Add Sample_Data to path (from Code/data_processing, up to root, then into Sample_Data)
SAMPLE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Sample_Data'))
sys.path.append(SAMPLE_DATA_PATH)

# Now import from vector_store
from vector_store.weaviate_client import get_weaviate_client, check_weaviate_connection


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from extract import extract
from transform import transform
from load import load

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_market_data(data):
    """Validate market data structure"""
    if not data:
        return False
        
    required_fields = {'source', 'data'}
    for item in data:
        if not all(field in item for field in required_fields):
            logger.error(f"Missing required fields in data item: {item}")
            return False
    return True

def etl_pipeline():
    """Execute the complete ETL pipeline for market data"""
    start_time = datetime.now()
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting Market Data ETL Pipeline at {start_time}")
    logger.info(f"{'='*50}\n")

    try:
        # Check Weaviate connection
        client = get_weaviate_client()
        if not check_weaviate_connection(client):
            logger.error("❌ Cannot connect to Weaviate. Stopping pipeline.")
            return
        client.close()

        # Extract
        logger.info("1. Extraction Phase")
        logger.info("-"*20)
        extract_start_time = datetime.now()
        raw_data = extract()
        
        if not check_market_data(raw_data):
            logger.error("❌ Invalid market data structure. Stopping pipeline.")
            return
            
        extract_duration = datetime.now() - extract_start_time
        logger.info(f"✅ Extraction complete - {len(raw_data)} records retrieved in {extract_duration}\n")

        # Transform
        logger.info("2. Transformation Phase")
        logger.info("-"*20)
        transform_start_time = datetime.now()
        transformed_data = transform(raw_data)
        
        if not check_market_data(transformed_data):
            logger.error("❌ Invalid transformed data structure. Stopping pipeline.")
            return
            
        transform_duration = datetime.now() - transform_start_time
        logger.info(f"✅ Transformation complete - {len(transformed_data)} records processed in {transform_duration}\n")

        # Load
        logger.info("3. Loading Phase")
        logger.info("-"*20)
        load_start_time = datetime.now()
        load(transformed_data)
        load_duration = datetime.now() - load_start_time
        logger.info(f"✅ Loading complete in {load_duration}\n")

        # Summary
        end_time = datetime.now()
        total_duration = end_time - start_time
        logger.info(f"{'='*50}")
        logger.info(f"Market Data ETL Pipeline completed successfully at {end_time}")
        logger.info(f"Total duration: {total_duration}")
        logger.info(f"{'='*50}\n")

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}")
        logger.error("\nFull error traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    etl_pipeline()