from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database(db_connection_string):
    """Set up the TimescaleDB database with required tables and extensions"""
    try:
        engine = create_engine(db_connection_string)
        
        with engine.connect() as conn:
            # Enable TimescaleDB extension
            logger.info("Enabling TimescaleDB extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            
            # Create the crypto prices table
            logger.info("Creating crypto_prices table...")
            conn.execute(text("""
                DROP TABLE IF EXISTS crypto_prices CASCADE;
                
                CREATE TABLE crypto_prices (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    exchange VARCHAR(20) NOT NULL,
                    price NUMERIC,
                    volume NUMERIC,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Convert to TimescaleDB hypertable
            logger.info("Converting to hypertable...")
            conn.execute(text("""
                SELECT create_hypertable('crypto_prices', 'time');
            """))
            
            # Create indexes
            logger.info("Creating indexes...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_crypto_prices_symbol ON crypto_prices(symbol);
                CREATE INDEX IF NOT EXISTS idx_crypto_prices_exchange ON crypto_prices(exchange);
            """))
            
            # Create materialized view for hourly aggregates
            logger.info("Creating materialized views...")
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS crypto_prices_hourly AS
                SELECT
                    time_bucket('1 hour', time) AS hour,
                    symbol,
                    exchange,
                    AVG(price) as avg_price,
                    SUM(volume) as total_volume,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    COUNT(*) as num_trades
                FROM crypto_prices
                GROUP BY time_bucket('1 hour', time), symbol, exchange
                ORDER BY hour DESC;
            """))
            
            conn.commit()
            logger.info("Database setup completed successfully!")
            
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

if __name__ == "__main__":
    db_connection = "postgresql://postgres:hamza@localhost:5432/crypto_db"
    setup_database(db_connection)