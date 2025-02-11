import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiExchangeETL:
    def __init__(self, db_connection_string):
        logger.info("Initializing MultiExchangeETL...")
        self.db_connection = db_connection_string
        try:
            self.engine = create_engine(db_connection_string)
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                logger.info(f"Connected to database: {result.scalar()}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    async def fetch_coincap_data(self, session, symbol):
        logger.info(f"Fetching CoinCap data for {symbol}...")
        url = f'https://api.coincap.io/v2/assets/{symbol}/history'
        params = {
            'interval': 'm1',
            'start': int((datetime.now().timestamp() - 3600) * 1000),  # Last hour
            'end': int(datetime.now().timestamp() * 1000)
        }
        
        try:
            async with session.get(url, params=params) as response:
                logger.info(f"CoinCap response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data.get('data', []))
                    df['exchange'] = 'coincap'
                    df['symbol'] = symbol.upper()
                    logger.info(f"Fetched {len(df)} records from CoinCap")
                    return df
                else:
                    logger.error(f"CoinCap API error: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching CoinCap data: {e}")
            return None

    async def fetch_binance_data(self, session, symbol):
        logger.info(f"Fetching Binance data for {symbol}...")
        symbol_map = {'bitcoin': 'BTC', 'ethereum': 'ETH'}
        binance_symbol = f"{symbol_map.get(symbol, symbol.upper())}USDT"
        
        url = f'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': binance_symbol,
            'interval': '1m',
            'limit': 60  # Last hour
        }
        
        try:
            async with session.get(url, params=params) as response:
                logger.info(f"Binance response status: {response.status}")
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                                   'volume', 'close_time', 'quote_volume', 'trades', 
                                                   'taker_buy_base', 'taker_buy_quote', 'ignore'])
                    df['exchange'] = 'binance'
                    df['symbol'] = symbol_map.get(symbol, symbol).upper()
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['price'] = pd.to_numeric(df['close'])
                    df['volume'] = pd.to_numeric(df['volume'])
                    logger.info(f"Fetched {len(df)} records from Binance")
                    return df[['timestamp', 'symbol', 'exchange', 'price', 'volume']]
                else:
                    logger.error(f"Binance API error: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching Binance data: {e}")
            return None

    def transform_data(self, dfs):
        logger.info("Transforming data...")
        combined_data = []
        
        for df in dfs:
            if df is not None and not df.empty:
                try:
                    # Ensure timestamp column is renamed to 'time' for TimescaleDB
                    if 'timestamp' in df.columns:
                        df = df.rename(columns={'timestamp': 'time'})
                    
                    # Ensure all required columns exist
                    required_columns = ['time', 'symbol', 'exchange', 'price', 'volume']
                    if not all(col in df.columns for col in required_columns):
                        logger.warning(f"Missing columns in DataFrame. Found: {df.columns.tolist()}")
                        continue
                    
                    # Select and reorder columns
                    df = df[required_columns]
                    combined_data.append(df)
                    logger.info(f"Transformed {len(df)} records from {df['exchange'].iloc[0]}")
                except Exception as e:
                    logger.error(f"Error transforming data: {e}")
                    continue
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Total transformed records: {len(result)}")
            return result
        else:
            logger.warning("No data to transform")
            return None

    def load_to_timescaledb(self, df):
        if df is None or df.empty:
            logger.warning("No data to load")
            return
            
        try:
            logger.info("Loading data to TimescaleDB...")
            
            # Load data
            df.to_sql('crypto_prices', 
                     self.engine, 
                     if_exists='append',
                     index=False,
                     method='multi',
                     chunksize=1000)
            
            # Verify the load
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT exchange, symbol, COUNT(*) as count 
                    FROM crypto_prices 
                    WHERE time >= NOW() - INTERVAL '1 hour'
                    GROUP BY exchange, symbol
                """))
                logger.info("Data loaded in the last hour:")
                for row in result:
                    logger.info(f"{row.exchange} - {row.symbol}: {row.count} records")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.error("DataFrame info:")
            logger.error(df.info())

    async def run_etl(self, symbols=['bitcoin', 'ethereum']):
        logger.info(f"Starting ETL process for symbols: {symbols}")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                tasks.extend([
                    self.fetch_coincap_data(session, symbol),
                    self.fetch_binance_data(session, symbol)
                ])
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in results if isinstance(r, pd.DataFrame)]
            
            if valid_results:
                transformed_data = self.transform_data(valid_results)
                self.load_to_timescaledb(transformed_data)
                logger.info("ETL process completed successfully")
            else:
                logger.error("No valid data received from any source")

def main():
    logger.info("Starting crypto multi-exchange ETL process...")
    try:
        db_connection = "postgresql://postgres:hamza@localhost:5432/crypto_db"
        etl = MultiExchangeETL(db_connection)
        asyncio.run(etl.run_etl())
        logger.info("ETL process finished successfully")
    except Exception as e:
        logger.error(f"ETL process failed: {e}")
        raise

if __name__ == "__main__":
    main()