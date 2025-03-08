# File path: Code/data_processing/time_series_manager.py

import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class TimeSeriesManager:
    """
    Manager for loading, processing, and storing time series data
    in the CryptoTimeSeries collection.
    """
    
    def __init__(self, csv_loader=None, storage_manager=None):
        """
        Initialize the time series manager.
        
        Args:
            csv_loader: CSV loader instance (will be created if None)
            storage_manager: Storage manager instance (will be created if None)
        """
        # Initialize CSV loader if not provided
        if csv_loader is None:
            from Code.data_processing.csv_loader import CryptoCSVLoader
            self.csv_loader = CryptoCSVLoader()
        else:
            self.csv_loader = csv_loader
        
        # Initialize storage manager if not provided
        if storage_manager is None:
            from Sample_Data.vector_store.storage_manager import StorageManager
            self.storage = StorageManager()
            self.storage.connect()
        else:
            self.storage = storage_manager
        
        # Track processed symbols
        self.processed_symbols = set()
        
        logger.info("Time Series Manager initialized")
    
    def load_and_store_all(self, force: bool = False) -> Dict[str, int]:
        """
        Load and store all available time series data.
        
        Args:
            force: Force update even if data exists
            
        Returns:
            Dict with statistics on processed data
        """
        logger.info("Loading and storing all time series data")
        
        # Get available symbols
        symbols = self.csv_loader.get_available_symbols()
        logger.info(f"Found {len(symbols)} available symbols")
        
        results = {
            "total_symbols": len(symbols),
            "success_count": 0,
            "failure_count": 0,
            "skipped_count": 0,
            "total_data_points": 0
        }
        
        for symbol in symbols:
            try:
                # Check if we need to process this symbol
                if not force and symbol in self.processed_symbols:
                    logger.info(f"Symbol {symbol} already processed, skipping")
                    results["skipped_count"] += 1
                    continue
                
                # Check if data already exists in Weaviate
                existing_data = self.storage.retrieve_time_series(symbol, limit=1)
                if existing_data and not force:
                    logger.info(f"Data for {symbol} already exists in Weaviate, skipping")
                    self.processed_symbols.add(symbol)
                    results["skipped_count"] += 1
                    continue
                
                # Load data from CSV
                logger.info(f"Loading data for {symbol}")
                data = self.csv_loader.load_historical_data(symbol)
                
                if not data:
                    logger.warning(f"No data found for {symbol}")
                    results["failure_count"] += 1
                    continue
                
                # Store in Weaviate
                logger.info(f"Storing {len(data)} data points for {symbol}")
                success = self.storage.store_time_series(data)
                
                if success:
                    logger.info(f"Successfully stored data for {symbol}")
                    self.processed_symbols.add(symbol)
                    results["success_count"] += 1
                    results["total_data_points"] += len(data)
                else:
                    logger.error(f"Failed to store data for {symbol}")
                    results["failure_count"] += 1
                
                # Small delay to avoid overwhelming Weaviate
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results["failure_count"] += 1
        
        logger.info(f"Completed processing {results['total_symbols']} symbols")
        return results
    
    def load_and_store_symbol(self, symbol: str, force: bool = False) -> bool:
        """
        Load and store time series data for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            force: Force update even if data exists
            
        Returns:
            True if successful
        """
        logger.info(f"Loading and storing data for {symbol}")
        
        try:
            # Check if symbol already processed
            if not force and symbol in self.processed_symbols:
                logger.info(f"Symbol {symbol} already processed, skipping")
                return True
            
            # Check if data already exists in Weaviate
            existing_data = self.storage.retrieve_time_series(symbol, limit=1)
            if existing_data and not force:
                logger.info(f"Data for {symbol} already exists in Weaviate, skipping")
                self.processed_symbols.add(symbol)
                return True
            
            # Load data from CSV
            data = self.csv_loader.load_historical_data(symbol)
            
            if not data:
                logger.warning(f"No data found for {symbol}")
                return False
            
            # Store in Weaviate
            success = self.storage.store_time_series(data)
            
            if success:
                logger.info(f"Successfully stored {len(data)} data points for {symbol}")
                self.processed_symbols.add(symbol)
                return True
            else:
                logger.error(f"Failed to store data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return False
    
    def get_processed_symbols(self) -> List[str]:
        """Get list of processed symbols"""
        return list(self.processed_symbols)
    
    def close(self):
        """Close the storage connection"""
        if hasattr(self.storage, 'close'):
            self.storage.close()

# Example usage
if __name__ == "__main__":
    manager = TimeSeriesManager()
    
    try:
        # Either process all symbols
        results = manager.load_and_store_all()
        print(f"Processed {results['success_count']} symbols successfully")
        print(f"Total data points stored: {results['total_data_points']}")
        
        # Or process a specific symbol
        # success = manager.load_and_store_symbol("BTCUSD")
        # print(f"Successfully processed BTCUSD: {success}")
    finally:
        manager.close()