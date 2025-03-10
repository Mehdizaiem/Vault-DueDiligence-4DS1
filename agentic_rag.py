# File path: agentic_rag.py
#!/usr/bin/env python
import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import time
import threading
import json
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_due_diligence.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

class CryptoDueDiligenceSystem:
    """
    Main orchestrator for the Crypto Due Diligence system.
    
    Manages data acquisition, processing, storage, retrieval, and analysis
    across all collections:
    - CryptoDueDiligenceDocuments (using all-MPNet embeddings)
    - CryptoNewsSentiment (using FinBERT embeddings)
    - MarketMetrics (no embeddings)
    - CryptoTimeSeries (no embeddings)
    - OnChainAnalytics (no embeddings)
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the system with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.collector = None
        self.processor = None
        self.storage = None
        self.time_series_manager = None
        self.onchain_manager = None
        
        # Create data directory
        os.makedirs(self.config.get("data_dir", "data"), exist_ok=True)
        
        # Track last operation times
        self.last_news_scrape = None
        self.last_market_update = None
        self.last_forecast_update = None
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "crypto_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"],
            "news_sources": ["CoinDesk", "Cointelegraph"],
            "scrape_interval_hours": 4,
            "market_update_interval_minutes": 15,
            "forecast_interval_hours": 12,
            "forecast_days_ahead": 7,
            "data_dir": "data",
            "use_finbert": True
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                # Update with any missing defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                        
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Save default config
            try:
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving default config: {e}")
                
            return default_config
    
    def _initialize_embeddings(self):
        """Initialize embedding models"""
        try:
            # Import the embedding module to trigger initialization
            from Sample_Data.vector_store.embed import (
                initialize_models, 
                generate_mpnet_embedding, 
                generate_finbert_embedding
            )
            
            # Explicitly initialize models
            initialize_models()
            logger.info("Embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
    
    def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize data collector
            from Code.data_acquisition.data_collector import DataCollector
            self.collector = DataCollector()
            logger.info("Data collector initialized")
            
            # Initialize data processor
            from Code.data_processing.processor import DataProcessor
            self.processor = DataProcessor(use_finbert=self.config.get("use_finbert", True))
            logger.info("Data processor initialized")
            
            # Initialize storage manager
            from Sample_Data.vector_store.storage_manager import StorageManager
            self.storage = StorageManager()
            
            # Initialize time series manager
            from Code.data_processing.time_series_manager import TimeSeriesManager
            self.time_series_manager = TimeSeriesManager(storage_manager=self.storage)
            logger.info("Time series manager initialized")
            
            # Initialize on-chain manager - UNCOMMENTED THIS SECTION
            try:
                from Code.data_processing.onchain_manager import OnChainManager
                self.onchain_manager = OnChainManager(storage_manager=self.storage)
                logger.info("OnChain manager initialized")
            except ImportError as e:
                logger.warning(f"OnChain manager could not be initialized: {e}")
                self.onchain_manager = None
            
            # Set up schemas
            self.storage.setup_schemas()
            logger.info("Storage manager initialized and schemas set up")
            
            logger.info("All system components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    def update_market_data(self, force: bool = False):
        """
        Update market data for all configured symbols.
        
        Args:
            force (bool): Force update even if interval hasn't elapsed
        """
        # Check if update is needed
        now = datetime.now()
        interval_minutes = self.config.get("market_update_interval_minutes", 15)
        
        if not force and self.last_market_update and \
           (now - self.last_market_update).total_seconds() < interval_minutes * 60:
            logger.info(f"Market data update skipped (last update: {self.last_market_update})")
            return
        
        logger.info("Updating market data")
        
        try:
            # Get symbols from config
            symbols = [s.replace("USDT", "") for s in self.config.get("crypto_symbols", [])]
            
            # Fetch market data
            market_data = self.collector.fetch_market_data(symbols)
            
            if market_data:
                # Store in Weaviate
                success = self.storage.store_market_data(market_data)
                
                if success:
                    logger.info(f"Market data updated successfully for {len(market_data)} symbols")
                    self.last_market_update = now
                else:
                    logger.error("Failed to store market data")
            else:
                logger.warning("No market data retrieved")
                
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def update_historical_data(self, force: bool = False):
        """
        Update historical time series data from CSV files.
        
        Args:
            force (bool): Force update even if data exists
        """
        logger.info("Updating historical data from CSV files")
        
        try:
            # Use the time series manager to load and store CSV data
            results = self.time_series_manager.load_and_store_all(force=force)
            
            logger.info(f"Historical data update completed:")
            logger.info(f"  Successfully processed {results['success_count']}/{results['total_symbols']} symbols")
            logger.info(f"  Total data points stored: {results['total_data_points']}")
                
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
    
    def update_news(self, force: bool = False):
        """
        Update crypto news with sentiment analysis.
        
        Args:
            force (bool): Force update even if interval hasn't elapsed
        """
        # Check if update is needed
        now = datetime.now()
        interval_hours = self.config.get("scrape_interval_hours", 4)
        
        if not force and self.last_news_scrape and \
           (now - self.last_news_scrape).total_seconds() < interval_hours * 3600:
            logger.info(f"News update skipped (last update: {self.last_news_scrape})")
            return
        
        logger.info("Updating crypto news")
        
        try:
            # Fetch news
            limit_per_source = self.config.get("news_limit_per_source", 10)
            news_articles = self.collector.fetch_news(limit_per_source=limit_per_source)
            
            if news_articles:
                # Process with sentiment analysis
                processed_articles = self.processor.analyze_news_batch(news_articles)
                
                # Store in Weaviate
                stored_count = 0
                for article in processed_articles:
                    if self.storage.store_news_article(article):
                        stored_count += 1
                
                logger.info(f"News updated successfully ({stored_count}/{len(processed_articles)} articles stored)")
                self.last_news_scrape = now
            else:
                logger.warning("No news articles retrieved")
                
        except Exception as e:
            logger.error(f"Error updating news: {e}")
    
    def update_forecasts(self, force: bool = False):
        """
        Update price forecasts for all configured symbols.
        
        Args:
            force (bool): Force update even if interval hasn't elapsed
        """
        # Check if update is needed
        now = datetime.now()
        interval_hours = self.config.get("forecast_interval_hours", 12)
        
        if not force and self.last_forecast_update and \
           (now - self.last_forecast_update).total_seconds() < interval_hours * 3600:
            logger.info(f"Forecast update skipped (last update: {self.last_forecast_update})")
            return
        
        logger.info("Updating price forecasts")
        
        try:
            # Get symbols from config
            symbols = self.config.get("crypto_symbols", [])
            days_ahead = self.config.get("forecast_days_ahead", 7)
            
            for symbol in symbols:
                try:
                    # Get historical data for forecasting
                    historical_data = self.storage.retrieve_time_series(symbol, interval="1d", limit=100)
                    
                    if not historical_data:
                        logger.warning(f"No historical data available for {symbol}, skipping forecast")
                        continue
                    
                    # Generate forecast
                    forecast = self.processor.forecast_prices(
                        symbol=symbol,
                        historical_data=historical_data,
                        days_ahead=days_ahead,
                        use_sentiment=True
                    )
                    
                    if "error" in forecast:
                        logger.error(f"Error forecasting {symbol}: {forecast['error']}")
                        continue
                    
                    # TODO: Store forecast in Weaviate (when forecast schema is implemented)
                    logger.info(f"Forecast generated for {symbol} ({days_ahead} days ahead)")
                    
                except Exception as e:
                    logger.error(f"Error forecasting {symbol}: {e}")
                    continue
                
                # Slight delay between symbols
                time.sleep(1)
            
            self.last_forecast_update = now
            
        except Exception as e:
            logger.error(f"Error updating forecasts: {e}")
    
    def analyze_onchain(self, address: str, blockchain: str = "ethereum", 
                        related_fund: Optional[str] = None) -> Dict:
        """
        Analyze on-chain data for a specific address.
        
        Args:
            address (str): Blockchain address
            blockchain (str): Blockchain name
            related_fund (str, optional): Related fund name
            
        Returns:
            Dict: Analysis results
        """
        logger.info(f"Analyzing on-chain data for {address} on {blockchain}")
        
        try:
            # Check if onchain_manager is available
            if self.onchain_manager is None:
                # Try direct analysis as a fallback
                try:
                    from Sample_Data.onchain_analytics.analyzers.wallet_analyzer import WalletAnalyzer
                    from Sample_Data.onchain_analytics.models.weaviate_storage import store_wallet_analysis
                    
                    analyzer = WalletAnalyzer()
                    analysis = analyzer.analyze_ethereum_wallet(address)
                    
                    if "error" not in analysis and related_fund:
                        store_wallet_analysis(analysis, related_fund)
                    
                    return analysis
                except Exception as direct_error:
                    logger.error(f"Error in direct wallet analysis: {direct_error}")
                    return {"error": f"OnChain manager not available and direct analysis failed: {str(direct_error)}"}
            
            # Use the onchain manager to analyze and store the data
            return self.onchain_manager.analyze_and_store(address, blockchain, related_fund)
                
        except Exception as e:
            logger.error(f"Error analyzing on-chain data: {e}")
            return {"error": str(e)}
    
    def store_document(self, content: str, filename: str, document_type: Optional[str] = None,
                     title: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Store a document in the CryptoDueDiligenceDocuments collection.
        
        Args:
            content (str): Document content
            filename (str): Source filename
            document_type (str, optional): Document type
            title (str, optional): Document title
            metadata (Dict, optional): Additional metadata
            
        Returns:
            bool: Success status
        """
        logger.info(f"Storing document: {filename}")
        
        try:
            # Determine document type if not provided
            if not document_type:
                document_type = self._infer_document_type(filename)
            
            # Use filename as title if not provided
            if not title:
                title = os.path.splitext(os.path.basename(filename))[0]
            
            # Extract features from document
            features = self.processor.extract_features_from_document({
                "content": content,
                "document_type": document_type
            })
            
            # Prepare document with features
            document = {
                "content": content,
                "source": filename,
                "document_type": document_type,
                "title": title,
                "risk_score": features.get("risk_score", 0)
            }
            
            # Add keywords if available
            if "keywords" in features:
                document["keywords"] = features["keywords"]
            
            # Add additional metadata if provided
            if metadata:
                document.update(metadata)
            
            # Store in Weaviate
            success = self.storage.store_due_diligence_document(document)
            
            if success:
                logger.info(f"Document {filename} stored successfully")
                return True
            else:
                logger.error(f"Failed to store document {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def _infer_document_type(self, filename: str) -> str:
        """
        Infer document type from filename.
        
        Args:
            filename (str): Source filename
            
        Returns:
            str: Inferred document type
        """
        filename_lower = filename.lower()
        
        if "whitepaper" in filename_lower:
            return "whitepaper"
        elif "audit" in filename_lower:
            return "audit_report"
        elif "regulation" in filename_lower or "compliance" in filename_lower:
            return "regulatory_filing"
        elif "report" in filename_lower or "analysis" in filename_lower:
            return "due_diligence_report"
        else:
            return "project_documentation"  # Default
    
    def search(self, query: str, collection: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Search for content across collections.
        
        Args:
            query (str): Search query
            collection (str, optional): Specific collection to search
            limit (int): Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # If collection specified, search only that collection
            if collection:
                return self.storage.retrieve_documents(query, collection_name=collection, limit=limit)
            
            # Otherwise, search across collections
            results = []
            
            # Search CryptoDueDiligenceDocuments
            doc_results = self.storage.retrieve_documents(
                query, collection_name="CryptoDueDiligenceDocuments", limit=limit
            )
            for result in doc_results:
                result["type"] = "document"
                results.append(result)
            
            # Search CryptoNewsSentiment
            news_results = self.storage.retrieve_documents(
                query, collection_name="CryptoNewsSentiment", limit=limit
            )
            for result in news_results:
                result["type"] = "news"
                results.append(result)
            
            # Sort results by relevance (if available)
            if results and "relevance" in results[0]:
                results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            
            # Limit to requested number
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Get latest market data for a symbol.
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            Dict: Market data
        """
        logger.info(f"Getting market data for {symbol}")
        
        try:
            # Format symbol
            if not symbol.upper().endswith("USDT"):
                symbol = f"{symbol.upper()}USDT"
            
            # Get latest data from storage
            results = self.storage.retrieve_market_data(symbol, limit=1)
            
            if results:
                return results[0]
            
            # If no data in storage, fetch from API
            market_data = self.collector.fetch_market_data([symbol.replace("USDT", "")])
            
            if market_data:
                for data in market_data:
                    if data.get("symbol") == symbol:
                        # Store for future queries
                        self.storage.store_market_data(data)
                        return data
            
            return {"error": f"No market data found for {symbol}"}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    def get_historical_data(self, symbol: str, interval: str = "1d", limit: int = 100) -> List[Dict]:
        """
        Get historical time series data for a symbol.
        
        Args:
            symbol (str): Cryptocurrency symbol
            interval (str): Time interval (e.g., "1d", "1h")
            limit (int): Maximum number of data points
            
        Returns:
            List[Dict]: Historical data
        """
        logger.info(f"Getting historical data for {symbol}")
        
        try:
            # Format symbol
            if not symbol.upper().endswith("USDT"):
                symbol = f"{symbol.upper()}USDT"
            
            # Retrieve from Weaviate
            return self.storage.retrieve_time_series(symbol, interval=interval, limit=limit)
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def get_sentiment_analysis(self, asset: str) -> Dict:
        """
        Get sentiment analysis for a specific asset.
        
        Args:
            asset (str): Cryptocurrency asset (e.g., "bitcoin")
            
        Returns:
            Dict: Sentiment analysis results
        """
        logger.info(f"Getting sentiment analysis for {asset}")
        
        try:
            # Get sentiment stats from storage
            return self.storage.get_sentiment_stats(asset, days=7)
            
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return {"error": str(e)}
    
    def start_background_services(self):
        """Start background services for data updates"""
        def background_worker():
            """Background worker function to update data periodically"""
            while True:
                try:
                    # Update market data
                    self.update_market_data()
                    
                    # Update news (less frequently)
                    self.update_news()
                    
                    # Update forecasts (even less frequently)
                    self.update_forecasts()
                    
                except Exception as e:
                    logger.error(f"Error in background worker: {e}")
                
                # Sleep before next update cycle (use market update interval as base)
                time.sleep(self.config.get("market_update_interval_minutes", 15) * 60)
        
        # Start background thread
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
        
        logger.info("Background services started")
        return thread
    
    def run_cli(self):
        """Run an interactive CLI for the system"""
        if not self.collector or not self.processor or not self.storage:
            self.initialize()
        
        # Start background services
        self.start_background_services()
        
        print("\n=== Crypto Due Diligence System ===")
        print("Type 'help' for commands or 'exit' to quit")
        
        while True:
            try:
                command = input("\nCommand: ").strip()
                
                if command.lower() == "exit":
                    print("Exiting...")
                    break
                    
                elif command.lower() == "help":
                    print("\nAvailable commands:")
                    print("  market <symbol>   - Get market data for symbol")
                    print("  history <symbol>  - Get historical data for symbol")
                    print("  news              - Update news data")
                    print("  sentiment <asset> - Get sentiment analysis for asset")
                    print("  forecast <symbol> - Get price forecast for symbol")
                    print("  onchain <address> - Analyze blockchain address")
                    print("  search <query>    - Search for content")
                    print("  document <file>   - Store a document")
                    print("  load-csv          - Load historical data from CSV files")
                    print("  exit              - Exit the program")
                
                elif command.lower().startswith("market "):
                    symbol = command.split(" ", 1)[1]
                    result = self.get_market_data(symbol)
                    print(json.dumps(result, indent=2))
                
                elif command.lower().startswith("history "):
                    symbol = command.split(" ", 1)[1]
                    result = self.get_historical_data(symbol, limit=10)  # Limit to 10 for display
                    print(json.dumps(result, indent=2))
                
                elif command.lower() == "news":
                    self.update_news(force=True)
                    print("News update initiated")
                
                elif command.lower().startswith("sentiment "):
                    asset = command.split(" ", 1)[1]
                    result = self.get_sentiment_analysis(asset)
                    print(json.dumps(result, indent=2))
                
                elif command.lower().startswith("forecast "):
                    symbol = command.split(" ", 1)[1]
                    
                    # Format symbol
                    if not symbol.upper().endswith("USDT"):
                        symbol = f"{symbol.upper()}USDT"
                    
                    # Get historical data
                    historical_data = self.get_historical_data(symbol, interval="1d", limit=100)
                    
                    if not historical_data:
                        print(f"No historical data found for {symbol}")
                        continue
                    
                    # Generate forecast
                    forecast = self.processor.forecast_prices(
                        symbol=symbol,
                        historical_data=historical_data,
                        days_ahead=self.config.get("forecast_days_ahead", 7),
                        use_sentiment=True
                    )
                    
                    print(json.dumps(forecast, indent=2))
                
                elif command.lower().startswith("onchain "):
                    address = command.split(" ", 1)[1]
                    result = self.analyze_onchain(address)
                    print(json.dumps(result, indent=2))
                
                elif command.lower().startswith("search "):
                    query = command.split(" ", 1)[1]
                    results = self.search(query)
                    
                    if results:
                        print(f"Found {len(results)} results:")
                        for i, result in enumerate(results):
                            print(f"\n[{i+1}] {result.get('title', 'Untitled')} ({result.get('type', 'unknown')})")
                            if "content" in result:
                                content = result["content"]
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                print(f"  {content}")
                    else:
                        print("No results found")
                
                elif command.lower().startswith("document "):
                    file_path = command.split(" ", 1)[1]
                    
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        success = self.store_document(
                            content=content,
                            filename=os.path.basename(file_path)
                        )
                        
                        if success:
                            print(f"Document stored successfully: {file_path}")
                        else:
                            print(f"Failed to store document: {file_path}")
                            
                    except Exception as e:
                        print(f"Error: {e}")
                
                elif command.lower() == "load-csv":
                    print("Loading historical data from CSV files...")
                    self.update_historical_data(force=False)
                    print("CSV loading complete")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Clean up
        self.storage.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crypto Due Diligence System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--setup", action="store_true", help="Set up schemas only")
    parser.add_argument("--update-market", action="store_true", help="Update market data")
    parser.add_argument("--update-news", action="store_true", help="Update news data")
    parser.add_argument("--load-csv", action="store_true", help="Load historical data from CSV files")
    parser.add_argument("--interactive", action="store_true", help="Run interactive CLI")
    
    args = parser.parse_args()
    
    system = CryptoDueDiligenceSystem(config_path=args.config)
    
    # Initialize components
    if not system.initialize():
        logger.error("Failed to initialize system")
        return 1
    
    try:
        if args.setup:
            # Only set up schemas
            logger.info("Setup complete")
            return 0
        
        if args.update_market:
            system.update_market_data(force=True)
        
        if args.update_news:
            system.update_news(force=True)
        
        if args.load_csv:
            system.update_historical_data(force=True)
        
        if args.interactive or not any([args.setup, args.update_market, args.update_news, args.load_csv]):
            # Run interactive CLI
            system.run_cli()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        # Clean up
        if system.storage:
            system.storage.close()

if __name__ == "__main__":
    sys.exit(main())