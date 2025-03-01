#!/usr/bin/env python
import os
import logging
from datetime import datetime
import argparse
import sys
from pathlib import Path
import pandas as pd
import threading
import time
import traceback
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing optional LangChain components
LANGCHAIN_AVAILABLE = False
try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.tools import Tool
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
    logger.info("LangChain successfully imported")
except ImportError:
    logger.warning("LangChain not available. Agentic features will be limited.")

class AgenticRagSystem:
    """Orchestrates the agentic RAG system for crypto due diligence"""
    
    def __init__(self):
        """Initialize components and tools"""
        self.client = None
        self.scraper = None
        self.sentiment_analyzer = None
        self.forecaster = None
        self.agent_executor = None
        self.config = self._load_config()
        
        # Track system state
        self.is_initialized = False
        self.last_scrape_time = None
        self.last_sentiment_time = None
        self.last_forecast_time = None
    
    def _load_config(self):
        """Load configuration from config file"""
        config_path = os.path.join(project_root, "config.json")
        default_config = {
            "crypto_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"],
            "news_sources": ["CoinDesk", "Cointelegraph"],
            "scrape_interval_hours": 4,
            "sentiment_interval_hours": 6,
            "forecast_interval_hours": 12,
            "forecast_days_ahead": 7,
            "use_openai": False,
            "data_dir": "data"
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    # Update with any missing default values
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Save default config
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def initialize(self):
        """Initialize all components of the system"""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing Agentic RAG System...")
            
            # Create data directory
            os.makedirs(self.config["data_dir"], exist_ok=True)
            
            # Initialize Weaviate client
            from Sample_Data.vector_store.weaviate_client import get_weaviate_client
            self.client = get_weaviate_client()
            
            # Initialize components
            from Code.data_acquisition.blockchain_collectors.news_scraper import CryptoNewsScraper
            from Code.data_processing.sentiment_analyzer import CryptoSentimentAnalyzer
            from Code.data_processing.crypto_forecaster import CryptoForecaster
            
            self.scraper = CryptoNewsScraper()
            self.sentiment_analyzer = CryptoSentimentAnalyzer()
            self.forecaster = CryptoForecaster(model_dir=os.path.join(self.config["data_dir"], "models"))
            
            # Set up LangChain agent tools if available
            if LANGCHAIN_AVAILABLE and self.config["use_openai"]:
                self._setup_agent_tools()
            
            self.is_initialized = True
            logger.info("Agentic RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _setup_agent_tools(self):
        """Set up LangChain agent tools for the agentic RAG system"""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available. Skipping agent setup.")
            return
            
        # Define tools for the agent
        tools = [
            Tool(
                name="search_crypto_documents",
                func=lambda query: self._search_documents(query),
                description="Search for information in the crypto due diligence documents. Input should be a search query."
            ),
            Tool(
                name="get_market_data",
                func=lambda symbol: self._get_market_data(symbol),
                description="Get current market data for a cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT)."
            ),
            Tool(
                name="get_sentiment_analysis",
                func=lambda symbol: self._get_sentiment(symbol),
                description="Get recent sentiment analysis for a cryptocurrency (e.g., BTC, ETH)."
            ),
            Tool(
                name="get_price_forecast",
                func=lambda symbol: self._get_forecast(symbol),
                description="Get price forecast for a cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT)."
            )
        ]
        
        # Only create the LangChain agent if OpenAI is enabled in config
        if self.config["use_openai"]:
            try:
                # Create LangChain agent
                llm = ChatOpenAI(temperature=0)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a crypto due diligence agent with access to market data, news sentiment, and forecasting tools. Help users make informed decisions about crypto investments."),
                    ("user", "{input}")
                ])
                agent = create_openai_functions_agent(llm, tools, prompt)
                self.agent_executor = AgentExecutor(agent=agent, tools=tools)
                logger.info("LangChain agent set up successfully")
            except Exception as e:
                logger.error(f"Error setting up LangChain agent: {e}")
                logger.info("LangChain agent will not be available")
        else:
            logger.info("OpenAI integration disabled in config")
    
    def _search_documents(self, query):
        """Search crypto documents based on a query"""
        try:
            from Sample_Data.retrieval.search import search_documents
            results = search_documents(self.client, query, top_k=5)
            
            if not results:
                return "No matching documents found."
                
            formatted_results = []
            for idx, doc in enumerate(results):
                content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                formatted_results.append(f"Result {idx + 1} from {doc['source']}:\n{content_preview}\n")
                
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Document search error: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _get_market_data(self, symbol):
        """Get current market data for a crypto symbol"""
        try:
            # Query MarketMetrics collection
            collection = self.client.collections.get("MarketMetrics")
            
            # Build query
            from weaviate.classes.query import Filter
            response = collection.query.fetch_objects(
                filters=Filter.by_property("symbol").equal(symbol),
                return_properties=["symbol", "price", "market_cap", 
                                  "volume_24h", "price_change_24h", "timestamp"],
                limit=1,
                sort=[{"path": ["timestamp"], "order": "desc"}]
            )
            
            if not response.objects:
                return f"No market data found for {symbol}"
                
            data = response.objects[0].properties
            
            formatted_response = (
                f"Market data for {symbol}:\n"
                f"Price: ${data['price']:,.2f}\n"
                f"Market Cap: ${data['market_cap']:,.2f}\n"
                f"24h Volume: ${data['volume_24h']:,.2f}\n"
                f"24h Change: {data['price_change_24h']}%\n"
                f"Timestamp: {data['timestamp']}"
            )
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return f"Error getting market data: {str(e)}"
    
    def _get_sentiment(self, symbol):
        """Get recent sentiment analysis for a crypto symbol"""
        try:
            from Sample_Data.vector_store.sentiment_store import get_sentiment_stats
            
            # Get sentiment stats for the symbol
            stats = get_sentiment_stats(symbol, days=7)
            
            if stats["total_articles"] == 0:
                return f"No sentiment data found for {symbol}"
            
            # Format the response
            formatted_response = f"Sentiment analysis for {symbol}:\n"
            formatted_response += f"Overall sentiment score: {stats['avg_sentiment']:.2f}\n"
            
            # Convert score to label
            if stats['avg_sentiment'] > 0.6:
                sentiment_label = "POSITIVE"
            elif stats['avg_sentiment'] < 0.4:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
                
            formatted_response += f"Overall sentiment: {sentiment_label}\n"
            formatted_response += f"Based on {stats['total_articles']} recent articles\n\n"
            
            # Add distribution
            formatted_response += "Sentiment distribution:\n"
            for label, count in stats['sentiment_distribution'].items():
                percentage = (count / stats['total_articles']) * 100 if stats['total_articles'] > 0 else 0
                formatted_response += f"{label}: {count} articles ({percentage:.1f}%)\n"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return f"Error getting sentiment analysis: {str(e)}"
    
    def _get_forecast(self, symbol):
        """Get price forecast for a crypto symbol"""
        try:
            predictions = self.forecaster.predict(symbol, days_ahead=self.config["forecast_days_ahead"])
            
            if predictions is None:
                return f"No forecast available for {symbol}"
            
            # Create forecast response
            formatted_response = f"Price forecast for {symbol} (next {len(predictions)} days):\n"
            
            for _, row in predictions.iterrows():
                formatted_response += f"- {row['date'].strftime('%Y-%m-%d')}: ${row['predicted_price']:.2f} "
                formatted_response += f"({row['change_pct']:+.2f}%)\n"
            
            # Add path to saved plot
            plot_path = f"{symbol}_forecast.png"
            if os.path.exists(plot_path):
                formatted_response += f"\nForecast chart saved to: {plot_path}"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return f"Error generating forecast: {str(e)}"
    
    def scrape_news(self):
        """Scrape crypto news"""
        if not self.is_initialized:
            self.initialize()
            
        logger.info("Starting news scraping...")
        
        try:
            # Set output file in data directory
            output_file = os.path.join(self.config["data_dir"], "crypto_news.csv")
            
            # Run scraper
            news_df = self.scraper.run(limit_per_source=10, output_file=output_file)
            
            self.last_scrape_time = datetime.now()
            logger.info(f"News scraping completed. Scraped {len(news_df)} articles.")
            
            return news_df
            
        except Exception as e:
            logger.error(f"News scraping error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def analyze_sentiment(self):
        """Analyze sentiment of scraped news"""
        if not self.is_initialized:
            self.initialize()
            
        logger.info("Starting sentiment analysis...")
        
        try:
            # Check if we have news data
            input_file = os.path.join(self.config["data_dir"], "crypto_news.csv")
            if not os.path.exists(input_file):
                logger.warning("No news data found. Run scrape_news first.")
                return None
                
            # Run sentiment analysis
            results_df = self.sentiment_analyzer.run(input_file=input_file, store_in_db=True)
            
            self.last_sentiment_time = datetime.now()
            logger.info(f"Sentiment analysis completed. Analyzed {len(results_df)} articles.")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def update_forecasts(self):
        """Update price forecasts for configured symbols"""
        if not self.is_initialized:
            self.initialize()
            
        logger.info("Updating price forecasts...")
        
        results = {}
        for symbol in self.config["crypto_symbols"]:
            try:
                logger.info(f"Generating forecast for {symbol}...")
                
                # Train/update model
                self.forecaster.train(symbol)
                
                # Generate predictions
                predictions = self.forecaster.predict(symbol, days_ahead=self.config["forecast_days_ahead"])
                
                if predictions is not None:
                    # Plot predictions
                    self.forecaster.plot_predictions(symbol, predictions)
                    results[symbol] = predictions
                    logger.info(f"Forecast for {symbol} completed successfully")
                else:
                    logger.warning(f"No predictions generated for {symbol}")
                
            except Exception as e:
                logger.error(f"Error forecasting {symbol}: {e}")
                continue
        
        self.last_forecast_time = datetime.now()
        logger.info(f"Forecast updates completed for {len(results)} symbols")
        
        return results
    
    def process_user_query(self, query):
        """Process a user query using the agent"""
        if not self.is_initialized:
            self.initialize()
            
        # If LangChain agent is available, use it
        if self.agent_executor:
            try:
                logger.info(f"Processing user query with agent: {query}")
                result = self.agent_executor.invoke({"input": query})
                return result["output"]
            except Exception as e:
                logger.error(f"Agent error: {e}")
                # Fall back to simple document search if agent fails
                logger.info("Falling back to simple document search")
                return self._search_documents(query)
        else:
            # If no agent is available, just do a document search
            logger.info(f"Processing user query with document search: {query}")
            return self._search_documents(query)
    
    def store_document(self, text, filename, document_type=None):
        """Store a document in the vector store"""
        if not self.is_initialized:
            self.initialize()
            
        try:
            from Sample_Data.vector_store.store import store_document
            
            logger.info(f"Storing document: {filename}")
            store_document(self.client, text, filename, document_type)
            logger.info(f"Document stored successfully: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def start_background_services(self):
        """Start background services for periodic data updates"""
        if not self.is_initialized:
            self.initialize()
            
        def background_worker():
            while True:
                try:
                    # Check if we need to scrape news
                    if (self.last_scrape_time is None or 
                        (datetime.now() - self.last_scrape_time).total_seconds() > 
                        self.config["scrape_interval_hours"] * 3600):
                        logger.info("Starting scheduled news scraping")
                        self.scrape_news()
                    
                    # Check if we need to analyze sentiment
                    if (self.last_sentiment_time is None or 
                        (datetime.now() - self.last_sentiment_time).total_seconds() > 
                        self.config["sentiment_interval_hours"] * 3600):
                        logger.info("Starting scheduled sentiment analysis")
                        self.analyze_sentiment()
                    
                    # Check if we need to update forecasts
                    if (self.last_forecast_time is None or 
                        (datetime.now() - self.last_forecast_time).total_seconds() > 
                        self.config["forecast_interval_hours"] * 3600):
                        logger.info("Starting scheduled forecast updates")
                        self.update_forecasts()
                    
                except Exception as e:
                    logger.error(f"Background service error: {e}")
                    logger.error(traceback.format_exc())
                
                # Sleep for 15 minutes before checking again
                time.sleep(15 * 60)
        
        # Start background thread
        logger.info("Starting background services")
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
        
        return thread
    
    def run_cli(self):
        """Run the system in command-line interface mode"""
        if not self.is_initialized:
            self.initialize()
            
        # Start background services
        self.start_background_services()
        
        print("\n=== Crypto Due Diligence Agentic RAG System ===")
        print("Type 'help' for available commands or 'exit' to quit")
        
        while True:
            try:
                command = input("\nEnter command or query: ").strip()
                
                if command.lower() == 'exit':
                    break
                    
                elif command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  scrape      - Scrape latest crypto news")
                    print("  sentiment   - Analyze sentiment of scraped news")
                    print("  forecast    - Update price forecasts")
                    print("  market BTC  - Get market data (replace BTC with any symbol)")
                    print("  sentiment BTC - Get sentiment analysis (replace BTC with any symbol)")
                    print("  forecast BTC - Get price forecast (replace BTC with any symbol)")
                    print("  store [file] [type] - Store a document (provide path and optional type)")
                    print("  exit        - Exit the program")
                    print("\nOr enter any question to search the knowledge base")
                    
                elif command.lower() == 'scrape':
                    news_df = self.scrape_news()
                    if news_df is not None:
                        print(f"Scraped {len(news_df)} articles")
                        
                elif command.lower() == 'sentiment':
                    results_df = self.analyze_sentiment()
                    if results_df is not None:
                        sentiment_counts = results_df['sentiment_label'].value_counts()
                        print(f"Sentiment analysis results:")
                        for label, count in sentiment_counts.items():
                            print(f"  {label}: {count}")
                            
                elif command.lower() == 'forecast':
                    results = self.update_forecasts()
                    if results:
                        print(f"Updated forecasts for {len(results)} symbols")
                        
                elif command.lower().startswith('market '):
                    symbol = command.split(' ')[1].upper()
                    result = self._get_market_data(symbol)
                    print(result)
                    
                elif command.lower().startswith('sentiment '):
                    symbol = command.split(' ')[1].upper()
                    result = self._get_sentiment(symbol)
                    print(result)
                    
                elif command.lower().startswith('forecast '):
                    symbol = command.split(' ')[1].upper()
                    if not symbol.endswith('USDT'):
                        symbol = f"{symbol}USDT"
                    result = self._get_forecast(symbol)
                    print(result)
                    
                elif command.lower().startswith('store '):
                    parts = command.split(' ')
                    if len(parts) < 2:
                        print("Usage: store [file] [type]")
                        continue
                        
                    file_path = parts[1]
                    doc_type = parts[2] if len(parts) > 2 else None
                    
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                        
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            
                        success = self.store_document(text, os.path.basename(file_path), doc_type)
                        if success:
                            print(f"Document stored successfully: {file_path}")
                        else:
                            print(f"Failed to store document: {file_path}")
                    except Exception as e:
                        print(f"Error reading file: {e}")
                    
                else:
                    # Assume it's a query
                    result = self.process_user_query(command)
                    print(result)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
        print("Exiting...")
        # Close Weaviate client
        if self.client:
            self.client.close()

def main():
    parser = argparse.ArgumentParser(description="Crypto Due Diligence Agentic RAG System")
    parser.add_argument('--scrape', action='store_true', help='Scrape latest crypto news')
    parser.add_argument('--sentiment', action='store_true', help='Analyze sentiment of scraped news')
    parser.add_argument('--forecast', action='store_true', help='Update price forecasts')
    parser.add_argument('--query', type=str, help='Process a user query')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive CLI mode')
    
    args = parser.parse_args()
    
    system = AgenticRagSystem()
    system.initialize()
    
    try:
        if args.scrape:
            system.scrape_news()
            
        if args.sentiment:
            system.analyze_sentiment()
            
        if args.forecast:
            system.update_forecasts()
            
        if args.query:
            result = system.process_user_query(args.query)
            print(result)
            
        if args.interactive or not any([args.scrape, args.sentiment, args.forecast, args.query]):
            system.run_cli()
            
    finally:
        # Clean up
        if system.client:
            system.client.close()

if __name__ == "__main__":
    main()