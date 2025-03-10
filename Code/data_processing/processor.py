# File path: Code/data_processing/processor.py
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class DataProcessor:
    """
    Processes data for the crypto due diligence platform:
    - Sentiment analysis for news articles
    - Time series forecasting for price data
    - Feature extraction for on-chain analytics
    """
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize the data processor
        
        Args:
            use_finbert (bool): Whether to use FinBERT for sentiment analysis
        """
        self.models_dir = os.path.join(project_root, "data", "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.use_finbert = use_finbert
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        # Initialize sentiment model if needed
        if self.use_finbert:
            self._initialize_sentiment_model()
    
    def _initialize_sentiment_model(self):
        """Initialize FinBERT sentiment model"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = "yiyanghkust/finbert-tone"
            logger.info(f"Loading FinBERT sentiment model: {model_name}")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info("FinBERT sentiment model loaded successfully")
            return True
        except ImportError:
            logger.warning("Transformers library not available. Falling back to rule-based sentiment.")
            self.use_finbert = False
            return False
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            self.use_finbert = False
            return False
    
    def analyze_sentiment(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of text using FinBERT or rule-based approach
        
        Args:
            text (str): Text to analyze
            title (str, optional): Title of the article for enhanced analysis
            
        Returns:
            Dict: Sentiment analysis results including label and score
        """
        # Combine title and text if both provided
        if title:
            combined_text = f"{title} {text}"
        else:
            combined_text = text
            
        # Use FinBERT if available
        if self.use_finbert and self.finbert_model and self.finbert_tokenizer:
            try:
                # Truncate text to fit BERT's maximum length
                max_length = 512
                
                # Tokenize the text
                inputs = self.finbert_tokenizer(combined_text, return_tensors="pt", 
                                               padding=True, truncation=True, max_length=max_length)
                
                # Get sentiment predictions
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(scores, dim=1)
                
                # Map the prediction to sentiment label
                # FinBERT tone model has 3 labels: 0 = neutral, 1 = positive, 2 = negative
                sentiment_map = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
                sentiment_label = sentiment_map[predictions.item()]
                
                # Get the probability scores
                probs = scores.squeeze().tolist()
                
                # Normalize to get a score between 0-1 where:
                # 0 = most negative, 0.5 = neutral, 1 = most positive
                # Positive score minus negative score, normalized to 0-1 range
                sentiment_score = (probs[1] - probs[2] + 1) / 2
                
                return {
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "confidence": max(probs),
                    "raw_scores": {
                        "neutral": probs[0],
                        "positive": probs[1],
                        "negative": probs[2]
                    },
                    "method": "finbert"
                }
                
            except Exception as e:
                logger.error(f"Error in FinBERT sentiment analysis: {e}")
                # Fall back to rule-based approach
                
        # Use rule-based approach
        return self._rule_based_sentiment(combined_text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based sentiment analysis
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        text_lower = text.lower()
        
        # Define sentiment lexicons
        positive_words = [
            "bullish", "growth", "profit", "gain", "surge", "rise", "positive", 
            "promising", "success", "opportunity", "uptrend", "breakthrough", 
            "milestone", "achievement", "progress", "improve", "advantage",
            "rally", "boom", "soar", "jump", "recovery", "innovation", "partnership",
            "adoption", "agreement", "prospect", "potential", "upgrade"
        ]
        
        negative_words = [
            "bearish", "crash", "loss", "decline", "fall", "drop", "negative", 
            "concern", "risk", "threat", "downtrend", "failure", "problem", 
            "issue", "disaster", "crisis", "danger", "worry", "bear", "sell",
            "plunge", "plummet", "tumble", "weaken", "fear", "uncertainty",
            "volatility", "vulnerability", "weakness", "downgrade", "regulatory",
            "investigation", "lawsuit", "fine", "penalty", "scam", "fraud"
        ]
        
        # Count word occurrences
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate sentiment score (0-1 range)
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0.5  # Neutral
        else:
            sentiment_score = positive_count / total_count
            
        # Determine sentiment label
        if sentiment_score > 0.6:
            sentiment_label = "POSITIVE"
        elif sentiment_score < 0.4:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
            
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "confidence": max(positive_count, negative_count) / (total_count + 1),
            "method": "rule-based"
        }
    
    def analyze_news_batch(self, news_articles: List[Dict]) -> List[Dict]:
        """
        Process a batch of news articles with sentiment analysis
        
        Args:
            news_articles (List[Dict]): List of news articles
            
        Returns:
            List[Dict]: Processed articles with sentiment analysis
        """
        logger.info(f"Analyzing sentiment for {len(news_articles)} news articles")
        
        processed_articles = []
        
        for i, article in enumerate(news_articles):
            try:
                # Extract content and title
                content = article.get("content", "")
                title = article.get("title", "")
                
                if not content and not title:
                    logger.warning(f"Skipping article {i} - no content or title")
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(content, title)
                
                # Copy the article and add sentiment analysis
                processed_article = article.copy()
                processed_article.update({
                    "sentiment_label": sentiment["sentiment_label"],
                    "sentiment_score": sentiment["sentiment_score"],
                    "analyzed_at": datetime.now().isoformat()
                })
                
                # Extract crypto assets mentioned
                processed_article["related_assets"] = self._extract_crypto_mentions(content, title)
                
                processed_articles.append(processed_article)
                
                # Log progress for every 10 articles
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(news_articles)} articles")
                    
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue
        
        logger.info(f"Completed sentiment analysis for {len(processed_articles)}/{len(news_articles)} articles")
        return processed_articles
    
    def _extract_crypto_mentions(self, content: str, title: Optional[str] = None) -> List[str]:
        """
        Extract cryptocurrency mentions from text
        
        Args:
            content (str): Article content
            title (str, optional): Article title
            
        Returns:
            List[str]: List of cryptocurrencies mentioned
        """
        # Combine title and content if title is provided
        if title:
            text = f"{title} {content}"
        else:
            text = content
            
        text_lower = text.lower()
        
        # Common cryptocurrencies to look for
        crypto_terms = {
            "bitcoin": ["bitcoin", "btc", "satoshi"],
            "ethereum": ["ethereum", "eth", "ether", "erc20", "erc721"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb", "bsc"],
            "cardano": ["cardano", "ada"],
            "ripple": ["ripple", "xrp"],
            "dogecoin": ["dogecoin", "doge"],
            "polkadot": ["polkadot", "dot"],
            "avalanche": ["avalanche", "avax"],
            "tron": ["tron", "trx"],
            "litecoin": ["litecoin", "ltc"],
            "chainlink": ["chainlink", "link"],
            "polygon": ["polygon", "matic"],
            "uniswap": ["uniswap", "uni"],
            "stablecoin": ["usdt", "tether", "usdc", "dai", "busd"]
        }
        
        # Find mentions
        mentions = []
        for crypto, terms in crypto_terms.items():
            for term in terms:
                if term in text_lower:
                    mentions.append(crypto)
                    break  # Only count each crypto once
        
        return mentions
    
    def forecast_prices(self, symbol: str, historical_data: List[Dict], 
                       days_ahead: int = 7, use_sentiment: bool = True) -> Dict[str, Any]:
        """
        Forecast cryptocurrency prices
        
        Args:
            symbol (str): Cryptocurrency symbol
            historical_data (List[Dict]): Historical price data
            days_ahead (int): Number of days to forecast
            use_sentiment (bool): Whether to incorporate sentiment data
            
        Returns:
            Dict: Forecast results
        """
        logger.info(f"Forecasting prices for {symbol} ({days_ahead} days ahead)")
        
        # Convert to DataFrame
        if not historical_data:
            logger.error(f"No historical data provided for {symbol}")
            return {"error": "No historical data provided"}
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Ensure required columns exist
            required_columns = ["timestamp", "close", "volume"]
            for column in required_columns:
                if column not in df.columns:
                    logger.error(f"Missing required column: {column}")
                    return {"error": f"Missing required column: {column}"}
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Check if we have enough data
            if len(df) < 30:  # Need at least 30 data points
                logger.warning(f"Insufficient historical data for {symbol}: {len(df)} points")
                return {"error": "Insufficient historical data (need at least 30 data points)"}
            
            # Prepare features
            features_df = self._prepare_forecast_features(df)
            
            # Get sentiment data if requested
            if use_sentiment:
                sentiment_df = self._get_sentiment_for_symbol(symbol)
                if sentiment_df is not None and not sentiment_df.empty:
                    # Merge sentiment data
                    features_df = self._add_sentiment_features(features_df, sentiment_df)
            
            # Train or load forecast model
            model = self._get_forecast_model(symbol, features_df)
            
            # Make predictions
            predictions = self._generate_predictions(model, features_df, days_ahead)
            
            # Add metadata
            result = {
                "symbol": symbol,
                "forecast_date": datetime.now().isoformat(),
                "days_ahead": days_ahead,
                "predictions": predictions,
                "use_sentiment": use_sentiment
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error forecasting prices for {symbol}: {e}")
            return {"error": str(e)}
    
    def _prepare_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for price forecasting
        
        Args:
            df (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: DataFrame with prepared features
        """
        # Create copy to avoid modifying the original
        features_df = df.copy()
        
        # Calculate technical indicators
        # Moving averages
        features_df['ma_7'] = features_df['close'].rolling(window=7).mean()
        features_df['ma_14'] = features_df['close'].rolling(window=14).mean()
        features_df['ma_30'] = features_df['close'].rolling(window=30).mean()
        
        # Exponential moving averages
        features_df['ema_12'] = features_df['close'].ewm(span=12, adjust=False).mean()
        features_df['ema_26'] = features_df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        features_df['macd'] = features_df['ema_12'] - features_df['ema_26']
        features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        features_df['ma_20'] = features_df['close'].rolling(window=20).mean()
        features_df['std_20'] = features_df['close'].rolling(window=20).std()
        features_df['upper_band'] = features_df['ma_20'] + (features_df['std_20'] * 2)
        features_df['lower_band'] = features_df['ma_20'] - (features_df['std_20'] * 2)
        features_df['bb_width'] = (features_df['upper_band'] - features_df['lower_band']) / features_df['ma_20']
        
        # Rate of change
        features_df['price_roc'] = features_df['close'].pct_change(periods=1)
        features_df['volume_roc'] = features_df['volume'].pct_change(periods=1)
        
        # Relative Strength Index (RSI)
        delta = features_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target variable (next day's price)
        features_df['target'] = features_df['close'].shift(-1)
        
        # Filter out rows with NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def _get_sentiment_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get sentiment data for a specific symbol
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            pd.DataFrame or None: Sentiment data
        """
        # This is a placeholder - in a real implementation, you'd query your sentiment database
        # For demo purposes, we'll just return None
        logger.info(f"Getting sentiment data for {symbol} (placeholder)")
        return None
    
    def _add_sentiment_features(self, features_df: pd.DataFrame, 
                               sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment features to the forecast features
        
        Args:
            features_df (pd.DataFrame): DataFrame with forecast features
            sentiment_df (pd.DataFrame): DataFrame with sentiment data
            
        Returns:
            pd.DataFrame: Combined features
        """
        # This is a placeholder - in a real implementation, you'd merge the sentiment data
        logger.info("Adding sentiment features (placeholder)")
        return features_df
    
    def _get_forecast_model(self, symbol: str, features_df: pd.DataFrame):
        """
        Get or train a forecast model for the symbol
        
        Args:
            symbol (str): Cryptocurrency symbol
            features_df (pd.DataFrame): Features for training
            
        Returns:
            object: Trained model
        """
        # In a real implementation, you'd train a proper ML model
        # For demo purposes, we'll use a simple moving average model
        logger.info(f"Getting forecast model for {symbol} (using simple MA)")
        
        # Create a simple moving average "model"
        ma_periods = [7, 14, 30]
        weights = [0.5, 0.3, 0.2]  # Higher weight for recent data
        
        # Calculate weighted average of MAs
        recent_ma = {}
        for period, weight in zip(ma_periods, weights):
            ma_col = f'ma_{period}'
            if ma_col in features_df.columns:
                recent_ma[period] = features_df[ma_col].iloc[-1]
        
        # Simple "model" is just the parameters for prediction
        model = {
            "type": "weighted_ma",
            "recent_price": features_df['close'].iloc[-1],
            "recent_ma": recent_ma,
            "ma_weights": weights,
            "avg_daily_change": features_df['price_roc'].mean(),
            "volatility": features_df['price_roc'].std()
        }
        
        return model
    
    def _generate_predictions(self, model, features_df: pd.DataFrame, days_ahead: int) -> List[Dict]:
        """
        Generate price predictions using the model
        
        Args:
            model: Trained forecasting model
            features_df (pd.DataFrame): Features data
            days_ahead (int): Number of days to forecast
            
        Returns:
            List[Dict]: Price predictions
        """
        # Get the last date in the data
        last_date = features_df['timestamp'].iloc[-1]
        
        # Get the last price
        current_price = features_df['close'].iloc[-1]
        
        # For our simple model, just use weighted MA with some randomness
        predictions = []
        
        for day in range(1, days_ahead + 1):
            # Calculate prediction date
            prediction_date = last_date + timedelta(days=day)
            
            # For simple model, add some trending based on recent performance
            if model["type"] == "weighted_ma":
                # Calculate weighted average of recent MAs
                weighted_ma = 0
                total_weight = 0
                
                for period, weight in zip(model["recent_ma"].keys(), model["ma_weights"]):
                    weighted_ma += model["recent_ma"][period] * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_ma /= total_weight
                else:
                    weighted_ma = current_price
                
                # Add trend component
                trend = 1.0 + (model["avg_daily_change"] * day)
                
                # Add volatility component (random)
                volatility = model["volatility"] * np.random.normal(0, 1)
                
                # Calculate predicted price
                predicted_price = weighted_ma * trend * (1 + volatility * 0.5)
                
                # Ensure price is positive
                predicted_price = max(0.01, predicted_price)
                
                # Calculate change percentage from current price
                change_pct = ((predicted_price - current_price) / current_price) * 100
                
                # Add confidence bounds
                lower_bound = predicted_price * (1 - (day * 0.01) - (volatility * 0.1))
                upper_bound = predicted_price * (1 + (day * 0.01) + (volatility * 0.1))
                
                predictions.append({
                    "date": prediction_date.isoformat(),
                    "predicted_price": predicted_price,
                    "change_pct": change_pct,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound
                })
        
        return predictions
    
    def extract_features_from_document(self, document: Dict) -> Dict[str, Any]:
        """
        Extract features from a document for due diligence
        
        Args:
            document (Dict): Document data
            
        Returns:
            Dict: Extracted features
        """
        # Import feature extractor if available
        try:
            from Sample_Data.feature_extraction import CryptoFeatureExtractor
            extractor = CryptoFeatureExtractor()
            
            # Extract features
            content = document.get("content", "")
            document_type = document.get("document_type")
            
            if not content:
                return {"error": "No document content provided"}
                
            features = extractor.extract_features(content, document_type)
            return features
            
        except ImportError:
            logger.warning("CryptoFeatureExtractor not available. Using basic extraction.")
            return self._basic_feature_extraction(document)
    
    def _basic_feature_extraction(self, document: Dict) -> Dict[str, Any]:
        """
        Basic feature extraction for documents
        
        Args:
            document (Dict): Document data
            
        Returns:
            Dict: Basic extracted features
        """
        content = document.get("content", "")
        
        if not content:
            return {"error": "No document content provided"}
            
        # Basic text statistics
        word_count = len(content.split())
        
        # Extract cryptocurrency mentions
        crypto_mentions = {}
        content_lower = content.lower()
        
        # Common cryptocurrencies to look for
        crypto_terms = {
            "bitcoin": ["bitcoin", "btc"],
            "ethereum": ["ethereum", "eth"],
            "solana": ["solana", "sol"],
            "binance": ["binance", "bnb"],
            "cardano": ["cardano", "ada"]
        }
        
        for crypto, terms in crypto_terms.items():
            for term in terms:
                count = content_lower.count(term)
                if count > 0:
                    crypto_mentions[crypto] = count
        
        # Risk assessment
        risk_terms = ["risk", "fraud", "scam", "hack", "vulnerability", "breach", "attack", 
                     "liability", "loss", "security"]
        risk_count = sum(content_lower.count(term) for term in risk_terms)
        
        # Very simple risk score calculation
        risk_score = min(100, (risk_count / word_count) * 1000) if word_count > 0 else 0
        
        return {
            "word_count": word_count,
            "crypto_mentions": crypto_mentions,
            "risk_score": risk_score,
            "risk_count": risk_count
        }
    
    def process_onchain_data(self, data: Dict) -> Dict:
        """
        Process on-chain analytics data
        
        Args:
            data (Dict): Raw on-chain data
            
        Returns:
            Dict: Processed on-chain analytics
        """
        # For now, just return the data as-is
        # In a real implementation, you might add more analysis or features
        return data

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Test sentiment analysis
    sentiment = processor.analyze_sentiment(
        "Bitcoin price surges to new all-time high as institutional adoption increases."
    )
    print(f"Sentiment: {sentiment['sentiment_label']}, Score: {sentiment['sentiment_score']:.2f}")
    
    # Test document feature extraction
    doc = {
        "content": "This whitepaper introduces a new cryptocurrency that aims to revolutionize the payment industry. Bitcoin and Ethereum have paved the way for blockchain technology, but our solution addresses their limitations with a more scalable and energy-efficient approach.",
        "document_type": "whitepaper"
    }
    features = processor.extract_features_from_document(doc)
    print(f"Document features: {features}")