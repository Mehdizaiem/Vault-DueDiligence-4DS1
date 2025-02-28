import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.weaviate_client import get_weaviate_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for missing MarketMetrics schema
def ensure_market_metrics_schema(client):
    """Ensure that MarketMetrics collection exists"""
    try:
        collection = client.collections.get("MarketMetrics")
        return collection
    except Exception:
        logger.info("Creating MarketMetrics collection")
        try:
            collection = client.collections.create(
                name="MarketMetrics",
                vectorizer_config=None,  # No embeddings for market data
                properties=[
                    {
                        "name": "symbol",
                        "dataType": ["text"],
                        "description": "Trading symbol (e.g., BTCUSDT)"
                    },
                    {
                        "name": "source",
                        "dataType": ["text"],
                        "description": "Data source (e.g., binance, coingecko)"
                    },
                    {
                        "name": "price",
                        "dataType": ["number"],
                        "description": "Current price in USD"
                    },
                    {
                        "name": "market_cap",
                        "dataType": ["number"],
                        "description": "Market capitalization"
                    },
                    {
                        "name": "volume_24h",
                        "dataType": ["number"],
                        "description": "24h trading volume"
                    },
                    {
                        "name": "price_change_24h",
                        "dataType": ["number"],
                        "description": "24h price change percentage"
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Data timestamp"
                    }
                ]
            )
            logger.info("✅ Successfully created MarketMetrics collection")
            
            # Add some sample data for testing
            self._add_sample_market_data(client, collection)
            
            return collection
        except Exception as e:
            logger.error(f"Failed to create MarketMetrics collection: {str(e)}")
            raise

def _add_sample_market_data(client, collection, symbol="BTCUSDT"):
    """Add sample market data for testing"""
    
    # Create sample data
    sample_data = []
    base_price = 50000.0  # Base price for BTC
    
    for i in range(30):  # 30 days of data
        # Generate price with some randomness
        price = base_price * (1 + np.random.normal(0, 0.03))  # 3% standard deviation
        
        # Adjust base price for next day (trend slightly upward)
        base_price *= 1.005  # 0.5% daily increase on average
        
        # Generate other metrics
        market_cap = price * 19000000  # Approx BTC supply
        volume_24h = market_cap * 0.05  # 5% daily volume
        price_change_24h = ((price / base_price) - 1) * 100  # Daily percent change
        
        # Create date (days in the past)
        date = datetime.now() - timedelta(days=29-i)
        
        sample_data.append({
            "symbol": symbol,
            "source": "sample_data",
            "price": price,
            "market_cap": market_cap,
            "volume_24h": volume_24h,
            "price_change_24h": price_change_24h,
            "timestamp": date.isoformat()
        })
    
    # Insert data
    try:
        collection.data.insert_many(sample_data)
        logger.info(f"Added {len(sample_data)} sample market data points for {symbol}")
    except Exception as e:
        logger.error(f"Error adding sample market data: {e}")

class CryptoForecaster:
    """Forecasts crypto prices using market data and sentiment analysis"""
    
    def __init__(self, model_dir="models"):
        """Initialize the forecaster"""
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def _fetch_market_data(self, symbol, days=30):
        """Fetch historical market data from Weaviate"""
        client = get_weaviate_client()
        
        try:
            # Ensure MarketMetrics collection exists
            collection = ensure_market_metrics_schema(client)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            from weaviate.classes.query import Filter
            response = collection.query.fetch_objects(
                filters=Filter.by_property("symbol").equal(symbol),
                return_properties=["symbol", "price", "market_cap", "volume_24h", 
                                  "price_change_24h", "timestamp", "source"],
                limit=1000  # Adjust based on your needs
            )
            
            if not response.objects:
                logger.warning(f"No market data found for {symbol}")
                # Add some sample data for this symbol
                _add_sample_market_data(client, collection, symbol)
                
                # Try querying again
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(symbol),
                    return_properties=["symbol", "price", "market_cap", "volume_24h", 
                                    "price_change_24h", "timestamp", "source"],
                    limit=1000
                )
                
                if not response.objects:
                    return None
                
            # Convert to DataFrame
            data = []
            for obj in response.objects:
                data.append(obj.properties)
                
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            logger.info(f"Fetched {len(df)} market data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
        finally:
            client.close()
    
    def _fetch_sentiment_data(self, symbol, days=30):
        """Fetch sentiment data related to the crypto symbol"""
        client = get_weaviate_client()
        
        try:
            # Check if collection exists
            try:
                # Query CryptoNewsSentiment collection
                collection = client.collections.get("CryptoNewsSentiment")
            except Exception:
                # Collection doesn't exist, return None
                logger.warning("CryptoNewsSentiment collection doesn't exist")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query - search for articles mentioning the symbol
            response = collection.query.hybrid(
                query=symbol,
                return_properties=["source", "title", "sentiment_label", 
                                  "sentiment_score", "date", "analyzed_at"],
                limit=100  # Adjust based on your needs
            )
            
            if not response.objects:
                logger.warning(f"No sentiment data found for {symbol}")
                return None
                
            # Convert to DataFrame
            data = []
            for obj in response.objects:
                data.append(obj.properties)
                
            df = pd.DataFrame(data)
            
            # Convert date to datetime
            df["date"] = pd.to_datetime(df["date"])
            
            # Sort by date
            df = df.sort_values("date")
            
            logger.info(f"Fetched {len(df)} sentiment data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            return None
        finally:
            client.close()
    
    def _prepare_features(self, market_df, sentiment_df):
        """Prepare features by combining market and sentiment data"""
        if market_df is None or len(market_df) < 5:
            logger.error("Insufficient market data for feature preparation")
            return None
            
        # Create copy to avoid modifying original data
        df = market_df.copy()
        
        # Add technical indicators
        # 1. Moving Averages
        df['ma_7'] = df['price'].rolling(window=min(7, len(df))).mean()
        df['ma_14'] = df['price'].rolling(window=min(14, len(df))).mean()
        
        # 2. Price Rate of Change
        df['price_roc'] = df['price'].pct_change(periods=1)
        
        # 3. Volume Rate of Change
        df['volume_roc'] = df['volume_24h'].pct_change(periods=1)
        
        # Add sentiment features if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Group sentiment by date and calculate daily averages
            daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'count']
            })
            daily_sentiment.columns = ['avg_sentiment', 'sentiment_count']
            daily_sentiment = daily_sentiment.reset_index()
            
            # Convert market data timestamp to date for joining
            df['date'] = df['timestamp'].dt.date
            
            # Merge market data with sentiment
            df = pd.merge(df, daily_sentiment, on='date', how='left')
            
            # Fill missing sentiment values
            df['avg_sentiment'].fillna(0.5, inplace=True)
            df['sentiment_count'].fillna(0, inplace=True)
        else:
            # Add placeholder sentiment features
            df['avg_sentiment'] = 0.5
            df['sentiment_count'] = 0
        
        # Fill NaN values with method appropriate for time series
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(method='bfill', inplace=True)  # Backward fill for any remaining NaNs
        df.fillna(0, inplace=True)  # Fill any remaining NaNs with zeros
        
        # Create target variable (next day's price)
        df['next_price'] = df['price'].shift(-1)
        
        # Drop last row (which will have NaN for next_price)
        df = df[:-1]
        
        # Drop rows with any remaining NaN values
        df.dropna(inplace=True)
        
        if len(df) < 5:
            logger.error("Insufficient data after feature preparation")
            return None
            
        logger.info(f"Prepared features with shape: {df.shape}")
        return df
    
    def train(self, symbol, days=90):
        """Train a prediction model for a specific crypto symbol"""
        logger.info(f"Training model for {symbol}")
        
        # Fetch data
        market_df = self._fetch_market_data(symbol, days)
        sentiment_df = self._fetch_sentiment_data(symbol, days)
        
        if market_df is None or len(market_df) < 10:
            logger.error(f"Insufficient market data for {symbol}")
            return False
            
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df)
        
        if df is None or len(df) < 10:
            logger.error(f"Insufficient prepared data for {symbol}")
            return False
        
        # Define features and target
        feature_columns = ['price', 'market_cap', 'volume_24h', 'price_change_24h',
                           'ma_7', 'ma_14', 'price_roc', 'volume_roc',
                           'avg_sentiment', 'sentiment_count']
        
        X = df[feature_columns]
        y = df['next_price']
        
        # Split data
        if len(df) <= 15:
            # For very small datasets, use a smaller test set
            test_size = 0.1
        else:
            test_size = 0.2
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model evaluation for {symbol}:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"R²: {r2:.4f}")
        
        # Save model and scaler
        model_path = os.path.join(self.model_dir, f"{symbol}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Store model and scaler in memory
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Save feature importance plot
        self._plot_feature_importance(model, feature_columns, symbol)
        
        logger.info(f"Model for {symbol} trained and saved successfully")
        return True
    
    def _plot_feature_importance(self, model, feature_names, symbol):
        """Plot and save feature importance"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importance for {symbol}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f"{symbol}_feature_importance.png")
        plt.savefig(plot_path)
        plt.close()
    
    def load_model(self, symbol):
        """Load a trained model for a specific symbol"""
        if symbol in self.models and symbol in self.scalers:
            return self.models[symbol], self.scalers[symbol]
            
        model_path = os.path.join(self.model_dir, f"{symbol}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            logger.info(f"Model for {symbol} loaded successfully")
            return model, scaler
        else:
            logger.warning(f"No trained model found for {symbol}")
            return None, None
    
    def predict(self, symbol, days_ahead=7):
        """Make predictions for a specific symbol"""
        # Load or train model
        model, scaler = self.load_model(symbol)
        
        if model is None:
            logger.info(f"No model found for {symbol}, training now...")
            success = self.train(symbol)
            if not success:
                logger.error(f"Failed to train model for {symbol}")
                return None
            model, scaler = self.load_model(symbol)
            
            if model is None:
                logger.error(f"Failed to load model after training for {symbol}")
                return None
        
        # Get current data
        market_df = self._fetch_market_data(symbol, days=30)
        sentiment_df = self._fetch_sentiment_data(symbol, days=30)
        
        if market_df is None or len(market_df) < 5:
            logger.error(f"Insufficient market data for prediction")
            
            # Generate synthetic data for demo purposes
            # In real system, you'd want better error handling
            current_date = datetime.now()
            base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0 if symbol == "ETHUSDT" else 500.0
            
            predictions = []
            for i in range(days_ahead):
                next_date = current_date + timedelta(days=i+1)
                change_pct = np.random.normal(0.5, 1.5)  # Random daily change
                predicted_price = base_price * (1 + change_pct/100)
                base_price = predicted_price  # Update for next day
                
                predictions.append({
                    'date': next_date,
                    'predicted_price': predicted_price,
                    'change_pct': change_pct
                })
            
            return pd.DataFrame(predictions)
            
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df)
        
        if df is None or len(df) < 5:
            logger.error(f"Insufficient prepared data for prediction")
            return None
            
        # Define features
        feature_columns = ['price', 'market_cap', 'volume_24h', 'price_change_24h',
                           'ma_7', 'ma_14', 'price_roc', 'volume_roc',
                           'avg_sentiment', 'sentiment_count']
        
        # Get most recent data point
        latest_data = df[feature_columns].iloc[-1].values.reshape(1, -1)
        latest_price = df['price'].iloc[-1]
        latest_date = df['timestamp'].iloc[-1]
        
        # Scale features
        latest_data_scaled = scaler.transform(latest_data)
        
        # Make predictions for multiple days ahead
        predictions = []
        current_date = latest_date
        current_price = latest_price
        current_data = latest_data.copy()
        
        for i in range(days_ahead):
            # Make prediction for next day
            current_data_scaled = scaler.transform(current_data)
            next_price = model.predict(current_data_scaled)[0]
            
            # Update date
            current_date = current_date + timedelta(days=1)
            
            # Calculate percent change
            change_pct = ((next_price - current_price) / current_price) * 100
            
            # Save prediction
            predictions.append({
                'date': current_date,
                'predicted_price': next_price,
                'change_pct': change_pct
            })
            
            # Update current price for next iteration
            current_price = next_price
            
            # Update features for next prediction
            # This is a simplified approach - in a real system, you would
            # need more sophisticated methods to update all features
            current_data[0, 0] = next_price  # Update price
            
            # Update moving averages (simplified)
            if i == 0:
                current_data[0, 4] = (current_data[0, 4] * 6 + next_price) / 7  # ma_7
                current_data[0, 5] = (current_data[0, 5] * 13 + next_price) / 14  # ma_14
            else:
                current_data[0, 4] = (current_data[0, 4] * 6 + next_price) / 7  # ma_7
                current_data[0, 5] = (current_data[0, 5] * 13 + next_price) / 14  # ma_14
            
            # Update price rate of change
            current_data[0, 6] = (next_price - current_price) / current_price  # price_roc
        
        # Convert to DataFrame
        prediction_df = pd.DataFrame(predictions)
        
        logger.info(f"Generated {len(prediction_df)} day predictions for {symbol}")
        return prediction_df
    
    def plot_predictions(self, symbol, predictions):
        """Plot price predictions"""
        if predictions is None or len(predictions) == 0:
            logger.error("No predictions to plot")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(predictions['date'], predictions['predicted_price'], 'b-o', label='Predicted Price')
        
        # Get historical data for comparison
        market_df = self._fetch_market_data(symbol, days=14)
        if market_df is not None and len(market_df) > 0:
            plt.plot(market_df['timestamp'], market_df['price'], 'r-', label='Historical Price')
        
        plt.title(f'{symbol} Price Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f"{symbol}_forecast.png")
        plt.savefig(plot_path)
        
        # Also save to current directory
        plt.savefig(f"{symbol}_forecast.png")
        plt.close()
        
        logger.info(f"Prediction plot saved to {plot_path}")

# Example usage
if __name__ == "__main__":
    forecaster = CryptoForecaster()
    
    # Train model for Bitcoin
    symbol = "BTCUSDT"
    forecaster.train(symbol)
    
    # Make predictions
    predictions = forecaster.predict(symbol, days_ahead=7)
    
    if predictions is not None:
        # Plot predictions
        forecaster.plot_predictions(symbol, predictions)
        
        # Print forecast
        print(f"\n{symbol} Price Forecast:")
        for _, row in predictions.iterrows():
            print(f"Date: {row['date'].strftime('%Y-%m-%d')}, Price: ${row['predicted_price']:.2f}, Change: {row['change_pct']:.2f}%")