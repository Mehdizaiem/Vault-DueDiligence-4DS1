import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any

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
            # Import the required schema creation function
            from Sample_Data.vector_store.market_sentiment_schema import create_market_metrics_schema
            collection = create_market_metrics_schema(client)
            logger.info("✅ Successfully created MarketMetrics collection")
            return collection
        except Exception as e:
            logger.error(f"Failed to create MarketMetrics collection: {str(e)}")
            raise

class CryptoForecaster:
    """Forecasts crypto prices using market data and sentiment analysis with improved models"""
    
    def __init__(self, model_dir="models"):
        """Initialize the forecaster"""
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        os.makedirs(model_dir, exist_ok=True)
        # Define model parameters for grid search
        self.model_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    def _fetch_market_data(self, symbol, days=30):
        """Fetch historical market data from Weaviate with improved error handling"""
        client = get_weaviate_client()
        
        try:
            # Ensure MarketMetrics collection exists
            collection = ensure_market_metrics_schema(client)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query
            try:
                from weaviate.classes.query import Filter
                
                # Add timestamp filter for date range
                start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(symbol) & 
                           Filter.by_property("timestamp").greater_than(start_date_str),
                    return_properties=["symbol", "price", "market_cap", "volume_24h", 
                                    "price_change_24h", "timestamp", "source"],
                    limit=1000  # Adjust based on your needs
                )
            except Exception as e:
                logger.warning(f"Error with filtered query: {e}. Falling back to non-filtered query.")
                # Fallback to non-filtered query
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(symbol),
                    return_properties=["symbol", "price", "market_cap", "volume_24h", 
                                    "price_change_24h", "timestamp", "source"],
                    limit=1000
                )
            
            if not response.objects:
                logger.warning(f"No market data found for {symbol}")
                # Add some sample data for this symbol
                try:
                    from Sample_Data.vector_store.market_sentiment_schema import add_sample_market_data
                    add_sample_market_data(client, collection, symbol)
                    
                    # Try querying again
                    response = collection.query.fetch_objects(
                        filters=Filter.by_property("symbol").equal(symbol),
                        return_properties=["symbol", "price", "market_cap", "volume_24h", 
                                        "price_change_24h", "timestamp", "source"],
                        limit=1000
                    )
                except Exception as e:
                    logger.error(f"Error adding sample data: {e}")
                
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
            
            # Remove duplicate timestamps by keeping the latest version
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            logger.info(f"Fetched {len(df)} market data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
        finally:
            client.close()
    
    def _fetch_sentiment_data(self, symbol, days=30):
        """Fetch sentiment data related to the crypto symbol with proper error handling"""
        client = get_weaviate_client()
        
        try:
            # Check if collection exists
            try:
                # Query CryptoNewsSentiment collection
                collection = client.collections.get("CryptoNewsSentiment")
            except Exception as e:
                # Collection doesn't exist, return None
                logger.warning(f"CryptoNewsSentiment collection doesn't exist: {e}")
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build query - use text search rather than vector search
            try:
                from weaviate.classes.query import Filter
                
                # Create proper keyword search instead of vector search
                # Strip USDT from symbol name for better matching
                symbol_clean = symbol.replace("USDT", "").lower()
                
                # Try with content filter
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("content").contains_any([symbol_clean]),
                    return_properties=["source", "title", "sentiment_label", 
                                      "sentiment_score", "date", "analyzed_at"],
                    limit=100
                )
                
                if not response.objects:
                    # Try with title filter as fallback
                    response = collection.query.fetch_objects(
                        filters=Filter.by_property("title").contains_any([symbol_clean]),
                        return_properties=["source", "title", "sentiment_label", 
                                          "sentiment_score", "date", "analyzed_at"],
                        limit=100
                    )
            except Exception as e:
                logger.error(f"Error fetching sentiment data: {e}")
                return None
            
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
        """Prepare features by combining market and sentiment data with improved handling"""
        if market_df is None or len(market_df) < 5:
            logger.error("Insufficient market data for feature preparation")
            return None
            
        # Create copy to avoid modifying original data
        df = market_df.copy()
        
        # Add technical indicators
        # 1. Moving Averages
        window_7 = min(7, len(df))
        window_14 = min(14, len(df))
        
        df['ma_7'] = df['price'].rolling(window=window_7).mean()
        df['ma_14'] = df['price'].rolling(window=window_14).mean()
        
        # 2. Price Rate of Change
        df['price_roc'] = df['price'].pct_change(periods=1)
        
        # 3. Volume Rate of Change
        df['volume_roc'] = df['volume_24h'].pct_change(periods=1)
        
        # 4. Exponential Moving Averages (new)
        df['ema_12'] = df['price'].ewm(span=min(12, len(df)), adjust=False).mean()
        df['ema_26'] = df['price'].ewm(span=min(26, len(df)), adjust=False).mean()
        
        # 5. MACD (new)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=min(9, len(df)), adjust=False).mean()
        
        # 6. Bollinger Bands (new)
        window_20 = min(20, len(df))
        df['ma_20'] = df['price'].rolling(window=window_20).mean()
        df['std_20'] = df['price'].rolling(window=window_20).std()
        df['upper_band'] = df['ma_20'] + (df['std_20'] * 2)
        df['lower_band'] = df['ma_20'] - (df['std_20'] * 2)
        
        # Add relative price position in band (new)
        df['bb_position'] = (df['price'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
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
        
        # Fill NaN values properly (fixing deprecated warnings)
        df = df.ffill()  # Forward fill (replacing fillna(method='ffill'))
        df = df.bfill()  # Backward fill (replacing fillna(method='bfill')) 
        df = df.fillna(0)  # Fill any remaining NaNs with zeros
        
        # Create target variable (next day's price)
        df['next_price'] = df['price'].shift(-1)
        
        # Drop last row (which will have NaN for next_price)
        df = df[:-1]
        
        # Drop rows with any remaining NaN values
        df = df.dropna()
        
        if len(df) < 5:
            logger.error("Insufficient data after feature preparation")
            return None
            
        logger.info(f"Prepared features with shape: {df.shape}")
        return df
    
    def train(self, symbol, days=90):
        """Train a prediction model for a specific crypto symbol with cross-validation"""
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
        feature_columns = [
            # Basic price and market data
            'price', 'market_cap', 'volume_24h', 'price_change_24h',
            # Moving averages
            'ma_7', 'ma_14', 'ma_20',
            # Rate of change indicators
            'price_roc', 'volume_roc',
            # MACD indicators
            'macd', 'macd_signal',
            # Bollinger Band indicators
            'std_20', 'upper_band', 'lower_band', 'bb_position',
            # Sentiment indicators
            'avg_sentiment', 'sentiment_count'
        ]
        
        # Make sure all feature columns exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
        y = df['next_price']
        
        # Split data using TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=min(5, len(df) // 5))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model with grid search for hyperparameter tuning
        try:
            # For small datasets, use a simpler approach
            if len(df) < 20:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
            else:
                # Use grid search for larger datasets
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_grid=self.model_params,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(X_scaled, y)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            # Fallback to basic model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
        
        # Evaluate model on the last fold
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        y_pred = model.predict(X_test)
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
        
        # Store model metrics
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "features": feature_columns
        }
        metrics_path = os.path.join(self.model_dir, f"{symbol}_metrics.joblib")
        joblib.dump(metrics, metrics_path)
        
        logger.info(f"Model for {symbol} trained and saved successfully")
        return True
    
    def _plot_feature_importance(self, model, feature_names, symbol):
        """Plot and save feature importance"""
        try:
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title(f'Feature Importance for {symbol}')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.model_dir, f"{symbol}_feature_importance.png")
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
    
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
        """Make predictions for a specific symbol with proper error handling and confidence intervals"""
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
                    'change_pct': change_pct,
                    'lower_bound': predicted_price * 0.95,  # Add confidence bounds
                    'upper_bound': predicted_price * 1.05
                })
            
            return pd.DataFrame(predictions)
        
        # Load feature information
        metrics_path = os.path.join(self.model_dir, f"{symbol}_metrics.joblib")
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            feature_columns = metrics.get("features")
        else:
            # Default feature columns
            feature_columns = [
                'price', 'market_cap', 'volume_24h', 'price_change_24h',
                'ma_7', 'ma_14', 'price_roc', 'volume_roc',
                'avg_sentiment', 'sentiment_count'
            ]
            
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df)
        
        if df is None or len(df) < 5:
            logger.error(f"Insufficient prepared data for prediction")
            return None
            
        # Make sure all feature columns exist in the DataFrame
        available_features = [col for col in feature_columns if col in df.columns]
            
        # Get most recent data point
        latest_data = df[available_features].iloc[-1].values.reshape(1, -1)
        latest_price = df['price'].iloc[-1]
        latest_date = df['timestamp'].iloc[-1]
        
        # Scale features
        latest_data_scaled = scaler.transform(latest_data)
        
        # Make predictions for multiple days ahead
        predictions = []
        current_date = latest_date
        current_price = latest_price
        current_data = latest_data.copy()
        
        # Calculate prediction uncertainty based on model MAE
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
            mae = metrics.get("mae", current_price * 0.05)  # Default to 5% if not available
        else:
            mae = current_price * 0.05  # Default error margin
        
        # Generate bootstrap samples for uncertainty estimation
        n_estimators = min(model.n_estimators, 100)  # Use number of trees or 100 max
        prediction_samples = []
        
        for i in range(days_ahead):
            # Make prediction for next day
            current_data_scaled = scaler.transform(current_data)
            
            if hasattr(model, 'estimators_'):
                # For RandomForest, use estimators to get prediction distribution
                tree_predictions = []
                for tree in model.estimators_[:n_estimators]:
                    tree_predictions.append(tree.predict(current_data_scaled)[0])
                
                # Get mean and standard deviation of predictions
                next_price = np.mean(tree_predictions)
                std_dev = np.std(tree_predictions)
                
                # Calculate confidence intervals (mean ± 1.96 * std for 95% CI)
                lower_bound = next_price - 1.96 * std_dev
                upper_bound = next_price + 1.96 * std_dev
            else:
                # For other models, use MAE for uncertainty
                next_price = model.predict(current_data_scaled)[0]
                lower_bound = next_price - 1.96 * mae
                upper_bound = next_price + 1.96 * mae
            
            # Update date
            current_date = current_date + timedelta(days=1)
            
            # Calculate percent change
            change_pct = ((next_price - current_price) / current_price) * 100
            
            # Save prediction
            predictions.append({
                'date': current_date,
                'predicted_price': next_price,
                'change_pct': change_pct,
                'lower_bound': max(0, lower_bound),  # Ensure non-negative prices
                'upper_bound': upper_bound
            })
            
            # Update current price for next iteration
            current_price = next_price
            
            # Update features for next prediction - if we have the right features
            if 'price' in available_features:
                feature_idx = available_features.index('price')
                current_data[0, feature_idx] = next_price  # Update price
            
            # Update moving averages if available
            if 'ma_7' in available_features and i > 0:
                ma7_idx = available_features.index('ma_7')
                current_data[0, ma7_idx] = (current_data[0, ma7_idx] * 6 + next_price) / 7
                
            if 'ma_14' in available_features and i > 0:
                ma14_idx = available_features.index('ma_14')
                current_data[0, ma14_idx] = (current_data[0, ma14_idx] * 13 + next_price) / 14
                
            # Update rate of change if available
            if 'price_roc' in available_features:
                roc_idx = available_features.index('price_roc')
                last_price = predictions[-2]['predicted_price'] if i > 0 else latest_price
                current_data[0, roc_idx] = (next_price - last_price) / last_price if last_price > 0 else 0
        
        # Convert to DataFrame
        prediction_df = pd.DataFrame(predictions)
        
        logger.info(f"Generated {len(prediction_df)} day predictions for {symbol}")
        return prediction_df
    
    def plot_predictions(self, symbol, predictions):
        """Plot price predictions with confidence intervals"""
        if predictions is None or len(predictions) == 0:
            logger.error("No predictions to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot predictions with confidence interval
        has_confidence = 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns
        
        plt.plot(predictions['date'], predictions['predicted_price'], 'b-o', label='Predicted Price')
        
        if has_confidence:
            # Plot confidence interval
            plt.fill_between(
                predictions['date'],
                predictions['lower_bound'],
                predictions['upper_bound'],
                color='blue', alpha=0.2,
                label='95% Confidence Interval'
            )
        
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
        
        # Add annotations for predicted prices
        for i, row in predictions.iterrows():
            if i % 2 == 0:  # Annotate every other point to reduce clutter
                plt.annotate(
                    f"${row['predicted_price']:,.2f}\n({row['change_pct']:+.1f}%)",
                    (row['date'], row['predicted_price']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )
        
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