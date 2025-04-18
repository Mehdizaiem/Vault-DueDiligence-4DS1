
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from typing import Dict, List, Optional, Union, Any

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from Sample_Data.vector_store.weaviate_client import get_weaviate_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_market_metrics_schema(client):
    try:
        return client.collections.get("MarketMetrics")
    except Exception:
        logger.info("Creating MarketMetrics collection")
        from Sample_Data.vector_store.market_sentiment_schema import create_market_metrics_schema
        return create_market_metrics_schema(client)


class CryptoForecaster:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        os.makedirs(model_dir, exist_ok=True)
        self.model_params = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

    def _fetch_market_data(self, symbol, days=30):
        client = get_weaviate_client()
        try:
            collection = ensure_market_metrics_schema(client)
            start_date_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            from weaviate.classes.query import Filter

            response = collection.query.fetch_objects(
                filters=Filter.by_property("symbol").equal(symbol) &
                        Filter.by_property("timestamp").greater_than(start_date_str),
                return_properties=["symbol", "price", "market_cap", "volume_24h",
                                   "price_change_24h", "timestamp", "source"],
                limit=1000
            )

            if not response.objects:
                logger.warning(f"No market data found for {symbol}")
                return None

            data = [obj.properties for obj in response.objects]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset="timestamp")
            logger.info(f"Fetched {len(df)} market data points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
        finally:
            client.close()

    def _fetch_sentiment_data(self, symbol, days=30):
        client = get_weaviate_client()
        try:
            collection = client.collections.get("CryptoNewsSentiment")
            from weaviate.classes.query import Filter
            symbol_clean = symbol.replace("USDT", "").lower()
            response = collection.query.fetch_objects(
                filters=Filter.by_property("content").contains_any([symbol_clean]),
                return_properties=["title", "sentiment_score", "date"],
                limit=100
            )
            data = [obj.properties for obj in response.objects]
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            logger.info(f"Fetched {len(df)} sentiment points for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            return None
        finally:
            client.close()

    def _prepare_features(self, market_df, sentiment_df):
        if market_df is None or len(market_df) < 5:
            logger.error("Insufficient market data")
            return None

        df = market_df.copy()

        # Adapt window sizes
        w7 = min(7, len(df))
        w14 = min(14, len(df))

        df["ma_7"] = df["price"].rolling(w7).mean()
        df["ma_14"] = df["price"].rolling(w14).mean()
        df["roc"] = df["price"].pct_change()
        df["vol_roc"] = df["volume_24h"].pct_change()

        df = df.ffill().bfill()

        if sentiment_df is not None and not sentiment_df.empty:
            sentiment_df["date"] = sentiment_df["date"].dt.date
            df["date"] = df["timestamp"].dt.date
            sentiment_daily = sentiment_df.groupby("date").agg({"sentiment_score": "mean"}).reset_index()
            df = df.merge(sentiment_daily, how="left", on="date")
            df["sentiment_score"] = df["sentiment_score"].fillna(0.5)
        else:
            df["sentiment_score"] = 0.5

        df["next_price"] = df["price"].shift(-1)
        df.dropna(inplace=True)

        if df.empty or len(df) < 5:
            logger.error("Not enough rows after feature engineering for prediction.")
            return None

        return df


    def train(self, symbol, days=90):
        market_df = self._fetch_market_data(symbol, days)
        sentiment_df = self._fetch_sentiment_data(symbol, days)
        df = self._prepare_features(market_df, sentiment_df)
        if df is None or len(df) < 10:
            logger.error("Training aborted: Not enough data")
            return False

        features = ["price", "market_cap", "volume_24h", "price_change_24h",
                    "ma_7", "ma_14", "roc", "vol_roc", "sentiment_score"]
        X = df[features]
        y = df["next_price"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)

        try:
            model = GridSearchCV(RandomForestRegressor(random_state=42), self.model_params,
                                 cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
            model.fit(X_scaled, y)
            model = model.best_estimator_
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)

        model_path = os.path.join(self.model_dir, f"{symbol}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        self.models[symbol] = model
        self.scalers[symbol] = scaler

        logger.info(f"Model trained and saved for {symbol}")
        return True

    def predict(self, symbol, days_ahead=7):
        model_path = os.path.join(self.model_dir, f"{symbol}_model.joblib")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.joblib")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.info("Model not found, training...")
            self.train(symbol)

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        market_df = self._fetch_market_data(symbol, days=30)
        sentiment_df = self._fetch_sentiment_data(symbol, days=30)
        df = self._prepare_features(market_df, sentiment_df)

        if df is None or df.empty:
            logger.error("No data to predict")
            return None

        last_row = df.iloc[-1:]
        features = ["price", "market_cap", "volume_24h", "price_change_24h",
                    "ma_7", "ma_14", "roc", "vol_roc", "sentiment_score"]
        base_price = last_row["price"].values[0]
        X_scaled = scaler.transform(last_row[features])

        future = []
        for i in range(days_ahead):
            pred = model.predict(X_scaled)[0]
            future.append({
                "date": datetime.now() + timedelta(days=i+1),
                "predicted_price": pred
            })
            X_scaled[0][0] = pred  # Update price

        return pd.DataFrame(future)

    def plot_predictions(self, symbol, predictions):
        if predictions is None or predictions.empty:
            logger.warning("No predictions to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(predictions["date"], predictions["predicted_price"], marker='o', label="Forecast")
        plt.title(f"{symbol} Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{symbol}_forecast.png")
        plt.close()
        logger.info(f"Plot saved: {symbol}_forecast.png")


if __name__ == "__main__":
    forecaster = CryptoForecaster()
    symbol = "BTCUSDT"
    forecaster.train(symbol)
    pred = forecaster.predict(symbol, days_ahead=7)
    if pred is not None:
        print(pred)
        forecaster.plot_predictions(symbol, pred)
