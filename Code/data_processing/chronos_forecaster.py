import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import os
import sys
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Try to import statsmodels and prophet, but provide graceful fallback
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChronosForecaster:
    """
    Advanced time series forecasting for cryptocurrency data using multiple models
    with confidence intervals, anomaly detection, and sentiment integration.
    """
    
    def __init__(self, model_dir="models/chronos"):
        """
        Initialize the ChronosForecaster.
        
        Args:
            model_dir: Directory for storing trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        
        logger.info(f"ChronosForecaster initialized with model_dir: {model_dir}")
        logger.info(f"SARIMAX available: {STATSMODELS_AVAILABLE}")
        logger.info(f"Prophet available: {PROPHET_AVAILABLE}")
    
    def _prepare_features(self, market_df, sentiment_df=None, onchain_df=None):
        """
        Prepare features by combining market, sentiment, and on-chain data.
        
        Args:
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            
        Returns:
            DataFrame with prepared features
        """
        if market_df is None or len(market_df) < 5:
            logger.error("Insufficient market data for feature preparation")
            return None
            
        # Create copy to avoid modifying original data
        df = market_df.copy()
        
        # Ensure timestamp is the index and is datetime
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('timestamp')
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Make sure price column exists (use close if no price)
        if 'price' not in df.columns and 'close' in df.columns:
            df['price'] = df['close']
        elif 'price' not in df.columns:
            logger.error("No price or close column found in market data")
            return None
        
        # Calculate price-based features
        # 1. Moving Averages (different window sizes)
        for window in [7, 14, 30]:
            window_size = min(window, len(df) - 1)  # Ensure window size <= data length
            if window_size > 1:
                df[f'ma_{window}'] = df['price'].rolling(window=window_size).mean()
        
        # 2. Exponential Moving Averages
        for span in [12, 26]:
            span_size = min(span, len(df) - 1)
            if span_size > 1:
                df[f'ema_{span}'] = df['price'].ewm(span=span_size, adjust=False).mean()
        
        # 3. MACD
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=min(9, len(df) - 1), adjust=False).mean()
        
        # 4. Bollinger Bands
        window_20 = min(20, len(df) - 1)
        if window_20 > 1:
            df['ma_20'] = df['price'].rolling(window=window_20).mean()
            df['std_20'] = df['price'].rolling(window=window_20).std()
            df['upper_band'] = df['ma_20'] + (df['std_20'] * 2)
            df['lower_band'] = df['ma_20'] - (df['std_20'] * 2)
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['ma_20']
            df['bb_position'] = (df['price'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        # 5. Rate of Change and Momentum
        df['price_roc_1'] = df['price'].pct_change(periods=1)
        if len(df) > 5:
            df['price_roc_5'] = df['price'].pct_change(periods=5)
        
        if 'volume' in df.columns or 'volume_24h' in df.columns:
            volume_col = 'volume' if 'volume' in df.columns else 'volume_24h'
            df['volume_roc_1'] = df[volume_col].pct_change(periods=1)
            df['price_volume_ratio'] = df['price'] / df[volume_col].replace(0, np.nan)
        
        # 6. RSI (Relative Strength Index)
        delta = df['price'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=min(14, len(df) - 1)).mean()
        avg_loss = loss.rolling(window=min(14, len(df) - 1)).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Add time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        
        # Add sentiment features if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Ensure sentiment_df index is datetime
            if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                if 'date' in sentiment_df.columns:
                    sentiment_df = sentiment_df.set_index('date')
                else:
                    sentiment_df.index = pd.to_datetime(sentiment_df.index)
            
            # Resample sentiment to match market data frequency
            daily_sentiment = sentiment_df.resample('D').mean()
            
            # Merge with market data
            df = df.join(daily_sentiment[['sentiment_score']], how='left')
            
            # Fill missing sentiment values with neutral (0.5)
            df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
            
            # Add sentiment momentum
            df['sentiment_momentum'] = df['sentiment_score'].diff()
        else:
            # Add placeholder sentiment
            df['sentiment_score'] = 0.5
            df['sentiment_momentum'] = 0
        
        # Add on-chain features if available
        if onchain_df is not None and not onchain_df.empty:
            # Similar processing as sentiment data
            # Merge relevant on-chain metrics
            pass
        
        # Handle missing values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Create target variable (next day's price)
        df['target_next_day'] = df['price'].shift(-1)
        
        # For multi-day forecasts, create additional targets
        for days in [3, 7, 14, 30]:
            if len(df) > days:
                df[f'target_{days}d'] = df['price'].shift(-days)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target_next_day'])
        
        return df
    
    def _detect_anomalies(self, df, symbol, retrain=False):
        """
        Detect anomalies in time series data using Isolation Forest.
        
        Args:
            df: DataFrame with time series data
            symbol: Trading symbol
            retrain: Whether to retrain the anomaly detector
            
        Returns:
            DataFrame with anomaly scores
        """
        if df is None or len(df) < 10:
            logger.error("Insufficient data for anomaly detection")
            return df
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Features for anomaly detection
        anomaly_features = ['price_roc_1', 'volume_roc_1', 'rsi_14', 'bb_position']
        available_features = [f for f in anomaly_features if f in df.columns]
        
        if len(available_features) < 2:
            logger.warning("Insufficient features for anomaly detection")
            result_df['anomaly_score'] = 0
            result_df['is_anomaly'] = False
            return result_df
        
        # Prepare data for anomaly detection
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get or train anomaly detector
        if symbol not in self.anomaly_detectors or retrain:
            logger.info(f"Training anomaly detector for {symbol}")
            
            # Initialize and train Isolation Forest
            isolation_forest = IsolationForest(
                n_estimators=100, 
                contamination=0.05,  # Assume ~5% of data points are anomalies
                random_state=42
            )
            
            isolation_forest.fit(X)
            self.anomaly_detectors[symbol] = isolation_forest
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{symbol}_anomaly_detector.joblib")
            joblib.dump(isolation_forest, model_path)
        else:
            isolation_forest = self.anomaly_detectors[symbol]
        
        # Generate anomaly scores (-1 for anomalies, 1 for normal)
        # Convert to 0-1 scale where higher values indicate stronger anomalies
        raw_scores = isolation_forest.decision_function(X)
        normalized_scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
        
        # Add scores to dataframe
        result_df['anomaly_score'] = normalized_scores
        result_df['is_anomaly'] = isolation_forest.predict(X) == -1
        
        # Add contextual information about anomalies
        result_df['anomaly_type'] = None
        
        # Investigate price anomalies
        price_anomalies = result_df[result_df['is_anomaly'] & (result_df['price_roc_1'].abs() > 0.03)]
        result_df.loc[price_anomalies.index, 'anomaly_type'] = 'price_movement'
        
        # Investigate volume anomalies
        if 'volume_roc_1' in result_df.columns:
            volume_anomalies = result_df[result_df['is_anomaly'] & (result_df['volume_roc_1'].abs() > 0.5)]
            result_df.loc[volume_anomalies.index, 'anomaly_type'] = 'volume_spike'
        
        # Count anomalies
        anomaly_count = result_df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies in {len(result_df)} data points for {symbol}")
        
        return result_df
    
    def train_short_term_model(self, symbol, market_df, sentiment_df=None, onchain_df=None):
        """
        Train a short-term (24h) forecasting model using Random Forest and ARIMA.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            
        Returns:
            Dict with trained models and metadata
        """
        logger.info(f"Training short-term model for {symbol}")
        
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df, onchain_df)
        if df is None or len(df) < 20:
            logger.error(f"Insufficient data for training short-term model for {symbol}")
            return None
        
        # Detect anomalies
        df = self._detect_anomalies(df, symbol, retrain=True)
        
        # Define features and target
        target = 'target_next_day'
        
        # Select features for prediction
        price_features = ['price', 'ma_7', 'ma_14', 'ema_12', 'ema_26', 'macd', 'rsi_14', 'bb_position']
        pattern_features = ['price_roc_1', 'volume_roc_1', 'anomaly_score']
        sentiment_features = ['sentiment_score', 'sentiment_momentum']
        time_features = ['day_of_week', 'month', 'is_weekend']
        
        # Use features that are available in the dataframe
        all_features = price_features + pattern_features + sentiment_features + time_features
        available_features = [f for f in all_features if f in df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Very few features available for {symbol}: {available_features}")
        
        X = df[available_features].values
        y = df[target].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_scaled, y)
        
        # Train ARIMA model for time series component
        arima_model = None
        if STATSMODELS_AVAILABLE:
            try:
                # Use last 60 days for ARIMA (or less if not available)
                arima_data = df['price'].iloc[-min(60, len(df)):]
                
                # Find optimal ARIMA parameters
                arima_model = SARIMAX(
                    arima_data,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                
                logger.info(f"ARIMA model trained successfully for {symbol}")
            except Exception as e:
                logger.warning(f"Error training ARIMA model for {symbol}: {e}")
        
        # Save models and metadata
        model_info = {
            'random_forest': rf_model,
            'arima': arima_model,
            'scaler': scaler,
            'features': available_features,
            'last_price': df['price'].iloc[-1],
            'last_date': df.index[-1],
            'training_data_points': len(df),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save models to disk
        models_path = os.path.join(self.model_dir, f"{symbol}_short_term.joblib")
        joblib.dump(model_info, models_path)
        
        # Store in memory
        self.models[f"{symbol}_short"] = model_info
        
        # Evaluate on training data
        y_pred = rf_model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        logger.info(f"Training metrics for {symbol} short-term model:")
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Calculate feature importance
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            feature_importance = [(available_features[i], importance[i]) for i in range(len(available_features))]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Top features for {symbol}:")
            for feat, imp in feature_importance[:5]:
                logger.info(f"  {feat}: {imp:.4f}")
        
        return model_info
    
    def train_long_term_model(self, symbol, market_df, sentiment_df=None, onchain_df=None):
        """
        Train a long-term (7-30 day) forecasting model using Prophet or fallback method.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            
        Returns:
            Dict with trained model and metadata
        """
        logger.info(f"Training long-term model for {symbol}")
        
        # Prepare features - simpler for Prophet
        if market_df is None or len(market_df) < 30:
            logger.error(f"Insufficient data for training long-term model for {symbol}")
            return None
        
        # Create copy to avoid modifying original
        df = market_df.copy()
        
        # Ensure we have a timestamp column
        if 'timestamp' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.error("No timestamp column or datetime index found")
            return None
        
        # Reset index if timestamp is the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Make sure price column exists (use close if no price)
        if 'price' not in df.columns and 'close' in df.columns:
            df['price'] = df['close']
        elif 'price' not in df.columns:
            logger.error("No price or close column found in market data")
            return None
        
        # Try Prophet if available
        if PROPHET_AVAILABLE:
            return self._train_prophet_model(symbol, df, sentiment_df)
        else:
            logger.warning("Prophet not available, using Random Forest for long-term forecasting")
            return self._train_rf_long_term(symbol, df, sentiment_df)
    
    def _train_prophet_model(self, symbol, df, sentiment_df=None):
        """Train a Prophet model for long-term forecasting"""
        # Prepare data for Prophet (which expects 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['timestamp']),
            'y': df['price']
        })
        
        # Add sentiment as a regressor if available
        if sentiment_df is not None and not sentiment_df.empty:
            # Get average daily sentiment
            sentiment_df['date'] = pd.to_datetime(sentiment_df.index if isinstance(sentiment_df.index, pd.DatetimeIndex) 
                                                else sentiment_df['date'] if 'date' in sentiment_df.columns else None)
            
            if sentiment_df['date'] is not None:
                daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date)['sentiment_score'].mean().reset_index()
                daily_sentiment.columns = ['ds', 'sentiment']
                
                # Convert ds to datetime
                daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
                
                # Merge with prophet_df
                prophet_df = pd.merge(prophet_df, daily_sentiment, on='ds', how='left')
                
                # Fill missing sentiment with neutral (0.5)
                prophet_df['sentiment'] = prophet_df['sentiment'].fillna(0.5)
        
        # Initialize and train Prophet model
        try:
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            # Add sentiment as a regressor if available
            if 'sentiment' in prophet_df.columns:
                model.add_regressor('sentiment')
            
            # Add extra regressors from market data
            if 'volume' in df.columns or 'volume_24h' in df.columns:
                volume_col = 'volume' if 'volume' in df.columns else 'volume_24h'
                prophet_df['volume'] = df[volume_col]
                model.add_regressor('volume')
            
            # Fit the model
            model.fit(prophet_df)
            
            # Save model and metadata
            model_info = {
                'prophet_model': model,
                'prophet_df': prophet_df,
                'has_sentiment': 'sentiment' in prophet_df.columns,
                'has_volume': 'volume' in prophet_df.columns,
                'last_price': df['price'].iloc[-1],
                'last_date': pd.to_datetime(df['timestamp'].iloc[-1]),
                'training_data_points': len(df),
                'trained_at': datetime.now().isoformat()
            }
            
            # Save model to disk
            models_path = os.path.join(self.model_dir, f"{symbol}_long_term.joblib")
            joblib.dump(model_info, models_path)
            
            # Store in memory
            self.models[f"{symbol}_long"] = model_info
            
            # Evaluate on training data
            future = model.make_future_dataframe(periods=1)
            
            # Add regressors to future dataframe
            if 'sentiment' in prophet_df.columns:
                future['sentiment'] = prophet_df['sentiment'].iloc[-1]  # Use last sentiment value
            
            if 'volume' in prophet_df.columns:
                future['volume'] = prophet_df['volume'].iloc[-1]  # Use last volume value
            
            forecast = model.predict(future)
            
            # Compare prediction with actual
            forecast_dates = pd.to_datetime(forecast['ds']).dt.date
            prophet_dates = pd.to_datetime(prophet_df['ds']).dt.date
            
            # Find matching dates
            common_dates = set(forecast_dates).intersection(set(prophet_dates))
            
            if common_dates:
                # Filter to common dates
                forecast_filtered = forecast[forecast['ds'].dt.date.isin(common_dates)]
                prophet_filtered = prophet_df[prophet_df['ds'].dt.date.isin(common_dates)]
                
                # Sort by date
                forecast_filtered = forecast_filtered.sort_values('ds')
                prophet_filtered = prophet_filtered.sort_values('ds')
                
                # Calculate metrics
                mae = np.mean(np.abs(forecast_filtered['yhat'].values - prophet_filtered['y'].values))
                rmse = np.sqrt(np.mean((forecast_filtered['yhat'].values - prophet_filtered['y'].values) ** 2))
                
                logger.info(f"Training metrics for {symbol} long-term model:")
                logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            return model_info
        
        except Exception as e:
            logger.error(f"Error training Prophet model for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _train_rf_long_term(self, symbol, df, sentiment_df=None):
        """Fallback Random Forest model for long-term forecasting when Prophet is not available"""
        # Prepare more features for random forest
        prepared_df = self._prepare_features(df, sentiment_df)
        if prepared_df is None or len(prepared_df) < 30:
            logger.error(f"Insufficient prepared data for training long-term RF model for {symbol}")
            return None
            
        # Targets for different forecast horizons
        targets = {}
        for horizon in [7, 14, 30]:
            if f'target_{horizon}d' in prepared_df.columns:
                targets[horizon] = prepared_df[f'target_{horizon}d'].dropna()
                
        if not targets:
            logger.error("No valid targets for long-term forecasting")
            return None
            
        # Select features
        feature_cols = [col for col in prepared_df.columns if col not in 
                      ['target_next_day'] + [f'target_{d}d' for d in [3, 7, 14, 30]]]
        
        # Train models for different horizons
        models = {}
        for horizon, target_series in targets.items():
            # Filter to rows with this target
            valid_idx = target_series.index
            X = prepared_df.loc[valid_idx, feature_cols].values
            y = target_series.values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            rf_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_scaled, y)
            
            # Save model and scaler
            models[horizon] = {
                'model': rf_model,
                'scaler': scaler,
                'features': feature_cols
            }
            
            # Evaluate
            y_pred = rf_model.predict(X_scaled)
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            logger.info(f"{horizon}-day forecast model RMSE: {rmse:.4f}")
        
        # Create model info
        model_info = {
            'models': models,
            'features': feature_cols,
            'last_price': df['price'].iloc[-1],
            'last_date': pd.to_datetime(df['timestamp'].iloc[-1]),
            'training_data_points': len(df),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save to disk
        models_path = os.path.join(self.model_dir, f"{symbol}_long_term.joblib")
        joblib.dump(model_info, models_path)
        
        # Store in memory
        self.models[f"{symbol}_long"] = model_info
        
        return model_info
    
    def forecast(self, symbol, market_df, horizon='short', days_ahead=7, 
                sentiment_df=None, onchain_df=None, retrain=False):
        """
        Generate forecasts with confidence intervals.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            horizon: 'short' or 'long'
            days_ahead: Number of days to forecast
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            retrain: Whether to retrain the model
            
        Returns:
            DataFrame with forecasts
        """
        logger.info(f"Generating {horizon}-term {days_ahead}-day forecast for {symbol}")
        
        # Determine model key
        model_key = f"{symbol}_{horizon}"
        
        # Check if model exists
        model_exists = model_key in self.models
        
        # Load model from disk if not in memory
        if not model_exists:
            try:
                model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}_term.joblib")
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    model_exists = True
                    logger.info(f"Loaded {horizon}-term model for {symbol} from disk")
            except Exception as e:
                logger.warning(f"Failed to load model from disk: {e}")
        
        # Train model if it doesn't exist or retraining requested
        if not model_exists or retrain:
            logger.info(f"Training new {horizon}-term model for {symbol}")
            
            if horizon == 'short':
                model_info = self.train_short_term_model(symbol, market_df, sentiment_df, onchain_df)
            else:
                model_info = self.train_long_term_model(symbol, market_df, sentiment_df, onchain_df)
                
            if model_info is None:
                logger.error(f"Failed to train {horizon}-term model for {symbol}")
                return None
        else:
            model_info = self.models[model_key]
        
        # Generate forecast based on horizon
        if horizon == 'short':
            return self._generate_short_term_forecast(symbol, model_info, market_df, sentiment_df, days_ahead)
        else:
            return self._generate_long_term_forecast(symbol, model_info, market_df, sentiment_df, days_ahead)
    
    def _generate_short_term_forecast(self, symbol, model_info, market_df, sentiment_df=None, days_ahead=7):
        """
        Generate short-term forecast using ensemble of Random Forest and ARIMA.
        
        Args:
            symbol: Trading symbol
            model_info: Model information dictionary
            market_df: Market data
            sentiment_df: Sentiment data
            days_ahead: Days to forecast
            
        Returns:
            DataFrame with forecast
        """
        # Extract model components
        rf_model = model_info.get('random_forest')
        arima_model = model_info.get('arima')
        scaler = model_info.get('scaler')
        features = model_info.get('features')
        last_price = model_info.get('last_price')
        last_date = model_info.get('last_date')
        
        if not isinstance(last_date, datetime):
            last_date = pd.to_datetime(last_date)
        
        if rf_model is None or scaler is None or features is None:
            logger.error("Missing required model components for forecasting")
            return None
        
        # Prepare data for prediction
        prepared_df = self._prepare_features(market_df, sentiment_df)
        if prepared_df is None:
            logger.error("Failed to prepare features for forecasting")
            return None
            
        # Get the latest data point for initial prediction
        if len(prepared_df) > 0:
            latest_data = prepared_df.iloc[-1]
            feature_values = [latest_data[feature] if feature in latest_data else 0 for feature in features]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Scale the features
            scaled_features = scaler.transform(feature_array)
        else:
            logger.error("No data points available in prepared DataFrame")
            return None
        
        # Initialize predictions
        predictions = []
        latest_price = latest_data['price']
        current_date = latest_data.name if isinstance(latest_data.name, datetime) else last_date
        
        # Create future dates
        future_dates = [current_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Generate bootstrap samples for uncertainty
        bootstrap_samples = 100
        
        for day_idx, future_date in enumerate(future_dates):
            # 1. Generate Random Forest prediction
            if hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0:
                # Use the individual trees for prediction distribution
                tree_predictions = []
                for tree in rf_model.estimators_[:bootstrap_samples]:
                    tree_predictions.append(tree.predict(scaled_features)[0])
                
                rf_prediction = np.mean(tree_predictions)
                rf_std = np.std(tree_predictions)
            else:
                # Standard prediction
                rf_prediction = rf_model.predict(scaled_features)[0]
                rf_std = latest_price * 0.02  # Assume 2% standard deviation
            
            # 2. Generate ARIMA prediction if available
            arima_prediction = None
            arima_std = None
            
            if arima_model is not None:
                try:
                    # Forecast next step
                    arima_forecast = arima_model.get_forecast(steps=day_idx+1)
                    arima_prediction = arima_forecast.predicted_mean.iloc[-1]
                    arima_std = np.sqrt(arima_forecast.var_pred_mean.iloc[-1])
                except Exception as e:
                    logger.warning(f"ARIMA forecast failed: {e}")
            
            # 3. Ensemble predictions
            if arima_prediction is not None:
                # Weighted average of RF and ARIMA
                prediction = 0.7 * rf_prediction + 0.3 * arima_prediction
                # Combined uncertainty
                pred_std = np.sqrt(0.7**2 * rf_std**2 + 0.3**2 * arima_std**2)
            else:
                # Just use RF prediction
                prediction = rf_prediction
                pred_std = rf_std
            
            # 4. Calculate confidence intervals (95%)
            lower_bound = max(0, prediction - 1.96 * pred_std)
            upper_bound = prediction + 1.96 * pred_std
            
            # 5. Calculate percent change
            pct_change = ((prediction - latest_price) / latest_price) * 100
            
            # Store prediction
            predictions.append({
                'date': future_date,
                'forecast_price': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std_dev': pred_std,
                'change_pct': pct_change
            })
            
            # Update for next iteration
            latest_price = prediction
            
            # Update feature vector for next day's prediction
            # (Note: this is a simple approach - in a full system, we'd update all features properly)
            if 'price' in features:
                idx = features.index('price')
                feature_array[0, idx] = prediction
                
            # Update price_roc_1 if it exists
            if 'price_roc_1' in features and day_idx > 0:
                idx = features.index('price_roc_1')
                prev_price = predictions[day_idx-1]['forecast_price']
                feature_array[0, idx] = (prediction - prev_price) / prev_price
                
            # Rescale for next prediction
            scaled_features = scaler.transform(feature_array)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(predictions)
        
        return forecast_df
    
    def _generate_long_term_forecast(self, symbol, model_info, market_df, sentiment_df=None, days_ahead=30):
        """
        Generate long-term forecast using Prophet or Random Forest.
        
        Args:
            symbol: Trading symbol
            model_info: Model information dictionary
            market_df: Market data
            sentiment_df: Sentiment data
            days_ahead: Days to forecast
            
        Returns:
            DataFrame with forecast
        """
        # Check if this is a Prophet model
        if 'prophet_model' in model_info:
            return self._generate_prophet_forecast(symbol, model_info, sentiment_df, days_ahead)
        else:
            return self._generate_rf_long_term_forecast(symbol, model_info, market_df, sentiment_df, days_ahead)
    
    def _generate_prophet_forecast(self, symbol, model_info, sentiment_df=None, days_ahead=30):
        """Generate forecast using Prophet model"""
        prophet_model = model_info.get('prophet_model')
        has_sentiment = model_info.get('has_sentiment', False)
        has_volume = model_info.get('has_volume', False)
        last_date = model_info.get('last_date')
        
        if prophet_model is None:
            logger.error("Prophet model not found")
            return None
            
        try:
            # Create future dataframe
            future = prophet_model.make_future_dataframe(periods=days_ahead)
            
            # Add regressors if they were used in training
            if has_sentiment:
                # Use latest sentiment or default to neutral
                latest_sentiment = 0.5
                
                if sentiment_df is not None and not sentiment_df.empty:
                    latest_sentiment = sentiment_df['sentiment_score'].iloc[-1] if len(sentiment_df) > 0 else 0.5
                
                future['sentiment'] = latest_sentiment
            
            if has_volume:
                # Use last volume from training
                prophet_df = model_info.get('prophet_df')
                last_volume = prophet_df['volume'].iloc[-1] if prophet_df is not None else 0
                future['volume'] = last_volume
            
            # Generate forecast
            forecast = prophet_model.predict(future)
            
            # Keep only future days
            if last_date is not None:
                if not isinstance(last_date, datetime):
                    last_date = pd.to_datetime(last_date)
                
                future_forecast = forecast[forecast['ds'] > last_date]
            else:
                # Just take the last days_ahead rows
                future_forecast = forecast.iloc[-days_ahead:]
            
            # Format output
            result = pd.DataFrame({
                'date': future_forecast['ds'],
                'forecast_price': future_forecast['yhat'],
                'lower_bound': future_forecast['yhat_lower'],
                'upper_bound': future_forecast['yhat_upper'],
                'std_dev': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']) / 3.92  # 95% CI width ÷ 3.92
            })
            
            # Calculate percent change
            last_price = model_info.get('last_price', result['forecast_price'].iloc[0])
            result['change_pct'] = ((result['forecast_price'] - last_price) / last_price) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_rf_long_term_forecast(self, symbol, model_info, market_df, sentiment_df=None, days_ahead=30):
        """Generate long-term forecast using Random Forest models"""
        # Get model info
        models = model_info.get('models', {})
        features = model_info.get('features', [])
        last_price = model_info.get('last_price')
        last_date = model_info.get('last_date')
        
        if not models or not features:
            logger.error("Missing models or features for long-term forecasting")
            return None
            
        if not isinstance(last_date, datetime):
            last_date = pd.to_datetime(last_date)
        
        # Prepare input data
        prepared_df = self._prepare_features(market_df, sentiment_df)
        if prepared_df is None or len(prepared_df) == 0:
            logger.error("Failed to prepare features for long-term forecasting")
            return None
            
        # Get the latest data point
        latest_data = prepared_df.iloc[-1]
        
        # Create future dates
        future_dates = [latest_data.name + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Find the closest available horizon model
        available_horizons = sorted(models.keys())
        if not available_horizons:
            logger.error("No horizon models available")
            return None
            
        # Select suitable models based on forecast day
        predictions = []
        
        for day_idx, future_date in enumerate(future_dates):
            forecast_day = day_idx + 1
            
            # Find nearest horizon model
            nearest_horizon = min(available_horizons, key=lambda x: abs(x - forecast_day))
            model_dict = models[nearest_horizon]
            
            rf_model = model_dict['model']
            scaler = model_dict['scaler']
            model_features = model_dict['features']
            
            # Prepare feature array
            feature_values = [latest_data[feature] if feature in latest_data else 0 for feature in model_features]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            scaled_features = scaler.transform(feature_array)
            
            # Generate prediction with uncertainty
            if hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0:
                # Use individual trees for prediction distribution
                tree_predictions = []
                n_trees = min(100, len(rf_model.estimators_))
                
                for tree in rf_model.estimators_[:n_trees]:
                    tree_predictions.append(tree.predict(scaled_features)[0])
                
                prediction = np.mean(tree_predictions)
                pred_std = np.std(tree_predictions)
            else:
                prediction = rf_model.predict(scaled_features)[0]
                pred_std = last_price * (0.02 * forecast_day)  # Increasing uncertainty with time
            
            # Calculate confidence intervals
            lower_bound = max(0, prediction - 1.96 * pred_std)
            upper_bound = prediction + 1.96 * pred_std
            
            # Calculate percent change
            pct_change = ((prediction - last_price) / last_price) * 100
            
            # Store prediction
            predictions.append({
                'date': future_date,
                'forecast_price': prediction,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std_dev': pred_std,
                'change_pct': pct_change
            })
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(predictions)
        
        return forecast_df
    
    def evaluate_model(self, symbol, market_df, horizon='short'):
        """
        Evaluate model performance on historical data.
        
        Args:
            symbol: Trading symbol
            market_df: Historical market data
            horizon: 'short' or 'long'
            
        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating {horizon}-term model for {symbol}")
        
        # Determine model key
        model_key = f"{symbol}_{horizon}"
        
        # Check if model exists
        if model_key not in self.models:
            try:
                model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}_term.joblib")
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"Loaded {horizon}-term model for {symbol} from disk")
                else:
                    logger.error(f"No {horizon}-term model found for {symbol}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load model from disk: {e}")
                return None
        
        # Get model info
        model_info = self.models[model_key]
        
        # Prepare evaluation data
        prepared_df = self._prepare_features(market_df)
        if prepared_df is None or len(prepared_df) < 30:
            logger.error("Insufficient data for model evaluation")
            return None
        
        # Split data into train/test (use last 30 days as test)
        test_size = min(30, len(prepared_df) // 3)  # Use 1/3 of data or 30 days, whichever is smaller
        train_df = prepared_df.iloc[:-test_size]
        test_df = prepared_df.iloc[-test_size:]
        
        # Prepare metrics
        results = {
            'symbol': symbol,
            'horizon': horizon,
            'test_size': test_size,
            'evaluation_date': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Evaluate based on horizon
        if horizon == 'short':
            # Evaluate Random Forest model
            rf_model = model_info.get('random_forest')
            scaler = model_info.get('scaler')
            features = model_info.get('features')
            
            if rf_model is None or scaler is None or features is None:
                logger.error("Missing required components for short-term evaluation")
                return None
            
            # Extract features and target
            X_test = test_df[features].values
            y_test = test_df['target_next_day'].values
            
            # Scale features
            X_test_scaled = scaler.transform(X_test)
            
            # Generate predictions
            y_pred = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Store metrics
            results['metrics'] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            logger.info(f"Evaluation metrics for {symbol} short-term model:")
            logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
            
        else:
            # Evaluate long-term models
            # This would depend on whether it's a Prophet model or RF model
            if 'prophet_model' in model_info:
                # Evaluate Prophet model
                prophet_model = model_info.get('prophet_model')
                
                if prophet_model is None:
                    logger.error("Missing Prophet model for evaluation")
                    return None
                
                # Create historical dataframe for Prophet
                historical = prophet_model.history
                
                # Calculate metrics on historical predictions vs actual
                y_true = historical['y'].values
                y_pred = historical['yhat'].values
                
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                # Calculate MAPE
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                results['metrics'] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape)
                }
                
                logger.info(f"Evaluation metrics for {symbol} Prophet model:")
                logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                
            elif 'models' in model_info:
                # Evaluate RF long-term models
                models_dict = model_info.get('models', {})
                features = model_info.get('features', [])
                
                if not models_dict or not features:
                    logger.error("Missing required components for RF long-term evaluation")
                    return None
                
                # Evaluate each horizon model
                horizon_metrics = {}
                
                for horizon, model_dict in models_dict.items():
                    rf_model = model_dict['model']
                    scaler = model_dict['scaler']
                    
                    # Target for this horizon
                    target_col = f'target_{horizon}d'
                    if target_col not in test_df.columns:
                        continue
                    
                    # Extract features and target
                    valid_idx = test_df[target_col].dropna().index
                    if len(valid_idx) < 5:
                        continue
                        
                    X_test = test_df.loc[valid_idx, features].values
                    y_test = test_df.loc[valid_idx, target_col].values
                    
                    # Scale features
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Generate predictions
                    y_pred = rf_model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                    
                    horizon_metrics[str(horizon)] = {
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'mape': float(mape)
                    }
                    
                    logger.info(f"{horizon}-day forecast model metrics:")
                    logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                
                # Store all metrics
                results['metrics'] = horizon_metrics
        
        # Save evaluation results
        eval_path = os.path.join(self.model_dir, f"{symbol}_{horizon}_evaluation.joblib")
        joblib.dump(results, eval_path)
        
        return results
    
    def plot_forecast(self, symbol, forecast_df, market_df=None, output_path=None):
        """
        Plot forecast with confidence intervals.
        
        Args:
            symbol: Trading symbol
            forecast_df: DataFrame with forecast
            market_df: DataFrame with historical data
            output_path: Path to save the plot
            
        Returns:
            Path to saved plot or None
        """
        if forecast_df is None or len(forecast_df) == 0:
            logger.error("No forecast data to plot")
            return None
        
        try:
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot historical data if available
            if market_df is not None and len(market_df) > 0:
                # Ensure timestamp is datetime
                if 'timestamp' in market_df.columns:
                    x = pd.to_datetime(market_df['timestamp'])
                elif isinstance(market_df.index, pd.DatetimeIndex):
                    x = market_df.index
                else:
                    x = pd.to_datetime(market_df.index)
                
                # Get price column
                if 'price' in market_df.columns:
                    y = market_df['price']
                elif 'close' in market_df.columns:
                    y = market_df['close']
                else:
                    logger.warning("No price or close column found in market data")
                    y = None
                
                if x is not None and y is not None:
                    plt.plot(x, y, 'b-', label='Historical Price')
            
            # Plot forecast
            x_forecast = forecast_df['date']
            y_forecast = forecast_df['forecast_price']
            
            plt.plot(x_forecast, y_forecast, 'r-', label='Forecast')
            
            # Plot confidence intervals
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                plt.fill_between(
                    x_forecast,
                    forecast_df['lower_bound'],
                    forecast_df['upper_bound'],
                    color='red',
                    alpha=0.2,
                    label='95% Confidence Interval'
                )
            
            # Format plot
            plt.title(f'{symbol} Price Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # Format y-axis as currency
            from matplotlib.ticker import FuncFormatter
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
            
            # Add annotations for key points
            for i, (idx, row) in enumerate(forecast_df.iterrows()):
                if i % max(1, len(forecast_df) // 5) == 0:  # Label every 5th point
                    plt.annotate(
                        f"${row['forecast_price']:.2f}\n({row['change_pct']:+.1f}%)",
                        (row['date'], row['forecast_price']),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8
                    )
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot if path provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Forecast plot saved to {output_path}")
            else:
                # Save to default location
                os.makedirs('plots', exist_ok=True)
                filename = f"plots/{symbol}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename)
                logger.info(f"Forecast plot saved to {filename}")
                output_path = filename
            
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def detect_latest_anomalies(self, symbol, market_df, lookback_days=30):
        """
        Detect anomalies in recent market data.
        
        Args:
            symbol: Trading symbol
            market_df: Market data
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with anomalies
        """
        logger.info(f"Detecting anomalies for {symbol}")
        
        if market_df is None or len(market_df) == 0:
            logger.error("No market data provided for anomaly detection")
            return None
        
        # Prepare features
        df = self._prepare_features(market_df)
        if df is None:
            logger.error("Failed to prepare features for anomaly detection")
            return None
        
        # Calculate lookback period
        if lookback_days is not None and lookback_days > 0:
            lookback_start = df.index[-1] - pd.Timedelta(days=lookback_days)
            df = df[df.index >= lookback_start]
        
        # Detect anomalies
        df_with_anomalies = self._detect_anomalies(df, symbol)
        
        # Filter to anomalies only
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']].copy()
        
        if len(anomalies) == 0:
            logger.info(f"No anomalies detected for {symbol} in the last {lookback_days} days")
            return pd.DataFrame()
        
        # Add description
        anomalies['description'] = anomalies.apply(
            lambda row: self._get_anomaly_description(row, df_with_anomalies), 
            axis=1
        )
        
        # Select relevant columns
        result_cols = ['price', 'anomaly_score', 'anomaly_type', 'description']
        result_cols = [col for col in result_cols if col in anomalies.columns]
        
        return anomalies[result_cols]
    
    def _get_anomaly_description(self, row, df):
        """Generate descriptive text for an anomaly"""
        anomaly_type = row.get('anomaly_type')
        
        if anomaly_type == 'price_movement':
            # Calculate percent change
            price_change = row.get('price_roc_1', 0) * 100
            if abs(price_change) > 0.1:
                direction = 'increase' if price_change > 0 else 'decrease'
                return f"Unusual price {direction} of {abs(price_change):.2f}%"
            else:
                return "Abnormal price pattern"
                
        elif anomaly_type == 'volume_spike':
            # Calculate volume change
            volume_change = row.get('volume_roc_1', 0) * 100
            return f"Volume spike of {abs(volume_change):.2f}%"
            
        else:
            # Generic description
            return "Unusual market behavior detected"
    
    def auto_retrain(self, symbols, market_data_func, sentiment_data_func=None, force=False):
        """
        Auto-retrain models for multiple symbols.
        
        Args:
            symbols: List of symbols to train
            market_data_func: Function to get market data for a symbol
            sentiment_data_func: Function to get sentiment data for a symbol
            force: Force retraining regardless of last training date
            
        Returns:
            Dict with training results
        """
        logger.info(f"Auto-retraining models for {len(symbols)} symbols")
        
        results = {
            'total': len(symbols),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': {}
        }
        
        for symbol in symbols:
            symbol_results = {
                'short_term': {'status': 'skipped', 'error': None},
                'long_term': {'status': 'skipped', 'error': None}
            }
            
            try:
                # Check if models need retraining
                short_model_key = f"{symbol}_short"
                long_model_key = f"{symbol}_long"
                
                short_needs_training = force
                long_needs_training = force
                
                if not force:
                    # Check last training date for short-term model
                    if short_model_key in self.models:
                        trained_at = self.models[short_model_key].get('trained_at')
                        if trained_at:
                            trained_date = pd.to_datetime(trained_at)
                            days_since_training = (datetime.now() - trained_date).days
                            short_needs_training = days_since_training >= 7  # Retrain weekly
                    else:
                        # Check if model exists on disk
                        model_path = os.path.join(self.model_dir, f"{symbol}_short_term.joblib")
                        if not os.path.exists(model_path):
                            short_needs_training = True
                    
                    # Check long-term model
                    if long_model_key in self.models:
                        trained_at = self.models[long_model_key].get('trained_at')
                        if trained_at:
                            trained_date = pd.to_datetime(trained_at)
                            days_since_training = (datetime.now() - trained_date).days
                            long_needs_training = days_since_training >= 30  # Retrain monthly
                    else:
                        # Check if model exists on disk
                        model_path = os.path.join(self.model_dir, f"{symbol}_long_term.joblib")
                        if not os.path.exists(model_path):
                            long_needs_training = True
                
                # Get market data
                market_df = market_data_func(symbol)
                
                # Get sentiment data if available
                sentiment_df = None
                if sentiment_data_func:
                    sentiment_df = sentiment_data_func(symbol)
                
                # Train models as needed
                if short_needs_training:
                    logger.info(f"Training short-term model for {symbol}")
                    try:
                        result = self.train_short_term_model(symbol, market_df, sentiment_df)
                        if result:
                            symbol_results['short_term']['status'] = 'success'
                            results['success'] += 1
                        else:
                            symbol_results['short_term']['status'] = 'failed'
                            symbol_results['short_term']['error'] = 'Training failed'
                            results['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error training short-term model for {symbol}: {e}")
                        symbol_results['short_term']['status'] = 'failed'
                        symbol_results['short_term']['error'] = str(e)
                        results['failed'] += 1
                else:
                    logger.info(f"Skipping short-term model training for {symbol} (recent)")
                    results['skipped'] += 1
                
                if long_needs_training:
                    logger.info(f"Training long-term model for {symbol}")
                    try:
                        result = self.train_long_term_model(symbol, market_df, sentiment_df)
                        if result:
                            symbol_results['long_term']['status'] = 'success'
                            results['success'] += 1
                        else:
                            symbol_results['long_term']['status'] = 'failed'
                            symbol_results['long_term']['error'] = 'Training failed'
                            results['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error training long-term model for {symbol}: {e}")
                        symbol_results['long_term']['status'] = 'failed'
                        symbol_results['long_term']['error'] = str(e)
                        results['failed'] += 1
                else:
                    logger.info(f"Skipping long-term model training for {symbol} (recent)")
                    results['skipped'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                symbol_results['short_term']['status'] = 'failed'
                symbol_results['short_term']['error'] = str(e)
                symbol_results['long_term']['status'] = 'failed'
                symbol_results['long_term']['error'] = str(e)
                results['failed'] += 2
            
            # Store results for this symbol
            results['details'][symbol] = symbol_results
        
        logger.info(f"Auto-retraining completed: {results['success']} successful, {results['failed']} failed, {results['skipped']} skipped")
        
        return results

# Example usage when imported as a library
if __name__ == "__main__":
    # Example standalone usage
    forecaster = ChronosForecaster()
    logger.info("ChronosForecaster test initialization complete")