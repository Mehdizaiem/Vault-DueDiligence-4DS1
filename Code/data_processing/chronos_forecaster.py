import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
import os
import sys
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Try to import deep learning libraries, but provide graceful fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, save_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D
    from tensorflow.keras.layers import Bidirectional, BatchNormalization, TimeDistributed
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models disabled.")

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
    with confidence intervals, anomaly detection, sentiment integration, and
    ensemble learning.
    
    Supports:
    - Random Forest regression models
    - Prophet models
    - LSTM neural networks 
    - Transformer-based models
    - SARIMAX models
    - Ensemble forecasting combining multiple models
    - Anomaly detection
    - Confidence interval generation
    - Multi-horizon forecasting
    """
    
    def __init__(self, model_dir="models/chronos"):
        """
        Initialize the ChronosForecaster.
        
        Args:
            model_dir: Directory for storing trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        
        # Available model types
        self.available_models = ["random_forest", "prophet", "sarimax", "ensemble"]
        
        # Track deep learning availability
        self.dl_available = TENSORFLOW_AVAILABLE
        if self.dl_available:
            self.available_models.extend(["lstm", "transformer"])
            
            # Configure TensorFlow
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU support enabled with {len(gpus)} GPUs")
                else:
                    logger.info("No GPU found, using CPU for deep learning")
            except Exception as e:
                logger.warning(f"Error configuring GPU: {e}")
        
        logger.info(f"ChronosForecaster initialized with model_dir: {model_dir}")
        logger.info(f"Available models: {', '.join(self.available_models)}")
        logger.info(f"SARIMAX available: {STATSMODELS_AVAILABLE}")
        logger.info(f"Prophet available: {PROPHET_AVAILABLE}")
        logger.info(f"Deep learning available: {self.dl_available}")
    
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
            
            # Add volatility indicators
            df['volatility'] = df['std_20'] / df['ma_20']
        
        # 5. Rate of Change and Momentum
        df['price_roc_1'] = df['price'].pct_change(periods=1)
        if len(df) > 5:
            df['price_roc_5'] = df['price'].pct_change(periods=5)
        
        if 'volume' in df.columns or 'volume_24h' in df.columns:
            volume_col = 'volume' if 'volume' in df.columns else 'volume_24h'
            df['volume_roc_1'] = df[volume_col].pct_change(periods=1)
            df['price_volume_ratio'] = df['price'] / df[volume_col].replace(0, np.nan)
            
            # Volume moving averages
            df[f'volume_ma_7'] = df[volume_col].rolling(window=min(7, len(df) - 1)).mean()
            
            # Relative volume
            df['rel_volume'] = df[volume_col] / df[f'volume_ma_7']
        
        # 6. RSI (Relative Strength Index)
        rsi_period = min(14, len(df) - 1)
        if rsi_period > 2:
            delta = df['price'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Add time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek >= 5
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['day_of_year'] = df.index.dayofyear
        
        # Cyclic encoding of time features for better ML performance
        df['day_sin'] = np.sin(df['day_of_year'] * (2 * np.pi / 365.25))
        df['day_cos'] = np.cos(df['day_of_year'] * (2 * np.pi / 365.25))
        df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
        df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))
        df['weekday_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['weekday_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
        
        # Add lagged features (for capturing autocorrelation)
        for lag in [1, 3, 7, 14]:
            if len(df) > lag:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)
                df[f'price_diff_{lag}'] = df['price'].diff(lag)
                # Relative price compared to lagged value
                df[f'price_rel_{lag}'] = df['price'] / df[f'price_lag_{lag}']
        
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
            
            # Sentiment moving average
            df['sentiment_ma_7'] = df['sentiment_score'].rolling(min(7, len(df))).mean()
            
            # Sentiment-price correlation features
            # Correlation between sentiment and future price change (predictive)
            df['sent_price_corr'] = df['sentiment_score'].rolling(min(10, len(df))).corr(df['price'].pct_change().shift(-1))
        else:
            # Add placeholder sentiment
            df['sentiment_score'] = 0.5
            df['sentiment_momentum'] = 0
        
        # Add on-chain features if available
        if onchain_df is not None and not onchain_df.empty:
            # Ensure onchain_df index is datetime
            if not isinstance(onchain_df.index, pd.DatetimeIndex):
                if 'date' in onchain_df.columns or 'timestamp' in onchain_df.columns:
                    date_col = 'date' if 'date' in onchain_df.columns else 'timestamp'
                    onchain_df = onchain_df.set_index(date_col)
                else:
                    onchain_df.index = pd.to_datetime(onchain_df.index)
            
            # Identify numeric columns for onchain metrics
            onchain_numeric = onchain_df.select_dtypes(include=['float', 'int']).columns
            
            # Merge with market data
            df = df.join(onchain_df[onchain_numeric], how='left')
            
            # Fill missing values
            df[onchain_numeric] = df[onchain_numeric].fillna(method='ffill').fillna(0)
            
        # Handle missing values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Create target variable (next day's price)
        df['target_next_day'] = df['price'].shift(-1)
        
        # For multi-day forecasts, create additional targets
        for days in [3, 7, 14, 30]:
            if len(df) > days:
                df[f'target_{days}d'] = df['price'].shift(-days)
        
        # Calculate returns for various forecast horizons
        for days in [1, 3, 7, 14, 30]:
            if len(df) > days:
                df[f'return_{days}d'] = (df[f'target_{days}d'] / df['price']) - 1 if f'target_{days}d' in df.columns else np.nan
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target_next_day'])
        
        return df
    
    def _prepare_sequence_data(self, df: pd.DataFrame, target_col: str = 'price',
                              lookback: int = 30, forecast_horizon: int = 7, 
                              step: int = 1, feature_cols: List[str] = None,
                              test_split: float = 0.2) -> Tuple:
        """
        Prepare sequence data for LSTM and Transformer models.
        
        Args:
            df: DataFrame with time series data (should be sorted by time)
            target_col: Target column to predict
            lookback: Number of lookback steps (window size)
            forecast_horizon: Number of steps to forecast
            step: Step size for creating sequences
            feature_cols: Features to include in sequence data
            test_split: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test, scaler)
        """
        if not self.dl_available:
            logger.error("Deep learning libraries not available")
            return None
            
        if df is None or len(df) < lookback + forecast_horizon:
            logger.error("Insufficient data for sequence preparation")
            return None
        
        # Choose features to include
        if feature_cols is None:
            # Select numeric columns with some exceptions
            exclude_cols = [f'target_{d}d' for d in [1, 3, 7, 14, 30]]
            exclude_cols.extend([f'return_{d}d' for d in [1, 3, 7, 14, 30]])
            exclude_cols.extend(['target_next_day'])
            
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col not in exclude_cols]
        
        # Make sure target column is included
        if target_col not in feature_cols:
            feature_cols.append(target_col)
        
        # Select features and handle missing values
        df_selected = df[feature_cols].copy()
        df_selected = df_selected.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_selected)
        
        # Find target column index
        target_idx = feature_cols.index(target_col)
        
        # Create sequences
        X, y = [], []
        for i in range(0, len(scaled_data) - lookback - forecast_horizon + 1, step):
            X.append(scaled_data[i:(i + lookback)])
            
            # For single-step forecasting, just take the next value
            # For multi-step, take the next forecast_horizon values of the target
            if forecast_horizon == 1:
                y.append(scaled_data[i + lookback, target_idx])
            else:
                y.append(scaled_data[i + lookback:i + lookback + forecast_horizon, target_idx])
                
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logger.info(f"Prepared sequences with shape X: {X.shape}, y: {y.shape}")
        logger.info(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
        logger.info(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test, scaler, feature_cols
    
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
        anomaly_features = ['price_roc_1', 'volume_roc_1', 'rsi_14', 'bb_position', 'volatility']
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
        
        # Investigate volatility anomalies
        if 'volatility' in result_df.columns:
            volatility_anomalies = result_df[result_df['is_anomaly'] & (result_df['volatility'] > 0.1)]
            result_df.loc[volatility_anomalies.index, 'anomaly_type'] = 'high_volatility'
        
        # Count anomalies
        anomaly_count = result_df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies in {len(result_df)} data points for {symbol}")
        
        return result_df
    
    def train_short_term_model(self, symbol, market_df, sentiment_df=None, onchain_df=None):
        """
        Train a short-term (1-7 day) forecasting model using Random Forest.
        
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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [f'target_{d}d' for d in [1, 3, 7, 14, 30]]
        exclude_cols.extend([f'return_{d}d' for d in [1, 3, 7, 14, 30]])
        exclude_cols.append('target_next_day')
        
        available_features = [f for f in numeric_cols if f not in exclude_cols]
        
        if len(available_features) < 5:
            logger.warning(f"Very few features available for {symbol}: {available_features}")
        
        X = df[available_features].values
        y = df[target].values
        
        # Split into train/test
        test_size = min(30, len(df) // 3)
        X_train, X_test = X[:-test_size], X[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model with improved hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting as additional model
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        
        gb_model.fit(X_train_scaled, y_train)
        
        # Train ARIMA model for time series component if available
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
        
        # Evaluate models on test set
        rf_pred = rf_model.predict(X_test_scaled)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mae = mean_absolute_error(y_test, rf_pred)
        
        gb_pred = gb_model.predict(X_test_scaled)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        gb_mae = mean_absolute_error(y_test, gb_pred)
        
        logger.info(f"RF model RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
        logger.info(f"GB model RMSE: {gb_rmse:.4f}, MAE: {gb_mae:.4f}")
        
        # Create ensemble weights based on performance
        total_error = rf_rmse + gb_rmse
        rf_weight = 1 - (rf_rmse / total_error) if total_error > 0 else 0.5
        gb_weight = 1 - (gb_rmse / total_error) if total_error > 0 else 0.5
        
        # Normalize weights to sum to 1
        sum_weights = rf_weight + gb_weight
        rf_weight = rf_weight / sum_weights if sum_weights > 0 else 0.5
        gb_weight = gb_weight / sum_weights if sum_weights > 0 else 0.5
        
        logger.info(f"Ensemble weights - RF: {rf_weight:.2f}, GB: {gb_weight:.2f}")
        
        # Save models and metadata
        model_info = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'arima': arima_model,
            'scaler': scaler,
            'features': available_features,
            'last_price': df['price'].iloc[-1],
            'last_date': df.index[-1],
            'training_data_points': len(df),
            'test_metrics': {
                'rf_rmse': rf_rmse,
                'rf_mae': rf_mae,
                'gb_rmse': gb_rmse,
                'gb_mae': gb_mae
            },
            'ensemble_weights': {
                'rf_weight': rf_weight,
                'gb_weight': gb_weight
            },
            'trained_at': datetime.now().isoformat()
        }
        
        # Save models to disk
        models_path = os.path.join(self.model_dir, f"{symbol}_short_term.joblib")
        joblib.dump(model_info, models_path)
        
        # Store in memory
        self.models[f"{symbol}_short"] = model_info
        
        # Calculate feature importance
        if hasattr(rf_model, 'feature_importances_'):
            importance = rf_model.feature_importances_
            feature_importance = [(available_features[i], importance[i]) for i in range(len(available_features))]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Top features for {symbol}:")
            for feat, imp in feature_importance[:5]:
                logger.info(f"  {feat}: {imp:.4f}")
        
        return model_info
    
    def train_lstm_model(self, symbol, market_df, sentiment_df=None, onchain_df=None, 
                       forecast_horizon=7, lookback=30):
        """
        Train an LSTM model for time series forecasting.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            forecast_horizon: Number of days to forecast
            lookback: Number of previous time steps to use
            
        Returns:
            Dict with trained model and metadata
        """
        if not self.dl_available:
            logger.error("TensorFlow not available for LSTM model training")
            return {"error": "TensorFlow not available"}
        
        logger.info(f"Training LSTM model for {symbol}")
        
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df, onchain_df)
        if df is None or len(df) < lookback + forecast_horizon + 30:  # Extra data needed for train/test
            logger.error(f"Insufficient data for LSTM model for {symbol}")
            return {"error": "Insufficient data for LSTM model"}
        
        # Prepare sequence data
        sequence_data = self._prepare_sequence_data(
            df=df,
            target_col='price',
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            feature_cols=None  # Use default features
        )
        
        if sequence_data is None:
            return {"error": "Failed to prepare sequence data"}
        
        X_train, y_train, X_test, y_test, scaler, features = sequence_data
        
        # Get dimensions
        n_features = X_train.shape[2]
        input_shape = (lookback, n_features)
        
        # Determine if multi-step or single-step output
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Build LSTM model
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(output_size))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        
        # Add model checkpoint
        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.h5")
        callbacks.append(ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ))
        
        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                epochs=200,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            evaluation = model.evaluate(X_test, y_test, verbose=0)
            test_loss = evaluation[0]
            test_mae = evaluation[1]
            
            logger.info(f"LSTM model training complete - test loss: {test_loss:.4f}, test MAE: {test_mae:.4f}")
            
            # Save model info
            model_info = {
                'model': model,
                'model_type': 'lstm',
                'scaler': scaler,
                'features': features,
                'lookback': lookback,
                'forecast_horizon': forecast_horizon,
                'input_shape': input_shape,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'training_history': history.history,
                'last_date': df.index[-1],
                'last_price': df['price'].iloc[-1],
                'trained_at': datetime.now().isoformat()
            }
            
            # Store in memory
            self.models[f"{symbol}_lstm"] = model_info
            
            # Save training history plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('LSTM Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('LSTM Training MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"{symbol}_lstm_training.png"))
            plt.close()
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"LSTM training failed: {str(e)}"}
    
    def train_transformer_model(self, symbol, market_df, sentiment_df=None, onchain_df=None,
                              forecast_horizon=7, lookback=30):
        """
        Train a Transformer model for time series forecasting.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            forecast_horizon: Number of days to forecast
            lookback: Number of previous time steps to use
            
        Returns:
            Dict with trained model and metadata
        """
        if not self.dl_available:
            logger.error("TensorFlow not available for Transformer model training")
            return {"error": "TensorFlow not available"}
        
        logger.info(f"Training Transformer model for {symbol}")
        
        # Prepare features
        df = self._prepare_features(market_df, sentiment_df, onchain_df)
        if df is None or len(df) < lookback + forecast_horizon + 30:
            logger.error(f"Insufficient data for Transformer model for {symbol}")
            return {"error": "Insufficient data for Transformer model"}
        
        # Prepare sequence data
        sequence_data = self._prepare_sequence_data(
            df=df,
            target_col='price',
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            feature_cols=None  # Use default features
        )
        
        if sequence_data is None:
            return {"error": "Failed to prepare sequence data"}
        
        X_train, y_train, X_test, y_test, scaler, features = sequence_data
        
        # Get dimensions
        seq_length, n_features = X_train.shape[1], X_train.shape[2]
        input_shape = (seq_length, n_features)
        
        # Determine if multi-step or single-step output
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        # Build Transformer model
        inputs = Input(shape=input_shape)
        
        # 1D convolution to capture local patterns
        x = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(inputs)
        
        # Add positional encoding (simplified)
        x = x + tf.keras.layers.Embedding(input_dim=seq_length, output_dim=128)(
            tf.range(start=0, limit=seq_length, delta=1))
        
        # Transformer layers
        for _ in range(2):
            # Multi-head self attention
            attention_output = MultiHeadAttention(
                key_dim=64, num_heads=4, dropout=0.1
            )(x, x)
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = Dense(256, activation="relu")(x)
            ffn = Dense(128)(ffn)
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation="relu")(x)
        outputs = Dense(output_size)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        ]
        
        # Add model checkpoint
        model_path = os.path.join(self.model_dir, f"{symbol}_transformer.h5")
        callbacks.append(ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ))
        
        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                epochs=200,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            evaluation = model.evaluate(X_test, y_test, verbose=0)
            test_loss = evaluation[0]
            test_mae = evaluation[1]
            
            logger.info(f"Transformer model training complete - test loss: {test_loss:.4f}, test MAE: {test_mae:.4f}")
            
            # Save model info
            model_info = {
                'model': model,
                'model_type': 'transformer',
                'scaler': scaler,
                'features': features,
                'lookback': lookback,
                'forecast_horizon': forecast_horizon,
                'input_shape': input_shape,
                'test_loss': test_loss,
                'test_mae': test_mae,
                'training_history': history.history,
                'last_date': df.index[-1],
                'last_price': df['price'].iloc[-1],
                'trained_at': datetime.now().isoformat()
            }
            
            # Store in memory
            self.models[f"{symbol}_transformer"] = model_info
            
            # Save training history plot
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Transformer Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Transformer Training MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"{symbol}_transformer_training.png"))
            plt.close()
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": f"Transformer training failed: {str(e)}"}
    
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
        
        # Add additional regressors from market data
        if 'volume' in df.columns or 'volume_24h' in df.columns:
            volume_col = 'volume' if 'volume' in df.columns else 'volume_24h'
            prophet_df['volume'] = df[volume_col]
            
            # Normalize volume
            max_volume = prophet_df['volume'].max()
            if max_volume > 0:
                prophet_df['volume'] = prophet_df['volume'] / max_volume
            
        # Initialize Prophet model with optimized parameters
        model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility in detecting changepoints
            seasonality_prior_scale=10.0,  # Stronger seasonality
            daily_seasonality=True,        # Enable daily patterns
            weekly_seasonality=True,       # Enable weekly patterns
            yearly_seasonality=True,       # Enable yearly patterns
            changepoint_range=0.95,        # Allow changepoints in most of the history
            interval_width=0.95            # 95% prediction intervals
        )
        
        # Add regressors if available
        if 'sentiment' in prophet_df.columns:
            model.add_regressor('sentiment', standardize=True)
            logger.info("Added sentiment as regressor to Prophet model")
        
        if 'volume' in prophet_df.columns:
            model.add_regressor('volume', standardize=True)
            logger.info("Added volume as regressor to Prophet model")
        
        # Add extra seasonality components
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Fit model with try/except to handle potential errors
        try:
            model.fit(prophet_df)
            logger.info(f"Prophet model trained successfully for {symbol}")
            
            # Create future dataframe for evaluation
            future = model.make_future_dataframe(periods=30)
            
            # Add regressor values to future dataframe
            if 'sentiment' in prophet_df.columns:
                # Use last sentiment value for future
                future['sentiment'] = prophet_df['sentiment'].iloc[-1]
            
            if 'volume' in prophet_df.columns:
                # Use last volume value for future
                future['volume'] = prophet_df['volume'].iloc[-1]
            
            # Generate forecast for future dates
            forecast = model.predict(future)
            
            # Save model info
            model_info = {
                'prophet_model': model,
                'prophet_df': prophet_df,
                'has_sentiment': 'sentiment' in prophet_df.columns,
                'has_volume': 'volume' in prophet_df.columns,
                'forecast': forecast,
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
            
            # Plot components
            try:
                fig1 = model.plot_components(forecast)
                fig1.savefig(os.path.join(self.model_dir, f"{symbol}_prophet_components.png"))
                plt.close(fig1)
                
                fig2 = model.plot(forecast)
                fig2.savefig(os.path.join(self.model_dir, f"{symbol}_prophet_forecast.png"))
                plt.close(fig2)
            except Exception as plot_e:
                logger.warning(f"Error plotting Prophet components: {plot_e}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _train_rf_long_term(self, symbol, df, sentiment_df=None):
        """Fallback Random Forest model for long-term forecasting when Prophet is not available"""
        # Create a time-aware feature set
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
            
            # Split into train/test
            test_size = min(30, len(X) // 5)
            X_train, X_test = X[:-test_size], X[-test_size:]
            y_train, y_test = y[:-test_size], y[-test_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Train Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                subsample=0.8
            )
            
            gb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_pred = rf_model.predict(X_test_scaled)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_mae = mean_absolute_error(y_test, rf_pred)
            
            gb_pred = gb_model.predict(X_test_scaled)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
            gb_mae = mean_absolute_error(y_test, gb_pred)
            
            # Calculate ensemble weights based on performance
            total_error = rf_rmse + gb_rmse
            rf_weight = 1 - (rf_rmse / total_error) if total_error > 0 else 0.5
            gb_weight = 1 - (gb_rmse / total_error) if total_error > 0 else 0.5
            
            # Normalize weights
            total_weight = rf_weight + gb_weight
            rf_weight /= total_weight
            gb_weight /= total_weight
            
            logger.info(f"{horizon}-day forecast model RMSE - RF: {rf_rmse:.4f}, GB: {gb_rmse:.4f}")
            logger.info(f"{horizon}-day forecast ensemble weights - RF: {rf_weight:.2f}, GB: {gb_weight:.2f}")
            
            # Save model and scaler
            models[horizon] = {
                'rf_model': rf_model,
                'gb_model': gb_model,
                'scaler': scaler,
                'features': feature_cols,
                'metrics': {
                    'rf_rmse': rf_rmse,
                    'rf_mae': rf_mae,
                    'gb_rmse': gb_rmse,
                    'gb_mae': gb_mae
                },
                'weights': {
                    'rf_weight': rf_weight,
                    'gb_weight': gb_weight
                }
            }
        
        # Create model info
        model_info = {
            'models': models,
            'features': feature_cols,
            'last_price': df['price'].iloc[-1],
            'last_date': pd.to_datetime(df['timestamp'].iloc[-1]),
            'training_data_points': len(df),
            'horizons': list(models.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save to disk
        models_path = os.path.join(self.model_dir, f"{symbol}_long_term.joblib")
        joblib.dump(model_info, models_path)
        
        # Store in memory
        self.models[f"{symbol}_long"] = model_info
        
        return model_info
    
    def train_ensemble_model(self, symbol, market_df, sentiment_df=None, onchain_df=None,
                           days_ahead=7, include_dl=True):
        """
        Train an ensemble of multiple model types for improved forecasting.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            sentiment_df: DataFrame with sentiment data
            onchain_df: DataFrame with on-chain metrics
            days_ahead: Number of days to forecast
            include_dl: Whether to include deep learning models
            
        Returns:
            Dict with ensemble model information
        """
        logger.info(f"Training ensemble model for {symbol} ({days_ahead} days ahead)")
        
        ensemble_models = {}
        model_weights = {}
        
        # 1. Short-term model
        short_term_model = self.train_short_term_model(symbol, market_df, sentiment_df, onchain_df)
        if short_term_model is not None:
            ensemble_models['short_term'] = short_term_model
            
            # Get metrics
            test_metrics = short_term_model.get('test_metrics', {})
            rf_rmse = test_metrics.get('rf_rmse', 1.0)
            gb_rmse = test_metrics.get('gb_rmse', 1.0)
            
            # Calculate weights based on inverse RMSE
            model_weights['short_term_rf'] = 1.0 / max(rf_rmse, 0.001)
            model_weights['short_term_gb'] = 1.0 / max(gb_rmse, 0.001)
        
        # 2. Long-term model if forecasting more than 3 days ahead
        if days_ahead >= 3:
            long_term_model = self.train_long_term_model(symbol, market_df, sentiment_df, onchain_df)
            if long_term_model is not None:
                ensemble_models['long_term'] = long_term_model
                
                # For Prophet models, use fixed weight
                if 'prophet_model' in long_term_model:
                    model_weights['prophet'] = 0.8  # Default for Prophet
                elif 'models' in long_term_model:
                    # Get closest horizon model
                    horizons = long_term_model.get('horizons', [])
                    if horizons:
                        closest_horizon = min(horizons, key=lambda x: abs(x - days_ahead))
                        horizon_model = long_term_model['models'][closest_horizon]
                        
                        # Get metrics
                        metrics = horizon_model.get('metrics', {})
                        rf_rmse = metrics.get('rf_rmse', 1.0)
                        gb_rmse = metrics.get('gb_rmse', 1.0)
                        
                        # Calculate weights based on inverse RMSE
                        model_weights['long_term_rf'] = 1.0 / max(rf_rmse, 0.001)
                        model_weights['long_term_gb'] = 1.0 / max(gb_rmse, 0.001)
        
        # 3. LSTM model if deep learning is available and requested
        if self.dl_available and include_dl:
            lstm_model = self.train_lstm_model(
                symbol, market_df, sentiment_df, onchain_df,
                forecast_horizon=days_ahead, lookback=30
            )
            
            if lstm_model is not None and not isinstance(lstm_model, dict) or \
               (isinstance(lstm_model, dict) and 'error' not in lstm_model):
                ensemble_models['lstm'] = lstm_model
                
                # Get LSTM test metrics
                test_mae = lstm_model.get('test_mae', 1.0)
                model_weights['lstm'] = 1.0 / max(test_mae, 0.001)
        
        # 4. Transformer model for longer sequences if deep learning is available
        if self.dl_available and include_dl and days_ahead >= 5:
            transformer_model = self.train_transformer_model(
                symbol, market_df, sentiment_df, onchain_df,
                forecast_horizon=days_ahead, lookback=60
            )
            
            if transformer_model is not None and not isinstance(transformer_model, dict) or \
               (isinstance(transformer_model, dict) and 'error' not in transformer_model):
                ensemble_models['transformer'] = transformer_model
                
                # Get Transformer test metrics
                test_mae = transformer_model.get('test_mae', 1.0)
                model_weights['transformer'] = 1.0 / max(test_mae, 0.001)
        
        # 5. SARIMA model if available
        if STATSMODELS_AVAILABLE and days_ahead <= 14:  # SARIMA is good for shorter forecasts
            try:
                # Prepare time series data
                price_series = market_df['price'].copy() if 'price' in market_df.columns else market_df['close'].copy()
                
                # Ensure index is datetime
                if 'timestamp' in market_df.columns:
                    price_series.index = pd.to_datetime(market_df['timestamp'])
                elif not isinstance(price_series.index, pd.DatetimeIndex):
                    price_series.index = pd.date_range(end=datetime.now(), periods=len(price_series), freq='D')
                
                # Fit SARIMA model
                sarima_model = SARIMAX(
                    price_series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7) if len(price_series) >= 14 else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                
                # Save in ensemble models
                ensemble_models['sarima'] = {
                    'model': sarima_model,
                    'last_price': price_series.iloc[-1],
                    'last_date': price_series.index[-1]
                }
                
                # Add fixed weight for SARIMA
                model_weights['sarima'] = 0.6  # Default weight for SARIMA
                
                logger.info(f"Added SARIMA model to ensemble for {symbol}")
                
            except Exception as e:
                logger.warning(f"Error training SARIMA model: {e}")
        
        # Normalize weights to sum to 1.0
        if model_weights:
            total_weight = sum(model_weights.values())
            for key in model_weights:
                model_weights[key] /= total_weight
        
        # Create ensemble model info
        ensemble_info = {
            'symbol': symbol,
            'forecast_horizon': days_ahead,
            'models': ensemble_models,
            'weights': model_weights,
            'last_price': market_df['price'].iloc[-1] if 'price' in market_df.columns else market_df['close'].iloc[-1],
            'last_date': market_df.index[-1] if isinstance(market_df.index, pd.DatetimeIndex) else 
                       pd.to_datetime(market_df['timestamp'].iloc[-1]) if 'timestamp' in market_df.columns else None,
            'training_data_points': len(market_df),
            'trained_at': datetime.now().isoformat()
        }
        
        # Save to disk
        ensemble_path = os.path.join(self.model_dir, f"{symbol}_ensemble.joblib")
        joblib.dump(ensemble_info, ensemble_path)
        
        # Store in memory
        self.models[f"{symbol}_ensemble"] = ensemble_info
        
        logger.info(f"Ensemble model trained for {symbol} with {len(model_weights)} components")
        logger.info(f"Model weights: {model_weights}")
        
        return ensemble_info
    
    def forecast(self, symbol, market_df, horizon='short', days_ahead=7, 
                sentiment_df=None, onchain_df=None, retrain=False):
        """
        Generate forecasts with confidence intervals.
        
        Args:
            symbol: Trading symbol
            market_df: DataFrame with market data
            horizon: 'short', 'long', 'lstm', 'transformer', or 'ensemble'
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
                if horizon in ['lstm', 'transformer', 'ensemble']:
                    model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}.joblib")
                
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
            elif horizon == 'long':
                model_info = self.train_long_term_model(symbol, market_df, sentiment_df, onchain_df)
            elif horizon == 'lstm':
                model_info = self.train_lstm_model(symbol, market_df, sentiment_df, onchain_df, 
                                                 forecast_horizon=days_ahead)
            elif horizon == 'transformer':
                model_info = self.train_transformer_model(symbol, market_df, sentiment_df, onchain_df,
                                                        forecast_horizon=days_ahead)
            elif horizon == 'ensemble':
                model_info = self.train_ensemble_model(symbol, market_df, sentiment_df, onchain_df,
                                                     days_ahead=days_ahead)
            else:
                logger.error(f"Unknown forecast horizon: {horizon}")
                return None
                
            if model_info is None:
                logger.error(f"Failed to train {horizon}-term model for {symbol}")
                return None
        else:
            model_info = self.models[model_key]
        
        # Generate forecast based on horizon
        if horizon == 'short':
            return self._generate_short_term_forecast(symbol, model_info, market_df, sentiment_df, days_ahead)
        elif horizon == 'long':
            return self._generate_long_term_forecast(symbol, model_info, market_df, sentiment_df, days_ahead)
        elif horizon == 'lstm':
            return self._generate_dl_forecast(symbol, model_info, market_df, days_ahead)
        elif horizon == 'transformer':
            return self._generate_dl_forecast(symbol, model_info, market_df, days_ahead)
        elif horizon == 'ensemble':
            return self._generate_ensemble_forecast(symbol, model_info, market_df, sentiment_df, onchain_df, days_ahead)
        else:
            logger.error(f"Unknown forecast horizon: {horizon}")
            return None
    
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
        gb_model = model_info.get('gradient_boosting')
        arima_model = model_info.get('arima')
        scaler = model_info.get('scaler')
        features = model_info.get('features')
        last_price = model_info.get('last_price')
        last_date = model_info.get('last_date')
        ensemble_weights = model_info.get('ensemble_weights', {})
        
        if not isinstance(last_date, datetime):
            last_date = pd.to_datetime(last_date)
        
        if rf_model is None or scaler is None or features is None:
            logger.error("Missing required model components for forecasting")
            return None
        
        # Check if we have gradient boosting model
        has_gb = gb_model is not None
        
        # Get weights for ensemble
        rf_weight = ensemble_weights.get('rf_weight', 0.6)
        gb_weight = ensemble_weights.get('gb_weight', 0.4) if has_gb else 0
        
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
            
            # 2. Generate Gradient Boosting prediction if available
            gb_prediction = None
            gb_std = None
            
            if has_gb:
                gb_prediction = gb_model.predict(scaled_features)[0]
                # For GB, use fixed percentage of prediction as std
                gb_std = gb_prediction * 0.03  # Assume 3% standard deviation
            
            # 3. Generate ARIMA prediction if available
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
            
            # 4. Ensemble predictions with weighted average
            prediction = 0
            weights_sum = 0
            
            # Add RF prediction with weight
            if rf_prediction is not None:
                prediction += rf_weight * rf_prediction
                weights_sum += rf_weight
            
            # Add GB prediction with weight if available
            if gb_prediction is not None:
                prediction += gb_weight * gb_prediction
                weights_sum += gb_weight
            
            # Add ARIMA prediction with fixed weight if available
            arima_weight = 0.2  # Fixed weight for ARIMA
            if arima_prediction is not None:
                prediction += arima_weight * arima_prediction
                weights_sum += arima_weight
            
            # Normalize prediction by weights sum
            if weights_sum > 0:
                prediction /= weights_sum
            else:
                # Fallback - just use RF prediction
                prediction = rf_prediction if rf_prediction is not None else latest_price
            
            # 5. Calculate combined uncertainty
            # Weighted average of component variances
            variance = 0
            
            if rf_prediction is not None and rf_std is not None:
                variance += (rf_weight / weights_sum)**2 * rf_std**2 if weights_sum > 0 else rf_std**2
                
            if gb_prediction is not None and gb_std is not None:
                variance += (gb_weight / weights_sum)**2 * gb_std**2 if weights_sum > 0 else gb_std**2
                
            if arima_prediction is not None and arima_std is not None:
                variance += (arima_weight / weights_sum)**2 * arima_std**2 if weights_sum > 0 else arima_std**2
            
            pred_std = np.sqrt(variance)
            
            # Add uncertainty based on forecast horizon (further out = more uncertain)
            horizon_uncertainty = 0.01 * (day_idx + 1)  # 1% per day ahead
            pred_std = np.sqrt(pred_std**2 + (prediction * horizon_uncertainty)**2)
            
            # 6. Calculate confidence intervals (95%)
            lower_bound = max(0, prediction - 1.96 * pred_std)
            upper_bound = prediction + 1.96 * pred_std
            
            # 7. Calculate percent change
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
            # This is a simple approach - in a real forecaster we'd update all features
            for feature in ['price', 'ma_7', 'ma_14', 'ema_12', 'ema_26']:
                if feature in features:
                    idx = features.index(feature)
                    if feature == 'price':
                        feature_array[0, idx] = prediction
                    elif feature.startswith('ma_'):
                        # Simple update for moving averages
                        window = int(feature.split('_')[1])
                        if window > 0:
                            feature_array[0, idx] = (feature_array[0, idx] * (window - 1) + prediction) / window
                    elif feature.startswith('ema_'):
                        # Simple update for exponential moving averages
                        span = int(feature.split('_')[1])
                        alpha = 2 / (span + 1)
                        feature_array[0, idx] = feature_array[0, idx] * (1 - alpha) + prediction * alpha
            
            # Update price_roc_1 if it exists
            if 'price_roc_1' in features and day_idx > 0:
                idx = features.index('price_roc_1')
                prev_price = predictions[day_idx-1]['forecast_price']
                feature_array[0, idx] = (prediction - prev_price) / prev_price if prev_price > 0 else 0
                
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
            
            rf_model = model_dict['rf_model']
            gb_model = model_dict['gb_model']
            scaler = model_dict['scaler']
            model_features = model_dict['features']
            weights = model_dict.get('weights', {'rf_weight': 0.6, 'gb_weight': 0.4})
            
            # Prepare feature array
            feature_values = [latest_data[feature] if feature in latest_data else 0 for feature in model_features]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            scaled_features = scaler.transform(feature_array)
            
            # Generate RF prediction with uncertainty
            if hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0:
                # Use individual trees for prediction distribution
                tree_predictions = []
                n_trees = min(100, len(rf_model.estimators_))
                
                for tree in rf_model.estimators_[:n_trees]:
                    tree_predictions.append(tree.predict(scaled_features)[0])
                
                rf_prediction = np.mean(tree_predictions)
                rf_std = np.std(tree_predictions)
            else:
                rf_prediction = rf_model.predict(scaled_features)[0]
                rf_std = last_price * (0.02 * forecast_day)  # Increasing uncertainty with time
                
            # Generate GB prediction
            gb_prediction = gb_model.predict(scaled_features)[0]
            gb_std = last_price * (0.03 * forecast_day)  # Gradient boosting typically has higher variance
            
            # Weight the predictions
            rf_weight = weights['rf_weight']
            gb_weight = weights['gb_weight']
            
            # Combine predictions
            prediction = (rf_prediction * rf_weight) + (gb_prediction * gb_weight)
            
            # Calculate uncertainty as weighted combination
            variance = (rf_weight**2 * rf_std**2) + (gb_weight**2 * gb_std**2)
            pred_std = np.sqrt(variance)
            
            # Add additional uncertainty for time horizon
            horizon_uncertainty = 0.005 * forecast_day  # 0.5% per day
            pred_std = np.sqrt(pred_std**2 + (prediction * horizon_uncertainty)**2)
            
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
    
    def _generate_dl_forecast(self, symbol, model_info, market_df, days_ahead=7):
        """
        Generate forecast using a deep learning model (LSTM or Transformer).
        
        Args:
            symbol: Trading symbol
            model_info: Model information dictionary
            market_df: Market data
            days_ahead: Number of days to forecast
            
        Returns:
            DataFrame with forecast
        """
        if not self.dl_available:
            logger.error("TensorFlow not available for deep learning forecast")
            return None
        
        # Extract model components
        model = model_info.get('model')
        model_type = model_info.get('model_type')
        scaler = model_info.get('scaler')
        features = model_info.get('features')
        lookback = model_info.get('lookback', 30)
        forecast_horizon = model_info.get('forecast_horizon', 1)
        last_price = model_info.get('last_price')
        last_date = model_info.get('last_date')
        
        if model is None or scaler is None:
            logger.error(f"Missing model components for {model_type} forecast")
            return None
        
        if not isinstance(last_date, datetime):
            last_date = pd.to_datetime(last_date)
        
        # Prepare features
        df = self._prepare_features(market_df)
        if df is None or len(df) < lookback:
            logger.error(f"Insufficient data for {model_type} forecast")
            return None
        
        try:
            # Create sequence input
            # Get the last lookback points
            sequence_data = df.iloc[-lookback:].copy()
            
            # Only use features that were used in training
            if features:
                sequence_features = [f for f in features if f in sequence_data.columns]
                sequence_data = sequence_data[sequence_features]
            else:
                # Use all numeric columns
                sequence_data = sequence_data.select_dtypes(include=[np.number])
            
            # Scale the data
            sequence_scaled = scaler.transform(sequence_data)
            
            # Reshape for model input (add batch dimension)
            sequence_scaled = np.expand_dims(sequence_scaled, axis=0)
            
            # Check if the model returns multiple steps or single step
            if forecast_horizon > 1 and model_type == 'lstm':
                # Model is designed for multi-step forecast
                predictions = model.predict(sequence_scaled)[0]
                # predictions is now a sequence of forecast_horizon values
                
                # Make sure we have enough predictions
                if len(predictions) < days_ahead:
                    # Need to forecast more than the model was trained for
                    # We can either generate iteratively or pad with the last prediction
                    last_pred = predictions[-1]
                    pad_length = days_ahead - len(predictions)
                    predictions = np.append(predictions, [last_pred] * pad_length)
            else:
                # Need to generate predictions iteratively
                predictions = []
                current_sequence = sequence_scaled.copy()
                
                # For each day ahead, generate a prediction and update the sequence
                for _ in range(days_ahead):
                    # Get prediction for next step
                    next_pred = model.predict(current_sequence)[0]
                    
                    # For multi-output models, take the first value
                    if isinstance(next_pred, np.ndarray) and len(next_pred.shape) > 0:
                        next_value = next_pred[0]
                    else:
                        next_value = next_pred
                    
                    predictions.append(next_value)
                    
                    # Update sequence for next prediction by rolling the window
                    # and adding the new prediction
                    rolled_sequence = np.roll(current_sequence[0], -1, axis=0)
                    
                    # Find the index of the price column in the features
                    price_idx = 0  # Default to first column
                    if 'price' in features:
                        price_idx = features.index('price')
                    
                    # Replace the last value with the prediction
                    rolled_sequence[-1, price_idx] = next_value
                    
                    # Update current sequence
                    current_sequence = np.expand_dims(rolled_sequence, axis=0)
            
            # Convert predictions back to original scale
            # Create a dummy array with zeros except for the target value
            dummy = np.zeros((len(predictions), len(sequence_data.columns)))
            
            # Find price column index
            price_idx = 0
            if 'price' in sequence_data.columns:
                price_idx = list(sequence_data.columns).index('price')
            
            # Set the price column to predictions
            dummy[:, price_idx] = predictions
            
            # Inverse transform
            unscaled_predictions = scaler.inverse_transform(dummy)[:, price_idx]
            
            # Create forecast DataFrame
            future_dates = [last_date + timedelta(days=i+1) for i in range(len(unscaled_predictions))]
            
            # Calculate percent changes
            first_price = market_df['price'].iloc[-1] if 'price' in market_df.columns else market_df['close'].iloc[-1]
            
            # Create result DataFrame
            result = pd.DataFrame({
                'date': future_dates,
                'forecast_price': unscaled_predictions,
                'change_pct': [((price / first_price) - 1) * 100 for price in unscaled_predictions]
            })
            
            # Add confidence intervals
            # DL models don't naturally provide confidence intervals, so we'll create simple ones
            uncertainty_factor = 0.02  # Base uncertainty (2%)
            for i, row in result.iterrows():
                day = i + 1
                # Increase uncertainty with forecast horizon
                uncertainty = uncertainty_factor * day
                std_dev = row['forecast_price'] * uncertainty
                
                result.loc[i, 'std_dev'] = std_dev
                result.loc[i, 'lower_bound'] = max(0, row['forecast_price'] - 1.96 * std_dev)
                result.loc[i, 'upper_bound'] = row['forecast_price'] + 1.96 * std_dev
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating {model_type} forecast: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_ensemble_forecast(self, symbol, model_info, market_df, sentiment_df=None, 
                                 onchain_df=None, days_ahead=7):
        """
        Generate forecast using an ensemble of multiple models.
        
        Args:
            symbol: Trading symbol
            model_info: Ensemble model information
            market_df: Market data
            sentiment_df: Sentiment data
            onchain_df: On-chain metrics
            days_ahead: Days to forecast
            
        Returns:
            DataFrame with forecast
        """
        # Extract ensemble components
        models = model_info.get('models', {})
        weights = model_info.get('weights', {})
        
        if not models or not weights:
            logger.error("Missing models or weights for ensemble forecast")
            return None
        
        # Generate individual model forecasts
        forecasts = {}
        
        # 1. Short-term model forecast
        if 'short_term' in models:
            short_term_model = models['short_term']
            short_forecast = self._generate_short_term_forecast(
                symbol, short_term_model, market_df, sentiment_df, days_ahead
            )
            if short_forecast is not None:
                forecasts['short_term'] = short_forecast
        
        # 2. Long-term model forecast
        if 'long_term' in models:
            long_term_model = models['long_term']
            long_forecast = self._generate_long_term_forecast(
                symbol, long_term_model, market_df, sentiment_df, days_ahead
            )
            if long_forecast is not None:
                forecasts['long_term'] = long_forecast
        
        # 3. LSTM model forecast
        if 'lstm' in models:
            lstm_model = models['lstm']
            lstm_forecast = self._generate_dl_forecast(
                symbol, lstm_model, market_df, days_ahead
            )
            if lstm_forecast is not None:
                forecasts['lstm'] = lstm_forecast
        
        # 4. Transformer model forecast
        if 'transformer' in models:
            transformer_model = models['transformer']
            transformer_forecast = self._generate_dl_forecast(
                symbol, transformer_model, market_df, days_ahead
            )
            if transformer_forecast is not None:
                forecasts['transformer'] = transformer_forecast
                
        # 5. SARIMA model forecast
        if 'sarima' in models:
            sarima_model = models['sarima']['model']
            sarima_last_date = models['sarima']['last_date']
            
            try:
                # Generate forecast
                sarima_forecast = sarima_model.get_forecast(steps=days_ahead)
                sarima_predicted = sarima_forecast.predicted_mean
                sarima_conf_int = sarima_forecast.conf_int(alpha=0.05)
                
                # Create dates
                if isinstance(sarima_last_date, pd.Timestamp):
                    sarima_dates = [sarima_last_date + pd.Timedelta(days=i+1) for i in range(days_ahead)]
                else:
                    sarima_dates = [pd.Timestamp.now() + pd.Timedelta(days=i+1) for i in range(days_ahead)]
                
                # Create DataFrame
                sarima_df = pd.DataFrame({
                    'date': sarima_dates,
                    'forecast_price': sarima_predicted.values,
                    'lower_bound': sarima_conf_int.iloc[:, 0].values,
                    'upper_bound': sarima_conf_int.iloc[:, 1].values,
                    'std_dev': (sarima_conf_int.iloc[:, 1].values - sarima_conf_int.iloc[:, 0].values) / 3.92
                })
                
                # Calculate percent change
                first_price = market_df['price'].iloc[-1] if 'price' in market_df.columns else market_df['close'].iloc[-1]
                sarima_df['change_pct'] = ((sarima_df['forecast_price'] / first_price) - 1) * 100
                
                forecasts['sarima'] = sarima_df
                
            except Exception as e:
                logger.warning(f"Error generating SARIMA forecast: {e}")
        
        # Check if we have any valid forecasts
        if not forecasts:
            logger.error("No valid forecasts generated for ensemble")
            return None
        
        # Create ensemble forecast by weighted averaging
        # First, create a common set of dates
        all_dates = set()
        for forecast in forecasts.values():
            all_dates.update(pd.to_datetime(forecast['date']))
        
        all_dates = sorted(all_dates)
        future_dates = all_dates[:days_ahead]
        
        # Initialize result DataFrame
        ensemble_forecast = pd.DataFrame({
            'date': future_dates,
            'forecast_price': [0.0] * len(future_dates),
            'lower_bound': [0.0] * len(future_dates),
            'upper_bound': [0.0] * len(future_dates),
            'std_dev': [0.0] * len(future_dates),
            'change_pct': [0.0] * len(future_dates),
            'weight_sum': [0.0] * len(future_dates),
            'model_count': [0] * len(future_dates)
        })
        
        # Process each forecast
        for model_name, forecast_df in forecasts.items():
            # Skip if no matching weights
            weight_keys = [k for k in weights.keys() if k.startswith(model_name)]
            if not weight_keys:
                logger.warning(f"No weights found for {model_name}")
                continue
                
            # Get model weight - if multiple weights for same model type, use average
            model_weight = sum(weights[k] for k in weight_keys) / len(weight_keys)
            
            # Convert forecast dates to datetime for safer comparison
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Find matching rows in both DataFrames
            for i, row in ensemble_forecast.iterrows():
                date = row['date']
                # Find matching date in forecast_df
                match = forecast_df[forecast_df['date'] == date]
                
                if not match.empty:
                    # Add weighted contribution to ensemble
                    ensemble_forecast.loc[i, 'forecast_price'] += match['forecast_price'].iloc[0] * model_weight
                    ensemble_forecast.loc[i, 'weight_sum'] += model_weight
                    ensemble_forecast.loc[i, 'model_count'] += 1
                    
                    # Add uncertainty information if available
                    if 'lower_bound' in match.columns and 'upper_bound' in match.columns:
                        ensemble_forecast.loc[i, 'lower_bound'] += match['lower_bound'].iloc[0] * model_weight
                        ensemble_forecast.loc[i, 'upper_bound'] += match['upper_bound'].iloc[0] * model_weight
                    
                    if 'std_dev' in match.columns:
                        # Weighted variance (summing squares of std_dev)
                        ensemble_forecast.loc[i, 'std_dev'] += (match['std_dev'].iloc[0] ** 2) * (model_weight ** 2)
        
        # Finalize predictions by normalizing by weight sum
        for i, row in ensemble_forecast.iterrows():
            weight_sum = row['weight_sum']
            if weight_sum > 0:
                ensemble_forecast.loc[i, 'forecast_price'] /= weight_sum
                ensemble_forecast.loc[i, 'lower_bound'] /= weight_sum
                ensemble_forecast.loc[i, 'upper_bound'] /= weight_sum
                
                # Convert variance to std_dev
                ensemble_forecast.loc[i, 'std_dev'] = np.sqrt(ensemble_forecast.loc[i, 'std_dev'])
        
        # Calculate percent change from last known price
        last_price = market_df['price'].iloc[-1] if 'price' in market_df.columns else market_df['close'].iloc[-1]
        ensemble_forecast['change_pct'] = ((ensemble_forecast['forecast_price'] / last_price) - 1) * 100
        
        # Clean up intermediate columns
        ensemble_forecast = ensemble_forecast.drop(['weight_sum', 'model_count'], axis=1)
        
        return ensemble_forecast
    
    def plot_forecast(self, symbol, forecast_df, market_df=None, output_path=None, 
                     show_anomalies=True, confidence_interval=True):
        """
        Plot price forecasts with historical data and confidence intervals.
        
        Args:
            symbol: Trading symbol
            forecast_df: DataFrame with forecast
            market_df: DataFrame with historical data
            output_path: Path to save the plot
            show_anomalies: Whether to highlight anomalies
            confidence_interval: Whether to show confidence intervals
            
        Returns:
            Path to saved plot or None
        """
        if forecast_df is None or len(forecast_df) == 0:
            logger.error("No forecast data to plot")
            return None
        
        try:
            # Create figure with improved aesthetics
            plt.figure(figsize=(12, 6))
            plt.style.use('seaborn-v0_8-whitegrid')
            
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
                    plt.plot(x, y, 'b-', linewidth=2, label='Historical Price')
                    
                    # Plot anomalies if requested
                    if show_anomalies and 'is_anomaly' in market_df.columns:
                        anomalies = market_df[market_df['is_anomaly']]
                        if not anomalies.empty:
                            if 'timestamp' in anomalies.columns:
                                anomaly_x = pd.to_datetime(anomalies['timestamp'])
                            elif isinstance(anomalies.index, pd.DatetimeIndex):
                                anomaly_x = anomalies.index
                            else:
                                anomaly_x = pd.to_datetime(anomalies.index)
                                
                            anomaly_y = anomalies['price'] if 'price' in anomalies.columns else anomalies['close']
                            plt.scatter(anomaly_x, anomaly_y, color='red', s=50, zorder=10, label='Anomalies')
            
            # Plot forecast
            x_forecast = pd.to_datetime(forecast_df['date'])
            y_forecast = forecast_df['forecast_price']
            
            plt.plot(x_forecast, y_forecast, 'r-', linewidth=2, label='Forecast')
            
            # Plot confidence intervals
            if confidence_interval and 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                plt.fill_between(
                    x_forecast,
                    forecast_df['lower_bound'],
                    forecast_df['upper_bound'],
                    color='red',
                    alpha=0.2,
                    label='95% Confidence Interval'
                )
            
            # Format plot
            plt.title(f'{symbol} Price Forecast', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            from matplotlib.ticker import FuncFormatter
            plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.2f}'))
            
            # Add annotations for key forecast points
            annotation_indices = []
            if len(forecast_df) <= 5:
                # If few points, label all
                annotation_indices = list(range(len(forecast_df)))
            else:
                # Otherwise label first, last, and some in between
                annotation_indices = [0, len(forecast_df)//2, -1]
            
            for i in annotation_indices:
                row = forecast_df.iloc[i]
                plt.annotate(
                    f"${row['forecast_price']:.2f}\n({row['change_pct']:+.1f}%)",
                    (pd.to_datetime(row['date']), row['forecast_price']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
            
            # Tight layout
            plt.tight_layout()
            
            # Save plot if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {output_path}")
            else:
                # Save to default location
                os.makedirs('plots', exist_ok=True)
                filename = f"plots/{symbol}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {filename}")
                output_path = filename
            
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def evaluate_model(self, symbol, market_df, horizon='short'):
        """
        Evaluate model performance on historical data.
        
        Args:
            symbol: Trading symbol
            market_df: Historical market data
            horizon: 'short', 'long', 'lstm', 'transformer', or 'ensemble'
            
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
                if horizon in ['lstm', 'transformer', 'ensemble']:
                    model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}.joblib")
                
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
            # Evaluate Random Forest and Gradient Boosting models
            rf_model = model_info.get('random_forest')
            gb_model = model_info.get('gradient_boosting', None)
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
            rf_pred = rf_model.predict(X_test_scaled)
            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rf_r2 = r2_score(y_test, rf_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
            
            metrics = {
                'rf_mae': float(rf_mae),
                'rf_rmse': float(rf_rmse),
                'rf_r2': float(rf_r2),
                'rf_mape': float(rf_mape)
            }
            
            # Evaluate GB model if available
            if gb_model is not None:
                gb_pred = gb_model.predict(X_test_scaled)
                gb_mae = mean_absolute_error(y_test, gb_pred)
                gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
                gb_r2 = r2_score(y_test, gb_pred)
                gb_mape = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100
                
                metrics.update({
                    'gb_mae': float(gb_mae),
                    'gb_rmse': float(gb_rmse),
                    'gb_r2': float(gb_r2),
                    'gb_mape': float(gb_mape)
                })
                
                # Evaluate ensemble if weights are available
                ensemble_weights = model_info.get('ensemble_weights', {})
                if ensemble_weights:
                    rf_weight = ensemble_weights.get('rf_weight', 0.5)
                    gb_weight = ensemble_weights.get('gb_weight', 0.5)
                    
                    # Generate ensemble predictions
                    ensemble_pred = (rf_pred * rf_weight) + (gb_pred * gb_weight)
                    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
                    
                    metrics.update({
                        'ensemble_mae': float(ensemble_mae),
                        'ensemble_rmse': float(ensemble_rmse),
                        'ensemble_r2': float(ensemble_r2),
                        'ensemble_mape': float(ensemble_mape)
                    })
            
            results['metrics'] = metrics
            
            logger.info(f"Evaluation metrics for {symbol} short-term model:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
                
        elif horizon == 'long':
            # Evaluate Prophet or RF long-term models
            if 'prophet_model' in model_info:
                # Evaluate Prophet model
                prophet_model = model_info.get('prophet_model')
                
                # Get historical forecasts from model
                if 'forecast' in model_info:
                    forecast = model_info['forecast']
                else:
                    # Generate new forecast
                    forecast = prophet_model.predict(prophet_model.history)
                
                # Extract actual and predicted values
                actuals = []
                preds = []
                
                # Match dates between forecast and actual data
                for idx, row in test_df.iterrows():
                    date = idx
                    
                    # Find matching date in forecast
                    forecast_match = forecast[forecast['ds'] == date]
                    
                    if not forecast_match.empty:
                        actuals.append(row['price'])
                        preds.append(forecast_match['yhat'].iloc[0])
                
                if actuals and preds:
                    prophet_mae = mean_absolute_error(actuals, preds)
                    prophet_rmse = np.sqrt(mean_squared_error(actuals, preds))
                    prophet_mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100
                    
                    results['metrics'] = {
                        'prophet_mae': float(prophet_mae),
                        'prophet_rmse': float(prophet_rmse),
                        'prophet_mape': float(prophet_mape)
                    }
                    
                    logger.info(f"Evaluation metrics for {symbol} Prophet model:")
                    logger.info(f"  MAE: {prophet_mae:.4f}, RMSE: {prophet_rmse:.4f}, MAPE: {prophet_mape:.2f}%")
                    
            elif 'models' in model_info:
                # Evaluate RF long-term models
                horizon_models = model_info.get('models', {})
                metrics = {}
                
                for horizon_days, model_dict in horizon_models.items():
                    # Get target column
                    target_col = f'target_{horizon_days}d'
                    
                    if target_col not in test_df.columns:
                        logger.warning(f"Target column {target_col} not found in test data")
                        continue
                    
                    # Extract valid rows
                    valid_idx = test_df[target_col].dropna().index
                    if len(valid_idx) < 5:
                        logger.warning(f"Insufficient test data for {horizon_days}-day evaluation")
                        continue
                    
                    # Get features and target
                    features = model_dict['features']
                    X_test = test_df.loc[valid_idx, features].values
                    y_test = test_df.loc[valid_idx, target_col].values
                    
                    # Scale features
                    scaler = model_dict['scaler']
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Get RF predictions
                    rf_model = model_dict['rf_model']
                    rf_pred = rf_model.predict(X_test_scaled)
                    rf_mae = mean_absolute_error(y_test, rf_pred)
                    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
                    rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
                    
                    # Get GB predictions if available
                    if 'gb_model' in model_dict:
                        gb_model = model_dict['gb_model']
                        gb_pred = gb_model.predict(X_test_scaled)
                        gb_mae = mean_absolute_error(y_test, gb_pred)
                        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
                        gb_mape = np.mean(np.abs((y_test - gb_pred) / y_test)) * 100
                        
                        # Get ensemble weights
                        weights = model_dict.get('weights', {'rf_weight': 0.5, 'gb_weight': 0.5})
                        rf_weight = weights.get('rf_weight', 0.5)
                        gb_weight = weights.get('gb_weight', 0.5)
                        
                        # Generate ensemble predictions
                        ensemble_pred = (rf_pred * rf_weight) + (gb_pred * gb_weight)
                        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                        ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
                        
                        metrics[f'{horizon_days}d'] = {
                            'rf_mae': float(rf_mae),
                            'rf_rmse': float(rf_rmse),
                            'rf_mape': float(rf_mape),
                            'gb_mae': float(gb_mae),
                            'gb_rmse': float(gb_rmse),
                            'gb_mape': float(gb_mape),
                            'ensemble_mae': float(ensemble_mae),
                            'ensemble_rmse': float(ensemble_rmse),
                            'ensemble_mape': float(ensemble_mape)
                        }
                    else:
                        metrics[f'{horizon_days}d'] = {
                            'rf_mae': float(rf_mae),
                            'rf_rmse': float(rf_rmse),
                            'rf_mape': float(rf_mape)
                        }
                    
                    logger.info(f"Evaluation metrics for {symbol} {horizon_days}-day forecast:")
                    logger.info(f"  RF - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}, MAPE: {rf_mape:.2f}%")
                
                results['metrics'] = metrics
        
        elif horizon in ['lstm', 'transformer']:
            # Evaluate deep learning models
            if not self.dl_available:
                logger.error("TensorFlow not available for DL model evaluation")
                return None
                
            model = model_info.get('model')
            scaler = model_info.get('scaler')
            features = model_info.get('features')
            lookback = model_info.get('lookback', 30)
            forecast_horizon = model_info.get('forecast_horizon', 1)
            
            if model is None or scaler is None:
                logger.error(f"Missing required components for {horizon} evaluation")
                return None
                
            # Prepare sequence data
            try:
                # Get the last lookback points for each test point
                test_sequences = []
                test_targets = []
                
                for i in range(len(test_df)):
                    if i + lookback < len(test_df):
                        # Create sequence
                        seq_data = test_df.iloc[i:i+lookback]
                        
                        # Only use features that were used in training
                        if features:
                            seq_features = [f for f in features if f in seq_data.columns]
                            seq_data = seq_data[seq_features]
                        else:
                            # Use all numeric columns
                            seq_data = seq_data.select_dtypes(include=[np.number])
                        
                        # Scale the data
                        seq_scaled = scaler.transform(seq_data)
                        
                        # Get target
                        if forecast_horizon == 1:
                            target = test_df.iloc[i+lookback]['price']
                            test_sequences.append(seq_scaled)
                            test_targets.append(target)
                        elif i + lookback + forecast_horizon <= len(test_df):
                            targets = test_df.iloc[i+lookback:i+lookback+forecast_horizon]['price'].values
                            test_sequences.append(seq_scaled)
                            test_targets.append(targets)
                
                # Convert to arrays
                X_test = np.array(test_sequences)
                y_test = np.array(test_targets)
                
                # Generate predictions
                y_pred = model.predict(X_test)
                
                # For multi-step output, flatten both arrays for comparison
                if len(y_test.shape) > 1:
                    y_test_flat = y_test.flatten()
                    y_pred_flat = y_pred.flatten()
                else:
                    y_test_flat = y_test
                    y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
                
                # Calculate metrics
                dl_mae = mean_absolute_error(y_test_flat, y_pred_flat)
                dl_rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
                dl_mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100
                
                results['metrics'] = {
                    f'{horizon}_mae': float(dl_mae),
                    f'{horizon}_rmse': float(dl_rmse),
                    f'{horizon}_mape': float(dl_mape)
                }
                
                logger.info(f"Evaluation metrics for {symbol} {horizon} model:")
                logger.info(f"  MAE: {dl_mae:.4f}, RMSE: {dl_rmse:.4f}, MAPE: {dl_mape:.2f}%")
                
            except Exception as e:
                logger.error(f"Error evaluating {horizon} model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results['metrics'] = {f'{horizon}_error': str(e)}
                
        elif horizon == 'ensemble':
            # Evaluate ensemble model
            ensemble_models = model_info.get('models', {})
            weights = model_info.get('weights', {})
            
            if not ensemble_models:
                logger.error("No models found in ensemble")
                return None
                
            # Evaluate each component model and the ensemble
            component_metrics = {}
            ensemble_preds = {}
            
            for model_type, model_data in ensemble_models.items():
                if model_type == 'short_term':
                    # Evaluate short-term model
                    short_metrics = self.evaluate_model(symbol, market_df, 'short')
                    if short_metrics and 'metrics' in short_metrics:
                        component_metrics['short_term'] = short_metrics['metrics']
                
                elif model_type == 'long_term':
                    # Evaluate long-term model
                    long_metrics = self.evaluate_model(symbol, market_df, 'long')
                    if long_metrics and 'metrics' in long_metrics:
                        component_metrics['long_term'] = long_metrics['metrics']
                
                elif model_type in ['lstm', 'transformer']:
                    # Evaluate deep learning model
                    dl_metrics = self.evaluate_model(symbol, market_df, model_type)
                    if dl_metrics and 'metrics' in dl_metrics:
                        component_metrics[model_type] = dl_metrics['metrics']
            
            # Calculate overall ensemble metrics
            if component_metrics:
                results['component_metrics'] = component_metrics
                
                # Calculate weighted average of metrics
                ensemble_metrics = {}
                metric_keys = ['mae', 'rmse', 'mape']
                
                for metric in metric_keys:
                    weighted_sum = 0
                    weight_sum = 0
                    
                    for model_type, metrics in component_metrics.items():
                        # Find matching weight keys
                        matching_weights = [w for w_key, w in weights.items() if w_key.startswith(model_type)]
                        
                        if matching_weights:
                            model_weight = sum(matching_weights) / len(matching_weights)
                            
                            # Find metrics for this model type
                            model_metrics = [v for k, v in metrics.items() if metric in k.lower()]
                            
                            if model_metrics:
                                # Average of metrics if multiple
                                avg_metric = sum(model_metrics) / len(model_metrics)
                                weighted_sum += avg_metric * model_weight
                                weight_sum += model_weight
                    
                    if weight_sum > 0:
                        ensemble_metrics[f'ensemble_{metric}'] = weighted_sum / weight_sum
                
                results['metrics'] = ensemble_metrics
                
                logger.info(f"Ensemble evaluation metrics for {symbol}:")
                for metric, value in ensemble_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                
            else:
                logger.warning("No component metrics available for ensemble evaluation")
        
        # Save evaluation results
        eval_path = os.path.join(self.model_dir, f"{symbol}_{horizon}_evaluation.joblib")
        joblib.dump(results, eval_path)
        
        return results
    
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
            if isinstance(df.index, pd.DatetimeIndex):
                lookback_start = df.index[-1] - pd.Timedelta(days=lookback_days)
                df = df[df.index >= lookback_start]
            else:
                # Use last N rows
                df = df.iloc[-min(lookback_days, len(df)):]
        
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
        
        # Add risk assessment
        anomalies['risk_level'] = anomalies.apply(
            lambda row: 'high' if row['anomaly_score'] > 0.8 else 
                        'medium' if row['anomaly_score'] > 0.5 else 'low',
            axis=1
        )
        
        # Select relevant columns
        result_cols = ['price', 'anomaly_score', 'anomaly_type', 'description', 'risk_level']
        additional_cols = ['volume_roc_1', 'price_roc_1', 'rsi_14', 'volatility']
        
        # Add any available additional columns
        for col in additional_cols:
            if col in anomalies.columns:
                result_cols.append(col)
        
        result_cols = [col for col in result_cols if col in anomalies.columns]
        
        # Sort by anomaly score
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        return anomalies[result_cols]
    
    def _get_anomaly_description(self, row, df):
        """Generate descriptive text for an anomaly"""
        anomaly_type = row.get('anomaly_type')
        
        if anomaly_type == 'price_movement':
            # Calculate percent change
            price_change = row.get('price_roc_1', 0) * 100
            if price_change > 5:
                return f"Unusual price surge of {price_change:.2f}%"
            elif price_change < -5:
                return f"Unusual price drop of {abs(price_change):.2f}%"
            else:
                return "Abnormal price pattern"
                
        elif anomaly_type == 'volume_spike':
            # Calculate volume change
            volume_change = row.get('volume_roc_1', 0) * 100
            return f"Volume spike of {abs(volume_change):.2f}%"
            
        elif anomaly_type == 'high_volatility':
            # Calculate volatility
            volatility = row.get('volatility', 0) * 100
            return f"Abnormally high volatility of {volatility:.2f}%"
            
        else:
            # Generic description
            rsi = row.get('rsi_14', 0)
            if rsi > 70:
                return f"Overbought conditions (RSI: {rsi:.1f})"
            elif rsi < 30:
                return f"Oversold conditions (RSI: {rsi:.1f})"
            else:
                return "Unusual market behavior detected"
    
    def generate_market_insights(self, symbol, market_df, forecast_df=None, 
                               sentiment_df=None, days=30):
        """
        Generate market insights for a cryptocurrency.
        
        Args:
            symbol: Trading symbol
            market_df: Market data
            forecast_df: Forecast data (optional)
            sentiment_df: Sentiment data (optional)
            days: Number of days to analyze
            
        Returns:
            Dict with market insights
        """
        logger.info(f"Generating market insights for {symbol}")
        
        if market_df is None or len(market_df) == 0:
            logger.error("No market data provided for insights")
            return {"error": "No market data provided"}
        
        # Prepare features
        df = self._prepare_features(market_df)
        if df is None:
            logger.error("Failed to prepare features for insights")
            return {"error": "Failed to prepare features"}
        
        # Limit to recent days
        if days and isinstance(df.index, pd.DatetimeIndex):
            lookback_start = df.index[-1] - pd.Timedelta(days=days)
            recent_df = df[df.index >= lookback_start].copy()
        else:
            # Use last N rows
            recent_df = df.iloc[-min(days, len(df)):].copy()
        
        # Calculate current price and recent change
        current_price = recent_df['price'].iloc[-1]
        day_change = recent_df['price'].pct_change().iloc[-1] * 100
        
        # Calculate recent highs and lows
        recent_high = recent_df['price'].max()
        recent_low = recent_df['price'].min()
        
        # Calculate average volume
        if 'volume' in recent_df.columns:
            avg_volume = recent_df['volume'].mean()
        elif 'volume_24h' in recent_df.columns:
            avg_volume = recent_df['volume_24h'].mean()
        else:
            avg_volume = None
        
        # Calculate technical indicators
        last_rsi = recent_df['rsi_14'].iloc[-1] if 'rsi_14' in recent_df.columns else None
        
        # Determine market trend
        price_sma_20 = recent_df['ma_20'].iloc[-1] if 'ma_20' in recent_df.columns else None
        
        if price_sma_20 is not None:
            trend = "bullish" if current_price > price_sma_20 else "bearish"
        else:
            # Use simple 7-day trend
            week_change = ((current_price / recent_df['price'].iloc[0]) - 1) * 100
            trend = "bullish" if week_change > 0 else "bearish"
        
        # Calculate volatility
        volatility = recent_df['price'].pct_change().std() * 100
        
        # Calculate support and resistance levels using recent price action
        support_levels = []
        resistance_levels = []
        
        # Simple method looking at recent lows and highs
        price_data = recent_df['price'].values
        for i in range(1, len(price_data)-1):
            # Check for support (local minimum)
            if price_data[i] < price_data[i-1] and price_data[i] < price_data[i+1]:
                support_levels.append(price_data[i])
            
            # Check for resistance (local maximum)
            if price_data[i] > price_data[i-1] and price_data[i] > price_data[i+1]:
                resistance_levels.append(price_data[i])
        
        # Keep only the closest levels to current price
        if support_levels:
            # Sort by distance from current price (only levels below current price)
            valid_supports = [s for s in support_levels if s < current_price]
            if valid_supports:
                support_levels = sorted(valid_supports, key=lambda x: abs(current_price - x))[:3]
        
        if resistance_levels:
            # Sort by distance from current price (only levels above current price)
            valid_resistances = [r for r in resistance_levels if r > current_price]
            if valid_resistances:
                resistance_levels = sorted(valid_resistances, key=lambda x: abs(current_price - x))[:3]
        
        # Get forecast insights if available
        forecast_insights = {}
        if forecast_df is not None and len(forecast_df) > 0:
            forecast_insights = {
                'next_7d_forecast': forecast_df['forecast_price'].iloc[min(6, len(forecast_df)-1)] if len(forecast_df) > 6 else None,
                'next_30d_forecast': forecast_df['forecast_price'].iloc[-1] if len(forecast_df) >= 30 else None,
                'forecast_direction': "up" if forecast_df['change_pct'].iloc[-1] > 0 else "down",
                'forecast_change_pct': forecast_df['change_pct'].iloc[-1],
                'max_upside': forecast_df['upper_bound'].max() if 'upper_bound' in forecast_df.columns else None,
                'max_downside': forecast_df['lower_bound'].min() if 'lower_bound' in forecast_df.columns else None
            }
        
        # Get sentiment insights if available
        sentiment_insights = {}
        if sentiment_df is not None and len(sentiment_df) > 0:
            avg_sentiment = sentiment_df['sentiment_score'].mean() if 'sentiment_score' in sentiment_df.columns else 0.5
            
            sentiment_insights = {
                'avg_sentiment': avg_sentiment,
                'sentiment_trend': "improving" if sentiment_df['sentiment_score'].iloc[-1] > avg_sentiment else "declining",
                'sentiment_vs_price_correlation': sentiment_df['sentiment_score'].corr(recent_df['price']) if len(sentiment_df) == len(recent_df) else None
            }
        
        # Detect recent anomalies
        df_with_anomalies = self._detect_anomalies(recent_df, symbol)
        recent_anomalies = df_with_anomalies[df_with_anomalies['is_anomaly']]
        
        # Compile results
        insights = {
            'symbol': symbol,
            'current_price': current_price,
            'day_change_pct': day_change,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'volatility': volatility,
            'rsi': last_rsi,
            'avg_volume': avg_volume,
            'trend': trend,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'anomalies_count': len(recent_anomalies),
            'generated_at': datetime.now().isoformat()
        }
        
        # Add forecast insights if available
        if forecast_insights:
            insights['forecast'] = forecast_insights
        
        # Add sentiment insights if available
        if sentiment_insights:
            insights['sentiment'] = sentiment_insights
        
        # Generate a market summary
        summary = []
        
        # Add price status
        summary.append(f"{symbol} is currently trading at ${current_price:.2f}, {day_change:+.2f}% over the last day.")
        
        # Add trend analysis
        if trend == "bullish":
            summary.append(f"The market is in a bullish trend, trading above the 20-day moving average.")
        else:
            summary.append(f"The market is in a bearish trend, trading below the 20-day moving average.")
        
        # Add RSI analysis
        if last_rsi is not None:
            if last_rsi > 70:
                summary.append(f"RSI of {last_rsi:.1f} indicates overbought conditions.")
            elif last_rsi < 30:
                summary.append(f"RSI of {last_rsi:.1f} indicates oversold conditions.")
            else:
                summary.append(f"RSI of {last_rsi:.1f} indicates neutral momentum.")
        
        # Add volatility analysis
        if volatility > 5:
            summary.append(f"Volatility is high at {volatility:.2f}%, suggesting elevated risk.")
        elif volatility < 2:
            summary.append(f"Volatility is low at {volatility:.2f}%, suggesting potential range-bound trading.")
        else:
            summary.append(f"Volatility is moderate at {volatility:.2f}%.")
        
        # Add support/resistance analysis
        if support_levels and support_levels[0] < current_price:
            summary.append(f"Nearest support level at ${support_levels[0]:.2f}.")
        
        if resistance_levels and resistance_levels[0] > current_price:
            summary.append(f"Nearest resistance level at ${resistance_levels[0]:.2f}.")
        
        # Add forecast insights
        if 'forecast' in insights:
            forecast_direction = insights['forecast']['forecast_direction']
            forecast_change = insights['forecast']['forecast_change_pct']
            
            if forecast_direction == "up":
                summary.append(f"The forecast is bullish with a projected {forecast_change:.2f}% increase.")
            else:
                summary.append(f"The forecast is bearish with a projected {abs(forecast_change):.2f}% decrease.")
        
        # Add sentiment insights
        if 'sentiment' in insights:
            avg_sentiment = insights['sentiment']['avg_sentiment']
            sentiment_trend = insights['sentiment']['sentiment_trend']
            
            if avg_sentiment > 0.6:
                sentiment_text = "strongly positive"
            elif avg_sentiment > 0.55:
                sentiment_text = "positive"
            elif avg_sentiment < 0.4:
                sentiment_text = "negative"
            elif avg_sentiment < 0.45:
                sentiment_text = "somewhat negative"
            else:
                sentiment_text = "neutral"
                
            summary.append(f"Market sentiment is {sentiment_text} and {sentiment_trend}.")
        
        # Add anomaly warnings
        if insights['anomalies_count'] > 0:
            summary.append(f"Detected {insights['anomalies_count']} market anomalies in the recent data, indicating unusual behavior.")
        
        insights['market_summary'] = " ".join(summary)
        
        return insights
    
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
                'long_term': {'status': 'skipped', 'error': None},
                'ensemble': {'status': 'skipped', 'error': None}
            }
            
            # Add deep learning models if available
            if self.dl_available:
                symbol_results['lstm'] = {'status': 'skipped', 'error': None}
                symbol_results['transformer'] = {'status': 'skipped', 'error': None}
            
            try:
                # Check if models need retraining
                short_model_key = f"{symbol}_short"
                long_model_key = f"{symbol}_long"
                ensemble_model_key = f"{symbol}_ensemble"
                lstm_model_key = f"{symbol}_lstm"
                transformer_model_key = f"{symbol}_transformer"
                
                short_needs_training = force
                long_needs_training = force
                ensemble_needs_training = force
                lstm_needs_training = force and self.dl_available
                transformer_needs_training = force and self.dl_available
                
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
                    
                    # Check ensemble model
                    if ensemble_model_key in self.models:
                        trained_at = self.models[ensemble_model_key].get('trained_at')
                        if trained_at:
                            trained_date = pd.to_datetime(trained_at)
                            days_since_training = (datetime.now() - trained_date).days
                            ensemble_needs_training = days_since_training >= 14  # Retrain biweekly
                    else:
                        # Check if model exists on disk
                        model_path = os.path.join(self.model_dir, f"{symbol}_ensemble.joblib")
                        if not os.path.exists(model_path):
                            ensemble_needs_training = True
                    
                    # Check LSTM model
                    if self.dl_available and lstm_model_key in self.models:
                        trained_at = self.models[lstm_model_key].get('trained_at')
                        if trained_at:
                            trained_date = pd.to_datetime(trained_at)
                            days_since_training = (datetime.now() - trained_date).days
                            lstm_needs_training = days_since_training >= 14  # Retrain biweekly
                    elif self.dl_available:
                        # Check if model exists on disk
                        model_path = os.path.join(self.model_dir, f"{symbol}_lstm.h5")
                        if not os.path.exists(model_path):
                            lstm_needs_training = True
                    
                    # Check Transformer model
                    if self.dl_available and transformer_model_key in self.models:
                        trained_at = self.models[transformer_model_key].get('trained_at')
                        if trained_at:
                            trained_date = pd.to_datetime(trained_at)
                            days_since_training = (datetime.now() - trained_date).days
                            transformer_needs_training = days_since_training >= 14  # Retrain biweekly
                    elif self.dl_available:
                        # Check if model exists on disk
                        model_path = os.path.join(self.model_dir, f"{symbol}_transformer.h5")
                        if not os.path.exists(model_path):
                            transformer_needs_training = True
                
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
                    logger.info(f"Skipping short-term model training for {symbol}")
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
                    logger.info(f"Skipping long-term model training for {symbol}")
                    results['skipped'] += 1
                
                # Train LSTM model if needed and available
                if lstm_needs_training and self.dl_available:
                    logger.info(f"Training LSTM model for {symbol}")
                    try:
                        result = self.train_lstm_model(symbol, market_df, sentiment_df)
                        if result and not (isinstance(result, dict) and 'error' in result):
                            symbol_results['lstm']['status'] = 'success'
                            results['success'] += 1
                        else:
                            symbol_results['lstm']['status'] = 'failed'
                            symbol_results['lstm']['error'] = result.get('error', 'Training failed') if isinstance(result, dict) else 'Training failed'
                            results['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error training LSTM model for {symbol}: {e}")
                        symbol_results['lstm']['status'] = 'failed'
                        symbol_results['lstm']['error'] = str(e)
                        results['failed'] += 1
                elif self.dl_available:
                    logger.info(f"Skipping LSTM model training for {symbol}")
                    results['skipped'] += 1
                
                # Train Transformer model if needed and available
                if transformer_needs_training and self.dl_available:
                    logger.info(f"Training Transformer model for {symbol}")
                    try:
                        result = self.train_transformer_model(symbol, market_df, sentiment_df)
                        if result and not (isinstance(result, dict) and 'error' in result):
                            symbol_results['transformer']['status'] = 'success'
                            results['success'] += 1
                        else:
                            symbol_results['transformer']['status'] = 'failed'
                            symbol_results['transformer']['error'] = result.get('error', 'Training failed') if isinstance(result, dict) else 'Training failed'
                            results['failed'] += 1
                    except Exception as e:
                        logger.error(f"Error training Transformer model for {symbol}: {e}")
                        symbol_results['transformer']['status'] = 'failed'
                        symbol_results['transformer']['error'] = str(e)
                        results['failed'] += 1
                elif self.dl_available:
                    logger.info(f"Skipping Transformer model training for {symbol}")
                    results['skipped'] += 1
                
                # Train ensemble model if needed and at least one component model is available
                if ensemble_needs_training:
                    # Only train ensemble if at least one component model was successfully trained
                    components_trained = (
                        symbol_results['short_term']['status'] == 'success' or
                        symbol_results['long_term']['status'] == 'success'
                    )
                    
                    if self.dl_available:
                        components_trained = components_trained or (
                            symbol_results['lstm']['status'] == 'success' or
                            symbol_results['transformer']['status'] == 'success'
                        )
                    
                    if components_trained:
                        logger.info(f"Training ensemble model for {symbol}")
                        try:
                            result = self.train_ensemble_model(symbol, market_df, sentiment_df)
                            if result:
                                symbol_results['ensemble']['status'] = 'success'
                                results['success'] += 1
                            else:
                                symbol_results['ensemble']['status'] = 'failed'
                                symbol_results['ensemble']['error'] = 'Training failed'
                                results['failed'] += 1
                        except Exception as e:
                            logger.error(f"Error training ensemble model for {symbol}: {e}")
                            symbol_results['ensemble']['status'] = 'failed'
                            symbol_results['ensemble']['error'] = str(e)
                            results['failed'] += 1
                    else:
                        logger.info(f"Skipping ensemble model training for {symbol} due to missing components")
                        symbol_results['ensemble']['status'] = 'skipped'
                        symbol_results['ensemble']['error'] = 'No component models available'
                        results['skipped'] += 1
                else:
                    logger.info(f"Skipping ensemble model training for {symbol}")
                    results['skipped'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                for model_type in symbol_results:
                    if symbol_results[model_type]['status'] == 'skipped':
                        symbol_results[model_type]['status'] = 'failed'
                        symbol_results[model_type]['error'] = str(e)
                results['failed'] += len(symbol_results)
            
            # Store results for this symbol
            results['details'][symbol] = symbol_results
        
        logger.info(f"Auto-retraining completed: {results['success']} successful, {results['failed']} failed, {results['skipped']} skipped")
        
        return results

# Example usage when imported as a library
if __name__ == "__main__":
    # Example standalone usage
    forecaster = ChronosForecaster()
    
    print("\nChronosForecaster - Next-Gen Crypto Forecasting")
    print("===============================================")
    print(f"Available models: {', '.join(forecaster.available_models)}")
    print(f"Deep learning available: {forecaster.dl_available}")
    print(f"SARIMAX available: {STATSMODELS_AVAILABLE}")
    print(f"Prophet available: {PROPHET_AVAILABLE}")
    print("\nRun your forecasting script or import this module to use the forecaster.")