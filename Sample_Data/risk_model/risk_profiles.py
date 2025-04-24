# File path: Sample_Data/risk_model/new_risk.py

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import xgboost as xgb
from typing import Dict, List, Any, Optional, Tuple
import weaviate # Make sure this is imported
import re # Import regex for extracting days

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Weaviate client
from Sample_Data.vector_store.weaviate_client import get_weaviate_client
from weaviate.classes.query import Filter, Sort
# Import aggregation classes specifically (needed for _get_available_symbols)
from weaviate.classes.aggregate import GroupByAggregate


class CryptoRiskScoreGenerator:
    # --- Keep existing methods (__init__, connect, close, _get_date_range, etc.) ---
    def __init__(self):
        """Initialize the risk score generator"""
        self.client = None
        self.xgb_model = None
        self.feature_columns = None

    def connect(self):
        """Connect to Weaviate"""
        if self.client is None:
            try: # Add try-except for robustness
                self.client = get_weaviate_client()
                logger.info("Weaviate client obtained.")
            except Exception as e:
                logger.error(f"Failed to get Weaviate client: {e}")
                self.client = None
                return False
        if self.client:
             try:
                 is_live = self.client.is_live()
                 if is_live:
                     # logger.info("Weaviate connection is live.") # Optional: reduce verbosity
                     return True
                 else:
                     logger.warning("Weaviate client is not live.")
                     self.client = None
                     return False
             except Exception as e:
                 logger.error(f"Error checking Weaviate liveness: {e}")
                 self.client = None
                 return False
        return False

    def close(self):
        """Close the Weaviate connection"""
        if self.client:
            try:
                self.client.close()
                # logger.info("Weaviate connection closed.") # Optional: reduce verbosity
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")
            finally:
                 self.client = None

    def _get_date_range(self, days: int = 3) -> Tuple[str, str]:
        """
        Get the date range for data extraction (using UTC).

        Args:
            days (int): Number of days to look back

        Returns:
            Tuple[str, str]: Start and end dates in ISO format (with Z)
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        # Ensure 'Z' is added for Weaviate DATE type compatibility
        return start_date.isoformat().replace("+00:00", "Z"), end_date.isoformat().replace("+00:00", "Z")


    def _get_date_range_objects(self, days: int = 3) -> Tuple[datetime, datetime]:
       # ... (keep this method as is) ...
        end_date_dt = datetime.now(timezone.utc)
        start_date_dt = end_date_dt - timedelta(days=days)
        return start_date_dt, end_date_dt

    def _get_market_metrics(self, symbol: str, start_date_dt: datetime, end_date_dt: datetime) -> pd.DataFrame:
        # ... (keep the latest robust version of this method) ...
        # Includes symbol variations, TEXT date parsing, python filtering
        if not self.client: logger.error("Cannot get market metrics: Weaviate client not connected."); return pd.DataFrame()
        logger.warning("Fetching recent market metrics and filtering dates in Python due to TEXT date field in Weaviate. This might be inefficient.")
        try:
            collection = self.client.collections.get("MarketMetrics")
            # ... (symbol variation filters) ...
            symbol_filter = Filter.by_property("symbol").equal(symbol)
            symbol_usdt = f"{symbol}USDT"; symbol_usdt_filter = Filter.by_property("symbol").equal(symbol_usdt)
            symbol_usd = f"{symbol}USD"; symbol_usd_filter = Filter.by_property("symbol").equal(symbol_usd)
            symbol_lower = symbol.lower(); symbol_lower_filter = Filter.by_property("symbol").equal(symbol_lower)
            symbol_lower_usdt_filter = Filter.by_property("symbol").equal(symbol_usdt.lower())
            symbol_lower_usd_filter = Filter.by_property("symbol").equal(symbol_usd.lower())
            combined_symbol_filter = (symbol_filter | symbol_usdt_filter | symbol_usd_filter | symbol_lower_filter | symbol_lower_usdt_filter | symbol_lower_usd_filter)
            fetch_limit = 2000
            logger.info(f"Querying latest {fetch_limit} MarketMetrics for '{symbol}' (or variations) (sorting by TEXT timestamp)")
            response = collection.query.fetch_objects(filters=combined_symbol_filter, sort=Sort.by_property("timestamp", ascending=False), limit=fetch_limit)
            data = [obj.properties for obj in response.objects];
            if not data: logger.warning(f"No market metrics found for {symbol}"); return pd.DataFrame()
            df = pd.DataFrame(data)
            if 'timestamp' not in df.columns: logger.error(f"The 'timestamp' column is missing..."); return pd.DataFrame()
            df_processed = df.copy(); original_timestamps = df_processed['timestamp'].copy()
            df_processed['timestamp_dt'] = pd.to_datetime(df_processed['timestamp'], errors='coerce', utc=True)
            failed_mask = df_processed['timestamp_dt'].isna()
            if failed_mask.any(): # ... (log failures) ...
                 logger.warning(f"Failed to parse {failed_mask.sum()} timestamp string(s) in MarketMetrics for {symbol}.")
                 failed_indices = df_processed[failed_mask].index
                 for i, idx in enumerate(failed_indices[:5]): logger.warning(f"  Example failed timestamp string: '{original_timestamps.loc[idx]}'")
            df_processed.dropna(subset=['timestamp_dt'], inplace=True)
            if df_processed.empty: logger.warning(f"No valid/parseable timestamps..."); return pd.DataFrame()
            logger.info(f"Filtering {len(df_processed)} records (with valid timestamps) between {start_date_dt} and {end_date_dt} in Python")
            df_filtered = df_processed[(df_processed['timestamp_dt'] >= start_date_dt) & (df_processed['timestamp_dt'] <= end_date_dt)].copy()
            if df_filtered.empty: logger.warning(f"No market metrics for {symbol} within the date range..."); return pd.DataFrame()
            df_filtered.set_index('timestamp_dt', inplace=True); df_filtered.sort_index(inplace=True)
            logger.info(f"Retained {len(df_filtered)} market metrics for {symbol} after Python date filtering.")
            return df_filtered
        except Exception as e: logger.error(f"Error getting market metrics: {e}", exc_info=True); return pd.DataFrame()


    def _get_sentiment_data(self, symbol: str, start_date_dt: datetime, end_date_dt: datetime) -> pd.DataFrame:
        # ... (keep the latest robust version of this method) ...
        # Includes symbol variations, row-by-row multi-strategy date parsing, python filtering
        if not self.client: logger.error("Cannot get sentiment data: Weaviate client not connected."); return pd.DataFrame()
        logger.warning("Fetching recent sentiment data and filtering dates in Python due to TEXT date field in Weaviate. This might be inefficient.")
        try:
            collection = self.client.collections.get("CryptoNewsSentiment")
            # ... (symbol variation filters) ...
            symbol_lower = symbol.lower()
            symbol_content_filter = Filter.by_property("content").like(f"*{symbol}*") | Filter.by_property("content").like(f"*{symbol_lower}*")
            symbol_title_filter = Filter.by_property("title").like(f"*{symbol}*") | Filter.by_property("title").like(f"*{symbol_lower}*")
            symbol_assets_filter = Filter.by_property("related_assets").contains_any([symbol_lower, symbol.upper()])
            symbol_filter = symbol_content_filter | symbol_title_filter | symbol_assets_filter
            fetch_limit = 1000
            logger.info(f"Querying latest {fetch_limit} CryptoNewsSentiment related to {symbol} (sorting by TEXT date)")
            response = collection.query.fetch_objects(filters=symbol_filter, sort=Sort.by_property("date", ascending=False), limit=fetch_limit)
            data = [obj.properties for obj in response.objects]
            if not data: logger.warning(f"No sentiment data found for {symbol}"); return pd.DataFrame()
            df = pd.DataFrame(data)
            if 'date' not in df.columns: logger.error(f"The 'date' column is missing..."); return pd.DataFrame()
            df_processed = df.copy(); original_dates = df_processed['date'].copy()
            parsed_datetimes = []; parse_errors = 0; error_examples = []
            for date_str in df_processed['date']: # Row-by-row parsing
                parsed_dt = pd.NaT
                if isinstance(date_str, str):
                    cleaned_str = date_str.strip()
                    if not cleaned_str: parse_errors += 1; error_examples.append(f"'{date_str}' (Empty string)"); parsed_datetimes.append(parsed_dt); continue
                    try: # Attempt 1
                        dt_attempt = pd.to_datetime(cleaned_str, errors='raise', utc=True, infer_datetime_format=True); parsed_dt = dt_attempt
                    except (ValueError, TypeError):
                        try: # Attempt 2
                            if len(cleaned_str) > 18 and cleaned_str[10] == 'T' and not (cleaned_str.endswith('Z') or '+' in cleaned_str[19:] or '-' in cleaned_str[19:]):
                                dt_attempt = pd.to_datetime(cleaned_str + 'Z', errors='raise', utc=True); parsed_dt = dt_attempt
                            else: raise ValueError("String already had Z or didn't match pattern")
                        except (ValueError, TypeError):
                            try: # Attempt 3
                                dt_attempt = pd.to_datetime(cleaned_str[:10], format='%Y-%m-%d', errors='raise'); parsed_dt = dt_attempt.tz_localize('UTC')
                            except (ValueError, TypeError, IndexError): parse_errors += 1; 
                            if len(error_examples) < 5: error_examples.append(f"'{date_str}'")
                            else: parse_errors += 1; 
                            if len(error_examples) < 5: error_examples.append(f"'{date_str}' (Non-string type: {type(date_str).__name__})")
                parsed_datetimes.append(parsed_dt)
            df_processed['date_dt'] = parsed_datetimes
            if parse_errors > 0: # Log errors
                logger.warning(f"Failed to parse {parse_errors} date string(s) using multiple strategies in CryptoNewsSentiment for {symbol}.")
                for example in error_examples: logger.warning(f"  Example original failed date string: {example}")
            df_processed.dropna(subset=['date_dt'], inplace=True)
            if df_processed.empty: logger.warning(f"No valid/parseable dates found..."); return pd.DataFrame()
            logger.info(f"Filtering {len(df_processed)} records (with valid dates) between {start_date_dt} and {end_date_dt} in Python")
            df_filtered = df_processed[(df_processed['date_dt'] >= start_date_dt) & (df_processed['date_dt'] <= end_date_dt)].copy()
            if df_filtered.empty: logger.warning(f"No sentiment data for {symbol} within the date range..."); return pd.DataFrame()
            df_filtered.set_index('date_dt', inplace=True); df_filtered.sort_index(inplace=True)
            logger.info(f"Retained {len(df_filtered)} sentiment articles related to {symbol} after Python date filtering.")
            return df_filtered
        except Exception as e: logger.error(f"Error getting sentiment data: {e}", exc_info=True); return pd.DataFrame()


    def _engineer_market_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        # ... (keep the latest robust version of this method) ...
        if market_df.empty: return pd.DataFrame()
        numeric_cols = ['price', 'market_cap', 'volume_24h', 'price_change_24h']
        for col in numeric_cols:
            if col in market_df.columns: market_df.loc[:, col] = pd.to_numeric(market_df[col], errors='coerce')
        if market_df.index.empty: logger.warning("Market DataFrame has an empty index..."); return pd.DataFrame()
        features = pd.DataFrame(index=[market_df.index.max()])
        try:
            if 'price' in market_df.columns and market_df['price'].notna().any(): #... (price feature logic) ...
                 price_series = market_df['price']; features['price_last'] = price_series.iloc[-1]; features['price_mean'] = price_series.mean()
                 features['price_std'] = price_series.std(); features['price_max'] = price_series.max(); features['price_min'] = price_series.min()
                 price_range_val = features['price_max'].iloc[0] - features['price_min'].iloc[0]; features['price_range'] = price_range_val
                 price_mean_val = features['price_mean'].iloc[0]; features['price_range_pct'] = price_range_val / price_mean_val if price_mean_val != 0 else 0.0

            if 'volume_24h' in market_df.columns and market_df['volume_24h'].notna().any(): #... (volume feature logic) ...
                 volume_series = market_df['volume_24h']; features['volume_last'] = volume_series.iloc[-1]; features['volume_mean'] = volume_series.mean(); features['volume_std'] = volume_series.std()
                 if len(volume_series) > 1: vol_change = volume_series.pct_change().dropna(); features['volume_change'] = vol_change.mean() if not vol_change.empty else 0.0
                 else: features['volume_change'] = 0.0

            if 'market_cap' in market_df.columns and market_df['market_cap'].notna().any(): #... (mcap feature logic) ...
                 mcap_series = market_df['market_cap']; features['mcap_last'] = mcap_series.iloc[-1]; features['mcap_mean'] = mcap_series.mean()

            if 'price_change_24h' in market_df.columns and market_df['price_change_24h'].notna().any(): #... (price change logic) ...
                 change_series = market_df['price_change_24h']; features['price_change_mean'] = change_series.mean(); features['price_change_std'] = change_series.std()

            if 'price' in market_df.columns and len(market_df) > 1: #... (volatility/drawdown logic with safety checks) ...
                 price_series = market_df['price']; valid_price_data = price_series.notna().sum() > 1
                 if valid_price_data:
                      returns = price_series.pct_change().dropna()
                      if not returns.empty:
                           features['volatility'] = returns.std() if len(returns) > 1 else 0.0
                           pos_returns = returns[returns > 0]; neg_returns = returns[returns < 0]
                           if len(neg_returns) > 0: features['pos_neg_ratio'] = len(pos_returns) / len(neg_returns)
                           elif len(pos_returns) > 0: features['pos_neg_ratio'] = 1.0
                           else: features['pos_neg_ratio'] = 0.0
                           if len(returns) >= 1:
                                cumulative_returns = (1 + returns).cumprod()
                                if not cumulative_returns.empty:
                                     max_return = cumulative_returns.expanding().max(); safe_max_return = max_return.replace(0, np.nan)
                                     if not safe_max_return.isna().all():
                                          drawdown = (cumulative_returns / safe_max_return) - 1
                                          if not drawdown.empty and not drawdown.isna().all(): features['max_drawdown'] = drawdown.min()
                                          else: features['max_drawdown'] = 0.0
                                     else: features['max_drawdown'] = 0.0
                                else: features['max_drawdown'] = 0.0
                           else: features['max_drawdown'] = 0.0
                      else: features['volatility'] = 0.0; features['pos_neg_ratio'] = 0.0; features['max_drawdown'] = 0.0
                 else: features['volatility'] = 0.0; features['pos_neg_ratio'] = 0.0; features['max_drawdown'] = 0.0

            features.fillna(0.0, inplace=True)
            return features
        except Exception as e: logger.error(f"Error engineering market features: {e}", exc_info=True); return pd.DataFrame(index=features.index if 'features' in locals() else None)


    def _engineer_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        # ... (keep this method as is) ...
        if sentiment_df.empty: return pd.DataFrame()
        if 'sentiment_score' not in sentiment_df.columns: return pd.DataFrame()
        features = pd.DataFrame(index=[sentiment_df.index.max()])
        try:
            sentiment_df['sentiment_score'] = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce')
            features['sentiment_mean'] = sentiment_df['sentiment_score'].mean()
            features['sentiment_std'] = sentiment_df['sentiment_score'].std()
            features['sentiment_last'] = sentiment_df['sentiment_score'].iloc[-1] if not sentiment_df.empty else 0.5
            if 'sentiment_label' in sentiment_df.columns:
                sentiment_counts = sentiment_df['sentiment_label'].value_counts(normalize=True)
                features['sentiment_positive_ratio'] = sentiment_counts.get('POSITIVE', 0)
                features['sentiment_negative_ratio'] = sentiment_counts.get('NEGATIVE', 0)
                features['sentiment_neutral_ratio'] = sentiment_counts.get('NEUTRAL', 0)
                pos_count = sentiment_counts.get('POSITIVE', 0); neg_count = sentiment_counts.get('NEGATIVE', 0)
                features['pos_neg_sentiment_ratio'] = pos_count / neg_count if neg_count > 0 else 1
            if len(sentiment_df) >= 2:
                sentiment_df_sorted = sentiment_df.sort_index()
                half_idx = len(sentiment_df_sorted) // 2
                first_half = sentiment_df_sorted.iloc[:half_idx]; second_half = sentiment_df_sorted.iloc[half_idx:]
                first_half_avg = first_half['sentiment_score'].mean(); second_half_avg = second_half['sentiment_score'].mean()
                features['sentiment_trend'] = second_half_avg - first_half_avg
            else: features['sentiment_trend'] = 0
            features['news_count'] = len(sentiment_df)
            features.fillna(0.0, inplace=True) # Fill NaNs from calculations
            return features
        except Exception as e: logger.error(f"Error engineering sentiment features: {e}"); return pd.DataFrame()


    def _combine_features(self, market_features: pd.DataFrame, sentiment_features: pd.DataFrame) -> pd.DataFrame:
        # ... (keep this method as is) ...
        if market_features.empty and sentiment_features.empty: return pd.DataFrame()
        elif market_features.empty: return sentiment_features.fillna(0.0) # Ensure fillna
        elif sentiment_features.empty: return market_features.fillna(0.0) # Ensure fillna
        combined = pd.concat([market_features, sentiment_features], axis=1)
        combined.fillna(0.0, inplace=True) # Fill NaNs from concat
        return combined


    def _train_xgboost_model(self, training_data: Optional[pd.DataFrame] = None) -> None: # Made arg optional
        """
        Train or load an XGBoost model. Placeholder for now.
        """
        logger.warning("XGBoost model training/loading is not implemented. Using heuristic prediction.")
        # TODO: Implement loading a pre-trained model or training logic
        self.xgb_model = None # Explicitly None if not loaded/trained
        # Define expected feature columns based on engineering steps FOR HEURISTIC FALLBACK
        # This list should contain ALL possible features your engineering functions might create
        self.feature_columns = [
             'price_last', 'price_mean', 'price_std', 'price_max', 'price_min', 'price_range', 'price_range_pct',
             'volume_last', 'volume_mean', 'volume_std', 'volume_change', # Adjusted name
             'mcap_last', 'mcap_mean',
             'price_change_mean', 'price_change_std', 'volatility', 'pos_neg_ratio', 'max_drawdown',
             'sentiment_mean', 'sentiment_std', 'sentiment_last',
             'sentiment_positive_ratio', 'sentiment_negative_ratio', 'sentiment_neutral_ratio', 'pos_neg_sentiment_ratio',
             'sentiment_trend', 'news_count'
        ]


    def _predict_risk_score(self, features: pd.DataFrame) -> float:
        """
        Predict risk score from features using heuristics.
        """
        if features.empty:
            logger.warning("Features DataFrame is empty, returning default risk score 50.")
            return 50.0

        # Ensure all potential feature columns exist, fill with 0 if missing
        if self.feature_columns:
             for col in self.feature_columns:
                 if col not in features.columns:
                     features[col] = 0.0
        else:
             # Fallback if feature_columns not defined (e.g., if _train_xgboost_model wasn't called)
             logger.warning("Feature columns list not defined. Using only available columns for heuristics.")


        # Use .get(col, pd.Series([0.0])).iloc[0] for safe access to feature values
        # This handles missing columns AND ensures we get a scalar value
        get_val = lambda col: features.get(col, pd.Series([0.0])).iloc[0] if not features.get(col, pd.Series([])).empty else 0.0


        # Initialize score components
        price_risk = 0
        volatility_risk = 0
        sentiment_risk = 0
        volume_risk = 0

        # Price-based risk factors
        price_risk += get_val('price_range_pct') * 30
        price_risk += get_val('price_change_std') * 10
        volatility_risk += get_val('volatility') * 50
        volatility_risk += abs(get_val('max_drawdown')) * 20

        # Sentiment-based risk factors
        sentiment_mean_val = get_val('sentiment_mean')
        sentiment_risk += (1 - sentiment_mean_val if sentiment_mean_val is not None else 0.5) * 15 # Handle potential None if mean calc fails
        sentiment_risk += get_val('sentiment_std') * 10
        sentiment_risk += get_val('sentiment_negative_ratio') * 20
        sentiment_risk += max(0, -get_val('sentiment_trend') * 15)

        # Volume-related risk
        volume_mean_val = get_val('volume_mean')
        volume_std_val = get_val('volume_std')
        if volume_mean_val > 0:
            normalized_vol_std = volume_std_val / volume_mean_val
            volume_risk += normalized_vol_std * 15

        # Calculate total risk score (weighted sum)
        price_weight = 0.35
        volatility_weight = 0.35
        sentiment_weight = 0.2
        volume_weight = 0.1

        total_risk = (
            price_weight * price_risk +
            volatility_weight * volatility_risk +
            sentiment_weight * sentiment_risk +
            volume_weight * volume_risk
        )

        # Scale to 0-100 range
        risk_score = min(100, max(0, total_risk)) # np.clip(total_risk, 0, 100) is cleaner

        # Handle potential NaN result from calculations
        if pd.isna(risk_score):
             logger.warning("Calculated risk score is NaN, returning default 50.")
             return 50.0

        return float(risk_score)


    def _get_risk_category(self, risk_score: Optional[float]) -> str: # Accept None
        """
        Convert numeric risk score to category. Handles None score.
        """
        if risk_score is None or pd.isna(risk_score): # Check for None or NaN
            return "Undetermined"
        if risk_score < 20:
            return "Very Low"
        elif risk_score < 40:
            return "Low"
        elif risk_score < 60:
            return "Moderate"
        elif risk_score < 80:
            return "High"
        else:
            return "Very High"

    # --- ADD THIS NEW METHOD ---
    def _store_risk_profile(self, risk_result: Dict[str, Any]):
        """
        Stores the generated risk profile result in the RiskProfiles collection.

        Args:
            risk_result (Dict): The dictionary returned by generate_risk_score.
        """
        if not self.client or not self.client.is_live():
            logger.error("Cannot store risk profile: Weaviate client not connected or not live.")
            return False # Indicate failure

        collection_name = "RiskProfiles"
        try:
            collection = self.client.collections.get(collection_name)

            # Prepare properties, handling potential None values from calculation errors
            properties = {}
            properties["symbol"] = risk_result.get("symbol", "UNKNOWN")
            properties["analysis_timestamp"] = risk_result.get("analysis_timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

            # Handle score and category (provide defaults if None)
            score = risk_result.get("risk_score")
            properties["risk_score"] = float(score) if score is not None else -1.0 # Use -1 to indicate error/None score
            properties["risk_category"] = risk_result.get("risk_category", "Error" if risk_result.get("error") else "Undetermined")

            # Extract analysis period days
            analysis_period_str = risk_result.get("analysis_period", "0 days")
            days_match = re.search(r'\d+', analysis_period_str)
            properties["analysis_period_days"] = int(days_match.group(0)) if days_match else 0

            # Get data point counts
            data_points = risk_result.get("data_points", {})
            properties["market_data_points"] = data_points.get("market_metrics", 0)
            properties["sentiment_data_points"] = data_points.get("sentiment_articles", 0)

            # Get risk factors (ensure it's a list)
            factors = risk_result.get("risk_factors", [])
            properties["risk_factors"] = factors if isinstance(factors, list) else [str(factors)] # Ensure list

            # Get error message
            properties["calculation_error"] = risk_result.get("error", None) # Store error if present

            # Remove None values before insertion if desired by Weaviate version/setup
            # properties = {k: v for k, v in properties.items() if v is not None}

            # Insert data
            uuid_inserted = collection.data.insert(properties=properties)
            logger.info(f"Successfully stored risk profile for {properties['symbol']} with UUID: {uuid_inserted}")
            return True

        except Exception as e:
            logger.error(f"Error storing risk profile for symbol {risk_result.get('symbol', 'UNKNOWN')} in {collection_name}: {e}", exc_info=True)
            return False
    # --- END OF NEW METHOD ---
    

    def generate_risk_score(self, symbol: str, days: int = 3) -> Dict[str, Any]:
        """
        Generate risk score for a specific cryptocurrency and store the result.
        *** Assumes connection is managed by the caller (e.g., generate_multi_asset_risk_report) ***
        *** or needs explicit connect/close if called standalone. ***
        """
        # Initialize result dict - ensure UTC timestamp with Z
        result_dict = {
            "symbol": symbol.upper(),
            "risk_score": None,
            "risk_category": "Undetermined",
            "analysis_period": f"{days} days",
            "analysis_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "data_points": {"market_metrics": 0, "sentiment_articles": 0},
            "risk_factors": [],
            "error": None
        }

        # Removed the connect() call here - assume caller handles it
        # if not self.connect(): ...

        try:
            symbol = symbol.upper() # Standardize symbol format earlier
            result_dict["symbol"] = symbol # Update result dict

            start_date_dt, end_date_dt = self._get_date_range_objects(days)

            market_df = self._get_market_metrics(symbol, start_date_dt, end_date_dt)
            sentiment_df = self._get_sentiment_data(symbol, start_date_dt, end_date_dt)

            result_dict["data_points"]["market_metrics"] = len(market_df)
            result_dict["data_points"]["sentiment_articles"] = len(sentiment_df)

            if market_df.empty and sentiment_df.empty:
                 logger.warning(f"Insufficient data for risk assessment for {symbol}.")
                 result_dict["error"] = "Insufficient data for risk assessment"
                 self._store_risk_profile(result_dict) # Store result even if insufficient data
                 return result_dict

            self._train_xgboost_model() # Placeholder for model load/train

            market_features = self._engineer_market_features(market_df)
            sentiment_features = self._engineer_sentiment_features(sentiment_df)

            if market_features.empty and sentiment_features.empty and not (market_df.empty and sentiment_df.empty) :
                 # Only log error if data existed but features failed
                 logger.warning(f"Feature engineering failed for {symbol}.")
                 result_dict["error"] = "Feature engineering failed"
                 self._store_risk_profile(result_dict)
                 return result_dict

            combined_features = self._combine_features(market_features, sentiment_features)

            if combined_features.empty and not (market_df.empty and sentiment_df.empty):
                 logger.warning(f"Combined features are empty for {symbol}.")
                 result_dict["error"] = "Feature combination failed"
                 self._store_risk_profile(result_dict)
                 return result_dict

            risk_score = self._predict_risk_score(combined_features)
            risk_category = self._get_risk_category(risk_score)
            risk_factors = self._derive_risk_factors(combined_features, risk_score)

            # Update result dict with calculated values
            result_dict["risk_score"] = round(float(risk_score), 2) if not pd.isna(risk_score) else None
            result_dict["risk_category"] = risk_category if result_dict["risk_score"] is not None else "Undetermined"
            result_dict["risk_factors"] = risk_factors

            # --- Store SUCCESS result ---
            self._store_risk_profile(result_dict)
            # --- End Store ---

            return result_dict

        except Exception as e:
            logger.error(f"Error generating risk score for {symbol}: {e}", exc_info=True)
            result_dict["error"] = f"Calculation error: {str(e)}"
            result_dict["risk_score"] = None
            result_dict["risk_category"] = "Error"
            result_dict["risk_factors"] = ["Calculation error occurred"]
            # --- Store EXCEPTION result ---
            self._store_risk_profile(result_dict)
            # --- End Store ---
            return result_dict
        # --- REMOVED finally: self.close() ---
        # Connection management moved to generate_multi_asset_risk_report or __main__




    def _derive_risk_factors(self, features: pd.DataFrame, risk_score: float) -> List[str]:
       # ... (keep the latest robust version of this method) ...
        risk_factors = []
        try:
            get_val = lambda col: features.get(col, pd.Series([0.0])).iloc[0] if col in features and not features[col].empty else 0.0
            if get_val('volatility') > 0.05: risk_factors.append("High price volatility")
            if get_val('sentiment_negative_ratio') > 0.5: risk_factors.append("High negative sentiment")
            if get_val('sentiment_trend') < -0.1: risk_factors.append("Declining sentiment trend")
            if get_val('max_drawdown') < -0.15: risk_factors.append("Significant price drawdown")
            vol_mean_val = get_val('volume_mean'); vol_std_val = get_val('volume_std')
            if vol_mean_val > 0: vol_ratio = vol_std_val / vol_mean_val;
            if vol_mean_val > 0 and (vol_std_val / vol_mean_val) > 0.5: risk_factors.append("Unstable trading volume") # Combined check
            if get_val('price_change_std') > 5: risk_factors.append("Erratic price changes")
            if not risk_factors: # Add overall only if no specific factors
                if risk_score is not None: # Check score is not None
                    if risk_score > 80: risk_factors.append("Overall risk profile: Very High")
                    elif risk_score > 60: risk_factors.append("Overall risk profile: High")
            elif risk_score is not None: # Add overall high/v.high even if specific factors exist
                if risk_score > 80 and "Overall risk profile: Very High" not in risk_factors: risk_factors.insert(0,"Overall risk profile: Very High")
                elif risk_score > 60 and not any("Overall risk profile" in rf for rf in risk_factors): risk_factors.insert(0,"Overall risk profile: High")
            return risk_factors if risk_factors else []
        except Exception as e: logger.error(f"Error deriving risk factors: {e}", exc_info=True); return ["Unable to determine specific risk factors"]


    def _get_available_symbols(self, collection_name="MarketMetrics", min_recent_records=1) -> List[str]:
         # ... (keep this method as is - requires GroupByAggregate import) ...
        if not self.connect(): logger.error(f"Cannot get available symbols: Weaviate client not connected."); return []
        symbols = set()
        try:
            collection = self.client.collections.get(collection_name); logger.info(f"Querying for unique symbols in collection: {collection_name}")
            # Strategy 1: Get ALL unique symbols
            response = collection.aggregate.over_all(group_by=GroupByAggregate(prop="symbol")) # Use imported class
            if response.groups:
                for group in response.groups:
                    symbol = group.grouped_by.value;
                    if symbol and isinstance(symbol, str): symbols.add(symbol.upper())
            # Strategy 2: (Optional Date Check)
            if min_recent_records > 0 and symbols:
                logger.warning(f"Performing secondary check for recent activity (min {min_recent_records} records) for {len(symbols)} symbols. This might be slow.")
                check_days = 7; start_dt_check, end_dt_check = self._get_date_range_objects(check_days); recent_symbols = set()
                symbols_to_check = list(symbols)[:500]
                if len(symbols) > 500: logger.warning(f"Checking recent activity for only the first 500 symbols out of {len(symbols)}.")
                for sym in symbols_to_check:
                    try:
                        df_check = self._get_market_metrics(sym, start_dt_check, end_dt_check)
                        if len(df_check) >= min_recent_records: recent_symbols.add(sym.upper())
                    except Exception as e_inner: logger.error(f"Error checking recent data for symbol {sym}: {e_inner}"); continue
                logger.info(f"Found {len(recent_symbols)} symbols with at least {min_recent_records} records in the last ~{check_days} days.")
                symbols = recent_symbols
            final_symbol_list = sorted(list(symbols))
            logger.info(f"Returning {len(final_symbol_list)} unique symbols for analysis: {final_symbol_list[:20]}...")
            return final_symbol_list
        except Exception as e: logger.error(f"Error retrieving available symbols from {collection_name}: {e}", exc_info=True); return []


    def generate_multi_asset_risk_report(self, symbols: List[str], days: int = 3) -> Dict[str, Any]:
        """
        Generate risk scores for multiple assets and store each result.
        Manages its own connection lifecycle.
        """
        logger.info(f"Starting multi-asset risk report generation for {len(symbols)} symbols...")
        # Connect at the beginning of the multi-asset process
        if not self.connect():
            logger.error("Failed to connect to Weaviate for multi-asset report.")
            # Return structure indicating failure
            return {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "period_days": days, "assets_analyzed": 0, "successful_analyses": 0,
                "average_risk_score": None, "highest_risk_asset": None, "lowest_risk_asset": None,
                "assets": {}, "error": "Database connection failed"
            }

        results = {}
        highest_risk = {"symbol": None, "score": -1.0} # Use float
        lowest_risk = {"symbol": None, "score": 101.0} # Use float
        valid_scores = []

        for symbol in symbols:
            # generate_risk_score will handle its own try/except and storing
            # It will use the connection established by this method
            logger.info(f"--- Processing symbol: {symbol} ---")
            result = self.generate_risk_score(symbol, days) # generate_risk_score now also stores
            processed_symbol = result.get("symbol", symbol.upper())
            results[processed_symbol] = result # Store result in report dict too

            # Update highest/lowest based on the actual stored key and score
            # Only count as valid if score is calculated and no major error occurred
            if result.get("risk_score") is not None and result.get("error") is None:
                 score = result["risk_score"]
                 valid_scores.append(score)
                 if score > highest_risk["score"]:
                     highest_risk = {"symbol": processed_symbol, "score": score}
                 if score < lowest_risk["score"]:
                     lowest_risk = {"symbol": processed_symbol, "score": score}

        # Close the connection after processing all symbols
        self.close() # Manage connection here
        logger.info("--- Multi-asset risk report generation complete ---")

        avg_risk = sum(valid_scores) / len(valid_scores) if valid_scores else None

        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "period_days": days,
            "assets_requested": len(symbols),
            "assets_analyzed": len(results), # How many were attempted
            "successful_analyses": len(valid_scores), # How many yielded a score without error
            "average_risk_score": round(avg_risk, 2) if avg_risk is not None else None,
            "highest_risk_asset": highest_risk if highest_risk["symbol"] else None,
            "lowest_risk_asset": lowest_risk if lowest_risk["symbol"] else None,
            "assets": results # Contains individual results including potential errors
        }

    # --- Keep _derive_risk_factors, _get_available_symbols, _engineer_..., _predict_..., etc. ---
    def _derive_risk_factors(self, features: pd.DataFrame, risk_score: float) -> List[str]:
       # ... (keep the latest robust version of this method) ...
        risk_factors = []
        try:
            get_val = lambda col: features.get(col, pd.Series([0.0])).iloc[0] if col in features and not features[col].empty else 0.0
            if get_val('volatility') > 0.05: risk_factors.append("High price volatility")
            if get_val('sentiment_negative_ratio') > 0.5: risk_factors.append("High negative sentiment")
            if get_val('sentiment_trend') < -0.1: risk_factors.append("Declining sentiment trend")
            if get_val('max_drawdown') < -0.15: risk_factors.append("Significant price drawdown")
            vol_mean_val = get_val('volume_mean'); vol_std_val = get_val('volume_std')
            if vol_mean_val > 0 and (vol_std_val / vol_mean_val) > 0.5: risk_factors.append("Unstable trading volume") # Combined check
            if get_val('price_change_std') > 5: risk_factors.append("Erratic price changes")
            if not risk_factors:
                if risk_score is not None:
                    if risk_score > 80: risk_factors.append("Overall risk profile: Very High")
                    elif risk_score > 60: risk_factors.append("Overall risk profile: High")
            elif risk_score is not None:
                if risk_score > 80 and "Overall risk profile: Very High" not in risk_factors: risk_factors.insert(0,"Overall risk profile: Very High")
                elif risk_score > 60 and not any("Overall risk profile" in rf for rf in risk_factors): risk_factors.insert(0,"Overall risk profile: High")
            return risk_factors if risk_factors else []
        except Exception as e: logger.error(f"Error deriving risk factors: {e}", exc_info=True); 
        return ["Unable to determine specific risk factors"]


    def _get_available_symbols(self, collection_name="MarketMetrics", min_recent_records=1) -> List[str]:
         # ... (keep this method as is - requires GroupByAggregate import) ...
        if not self.connect(): logger.error(f"Cannot get available symbols: Weaviate client not connected."); return []
        symbols = set()
        try:
            collection = self.client.collections.get(collection_name); logger.info(f"Querying for unique symbols in collection: {collection_name}")
            response = collection.aggregate.over_all(group_by=GroupByAggregate(prop="symbol"))
            if response.groups:
                for group in response.groups:
                    symbol = group.grouped_by.value;
                    if symbol and isinstance(symbol, str): symbols.add(symbol.upper())
            if min_recent_records > 0 and symbols:
                logger.warning(f"Performing secondary check for recent activity (min {min_recent_records} records) for {len(symbols)} symbols. This might be slow.")
                check_days = 7; start_dt_check, end_dt_check = self._get_date_range_objects(check_days); recent_symbols = set()
                symbols_to_check = list(symbols)[:500]
                if len(symbols) > 500: logger.warning(f"Checking recent activity for only the first 500 symbols out of {len(symbols)}.")
                for sym in symbols_to_check:
                    try: 
                        df_check = self._get_market_metrics(sym, start_dt_check, end_dt_check);
                        if len(df_check) >= min_recent_records: recent_symbols.add(sym.upper())
                    except Exception as e_inner: logger.error(f"Error checking recent data for symbol {sym}: {e_inner}"); continue
                logger.info(f"Found {len(recent_symbols)} symbols with at least {min_recent_records} records in the last ~{check_days} days.")
                symbols = recent_symbols
            final_symbol_list = sorted(list(symbols))
            logger.info(f"Returning {len(final_symbol_list)} unique symbols for analysis: {final_symbol_list[:20]}...")
            return final_symbol_list
        except Exception as e: logger.error(f"Error retrieving available symbols from {collection_name}: {e}", exc_info=True); return []
# Example usage
if __name__ == "__main__":
    # --- Ensure necessary imports for this block ---
    import sys
    # --- End imports ---

    risk_generator = CryptoRiskScoreGenerator()

    # --- Define Symbols ---
    # Using a fixed list for clarity, but automatic discovery can be used by uncommenting below
    # print("Attempting to automatically discover symbols...")
    # available_symbols = risk_generator._get_available_symbols(collection_name="MarketMetrics", min_recent_records=1) # Requires connection management if used here
    available_symbols = ["BTC", "ETH", "SOL", "AVAX", "BNB", "ADA", "XRP", "LTC", "LINK", "DOGE", "SUI", "PI"] # Example list
    print(f"Using fixed list of symbols: {available_symbols}")

    if not available_symbols:
        print("No symbols defined to analyze. Exiting.")
        sys.exit(1)

    print(f"\n--- Generating Multi-Asset Risk Report for {len(available_symbols)} Symbols ---")
    # generate_multi_asset_risk_report now handles connect/close internally
    report = risk_generator.generate_multi_asset_risk_report(available_symbols, days=3)

    # --- Print Report Summary ---
    print("\nMulti-Asset Risk Report:")
    print(f"Timestamp: {report.get('timestamp')}")
    print(f"Assets Requested: {report.get('assets_requested')}")
    print(f"Successful Analyses: {report.get('successful_analyses')}")
    print(f"Average Risk Score: {report.get('average_risk_score', 'N/A')}")
    highest = report.get('highest_risk_asset')
    if highest and highest.get('symbol') is not None: print(f"Highest Risk: {highest['symbol']} ({highest.get('score', 'N/A'):.2f})")
    else: print("Highest Risk Asset: N/A (No successful analyses)")
    lowest = report.get('lowest_risk_asset')
    if lowest and lowest.get('symbol') is not None: print(f"Lowest Risk: {lowest['symbol']} ({lowest.get('score', 'N/A'):.2f})")
    else: print("Lowest Risk Asset: N/A (No successful analyses)")


    # --- Print Individual Details ---
    print("\n--- Individual Asset Risk Details ---")
    for symbol, details in report.get('assets', {}).items():
        print(f"\nSymbol: {symbol}")
        if details.get("error"):
            print(f"  Error: {details['error']}")
            print(f"  Data Points (Market/Sentiment): {details.get('data_points', {}).get('market_metrics', 0)} / {details.get('data_points', {}).get('sentiment_articles', 0)}")
        else:
            score = details.get('risk_score', 'N/A')
            category = details.get('risk_category', 'N/A')
            factors = details.get('risk_factors', [])
            print(f"  Risk Score: {score}")
            print(f"  Risk Category: {category}")
            print(f"  Data Points (Market/Sentiment): {details.get('data_points', {}).get('market_metrics', 0)} / {details.get('data_points', {}).get('sentiment_articles', 0)}")
            print(f"  Risk Factors:")
            if factors:
                for factor in factors: print(f"    - {factor}")
            else:
                 if score is None or score == 'N/A': print("    - Analysis incomplete or failed.")
                 else: print("    - None identified.")

    # No need to call risk_generator.close() here, generate_multi_asset_risk_report handles it.
    print("\nScript finished.")