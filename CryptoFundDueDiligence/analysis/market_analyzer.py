"""
Market Analyzer Module

This module analyzes market data for cryptocurrencies mentioned in fund documents.
It retrieves and processes data from MarketMetrics, CryptoTimeSeries, and other collections.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """
    Analyzes market data for cryptocurrencies mentioned in fund documents.
    """
    
    def __init__(self, data_retriever):
        """
        Initialize the market analyzer.
        
        Args:
            data_retriever: Object that retrieves data from collections
        """
        self.retriever = data_retriever
        
    def analyze_market_data(self, crypto_entities: List[str]) -> Dict[str, Any]:
        """
        Analyze market data for specified cryptocurrencies.
        
        Args:
            crypto_entities: List of cryptocurrency names/symbols
            
        Returns:
            Dict with market analysis data
        """
        logger.info(f"Analyzing market data for: {', '.join(crypto_entities)}")
        
        # Convert common names to symbols and normalize
        symbols = self._normalize_crypto_symbols(crypto_entities)
        
        # Get current market data
        current_market_data = self._get_market_data(symbols)
        
        # Get historical performance
        historical_performance = self._get_historical_performance(symbols)
        
        # Get volatility metrics
        volatility_metrics = self._calculate_volatility(symbols)
        
        # Get correlation data
        correlation_data = self._calculate_correlations(symbols)
        
        # Get forecasts if available
        forecast_data = self._get_forecasts(symbols)
        
        # Combine all data
        market_analysis = {
            "current_data": current_market_data,
            "historical_performance": historical_performance,
            "volatility": volatility_metrics,
            "correlations": correlation_data,
            "forecasts": forecast_data,
            "analysis_date": datetime.now().isoformat()
        }
        
        return market_analysis
    
    def _normalize_crypto_symbols(self, crypto_entities: List[str]) -> List[str]:
        """
        Convert common cryptocurrency names to trading symbols.
        
        Args:
            crypto_entities: List of crypto names or symbols
            
        Returns:
            List of normalized trading symbols
        """
        # Map common names to standard symbols
        name_to_symbol = {
            "bitcoin": "BTCUSDT",
            "btc": "BTCUSDT",
            "ethereum": "ETHUSDT",
            "eth": "ETHUSDT",
            "solana": "SOLUSDT",
            "sol": "SOLUSDT",
            "binance": "BNBUSDT",
            "bnb": "BNBUSDT",
            "binance coin": "BNBUSDT",
            "cardano": "ADAUSDT",
            "ada": "ADAUSDT",
            "ripple": "XRPUSDT",
            "xrp": "XRPUSDT",
            "polkadot": "DOTUSDT",
            "dot": "DOTUSDT",
            "dogecoin": "DOGEUSDT",
            "doge": "DOGEUSDT",
            "avalanche": "AVAXUSDT",
            "avax": "AVAXUSDT",
            "polygon": "MATICUSDT",
            "matic": "MATICUSDT"
        }
        
        symbols = []
        for entity in crypto_entities:
            entity_lower = entity.lower()
            if entity_lower in name_to_symbol:
                symbols.append(name_to_symbol[entity_lower])
            else:
                # Try to form a valid symbol
                if not entity.upper().endswith('USDT'):
                    symbols.append(f"{entity.upper()}USDT")
                else:
                    symbols.append(entity.upper())
        
        return list(set(symbols))  # Remove duplicates
    
    def _get_market_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current market data for specified symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict mapping symbols to their market data
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get latest market metrics from collector
                data = self.retriever.get_market_data(symbol, limit=1)
                
                if data and len(data) > 0:
                    # Take the most recent data point
                    market_data[symbol] = data[0]
                    logger.info(f"Retrieved market data for {symbol}")
                else:
                    logger.warning(f"No market data found for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving market data for {symbol}: {e}")
        
        return market_data
    
    def _get_historical_performance(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate historical performance metrics for specified symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict mapping symbols to their performance metrics
        """
        performance_data = {}
        
        # Define time periods to analyze
        periods = {
            "1m": 30,     # 30 days
            "3m": 90,     # 90 days
            "6m": 180,    # 180 days
            "1y": 365,    # 1 year
            "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days  # Year to date
        }
        
        for symbol in symbols:
            try:
                # Get historical time series data (use 1d interval for performance calculation)
                historical_data = self.retriever.get_historical_data(symbol, interval="1d", limit=365)  # Get up to 1 year
                
                if not historical_data or len(historical_data) < 2:
                    logger.warning(f"Insufficient historical data for {symbol}")
                    continue
                
                # Sort data by timestamp
                historical_data.sort(key=lambda x: x.get('timestamp', ''))
                
                # Calculate performance for each period
                performance = {}
                latest_price = float(historical_data[-1].get('close', 0))
                
                for period_name, days in periods.items():
                    # Find the starting price for this period
                    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                    
                    # Find the closest data point
                    start_data_point = None
                    for data_point in historical_data:
                        if data_point.get('timestamp', '') >= cutoff_date:
                            start_data_point = data_point
                            break
                    
                    # If we found a starting point, calculate performance
                    if start_data_point:
                        start_price = float(start_data_point.get('close', 0))
                        if start_price > 0:
                            period_change = (latest_price - start_price) / start_price
                            performance[period_name] = {
                                "start_date": start_data_point.get('timestamp', ''),
                                "start_price": start_price,
                                "end_price": latest_price,
                                "change_pct": period_change * 100,
                                "change_abs": latest_price - start_price
                            }
                
                # Calculate high-level metrics
                if len(historical_data) >= 30:  # At least 30 days of data
                    # Calculate all-time high and low in this dataset
                    prices = [float(data_point.get('close', 0)) for data_point in historical_data]
                    ath = max(prices)
                    atl = min(prices)
                    
                    # Calculate drawdown from ATH
                    drawdown_pct = ((latest_price - ath) / ath) * 100 if ath > 0 else 0
                    
                    # Add to performance data
                    performance["metrics"] = {
                        "all_time_high": ath,
                        "all_time_low": atl,
                        "current_drawdown_pct": drawdown_pct,
                        "days_analyzed": len(historical_data)
                    }
                
                performance_data[symbol] = performance
                logger.info(f"Calculated performance metrics for {symbol}")
                
            except Exception as e:
                logger.error(f"Error calculating performance for {symbol}: {e}")
        
        return performance_data
    
    def _calculate_volatility(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate volatility metrics for specified symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict mapping symbols to their volatility metrics
        """
        volatility_data = {}
        
        for symbol in symbols:
            try:
                # Get historical time series data
                historical_data = self.retriever.get_historical_data(symbol, interval="1d", limit=90)  # Last 90 days
                
                if not historical_data or len(historical_data) < 7:  # Need at least a week of data
                    logger.warning(f"Insufficient historical data for volatility calculation for {symbol}")
                    continue
                
                # Sort data by timestamp
                historical_data.sort(key=lambda x: x.get('timestamp', ''))
                
                # Calculate daily returns
                closing_prices = [float(data_point.get('close', 0)) for data_point in historical_data]
                daily_returns = []
                
                for i in range(1, len(closing_prices)):
                    if closing_prices[i-1] > 0:
                        daily_return = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
                        daily_returns.append(daily_return)
                
                if len(daily_returns) > 1:
                    # Calculate standard deviation of daily returns
                    daily_volatility = statistics.stdev(daily_returns)
                    
                    # Annualize volatility (standard approach: daily vol * sqrt(252))
                    annual_volatility = daily_volatility * (252 ** 0.5)
                    
                    # Calculate 7-day and 30-day rolling volatility if enough data
                    volatility_7d = None
                    volatility_30d = None
                    
                    if len(daily_returns) >= 7:
                        volatility_7d = statistics.stdev(daily_returns[-7:])
                    
                    if len(daily_returns) >= 30:
                        volatility_30d = statistics.stdev(daily_returns[-30:])
                    
                    volatility_data[symbol] = {
                        "daily_volatility": daily_volatility * 100,  # Convert to percentage
                        "annual_volatility": annual_volatility * 100,  # Convert to percentage
                        "volatility_7d": volatility_7d * 100 if volatility_7d else None,  # Convert to percentage
                        "volatility_30d": volatility_30d * 100 if volatility_30d else None,  # Convert to percentage
                        "max_daily_gain": max(daily_returns) * 100,  # Convert to percentage
                        "max_daily_loss": min(daily_returns) * 100,  # Convert to percentage
                        "days_analyzed": len(daily_returns)
                    }
                    
                    logger.info(f"Calculated volatility metrics for {symbol}")
                
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
        
        return volatility_data
    
    def _calculate_correlations(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between the specified symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict mapping symbol pairs to their correlation coefficient
        """
        if len(symbols) < 2:
            return {}  # Need at least 2 symbols for correlation
        
        correlation_data = {}
        
        try:
            # Get price data for all symbols (last 90 days)
            symbol_price_data = {}
            
            for symbol in symbols:
                # Get historical data
                historical_data = self.retriever.get_historical_data(symbol, interval="1d", limit=90)
                
                if not historical_data or len(historical_data) < 30:  # Need reasonable amount of data
                    logger.warning(f"Insufficient historical data for correlation calculation for {symbol}")
                    continue
                
                # Sort and extract closing prices with timestamps
                historical_data.sort(key=lambda x: x.get('timestamp', ''))
                price_data = {data_point.get('timestamp', ''): float(data_point.get('close', 0)) 
                             for data_point in historical_data}
                
                symbol_price_data[symbol] = price_data
            
            # Find common dates across all symbols
            common_dates = set()
            for symbol, price_data in symbol_price_data.items():
                if not common_dates:
                    common_dates = set(price_data.keys())
                else:
                    common_dates = common_dates.intersection(set(price_data.keys()))
            
            # Convert to sorted list
            common_dates = sorted(list(common_dates))
            
            if len(common_dates) < 30:  # Need at least 30 common data points
                logger.warning("Insufficient common data points for correlation calculation")
                return {}
            
            # Calculate correlations between each pair of symbols
            for i, symbol1 in enumerate(symbols):
                if symbol1 not in symbol_price_data:
                    continue
                    
                correlation_data[symbol1] = {}
                
                for j, symbol2 in enumerate(symbols):
                    if j <= i or symbol2 not in symbol_price_data:  # Avoid duplicate pairs and missing data
                        continue
                    
                    # Get price series for both symbols
                    prices1 = [symbol_price_data[symbol1][date] for date in common_dates]
                    prices2 = [symbol_price_data[symbol2][date] for date in common_dates]
                    
                    # Calculate correlation
                    correlation = self._calculate_correlation_coefficient(prices1, prices2)
                    
                    # Store correlation
                    correlation_data[symbol1][symbol2] = correlation
                    
                    # Also store the inverse relationship for easier lookup
                    if symbol2 not in correlation_data:
                        correlation_data[symbol2] = {}
                    correlation_data[symbol2][symbol1] = correlation
            
            logger.info(f"Calculated correlations between {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
        
        return correlation_data
    
    def _calculate_correlation_coefficient(self, series1: List[float], series2: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient between two price series.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        
        n = len(series1)
        
        # Calculate means
        mean1 = sum(series1) / n
        mean2 = sum(series2) / n
        
        # Calculate covariance and standard deviations
        covariance = 0.0
        variance1 = 0.0
        variance2 = 0.0
        
        for i in range(n):
            diff1 = series1[i] - mean1
            diff2 = series2[i] - mean2
            covariance += diff1 * diff2
            variance1 += diff1 * diff1
            variance2 += diff2 * diff2
        
        # Avoid division by zero
        if variance1 == 0 or variance2 == 0:
            return 0.0
            
        # Calculate correlation coefficient
        correlation = covariance / ((variance1 * variance2) ** 0.5)
        
        return correlation
    
    def _get_forecasts(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get price forecasts for specified symbols if available.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dict mapping symbols to their forecast data
        """
        forecast_data = {}
        
        for symbol in symbols:
            try:
                # Get latest forecast
                forecasts = self.retriever.get_forecasts(symbol, limit=1)
                
                if forecasts and len(forecasts) > 0:
                    forecast_data[symbol] = forecasts[0]
                    logger.info(f"Retrieved forecast data for {symbol}")
            except Exception as e:
                logger.error(f"Error retrieving forecast for {symbol}: {e}")
        
        return forecast_data
    
    def generate_market_report(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured market report based on analysis results.
        
        Args:
            market_analysis: Results from analyze_market_data
            
        Returns:
            Dict with formatted market report
        """
        report = {
            "current_prices": {},
            "performance_summary": {},
            "volatility_summary": {},
            "correlations_summary": {},
            "forecasts_summary": {},
            "analysis_date": market_analysis.get("analysis_date", datetime.now().isoformat())
        }
        
        # Process current prices
        for symbol, data in market_analysis.get("current_data", {}).items():
            report["current_prices"][symbol] = {
                "price": data.get("price", 0),
                "change_24h": data.get("price_change_24h", 0),
                "volume_24h": data.get("volume_24h", 0),
                "market_cap": data.get("market_cap", 0)
            }
        
        # Process performance summary
        for symbol, data in market_analysis.get("historical_performance", {}).items():
            report["performance_summary"][symbol] = {
                "last_month": data.get("1m", {}).get("change_pct", 0) if "1m" in data else None,
                "last_quarter": data.get("3m", {}).get("change_pct", 0) if "3m" in data else None,
                "year_to_date": data.get("ytd", {}).get("change_pct", 0) if "ytd" in data else None,
                "last_year": data.get("1y", {}).get("change_pct", 0) if "1y" in data else None,
                "all_time_high": data.get("metrics", {}).get("all_time_high", 0),
                "current_drawdown": data.get("metrics", {}).get("current_drawdown_pct", 0)
            }
        
        # Process volatility summary
        for symbol, data in market_analysis.get("volatility", {}).items():
            report["volatility_summary"][symbol] = {
                "annual_volatility": data.get("annual_volatility", 0),
                "daily_volatility": data.get("daily_volatility", 0),
                "max_daily_gain": data.get("max_daily_gain", 0),
                "max_daily_loss": data.get("max_daily_loss", 0)
            }
        
        # Process correlations (simplified view)
        correlations = market_analysis.get("correlations", {})
        if correlations:
            # Create a simplified correlation matrix
            correlation_matrix = {}
            
            for symbol1, corr_data in correlations.items():
                correlation_matrix[symbol1] = {}
                
                for symbol2, corr_value in corr_data.items():
                    # Round to 2 decimal places
                    correlation_matrix[symbol1][symbol2] = round(corr_value, 2)
            
            report["correlations_summary"] = correlation_matrix
        
        # Process forecasts
        for symbol, forecast in market_analysis.get("forecasts", {}).items():
            # Extract key forecast metrics
            forecast_price = forecast.get("final_forecast", 0)
            current_price = forecast.get("current_price", 0)
            
            # Calculate forecast change
            forecast_change_pct = 0
            if current_price > 0:
                forecast_change_pct = ((forecast_price - current_price) / current_price) * 100
            
            report["forecasts_summary"][symbol] = {
                "current_price": current_price,
                "forecast_price": forecast_price,
                "forecast_change_pct": forecast_change_pct,
                "forecast_period": forecast.get("days_ahead", 30),
                "trend": forecast.get("trend", "unknown"),
                "probability_increase": forecast.get("probability_increase", 0)
            }
        
        return report
    
    def get_market_sentiment(self, crypto_entities: List[str]) -> Dict[str, Any]:
        """
        Get market sentiment for specified cryptocurrencies.
        
        Args:
            crypto_entities: List of cryptocurrency names/symbols
            
        Returns:
            Dict with sentiment analysis data
        """
        sentiment_data = {}
        
        # Normalize crypto names for sentiment lookup
        crypto_names = []
        for entity in crypto_entities:
            entity_lower = entity.lower()
            # Get base name (without USDT, etc.)
            if entity_lower in ["bitcoin", "btc"]:
                crypto_names.append("bitcoin")
            elif entity_lower in ["ethereum", "eth"]:
                crypto_names.append("ethereum")
            elif entity_lower in ["solana", "sol"]:
                crypto_names.append("solana")
            else:
                # Remove common suffixes
                clean_name = entity_lower.replace("usdt", "").replace("usd", "").strip()
                crypto_names.append(clean_name)
        
        # Get sentiment for each crypto
        for crypto in crypto_names:
            try:
                # Get sentiment data from retriever
                crypto_sentiment = self.retriever.get_sentiment_stats(crypto)
                
                if crypto_sentiment and not isinstance(crypto_sentiment, str):
                    sentiment_data[crypto] = crypto_sentiment
            except Exception as e:
                logger.error(f"Error retrieving sentiment for {crypto}: {e}")
        
        return sentiment_data