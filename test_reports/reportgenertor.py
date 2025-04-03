#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Due Diligence Data Analysis Report Generator

This script generates a comprehensive analysis report of all data
collected in the crypto due diligence system collections:
- CryptoDueDiligenceDocuments
- CryptoNewsSentiment 
- MarketMetrics
- CryptoTimeSeries
- OnChainAnalytics

The report is saved as both Markdown and HTML files for easy viewing.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from collections import Counter
import re
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Try to import Weaviate client
try:
    from Sample_Data.vector_store.weaviate_client import get_weaviate_client
    from weaviate.classes.query import Filter, Sort
except ImportError:
    logger.error("Failed to import Weaviate client. Make sure it's installed.")
    sys.exit(1)

# Set up matplotlib for non-interactive mode
plt.switch_backend('agg')

# Define output directory
OUTPUT_DIR = os.path.join(project_root, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define output files
REPORT_DATE = datetime.now().strftime("%Y-%m-%d")
MARKDOWN_FILE = os.path.join(OUTPUT_DIR, f"crypto_analysis_report_{REPORT_DATE}.md")
HTML_FILE = os.path.join(OUTPUT_DIR, f"crypto_analysis_report_{REPORT_DATE}.html")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

class ReportGenerator:
    """Generates comprehensive analysis report of crypto due diligence data"""
    
    def __init__(self):
        """Initialize the report generator"""
        self.client = None
        self.report_md = []  # Store markdown content
        self.figures = []    # Track generated figures
    
    def connect_to_weaviate(self):
        """Connect to Weaviate database"""
        try:
            self.client = get_weaviate_client()
            logger.info("Connected to Weaviate database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            return False
    
    def close_connection(self):
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")
    
    def add_to_report(self, content):
        """Add content to the report"""
        self.report_md.append(content)
    
    def generate_report(self):
        """Generate the full analysis report"""
        if not self.connect_to_weaviate():
            logger.error("Cannot generate report without Weaviate connection")
            return False
        
        try:
            # Report header
            self.add_to_report(f"# Crypto Due Diligence Data Analysis Report\n")
            self.add_to_report(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Table of contents
            self.add_to_report("## Table of Contents\n")
            self.add_to_report("1. [Executive Summary](#executive-summary)")
            self.add_to_report("2. [Market Data Analysis](#market-data-analysis)")
            self.add_to_report("3. [Time Series Analysis](#time-series-analysis)")
            self.add_to_report("4. [News Sentiment Analysis](#news-sentiment-analysis)")
            self.add_to_report("5. [Document Analysis](#document-analysis)")
            self.add_to_report("6. [On-Chain Analytics](#on-chain-analytics)")
            self.add_to_report("7. [Cross-Collection Insights](#cross-collection-insights)")
            self.add_to_report("8. [Data Quality Assessment](#data-quality-assessment)")
            self.add_to_report("9. [Recommendations](#recommendations)\n")
            
            # Generate each section
            self.generate_executive_summary()
            self.analyze_market_data()
            self.analyze_time_series()
            self.analyze_news_sentiment()
            self.analyze_documents()
            self.analyze_onchain_data()
            self.generate_cross_collection_insights()
            self.assess_data_quality()
            self.generate_recommendations()
            
            # Save report to files
            self.save_report()
            
            logger.info(f"Report generated successfully at {MARKDOWN_FILE} and {HTML_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self.close_connection()
    
    def generate_executive_summary(self):
        """Generate executive summary of the data collections"""
        self.add_to_report("## Executive Summary\n")
        
        try:
            # Get collection statistics
            collection_stats = {}
            total_objects = 0
            
            for collection_name in [
                "CryptoDueDiligenceDocuments", 
                "CryptoNewsSentiment",
                "MarketMetrics", 
                "CryptoTimeSeries",
                "OnChainAnalytics"
            ]:
                try:
                    collection = self.client.collections.get(collection_name)
                    count_result = collection.aggregate.over_all(total_count=True)
                    count = count_result.total_count
                    collection_stats[collection_name] = count
                    total_objects += count
                except Exception as e:
                    logger.warning(f"Could not get stats for {collection_name}: {e}")
                    collection_stats[collection_name] = 0
            
            # Add summary statistics
            self.add_to_report("This report analyzes the data collected in the Crypto Due Diligence system. The system contains:")
            
            for collection, count in collection_stats.items():
                self.add_to_report(f"- **{collection}**: {count:,} records")
            
            self.add_to_report(f"\n**Total objects across all collections**: {total_objects:,}\n")
            
            # Generate and add summary figure
            plt.figure(figsize=(10, 6))
            bars = plt.bar(collection_stats.keys(), collection_stats.values(), color=sns.color_palette("viridis", len(collection_stats)))
            plt.xticks(rotation=45, ha='right')
            plt.title('Number of Records by Collection')
            plt.ylabel('Count (log scale)')
            plt.yscale('log')  # Use log scale for better visibility with large differences
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            fig_path = os.path.join(FIGURES_DIR, "collection_stats.png")
            plt.savefig(fig_path)
            plt.close()
            
            self.add_to_report(f"![Collection Statistics]({fig_path})\n")
            self.figures.append(fig_path)
            
            # Add key insights summary
            self.add_to_report("### Key Insights\n")
            
            # Market data insights
            if collection_stats.get("MarketMetrics", 0) > 0:
                collection = self.client.collections.get("MarketMetrics")
                
                # Try to get latest market data for BTC
                btc_data = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal("BTCUSDT"),
                    limit=1,
                    sort=Sort.by_property("timestamp", ascending=False)
                )
                
                if btc_data.objects:
                    btc_price = btc_data.objects[0].properties.get("price", "N/A")
                    self.add_to_report(f"- Current BTC price: ${btc_price:,.2f}" if isinstance(btc_price, (int, float)) else f"- Current BTC price: {btc_price}")
            
            # News sentiment insights
            if collection_stats.get("CryptoNewsSentiment", 0) > 0:
                collection = self.client.collections.get("CryptoNewsSentiment")
                
                # Get sentiment distribution
                sentiments = collection.query.fetch_objects(
                    return_properties=["sentiment_label"],
                    limit=1000
                )
                
                if sentiments.objects:
                    sentiment_counts = Counter([obj.properties.get("sentiment_label", "UNKNOWN") for obj in sentiments.objects])
                    total = sum(sentiment_counts.values())
                    
                    if total > 0:
                        positive_pct = sentiment_counts.get("POSITIVE", 0) / total * 100
                        self.add_to_report(f"- Current market sentiment: {positive_pct:.1f}% positive")
            
            # Time series insights
            if collection_stats.get("CryptoTimeSeries", 0) > 0:
                self.add_to_report(f"- Historical price data spans {collection_stats.get('CryptoTimeSeries', 0):,} data points")
            
            # Document insights
            if collection_stats.get("CryptoDueDiligenceDocuments", 0) > 0:
                self.add_to_report(f"- {collection_stats.get('CryptoDueDiligenceDocuments', 0):,} due diligence documents analyzed")
            
            self.add_to_report("\n")
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            self.add_to_report(f"*Error generating executive summary: {e}*\n")
    
    def analyze_market_data(self):
        """Analyze market metrics data"""
        self.add_to_report("## Market Data Analysis\n")
        
        try:
            collection = self.client.collections.get("MarketMetrics")
            
            # Get all market data
            market_data = collection.query.fetch_objects(
                return_properties=["symbol", "price", "market_cap", "volume_24h", "price_change_24h", "timestamp", "source"],
                limit=1000
            )
            
            if not market_data.objects:
                self.add_to_report("No market data available for analysis.\n")
                return
            
            # Convert to DataFrame
            data = []
            for obj in market_data.objects:
                if obj.properties:
                    data.append(obj.properties)
            
            if not data:
                self.add_to_report("No market data properties available for analysis.\n")
                return
                
            df = pd.DataFrame(data)
            
            # Convert timestamp to datetime if it's a string
            if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get latest data for each symbol
            latest_data = df.sort_values('timestamp').groupby('symbol').last().reset_index()
            
            # Print latest market data table
            self.add_to_report("### Latest Market Data\n")
            
            # Create markdown table
            table_rows = ["| Symbol | Price (USD) | 24h Change | 24h Volume | Market Cap | Source |"]
            table_rows.append("| ------ | ---------- | ---------- | ---------- | ---------- | ------ |")
            
            for _, row in latest_data.iterrows():
                symbol = row.get('symbol', 'N/A')
                price = f"${row.get('price', 0):,.2f}" if 'price' in row and pd.notna(row['price']) else 'N/A'
                change = f"{row.get('price_change_24h', 0):+.2f}%" if 'price_change_24h' in row and pd.notna(row['price_change_24h']) else 'N/A'
                volume = f"${row.get('volume_24h', 0):,.0f}" if 'volume_24h' in row and pd.notna(row['volume_24h']) else 'N/A'
                market_cap = f"${row.get('market_cap', 0):,.0f}" if 'market_cap' in row and pd.notna(row['market_cap']) else 'N/A'
                source = row.get('source', 'N/A')
                
                table_rows.append(f"| {symbol} | {price} | {change} | {volume} | {market_cap} | {source} |")
            
            self.add_to_report("\n".join(table_rows) + "\n")
            
            # Create price comparison chart
            if len(latest_data) > 1:
                plt.figure(figsize=(12, 6))
                
                # Only include symbols with valid price data
                plot_data = latest_data[pd.notna(latest_data['price'])]
                
                if len(plot_data) > 0:
                    # Sort by price
                    plot_data = plot_data.sort_values('price', ascending=False)
                    
                    # Plot prices
                    colors = ['g' if val >= 0 else 'r' for val in plot_data['price_change_24h']]
                    bars = plt.bar(plot_data['symbol'], plot_data['price'], color=colors)
                    
                    plt.title('Current Prices by Symbol')
                    plt.ylabel('Price (USD)')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add price labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                                f'${height:,.2f}', ha='center', va='bottom', rotation=0)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "price_comparison.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Price Comparison]({fig_path})\n")
                    self.figures.append(fig_path)
            
            # Create 24h change comparison chart
            if len(latest_data) > 1:
                plt.figure(figsize=(12, 6))
                
                # Only include symbols with valid price change data
                plot_data = latest_data[pd.notna(latest_data['price_change_24h'])]
                
                if len(plot_data) > 0:
                    # Sort by price change
                    plot_data = plot_data.sort_values('price_change_24h', ascending=False)
                    
                    # Plot price changes
                    colors = ['g' if val >= 0 else 'r' for val in plot_data['price_change_24h']]
                    bars = plt.bar(plot_data['symbol'], plot_data['price_change_24h'], color=colors)
                    
                    plt.title('24h Price Change by Symbol')
                    plt.ylabel('Price Change (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    
                    # Add price change labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3 if height >= 0 else height - 0.8,
                                f'{height:+.2f}%', ha='center', va='bottom', rotation=0)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "price_change_comparison.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Price Change Comparison]({fig_path})\n")
                    self.figures.append(fig_path)
            
            # Market insight summary
            self.add_to_report("### Market Insights\n")
            
            # Calculate high-level stats
            if 'price_change_24h' in df.columns:
                positive_changes = (df['price_change_24h'] > 0).sum()
                total_changes = df['price_change_24h'].count()
                
                if total_changes > 0:
                    market_sentiment = positive_changes / total_changes
                    
                    if market_sentiment > 0.7:
                        sentiment_desc = "strongly bullish"
                    elif market_sentiment > 0.6:
                        sentiment_desc = "bullish"
                    elif market_sentiment > 0.4:
                        sentiment_desc = "neutral"
                    elif market_sentiment > 0.3:
                        sentiment_desc = "bearish"
                    else:
                        sentiment_desc = "strongly bearish"
                    
                    self.add_to_report(f"- Based on price movements, the current market sentiment appears to be **{sentiment_desc}** with {positive_changes} out of {total_changes} ({market_sentiment:.1%}) symbols showing positive 24-hour price change.")
            
            # Highlight best and worst performers
            if len(latest_data) > 1 and 'price_change_24h' in latest_data.columns:
                best_performer = latest_data.loc[latest_data['price_change_24h'].idxmax()]
                worst_performer = latest_data.loc[latest_data['price_change_24h'].idxmin()]
                
                self.add_to_report(f"- Best 24h performer: **{best_performer['symbol']}** with **{best_performer['price_change_24h']:+.2f}%** change")
                self.add_to_report(f"- Worst 24h performer: **{worst_performer['symbol']}** with **{worst_performer['price_change_24h']:+.2f}%** change")
            
            # Volume analysis
            if 'volume_24h' in latest_data.columns:
                highest_volume = latest_data.loc[latest_data['volume_24h'].idxmax()]
                
                self.add_to_report(f"- Highest trading volume: **{highest_volume['symbol']}** with **${highest_volume['volume_24h']:,.0f}** in 24h volume")
                
                # Volume to market cap ratio (liquidity indicator)
                latest_data['volume_to_mcap'] = latest_data['volume_24h'] / latest_data['market_cap'] * 100
                highest_liquidity = latest_data.loc[latest_data['volume_to_mcap'].idxmax()]
                
                self.add_to_report(f"- Highest liquidity (volume/market cap): **{highest_liquidity['symbol']}** with **{highest_liquidity['volume_to_mcap']:.2f}%** ratio")
            
            self.add_to_report("\n")
            
        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            self.add_to_report(f"*Error analyzing market data: {e}*\n")
    
    def analyze_time_series(self):
        """Analyze time series data"""
        self.add_to_report("## Time Series Analysis\n")
        
        try:
            collection = self.client.collections.get("CryptoTimeSeries")
            
            # Get available symbols
            symbols_result = collection.aggregate.over_all(
                group_by="symbol",
                total_count=True
            )
            
            if not symbols_result.groups:
                self.add_to_report("No time series data available for analysis.\n")
                return
            
            # Extract symbols and counts
            symbols = []
            for group in symbols_result.groups:
                if hasattr(group, "grouped_by") and hasattr(group.grouped_by, "value"):
                    symbols.append((group.grouped_by.value, group.total_count))
            
            # Sort by count (descending)
            symbols.sort(key=lambda x: x[1], reverse=True)
            
            self.add_to_report(f"The time series collection contains data for {len(symbols)} symbols. Top symbols by data points:\n")
            
            # Create symbol count table
            table_rows = ["| Symbol | Data Points |"]
            table_rows.append("| ------ | ----------- |")
            
            for symbol, count in symbols[:10]:  # Show top 10
                table_rows.append(f"| {symbol} | {count:,} |")
            
            self.add_to_report("\n".join(table_rows) + "\n")
            
            # Create data points by symbol chart
            plt.figure(figsize=(12, 6))
            
            symbol_names = [s[0] for s in symbols[:10]]
            symbol_counts = [s[1] for s in symbols[:10]]
            
            bars = plt.bar(symbol_names, symbol_counts, color=sns.color_palette("viridis", len(symbol_names)))
            plt.title('Data Points by Symbol')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height):,}', ha='center', va='bottom', rotation=0)
            
            plt.tight_layout()
            fig_path = os.path.join(FIGURES_DIR, "datapoints_by_symbol.png")
            plt.savefig(fig_path)
            plt.close()
            
            self.add_to_report(f"![Data Points by Symbol]({fig_path})\n")
            self.figures.append(fig_path)
            
            # Analyze BTC price trend
            self.add_to_report("### Bitcoin Price Trend\n")
            
            # Get BTC data
            btc_data = collection.query.fetch_objects(
                filters=Filter.by_property("symbol").equal("BTCUSDT"),
                return_properties=["timestamp", "close", "volume", "high", "low"],
                limit=200,  # Last 200 data points
                sort=Sort.by_property("timestamp", ascending=True)
            )
            
            if btc_data.objects:
                # Convert to DataFrame
                data = []
                for obj in btc_data.objects:
                    if obj.properties:
                        data.append(obj.properties)
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Sort by timestamp
                    df = df.sort_values('timestamp')
                    
                    # Plot BTC price trend
                    plt.figure(figsize=(14, 7))
                    
                    plt.plot(df['timestamp'], df['close'], 'b-', linewidth=2)
                    
                    plt.title('Bitcoin Price Trend')
                    plt.ylabel('Price (USD)')
                    plt.grid(True, alpha=0.3)
                    
                    # Format y-axis as dollars
                    plt.gca().yaxis.set_major_formatter('${x:,.0f}')
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "btc_price_trend.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Bitcoin Price Trend]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Calculate key metrics
                    self.add_to_report("**Key Bitcoin Metrics:**\n")
                    
                    latest_price = df['close'].iloc[-1]
                    highest_price = df['high'].max()
                    lowest_price = df['low'].min()
                    
                    # Calculate price changes
                    week_ago_idx = df[df['timestamp'] >= df['timestamp'].iloc[-1] - pd.Timedelta(days=7)].index[0]
                    week_ago_price = df.loc[week_ago_idx, 'close']
                    week_change_pct = (latest_price - week_ago_price) / week_ago_price * 100
                    
                    self.add_to_report(f"- Current price: ${latest_price:,.2f}")
                    self.add_to_report(f"- 7-day change: {week_change_pct:+.2f}%")
                    self.add_to_report(f"- Highest price (period): ${highest_price:,.2f}")
                    self.add_to_report(f"- Lowest price (period): ${lowest_price:,.2f}")
                    self.add_to_report(f"- Volatility (stddev of daily returns): {df['close'].pct_change().std() * 100:.2f}%\n")
                    
                    # Candlestick chart (requires mplfinance)
                    try:
                        import mplfinance as mpf
                        
                        # Prepare data for candlestick chart
                        ohlc_df = df.copy()
                        
                        # Need to add 'open' if not present (use previous close)
                        if 'open' not in ohlc_df.columns:
                            ohlc_df['open'] = ohlc_df['close'].shift(1)
                            ohlc_df['open'].fillna(ohlc_df['close'].iloc[0], inplace=True)
                        
                        # Set timestamp as index
                        ohlc_df.set_index('timestamp', inplace=True)
                        
                        # Filter last 30 data points for better visibility
                        ohlc_df = ohlc_df.iloc[-30:]
                        
                        # Create candlestick chart
                        fig_path = os.path.join(FIGURES_DIR, "btc_candlestick.png")
                        
                        mpf.plot(
                            ohlc_df,
                            type='candle',
                            style='yahoo',
                            title='Bitcoin Price (Candlestick)',
                            ylabel='Price (USD)',
                            volume=True if 'volume' in ohlc_df.columns else False,
                            figsize=(14, 7),
                            savefig=fig_path
                        )
                        
                        self.add_to_report(f"![Bitcoin Candlestick Chart]({fig_path})\n")
                        self.figures.append(fig_path)
                    except ImportError:
                        logger.warning("mplfinance package not installed. Skipping candlestick chart.")
                    except Exception as e:
                        logger.error(f"Error creating candlestick chart: {e}")
            else:
                self.add_to_report("No Bitcoin price data available for analysis.\n")
                
            # Price volatility comparison
            self.add_to_report("### Volatility Comparison\n")
            
            # Calculate volatility for multiple assets
            volatility_data = {}
            
            for symbol, df_data in symbol_data.items():
                if 'close' in df_data.columns and len(df_data) > 5:
                    # Calculate daily returns
                    df_data['pct_change'] = df_data['close'].pct_change()
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = df_data['pct_change'].std() * 100  # Convert to percentage
                    volatility_data[symbol] = volatility
            
            if volatility_data:
                # Create volatility comparison chart
                plt.figure(figsize=(12, 6))
                
                # Sort by volatility
                volatility_data = {k: v for k, v in sorted(volatility_data.items(), key=lambda item: item[1], reverse=True)}
                
                bars = plt.bar(volatility_data.keys(), volatility_data.values(), color=sns.color_palette("viridis", len(volatility_data)))
                plt.title('Volatility by Asset (StdDev of Daily Returns)')
                plt.ylabel('Volatility (%)')
                plt.xticks(rotation=45, ha='right')
                
                # Add volatility labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}%', ha='center', va='bottom', rotation=0)
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "volatility_comparison.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Volatility Comparison]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Add volatility insights
                self.add_to_report("**Volatility Insights:**\n")
                
                highest_vol = max(volatility_data.items(), key=lambda x: x[1])
                lowest_vol = min(volatility_data.items(), key=lambda x: x[1])
                
                self.add_to_report(f"- Most volatile asset: **{highest_vol[0]}** with {highest_vol[1]:.2f}% volatility")
                self.add_to_report(f"- Least volatile asset: **{lowest_vol[0]}** with {lowest_vol[1]:.2f}% volatility")
                self.add_to_report(f"- Average volatility across assets: {sum(volatility_data.values())/len(volatility_data):.2f}%\n")
            
            # Correlation analysis
            self.add_to_report("### Cross-Asset Correlation\n")
            
            # Get top 5 symbols for correlation analysis
            top_symbols = [s[0] for s in symbols[:5]]
            
            symbol_data = {}
            for symbol in top_symbols:
                # Get data for each symbol
                data = collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal(symbol),
                    return_properties=["timestamp", "close"],
                    limit=100,  # Last 100 data points
                    sort=Sort.by_property("timestamp", ascending=True)
                )
                
                if data.objects:
                    # Convert to DataFrame
                    df_data = []
                    for obj in data.objects:
                        if obj.properties:
                            df_data.append(obj.properties)
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Sort by timestamp
                        df = df.sort_values('timestamp')
                        
                        symbol_data[symbol] = df
            
            if len(symbol_data) > 1:
                # Merge price data to calculate correlations
                merged_df = None
                
                for symbol, df in symbol_data.items():
                    if merged_df is None:
                        merged_df = df[['timestamp', 'close']].copy()
                        merged_df.rename(columns={'close': symbol}, inplace=True)
                    else:
                        # Find closest timestamps and merge
                        temp_df = df[['timestamp', 'close']].copy()
                        temp_df.rename(columns={'close': symbol}, inplace=True)
                        
                        # Simple merge on closest timestamp
                        merged_df = pd.merge_asof(merged_df, temp_df, on='timestamp')
                
                if merged_df is not None and len(merged_df.columns) > 2:  # timestamp + at least 2 symbols
                    # Calculate correlation matrix
                    correlation_matrix = merged_df.drop('timestamp', axis=1).corr()
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                    plt.title('Price Correlation Between Assets')
                    plt.tight_layout()
                    
                    fig_path = os.path.join(FIGURES_DIR, "price_correlation.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Price Correlation Matrix]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Add correlation insights
                    high_corr_pairs = []
                    low_corr_pairs = []
                    
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            sym1 = correlation_matrix.columns[i]
                            sym2 = correlation_matrix.columns[j]
                            corr = correlation_matrix.iloc[i, j]
                            
                            if corr > 0.8:
                                high_corr_pairs.append((sym1, sym2, corr))
                            elif corr < 0.2:
                                low_corr_pairs.append((sym1, sym2, corr))
                    
                    if high_corr_pairs:
                        self.add_to_report("**Highly correlated pairs:**\n")
                        for sym1, sym2, corr in high_corr_pairs:
                            self.add_to_report(f"- {sym1} and {sym2}: {corr:.2f}")
                        self.add_to_report("")
                    
                    if low_corr_pairs:
                        self.add_to_report("**Weakly correlated pairs (potential diversification):**\n")
                        for sym1, sym2, corr in low_corr_pairs:
                            self.add_to_report(f"- {sym1} and {sym2}: {corr:.2f}")
                        self.add_to_report("")
        except Exception as e:
            logger.error(f"Error analyzing time series data: {e}")
            self.add_to_report(f"*Error analyzing time series data: {e}*\n")
    
    def analyze_news_sentiment(self):
        """Analyze news sentiment data"""
        self.add_to_report("## News Sentiment Analysis\n")
        
        try:
            collection = self.client.collections.get("CryptoNewsSentiment")
            
            # Get all sentiment data
            sentiment_data = collection.query.fetch_objects(
                return_properties=["title", "source", "sentiment_label", "sentiment_score", "date", "content"],
                limit=1000
            )
            
            if not sentiment_data.objects:
                self.add_to_report("No sentiment data available for analysis.\n")
                return
            
            # Convert to DataFrame
            data = []
            for obj in sentiment_data.objects:
                if obj.properties:
                    data.append(obj.properties)
            
            if not data:
                self.add_to_report("No sentiment data properties available for analysis.\n")
                return
                
            df = pd.DataFrame(data)
            
            # Convert date to datetime if it's a string
            if 'date' in df.columns and isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Overall sentiment distribution
            self.add_to_report("### Overall Sentiment Distribution\n")
            
            if 'sentiment_label' in df.columns:
                # Count sentiment labels
                sentiment_counts = df['sentiment_label'].value_counts()
                
                # Create a pie chart
                plt.figure(figsize=(10, 6))
                
                colors = {'POSITIVE': 'green', 'NEUTRAL': 'gray', 'NEGATIVE': 'red'}
                sentiment_colors = [colors.get(label, 'blue') for label in sentiment_counts.index]
                
                plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                        startangle=90, colors=sentiment_colors)
                plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                plt.title('Distribution of Sentiment in News Articles')
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "sentiment_distribution.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Sentiment Distribution]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Calculate percentages
                total_articles = sentiment_counts.sum()
                positive_pct = sentiment_counts.get('POSITIVE', 0) / total_articles * 100 if total_articles > 0 else 0
                negative_pct = sentiment_counts.get('NEGATIVE', 0) / total_articles * 100 if total_articles > 0 else 0
                neutral_pct = sentiment_counts.get('NEUTRAL', 0) / total_articles * 100 if total_articles > 0 else 0
                
                self.add_to_report(f"Analysis of {total_articles} news articles shows:")
                self.add_to_report(f"- Positive sentiment: {positive_pct:.1f}% ({sentiment_counts.get('POSITIVE', 0)} articles)")
                self.add_to_report(f"- Neutral sentiment: {neutral_pct:.1f}% ({sentiment_counts.get('NEUTRAL', 0)} articles)")
                self.add_to_report(f"- Negative sentiment: {negative_pct:.1f}% ({sentiment_counts.get('NEGATIVE', 0)} articles)\n")
                
                # Sentiment over time
                if 'date' in df.columns and not df['date'].isna().all():
                    self.add_to_report("### Sentiment Trend Over Time\n")
                    
                    # Add a day column for grouping
                    df['day'] = df['date'].dt.date
                    
                    # Group by day and calculate average sentiment score
                    daily_sentiment = df.groupby('day')['sentiment_score'].agg(['mean', 'count']).reset_index()
                    daily_sentiment.columns = ['day', 'avg_sentiment', 'article_count']
                    
                    # Sort by date
                    daily_sentiment = daily_sentiment.sort_values('day')
                    
                    # Plot sentiment trend
                    plt.figure(figsize=(14, 7))
                    
                    # Plot average sentiment
                    plt.plot(daily_sentiment['day'], daily_sentiment['avg_sentiment'], 'b-', linewidth=2)
                    
                    # Add reference line for neutral sentiment
                    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
                    
                    # Shade areas above/below neutral
                    plt.fill_between(daily_sentiment['day'], daily_sentiment['avg_sentiment'], 0.5,
                                    where=(daily_sentiment['avg_sentiment'] >= 0.5),
                                    interpolate=True, color='green', alpha=0.3)
                    plt.fill_between(daily_sentiment['day'], daily_sentiment['avg_sentiment'], 0.5,
                                    where=(daily_sentiment['avg_sentiment'] <= 0.5),
                                    interpolate=True, color='red', alpha=0.3)
                    
                    plt.title('Average Sentiment Score Over Time')
                    plt.ylabel('Sentiment Score (0-1)')
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "sentiment_trend.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Sentiment Trend]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Calculate recent sentiment trend
                    if len(daily_sentiment) > 3:
                        last_days = min(7, len(daily_sentiment))
                        recent_sentiment = daily_sentiment.iloc[-last_days:]
                        
                        avg_recent = recent_sentiment['avg_sentiment'].mean()
                        
                        if len(daily_sentiment) > last_days:
                            previous_period = daily_sentiment.iloc[-(last_days*2):-last_days]
                            avg_previous = previous_period['avg_sentiment'].mean()
                            
                            sentiment_change = avg_recent - avg_previous
                            
                            if sentiment_change > 0.05:
                                trend_desc = "improving"
                            elif sentiment_change < -0.05:
                                trend_desc = "deteriorating"
                            else:
                                trend_desc = "stable"
                            
                            self.add_to_report(f"The average sentiment score over the last {last_days} days is {avg_recent:.2f}, indicating a **{trend_desc}** trend compared to the previous period ({sentiment_change:+.2f}).\n")
                        else:
                            self.add_to_report(f"The average sentiment score over the last {last_days} days is {avg_recent:.2f}.\n")
                
                # Recent news summary
                self.add_to_report("### Recent News Highlights\n")
                
                if 'date' in df.columns and not df['date'].isna().all():
                    # Sort by date (descending)
                    recent_news = df.sort_values('date', ascending=False).head(5)
                    
                    self.add_to_report("**Most recent news articles:**\n")
                    
                    for _, article in recent_news.iterrows():
                        title = article.get('title', 'No title')
                        source = article.get('source', 'Unknown source')
                        sentiment = article.get('sentiment_label', 'NEUTRAL')
                        date = article.get('date', '')
                        
                        if hasattr(date, 'strftime'):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        
                        # Get emoji based on sentiment
                        if sentiment == 'POSITIVE':
                            emoji = "📈"
                        elif sentiment == 'NEGATIVE':
                            emoji = "📉"
                        else:
                            emoji = "⚖️"
                        
                        self.add_to_report(f"- {date_str} | {emoji} **{title}** - *{source}*")
                    
                    self.add_to_report("")
                
                # Top positive and negative articles
                if 'sentiment_score' in df.columns:
                    most_positive = df.nlargest(3, 'sentiment_score')
                    most_negative = df.nsmallest(3, 'sentiment_score')
                    
                    self.add_to_report("**Most positive news articles:**\n")
                    
                    for _, article in most_positive.iterrows():
                        title = article.get('title', 'No title')
                        source = article.get('source', 'Unknown source')
                        score = article.get('sentiment_score', 0)
                        
                        self.add_to_report(f"- 📈 **{title}** - *{source}* (Score: {score:.2f})")
                    
                    self.add_to_report("\n**Most negative news articles:**\n")
                    
                    for _, article in most_negative.iterrows():
                        title = article.get('title', 'No title')
                        source = article.get('source', 'Unknown source')
                        score = article.get('sentiment_score', 0)
                        
                        self.add_to_report(f"- 📉 **{title}** - *{source}* (Score: {score:.2f})")
                    
                    self.add_to_report("")
                
                # Source analysis
                if 'source' in df.columns:
                    self.add_to_report("### News Source Analysis\n")
                    
                    # Count articles by source
                    source_counts = df['source'].value_counts().reset_index()
                    source_counts.columns = ['source', 'count']
                    
                    # Create a bar chart of sources
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(source_counts['source'], source_counts['count'], color='skyblue')
                    plt.title('Number of Articles by Source')
                    plt.ylabel('Article Count')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                str(int(height)), ha='center', va='bottom', rotation=0)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "articles_by_source.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Articles by Source]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Calculate average sentiment by source
                    if 'sentiment_score' in df.columns:
                        source_sentiment = df.groupby('source')['sentiment_score'].mean().reset_index()
                        source_sentiment = source_sentiment.sort_values('sentiment_score', ascending=False)
                        
                        # Create a bar chart of sentiment by source
                        plt.figure(figsize=(12, 6))
                        
                        # Color bars based on sentiment score
                        colors = ['green' if score > 0.6 else 'red' if score < 0.4 else 'gray' 
                                for score in source_sentiment['sentiment_score']]
                        
                        bars = plt.bar(source_sentiment['source'], source_sentiment['sentiment_score'], color=colors)
                        plt.title('Average Sentiment Score by Source')
                        plt.ylabel('Sentiment Score (0-1)')
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add neutral line
                        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
                        
                        # Add score labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
                        
                        plt.tight_layout()
                        fig_path = os.path.join(FIGURES_DIR, "sentiment_by_source.png")
                        plt.savefig(fig_path)
                        plt.close()
                        
                        self.add_to_report(f"![Sentiment by Source]({fig_path})\n")
                        self.figures.append(fig_path)
                        
                        # Add source sentiment insights
                        self.add_to_report("**Source Sentiment Analysis:**\n")
                        
                        most_positive_source = source_sentiment.iloc[0]
                        most_negative_source = source_sentiment.iloc[-1]
                        
                        self.add_to_report(f"- Most positive source: **{most_positive_source['source']}** with average sentiment score of {most_positive_source['sentiment_score']:.2f}")
                        self.add_to_report(f"- Most negative source: **{most_negative_source['source']}** with average sentiment score of {most_negative_source['sentiment_score']:.2f}")
                        self.add_to_report("")
                
                # Word cloud of content
                try:
                    from wordcloud import WordCloud
                    
                    if 'content' in df.columns:
                        self.add_to_report("### News Content Word Cloud\n")
                        
                        # Combine all content
                        all_text = " ".join(df['content'].fillna(""))
                        
                        # Create word cloud
                        wordcloud = WordCloud(
                            width=800, height=400,
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            contour_width=3
                        ).generate(all_text)
                        
                        plt.figure(figsize=(16, 8))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        plt.tight_layout()
                        
                        fig_path = os.path.join(FIGURES_DIR, "news_wordcloud.png")
                        plt.savefig(fig_path)
                        plt.close()
                        
                        self.add_to_report(f"![News Content Word Cloud]({fig_path})\n")
                        self.figures.append(fig_path)
                except ImportError:
                    logger.warning("WordCloud package not installed. Skipping word cloud generation.")
            else:
                self.add_to_report("No sentiment labels found in the data.\n")
        
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            self.add_to_report(f"*Error analyzing news sentiment: {e}*\n")
    
    def analyze_documents(self):
        """Analyze due diligence documents"""
        self.add_to_report("## Document Analysis\n")
        
        try:
            collection = self.client.collections.get("CryptoDueDiligenceDocuments")
            
            # Get document data
            doc_data = collection.query.fetch_objects(
                return_properties=["title", "document_type", "source", "word_count", "extracted_risk_score"],
                limit=1000
            )
            
            if not doc_data.objects:
                self.add_to_report("No document data available for analysis.\n")
                return
            
            # Convert to DataFrame
            data = []
            for obj in doc_data.objects:
                if obj.properties:
                    data.append(obj.properties)
            
            if not data:
                self.add_to_report("No document properties available for analysis.\n")
                return
                
            df = pd.DataFrame(data)
            
            # Document type distribution
            self.add_to_report("### Document Type Distribution\n")
            
            if 'document_type' in df.columns:
                # Count document types
                type_counts = df['document_type'].value_counts()
                
                # Create a pie chart
                plt.figure(figsize=(10, 6))
                
                plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', 
                        startangle=90, colors=sns.color_palette("viridis", len(type_counts)))
                plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
                plt.title('Distribution of Document Types')
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "document_type_distribution.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Document Type Distribution]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Calculate percentages
                total_docs = type_counts.sum()
                
                self.add_to_report(f"Analysis of {total_docs} documents shows the following distribution:")
                
                for doc_type, count in type_counts.items():
                    percentage = count / total_docs * 100
                    self.add_to_report(f"- **{doc_type}**: {count} documents ({percentage:.1f}%)")
                
                self.add_to_report("")
            
            # Word count distribution
            if 'word_count' in df.columns:
                self.add_to_report("### Document Length Analysis\n")
                
                # Remove any NaN values
                word_counts = df['word_count'].dropna()
                
                if not word_counts.empty:
                    # Calculate statistics
                    avg_words = word_counts.mean()
                    median_words = word_counts.median()
                    max_words = word_counts.max()
                    min_words = word_counts.min()
                    
                    self.add_to_report(f"**Document length statistics:**")
                    self.add_to_report(f"- Average word count: {avg_words:.0f} words")
                    self.add_to_report(f"- Median word count: {median_words:.0f} words")
                    self.add_to_report(f"- Longest document: {max_words:.0f} words")
                    self.add_to_report(f"- Shortest document: {min_words:.0f} words")
                    self.add_to_report("")
                    
                    # Create histogram of word counts
                    plt.figure(figsize=(12, 6))
                    
                    plt.hist(word_counts, bins=20, alpha=0.7, color='skyblue')
                    plt.axvline(avg_words, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_words:.0f}')
                    plt.axvline(median_words, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_words:.0f}')
                    
                    plt.title('Distribution of Document Lengths')
                    plt.xlabel('Word Count')
                    plt.ylabel('Number of Documents')
                    plt.legend()
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "word_count_distribution.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Word Count Distribution]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Word count by document type
                    if 'document_type' in df.columns:
                        # Group by document type and calculate statistics
                        type_word_counts = df.groupby('document_type')['word_count'].agg(['mean', 'median', 'count']).reset_index()
                        type_word_counts = type_word_counts.sort_values('mean', ascending=False)
                        
                        # Create a bar chart
                        plt.figure(figsize=(12, 6))
                        
                        bars = plt.bar(type_word_counts['document_type'], type_word_counts['mean'], color=sns.color_palette("viridis", len(type_word_counts)))
                        
                        plt.title('Average Word Count by Document Type')
                        plt.xlabel('Document Type')
                        plt.ylabel('Average Word Count')
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add count labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{int(height):,}', ha='center', va='bottom', rotation=0)
                        
                        plt.tight_layout()
                        fig_path = os.path.join(FIGURES_DIR, "word_count_by_type.png")
                        plt.savefig(fig_path)
                        plt.close()
                        
                        self.add_to_report(f"![Word Count by Document Type]({fig_path})\n")
                        self.figures.append(fig_path)
                        
                        # Add insights
                        self.add_to_report("**Document length by type:**\n")
                        
                        for _, row in type_word_counts.iterrows():
                            doc_type = row['document_type']
                            mean_words = row['mean']
                            median_words = row['median']
                            count = row['count']
                            
                            self.add_to_report(f"- **{doc_type}** ({count} documents): Average {mean_words:.0f} words, Median {median_words:.0f} words")
                        
                        self.add_to_report("")
            
            # Risk score analysis
            if 'extracted_risk_score' in df.columns:
                self.add_to_report("### Risk Assessment\n")
                
                # Remove any NaN values
                risk_scores = df['extracted_risk_score'].dropna()
                
                if not risk_scores.empty:
                    # Calculate statistics
                    avg_risk = risk_scores.mean()
                    median_risk = risk_scores.median()
                    high_risk_count = (risk_scores > 70).sum()
                    low_risk_count = (risk_scores < 30).sum()
                    
                    self.add_to_report(f"**Risk score statistics:**")
                    self.add_to_report(f"- Average risk score: {avg_risk:.1f}/100")
                    self.add_to_report(f"- Median risk score: {median_risk:.1f}/100")
                    self.add_to_report(f"- High risk documents (>70): {high_risk_count} ({high_risk_count/len(risk_scores)*100:.1f}%)")
                    self.add_to_report(f"- Low risk documents (<30): {low_risk_count} ({low_risk_count/len(risk_scores)*100:.1f}%)")
                    self.add_to_report("")
                    
                    # Create histogram of risk scores
                    plt.figure(figsize=(12, 6))
                    
                    plt.hist(risk_scores, bins=20, alpha=0.7, color='salmon')
                    plt.axvline(avg_risk, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {avg_risk:.1f}')
                    plt.axvline(median_risk, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_risk:.1f}')
                    
                    # Add risk zones
                    plt.axvspan(0, 30, alpha=0.2, color='green', label='Low Risk')
                    plt.axvspan(30, 70, alpha=0.2, color='yellow', label='Medium Risk')
                    plt.axvspan(70, 100, alpha=0.2, color='red', label='High Risk')
                    
                    plt.title('Distribution of Risk Scores')
                    plt.xlabel('Risk Score')
                    plt.ylabel('Number of Documents')
                    plt.legend()
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "risk_score_distribution.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Risk Score Distribution]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Risk score by document type
                    if 'document_type' in df.columns:
                        # Group by document type and calculate statistics
                        type_risk_scores = df.groupby('document_type')['extracted_risk_score'].agg(['mean', 'median', 'count']).reset_index()
                        type_risk_scores = type_risk_scores.sort_values('mean', ascending=False)
                        
                        # Create a bar chart
                        plt.figure(figsize=(12, 6))
                        
                        # Color bars based on risk score
                        colors = ['red' if score > 70 else 'orange' if score > 30 else 'green' 
                                for score in type_risk_scores['mean']]
                        
                        bars = plt.bar(type_risk_scores['document_type'], type_risk_scores['mean'], color=colors)
                        
                        plt.title('Average Risk Score by Document Type')
                        plt.xlabel('Document Type')
                        plt.ylabel('Average Risk Score')
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add score labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    f'{height:.1f}', ha='center', va='bottom', rotation=0)
                        
                        plt.tight_layout()
                        fig_path = os.path.join(FIGURES_DIR, "risk_score_by_type.png")
                        plt.savefig(fig_path)
                        plt.close()
                        
                        self.add_to_report(f"![Risk Score by Document Type]({fig_path})\n")
                        self.figures.append(fig_path)
                        
                        # Add insights
                        self.add_to_report("**Risk scores by document type:**\n")
                        
                        for _, row in type_risk_scores.iterrows():
                            doc_type = row['document_type']
                            mean_risk = row['mean']
                            median_risk = row['median']
                            count = row['count']
                            
                            # Classify risk level
                            if mean_risk > 70:
                                risk_level = "High"
                            elif mean_risk > 30:
                                risk_level = "Medium"
                            else:
                                risk_level = "Low"
                            
                            self.add_to_report(f"- **{doc_type}** ({count} documents): Average risk {mean_risk:.1f}/100 ({risk_level})")
                        
                        self.add_to_report("")
            
            # Document sources analysis
            if 'source' in df.columns:
                self.add_to_report("### Document Source Analysis\n")
                
                # Get top sources
                source_counts = df['source'].value_counts().head(10)
                
                # Create a bar chart
                plt.figure(figsize=(12, 6))
                
                bars = plt.bar(source_counts.index, source_counts.values, color='lightgreen')
                plt.title('Top Document Sources')
                plt.ylabel('Number of Documents')
                plt.xticks(rotation=45, ha='right')
                
                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            str(int(height)), ha='center', va='bottom', rotation=0)
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "document_sources.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Document Sources]({fig_path})\n")
                self.figures.append(fig_path)
        
        except Exception as e:
            logger.error(f"Error analyzing documents: {e}")
            self.add_to_report(f"*Error analyzing documents: {e}*\n")
    
    def analyze_onchain_data(self):
        """Analyze on-chain analytics data"""
        self.add_to_report("## On-Chain Analytics\n")
        
        try:
            collection = self.client.collections.get("OnChainAnalytics")
            
            # Get on-chain data
            onchain_data = collection.query.fetch_objects(
                return_properties=["address", "blockchain", "entity_type", "transaction_count", 
                                 "token_transaction_count", "balance", "active_days",
                                 "unique_interactions", "contract_interactions", "tokens",
                                 "risk_score", "risk_level"],
                limit=1000
            )
            
            if not onchain_data.objects:
                self.add_to_report("No on-chain analytics data available for analysis.\n")
                return
            
            # Convert to DataFrame
            data = []
            for obj in onchain_data.objects:
                if obj.properties:
                    data.append(obj.properties)
            
            if not data:
                self.add_to_report("No on-chain analytics properties available for analysis.\n")
                return
                
            df = pd.DataFrame(data)
            
            # Summary table
            self.add_to_report("### On-Chain Wallet Summary\n")
            
            # Create summary table
            table_rows = ["| Address | Blockchain | Entity Type | Tx Count | Token Tx Count | Balance | Risk Level |"]
            table_rows.append("| ------- | ---------- | ----------- | -------- | ------------- | ------- | ---------- |")
            
            for _, row in df.iterrows():
                address = row.get('address', 'N/A')
                # Format address for display (truncate middle)
                if len(address) > 16:
                    address = f"{address[:8]}...{address[-8:]}"
                
                blockchain = row.get('blockchain', 'N/A')
                entity_type = row.get('entity_type', 'N/A')
                tx_count = row.get('transaction_count', 0)
                token_tx_count = row.get('token_transaction_count', 0)
                balance = row.get('balance', 0)
                risk_level = row.get('risk_level', 'N/A')
                
                table_rows.append(f"| {address} | {blockchain} | {entity_type} | {tx_count:,} | {token_tx_count:,} | {balance:.4f} | {risk_level} |")
            
            self.add_to_report("\n".join(table_rows) + "\n")
            
            # Transaction analysis
            if 'transaction_count' in df.columns and 'token_transaction_count' in df.columns:
                self.add_to_report("### Transaction Analysis\n")
                
                # Create transaction bar chart
                plt.figure(figsize=(10, 6))
                
                # For each address, create a stacked bar of regular and token transactions
                addresses = []
                for i, row in df.iterrows():
                    address = row.get('address', 'Unknown')
                    if len(address) > 16:
                        address = f"{address[:8]}...{address[-4:]}"
                    addresses.append(address)
                
                regular_txs = df['transaction_count'].values
                token_txs = df['token_transaction_count'].values
                
                # Create stacked bar chart
                plt.bar(addresses, regular_txs, label='Regular Transactions')
                plt.bar(addresses, token_txs, bottom=regular_txs, label='Token Transactions')
                
                plt.title('Transaction Distribution by Address')
                plt.xlabel('Address')
                plt.ylabel('Transaction Count')
                plt.legend()
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "onchain_transactions.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Transaction Distribution]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Calculate statistics
                total_regular_txs = df['transaction_count'].sum()
                total_token_txs = df['token_transaction_count'].sum()
                total_txs = total_regular_txs + total_token_txs
                
                self.add_to_report(f"**Transaction Statistics:**")
                self.add_to_report(f"- Total transactions: {total_txs:,}")
                self.add_to_report(f"- Regular transactions: {total_regular_txs:,} ({total_regular_txs/total_txs*100:.1f}%)")
                self.add_to_report(f"- Token transactions: {total_token_txs:,} ({total_token_txs/total_txs*100:.1f}%)")
                self.add_to_report("")
            
            # Network analysis
            if 'unique_interactions' in df.columns and 'contract_interactions' in df.columns:
                self.add_to_report("### Network Analysis\n")
                
                # Create network interaction chart
                plt.figure(figsize=(10, 6))
                
                # For each address, create a stacked bar of unique interactions and contract interactions
                addresses = []
                for i, row in df.iterrows():
                    address = row.get('address', 'Unknown')
                    if len(address) > 16:
                        address = f"{address[:8]}...{address[-4:]}"
                    addresses.append(address)
                
                unique_interactions = df['unique_interactions'].values
                contract_interactions = df['contract_interactions'].values
                
                # Create bar chart
                plt.bar(addresses, unique_interactions, label='Unique Interactions')
                plt.bar(addresses, contract_interactions, alpha=0.7, label='Contract Interactions')
                
                plt.title('Network Interactions by Address')
                plt.xlabel('Address')
                plt.ylabel('Interaction Count')
                plt.legend()
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "onchain_network.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Network Interactions]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Calculate statistics
                avg_unique = df['unique_interactions'].mean()
                avg_contracts = df['contract_interactions'].mean()
                
                self.add_to_report(f"**Network Statistics:**")
                self.add_to_report(f"- Average unique interactions per address: {avg_unique:.1f}")
                self.add_to_report(f"- Average contract interactions per address: {avg_contracts:.1f}")
                self.add_to_report("")
            
            # Token analysis
            if 'tokens' in df.columns:
                self.add_to_report("### Token Analysis\n")
                
                # Collect all tokens
                all_tokens = []
                for i, row in df.iterrows():
                    tokens = row.get('tokens', [])
                    if tokens:
                        all_tokens.extend(tokens)
                
                # Count token occurrences
                token_counts = Counter(all_tokens)
                
                if token_counts:
                    # Get top tokens
                    top_tokens = token_counts.most_common(10)
                    
                    # Create token distribution chart
                    plt.figure(figsize=(12, 6))
                    
                    token_names = [t[0] for t in top_tokens]
                    token_counts_values = [t[1] for t in top_tokens]
                    
                    bars = plt.bar(token_names, token_counts_values, color='lightblue')
                    
                    plt.title('Most Common Tokens Across Addresses')
                    plt.xlabel('Token')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                str(int(height)), ha='center', va='bottom', rotation=0)
                    
                    plt.tight_layout()
                    fig_path = os.path.join(FIGURES_DIR, "onchain_tokens.png")
                    plt.savefig(fig_path)
                    plt.close()
                    
                    self.add_to_report(f"![Token Distribution]({fig_path})\n")
                    self.figures.append(fig_path)
                    
                    # Token insights
                    self.add_to_report(f"**Token Insights:**")
                    self.add_to_report(f"- Total unique tokens: {len(token_counts)}")
                    self.add_to_report(f"- Most common token: **{top_tokens[0][0]}** (found in {top_tokens[0][1]} addresses)")
                    self.add_to_report("")
                    
                    # List top tokens
                    self.add_to_report("**Top tokens by occurrence:**")
                    for token, count in top_tokens:
                        self.add_to_report(f"- {token}: {count} occurrences")
                    self.add_to_report("")
            
            # Risk analysis
            if 'risk_score' in df.columns and 'risk_level' in df.columns:
                self.add_to_report("### Risk Analysis\n")
                
                # Create risk score chart
                plt.figure(figsize=(10, 6))
                
                # For each address, show risk score
                addresses = []
                for i, row in df.iterrows():
                    address = row.get('address', 'Unknown')
                    if len(address) > 16:
                        address = f"{address[:8]}...{address[-4:]}"
                    addresses.append(address)
                
                risk_scores = df['risk_score'].values
                
                # Color bars based on risk level
                colors = ['green' if score < 30 else 'orange' if score < 70 else 'red' for score in risk_scores]
                
                # Create bar chart
                bars = plt.bar(addresses, risk_scores, color=colors)
                
                plt.title('Risk Scores by Address')
                plt.xlabel('Address')
                plt.ylabel('Risk Score (0-100)')
                
                # Add score labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.0f}', ha='center', va='bottom', rotation=0)
                
                # Add risk zones
                plt.axhspan(0, 30, alpha=0.2, color='green', label='Low Risk')
                plt.axhspan(30, 70, alpha=0.2, color='orange', label='Medium Risk')
                plt.axhspan(70, 100, alpha=0.2, color='red', label='High Risk')
                
                plt.legend()
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "onchain_risk.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Risk Scores]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Risk distribution
                risk_counts = df['risk_level'].value_counts()
                
                # Create pie chart of risk levels
                plt.figure(figsize=(8, 8))
                
                colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Very Low': 'lightgreen', 'Very High': 'darkred'}
                risk_colors = [colors.get(level, 'gray') for level in risk_counts.index]
                
                plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
                        startangle=90, colors=risk_colors)
                plt.axis('equal')
                plt.title('Risk Level Distribution')
                
                plt.tight_layout()
                fig_path = os.path.join(FIGURES_DIR, "onchain_risk_levels.png")
                plt.savefig(fig_path)
                plt.close()
                
                self.add_to_report(f"![Risk Level Distribution]({fig_path})\n")
                self.figures.append(fig_path)
                
                # Risk insights
                self.add_to_report(f"**Risk Insights:**")
                
                avg_risk = df['risk_score'].mean()
                self.add_to_report(f"- Average risk score: {avg_risk:.1f}/100")
                
                for level, count in risk_counts.items():
                    self.add_to_report(f"- {level} risk addresses: {count} ({count/len(df)*100:.1f}%)")
                
                self.add_to_report("")
        
        except Exception as e:
            logger.error(f"Error analyzing on-chain data: {e}")
            self.add_to_report(f"*Error analyzing on-chain data: {e}*\n")
    
    def generate_cross_collection_insights(self):
        """Generate insights across collections"""
        self.add_to_report("## Cross-Collection Insights\n")
        
        try:
            # Sentiment vs. Price correlation
            self.add_to_report("### Sentiment vs. Price Correlation\n")
            
            try:
                # Get sentiment data
                sentiment_collection = self.client.collections.get("CryptoNewsSentiment")
                
                # Get sentiment data with dates
                sentiment_data = sentiment_collection.query.fetch_objects(
                    return_properties=["date", "sentiment_score", "sentiment_label"],
                    limit=1000
                )
                
                # Get price data
                market_collection = self.client.collections.get("MarketMetrics")
                
                # Get Bitcoin price data
                price_data = market_collection.query.fetch_objects(
                    filters=Filter.by_property("symbol").equal("BTCUSDT"),
                    return_properties=["timestamp", "price", "price_change_24h"],
                    limit=1000
                )
                
                # If we have both types of data, try to correlate them
                if sentiment_data.objects and price_data.objects:
                    # Convert to DataFrame
                    sentiment_df = pd.DataFrame([obj.properties for obj in sentiment_data.objects if obj.properties])
                    price_df = pd.DataFrame([obj.properties for obj in price_data.objects if obj.properties])
                    
                    # Convert dates to datetime
                    if 'date' in sentiment_df.columns and isinstance(sentiment_df['date'].iloc[0], str):
                        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
                    
                    if 'timestamp' in price_df.columns and isinstance(price_df['timestamp'].iloc[0], str):
                        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], errors='coerce')
                    
                    # Group sentiment by day
                    if 'date' in sentiment_df.columns and not sentiment_df['date'].isna().all():
                        sentiment_df['day'] = sentiment_df['date'].dt.date
                        daily_sentiment = sentiment_df.groupby('day')['sentiment_score'].mean().reset_index()
                        
                        # Group price by day
                        if 'timestamp' in price_df.columns and not price_df['timestamp'].isna().all():
                            price_df['day'] = price_df['timestamp'].dt.date
                            daily_price = price_df.groupby('day')['price'].mean().reset_index()
                            
                            # Merge datasets on day
                            merged_df = pd.merge(daily_sentiment, daily_price, on='day', how='inner')
                            
                            if len(merged_df) > 5:  # Need at least a few data points
                                # Create scatter plot
                                plt.figure(figsize=(10, 6))
                                
                                plt.scatter(merged_df['sentiment_score'], merged_df['price'], alpha=0.7)
                                
                                # Add trend line
                                z = np.polyfit(merged_df['sentiment_score'], merged_df['price'], 1)
                                p = np.poly1d(z)
                                plt.plot(merged_df['sentiment_score'], p(merged_df['sentiment_score']), "r--")
                                
                                plt.title('Bitcoin Price vs. Sentiment Score')
                                plt.xlabel('Sentiment Score (0-1)')
                                plt.ylabel('Price (USD)')
                                plt.grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                fig_path = os.path.join(FIGURES_DIR, "sentiment_price_correlation.png")
                                plt.savefig(fig_path)
                                plt.close()
                                
                                self.add_to_report(f"![Sentiment vs. Price Correlation]({fig_path})\n")
                                self.figures.append(fig_path)
                                
                                # Calculate correlation
                                correlation = merged_df['sentiment_score'].corr(merged_df['price'])
                                
                                # Interpret correlation
                                if correlation > 0.7:
                                    correlation_desc = "strong positive"
                                elif correlation > 0.3:
                                    correlation_desc = "moderate positive"
                                elif correlation > 0:
                                    correlation_desc = "weak positive"
                                elif correlation > -0.3:
                                    correlation_desc = "weak negative"
                                elif correlation > -0.7:
                                    correlation_desc = "moderate negative"
                                else:
                                    correlation_desc = "strong negative"
                                
                                self.add_to_report(f"Analysis shows a **{correlation_desc}** correlation ({correlation:.2f}) between sentiment scores and Bitcoin price, suggesting that {'' if correlation > 0 else 'not '} sentiment tends to move with price.\n")
            except Exception as e:
                logger.error(f"Error generating sentiment vs. price correlation: {e}")
                self.add_to_report("*Unable to generate sentiment vs. price correlation due to data limitations.*\n")
            
            # Document risk vs. market activity
            self.add_to_report("### Document Risk vs. Market Activity\n")
            
            try:
                # Get document risk data
                doc_collection = self.client.collections.get("CryptoDueDiligenceDocuments")
                
                # Get document data with risk scores
                doc_data = doc_collection.query.fetch_objects(
                    return_properties=["document_type", "extracted_risk_score"],
                    limit=1000
                )
                
                # Get market volatility
                market_collection = self.client.collections.get("MarketMetrics")
                
                # Get price change data
                volatility_data = market_collection.query.fetch_objects(
                    return_properties=["symbol", "price_change_24h"],
                    limit=1000
                )
                
                # If we have both types of data, try to analyze them
                if doc_data.objects and volatility_data.objects:
                    # Calculate average risk score
                    risk_scores = [obj.properties.get("extracted_risk_score", 0) for obj in doc_data.objects 
                                  if obj.properties and "extracted_risk_score" in obj.properties]
                    
                    if risk_scores:
                        avg_risk = sum(risk_scores) / len(risk_scores)
                        
                        # Calculate average volatility
                        volatilities = [abs(obj.properties.get("price_change_24h", 0)) for obj in volatility_data.objects
                                      if obj.properties and "price_change_24h" in obj.properties]
                        
                        if volatilities:
                            avg_volatility = sum(volatilities) / len(volatilities)
                            
                            self.add_to_report(f"The average document risk score is **{avg_risk:.1f}/100**, while market volatility (average absolute price change) is **{avg_volatility:.1f}%**.\n")
                            
                            # Classify market conditions based on volatility
                            if avg_volatility > 5:
                                market_condition = "highly volatile"
                            elif avg_volatility > 2:
                                market_condition = "moderately volatile"
                            else:
                                market_condition = "relatively stable"
                            
                            self.add_to_report(f"The current market appears to be **{market_condition}**.")
                            
                            # Relate document risk to market conditions
                            if avg_risk > 70 and avg_volatility > 3:
                                self.add_to_report("The high document risk scores align with the current market volatility, suggesting increased investor caution during turbulent market conditions.")
                            elif avg_risk > 70 and avg_volatility <= 3:
                                self.add_to_report("Despite relatively stable market conditions, document risk scores remain high, potentially indicating underlying concerns not yet reflected in market prices.")
                            elif avg_risk <= 30 and avg_volatility > 3:
                                self.add_to_report("Despite significant market volatility, document risk assessment remains low, suggesting confidence in long-term fundamentals.")
                            elif avg_risk <= 30 and avg_volatility <= 3:
                                self.add_to_report("Both document risk assessment and market volatility are low, indicating a period of stability and confidence.")
                            else:
                                self.add_to_report("Document risk scores and market volatility show moderate levels, reflecting balanced market sentiment.")
            except Exception as e:
                logger.error(f"Error generating document risk vs. market activity analysis: {e}")
                self.add_to_report("*Unable to generate document risk vs. market activity analysis due to data limitations.*\n")
            
            self.add_to_report("")
            
            # OnChain Activity vs. News Volume
            self.add_to_report("### OnChain Activity vs. News Volume\n")
            
            try:
                # Placeholder for more complex cross-collection analysis
                self.add_to_report("*This analysis would explore the relationship between on-chain transaction activity and news coverage volume/sentiment, but requires more data for meaningful insights.*\n")
            except Exception as e:
                logger.error(f"Error generating cross-collection insights: {e}")
                self.add_to_report(f"*Error generating cross-collection insights: {e}*\n")
        
        except Exception as e:
            logger.error(f"Error generating cross-collection insights: {e}")
            self.add_to_report(f"*Error generating cross-collection insights: {e}*\n")
    
    def assess_data_quality(self):
        """Assess data quality across collections"""
        self.add_to_report("## Data Quality Assessment\n")
        
        try:
            # Get collection statistics
            collection_stats = {}
            
            for collection_name in [
                "CryptoDueDiligenceDocuments", 
                "CryptoNewsSentiment",
                "MarketMetrics", 
                "CryptoTimeSeries",
                "OnChainAnalytics"
            ]:
                try:
                    collection = self.client.collections.get(collection_name)
                    count_result = collection.aggregate.over_all(total_count=True)
                    collection_stats[collection_name] = count_result.total_count
                except Exception as e:
                    logger.warning(f"Could not get stats for {collection_name}: {e}")
                    collection_stats[collection_name] = 0
            
            # Data completeness
            self.add_to_report("### Data Completeness\n")
            
            completeness_scores = {}
            
            # Assign completeness scores based on record counts
            for collection, count in collection_stats.items():
                if collection == "CryptoTimeSeries":
                    # For time series, we expect thousands of records
                    if count > 10000:
                        score = "Excellent"
                    elif count > 5000:
                        score = "Good"
                    elif count > 1000:
                        score = "Fair"
                    else:
                        score = "Poor"
                elif collection == "CryptoNewsSentiment":
                    # For news sentiment, we expect hundreds of records
                    if count > 500:
                        score = "Excellent"
                    elif count > 100:
                        score = "Good"
                    elif count > 50:
                        score = "Fair"
                    else:
                        score = "Poor"
                elif collection == "MarketMetrics":
                    # For market metrics, expect dozens of records
                    if count > 100:
                        score = "Excellent"
                    elif count > 50:
                        score = "Good"
                    elif count > 20:
                        score = "Fair"
                    else:
                        score = "Poor"
                elif collection == "CryptoDueDiligenceDocuments":
                    # For documents, expect dozens of records
                    if count > 100:
                        score = "Excellent"
                    elif count > 50:
                        score = "Good"
                    elif count > 20:
                        score = "Fair"
                    else:
                        score = "Poor"
                elif collection == "OnChainAnalytics":
                    # For on-chain, even a few records is good
                    if count > 10:
                        score = "Excellent"
                    elif count > 5:
                        score = "Good"
                    elif count > 1:
                        score = "Fair"
                    else:
                        score = "Poor"
                
                completeness_scores[collection] = score
            
            # Create completeness table
            table_rows = ["| Collection | Record Count | Completeness |"]
            table_rows.append("| ---------- | ------------ | ------------ |")
            
            for collection, count in collection_stats.items():
                score = completeness_scores.get(collection, "Unknown")
                table_rows.append(f"| {collection} | {count:,} | {score} |")
            
            self.add_to_report("\n".join(table_rows) + "\n")
            
            # Data quality issues
            self.add_to_report("### Data Quality Issues\n")
            
            quality_issues = []
            
            # Check for potential quality issues
            try:
                # Check time series data
                if collection_stats.get("CryptoTimeSeries", 0) > 0:
                    collection = self.client.collections.get("CryptoTimeSeries")
                    
                    # Check for duplicate timestamps
                    try:
                        # Group by symbol and timestamp to see if there are duplicates
                        timestamp_counts = collection.aggregate.over_all(
                            group_by=["symbol", "timestamp"],
                            total_count=True
                        )
                        
                        duplicate_count = sum(1 for group in timestamp_counts.groups if group.total_count > 1)
                        
                        if duplicate_count > 0:
                            quality_issues.append(f"- Found {duplicate_count} potential duplicate timestamp entries in time series data")
                    except Exception as e:
                        logger.warning(f"Could not check for duplicate timestamps: {e}")
                
                # Check market data currency
                if collection_stats.get("MarketMetrics", 0) > 0:
                    collection = self.client.collections.get("MarketMetrics")
                    
                    # Get most recent data point
                    latest_data = collection.query.fetch_objects(
                        return_properties=["timestamp"],
                        limit=1,
                        sort=Sort.by_property("timestamp", ascending=False)
                    )
                    
                    if latest_data.objects and latest_data.objects[0].properties:
                        latest_timestamp = latest_data.objects[0].properties.get("timestamp", "")
                        
                        if latest_timestamp:
                            try:
                                latest_date = pd.to_datetime(latest_timestamp)
                                days_old = (datetime.now() - latest_date).days
                                
                                if days_old > 7:
                                    quality_issues.append(f"- Market data may be stale: most recent data is {days_old} days old")
                            except:
                                quality_issues.append("- Could not determine age of market data")
                
                # Check sentiment data coverage
                if collection_stats.get("CryptoNewsSentiment", 0) > 0:
                    collection = self.client.collections.get("CryptoNewsSentiment")
                    
                    # Check sentiment distribution
                    sentiments = collection.query.fetch_objects(
                        return_properties=["sentiment_label"],
                        limit=1000
                    )
                    
                    if sentiments.objects:
                        sentiment_counts = Counter([obj.properties.get("sentiment_label", "UNKNOWN") for obj in sentiments.objects
                                                  if obj.properties and "sentiment_label" in obj.properties])
                        
                        total = sum(sentiment_counts.values())
                        
                        if total > 0:
                            # Check for skewed sentiment distribution
                            neutral_pct = sentiment_counts.get("NEUTRAL", 0) / total * 100
                            
                            if neutral_pct > 80:
                                quality_issues.append(f"- Sentiment data may be biased: {neutral_pct:.1f}% of articles are classified as neutral")
            except Exception as e:
                logger.error(f"Error checking data quality: {e}")
            
            # Report quality issues
            if quality_issues:
                self.add_to_report("The following potential data quality issues were identified:\n")
                
                for issue in quality_issues:
                    self.add_to_report(issue)
                
                self.add_to_report("")
            else:
                self.add_to_report("No significant data quality issues were identified.\n")
            
            # Data coverage
            self.add_to_report("### Data Coverage\n")
            
            # Check time range for time series data
            try:
                if collection_stats.get("CryptoTimeSeries", 0) > 0:
                    collection = self.client.collections.get("CryptoTimeSeries")
                    
                    # Get earliest and latest dates
                    earliest_data = collection.query.fetch_objects(
                        return_properties=["timestamp"],
                        limit=1,
                        sort=Sort.by_property("timestamp", ascending=True)
                    )
                    
                    latest_data = collection.query.fetch_objects(
                        return_properties=["timestamp"],
                        limit=1,
                        sort=Sort.by_property("timestamp", ascending=False)
                    )
                    
                    if earliest_data.objects and latest_data.objects:
                        earliest_timestamp = earliest_data.objects[0].properties.get("timestamp", "")
                        latest_timestamp = latest_data.objects[0].properties.get("timestamp", "")
                        
                        if earliest_timestamp and latest_timestamp:
                            try:
                                earliest_date = pd.to_datetime(earliest_timestamp)
                                latest_date = pd.to_datetime(latest_timestamp)
                                
                                date_range = (latest_date - earliest_date).days
                                
                                self.add_to_report(f"Time series data spans approximately {date_range} days, from {earliest_date.date()} to {latest_date.date()}.\n")
                            except:
                                pass
            except Exception as e:
                logger.error(f"Error checking time series coverage: {e}")
            
            # Symbol coverage
            try:
                if collection_stats.get("CryptoTimeSeries", 0) > 0:
                    collection = self.client.collections.get("CryptoTimeSeries")
                    
                    # Get symbol counts
                    symbols_result = collection.aggregate.over_all(
                        group_by="symbol",
                        total_count=True
                    )
                    
                    if symbols_result.groups:
                        symbol_count = len(symbols_result.groups)
                        self.add_to_report(f"Data covers {symbol_count} different cryptocurrency symbols.\n")
            except Exception as e:
                logger.error(f"Error checking symbol coverage: {e}")
        
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            self.add_to_report(f"*Error assessing data quality: {e}*\n")
    
    def generate_recommendations(self):
        """Generate recommendations based on data analysis"""
        self.add_to_report("## Recommendations\n")
        
        try:
            # Data collection recommendations
            self.add_to_report("### Data Collection Recommendations\n")
            
            # Collection-specific recommendations
            collection_stats = {}
            
            for collection_name in [
                "CryptoDueDiligenceDocuments", 
                "CryptoNewsSentiment",
                "MarketMetrics", 
                "CryptoTimeSeries",
                "OnChainAnalytics"
            ]:
                try:
                    collection = self.client.collections.get(collection_name)
                    count_result = collection.aggregate.over_all(total_count=True)
                    collection_stats[collection_name] = count_result.total_count
                except Exception as e:
                    logger.warning(f"Could not get stats for {collection_name}: {e}")
                    collection_stats[collection_name] = 0
            
            # Time series recommendations
            if collection_stats.get("CryptoTimeSeries", 0) < 10000:
                self.add_to_report("- **Time Series Data**: Consider increasing historical data coverage by loading more CSV files or connecting to additional exchange APIs to fetch longer time periods. Comprehensive historical data improves forecasting model accuracy.")
            
            # News sentiment recommendations
            if collection_stats.get("CryptoNewsSentiment", 0) < 500:
                self.add_to_report("- **News Sentiment**: Expand news sources coverage to include additional reputable crypto news outlets. Consider implementing real-time sentiment analysis to capture market reactions to breaking news.")
            
            # Market metrics recommendations
            if collection_stats.get("MarketMetrics", 0) < 100:
                self.add_to_report("- **Market Data**: Increase the frequency of market data collection, especially during high volatility periods. Consider adding additional market indicators such as order book depth, trading volume by exchange, and liquidity metrics.")
            
            # Document recommendations
            if collection_stats.get("CryptoDueDiligenceDocuments", 0) < 50:
                self.add_to_report("- **Due Diligence Documents**: Expand the document collection to include more whitepapers, regulatory filings, and audit reports. Consider implementing automated document classification to better organize the repository.")
            
            # On-chain recommendations
            if collection_stats.get("OnChainAnalytics", 0) < 20:
                self.add_to_report("- **On-Chain Data**: Analyze more wallet addresses, especially those belonging to major players in the ecosystem. Consider implementing cross-chain analysis to track fund flows between different blockchains.")
            
            # General data quality recommendations
            self.add_to_report("\n### Data Quality Recommendations\n")
            
            # Check for potential gaps in time series data
            self.add_to_report("- **Data Consistency**: Implement automated checks for gaps in time series data, especially during weekend periods or high volatility events.")
            
            # Recommend regular backups
            self.add_to_report("- **Data Backup**: Establish a regular backup schedule for all collections to prevent data loss.")
            
            # Recommend data normalization
            self.add_to_report("- **Data Standardization**: Standardize naming conventions and data formats across all collections to improve cross-collection analytics.")
            
            # Analysis recommendations
            self.add_to_report("\n### Analysis Enhancement Recommendations\n")
            
            # Advanced analytics recommendations
            self.add_to_report("- **Advanced Analytics**: Implement machine learning models for anomaly detection in both price movements and on-chain transactions.")
            
            # Correlation analysis
            self.add_to_report("- **Correlation Analysis**: Expand cross-collection insights by systematically analyzing correlations between news sentiment, market movements, and on-chain activities.")
            
            # Visualization improvements
            self.add_to_report("- **Interactive Visualizations**: Develop interactive dashboards for real-time monitoring of key metrics across all data collections.")
            
            # API integration
            self.add_to_report("- **External API Integration**: Connect with additional data sources such as social media sentiment APIs, macroeconomic indicators, and regulatory news feeds.")
            
            # Alert system
            self.add_to_report("- **Alert System**: Implement an automated alert system for significant changes in sentiment, unusual on-chain transactions, or price volatility thresholds.")
            
            # Report automation
            self.add_to_report("\n### Reporting Recommendations\n")
            
            # Regular report generation
            self.add_to_report("- **Automated Reporting**: Schedule regular automated report generation for daily, weekly, and monthly summaries.")
            
            # Custom report templates
            self.add_to_report("- **Custom Templates**: Develop specific report templates for different stakeholders (investors, analysts, compliance officers) with relevant focus areas.")
            
            # Export formats
            self.add_to_report("- **Multiple Export Formats**: Add support for exporting reports in various formats including PDF, interactive HTML, and presentation slides.")
            
            self.add_to_report("\n")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            self.add_to_report(f"*Error generating recommendations: {e}*\n")
    
    def save_report(self):
        """Save the generated report to files"""
        try:
            # Save Markdown file with UTF-8 encoding
            with open(MARKDOWN_FILE, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.report_md))
            logger.info(f"Saved Markdown report to {MARKDOWN_FILE}")
            
            # Convert to HTML
            try:
                import markdown
                html_content = markdown.markdown("\n".join(self.report_md), extensions=['tables', 'fenced_code'])
                
                # Add HTML header with basic styling
                html_full = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Crypto Due Diligence Data Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #3498db; border-bottom: 1px solid #3498db; padding-bottom: 5px; }}
                        h3 {{ color: #2980b9; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        img {{ max-width: 100%; height: auto; }}
                        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
                        pre {{ background: #f4f4f4; padding: 10px; border-radius: 3px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    {html_content}
                </body>
                </html>
                """
                
                with open(HTML_FILE, 'w', encoding='utf-8') as f:
                    f.write(html_full)
                logger.info(f"Saved HTML report to {HTML_FILE}")
                
            except ImportError:
                logger.warning("Could not convert to HTML: markdown package not installed")
                
                # Simple HTML conversion as fallback
                html_content = "\n".join(self.report_md).replace("\n", "<br>")
                with open(HTML_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"<html><body>{html_content}</body></html>")
                logger.info(f"Saved basic HTML report to {HTML_FILE}")
                
        except Exception as e:
            logger.error(f"Error saving report: {e}")