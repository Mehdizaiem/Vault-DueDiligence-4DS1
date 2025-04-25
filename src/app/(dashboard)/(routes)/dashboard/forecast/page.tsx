"use client";

import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Calendar,
  BarChart, 
  RefreshCcw,
  Info,
  LineChart,
  FileQuestion,
  Terminal
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area } from 'recharts';

// Interface for cryptocurrency data
interface CryptoData {
  symbol: string;
  price: number;
  price_change_24h?: number;
  price_change_percentage_24h?: number;
}

// Interface for forecast data
interface ForecastData {
  symbol: string;
  forecast_timestamp: string;
  model_name: string;
  model_type?: string;
  days_ahead: number;
  current_price: number;
  forecast_dates: string[];
  forecast_values: number[];
  lower_bounds: number[];
  upper_bounds: number[];
  final_forecast: number;
  change_pct: number;
  trend: string;
  probability_increase: number;
  average_uncertainty: number;
  insight: string;
}

// Interface for combined historical and forecast data for chart
interface ChartData {
  date: string;
  price?: number;
  forecast?: number;
  lowerBound?: number;
  upperBound?: number;
}

export default function ForecastPage() {
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [timeframe, setTimeframe] = useState('30d');
  const [isLoading, setIsLoading] = useState(true);
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Available cryptocurrencies
  const availableCryptos = [
    { symbol: 'BTC', name: 'Bitcoin' },
    { symbol: 'ETH', name: 'Ethereum' },
    { symbol: 'SOL', name: 'Solana' },
    { symbol: 'ADA', name: 'Cardano' },
    { symbol: 'DOT', name: 'Polkadot' },
    { symbol: 'XRP', name: 'XRP' },
  ];

  // Market indicators based on forecast data
  const getMarketIndicators = () => [
    { 
      name: 'Market Sentiment', 
      value: forecastData?.trend?.includes('bullish') ? 'Bullish' : 
             forecastData?.trend?.includes('bearish') ? 'Bearish' : 'Neutral',
      status: forecastData?.trend?.includes('bullish') ? 'positive' : 
              forecastData?.trend?.includes('bearish') ? 'negative' : 'neutral' 
    },
    { 
      name: 'Forecast Confidence', 
      value: `${Math.round(forecastData?.probability_increase || 50)}%`,
      status: (forecastData?.probability_increase || 50) > 65 ? 'positive' : 
              (forecastData?.probability_increase || 50) < 35 ? 'negative' : 'neutral'
    },
    { 
      name: 'Volatility Outlook', 
      value: (forecastData?.average_uncertainty || 0) < 10 ? 'Low' :
             (forecastData?.average_uncertainty || 0) < 20 ? 'Moderate' : 'High',
      status: (forecastData?.average_uncertainty || 0) < 10 ? 'positive' : 
              (forecastData?.average_uncertainty || 0) < 20 ? 'neutral' : 'negative'
    },
    { 
      name: 'Price Projection', 
      value: (forecastData?.change_pct || 0) > 0 ? `+${forecastData?.change_pct.toFixed(1)}%` : 
             `${forecastData?.change_pct.toFixed(1)}%`,
      status: (forecastData?.change_pct || 0) > 5 ? 'positive' : 
              (forecastData?.change_pct || 0) < -5 ? 'negative' : 'neutral'
    }
  ];

  // Fetch crypto data from API (prices, forecast, and historical data)
  const fetchCryptoData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/forecasts?symbol=${selectedCrypto}&timeframe=${timeframe}`);
      
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      
      const data = await response.json();
      
      // Process prices
      if (data.prices && Array.isArray(data.prices) && data.prices.length > 0) {
        setCryptoData(data.prices);
      } else {
        // Set placeholder price data if none is available
        const placeholderPrices = availableCryptos.map(crypto => ({
          symbol: `${crypto.symbol}USD`,
          price: crypto.symbol === 'BTC' ? 67500 : 
                  crypto.symbol === 'ETH' ? 3450 : 
                  crypto.symbol === 'SOL' ? 142 : 
                  crypto.symbol === 'ADA' ? 0.57 : 
                  crypto.symbol === 'DOT' ? 7.85 : 0.64,
          price_change_percentage_24h: 0
        }));
        
        setCryptoData(placeholderPrices);
        console.warn("No price data available, using placeholder values");
      }
      
      // Process forecast data
      if (data.forecast) {
        setForecastData(data.forecast);
        createChartData(data.forecast, data.historicalData || []);
      } else {
        setForecastData(null);
        setChartData([]);
        setError(data.error || "No forecast data available");
      }
      
    } catch (err) {
      console.error("Error fetching crypto data:", err);
      setError("Failed to load cryptocurrency data. Please try again later.");
      // Keep any existing crypto price data to avoid completely empty UI
    } finally {
      setIsLoading(false);
    }
  };

  // Create chart data from historical and forecast data
  const createChartData = (forecast: ForecastData, historicalData: any[] = []) => {
    try {
      if (historicalData.length > 0) {
        // Process historical data
        const historical = historicalData.map((point) => ({
          date: point.timestamp || point.date,
          price: point.price || point.close,
        }));
        
        // Process forecast data
        const forecastPoints = forecast.forecast_dates.map((date, i) => ({
          date,
          forecast: forecast.forecast_values[i],
          lowerBound: forecast.lower_bounds[i],
          upperBound: forecast.upper_bounds[i]
        }));
        
        // Combine both datasets
        setChartData([...historical, ...forecastPoints]);
      } else {
        // Create chart data with only forecast
        const forecastPoints = forecast.forecast_dates.map((date, i) => ({
          date,
          forecast: forecast.forecast_values[i],
          lowerBound: forecast.lower_bounds[i],
          upperBound: forecast.upper_bounds[i]
        }));
        
        // Add current price as the first point
        const currentDate = new Date();
        forecastPoints.unshift({
          date: currentDate.toISOString(),
          price: forecast.current_price
        });
        
        setChartData(forecastPoints);
      }
    } catch (err) {
      console.error("Error creating chart data:", err);
      setChartData([]);
      setError("Failed to process chart data");
    }
  };

  // Format price for display
  const formatPrice = (price: number) => {
    if (price >= 1000) {
      return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    } else if (price >= 1) {
      return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else {
      return `$${price.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 6 })}`;
    }
  };

  // Load data when component mounts or when selectedCrypto/timeframe changes
  useEffect(() => {
    fetchCryptoData();
  }, [selectedCrypto, timeframe]);

  // Handle refresh button click
  const handleRefresh = () => {
    fetchCryptoData();
  };

  // Find price data for a specific crypto
  const findCryptoPrice = (symbol: string) => {
    return cryptoData.find(data => 
      data.symbol === `${symbol}USD` || 
      data.symbol === `${symbol}USDT` ||
      data.symbol.replace('USD', '').replace('USDT', '') === symbol
    );
  };

  // Custom tooltip for chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const date = new Date(label).toLocaleDateString();
      return (
        <div className="bg-white p-3 border shadow-md rounded-md">
          <p className="font-medium text-sm mb-1">{date}</p>
          {payload.map((entry: any, index: number) => {
            if (entry.value !== null && entry.value !== undefined) {
              return (
                <p key={index} style={{ color: entry.color }} className="text-sm">
                  {entry.name}: {formatPrice(entry.value)}
                </p>
              );
            }
            return null;
          })}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto Forecast</h2>
            <p className="text-muted-foreground">
              Price predictions and market analysis for major cryptocurrencies
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Timeframe selector */}
            <div className="flex border rounded-lg overflow-hidden">
              {['7d', '30d', '90d'].map((time) => (
                <button
                  key={time}
                  onClick={() => setTimeframe(time)}
                  className={`px-4 py-2 text-sm font-medium ${
                    timeframe === time
                      ? 'bg-black text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {time}
                </button>
              ))}
            </div>
            
            <button 
              className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              <RefreshCcw size={16} className={isLoading ? 'animate-spin' : ''} />
              <span>{isLoading ? 'Loading...' : 'Refresh'}</span>
            </button>
          </div>
        </div>
        
        {/* Error message if any */}
        {error && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg relative mt-4">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {/* Market Indicators - only show if forecast data is available */}
        {forecastData && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
            {getMarketIndicators().map((indicator) => (
              <Card 
                key={indicator.name} 
                className="p-6"
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm text-gray-500">{indicator.name}</p>
                    <h3 className="text-2xl font-bold mt-1">{indicator.value}</h3>
                  </div>
                  <div className={`rounded-full p-2 ${
                    indicator.status === 'positive' ? 'bg-green-100' : 
                    indicator.status === 'negative' ? 'bg-red-100' : 'bg-gray-100'
                  }`}>
                    {indicator.status === 'positive' ? (
                      <TrendingUp className="h-5 w-5 text-green-600" />
                    ) : indicator.status === 'negative' ? (
                      <TrendingDown className="h-5 w-5 text-red-600" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-gray-600" />
                    )}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
        
        {/* Crypto Selector Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
          {availableCryptos.map((crypto) => {
            const priceData = findCryptoPrice(crypto.symbol);
            return (
              <Card 
                key={crypto.symbol} 
                className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedCrypto === crypto.symbol ? 'border-2 border-blue-500 bg-blue-50' : ''
                }`}
                onClick={() => setSelectedCrypto(crypto.symbol)}
              >
                <div className="flex flex-col items-center justify-center">
                  <div className="h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center mb-2">
                    <span className="font-bold text-sm">{crypto.symbol}</span>
                  </div>
                  <p className="text-sm font-medium">{crypto.name}</p>
                  
                  {priceData && (
                    <div className="mt-2 text-center">
                      <p className="text-sm font-bold">
                        {formatPrice(priceData.price)}
                      </p>
                      <p className={`text-xs ${
                        (priceData.price_change_percentage_24h || 0) >= 0 
                          ? 'text-green-600' 
                          : 'text-red-600'
                      }`}>
                        {(priceData.price_change_percentage_24h || 0) >= 0 ? '↑' : '↓'}
                        {Math.abs(priceData.price_change_percentage_24h || 0).toFixed(2)}%
                      </p>
                    </div>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
        
        {/* Price Chart */}
        <Card className="p-6 mt-6">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-lg font-semibold">Price Forecast</h3>
              <p className="text-sm text-gray-500">Historical data and price prediction</p>
            </div>
            <div className="flex gap-3">
              <button className="flex items-center gap-2 px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-lg text-sm hover:bg-indigo-100">
                <LineChart size={16} />
                <span>Line</span>
              </button>
              <button className="flex items-center gap-2 px-3 py-1.5 bg-white border text-gray-600 rounded-lg text-sm hover:bg-gray-50">
                <BarChart size={16} />
                <span>Bar</span>
              </button>
              <button className="flex items-center gap-2 px-3 py-1.5 bg-white border text-gray-600 rounded-lg text-sm hover:bg-gray-50">
                <Calendar size={16} />
                <span>Calendar</span>
              </button>
            </div>
          </div>
          
          {/* Chart */}
          {isLoading ? (
            <div className="h-80 flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
            </div>
          ) : chartData.length > 0 ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis 
                    dataKey="date" 
                    tickFormatter={(date) => new Date(date).toLocaleDateString()} 
                    minTickGap={30}
                  />
                  <YAxis 
                    domain={['auto', 'auto']}
                    tickFormatter={(value) => formatPrice(value)}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  
                  {/* Confidence interval area */}
                  <Area 
                    type="monotone"
                    dataKey="upperBound"
                    stroke="none"
                    fillOpacity={0.1}
                    fill="#16a34a"
                    name="Confidence Range"
                  />
                  <Area 
                    type="monotone"
                    dataKey="lowerBound"
                    stroke="none"
                    fillOpacity={0}
                    fill="#16a34a"
                    name="Confidence Range"
                  />
                  
                  {/* Historical price line */}
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#4f46e5" 
                    strokeWidth={2} 
                    dot={false} 
                    name="Historical Price" 
                    connectNulls
                  />
                  
                  {/* Forecast line */}
                  <Line 
                    type="monotone" 
                    dataKey="forecast" 
                    stroke="#16a34a" 
                    strokeWidth={2} 
                    strokeDasharray="5 5" 
                    dot={true} 
                    name="Forecast"
                    connectNulls
                  />
                  
                  {/* Reference line for today */}
                  <ReferenceLine 
                    x={new Date().toISOString()} 
                    stroke="#475569" 
                    strokeWidth={1.5} 
                    strokeDasharray="3 3"
                    label={{ 
                      value: 'Today', 
                      position: 'top', 
                      fill: '#475569',
                      fontSize: 12
                    }} 
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-80 flex flex-col items-center justify-center bg-gray-50 rounded-lg p-6">
              <FileQuestion size={64} className="text-gray-300 mb-4" />
              <h4 className="text-lg font-medium text-gray-600 mb-2">No Forecast Data Available</h4>
              <p className="text-gray-500 text-center mb-4">
                {error || `We couldn't find forecast data for ${selectedCrypto}. This might be because:`}
              </p>
              {!error && (
                <ul className="text-gray-500 text-sm list-disc pl-6 space-y-1">
                  <li>The forecast hasn't been generated yet</li>
                  <li>Historical data is missing for this cryptocurrency</li>
                  <li>The forecast service is currently unavailable</li>
                </ul>
              )}
              <button 
                onClick={handleRefresh}
                className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
              >
                Retry
              </button>
            </div>
          )}
        </Card>
        
        {/* Forecast Insights - only show if forecast data is available */}
        {forecastData && (
          <Card className="p-6 mt-6">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <h3 className="text-lg font-semibold">Forecast Insight</h3>
                  <Info size={16} className="text-gray-400" />
                </div>
                <p className="text-gray-600 mb-4">{forecastData.insight}</p>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Current Price</p>
                    <p className="text-xl font-bold">{formatPrice(forecastData.current_price)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Forecasted Price</p>
                    <p className="text-xl font-bold">{formatPrice(forecastData.final_forecast)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Predicted Change</p>
                    <p className={`text-xl font-bold ${forecastData.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {forecastData.change_pct >= 0 ? '+' : ''}{forecastData.change_pct.toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">Forecast Horizon</p>
                    <p className="text-xl font-bold">{forecastData.days_ahead} days</p>
                  </div>
                </div>
              </div>
              
              <div className="flex-1">
                <h3 className="text-lg font-semibold mb-3">Forecast Metrics</h3>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-500">Trend</span>
                      <span className={`text-sm font-medium ${
                        forecastData.trend.includes('bullish') ? 'text-green-600' : 
                        forecastData.trend.includes('bearish') ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        {forecastData.trend.charAt(0).toUpperCase() + forecastData.trend.slice(1)}
                      </span>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-500">Probability of Increase</span>
                      <span className="text-sm font-medium">{forecastData.probability_increase.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          forecastData.probability_increase >= 60 ? 'bg-green-500' : 
                          forecastData.probability_increase >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${forecastData.probability_increase}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-500">Uncertainty</span>
                      <span className="text-sm font-medium">±{forecastData.average_uncertainty.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          forecastData.average_uncertainty <= 10 ? 'bg-green-500' : 
                          forecastData.average_uncertainty <= 20 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${100 - Math.min(forecastData.average_uncertainty, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-500">Model</span>
                      <span className="text-sm font-medium">{forecastData.model_name}</span>
                    </div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm text-gray-500">Last Updated</span>
                      <span className="text-sm font-medium">
                        {new Date(forecastData.forecast_timestamp).toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        )}
        
        {/* No forecast data available - show CTA to run forecasting */}
        {!isLoading && !forecastData && (
          <Card className="p-6 mt-6">
            <div className="flex flex-col items-center text-center p-6">
              <Terminal size={64} className="text-gray-300 mb-4" />
              <h3 className="text-xl font-semibold mb-2">Generate Forecasts</h3>
              <p className="text-gray-600 mb-6">
                No forecast data is currently available for {selectedCrypto}. You can generate forecasts using the chronos_finetune.py script.
              </p>
              <div className="bg-gray-50 p-4 rounded-lg text-left w-full max-w-lg mb-6">
                <p className="text-sm font-medium mb-2">How to generate forecasts:</p>
                <div className="bg-black text-white p-3 rounded-lg overflow-auto">
                  <pre className="text-xs">python models/chronos/chronos_finetune.py --symbol {selectedCrypto} --days-ahead 7</pre>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  This will fine-tune the Chronos model for {selectedCrypto} and generate forecasts.
                </p>
              </div>
              <p className="text-sm text-gray-500">
                After running the script, refresh this page to see the forecast.
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}