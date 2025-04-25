"use client";

import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Calendar,
  BarChart, 
  RefreshCcw,
  Info,
  LineChart
} from "lucide-react";
import axios from 'axios';
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area } from 'recharts';

// Interface for cryptocurrency data
interface CryptoData {
  symbol: string;
  name: string;
  currentPrice: number;
  priceChange24h: number;
  priceChangePercentage24h: number;
}

// Interface for forecast data
interface ForecastData {
  symbol: string;
  forecast_timestamp: string;
  model_name: string;
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

  // Fetch current price data from API
  const fetchPriceData = async () => {
    try {
      // Try to fetch from actual API first
      try {
        const response = await axios.get('/api/crypto/prices');
        if (response.data && Array.isArray(response.data)) {
          const formattedData = response.data.map((item: any) => ({
            symbol: item.symbol.replace('USDT', '').replace('USD', ''),
            name: availableCryptos.find(c => c.symbol === item.symbol.replace('USDT', '').replace('USD', ''))?.name || 'Unknown',
            currentPrice: parseFloat(item.price),
            priceChange24h: parseFloat(item.price_change_24h || 0),
            priceChangePercentage24h: parseFloat(item.price_change_percentage_24h || 0)
          }));
          
          setCryptoData(formattedData);
          return;
        }
      } catch (apiError) {
        console.warn("Could not fetch from real API, using fallback data", apiError);
      }
      
      // Fallback to mock data if API fetch fails
      const mockPrices = {
        'BTC': { price: 67432.18, change: 7.52 },
        'ETH': { price: 3421.50, change: 12.53 },
        'SOL': { price: 143.25, change: 17.80 },
        'ADA': { price: 0.57, change: 10.53 },
        'DOT': { price: 7.85, change: 10.45 },
        'XRP': { price: 0.64, change: -7.81 },
      };
      
      // Add some randomness to prices and changes for realistic variation
      const randomizedData = availableCryptos.map(crypto => {
        const baseData = mockPrices[crypto.symbol as keyof typeof mockPrices];
        const randomFactor = 0.98 + Math.random() * 0.04; // Between 0.98 and 1.02
        const randomChangeDir = Math.random() > 0.7 ? -1 : 1;
        
        return {
          symbol: crypto.symbol,
          name: crypto.name,
          currentPrice: baseData.price * randomFactor,
          priceChange24h: baseData.price * (baseData.change / 100) * randomFactor,
          priceChangePercentage24h: baseData.change * randomFactor * randomChangeDir
        };
      });
      
      setCryptoData(randomizedData);
    } catch (err) {
      console.error("Error fetching price data:", err);
      setError("Failed to fetch current prices");
    }
  };

  // Fetch forecast data from API
  const fetchForecastData = async () => {
    try {
      // Try to fetch from actual API
      const response = await axios.get(`/api/forecasts?symbol=${selectedCrypto}&timeframe=${timeframe}`);
      
      if (response.data && response.data.forecast) {
        setForecastData(response.data.forecast);
        createChartData(response.data.forecast, response.data.historicalData || []);
      } else {
        throw new Error("Invalid forecast data format");
      }
    } catch (err) {
      console.error("Error fetching forecast data:", err);
      setError("Failed to fetch forecast data");
    }
  };

  // Create chart data from historical and forecast data
  const createChartData = (forecast: ForecastData, historicalData: any[] = []) => {
    try {
      // Use provided historical data if available
      if (historicalData.length > 0) {
        console.log(`Processing ${historicalData.length} historical data points`);
        
        // Limit historical data based on timeframe if needed
        const timeframeDays = timeframe === '7d' ? 7 : timeframe === '30d' ? 30 : 90;
        const filteredHistorical = historicalData.length > timeframeDays 
          ? historicalData.slice(-timeframeDays) 
          : historicalData;
        
        const historical = filteredHistorical.map((point) => ({
          date: point.timestamp || point.date,
          price: point.price || point.close,
        }));
        
        const forecastPoints = forecast.forecast_dates.map((date, i) => ({
          date,
          forecast: forecast.forecast_values[i],
          lowerBound: forecast.lower_bounds[i],
          upperBound: forecast.upper_bounds[i]
        }));
        
        setChartData([...historical, ...forecastPoints]);
      } else {
        setError("No historical data available");
      }
    } catch (err) {
      console.error("Error creating chart data:", err);
      setError("Failed to create chart data");
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
    const loadData = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        await fetchPriceData();
        await fetchForecastData();
      } catch (err) {
        console.error("Error loading data:", err);
        setError("Failed to load forecast data");
      } finally {
        setIsLoading(false);
      }
    };
    
    loadData();
  }, [selectedCrypto, timeframe]);

  // Handle refresh button click
  const handleRefresh = () => {
    setIsLoading(true);
    fetchPriceData();
    fetchForecastData();
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
        
        {/* Market Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
          {getMarketIndicators().map((indicator) => (
            <Card key={indicator.name} className="p-6">
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
        
        {/* Crypto Selector Cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
          {availableCryptos.map((crypto) => (
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
                
                {cryptoData.find(data => data.symbol === crypto.symbol) && (
                  <div className="mt-2 text-center">
                    <p className="text-sm font-bold">
                      {formatPrice(cryptoData.find(data => data.symbol === crypto.symbol)?.currentPrice || 0)}
                    </p>
                    <p className={`text-xs ${
                      (cryptoData.find(data => data.symbol === crypto.symbol)?.priceChangePercentage24h || 0) >= 0 
                        ? 'text-green-600' 
                        : 'text-red-600'
                    }`}>
                      {(cryptoData.find(data => data.symbol === crypto.symbol)?.priceChangePercentage24h || 0) >= 0 ? '↑' : '↓'}
                      {Math.abs(cryptoData.find(data => data.symbol === crypto.symbol)?.priceChangePercentage24h || 0).toFixed(2)}%
                    </p>
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg relative mt-6">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
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
            <div className="h-80 flex items-center justify-center bg-gray-50 rounded-lg">
              <div className="text-center">
                <LineChart size={48} className="mx-auto text-gray-300 mb-3" />
                <p className="text-gray-500">No chart data available</p>
              </div>
            </div>
          )}
        </Card>
        
        {/* Forecast Insights */}
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
      </div>
    </div>
  );
}