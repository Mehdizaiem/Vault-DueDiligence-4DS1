"use client";

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';
import { Loader2 } from 'lucide-react';

interface HistoricalData {
  date: string;
  price: number;
}

interface ForecastData {
  timestamp: string;
  predicted_price: number;
  confidence_interval_lower: number;
  confidence_interval_upper: number;
  model_name?: string;
  model_type?: string;
  days_ahead?: number;
  average_uncertainty?: number;
}

export default function ForecastPage() {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch historical data
        const historyResponse = await fetch('/api/historical?symbol=BTC_USD');
        if (!historyResponse.ok) {
          throw new Error('Failed to fetch historical data');
        }
        const historyData = await historyResponse.json();
        
        if (!historyData.success) {
          throw new Error(historyData.error || 'Failed to fetch historical data');
        }

        // Format historical data
        const formattedHistoricalData = historyData.data.map((item: any) => ({
          date: format(new Date(item.date), 'yyyy-MM-dd'),
          price: item.price
        }));

        setHistoricalData(formattedHistoricalData);

        // Fetch latest forecast
        try {
          const forecastResponse = await fetch('/api/forecast/latest?symbol=BTC_USD');
          if (!forecastResponse.ok) {
            throw new Error('Failed to fetch forecast data');
          }
          const forecastData = await forecastResponse.json();

          if (forecastData.success && forecastData.data && forecastData.data.length > 0) {
            // Format forecast data
            const formattedForecastData = forecastData.data.map((item: any) => ({
              timestamp: format(new Date(item.timestamp), 'yyyy-MM-dd'),
              predicted_price: item.predicted_price,
              confidence_interval_lower: item.confidence_interval_lower,
              confidence_interval_upper: item.confidence_interval_upper,
              model_name: item.model_name,
              model_type: item.model_type,
              days_ahead: item.days_ahead,
              average_uncertainty: item.average_uncertainty
            }));

            setForecastData(formattedForecastData);
          }
        } catch (forecastError) {
          console.error('Error fetching forecast:', forecastError);
          // Don't throw here, just continue with historical data
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-4rem)]">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Card className="p-4 border-red-500">
          <h2 className="text-lg font-semibold text-red-500">Error</h2>
          <p className="text-sm text-red-500">{error}</p>
        </Card>
      </div>
    );
  }

  if (historicalData.length === 0) {
    return (
      <div className="p-6">
        <Card className="p-4">
          <h2 className="text-lg font-semibold">No Data Available</h2>
          <p className="text-sm text-muted-foreground">No historical data is currently available.</p>
        </Card>
      </div>
    );
  }

  // Combine historical and forecast data for the chart
  const chartData = [
    ...historicalData.map(item => ({
      date: item.date,
      price: item.price,
      predicted: null,
      lower: null,
      upper: null
    })),
    ...(forecastData.length > 0 ? forecastData.map(item => ({
      date: item.timestamp,
      price: null,
      predicted: item.predicted_price,
      lower: item.confidence_interval_lower,
      upper: item.confidence_interval_upper
    })) : [])
  ].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

  // Calculate date range for x-axis
  const startDate = chartData.length > 0 ? new Date(chartData[0].date) : new Date();
  const endDate = chartData.length > 0 ? new Date(chartData[chartData.length - 1].date) : new Date();
  const monthDiff = (endDate.getFullYear() - startDate.getFullYear()) * 12 + endDate.getMonth() - startDate.getMonth();

  // Create ticks for every month if the range is less than 24 months, otherwise every 3 months
  const ticks: number[] = [];
  const tickInterval = monthDiff > 24 ? 3 : 1;
  let currentDate = new Date(startDate);
  while (currentDate <= endDate) {
    ticks.push(currentDate.getTime());
    currentDate.setMonth(currentDate.getMonth() + tickInterval);
  }

  const latestHistoricalPrice = historicalData[historicalData.length - 1]?.price || 0;
  const latestForecastPrice = forecastData[0]?.predicted_price;
  const priceDifference = latestForecastPrice 
    ? ((latestForecastPrice - latestHistoricalPrice) / latestHistoricalPrice) * 100 
    : 0;
  const trend = priceDifference > 0 ? 'bullish' : priceDifference < 0 ? 'bearish' : 'neutral';

  const getTrendColor = (trend: 'bullish' | 'bearish' | 'neutral') => {
    switch (trend.toLowerCase()) {
      case 'bullish':
        return 'bg-green-500';
      case 'bearish':
        return 'bg-red-500';
      default:
        return 'bg-yellow-500';
    }
  };

  return (
    <div className="container mx-auto p-4 space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>BTC/USD Price Forecast</CardTitle>
          <CardDescription>
            Historical data and price predictions for Bitcoin
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => format(new Date(value), 'MMM yyyy')}
                  ticks={ticks}
                  interval={0}
                  angle={-45}
                  textAnchor="end"
                  height={50}
                />
                <YAxis 
                  tickFormatter={(value) => `$${value.toLocaleString()}`}
                />
                <Tooltip
                  labelFormatter={(value) => format(new Date(value), 'MMM d, yyyy')}
                  formatter={(value: any, name: string) => {
                    if (value === null) return ['-', name];
                    return [`$${value.toLocaleString()}`, name];
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="price" 
                  stroke="#8884d8" 
                  name="Historical Price"
                  strokeWidth={2}
                  dot={false}
                />
                {forecastData.length > 0 && (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="predicted" 
                      stroke="#82ca9d" 
                      name="Predicted Price"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="upper" 
                      stroke="#82ca9d" 
                      name="Upper Bound"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      opacity={0.5}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="lower" 
                      stroke="#82ca9d" 
                      name="Lower Bound"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      dot={false}
                      opacity={0.5}
                    />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <h3 className="text-sm font-medium">Current Price</h3>
          <div className="text-2xl font-bold">${latestHistoricalPrice.toLocaleString()}</div>
          <p className="text-xs text-gray-500">
            as of {historicalData[historicalData.length - 1]?.date || 'N/A'}
          </p>
        </Card>

        {forecastData.length > 0 && (
          <>
            <Card className="p-4">
              <h3 className="text-sm font-medium">7-Day Forecast</h3>
              <div className="text-2xl font-bold">${latestForecastPrice?.toLocaleString()}</div>
              <p className={`text-xs ${priceDifference >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {priceDifference >= 0 ? '+' : ''}{priceDifference.toFixed(2)}% from current
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Model: {forecastData[0]?.model_name || 'N/A'}
              </p>
            </Card>

            <Card className="p-4">
              <h3 className="text-sm font-medium">Forecast Metrics</h3>
              <div className="space-y-2">
                <div>
                  <p className="text-xs text-gray-500">Uncertainty</p>
                  <p className="text-sm font-semibold">
                    ±{((forecastData[0]?.average_uncertainty || 0) * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Days Ahead</p>
                  <p className="text-sm font-semibold">{forecastData[0]?.days_ahead || 7} days</p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="text-sm font-medium">Price Range</h3>
              <div className="mt-2">
                <div className="flex justify-between items-center">
                  <div>
                    <p className="text-xs text-gray-500">Upper</p>
                    <p className="text-sm font-semibold">
                      ${forecastData[0]?.confidence_interval_upper.toLocaleString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-500">Lower</p>
                    <p className="text-sm font-semibold">
                      ${forecastData[0]?.confidence_interval_lower.toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className={`mt-2 px-2 py-1 rounded-full text-center text-white text-sm ${getTrendColor(trend)}`}>
                  {trend.toUpperCase()}
                </div>
              </div>
            </Card>
          </>
        )}
      </div>

      <Card className="p-4">
        <h2 className="text-lg font-semibold mb-4">Market Insight</h2>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {forecastData.length > 0 ? (
              <>
                {trend === 'bullish' 
                  ? 'The market shows a bullish trend with potential for upward movement. '
                  : trend === 'bearish'
                  ? 'The market indicates a bearish trend with possible downward pressure. '
                  : 'The market appears to be in a neutral state with no clear directional bias. '}
                The forecast model predicts a price range between ${forecastData[0]?.confidence_interval_lower.toLocaleString()} 
                and ${forecastData[0]?.confidence_interval_upper.toLocaleString()} over the next 
                {forecastData[0]?.days_ahead || 7} days.
              </>
            ) : (
              'Historical data available. Forecast data currently unavailable.'
            )}
          </p>
          {forecastData.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                <span className="font-medium">Model:</span> {forecastData[0]?.model_name} ({forecastData[0]?.model_type})
              </p>
              <p className="text-sm text-muted-foreground">
                <span className="font-medium">Forecast Range:</span> {forecastData[0]?.days_ahead} days
              </p>
              <p className="text-sm text-muted-foreground">
                <span className="font-medium">Uncertainty Level:</span> ±{((forecastData[0]?.average_uncertainty || 0) * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}