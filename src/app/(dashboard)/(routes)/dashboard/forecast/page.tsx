"use client";

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { 
  Loader2, 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  BarChart,
  DollarSign,
  Info,
  ShieldAlert
} from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { 
  getLatestForecast, 
  getHistoricalData, 
  combineDataForChart,
  calculateTrend,
  type ForecastData,
  type HistoricalData,
  type ChartData
} from '@/lib/forecast-manager';

// Import our custom components
import MetricCard from '@/components/ui/metric-card';
import ForecastChart from '@/components/ui/forecast-chart';
import ForecastInsight from '@/components/ui/forecast-insight';
import TimeFrameSelector from '@/components/ui/time-frame-selector';

export default function ForecastPage() {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimePeriod, setSelectedTimePeriod] = useState<'90d' | '180d' | '365d'>('365d');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch historical data
        const historical = await getHistoricalData('BTC_USD', 365);
        setHistoricalData(historical);

        // Fetch forecast data
        const forecast = await getLatestForecast('BTC_USD');
        setForecastData(forecast);

        // Combine data for chart
        setChartData(combineDataForChart(historical, forecast));
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Format price as currency
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(price);
  };

  // Format date for display
  const formatDate = (dateStr: string) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return new Intl.DateTimeFormat('en-US', { 
      month: 'long', 
      day: 'numeric', 
      year: 'numeric' 
    }).format(date);
  };

  if (loading) {
    return (
      <div className="flex-1 p-8 pt-6 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium">Loading forecast data...</h3>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8 pt-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  const latestHistoricalPrice = historicalData.length > 0 
    ? historicalData[historicalData.length - 1].price 
    : 0;
    
  const latestForecastPrice = forecastData?.predicted_price || 0;
  const trend = forecastData 
    ? calculateTrend(latestHistoricalPrice, latestForecastPrice)
    : 'neutral';
  const changePercent = forecastData 
    ? ((latestForecastPrice - latestHistoricalPrice) / latestHistoricalPrice) * 100
    : 0;

  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto Forecast</h2>
            <p className="text-muted-foreground">
              Price predictions and market analysis for cryptocurrencies
            </p>
          </div>
          
          <TimeFrameSelector 
            selected={selectedTimePeriod} 
            onChange={setSelectedTimePeriod} 
          />
        </div>

        <Card className="overflow-hidden">
          <CardHeader className="pb-0">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>BTC/USD Price Forecast</CardTitle>
                <CardDescription>
                  Historical data and price predictions for Bitcoin
                </CardDescription>
              </div>
              <div className="hidden md:flex items-center gap-2">
                <div className="flex items-center gap-1">
                  <div className="w-4 h-1 bg-[#4F46E5] rounded-full"></div>
                  <span className="text-xs text-gray-500">Historical</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-4 h-1 bg-[#10B981] rounded-full"></div>
                  <span className="text-xs text-gray-500">Forecast</span>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ForecastChart data={chartData} timeFrame={selectedTimePeriod} />
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Current Price"
            value={formatPrice(latestHistoricalPrice)}
            icon={<DollarSign className="h-5 w-5 text-blue-500" />}
            subtitle={`as of ${historicalData.length > 0 ? formatDate(historicalData[historicalData.length - 1].date) : 'N/A'}`}
            iconBgColor="bg-blue-100"
          />

          {forecastData && (
            <>
              <MetricCard
                title={`${forecastData.days_ahead || 3}-Day Forecast`}
                value={formatPrice(latestForecastPrice)}
                icon={<BarChart className="h-5 w-5 text-green-500" />}
                subtitle={`${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}% from current`}
                trend={changePercent >= 0 ? 'up' : 'down'}
                iconBgColor="bg-green-100"
              />

              <MetricCard
                title="Price Trend"
                value={trend.toUpperCase()}
                icon={changePercent >= 0 
                  ? <TrendingUp className="h-5 w-5 text-green-500" />
                  : <TrendingDown className="h-5 w-5 text-red-500" />
                }
                subtitle="Based on forecast analysis"
                iconBgColor={changePercent >= 0 ? "bg-green-100" : "bg-red-100"}
              />

              <MetricCard
                title="Uncertainty"
                value={`±${(forecastData.average_uncertainty || 10).toFixed(1)}%`}
                icon={<ShieldAlert className="h-5 w-5 text-orange-500" />}
                subtitle={`Model: ${forecastData.model_name || 'Chronos'}`}
                iconBgColor="bg-orange-100"
              />
            </>
          )}
        </div>

        <Card className="p-6 shadow-sm">
          <div className="flex items-start space-x-2 mb-4">
            <Info className="h-5 w-5 text-blue-500 mt-1" />
            <h3 className="text-lg font-semibold">Market Insight</h3>
          </div>
          
          <ForecastInsight 
            currentPrice={latestHistoricalPrice} 
            forecastData={forecastData} 
          />
        </Card>
      </div>
    </div>
  );
}