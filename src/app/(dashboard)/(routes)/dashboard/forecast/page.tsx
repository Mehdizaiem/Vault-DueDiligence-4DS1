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
import TimeFrameSelector from '@/components/ui/time-frame-selector';
import MetricCard from '@/components/ui/metric-card';
import ForecastChart from '@/components/ui/forecast-chart';
import ForecastInsight from '@/components/ui/forecast-insight';

interface HistoricalData {
  date: string;
  price: number;
}

interface ForecastData {
  id?: string;
  symbol: string;
  forecast_timestamp: string;
  model_name: string;
  model_type: string;
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

interface ChartData {
  date: string;
  price?: number;
  predicted?: number;
  lower?: number;
  upper?: number;
}

export default function ForecastPage() {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [forecastData, setForecastData] = useState<ForecastData | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimePeriod, setSelectedTimePeriod] = useState<'90d' | '180d' | '365d'>('365d');

  // Helper function to combine historical and forecast data for the chart
  function combineDataForChart(historical: HistoricalData[], forecast: ForecastData | null): ChartData[] {
    if (!historical || historical.length === 0) return [];

    // Get the last historical date
    const lastHistoricalDate = new Date(historical[historical.length - 1].date);
    
    const chartData = historical.map(item => ({
      date: item.date,
      price: item.price,
      predicted: undefined,
      lower: undefined,
      upper: undefined
    }));

    if (forecast?.forecast_values) {
      const forecastItems: ChartData[] = [];
      
      // Start forecast from the last historical date
      const lastHistoricalPoint: ChartData = {
        date: lastHistoricalDate.toISOString(),
        price: historical[historical.length - 1].price,
        predicted: forecast.forecast_values[0], // Connect to first forecast point
        lower: forecast.lower_bounds[0],
        upper: forecast.upper_bounds[0]
      };
      
      // Generate forecast points
      for (let i = 0; i < forecast.forecast_values.length; i++) {
        const forecastDate = new Date(lastHistoricalDate);
        forecastDate.setDate(forecastDate.getDate() + i + 1); // +1 to start from next day
        
        forecastItems.push({
          date: forecastDate.toISOString(),
          price: undefined,
          predicted: forecast.forecast_values[i],
          lower: forecast.lower_bounds[i],
          upper: forecast.upper_bounds[i]
        });
      }
      
      // Add the last historical point to connect the lines
      return [...chartData, lastHistoricalPoint, ...forecastItems];
    }

    return chartData;
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch historical data from our API
        const historicalResponse = await fetch(`/api/historical?symbol=BTC_USD&days=${selectedTimePeriod.replace('d', '')}`);
        const historicalResult = await historicalResponse.json();

        if (!historicalResult.success) {
          throw new Error(historicalResult.error || 'Failed to fetch historical data');
        }

        // Fetch forecast data from our Storage Manager via API
        const forecastResponse = await fetch('/api/forecasts?symbol=BTC_USD');
        const forecastResult = await forecastResponse.json();

        if (!forecastResult.success) {
          console.warn('Warning fetching forecast data:', forecastResult.error);
          // Continue with historical data even if forecast fails
        }

        const historicalDataArray = historicalResult.data || [];
        setHistoricalData(historicalDataArray);
        
        // Get the forecast data
        const forecastDataItem = forecastResult.success ? forecastResult.data : null;
        console.log('Forecast data:', forecastDataItem); // Debug log
        setForecastData(forecastDataItem);

        // Combine the data for the chart
        const combinedData = combineDataForChart(historicalDataArray, forecastDataItem);
        console.log('Combined chart data:', combinedData); // Debug log
        setChartData(combinedData);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedTimePeriod]);

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

  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-6">
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

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Current Price"
            value={new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(latestHistoricalPrice)}
            icon={<DollarSign className="h-5 w-5 text-blue-500" />}
            subtitle={`as of ${historicalData.length > 0 
              ? new Date(historicalData[historicalData.length - 1].date).toLocaleDateString() 
              : 'N/A'}`}
            iconBgColor="bg-blue-100"
          />

          {forecastData && (
            <>
              <MetricCard
                title={`${forecastData.days_ahead}-Day Forecast`}
                value={new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(forecastData.final_forecast)}
                icon={<BarChart className="h-5 w-5 text-green-500" />}
                subtitle={`${forecastData.change_pct >= 0 ? '+' : ''}${forecastData.change_pct.toFixed(2)}% from current`}
                trend={forecastData.change_pct >= 0 ? 'up' : 'down'}
                iconBgColor="bg-green-100"
              />

              <MetricCard
                title="Price Trend"
                value={forecastData.trend.charAt(0).toUpperCase() + forecastData.trend.slice(1)}
                icon={forecastData.trend === 'bullish' || forecastData.trend === 'strongly bullish'
                  ? <TrendingUp className="h-5 w-5 text-green-500" />
                  : <TrendingDown className="h-5 w-5 text-red-500" />
                }
                subtitle={`${forecastData.probability_increase.toFixed(1)}% chance of increase`}
                trend={forecastData.trend.includes('bullish') ? 'up' : 'down'}
                iconBgColor={forecastData.trend.includes('bullish') ? "bg-green-100" : "bg-red-100"}
              />

              <MetricCard
                title="Uncertainty"
                value={`±${forecastData.average_uncertainty.toFixed(1)}%`}
                icon={<ShieldAlert className="h-5 w-5 text-orange-500" />}
                subtitle={`Model: ${forecastData.model_name}`}
                iconBgColor="bg-orange-100"
              />
            </>
          )}
        </div>

        <div className="grid gap-6">
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

          {forecastData && (
            <Card>
              <CardHeader>
                <CardTitle>Forecast Analysis</CardTitle>
                <CardDescription>
                  Detailed insights and analysis of the price forecast
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ForecastInsight 
                  currentPrice={latestHistoricalPrice} 
                  forecastData={forecastData} 
                />
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}