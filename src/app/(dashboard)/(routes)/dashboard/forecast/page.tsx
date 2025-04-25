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
  ShieldAlert,
  Brain,
  LineChart,
  Clock,
  ChevronDown,
  ArrowRight,
  Sparkles,
  Target,
  Gauge,
  Calendar
} from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import TimeFrameSelector from '@/components/ui/time-frame-selector';
import MetricCard from '@/components/ui/metric-card';
import ForecastChart from '@/components/ui/forecast-chart';
import ForecastInsight from '@/components/ui/forecast-insight';
import { cn } from '@/lib/utils';

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

// Helper functions for formatting
const formatPrice = (price: number) => {
  return new Intl.NumberFormat('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2 
  }).format(price);
};

const formatDate = (dateStr: string) => {
  return new Date(dateStr).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const getConfidenceColor = (uncertainty: number) => {
  if (uncertainty <= 5) return 'text-green-600';
  if (uncertainty <= 10) return 'text-yellow-600';
  return 'text-red-600';
};

const PriceMetricCard = ({ 
  title, 
  price,
  change,
  date,
  type = 'current',
  icon: Icon,
}: { 
  title: string;
  price: number;
  change?: number;
  date?: string;
  type?: 'current' | 'forecast';
  icon: any;
}) => {
  const isPositiveChange = change && change > 0;
  const gradientColors = type === 'current' 
    ? 'from-blue-500 to-blue-600'
    : isPositiveChange 
      ? 'from-green-500 to-emerald-600'
      : 'from-red-500 to-rose-600';

  return (
    <div className="relative overflow-hidden rounded-xl border bg-white p-6 shadow-lg transition-all hover:shadow-xl">
      <div className="absolute inset-0 bg-gradient-to-br opacity-[0.03] from-gray-100 to-gray-200" />
      <div className="absolute -right-6 -top-6 h-24 w-24 rounded-full bg-gradient-to-br opacity-10" />
      
      <div className="relative">
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <Icon className="h-5 w-5" />
          <span>{title}</span>
        </div>

        <div className="mt-4 space-y-1">
          <div className={cn(
            "text-3xl font-bold tracking-tight bg-gradient-to-br bg-clip-text text-transparent",
            gradientColors
          )}>
            {formatPrice(price)}
          </div>
          
          {change && (
            <div className="flex items-center gap-2">
              <div className={cn(
                "flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium",
                isPositiveChange ? "bg-green-50 text-green-600" : "bg-red-50 text-red-600"
              )}>
                {isPositiveChange ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                {isPositiveChange ? '+' : ''}{change.toFixed(2)}%
              </div>
              {date && (
                <span className="text-xs text-muted-foreground">
                  as of {formatDate(date)}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const MetricTile = ({
  label,
  value,
  icon: Icon,
  color = "blue"
}: {
  label: string;
  value: string | number;
  icon: any;
  color?: "blue" | "green" | "red" | "purple" | "yellow";
}) => {
  const colors = {
    blue: "bg-blue-50 text-blue-600",
    green: "bg-green-50 text-green-600",
    red: "bg-red-50 text-red-600",
    purple: "bg-purple-50 text-purple-600",
    yellow: "bg-yellow-50 text-yellow-600",
  };

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-white p-4">
      <div className={cn("rounded-lg p-2", colors[color])}>
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="font-medium">{value}</p>
      </div>
    </div>
  );
};

const ConfidenceIndicator = ({ value }: { value: number }) => {
  const getColor = (v: number) => {
    if (v <= 5) return { bg: "bg-green-100", text: "text-green-700", ring: "ring-green-600" };
    if (v <= 10) return { bg: "bg-yellow-100", text: "text-yellow-700", ring: "ring-yellow-600" };
    return { bg: "bg-red-100", text: "text-red-700", ring: "ring-red-600" };
  };

  const { bg, text, ring } = getColor(value);
  const percentage = Math.min(100, Math.max(0, (value / 20) * 100));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">Uncertainty Level</span>
        <span className={cn("text-sm font-bold", text)}>±{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-100">
        <div 
          className={cn("h-full rounded-full transition-all", bg)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};

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
      <div className="flex-1 p-8 pt-6 flex items-center justify-center min-h-[600px] bg-gradient-to-br from-blue-50 to-purple-50">
        <div className="text-center space-y-4">
          <div className="relative">
            <div className="absolute inset-0 rounded-full animate-ping bg-blue-400 opacity-20" />
            <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto relative z-10" />
          </div>
          <h3 className="text-xl font-medium text-muted-foreground">Loading forecast data...</h3>
          <p className="text-sm text-muted-foreground">Please wait while we analyze market data</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8 pt-6 bg-gradient-to-br from-red-50 to-orange-50">
        <Alert variant="destructive" className="max-w-2xl mx-auto">
          <AlertCircle className="h-5 w-5" />
          <AlertTitle className="text-lg">Error Loading Data</AlertTitle>
          <AlertDescription className="mt-2">{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  const latestHistoricalPrice = historicalData.length > 0 
    ? historicalData[historicalData.length - 1].price 
    : 0;

  return (
    <div className="min-h-screen bg-[#fafafa]">
      <div className="flex-1 p-8 pt-6">
        <div className="space-y-8 max-w-[1400px] mx-auto">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Crypto Forecast
                </h2>
                <div className="flex h-6 items-center rounded-full bg-blue-50 px-2 text-xs font-medium text-blue-600">
                  AI-Powered
                </div>
              </div>
              <p className="text-muted-foreground mt-2 text-lg">
                Advanced price predictions and market analysis for Bitcoin
              </p>
            </div>
            
            <TimeFrameSelector 
              selected={selectedTimePeriod} 
              onChange={setSelectedTimePeriod} 
            />
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <PriceMetricCard
              title="Current Price"
              price={latestHistoricalPrice}
              date={historicalData.length > 0 ? historicalData[historicalData.length - 1].date : undefined}
              icon={DollarSign}
              type="current"
            />
            
            {forecastData && (
              <PriceMetricCard
                title={`${forecastData.days_ahead}-Day Forecast`}
                price={forecastData.final_forecast}
                change={forecastData.change_pct}
                icon={Target}
                type="forecast"
              />
            )}
          </div>

          {forecastData && (
            <div className="grid gap-6 md:grid-cols-3">
              <MetricTile
                label="Model"
                value={forecastData.model_name}
                icon={Brain}
                color="purple"
              />
              <MetricTile
                label="Generated"
                value={formatDate(forecastData.forecast_timestamp)}
                icon={Calendar}
                color="blue"
              />
              <MetricTile
                label="Probability of Increase"
                value={`${forecastData.probability_increase.toFixed(1)}%`}
                icon={Gauge}
                color={forecastData.probability_increase > 50 ? "green" : "red"}
              />
            </div>
          )}

          <Card className="overflow-hidden bg-white shadow-xl border-none">
            <CardHeader className="border-b bg-gray-50/50">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-xl">BTC/USD Price Forecast</CardTitle>
                  <CardDescription>
                    Historical data and AI-powered price predictions
                  </CardDescription>
                </div>
                <div className="hidden md:flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full" />
                    <span className="text-sm text-muted-foreground">Historical</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full" />
                    <span className="text-sm text-muted-foreground">Forecast</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-green-500/20 rounded-full" />
                    <span className="text-sm text-muted-foreground">Confidence</span>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-6">
              <div className="h-[400px]">
                <ForecastChart data={chartData} timeFrame={selectedTimePeriod} />
              </div>
            </CardContent>
          </Card>

          {forecastData && (
            <div className="grid gap-6 md:grid-cols-2">
              <Card className="bg-white shadow-lg border-none">
                <CardHeader className="border-b bg-gray-50/50">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-purple-500" />
                    <CardTitle>AI Analysis</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="prose prose-sm">
                    <p className="text-gray-600 leading-relaxed">
                      {forecastData.insight}
                    </p>
                  </div>
                  <div className="mt-6 space-y-6">
                    <ConfidenceIndicator value={forecastData.average_uncertainty} />
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Price Range</span>
                        <span className="font-medium">
                          {formatPrice(forecastData.lower_bounds[0])} - {formatPrice(forecastData.upper_bounds[0])}
                        </span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-white shadow-lg border-none">
                <CardHeader className="border-b bg-gray-50/50">
                  <div className="flex items-center gap-2">
                    <Target className="h-5 w-5 text-blue-500" />
                    <CardTitle>Forecast Details</CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="p-6">
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Model Type</p>
                        <p className="font-medium">{forecastData.model_type}</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm text-muted-foreground">Timeframe</p>
                        <p className="font-medium">{forecastData.days_ahead} days ahead</p>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Day 1 Forecast</span>
                        <span className="font-medium">{formatPrice(forecastData.forecast_values[0])}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Day 2 Forecast</span>
                        <span className="font-medium">{formatPrice(forecastData.forecast_values[1])}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Day 3 Forecast</span>
                        <span className="font-medium">{formatPrice(forecastData.forecast_values[2])}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}