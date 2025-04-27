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
  Calendar,
  Zap,
  Activity,
  BarChart4,
  Waves
} from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import TimeFrameSelector from '@/components/ui/time-frame-selector';
import MetricCard from '@/components/ui/metric-card';
import ForecastChart from '@/components/ui/forecast-chart';
import ForecastInsight from '@/components/ui/forecast-insight';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

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

// Enhanced helper functions
const formatPrice = (price: number) => {
  return new Intl.NumberFormat('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2 
  }).format(price);
};

const formatCompactPrice = (price: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1
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
  if (uncertainty <= 5) return { bg: 'bg-green-500/10', text: 'text-green-500', gradient: 'from-green-500' };
  if (uncertainty <= 10) return { bg: 'bg-yellow-500/10', text: 'text-yellow-500', gradient: 'from-yellow-500' };
  return { bg: 'bg-red-500/10', text: 'text-red-500', gradient: 'from-red-500' };
};

const PriceCard = ({ 
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
  const colors = type === 'current' 
    ? { gradient: 'from-blue-500 to-indigo-600', glow: 'group-hover:shadow-blue-500/25' }
    : isPositiveChange 
      ? { gradient: 'from-green-500 to-emerald-600', glow: 'group-hover:shadow-green-500/25' }
      : { gradient: 'from-red-500 to-rose-600', glow: 'group-hover:shadow-red-500/25' };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="group"
    >
      <div className={cn(
        "relative overflow-hidden rounded-xl border bg-white/50 backdrop-blur-lg p-6",
        "transition-all duration-300 hover:shadow-xl hover:scale-[1.02]",
        "shadow-lg", colors.glow
      )}>
        <div className="absolute inset-0 bg-gradient-to-br opacity-[0.08] from-white to-transparent" />
        <div className={cn(
          "absolute -right-12 -top-12 h-32 w-32 rounded-full bg-gradient-to-br opacity-20",
          colors.gradient
        )} />
        
        <div className="relative">
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <div className={cn("p-2 rounded-lg", type === 'current' ? 'bg-blue-500/10' : isPositiveChange ? 'bg-green-500/10' : 'bg-red-500/10')}>
              <Icon className={cn("h-5 w-5", type === 'current' ? 'text-blue-500' : isPositiveChange ? 'text-green-500' : 'text-red-500')} />
            </div>
            <span className="font-medium">{title}</span>
          </div>

          <div className="mt-4 space-y-2">
            <div className={cn(
              "text-4xl font-bold tracking-tight bg-gradient-to-br bg-clip-text text-transparent",
              colors.gradient
            )}>
              {formatPrice(price)}
            </div>
            
            {change && (
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                  isPositiveChange ? "bg-green-500/10 text-green-600" : "bg-red-500/10 text-red-600"
                )}>
                  {isPositiveChange ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                  {isPositiveChange ? '+' : ''}{change.toFixed(2)}%
                </div>
                {date && (
                  <span className="text-xs text-muted-foreground">
                    Updated {formatDate(date)}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

const MetricTile = ({
  label,
  value,
  icon: Icon,
  color = "blue",
  delay = 0
}: {
  label: string;
  value: string | number;
  icon: any;
  color?: "blue" | "green" | "red" | "purple" | "yellow";
  delay?: number;
}) => {
  const colors = {
    blue: { bg: "bg-blue-50", text: "text-blue-600", ring: "group-hover:ring-blue-500/20" },
    green: { bg: "bg-green-50", text: "text-green-600", ring: "group-hover:ring-green-500/20" },
    red: { bg: "bg-red-50", text: "text-red-600", ring: "group-hover:ring-red-500/20" },
    purple: { bg: "bg-purple-50", text: "text-purple-600", ring: "group-hover:ring-purple-500/20" },
    yellow: { bg: "bg-yellow-50", text: "text-yellow-600", ring: "group-hover:ring-yellow-500/20" },
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3, delay }}
      className="group"
    >
      <div className={cn(
        "flex items-center gap-3 rounded-lg border bg-white/50 backdrop-blur-lg p-4",
        "transition-all duration-300",
        "hover:shadow-lg hover:scale-[1.02]",
        "ring-1 ring-black/5",
        colors[color].ring
      )}>
        <div className={cn("rounded-lg p-2.5", colors[color].bg)}>
          <Icon className={cn("h-4 w-4", colors[color].text)} />
        </div>
        <div>
          <p className="text-sm text-muted-foreground font-medium">{label}</p>
          <p className={cn("font-semibold", colors[color].text)}>{value}</p>
        </div>
      </div>
    </motion.div>
  );
};

const ConfidenceIndicator = ({ value }: { value: number }) => {
  const colors = getConfidenceColor(value);
  const percentage = Math.min(100, Math.max(0, (value / 20) * 100));

  return (
    <motion.div 
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-2"
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-muted-foreground">Uncertainty Level</span>
        <span className={cn("text-sm font-bold", colors.text)}>±{value.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-100/50 overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={cn("h-full rounded-full", colors.bg)}
        />
      </div>
    </motion.div>
  );
};

const InsightCard = ({ 
  title, 
  icon: Icon, 
  iconColor,
  children 
}: { 
  title: string;
  icon: any;
  iconColor: string;
  children: React.ReactNode;
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="bg-white/50 backdrop-blur-lg shadow-lg border-none hover:shadow-xl transition-all duration-300">
        <CardHeader className="border-b bg-gray-50/50">
          <div className="flex items-center gap-2">
            <div className={cn("p-2 rounded-lg", `bg-${iconColor}-500/10`)}>
              <Icon className={cn("h-5 w-5", `text-${iconColor}-500`)} />
            </div>
            <CardTitle>{title}</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          {children}
        </CardContent>
      </Card>
    </motion.div>
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
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center space-y-4"
        >
          <div className="relative">
            <div className="absolute inset-0 rounded-full animate-ping bg-blue-400 opacity-20" />
            <div className="relative z-10 bg-white/50 backdrop-blur-lg rounded-2xl p-8 shadow-xl border border-blue-100">
              <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto" />
              <h3 className="text-xl font-medium text-muted-foreground mt-4">Analyzing Market Data</h3>
              <p className="text-sm text-muted-foreground mt-2">Please wait while we prepare your forecast...</p>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-8 pt-6 bg-gradient-to-br from-red-50 to-orange-50">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Alert variant="destructive" className="max-w-2xl mx-auto">
            <AlertCircle className="h-5 w-5" />
            <AlertTitle className="text-lg">Error Loading Data</AlertTitle>
            <AlertDescription className="mt-2">{error}</AlertDescription>
          </Alert>
        </motion.div>
      </div>
    );
  }

  const latestHistoricalPrice = historicalData.length > 0 
    ? historicalData[historicalData.length - 1].price 
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/20 to-purple-50/20">
      <div className="flex-1 p-8 pt-6">
        <div className="space-y-8 max-w-[1400px] mx-auto">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col md:flex-row md:items-center md:justify-between gap-6"
          >
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Crypto Forecast
                </h2>
                <div className="flex h-6 items-center gap-1.5 rounded-full bg-blue-50 px-2.5 text-xs font-medium text-blue-600">
                  <Zap className="h-3.5 w-3.5" />
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
          </motion.div>

          <div className="grid gap-6 md:grid-cols-2">
            <PriceCard
              title="Current Price"
              price={latestHistoricalPrice}
              date={historicalData.length > 0 ? historicalData[historicalData.length - 1].date : undefined}
              icon={DollarSign}
              type="current"
            />
            
            {forecastData && (
              <PriceCard
                title={`${forecastData.days_ahead}-Day Forecast`}
                price={forecastData.final_forecast}
                change={forecastData.change_pct}
                icon={Target}
                type="forecast"
              />
            )}
          </div>

          {forecastData && (
            <div className="grid gap-6 md:grid-cols-4">
              <MetricTile
                label="Model"
                value={forecastData.model_name}
                icon={Brain}
                color="purple"
                delay={0.1}
              />
              <MetricTile
                label="Generated"
                value={formatDate(forecastData.forecast_timestamp)}
                icon={Calendar}
                color="blue"
                delay={0.2}
              />
              <MetricTile
                label="Probability of Increase"
                value={`${forecastData.probability_increase.toFixed(1)}%`}
                icon={Gauge}
                color={forecastData.probability_increase > 50 ? "green" : "red"}
                delay={0.3}
              />
              <MetricTile
                label="Volatility"
                value={`±${forecastData.average_uncertainty.toFixed(1)}%`}
                icon={Activity}
                color={forecastData.average_uncertainty <= 5 ? "green" : forecastData.average_uncertainty <= 10 ? "yellow" : "red"}
                delay={0.4}
              />
            </div>
          )}

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="overflow-hidden bg-white/50 backdrop-blur-lg shadow-xl border-none hover:shadow-2xl transition-all duration-300">
              <CardHeader className="border-b bg-gray-50/50">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <BarChart4 className="h-5 w-5 text-blue-500" />
                      <CardTitle className="text-xl">BTC/USD Price Forecast</CardTitle>
                    </div>
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
          </motion.div>

          {forecastData && (
            <div className="grid gap-6 md:grid-cols-2">
              <InsightCard 
                title="AI Analysis" 
                icon={Sparkles}
                iconColor="purple"
              >
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
              </InsightCard>

              <InsightCard 
                title="Forecast Details" 
                icon={Target}
                iconColor="blue"
              >
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
                    {forecastData.forecast_values.map((value, index) => (
                      <motion.div 
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.5, delay: index * 0.1 }}
                        className="flex items-center justify-between bg-gray-50/50 rounded-lg p-3"
                      >
                        <span className="text-sm text-muted-foreground">Day {index + 1} Forecast</span>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{formatPrice(value)}</span>
                          <div className={cn(
                            "text-xs font-medium px-1.5 py-0.5 rounded-full",
                            value > latestHistoricalPrice ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                          )}>
                            {((value - latestHistoricalPrice) / latestHistoricalPrice * 100).toFixed(2)}%
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </InsightCard>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}