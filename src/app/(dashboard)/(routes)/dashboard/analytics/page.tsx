"use client";

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { 
  BarChart, 
  PieChart, 
  Filter, 
  Download, 
  CalendarRange,
  ChevronDown,
  CircleDollarSign,
  Activity,
  AlertCircle,
  RefreshCw,
  Newspaper,
  Clock,
  ExternalLink,
  BarChart2,
  FileText,
  ClipboardList,
  Shield,
  File,
  TrendingUp,
  TrendingDown,
  BadgePercent,
  Sparkles,
  Info,
  DollarSign,
  Target,
  Gauge,
  Loader2,
  Zap
} from "lucide-react";

// Define BarChart4 as a component since it's not in Lucide
const BarChart4 = Activity; // Using Activity icon as a replacement
import { 
  LineChart, 
  Line, 
  Pie, 
  Cell, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart as RechartsChart,
  BarChart as RechartBarChart,
  Bar,
  Area
} from 'recharts';
import { fetchAnalyticsData, AnalyticsResponse } from '@/services/analytics-service';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

// Type definitions
interface KPI {
  value: string;
  change: string;
  trend: 'up' | 'down' | 'neutral';
}

interface Kpis {
  market_cap?: KPI;
  asset_count?: KPI;
  price_change?: KPI;
  market_sentiment?: KPI;
}

interface Asset {
  name: string;
  value: number;
  color?: string;
}

interface PerformanceItem {
  symbol: string;
  timestamp: string;
  change_pct: number;
}

interface NewsItem {
  title: string;
  source: string;
  date: string;
  url: string;
  sentiment_score: number;
  sentiment_color: string;
  related_assets?: string[];
}

interface DueDiligenceItem {
  title: string;
  document_type: string;
  source: string;
  icon: string;
  keywords?: string[];
}

interface TopPerformer {
  symbol: string;
  change_pct: number;
  period: string;
  color: string;
}

const normalizeTrend = (trend: any): 'up' | 'down' | 'neutral' => {
  if (trend === 'up' || trend === 'down' || trend === 'neutral') {
    return trend;
  }
  return 'neutral';
};

const getSentimentColor = (score: number) => {
  if (score >= 0.5) return { text: 'text-green-600', bg: 'bg-green-600' };
  if (score > 0) return { text: 'text-emerald-600', bg: 'bg-emerald-600' };
  if (score === 0) return { text: 'text-gray-600', bg: 'bg-gray-600' };
  if (score > -0.5) return { text: 'text-amber-600', bg: 'bg-amber-600' };
  return { text: 'text-red-600', bg: 'bg-red-600' };
};

// Date range options
const DATE_RANGES = [
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' }
];

// Color mapping for consistent chart colors
const CRYPTO_COLOR_MAP: Record<string, string> = {
  'BTC': '#1F78B4', // Professional blue
  'ETH': '#33A02C', // Professional green
  'SOL': '#6A3D9A', // Professional purple
  'ADA': '#E31A1C', // Professional red
  'XRP': '#FF7F00', // Professional orange
  'BNB': '#FFD700', // Professional gold
  'DOGE': '#B15928', // Professional brown
  'DOT': '#CAB2D6', // Professional light purple
  'AVAX': '#FB9A99', // Professional light red
  'MATIC': '#A6CEE3', // Professional light blue
  'LINK': '#1F78B4', // Professional blue
  'TRX': '#B2DF8A', // Professional light green
  'LTC': '#666666', // Professional gray
  'UNI': '#E31A1C', // Professional red
  'BCH': '#FDBF6F', // Professional light orange
  'DEFAULT': '#7F7F7F' // Professional medium gray
};

// Background color map for UI components
const BG_COLOR_MAP: Record<string, string> = {
  'BTC': 'bg-blue-500',
  'ETH': 'bg-green-500',
  'SOL': 'bg-purple-600',
  'ADA': 'bg-red-600',
  'XRP': 'bg-orange-500',
  'BNB': 'bg-yellow-400',
  'DOGE': 'bg-amber-700',
  'DOT': 'bg-purple-300',
  'AVAX': 'bg-red-300',
  'MATIC': 'bg-blue-300',
  'LINK': 'bg-blue-500',
  'TRX': 'bg-green-300',
  'LTC': 'bg-gray-600',
  'UNI': 'bg-red-600',
  'BCH': 'bg-orange-300',
  'OTHER': 'bg-gray-500',
  'DEFAULT': 'bg-gray-500'
};

// Helper functions
const formatPrice = (price: number | string) => {
  if (typeof price === 'string') {
    if (price.startsWith('$')) {
      return price;
    }
    return `$${price}`;
  }
  
  return new Intl.NumberFormat('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2 
  }).format(price);
};

const formatCompactPrice = (price: number | string) => {
  if (typeof price === 'string') {
    if (price.startsWith('$')) {
      const numValue = parseFloat(price.replace(/[$,]/g, ''));
      if (!isNaN(numValue)) {
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          notation: 'compact',
          maximumFractionDigits: 1
        }).format(numValue);
      }
    }
    return price;
  }
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1
  }).format(price);
};

// MetricTile component
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

// PriceCard component
const PriceCard = ({ 
  title, 
  price,
  change,
  date,
  type = 'current',
  icon: Icon,
}: { 
  title: string;
  price: string | number;
  change?: string | number;
  date?: string;
  type?: 'current' | 'forecast';
  icon: any;
}) => {
  let changeNum = 0;
  if (typeof change === 'string') {
    changeNum = parseFloat(change.replace(/[+%]/g, ''));
  } else if (typeof change === 'number') {
    changeNum = change;
  }
  
  const isPositiveChange = changeNum > 0;
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
              {typeof price === 'number' ? formatPrice(price) : price}
            </div>
            
            {change && (
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium",
                  isPositiveChange ? "bg-green-500/10 text-green-600" : "bg-red-500/10 text-red-600"
                )}>
                  {isPositiveChange ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />}
                  {typeof change === 'number' ? (isPositiveChange ? '+' : '') + change.toFixed(2) + '%' : change}
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

// InsightCard component
const InsightCard = ({ 
  title, 
  icon: Icon, 
  iconColor,
  className,
  children 
}: { 
  title: string;
  icon: any;
  iconColor: string;
  className?: string;
  children: React.ReactNode;
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={className}
    >
      <Card className="bg-white/50 backdrop-blur-lg shadow-lg border-none hover:shadow-xl transition-all duration-300">
        <CardHeader className="border-b bg-gray-50/50">
          <div className="flex items-center gap-2">
            <div className={`p-2 rounded-lg ${`bg-${iconColor}-500/10`}`}>
              <Icon className={`h-5 w-5 ${`text-${iconColor}-500`}`} />
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

// Format date utility function
const formatDate = (dateString: string): string => {
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  } catch {
    return dateString;
  }
};

export default function AnalyticsPage() {
  // States
  const [dateRange, setDateRange] = useState<string>('30d');
  const [kpis, setKpis] = useState<Kpis>({});
  const [assetDistribution, setAssetDistribution] = useState<Asset[]>([]);
  const [portfolioPerformance, setPortfolioPerformance] = useState<PerformanceItem[]>([]);
  const [recentNews, setRecentNews] = useState<NewsItem[]>([]);
  const [dueDiligence, setDueDiligence] = useState<DueDiligenceItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState<number>(0);
  const [activeSymbols, setActiveSymbols] = useState<string[]>(['All']);
  const [showFilterMenu, setShowFilterMenu] = useState<boolean>(false);
  const [filteredSymbols, setFilteredSymbols] = useState<string[]>([]);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [showDatePicker, setShowDatePicker] = useState<boolean>(false);
  const [customDateRange, setCustomDateRange] = useState<{start: string; end: string}>({
    start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0]
  });

  // Load data on component mount or when filters change
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchAnalyticsData({
          dateRange,
          symbols: activeSymbols.includes('All') ? [] : activeSymbols
        });
  
        if (data.error) {
          setError(data.error);
          console.error("Error from API:", data.error);
        } else {
          setError(null);
        }
  
        // Normalize KPI data to fix trend values
        const normalizedKpis: Kpis = {};
        
        if (data.kpis?.market_cap) {
          normalizedKpis.market_cap = {
            ...data.kpis.market_cap,
            trend: normalizeTrend(data.kpis.market_cap.trend)
          };
        }
        
        if (data.kpis?.asset_count) {
          normalizedKpis.asset_count = {
            ...data.kpis.asset_count,
            trend: normalizeTrend(data.kpis.asset_count.trend)
          };
        }
        
        if (data.kpis?.price_change) {
          normalizedKpis.price_change = {
            ...data.kpis.price_change,
            trend: normalizeTrend(data.kpis.price_change.trend)
          };
        }
        
        if (data.kpis?.market_sentiment) {
          normalizedKpis.market_sentiment = {
            ...data.kpis.market_sentiment,
            trend: normalizeTrend(data.kpis.market_sentiment.trend)
          };
        }
  
        setKpis(normalizedKpis);
        setAssetDistribution(data.asset_distribution || []);
        setPortfolioPerformance(data.portfolio_performance || []);
        setRecentNews(data.recent_news || []);
        setDueDiligence(data.due_diligence || []);
      } catch (error) {
        console.error("Error fetching analytics data:", error);
        setError(error instanceof Error ? error.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };
  
    loadData();
  }, [retryCount, dateRange, activeSymbols]);

  // Get unique symbols for filtering
  const uniqueSymbols = useMemo(() => {
    const symbols = new Set<string>();
    
    portfolioPerformance.forEach(item => {
      if (item.symbol) {
        const cleanSymbol = item.symbol.replace('USDT', '');
        if (cleanSymbol) symbols.add(cleanSymbol);
      }
    });
    
    assetDistribution.forEach(item => {
      if (item.name && item.name !== 'Other') {
        symbols.add(item.name);
      }
    });
    
    return ['All', ...Array.from(symbols)];
  }, [portfolioPerformance, assetDistribution]);

  // Handle filter actions
  const handleSymbolFilter = useCallback((symbol: string) => {
    setActiveSymbols(prev => {
      if (symbol === 'All') {
        return ['All'];
      }
      
      const newActiveSymbols = prev.includes('All') 
        ? [symbol]
        : prev.includes(symbol)
          ? prev.filter(s => s !== symbol)
          : [...prev, symbol];
      
      return newActiveSymbols.length ? newActiveSymbols : ['All'];
    });
  }, []);

  const handleRetry = useCallback(() => {
    setRetryCount(prev => prev + 1);
  }, []);

  const handleDateRangeChange = useCallback((range: string) => {
    setDateRange(range);
    setShowDatePicker(false);
  }, []);

  const handleCustomDateChange = useCallback(() => {
    setShowDatePicker(false);
    console.log(`Custom date range: ${customDateRange.start} to ${customDateRange.end}`);
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [customDateRange]);

  const handleExport = useCallback(() => {
    setIsExporting(true);
    
    try {
      const data = {
        kpis,
        assetDistribution,
        portfolioPerformance,
        recentNews: recentNews.map(({title, source, date, sentiment_score}) => 
          ({title, source, date, sentiment_score}))
      };
      
      const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `crypto-analytics-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      setTimeout(() => {
        setIsExporting(false);
        alert('Data exported successfully!');
      }, 500);
    } catch (error) {
      console.error('Export error:', error);
      setIsExporting(false);
      alert('Export failed: ' + (error instanceof Error ? error.message : 'Unknown error'));
    }
  }, [kpis, assetDistribution, portfolioPerformance, recentNews]);

  const toggleFilterMenu = useCallback(() => {
    if (!showFilterMenu) {
      setFilteredSymbols(
        activeSymbols.includes('All') ? [] : [...activeSymbols]
      );
    }
    setShowFilterMenu(prev => !prev);
  }, [showFilterMenu, activeSymbols]);

  const applySymbolFilters = useCallback(() => {
    if (filteredSymbols.length === 0) {
      setActiveSymbols(['All']);
    } else {
      setActiveSymbols(filteredSymbols);
    }
    setShowFilterMenu(false);
  }, [filteredSymbols]);

  const toggleSymbolFilter = useCallback((symbol: string) => {
    setFilteredSymbols(prev => {
      if (prev.includes(symbol)) {
        return prev.filter(s => s !== symbol);
      } else {
        return [...prev, symbol];
      }
    });
  }, []);

  // Prepare KPI display
  const kpiItems = useMemo(() => [
    {
      label: 'Market Cap',
      value: kpis.market_cap?.value || '$0',
      change: kpis.market_cap?.change || '+0%',
      trend: kpis.market_cap?.trend || 'neutral',
      icon: CircleDollarSign,
      color: 'text-green-500',
      bgColor: 'bg-green-100'
    },
    {
      label: 'Asset Count',
      value: kpis.asset_count?.value || '0',
      change: kpis.asset_count?.change || '+0',
      trend: kpis.asset_count?.trend || 'neutral',
      icon: FileText,
      color: 'text-blue-500',
      bgColor: 'bg-blue-100'
    },
    {
      label: 'Price Change',
      value: kpis.price_change?.value || '0%',
      change: kpis.price_change?.change || '+0%',
      trend: kpis.price_change?.trend || 'neutral',
      icon: BarChart2,
      color: 'text-violet-500',
      bgColor: 'bg-violet-100'
    },
    {
      label: 'Market Sentiment',
      value: kpis.market_sentiment?.value || '50%',
      change: kpis.market_sentiment?.change || '+0%',
      trend: kpis.market_sentiment?.trend || 'neutral',
      icon: Activity,
      color: 'text-pink-500',
      bgColor: 'bg-pink-100'
    }
  ], [kpis]);

  // Calculate top performers
  const topPerformers = useMemo(() => {
    if (!portfolioPerformance.length) return [];

    const bySymbol: Record<string, PerformanceItem[]> = {};
    
    portfolioPerformance.forEach(item => {
      if (!bySymbol[item.symbol]) {
        bySymbol[item.symbol] = [];
      }
      bySymbol[item.symbol].push(item);
    });
    
    const performers: TopPerformer[] = [];
    
    Object.entries(bySymbol).forEach(([symbol, items]) => {
      if (items.length < 2) return;
      
      items.sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      
      const first = items[0];
      const last = items[items.length - 1];
      
      const change_pct = last.change_pct - first.change_pct;
      
      const cleanSymbol = symbol.replace('USDT', '');
      
      performers.push({
        symbol: cleanSymbol,
        change_pct,
        period: dateRange,
        color: change_pct >= 0 ? 'text-green-500' : 'text-red-500'
      });
    });
    
    performers.sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct));
    
    return performers.slice(0, 5);
  }, [portfolioPerformance, dateRange]);

  // Prepare Portfolio Performance Chart Data
  const preparePricePerformanceData = useCallback(() => {
    if (!portfolioPerformance.length) return { chartData: [], lines: [] };
  
    const cleanSymbolName = (symbol: string): string => {
      return symbol.replace(/USDT$|USD$|usdt$|usd$/i, '');
    };
  
    const validSymbols = new Set<string>();
    const symbolMap: Record<string, string> = {};
    
    portfolioPerformance.forEach(item => {
      if (item.symbol) {
        const cleanSymbol = cleanSymbolName(item.symbol);
        symbolMap[item.symbol] = cleanSymbol;
        if (activeSymbols.includes('All') || activeSymbols.includes(cleanSymbol)) {
          validSymbols.add(item.symbol);
        }
      }
    });

    const dataByDate: Record<string, any> = {};
  
    portfolioPerformance.forEach(item => {
      if (!item.symbol || !validSymbols.has(item.symbol)) return;

      try {
        let dateObj: Date;
        if (typeof item.timestamp === 'string') {
          if (item.timestamp.includes('T')) {
            dateObj = new Date(item.timestamp);
          } else {
            dateObj = new Date(item.timestamp);
          }
        } else {
          dateObj = new Date(item.timestamp);
        }
        
        if (isNaN(dateObj.getTime())) {
          console.warn(`Invalid date: ${item.timestamp}`);
          return;
        }

        const dateStr = dateObj.toLocaleDateString('en-US');
        if (!dataByDate[dateStr]) {
          dataByDate[dateStr] = { date: dateStr };
        }

        const cleanSymbol = symbolMap[item.symbol];
        dataByDate[dateStr][cleanSymbol] = item.change_pct;
      } catch (e) {
        console.warn(`Error processing data point: ${e}`);
      }
    });

    const chartData = Object.values(dataByDate).sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const uniqueCleanSymbols = new Set(Object.values(symbolMap));
  
    const lines = Array.from(uniqueCleanSymbols).map(symbol => {
      return {
        name: symbol,
        dataKey: symbol,
        stroke: CRYPTO_COLOR_MAP[symbol] || CRYPTO_COLOR_MAP.DEFAULT,
        strokeWidth: 2,
        type: 'monotone' as const
      };
    });

    return { chartData, lines };
  }, [portfolioPerformance, activeSymbols]);

  // Prepare Asset Distribution Chart Data
  const prepareAssetChartData = useCallback(() => {
    if (!assetDistribution.length) return { data: [] };
  
    const pieColors = [
      '#1F78B4', // blue
      '#33A02C', // green
      '#E31A1C', // red
      '#FF7F00', // orange
      '#6A3D9A', // purple
      '#B15928', // brown
      '#A6CEE3', // light blue
      '#B2DF8A', // light green
      '#FB9A99', // light red
      '#FDBF6F', // light orange
      '#CAB2D6', // light purple
      '#80B1D3', // sky blue
      '#8DD3C7', // mint
      '#BEBADA', // lavender
      '#FDB462', // peach
      '#BC80BD'  // lavender
    ];
  
    const data = assetDistribution.map((asset, index) => {
      const name = asset.name || 'Unknown';
      let hexColor = CRYPTO_COLOR_MAP[name] || pieColors[index % pieColors.length];
      
      return {
        name,
        value: asset.value || 0,
        color: hexColor
      };
    });
  
    return { data };
  }, [assetDistribution]);

  // Prepare data for top performers chart
  const prepareTopPerformersData = useCallback(() => {
    return topPerformers.map(performer => ({
      name: performer.symbol,
      value: Math.abs(performer.change_pct),
      isPositive: performer.change_pct >= 0,
      color: performer.change_pct >= 0 ? '#33A02C' : '#E31A1C'
    }));
  }, [topPerformers]);

  // Get icon component for document type
  const getDocumentIcon = useCallback((iconName: string) => {
    const icons: Record<string, any> = {
      'file-text': FileText,
      'clipboard-list': ClipboardList,
      'chart-bar': BarChart,
      'shield-check': Shield,
      'document': File
    };
    return icons[iconName] || File;
  }, []);

  // Error display component
  const ErrorDisplay = ({ message, onRetry }: { message: string; onRetry: () => void }) => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Alert variant="destructive" className="max-w-2xl mx-auto">
        <AlertCircle className="h-5 w-5" />
        <AlertTitle className="text-lg">Error Loading Analytics</AlertTitle>
        <AlertDescription>
          <p className="mt-2">{message}</p>
          <button 
            onClick={onRetry}
            className="mt-4 bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
            aria-label="Retry loading analytics data"
          >
            <RefreshCw size={16} aria-hidden="true" />
            Retry
          </button>
        </AlertDescription>
      </Alert>
    </motion.div>
  );

  // Loading skeleton component
  const LoadingSkeleton = () => (
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
            <h3 className="text-xl font-medium text-muted-foreground mt-4">Loading Analytics</h3>
            <p className="text-sm text-muted-foreground mt-2">Please wait while we process your data...</p>
          </div>
        </div>
      </motion.div>
    </div>
  );

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
                  Crypto Analytics
                </h2>
                <div className="flex h-6 items-center gap-1.5 rounded-full bg-blue-50 px-2.5 text-xs font-medium text-blue-600">
                  <Zap className="h-3.5 w-3.5" />
                  Real-time Data
                </div>
              </div>
              <p className="text-muted-foreground mt-2 text-lg">
                Insights & trend analysis for cryptoasset due diligence
              </p>
            </div>
          
            <div className="flex flex-wrap items-center gap-3">
              <div className="relative">
                <div className="flex items-center border rounded-lg overflow-hidden shadow-sm bg-white" role="radiogroup" aria-label="Date range selection">
                  {DATE_RANGES.map((range) => (
                    <button
                      key={range.value}
                      onClick={() => handleDateRangeChange(range.value)}
                      className={`px-3 py-2 text-sm font-medium transition-colors ${
                        dateRange === range.value
                          ? 'bg-black text-white'
                          : 'bg-white text-gray-600 hover:bg-gray-50'
                      }`}
                      aria-checked={dateRange === range.value}
                      role="radio"
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="relative">
                <button 
                  onClick={toggleFilterMenu}
                  className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50 shadow-sm" 
                  aria-label="Open filter options"
                  aria-expanded={showFilterMenu}
                >
                  <Filter size={16} aria-hidden="true" />
                  <span>Filter {activeSymbols.includes('All') ? '' : `(${activeSymbols.length})`}</span>
                  <ChevronDown size={16} aria-hidden="true" />
                </button>
                
                {showFilterMenu && (
                  <div className="absolute top-full mt-2 right-0 z-10 bg-white rounded-lg shadow-lg border p-4 w-64">
                    <h4 className="text-sm font-medium mb-3">Filter by Asset</h4>
                    <div className="max-h-48 overflow-y-auto">
                      {uniqueSymbols.filter(s => s !== 'All').map((symbol) => (
                        <div key={symbol} className="flex items-center mb-2">
                          <input 
                            type="checkbox" 
                            id={`filter-${symbol}`}
                            checked={filteredSymbols.includes(symbol)}
                            onChange={() => toggleSymbolFilter(symbol)}
                            className="mr-2"
                          />
                          <label htmlFor={`filter-${symbol}`} className="text-sm">{symbol}</label>
                        </div>
                      ))}
                    </div>
                    <div className="flex justify-end gap-2 mt-4">
                      <button 
                        onClick={() => {
                          setFilteredSymbols([]);
                          setShowFilterMenu(false);
                        }}
                        className="px-3 py-1 text-sm text-gray-600 border rounded hover:bg-gray-50"
                      >
                        Clear
                      </button>
                      <button 
                        onClick={applySymbolFilters}
                        className="px-3 py-1 text-sm bg-black text-white rounded hover:bg-gray-800"
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                )}
              </div>
              
              <button 
                onClick={handleExport}
                disabled={isExporting}
                className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-70 shadow-sm" 
                aria-label="Export data"
              >
                {isExporting ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    <span>Exporting...</span>
                  </>
                ) : (
                  <>
                    <Download size={16} aria-hidden="true" />
                    <span>Export</span>
                  </>
                )}
              </button>
            </div>
          </motion.div>
          
          {error && (
            <div className="mt-6">
              <ErrorDisplay message={error} onRetry={handleRetry} />
            </div>
          )}
          
          {loading ? (
            <LoadingSkeleton />
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
                {kpiItems.map((kpi, index) => (
                  <PriceCard
                    key={kpi.label}
                    title={kpi.label}
                    price={kpi.value}
                    change={kpi.change}
                    icon={kpi.icon}
                    type={kpi.trend === 'up' ? 'forecast' : 'current'}
                  />
                ))}
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="lg:col-span-2"
                >
                  <Card className="overflow-hidden bg-white/50 backdrop-blur-lg shadow-xl border-none hover:shadow-2xl transition-all duration-300">
                    <CardHeader className="border-b bg-gray-50/50">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <BarChart4 className="h-5 w-5 text-blue-500" />
                            <CardTitle className="text-xl">Crypto Price Performance</CardTitle>
                          </div>
                          <CardDescription>
                            Percentage change over time for major assets
                          </CardDescription>
                        </div>
                        <div className="hidden md:flex items-center flex-wrap gap-2" role="group" aria-label="Symbol filters">
                          {uniqueSymbols.slice(0, 5).map((symbol) => (
                            <button 
                              key={symbol}
                              onClick={() => handleSymbolFilter(symbol)}
                              className={`px-3 py-1 text-xs rounded-lg ${
                                activeSymbols.includes(symbol) || (symbol === 'All' && activeSymbols.includes('All'))
                                  ? 'bg-black text-white'
                                  : 'bg-white text-gray-600 border hover:bg-gray-50'
                              }`}
                              aria-pressed={activeSymbols.includes(symbol) || (symbol === 'All' && activeSymbols.includes('All'))}
                            >
                              {symbol}
                            </button>
                          ))}
                          {uniqueSymbols.length > 5 && (
                            <button 
                              onClick={toggleFilterMenu}
                              className="px-3 py-1 text-xs rounded-lg bg-white text-gray-600 border hover:bg-gray-50"
                            >
                              +{uniqueSymbols.length - 5} more
                            </button>
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-6">
                      <div className="h-[400px]">
                        {portfolioPerformance.length > 0 ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart 
                              data={preparePricePerformanceData().chartData}
                              margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                              <XAxis 
                                dataKey="date"
                                tick={{ fontSize: 12 }}
                              />
                              <YAxis 
                                tick={{ fontSize: 12 }}
                                tickFormatter={(value) => `${value}%`}
                              />
                              <Tooltip 
                                formatter={(value: number) => [`${value.toFixed(2)}%`, ""]}
                                labelFormatter={(label) => `Date: ${label}`}
                                contentStyle={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                              />
                              <Legend 
                                verticalAlign="top" 
                                height={36}
                                iconType="circle"
                                iconSize={8}
                              />
                              {preparePricePerformanceData().lines.map((line, index) => (
                                <Line
                                  key={index}
                                  type="monotone"
                                  dataKey={line.dataKey}
                                  name={line.name}
                                  stroke={line.stroke}
                                  strokeWidth={line.strokeWidth}
                                  dot={{ r: 2 }}
                                  activeDot={{ r: 4, stroke: line.stroke, strokeWidth: 2, fill: 'white' }}
                                />
                              ))}
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="bg-gray-50 border border-dashed rounded-xl h-full flex items-center justify-center" role="status">
                            <div className="text-center">
                              <BarChart size={48} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                              <p className="text-gray-500">No price performance data available</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <Card className="overflow-hidden bg-white/50 backdrop-blur-lg shadow-xl border-none hover:shadow-2xl transition-all duration-300">
                    <CardHeader className="border-b bg-gray-50/50">
                      <div className="flex items-center justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <PieChart className="h-5 w-5 text-purple-500" />
                            <CardTitle className="text-xl">Asset Distribution</CardTitle>
                          </div>
                          <CardDescription>
                            Market cap breakdown
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-6">
                      <div className="h-[300px] flex items-center justify-center">
                        {assetDistribution.length > 0 ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsChart>
                              <Pie 
                                data={prepareAssetChartData().data}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                outerRadius={100}
                                innerRadius={60}
                                label={({ name, value }) => `${name}: ${value}%`}
                                labelLine={false}
                              >
                                {prepareAssetChartData().data.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <Tooltip
                                formatter={(value: number) => [`${value.toFixed(2)}%`, ""]}
                                contentStyle={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                              />
                            </RechartsChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="text-center" role="status">
                            <PieChart size={48} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                            <p className="text-gray-500">No distribution data available</p>
                          </div>
                        )}
                      </div>
                      
                      <div className="space-y-3 mt-4">
                        {prepareAssetChartData().data.map((asset, index) => (
                          <div key={asset.name} className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <div
                                className="w-3 h-3 rounded-full"
                                style={{ backgroundColor: asset.color }}
                              ></div>
                              <span className="text-sm">{asset.name}</span>
                            </div>
                            <span className="font-medium">{asset.value}%</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
                <InsightCard 
                  title="Recent Market News" 
                  icon={Newspaper}
                  iconColor="blue"
                  className="lg:col-span-2"
                >
                  {recentNews.length > 0 ? (
                    <div className="space-y-4">
                      {recentNews.map((news, index) => (
                        <div key={index} className="border-b pb-4 last:border-0 last:pb-0 hover:bg-gray-50 p-3 rounded-lg transition-colors">
                          <div className="flex justify-between items-start mb-2">
                            <h4 className="font-medium text-gray-800">{news.title}</h4>
                            <span className={`text-xs font-medium ${getSentimentColor(news.sentiment_score).text} ml-2 px-2 py-1 rounded-full bg-opacity-10 ${getSentimentColor(news.sentiment_score).bg}`}>
                              {news.sentiment_score >= 0 ? '+' : ''}{news.sentiment_score}
                            </span>
                          </div>
                          <div className="flex justify-between text-sm text-gray-500">
                            <div className="flex items-center gap-2">
                              <span>{news.source}</span>
                              <span className="text-gray-300">•</span>
                              <div className="flex items-center gap-1">
                                <Clock size={14} aria-hidden="true" />
                                <span>{formatDate(news.date)}</span>
                              </div>
                            </div>
                            <a 
                              href={news.url} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="flex items-center gap-1 text-blue-500 hover:text-blue-700"
                              aria-label={`Read news article: ${news.title}`}
                            >
                              <span>Read</span>
                              <ExternalLink size={14} aria-hidden="true" />
                            </a>
                          </div>
                          {news.related_assets && news.related_assets.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-2">
                              {news.related_assets.map((asset, i) => (
                                <span key={i} className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-full">
                                  {asset}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center" role="status">
                      <Newspaper size={36} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                      <p className="text-gray-500">No recent news available</p>
                    </div>
                  )}
                </InsightCard>
                
                <InsightCard 
                  title="Due Diligence" 
                  icon={FileText}
                  iconColor="green"
                >
                  {dueDiligence.length > 0 ? (
                    <div className="space-y-4">
                      {dueDiligence.map((doc, index) => {
                        const IconComponent = getDocumentIcon(doc.icon);
                        return (
                          <div 
                            key={index} 
                            className="flex items-start gap-3 p-3 rounded-lg border hover:bg-gray-50 transition-colors cursor-pointer"
                            role="button"
                            tabIndex={0}
                            onKeyDown={(e) => e.key === 'Enter' && window.open(doc.source, '_blank')}
                            onClick={() => window.open(doc.source, '_blank')}
                          >
                            <div className="p-2 bg-green-100 text-green-600 rounded" aria-hidden="true">
                              <IconComponent size={20} />
                            </div>
                            <div className="overflow-hidden">
                              <h4 className="font-medium text-gray-800 truncate">{doc.title}</h4>
                              <div className="flex items-center text-sm text-gray-500">
                                <span className="truncate">{doc.document_type}</span>
                                <span className="mx-2 text-gray-300 flex-shrink-0">•</span>
                                <span className="truncate">{doc.source}</span>
                              </div>
                              {doc.keywords && doc.keywords.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {doc.keywords.slice(0, 3).map((keyword, i) => (
                                    <span key={i} className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded-full">
                                      {keyword}
                                    </span>
                                  ))}
                                  {doc.keywords.length > 3 && (
                                    <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded-full">
                                      +{doc.keywords.length - 3}
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center" role="status">
                      <FileText size={36} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                      <p className="text-gray-500">No due diligence documents available</p>
                    </div>
                  )}
                </InsightCard>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
                className="mt-8"
              >
                <Card className="overflow-hidden bg-white/50 backdrop-blur-lg shadow-xl border-none hover:shadow-2xl transition-all duration-300">
                  <CardHeader className="border-b bg-gray-50/50">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-yellow-500" />
                      <CardTitle className="text-xl">Top Currency Performance</CardTitle>
                      <div className="flex h-6 items-center gap-1.5 rounded-full bg-yellow-50 px-2.5 text-xs font-medium text-yellow-600">
                        <Target className="h-3.5 w-3.5" />
                        {dateRange} period
                      </div>
                    </div>
                    <CardDescription>
                      Leading performers over the selected time frame
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6">
                    {topPerformers.length > 0 ? (
                      <div>
                        <div className="h-[300px] mb-6">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartBarChart
                              data={prepareTopPerformersData()}
                              layout="vertical"
                              margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                              <XAxis 
                                type="number"
                                tickFormatter={(value) => `${value}%`}
                              />
                              <YAxis 
                                dataKey="name" 
                                type="category" 
                                width={80}
                                tick={{ fontSize: 12 }}
                              />
                              <Tooltip 
                                formatter={(value: number) => [`${value.toFixed(2)}%`, "Change"]}
                                contentStyle={{ backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                              />
                              <Bar dataKey="value">
                                {prepareTopPerformersData().map((entry, index) => (
                                  <Cell 
                                    key={`cell-${index}`} 
                                    fill={entry.color} 
                                  />
                                ))}
                              </Bar>
                            </RechartBarChart>
                          </ResponsiveContainer>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                          {topPerformers.map((performer, index) => (
                            <div key={index} className={`p-4 rounded-xl bg-white shadow-md transition-transform duration-200 hover:scale-105 border-l-4 ${
                              performer.change_pct >= 0 ? 'border-l-green-500' : 'border-l-red-500'
                            }`}>
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-bold text-sm">{performer.symbol}</span>
                                {performer.change_pct >= 0 ? (
                                  <TrendingUp className="text-green-500" size={16} />
                                ) : (
                                  <TrendingDown className="text-red-500" size={16} />
                                )}
                              </div>
                              <div className={`text-lg font-bold ${performer.color}`}>
                                {performer.change_pct >= 0 ? '+' : ''}{performer.change_pct.toFixed(2)}%
                              </div>
                              <div className="text-xs text-gray-500 mt-1">Last {performer.period}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center" role="status">
                        <BadgePercent size={36} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                        <p className="text-gray-500">No performance data available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}