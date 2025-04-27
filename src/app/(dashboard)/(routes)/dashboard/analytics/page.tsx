"use client";

import { useState, useEffect, useMemo, useCallback } from 'react';
import { Card } from "@/components/ui/card";
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
  ArrowUpRight,
  ArrowDownRight,
  BadgePercent,
  Sparkles
} from "lucide-react";
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
  Bar
} from 'recharts';

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
  color: string;
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

interface AnalyticsData {
  kpis: Kpis;
  asset_distribution: Asset[];
  portfolio_performance: PerformanceItem[];
  recent_news: NewsItem[];
  due_diligence: DueDiligenceItem[];
  error?: string;
}

interface TopPerformer {
  symbol: string;
  change_pct: number;
  period: string;
  color: string;
}

// Date range options
const DATE_RANGES = [
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
  { value: 'YTD', label: 'YTD' },
  { value: '1y', label: '1 Year' }
];

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

  // Custom fetch function
  const fetchAnalyticsData = useCallback(async (): Promise<AnalyticsData> => {
    try {
      const response = await fetch('/api/analytics', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store',
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('API response not OK:', errorData);
        return {
          error: errorData.error || 'Failed to fetch analytics data',
          kpis: {},
          asset_distribution: [],
          portfolio_performance: [],
          recent_news: [],
          due_diligence: [],
        };
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching analytics data:', error);
      return {
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        kpis: {},
        asset_distribution: [],
        portfolio_performance: [],
        recent_news: [],
        due_diligence: [],
      };
    }
  }, []);

  // Load data on component mount or when retry is triggered
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchAnalyticsData();

        if (data.error) {
          setError(data.error);
          console.error("Error from API:", data.error);
        } else {
          setError(null);
        }

        setKpis(data.kpis || {});
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
  }, [retryCount, fetchAnalyticsData]);

  // Get unique symbols for filtering
  const uniqueSymbols = useMemo(() => {
    const symbols = new Set<string>();
    portfolioPerformance.forEach(item => {
      if (item.symbol) {
        const cleanSymbol = item.symbol.replace('USDT', '');
        if (cleanSymbol) symbols.add(cleanSymbol);
      }
    });
    return ['All', ...Array.from(symbols)];
  }, [portfolioPerformance]);

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
    // In a real app, you would refetch data with the new date range
    // Here we'll just simulate it with a delay
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, []);

  const handleCustomDateChange = useCallback(() => {
    setShowDatePicker(false);
    // In a real app, you would use the custom date range to fetch data
    console.log(`Custom date range: ${customDateRange.start} to ${customDateRange.end}`);
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [customDateRange]);

  const handleExport = useCallback(() => {
    setIsExporting(true);
    // Simulate export process
    setTimeout(() => {
      setIsExporting(false);
      // In a real app, you would generate and download a CSV or PDF here
      alert('Data exported successfully!');
    }, 1500);
  }, []);

  const toggleFilterMenu = useCallback(() => {
    setShowFilterMenu(prev => !prev);
  }, []);

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

  // Prepare KPI display with fallbacks
  const kpiItems = useMemo(() => [
    {
      label: 'Market Cap',
      value: kpis.market_cap?.value || '$0M',
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
    
    // Group by symbol
    portfolioPerformance.forEach(item => {
      if (!bySymbol[item.symbol]) {
        bySymbol[item.symbol] = [];
      }
      bySymbol[item.symbol].push(item);
    });
    
    // Calculate change for each symbol
    const performers: TopPerformer[] = [];
    
    Object.entries(bySymbol).forEach(([symbol, items]) => {
      if (items.length < 2) return;
      
      // Sort by date
      items.sort((a, b) => 
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
      
      // Get first and last points
      const first = items[0];
      const last = items[items.length - 1];
      
      // Calculate percentage change
      const change_pct = last.change_pct - first.change_pct;
      
      performers.push({
        symbol: symbol.replace('USDT', ''),
        change_pct,
        period: dateRange,
        color: change_pct >= 0 ? 'text-green-500' : 'text-red-500'
      });
    });
    
    // Sort by performance (descending)
    performers.sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct));
    
    return performers.slice(0, 5);
  }, [portfolioPerformance, dateRange]);

  // Prepare Portfolio Performance Chart Data
  const preparePricePerformanceData = useCallback(() => {
    if (!portfolioPerformance.length) return { chartData: [], lines: [] };

    const colorMap: Record<string, string> = {
      'BTCUSDT': '#f7931a',
      'ETHUSDT': '#627eea',
      'SOLUSDT': '#14f195',
      'ADAUSDT': '#0033ad',
      'XRPUSDT': '#23272a',
      'BNBUSDT': '#f3ba2f',
      'DOGEUSDT': '#c39527',
      'TRXUSDT': '#cd7f32',
      'BCHUSDT': '#8b4513',
      'DEFAULT': '#a0a0a0'
    };

    const validSymbols = new Set<string>();
    portfolioPerformance.forEach(item => {
      if (item.symbol) {
        const cleanSymbol = item.symbol.replace('USDT', '');
        if (activeSymbols.includes('All') || activeSymbols.includes(cleanSymbol)) {
          validSymbols.add(item.symbol);
        }
      }
    });

    const dataByDate: Record<string, any> = {};
    portfolioPerformance.forEach(item => {
      if (!item.symbol || !validSymbols.has(item.symbol)) return;

      try {
        const date = new Date(item.timestamp);
        if (isNaN(date.getTime())) return;

        const dateStr = date.toLocaleDateString('en-US');
        if (!dataByDate[dateStr]) {
          dataByDate[dateStr] = { date: dateStr };
        }

        const cleanSymbol = item.symbol.replace('USDT', '');
        dataByDate[dateStr][cleanSymbol] = item.change_pct;
      } catch (e) {
        console.warn(`Invalid date: ${item.timestamp}`);
      }
    });

    const chartData = Object.values(dataByDate).sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    const lines = Array.from(validSymbols).map(symbol => {
      const cleanSymbol = symbol.replace('USDT', '');
      return {
        name: cleanSymbol,
        dataKey: cleanSymbol,
        stroke: colorMap[symbol] || colorMap.DEFAULT,
        fill: `${colorMap[symbol] || colorMap.DEFAULT}20`,
        type: 'monotone' as const
      };
    });

    return { chartData, lines };
  }, [portfolioPerformance, activeSymbols]);

  // Prepare Asset Distribution Chart Data
  const prepareAssetChartData = useCallback(() => {
    if (!assetDistribution.length) return { data: [] };

    const colorMap: Record<string, string> = {
      'bg-orange-500': '#f97316',
      'bg-indigo-500': '#6366f1',
      'bg-green-500': '#22c55e',
      'bg-gray-500': '#6b7280',
      'bg-blue-500': '#3b82f6',
      'bg-red-500': '#ef4444',
      'bg-yellow-500': '#eab308',
      'bg-purple-500': '#a855f7',
    };

    const data = assetDistribution.map(asset => ({
      name: asset.name || 'Unknown',
      value: asset.value || 0,
      color: colorMap[asset.color] || '#6b7280'
    }));

    return { data };
  }, [assetDistribution]);

  // Prepare data for top performers chart
  const prepareTopPerformersData = useCallback(() => {
    return topPerformers.map(performer => ({
      name: performer.symbol,
      value: Math.abs(performer.change_pct),
      isPositive: performer.change_pct >= 0
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

  // Format date
  const formatDate = useCallback((dateString: string): string => {
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
  }, []);

  // Error display component
  const ErrorDisplay = ({ message, onRetry }: { message: string; onRetry: () => void }) => (
    <div className="bg-red-50 border border-red-200 text-red-800 rounded-xl p-6 flex flex-col items-center" role="alert">
      <AlertCircle className="h-12 w-12 text-red-500 mb-4" aria-hidden="true" />
      <h3 className="text-lg font-semibold mb-2">Error Loading Analytics</h3>
      <p className="text-center mb-4">{message}</p>
      <button 
        onClick={onRetry}
        className="bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
        aria-label="Retry loading analytics data"
      >
        <RefreshCw size={16} aria-hidden="true" />
        Retry
      </button>
    </div>
  );

  // Loading skeleton component
  const LoadingSkeleton = () => (
    <div className="animate-pulse" role="status" aria-label="Loading analytics data">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white p-6 rounded-xl border">
            <div className="flex items-center gap-4">
              <div className="bg-gray-200 p-3 rounded-lg w-12 h-12"></div>
              <div className="space-y-2">
                <div className="h-4 bg-gray-200 rounded w-20"></div>
                <div className="h-6 bg-gray-200 rounded w-24"></div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
        <div className="bg-white p-6 rounded-xl border lg:col-span-2">
          <div className="flex justify-between items-center mb-6">
            <div className="space-y-2">
              <div className="h-5 bg-gray-200 rounded w-40"></div>
              <div className="h-4 bg-gray-200 rounded w-60"></div>
            </div>
            <div className="flex gap-2">
              <div className="h-8 bg-gray-200 rounded w-16"></div>
              <div className="h-8 bg-gray-200 rounded w-16"></div>
            </div>
          </div>
          <div className="h-64 bg-gray-100 rounded-xl"></div>
        </div>
        
        <div className="bg-white p-6 rounded-xl border">
          <div className="flex justify-between items-center mb-6">
            <div className="space-y-2">
              <div className="h-5 bg-gray-200 rounded w-40"></div>
              <div className="h-4 bg-gray-200 rounded w-48"></div>
            </div>
          </div>
          <div className="h-48 bg-gray-100 rounded-xl mb-4"></div>
          <div className="space-y-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="flex justify-between">
                <div className="h-4 bg-gray-200 rounded w-24"></div>
                <div className="h-4 bg-gray-200 rounded w-12"></div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-8">
        <div className="bg-white p-6 rounded-xl border">
          <div className="flex justify-between items-center mb-6">
            <div className="space-y-2">
              <div className="h-5 bg-gray-200 rounded w-40"></div>
              <div className="h-4 bg-gray-200 rounded w-48"></div>
            </div>
          </div>
          <div className="h-32 bg-gray-100 rounded-xl"></div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex-1 p-8 pt-6" role="main">
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto Analytics</h2>
            <p className="text-muted-foreground">
              Insights & trend analysis for cryptoasset due diligence
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <div className="flex items-center border rounded-lg overflow-hidden" role="radiogroup" aria-label="Date range selection">
                {DATE_RANGES.map((range) => (
                  <button
                    key={range.value}
                    onClick={() => handleDateRangeChange(range.value)}
                    className={`px-3 py-2 text-sm font-medium ${
                      dateRange === range.value
                        ? 'bg-black text-white'
                        : 'bg-white text-gray-600 hover:bg-gray-50'
                    }`}
                    aria-checked={dateRange === range.value}
                    role="radio"
                  >
                    {range.value}
                  </button>
                ))}
                <button 
                  onClick={() => setShowDatePicker(prev => !prev)}
                  className="flex items-center gap-2 px-3 py-2 text-sm bg-white text-gray-600 hover:bg-gray-50" 
                  aria-label="Select custom date range"
                  aria-expanded={showDatePicker}
                >
                  <CalendarRange size={16} aria-hidden="true" />
                  <span>Custom</span>
                </button>
              </div>
              
              {showDatePicker && (
                <div className="absolute top-full mt-2 right-0 z-10 bg-white rounded-lg shadow-lg border p-4 w-72">
                  <h4 className="text-sm font-medium mb-3">Custom Date Range</h4>
                  <div className="space-y-4">
                    <div className="flex flex-col">
                      <label className="text-xs text-gray-500 mb-1">Start Date</label>
                      <input 
                        type="date" 
                        value={customDateRange.start}
                        onChange={e => setCustomDateRange(prev => ({ ...prev, start: e.target.value }))}
                        className="border rounded p-2 text-sm"
                      />
                    </div>
                    <div className="flex flex-col">
                      <label className="text-xs text-gray-500 mb-1">End Date</label>
                      <input 
                        type="date" 
                        value={customDateRange.end}
                        onChange={e => setCustomDateRange(prev => ({ ...prev, end: e.target.value }))}
                        className="border rounded p-2 text-sm"
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <button 
                        onClick={() => setShowDatePicker(false)}
                        className="px-3 py-1 text-sm text-gray-600 border rounded hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      <button 
                        onClick={handleCustomDateChange}
                        className="px-3 py-1 text-sm bg-black text-white rounded hover:bg-gray-800"
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="relative">
              <button 
                onClick={toggleFilterMenu}
                className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50" 
                aria-label="Open filter options"
                aria-expanded={showFilterMenu}
              >
                <Filter size={16} aria-hidden="true" />
                <span>Filter</span>
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
              className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50 disabled:opacity-70" 
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
        </div>
        
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
              {kpiItems.map((kpi) => (
                <Card key={kpi.label} className="p-6 hover:shadow-md transition-shadow">
                  <div className="flex items-center gap-4">
                    <div className={`${kpi.bgColor} p-3 rounded-lg`} aria-hidden="true">
                      <kpi.icon className={`w-6 h-6 ${kpi.color}`} />
                    </div>
                    <div>
                      <p className="text-gray-500 text-sm">{kpi.label}</p>
                      <div className="flex items-center gap-2">
                        <h3 className="text-2xl font-bold">{kpi.value}</h3>
                        <span className={`text-xs font-medium ${kpi.trend === 'up' ? 'text-green-500' : kpi.trend === 'down' ? 'text-red-500' : 'text-gray-500'}`}>
                          {kpi.change}
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
              <Card className="p-6 lg:col-span-2 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-semibold">Crypto Price Performance</h3>
                    <p className="text-sm text-gray-500">Percentage change over time for major assets</p>
                  </div>
                  <div className="flex flex-wrap gap-2" role="group" aria-label="Symbol filters">
                    {uniqueSymbols.map((symbol) => (
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
                </div>
              </div>
              
              <div className="h-64">
                {portfolioPerformance.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart 
                      data={preparePricePerformanceData().chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis 
                        dataKey="date"
                        label={{ 
                          value: 'Date', 
                          position: 'insideBottom', 
                          offset: -10 
                        }}
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis 
                        label={{ 
                          value: 'Change %', 
                          angle: -90, 
                          position: 'insideLeft',
                          style: { textAnchor: 'middle' } 
                        }}
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
                          type={line.type}
                          dataKey={line.dataKey}
                          name={line.name}
                          stroke={line.stroke}
                          strokeWidth={2}
                          dot={{ r: 2 }}
                          activeDot={{ r: 4 }}
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
            </Card>
            
            <Card className="p-6 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-semibold">Asset Distribution</h3>
                  <p className="text-sm text-gray-500">Market cap breakdown</p>
                </div>
                <button className="text-gray-400 hover:text-gray-500" aria-label="Toggle asset distribution options">
                  <ChevronDown size={18} />
                </button>
              </div>
              
              <div className="flex items-center justify-center h-48 mb-4">
                {assetDistribution.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <RechartsChart>
                      <Pie 
                        data={prepareAssetChartData().data}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        innerRadius={0}
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
              
              <div className="space-y-3">
                {assetDistribution.map((asset) => (
                  <div key={asset.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${asset.color}`}></div>
                      <span className="text-sm">{asset.name}</span>
                    </div>
                    <span className="font-medium">{asset.value}%</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
            <Card className="p-6 lg:col-span-2 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-semibold">Recent Market News</h3>
                  <p className="text-sm text-gray-500">Latest news with sentiment analysis</p>
                </div>
                <button className="text-gray-400 hover:text-gray-500" aria-label="View more news">
                  <Newspaper size={18} />
                </button>
              </div>
              
              {recentNews.length > 0 ? (
                <div className="space-y-4">
                  {recentNews.map((news, index) => (
                    <div key={index} className="border-b pb-4 last:border-0 last:pb-0 hover:bg-gray-50 p-3 rounded-lg transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <h4 className="font-medium text-gray-800">{news.title}</h4>
                        <span className={`text-xs font-medium ${news.sentiment_color} ml-2 px-2 py-1 rounded-full bg-opacity-10 ${news.sentiment_color.replace('text', 'bg')}`}>
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
            </Card>
            
            <Card className="p-6 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-semibold">Due Diligence</h3>
                  <p className="text-sm text-gray-500">Recent documents & reports</p>
                </div>
                <button className="text-gray-400 hover:text-gray-500" aria-label="View more documents">
                  <FileText size={18} />
                </button>
              </div>
              
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
                      >
                        <div className="p-2 bg-blue-100 text-blue-600 rounded" aria-hidden="true">
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
            </Card>
          </div>

          {/* Top Currency Performance Section */}
          <div className="mt-8">
            <Card className="p-6 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <Sparkles className="text-yellow-500" size={20} />
                    Top Currency Performance
                  </h3>
                  <p className="text-sm text-gray-500">Leading performers over the {dateRange} period</p>
                </div>
              </div>
              
              {topPerformers.length > 0 ? (
                <div>
                  <div className="h-56">
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
                              fill={entry.isPositive ? '#22c55e' : '#ef4444'} 
                            />
                          ))}
                        </Bar>
                      </RechartBarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="mt-4">
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                      {topPerformers.map((performer, index) => (
                        <Card key={index} className={`p-4 border-l-4 ${
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
                        </Card>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center" role="status">
                  <BadgePercent size={36} className="mx-auto text-gray-300 mb-3" aria-hidden="true" />
                  <p className="text-gray-500">No performance data available</p>
                </div>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  </div>
);
}