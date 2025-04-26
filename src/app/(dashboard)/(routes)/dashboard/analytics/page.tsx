"use client";

import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { 
  BarChart, 
  PieChart, 
  TrendingUp, 
  Filter, 
  Download, 
  CalendarRange,
  ChevronDown,
  Search,
  FileText,
  CircleDollarSign,
  Activity,
  AlertCircle,
  RefreshCw,
  Newspaper,
  Clock,
  ExternalLink,
  BarChart2,
  FileText as FileTextIcon,
  ClipboardList,
  ChartBar,
  ShieldCheck,
  File
} from "lucide-react";
import { Bar, Line, Pie } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  ArcElement, 
  Tooltip, 
  Legend, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  PointElement, 
  LineElement 
} from 'chart.js';
import { fetchAnalyticsData, AnalyticsResponse } from '@/services/analytics-service';

// Register Chart.js components
ChartJS.register(
  ArcElement, 
  Tooltip, 
  Legend, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  PointElement,
  LineElement
);

// Define types for chart data
interface ChartPoint {
  timestamp: string;
  change_pct: number;
}

interface ColorSet {
  bg: string;
  border: string;
}

export default function AnalyticsPage() {
  const [dateRange, setDateRange] = useState('30d');
  const [kpis, setKpis] = useState<AnalyticsResponse['kpis']>({});
  const [assetDistribution, setAssetDistribution] = useState<AnalyticsResponse['asset_distribution']>([]);
  const [portfolioPerformance, setPortfolioPerformance] = useState<AnalyticsResponse['portfolio_performance']>([]);
  const [recentNews, setRecentNews] = useState<AnalyticsResponse['recent_news']>([]);
  const [dueDiligence, setDueDiligence] = useState<AnalyticsResponse['due_diligence']>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

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
  }, [retryCount]);

  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
  };

  // Prepare KPI display with fallbacks
  const kpiItems = [
    {
      label: 'Market Cap',
      value: kpis.market_cap?.value || '$0M',
      change: kpis.market_cap?.change || '+0%',
      trend: kpis.market_cap?.trend || 'up',
      icon: CircleDollarSign,
      color: 'text-green-500',
      bgColor: 'bg-green-100'
    },
    {
      label: 'Asset Count',
      value: kpis.asset_count?.value || '0',
      change: kpis.asset_count?.change || '+0',
      trend: kpis.asset_count?.trend || 'up',
      icon: FileText,
      color: 'text-blue-500',
      bgColor: 'bg-blue-100'
    },
    {
      label: 'Price Change',
      value: kpis.price_change?.value || '0%',
      change: kpis.price_change?.change || '+0%',
      trend: kpis.price_change?.trend || 'up',
      icon: BarChart2,
      color: 'text-violet-500',
      bgColor: 'bg-violet-100'
    },
    {
      label: 'Market Sentiment',
      value: kpis.market_sentiment?.value || '50%',
      change: kpis.market_sentiment?.change || '+0%',
      trend: kpis.market_sentiment?.trend || 'up',
      icon: Activity,
      color: 'text-pink-500',
      bgColor: 'bg-pink-100'
    }
  ];

  // Prepare Portfolio Performance Chart Data
  const preparePricePerformanceData = () => {
    // Group data by symbol
    const dataBySymbol: Record<string, ChartPoint[]> = {};
    
    portfolioPerformance.forEach(item => {
      const symbol = item.symbol || 'Unknown';
      if (!dataBySymbol[symbol]) {
        dataBySymbol[symbol] = [];
      }
      dataBySymbol[symbol].push({
        timestamp: item.timestamp,
        change_pct: item.change_pct
      });
    });
    
    // Prepare datasets
    const datasets: ChartDataset[] = [];
    const allLabels = new Set<string>();
    
    // Generate colors for each symbol
    const colorMap: Record<string, ColorSet> = {
      'BTCUSDT': { bg: 'rgba(247, 147, 26, 0.2)', border: 'rgba(247, 147, 26, 1)' },
      'ETHUSDT': { bg: 'rgba(98, 126, 234, 0.2)', border: 'rgba(98, 126, 234, 1)' },
      'SOLUSDT': { bg: 'rgba(20, 241, 149, 0.2)', border: 'rgba(20, 241, 149, 1)' },
      'ADAUSDT': { bg: 'rgba(0, 51, 173, 0.2)', border: 'rgba(0, 51, 173, 1)' },
      'XRPUSDT': { bg: 'rgba(35, 31, 32, 0.2)', border: 'rgba(35, 31, 32, 1)' },
      'BNBUSDT': { bg: 'rgba(243, 186, 47, 0.2)', border: 'rgba(243, 186, 47, 1)' },
      'DOGEUSDT': { bg: 'rgba(195, 149, 39, 0.2)', border: 'rgba(195, 149, 39, 1)' },
      'DEFAULT': { bg: 'rgba(160, 160, 160, 0.2)', border: 'rgba(160, 160, 160, 1)' }
    };
    
    // Sort data points by timestamp for each symbol
    Object.entries(dataBySymbol).forEach(([symbol, dataPoints]) => {
      // Sort by timestamp
      dataPoints.sort((a, b) => {
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      });
      
      // Add all timestamps to allLabels
      dataPoints.forEach(dp => {
        try {
          allLabels.add(new Date(dp.timestamp).toLocaleDateString());
        } catch (e) {
          console.error(`Invalid date: ${dp.timestamp}`);
        }
      });
      
      // Get color for this symbol
      const color = colorMap[symbol] || colorMap.DEFAULT;
      
      // Create dataset
      datasets.push({
        label: symbol.replace('USDT', ''),
        data: dataPoints.map(dp => ({
          x: new Date(dp.timestamp).toLocaleDateString(),
          y: dp.change_pct
        })),
        backgroundColor: color.bg,
        borderColor: color.border,
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.3,
        fill: true
      });
    });
    
    return {
      labels: Array.from(allLabels).sort((a, b) => new Date(a).getTime() - new Date(b).getTime()),
      datasets
    };
  };

  // Prepare Asset Distribution Chart Data
  const assetChartData = {
    labels: assetDistribution.map((asset) => asset.name),
    datasets: [
      {
        data: assetDistribution.map((asset) => asset.value),
        backgroundColor: assetDistribution.map((asset) => {
          // Map color class names to actual color values
          const colorMap: Record<string, string> = {
            'bg-orange-500': 'rgba(249, 115, 22, 0.8)',
            'bg-indigo-500': 'rgba(99, 102, 241, 0.8)',
            'bg-green-500': 'rgba(34, 197, 94, 0.8)',
            'bg-gray-500': 'rgba(107, 114, 128, 0.8)',
            'bg-blue-500': 'rgba(59, 130, 246, 0.8)',
            'bg-red-500': 'rgba(239, 68, 68, 0.8)',
            'bg-yellow-500': 'rgba(234, 179, 8, 0.8)',
            'bg-purple-500': 'rgba(168, 85, 247, 0.8)',
          };
          return colorMap[asset.color] || 'rgba(107, 114, 128, 0.8)';
        }),
        borderWidth: 1
      }
    ]
  };

  // Get icon component for document type
  const getDocumentIcon = (iconName: string) => {
    const icons: Record<string, React.ElementType> = {
      'file-text': FileTextIcon,
      'clipboard-list': ClipboardList,
      'chart-bar': ChartBar,
      'shield-check': ShieldCheck,
      'document': File
    };
    return icons[iconName] || File;
  };

  interface ChartDataset {
    label: string;
    data: { x: string; y: number }[];
    backgroundColor: string;
    borderColor: string;
    borderWidth: number;
    pointRadius: number;
    tension: number;
    fill: boolean;
  }
  
  // Error display component
  const ErrorDisplay = ({ message, onRetry }: { message: string, onRetry: () => void }) => (
    <div className="bg-red-50 border border-red-200 text-red-800 rounded-xl p-6 flex flex-col items-center">
      <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
      <h3 className="text-lg font-semibold mb-2">Error Loading Analytics</h3>
      <p className="text-center mb-4">{message}</p>
      <button 
        onClick={onRetry}
        className="bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
      >
        <RefreshCw size={16} />
        Retry
      </button>
    </div>
  );

  // Loading skeleton component
  const LoadingSkeleton = () => (
    <div className="animate-pulse">
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
    </div>
  );
  
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto Analytics</h2>
            <p className="text-muted-foreground">
              Insights & trend analysis for cryptoasset due diligence
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center border rounded-lg overflow-hidden">
              {['7d', '30d', '90d', 'YTD', '1y'].map((range) => (
                <button
                  key={range}
                  onClick={() => setDateRange(range)}
                  className={`px-3 py-2 text-sm font-medium ${
                    dateRange === range
                      ? 'bg-black text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {range}
                </button>
              ))}
              <button className="flex items-center gap-2 px-3 py-2 text-sm bg-white text-gray-600 hover:bg-gray-50">
                <CalendarRange size={16} />
                <span>Custom</span>
              </button>
            </div>
            
            <button className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
              <Filter size={16} />
              <span>Filter</span>
              <ChevronDown size={16} />
            </button>
            
            <button className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
              <Download size={16} />
              <span>Export</span>
            </button>
          </div>
        </div>
        
        {/* Error display */}
        {error && (
          <div className="mt-6">
            <ErrorDisplay message={error} onRetry={handleRetry} />
          </div>
        )}
        
        {/* Content area with loading state */}
        {loading ? (
          <LoadingSkeleton />
        ) : (
          <>
            {/* Key Performance Indicators */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
              {kpiItems.map((kpi) => (
                <Card key={kpi.label} className="p-6">
                  <div className="flex items-center gap-4">
                    <div className={`${kpi.bgColor} p-3 rounded-lg`}>
                      <kpi.icon className={`w-6 h-6 ${kpi.color}`} />
                    </div>
                    <div>
                      <p className="text-gray-500 text-sm">{kpi.label}</p>
                      <div className="flex items-center gap-2">
                        <h3 className="text-2xl font-bold">{kpi.value}</h3>
                        <span className={`text-xs font-medium ${kpi.trend === 'up' ? 'text-green-500' : 'text-red-500'}`}>
                          {kpi.change}
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            
            {/* Charts and News Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
              {/* Price Performance Chart */}
              <Card className="p-6 lg:col-span-2">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-semibold">Crypto Price Performance</h3>
                    <p className="text-sm text-gray-500">Percentage change over time for major assets</p>
                  </div>
                  <div className="flex gap-2">
                    <button className="bg-gray-100 text-gray-600 px-3 py-1 text-xs rounded-lg">
                      All
                    </button>
                    {portfolioPerformance.length > 0 && 
                      Array.from(new Set(portfolioPerformance.map(data => data.symbol?.replace("USDT", "") || "")))
                      .slice(0, 3)
                      .map((symbol) => (
                        <button key={symbol} className="bg-white text-gray-600 px-3 py-1 text-xs rounded-lg border">
                          {symbol}
                        </button>
                      ))
                    }
                  </div>
                </div>
                
                <div className="h-64">
                  {portfolioPerformance.length > 0 ? (
                    <Line 
                      data={preparePricePerformanceData()} 
                      options={{ 
                        maintainAspectRatio: false,
                        scales: {
                          y: {
                            title: {
                              display: true,
                              text: 'Change %'
                            }
                          }
                        },
                        interaction: {
                          mode: 'index',
                          intersect: false,
                        },
                        plugins: {
                          tooltip: {
                            callbacks: {
                              label: function(context: any) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
                              }
                            }
                          }
                        }
                      }} 
                    />
                  ) : (
                    <div className="bg-gray-50 border border-dashed rounded-xl h-full flex items-center justify-center">
                      <div className="text-center">
                        <BarChart size={48} className="mx-auto text-gray-300 mb-3" />
                        <p className="text-gray-500">No price performance data available</p>
                      </div>
                    </div>
                  )}
                </div>
              </Card>
              
              {/* Asset Distribution Chart */}
              <Card className="p-6">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-semibold">Asset Distribution</h3>
                    <p className="text-sm text-gray-500">Market cap breakdown</p>
                  </div>
                  <button className="text-gray-400 hover:text-gray-500">
                    <ChevronDown size={18} />
                  </button>
                </div>
                
                <div className="flex items-center justify-center h-48 mb-4">
                  {assetDistribution.length > 0 ? (
                    <Pie data={assetChartData} options={{ maintainAspectRatio: false }} />
                  ) : (
                    <div className="text-center">
                      <PieChart size={48} className="mx-auto text-gray-300 mb-3" />
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
            
            {/* Market News & Due Diligence Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
              {/* Recent News */}
              <Card className="p-6 lg:col-span-2">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-semibold">Recent Market News</h3>
                    <p className="text-sm text-gray-500">Latest news with sentiment analysis</p>
                  </div>
                  <button className="text-gray-400 hover:text-gray-500">
                    <Newspaper size={18} />
                  </button>
                </div>
                
                {recentNews && recentNews.length > 0 ? (
                  <div className="space-y-4">
                    {recentNews.map((news, index) => (
                      <div key={index} className="border-b pb-4 last:border-0 last:pb-0">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-medium text-gray-800">{news.title}</h4>
                          <span className={`text-xs font-medium ${news.sentiment_color} ml-2`}>
                            {news.sentiment_score >= 0 ? '+' : ''}{news.sentiment_score}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm text-gray-500">
                          <div className="flex items-center gap-2">
                            <span>{news.source}</span>
                            <span className="text-gray-300">•</span>
                            <div className="flex items-center gap-1">
                              <Clock size={14} />
                              <span>{new Date(news.date).toLocaleDateString()}</span>
                            </div>
                          </div>
                          <a href={news.url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 text-blue-500 hover:text-blue-700">
                            <span>Read</span>
                            <ExternalLink size={14} />
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
                  <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center">
                    <Newspaper size={36} className="mx-auto text-gray-300 mb-3" />
                    <p className="text-gray-500">No recent news available</p>
                  </div>
                )}
              </Card>
              
              {/* Due Diligence Documents */}
              <Card className="p-6">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-semibold">Due Diligence</h3>
                    <p className="text-sm text-gray-500">Recent documents & reports</p>
                  </div>
                  <button className="text-gray-400 hover:text-gray-500">
                    <FileTextIcon size={18} />
                  </button>
                </div>
                
                {dueDiligence && dueDiligence.length > 0 ? (
                  <div className="space-y-4">
                    {dueDiligence.map((doc, index) => {
                      const IconComponent = getDocumentIcon(doc.icon);
                      return (
                        <div key={index} className="flex items-start gap-3 p-3 rounded-lg border hover:bg-gray-50 transition-colors cursor-pointer">
                          <div className="p-2 bg-blue-100 text-blue-600 rounded">
                            <IconComponent size={20} />
                          </div>
                          <div>
                            <h4 className="font-medium text-gray-800">{doc.title}</h4>
                            <div className="flex items-center text-sm text-gray-500">
                              <span>{doc.document_type}</span>
                              <span className="mx-2 text-gray-300">•</span>
                              <span>{doc.source}</span>
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
                  <div className="bg-gray-50 border border-dashed rounded-xl p-8 text-center">
                    <FileTextIcon size={36} className="mx-auto text-gray-300 mb-3" />
                    <p className="text-gray-500">No due diligence documents available</p>
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