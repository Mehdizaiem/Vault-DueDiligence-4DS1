'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Line, Pie } from 'react-chartjs-2';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend, 
  PointElement, 
  LineElement, 
  ArcElement 
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend, 
  PointElement, 
  LineElement,
  ArcElement
);

type NewsItem = {
  id?: string;
  title: string;
  source: string;
  date: string;
  content: string;
  sentiment_label: string;
  sentiment_score: number;
  aspect?: string;
  url?: string;
  explanation?: string;
  cryptocurrency?: string; // Added field for filtering
};

// Number of items to load per infinite scroll trigger
const LOAD_INCREMENT = 6;

export default function NewsDashboard() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false); // New state for refresh animation
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [displayCount, setDisplayCount] = useState(LOAD_INCREMENT);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('news'); // Added for tab navigation
  const [cryptoFilter, setCryptoFilter] = useState('all'); // For crypto type filtering
  const [sourceFilter, setSourceFilter] = useState('all'); // For source filtering
  
  // Refs for infinite scroll
  const observer = useRef<IntersectionObserver | null>(null);
  const lastNewsElementRef = useCallback((node: HTMLDivElement | null) => {
    if (loading) return;
    if (observer.current) observer.current.disconnect();
    
    observer.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && filtered.length > displayCount) {
        setDisplayCount(prevCount => prevCount + LOAD_INCREMENT);
      }
    });
    
    if (node) observer.current.observe(node);
  }, [loading, displayCount]);

  const toggleExpanded = (id: string) => {
    setExpanded(prev => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const fetchNews = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/news_sentiment');
      const data = await res.json();
      if (Array.isArray(data.articles)) setNews(data.articles);
    } catch (err) {
      console.error('Fetch failed', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true); // Start refresh animation
    try {
      await fetch('/api/refresh_news');
      await fetchNews();
    } catch (err) {
      console.error('Refresh failed:', err);
    } finally {
      setRefreshing(false); // End refresh animation
      
      // Show toast notification
      const toast = document.getElementById('refresh-toast');
      if (toast) {
        toast.classList.remove('translate-y-16', 'opacity-0');
        toast.classList.add('translate-y-0', 'opacity-100');
        
        // Hide toast after 3 seconds
        setTimeout(() => {
          toast.classList.add('translate-y-16', 'opacity-0');
          toast.classList.remove('translate-y-0', 'opacity-100');
        }, 3000);
      }
    }
  };

  const formatDate = (d: string) => {
    const date = new Date(d);
    return isNaN(date.getTime()) ? 'N/A' : date.toLocaleDateString();
  };

  const getSentimentColor = (label?: string) => {
    const sentiment = label?.toLowerCase();
    if (sentiment === 'positive') return { bg: 'bg-green-100', text: 'text-green-800' };
    if (sentiment === 'negative') return { bg: 'bg-red-100', text: 'text-red-800' };
    return { bg: 'bg-yellow-100', text: 'text-yellow-800' }; // default to neutral
  };
  

  // All available cryptocurrencies
  const cryptocurrencies = ['Bitcoin', 'Ethereum', 'XRP', 'BNB', 'SOL', 'DOGE', 'ADA', 'LINK', 'AVAX'];
  
  // All available news sources
  const newsSources = Array.from(new Set(news.map(item => item.source))).sort();

  // Filtering logic with added filters
  const filtered = news.filter(item => {
    const textMatch = 
      item.title?.toLowerCase().includes(search.toLowerCase()) ||
      item.content?.toLowerCase().includes(search.toLowerCase());
    
    const sentimentMatch = 
      filter === 'all' ? true : item.sentiment_label?.toLowerCase() === filter;

    const cryptoMatch = 
      cryptoFilter === 'all' ? true : 
      (item.cryptocurrency?.toLowerCase() === cryptoFilter.toLowerCase() ||
       item.title?.toLowerCase().includes(cryptoFilter.toLowerCase()) ||
       item.content?.toLowerCase().includes(cryptoFilter.toLowerCase()));

    const sourceMatch = 
      sourceFilter === 'all' ? true : item.source === sourceFilter;
      
    return textMatch && sentimentMatch && cryptoMatch && sourceMatch;
  });

  // Items for display with infinite scroll
  const displayItems = filtered.slice(0, displayCount);

  // Function to export data as CSV
  const exportCSV = () => {
    const headers = ['Title', 'Source', 'Date', 'Sentiment', 'Score', 'Content', 'URL'];
    const csvContent = filtered.map(item => 
      [
        `"${item.title.replace(/"/g, '""')}"`,
        item.source,
        item.date,
        item.sentiment_label,
        item.sentiment_score,
        `"${item.content.replace(/"/g, '""')}"`,
        item.url || ''
      ].join(',')
    );
    
    const csv = [headers.join(','), ...csvContent].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `crypto_news_sentiment_${new Date().toISOString().split('T')[0]}.csv`);
    link.click();
  };

  // Function to export data as JSON
  const exportJSON = () => {
    const dataStr = JSON.stringify(filtered, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `crypto_news_sentiment_${new Date().toISOString().split('T')[0]}.json`);
    link.click();
  };

  // For sentiment visualization 
  const sentimentCounts = {
    positive: filtered.filter(item => item.sentiment_label?.toLowerCase().includes('positive')).length,
    neutral: filtered.filter(item => item.sentiment_label?.toLowerCase().includes('neutral')).length,
    negative: filtered.filter(item => item.sentiment_label?.toLowerCase().includes('negative')).length
  };
  

  // Group by date for trend chart
  const sentimentByDate = filtered.reduce((acc, item) => {
    const date = item.date.split('T')[0];
    if (!acc[date]) {
      acc[date] = { positive: 0, neutral: 0, negative: 0, total: 0 };
    }
    
    if (item.sentiment_label?.toLowerCase().includes('positive')) {
      acc[date].positive++;
    } else if (item.sentiment_label?.toLowerCase().includes('negative')) {
      acc[date].negative++;
    } else {
      acc[date].neutral++;
    }
    
    
    acc[date].total++;
    return acc;
  }, {} as Record<string, { positive: number, neutral: number, negative: number, total: number }>);

  // Sort dates for trend chart
  const sortedDates = Object.keys(sentimentByDate).sort();

  // Chart data
  const pieChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [sentimentCounts.positive, sentimentCounts.neutral, sentimentCounts.negative],
        backgroundColor: ['rgba(52, 211, 153, 0.8)', 'rgba(251, 191, 36, 0.8)', 'rgba(239, 68, 68, 0.8)'],
        borderColor: ['rgb(16, 185, 129)', 'rgb(245, 158, 11)', 'rgb(220, 38, 38)'],
        borderWidth: 1,
      },
    ],
  };

  const trendChartData = {
    labels: sortedDates,
    datasets: [
      {
        label: 'Positive',
        data: sortedDates.map(date => sentimentByDate[date].positive),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(52, 211, 153, 0.5)',
        tension: 0.2,
      },
      {
        label: 'Neutral',
        data: sortedDates.map(date => sentimentByDate[date].neutral),
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(251, 191, 36, 0.5)',
        tension: 0.2,
      },
      {
        label: 'Negative',
        data: sortedDates.map(date => sentimentByDate[date].negative),
        borderColor: 'rgb(220, 38, 38)',
        backgroundColor: 'rgba(239, 68, 68, 0.5)',
        tension: 0.2,
      },
    ],
  };

  return (
    <main className="bg-gray-50 min-h-screen">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <header className="mb-6">
          <div className="flex justify-between items-center mb-3">
            <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-2">
              📊 Crypto News Sentiment Dashboard
            </h1>
            
            {/* Enhanced refresh button with dynamic effects */}
            <div className="flex items-center">
              <span className="mr-2 text-sm text-gray-500">
                Last updated: {new Date().toLocaleDateString()}
              </span>
              <button
                onClick={handleRefresh}
                disabled={loading || refreshing}
                className={`
                  flex items-center gap-2
                  text-sm font-medium
                  px-4 py-2 rounded-lg
                  transition-all duration-300 ease-in-out
                  shadow-md hover:shadow-lg active:scale-95
                  ${refreshing 
                    ? 'bg-blue-200 text-blue-700 cursor-not-allowed' 
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                  }
                `}
                
              >
                <span className={`
                  inline-block
                  transition-transform
                  ${refreshing ? 'animate-spin' : 'group-hover:rotate-180'}
                `}>
                  {refreshing ? '⟳' : '↻'}
                </span>
                
              </button>
            </div>
          </div>

          {/* Enhanced Tabs */}
          <div className="border-b border-gray-200 mb-4">
            <ul className="flex flex-wrap -mb-px">
              <li className="mr-2">
                <button 
                  onClick={() => setActiveTab('news')}
                  className={`
                    inline-block py-2 px-4 text-sm font-medium 
                    transition-all duration-300 ease-in-out
                    relative
                    ${activeTab === 'news'
                      ? 'text-blue-600 border-b-2 border-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-1 after:bg-blue-600 after:rounded-t-lg'
                      : 'text-gray-500 hover:text-gray-600 hover:border-gray-300'}
                  `}
                >
                  News Feed
                </button>
              </li>
              <li className="mr-2">
                <button 
                  onClick={() => setActiveTab('analytics')}
                  className={`
                    inline-block py-2 px-4 text-sm font-medium 
                    transition-all duration-300 ease-in-out
                    relative
                    ${activeTab === 'analytics'
                      ? 'text-blue-600 border-b-2 border-blue-600 after:absolute after:bottom-0 after:left-0 after:right-0 after:h-1 after:bg-blue-600 after:rounded-t-lg'
                      : 'text-gray-500 hover:text-gray-600 hover:border-gray-300'}
                  `}
                >
                  Sentiment Analytics
                </button>
              </li>
            </ul>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-wrap gap-4 items-center justify-between mb-4">
            <input
              type="text"
              placeholder="🔍 Search title or content..."
              className="flex-1 p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setDisplayCount(LOAD_INCREMENT);
              }}
            />
            
            <div className="flex flex-wrap gap-2">
              {/* Export buttons with dynamic effects */}
              <div className="bg-white p-2 rounded-lg shadow-sm flex gap-2">
                <button 
                  onClick={exportCSV}
                  className="
                    relative overflow-hidden
                    px-3 py-1.5 
                    text-sm font-medium 
                    bg-gradient-to-r from-green-400 to-green-500
                    text-white 
                    rounded-md
                    transition-all duration-300
                    hover:shadow-md hover:from-green-500 hover:to-green-600
                    active:scale-95
                    before:absolute before:inset-0 before:bg-white before:opacity-0 before:hover:opacity-10 before:transition-opacity
                  "
                >
                  Export CSV
                </button>
                <button 
                  onClick={exportJSON}
                  className="
                    relative overflow-hidden
                    px-3 py-1.5 
                    text-sm font-medium 
                    bg-gradient-to-r from-purple-400 to-purple-500
                    text-white 
                    rounded-md
                    transition-all duration-300
                    hover:shadow-md hover:from-purple-500 hover:to-purple-600
                    active:scale-95
                    before:absolute before:inset-0 before:bg-white before:opacity-0 before:hover:opacity-10 before:transition-opacity
                  "
                >
                  Export JSON
                </button>
              </div>
            </div>
          </div>

          {/* Additional Filters with Dynamic Buttons */}
          <div className="bg-white p-4 rounded-lg shadow-sm mb-4 flex flex-wrap gap-4">
            {/* Sentiment Filter */}
            <div>
              <span className="text-sm text-gray-600 mr-2">Sentiment:</span>
              {['all', 'positive', 'neutral', 'negative'].map(type => {
                // Define color schemes for each sentiment type
                const colors = {
                  all: filter === 'all' ? 'from-blue-400 to-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100',
                  positive: filter === 'positive' ? 'from-green-400 to-green-500 text-white' : 'text-gray-700 hover:bg-gray-100',
                  neutral: filter === 'neutral' ? 'from-amber-400 to-amber-500 text-white' : 'text-gray-700 hover:bg-gray-100',
                  negative: filter === 'negative' ? 'from-rose-400 to-rose-500 text-white' : 'text-gray-700 hover:bg-gray-100'
                };
                
                const colorClass = filter === type ? 
                  colors[type as keyof typeof colors] : 
                  'text-gray-700 hover:from-gray-100 hover:to-gray-200';
                
                return (
                  <button
                    key={type}
                    onClick={() => {
                      setFilter(type);
                      setDisplayCount(LOAD_INCREMENT);
                    }}
                    className={`
                      relative overflow-hidden
                      px-3 py-1.5 
                      text-sm font-medium 
                      rounded-md mx-1
                      transition-all duration-300
                      shadow-sm
                      hover:shadow
                      active:scale-95
                      ${filter === type ? 
                        'bg-blue-600 text-white' : 
                        'bg-white text-gray-700 hover:bg-blue-100'}
                      
                    `}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                );
              })}
            </div>

            {/* Cryptocurrency Filter */}
            <div>
              <span className="text-sm text-gray-600 mr-2">Cryptocurrency:</span>
              <select
                value={cryptoFilter}
                onChange={(e) => {
                  setCryptoFilter(e.target.value);
                  setDisplayCount(LOAD_INCREMENT);
                }}
                className="
                  px-3 py-1.5 
                  text-sm font-medium 
                  bg-white border rounded-md
                  shadow-sm
                  transition-all duration-300
                  focus:ring-2 focus:ring-blue-300 focus:border-blue-500
                  cursor-pointer
                  appearance-none
                  bg-no-repeat bg-right
                  pr-8
                "
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236B7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                  backgroundSize: '1.5em'
                }}
              >
                <option value="all">All Cryptocurrencies</option>
                {cryptocurrencies.map(crypto => (
                  <option key={crypto} value={crypto}>{crypto}</option>
                ))}
              </select>
            </div>

            {/* Source Filter */}
            <div>
              <span className="text-sm text-gray-600 mr-2">Source:</span>
              <select
                value={sourceFilter}
                onChange={(e) => {
                  setSourceFilter(e.target.value);
                  setDisplayCount(LOAD_INCREMENT);
                }}
                className="
                  px-3 py-1.5 
                  text-sm font-medium 
                  bg-white border rounded-md
                  shadow-sm
                  transition-all duration-300
                  focus:ring-2 focus:ring-blue-300 focus:border-blue-500
                  cursor-pointer
                  appearance-none
                  bg-no-repeat bg-right
                  pr-8
                "
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%236B7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                  backgroundSize: '1.5em'
                }}
              >
                <option value="all">All Sources</option>
                {newsSources.map(source => (
                  <option key={source} value={source}>{source}</option>
                ))}
              </select>
            </div>
          </div>
        </header>

        {/* Success Toast Notification */}
        <div 
          id="refresh-toast" 
          className="fixed bottom-5 right-5 bg-gradient-to-r from-green-100 to-green-200 border-l-4 border-green-500 text-green-700 p-4 shadow-md rounded transition-all duration-300 transform translate-y-16 opacity-0 z-50 flex items-center"
        >
          <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          <span>Data refreshed successfully!</span>
        </div>

        {/* Loader */}
        {loading ? (
          <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-400 border-t-transparent"></div>
        </div>
        
        ) : (
          <>
            {activeTab === 'news' ? (
              /* Articles Grid */
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {displayItems.map((item, idx) => {
                    const { bg, text } = getSentimentColor(item.sentiment_label);

                    <span className={`${bg} ${text} px-2 py-1 rounded-full`}>
                      {item.sentiment_label}
                    </span>
                    
                    const isLastElement = idx === displayItems.length - 1;
                    
                    return (
                      <div
                        key={item.id || idx}
                        ref={isLastElement ? lastNewsElementRef : null}
                        className="bg-white rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition"
                      >
                        <div className="px-6 py-4">
                          <div className="flex justify-between items-start mb-2">
                            <h2 className="font-bold text-xl text-gray-800">{item.title}</h2>
                            <span className="text-xs bg-gray-100 px-2 py-1 rounded-full text-gray-700">{item.source}</span>
                          </div>
                          <p className="text-gray-600 mb-3 line-clamp-3">{item.content}</p>

                          <div className="flex flex-wrap gap-2 mb-3 text-sm">
                            {item.sentiment_label && (
                              <span className={`${bg} ${text} text-xs px-3 py-1 rounded-full shadow-sm font-medium`}>
                              {item.sentiment_label}
                            </span>
                            )}

                            {item.sentiment_score != null && !isNaN(item.sentiment_score) && (
                              <span className="bg-gray-200 text-gray-800 text-xs px-3 py-1 rounded-full shadow-sm font-medium">
                              Score: {item.sentiment_score.toFixed(2)}
                            </span>
                            )}

                            {item.aspect && (
                              <span className="bg-purple-100 text-purple-800 text-xs px-3 py-1 rounded-full shadow-sm font-medium">
                              {item.aspect}
                            </span>
                            )}
                            {item.cryptocurrency && (
                              <span className="bg-blue-100 text-blue-800 text-xs px-3 py-1 rounded-full shadow-sm font-medium">
                              {item.cryptocurrency}
                            </span>
                            )}
                          </div>

                          {/* Explanation toggle */}
                          {item.explanation && (
                            <button
                            onClick={() => toggleExpanded(item.id || idx.toString())}
                            className="
                              px-3 py-1 text-xs font-medium
                              text-blue-600 hover:text-white
                              border border-blue-600 hover:bg-blue-600
                              rounded-full transition-all duration-300
                              shadow-sm hover:shadow-md active:scale-95
                            "
                          >
                            {expanded.has(item.id || idx.toString()) ? "Hide Explanation" : "Show Explanation"}
                          </button>
                          
                          )}
                          {expanded.has(item.id || idx.toString()) && (
                            <p className="text-sm text-gray-700 mt-2 italic">{item.explanation}</p>
                          )}
                        </div>

                        <div className="bg-gray-50 px-6 py-3 flex justify-between items-center border-t">
                          <span className="text-xs text-gray-500">{formatDate(item.date)}</span>
                          {item.url && (
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="
                                text-blue-600 text-sm 
                                transition-all duration-300
                                relative
                                after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:bg-blue-600
                                after:scale-x-0 after:origin-bottom-right after:transition-transform
                                hover:after:scale-x-100 hover:after:origin-bottom-left
                              "
                            >
                              Read more →
                            </a>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {/* Loading indicator for infinite scroll */}
                {displayCount < filtered.length && (
                  <div className="flex justify-center my-6">
                    <div className="animate-pulse text-gray-500">Loading more...</div>
                  </div>
                )}
                
                {/* No results message */}
                {filtered.length === 0 && (
                  <div className="text-center py-10">
                    <p className="text-gray-500 text-xl">No news items match your current filters</p>
                  </div>
                )}
              </>
            ) : (
              
              /* Analytics View */
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Sentiment Analysis</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Pie Chart */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Sentiment Distribution</h3>
                    <div className="h-64">
                      <Pie 
                        data={pieChartData} 
                        options={{
                          plugins: {
                            legend: {
                              position: 'right',
                            },
                            tooltip: {
                              callbacks: {
                                label: function(tooltipItem) {
                                  const value = tooltipItem.raw as number;
                                  const total = sentimentCounts.positive + sentimentCounts.neutral + sentimentCounts.negative;
                                  const percentage = Math.round((value / total) * 100);
                                  return `${tooltipItem.label}: ${value} (${percentage}%)`;
                                }
                              }
                            }
                          },
                          maintainAspectRatio: false,
                        }}
                      />
                    </div>
                  </div>
                  
                  {/* Trend Chart */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Sentiment Trends Over Time</h3>
                    <div className="h-64">
                      <Line 
                        data={trendChartData}
                        options={{
                          plugins: {
                            legend: {
                              position: 'top',
                            },
                          },
                          scales: {
                            y: {
                              beginAtZero: true
                            }
                          },
                          maintainAspectRatio: false,
                        }}
                      />
                    </div>
                  </div>
                  
                  {/* Summary Statistics */}
                  <div className="lg:col-span-2 mt-4">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Summary Statistics</h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-100">
                        <h4 className="text-green-800 font-medium">Positive News</h4>
                        <p className="text-3xl font-bold text-green-600">{sentimentCounts.positive}</p>
                        <p className="text-sm text-green-700">
                          {Math.round(sentimentCounts.positive / filtered.length * 100) || 0}% of total
                        </p>
                      </div>
                      
                      <div className="bg-amber-50 p-4 rounded-lg border border-amber-100">
                        <h4 className="text-amber-800 font-medium">Neutral News</h4>
                        <p className="text-3xl font-bold text-amber-600">{sentimentCounts.neutral}</p>
                        <p className="text-sm text-amber-700">
                          {Math.round(sentimentCounts.neutral / filtered.length * 100) || 0}% of total
                        </p>
                      </div>
                      
                      <div className="bg-rose-50 p-4 rounded-lg border border-rose-100">
                        <h4 className="text-rose-800 font-medium">Negative News</h4>
                        <p className="text-3xl font-bold text-rose-600">{sentimentCounts.negative}</p>
                        <p className="text-sm text-rose-700">
                          {Math.round(sentimentCounts.negative / filtered.length * 100) || 0}% of total
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}