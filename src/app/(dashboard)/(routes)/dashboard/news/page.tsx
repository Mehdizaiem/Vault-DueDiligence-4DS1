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
  cryptocurrency?: string;
};

// Number of items to load per infinite scroll trigger
const LOAD_INCREMENT = 6;

export default function NewsDashboard() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [displayCount, setDisplayCount] = useState(LOAD_INCREMENT);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('news');
  const [cryptoFilter, setCryptoFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);
  const [refreshAnimation, setRefreshAnimation] = useState(false);
  
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
    setLoading(true);
    setRefreshAnimation(true);
    try {
      await fetch('/api/refresh_news');
      await fetchNews();
    } catch (err) {
      console.error('Refresh failed:', err);
    } finally {
      setLoading(false);
      setTimeout(() => setRefreshAnimation(false), 1000);
    }
  };

  const formatDate = (d: string) => {
    const date = new Date(d);
    return isNaN(date.getTime()) ? 'N/A' : date.toLocaleDateString();
  };

  const getSentimentColor = (s?: string) => {
    if (s?.toLowerCase().includes('positive')) return { bg: 'bg-indigo-100', text: 'text-indigo-800', border: 'border-indigo-200', dark: 'bg-indigo-200' };
    if (s?.toLowerCase().includes('negative')) return { bg: 'bg-pink-100', text: 'text-pink-800', border: 'border-pink-200', dark: 'bg-pink-200' };
    return { bg: 'bg-teal-100', text: 'text-teal-800', border: 'border-teal-200', dark: 'bg-teal-200' };
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

  // Chart data with updated colors
  const pieChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [sentimentCounts.positive, sentimentCounts.neutral, sentimentCounts.negative],
        backgroundColor: ['rgba(99, 102, 241, 0.7)', 'rgba(20, 184, 166, 0.7)', 'rgba(236, 72, 153, 0.7)'],
        borderColor: ['rgb(79, 70, 229)', 'rgb(13, 148, 136)', 'rgb(219, 39, 119)'],
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
        borderColor: 'rgb(79, 70, 229)',
        backgroundColor: 'rgba(99, 102, 241, 0.3)',
        tension: 0.3,
        fill: true,
      },
      {
        label: 'Neutral',
        data: sortedDates.map(date => sentimentByDate[date].neutral),
        borderColor: 'rgb(13, 148, 136)',
        backgroundColor: 'rgba(20, 184, 166, 0.3)',
        tension: 0.3,
        fill: true,
      },
      {
        label: 'Negative',
        data: sortedDates.map(date => sentimentByDate[date].negative),
        borderColor: 'rgb(219, 39, 119)',
        backgroundColor: 'rgba(236, 72, 153, 0.3)',
        tension: 0.3,
        fill: true,
      },
    ],
  };

  return (
    <main className="bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <header className="mb-6">
          <div className="flex justify-center items-center gap-4 mb-5">
            <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-600 to-indigo-600 flex items-center gap-2">
              📊 Crypto News Sentiment Dashboard
            </h1>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className={`bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-3 py-1.5 rounded-lg shadow-md hover:shadow-lg hover:from-indigo-600 hover:to-purple-600 transition-all duration-300 disabled:opacity-50 ${refreshAnimation ? 'animate-pulse' : ''}`}
            >
              {loading ? (
                <span className="inline-block animate-spin">🔄</span>
              ) : (
                <span>🔄</span>
              )}
            </button>
          </div>

          <div className="border-b border-gray-200 mb-6">
            <ul className="flex flex-wrap -mb-px">
              <li className="mr-2">
                <button 
                  onClick={() => setActiveTab('news')}
                  className={`inline-block py-2 px-4 text-sm font-medium rounded-t-lg transition-all duration-300 ${
                    activeTab === 'news'
                    ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50 shadow-inner'
                    : 'text-gray-500 hover:text-indigo-500 hover:bg-gray-100'
                  }`}
                >
                  News Feed
                </button>
              </li>
              <li className="mr-2">
                <button 
                  onClick={() => setActiveTab('analytics')}
                  className={`inline-block py-2 px-4 text-sm font-medium rounded-t-lg transition-all duration-300 ${
                    activeTab === 'analytics'
                    ? 'text-indigo-600 border-b-2 border-indigo-600 bg-indigo-50 shadow-inner'
                    : 'text-gray-500 hover:text-indigo-500 hover:bg-gray-100'
                  }`}
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
              className="flex-1 p-2 border border-gray-200 rounded-lg shadow-md focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 transition-all duration-300"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setDisplayCount(LOAD_INCREMENT);
              }}
            />
            
            <div className="flex flex-wrap gap-2">
              {/* Export buttons */}
              <div className="bg-white p-2 rounded-lg shadow-md flex gap-2">
                <button 
                  onClick={exportCSV}
                  className="px-3 py-1.5 text-sm font-medium bg-gradient-to-r from-teal-400 to-teal-500 text-white rounded-md shadow-sm hover:shadow-md transition-all duration-300 flex items-center gap-1 transform hover:scale-105"
                >
                  <span>📄</span> Export CSV
                </button>
                <button 
                  onClick={exportJSON}
                  className="px-3 py-1.5 text-sm font-medium bg-gradient-to-r from-purple-400 to-purple-500 text-white rounded-md shadow-sm hover:shadow-md transition-all duration-300 flex items-center gap-1 transform hover:scale-105"
                >
                  <span>📋</span> Export JSON
                </button>
              </div>
            </div>
          </div>

          {/* Additional Filters */}
          <div className="bg-white p-4 rounded-lg shadow-md mb-6 flex flex-wrap gap-4 backdrop-blur-sm bg-opacity-80">
            {/* Sentiment Filter */}
            <div>
              <span className="text-sm text-gray-600 mr-2">Sentiment:</span>
              <div className="inline-flex rounded-md shadow-sm">
                {['all', 'positive', 'neutral', 'negative'].map(type => {
                  // Select appropriate colors based on sentiment type
                  const getBgColor = () => {
                    if (filter === type) {
                      if (type === 'positive') return 'bg-gradient-to-r from-indigo-100 to-indigo-200 hover:from-indigo-200 hover:to-indigo-300';
                      if (type === 'neutral') return 'bg-gradient-to-r from-teal-100 to-teal-200 hover:from-teal-200 hover:to-teal-300';
                      if (type === 'negative') return 'bg-gradient-to-r from-pink-100 to-pink-200 hover:from-pink-200 hover:to-pink-300';
                      return 'bg-gradient-to-r from-blue-100 to-blue-200 hover:from-blue-200 hover:to-blue-300';
                    }
                    return 'bg-white hover:bg-gray-100';
                  };
                  
                  const getTextColor = () => {
                    if (filter === type) {
                      if (type === 'positive') return 'text-indigo-800';
                      if (type === 'neutral') return 'text-teal-800';
                      if (type === 'negative') return 'text-pink-800';
                      return 'text-blue-800';
                    }
                    return 'text-gray-700';
                  };
                  
                  // Apply special styling for first and last buttons in the group
                  const position = 
                    type === 'all' ? 'rounded-l-md' :
                    type === 'negative' ? 'rounded-r-md' : '';
                    
                  return (
                    <button
                      key={type}
                      onClick={() => {
                        setFilter(type);
                        setDisplayCount(LOAD_INCREMENT);
                      }}
                      className={`px-4 py-1.5 text-sm font-medium ${getBgColor()} ${getTextColor()} ${position} border border-gray-200 transition-all duration-300 shadow-sm transform hover:scale-105`}
                    >
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </button>
                  );
                })}
              </div>
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
                className="px-3 py-1.5 text-sm font-medium bg-white border rounded-md shadow-sm hover:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-300 transition-all duration-300"
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
                className="px-3 py-1.5 text-sm font-medium bg-white border rounded-md shadow-sm hover:border-indigo-300 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-300 transition-all duration-300"
              >
                <option value="all">All Sources</option>
                {newsSources.map(source => (
                  <option key={source} value={source}>{source}</option>
                ))}
              </select>
            </div>
          </div>
        </header>

        {/* Loader */}
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-indigo-200 border-t-4 border-t-indigo-600"></div>
          </div>
        ) : (
          <>
            {activeTab === 'news' ? (
              /* Articles Grid */
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {displayItems.map((item, idx) => {
                    const sentimentColor = getSentimentColor(item.sentiment_label);
                    const isLastElement = idx === displayItems.length - 1;
                    const cardId = item.id || idx.toString();
                    
                    return (
                      <div
                        key={cardId}
                        ref={isLastElement ? lastNewsElementRef : null}
                        className={`bg-white rounded-xl shadow-md border border-gray-100 hover:shadow-xl transition-all duration-300 transform ${hoveredCard === cardId ? 'scale-102' : ''}`}
                        onMouseEnter={() => setHoveredCard(cardId)} 
                        onMouseLeave={() => setHoveredCard(null)}
                      >
                        <div className="px-6 py-4">
                          <div className="flex justify-between items-start mb-2">
                            <h2 className="font-bold text-xl text-gray-800 group-hover:text-indigo-600 transition-colors duration-300">{item.title}</h2>
                            <span className="text-xs bg-gray-100 px-2 py-1 rounded-full text-gray-700 shadow-sm">{item.source}</span>
                          </div>
                          <p className="text-gray-600 mb-3 line-clamp-3">{item.content}</p>

                          <div className="flex flex-wrap gap-2 mb-3 text-sm">
                            {item.sentiment_label && (
                              <span className={`${sentimentColor.bg} ${sentimentColor.text} px-2 py-1 rounded-full shadow-sm`}>
                                {item.sentiment_label}
                              </span>
                            )}

                            {item.sentiment_score != null && !isNaN(item.sentiment_score) && (
                              <span className={`${sentimentColor.dark} ${sentimentColor.text} px-2 py-1 rounded-full shadow-sm`}>
                                Score: {item.sentiment_score.toFixed(2)}
                              </span>
                            )}

                            {item.aspect && (
                              <span className="bg-violet-100 text-violet-800 px-2 py-1 rounded-full shadow-sm">
                                {item.aspect}
                              </span>
                            )}
                            {item.cryptocurrency && (
                              <span className="bg-cyan-100 text-cyan-800 px-2 py-1 rounded-full shadow-sm">
                                {item.cryptocurrency}
                              </span>
                            )}
                          </div>

                          {/* Explanation toggle */}
                          {item.explanation && (
                            <button
                              onClick={() => toggleExpanded(cardId)}
                              className={`text-sm hover:underline flex items-center gap-1 transition-all duration-300 ${
                                expanded.has(cardId) 
                                ? "text-indigo-700" 
                                : "text-indigo-600"
                              }`}
                            >
                              {expanded.has(cardId) ? (
                                <>
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                  </svg>
                                  Hide Explanation
                                </>
                              ) : (
                                <>
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                  Show Explanation
                                </>
                              )}
                            </button>
                          )}
                          {expanded.has(cardId) && (
                            <div className="animate-fadeIn">
                              <p className="text-sm text-gray-700 mt-2 italic bg-gray-50 p-3 rounded-lg border-l-2 border-indigo-300 shadow-sm">{item.explanation}</p>
                            </div>
                          )}
                        </div>

                        <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-3 flex justify-between items-center border-t rounded-b-xl">
                          <span className="text-xs text-gray-500">{formatDate(item.date)}</span>
                          {item.url && (
                            <a
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-indigo-600 text-sm hover:text-indigo-800 hover:underline flex items-center gap-1 transition-colors duration-300 group"
                            >
                              Read more
                              <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                              </svg>
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
                    <div className="animate-pulse text-indigo-500 font-medium">Loading more articles...</div>
                  </div>
                )}
                
                {/* No results message */}
                {filtered.length === 0 && (
                  <div className="text-center py-10 bg-white rounded-xl shadow-md">
                    <p className="text-gray-500 text-xl">No news items match your current filters</p>
                    <button 
                      onClick={() => {
                        setFilter('all');
                        setCryptoFilter('all');
                        setSourceFilter('all');
                        setSearch('');
                      }}
                      className="mt-4 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors duration-300"
                    >
                      Clear all filters
                    </button>
                  </div>
                )}
              </>
            ) : (
              /* Analytics View */
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-6">Sentiment Analysis</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Pie Chart */}
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md border border-gray-100">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Sentiment Distribution</h3>
                    <div className="h-64">
                      <Pie 
                        data={pieChartData} 
                        options={{
                          plugins: {
                            legend: {
                              position: 'right',
                              labels: {
                                font: {
                                  weight: 'bold'
                                },
                                padding: 20
                              }
                            },
                            tooltip: {
                              callbacks: {
                                label: function(tooltipItem) {
                                  const value = tooltipItem.raw as number;
                                  const total = sentimentCounts.positive + sentimentCounts.neutral + sentimentCounts.negative;
                                  const percentage = Math.round((value / total) * 100);
                                  return `${tooltipItem.label}: ${value} (${percentage}%)`;
                                }
                              },
                              backgroundColor: 'rgba(255, 255, 255, 0.9)',
                              titleColor: '#1f2937',
                              bodyColor: '#4f46e5',
                              borderColor: '#e5e7eb',
                              borderWidth: 1,
                              padding: 12,
                              displayColors: true,
                              boxPadding: 6
                            }
                          },
                          maintainAspectRatio: false,
                          animation: {
                            animateScale: true,
                            animateRotate: true,
                            duration: 2000
                          }
                        }}
                      />
                    </div>
                  </div>
                  
                  {/* Trend Chart */}
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg shadow-md border border-gray-100">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Sentiment Trends Over Time</h3>
                    <div className="h-64">
                      <Line 
                        data={trendChartData}
                        options={{
                          plugins: {
                            legend: {
                              position: 'top',
                              labels: {
                                font: {
                                  weight: 'bold'
                                },
                                padding: 20
                              }
                            },
                            tooltip: {
                              backgroundColor: 'rgba(255, 255, 255, 0.9)',
                              titleColor: '#1f2937',
                              bodyColor: '#4f46e5',
                              borderColor: '#e5e7eb',
                              borderWidth: 1
                            }
                          },
                          scales: {
                            y: {
                              beginAtZero: true,
                              grid: {
                                color: 'rgba(156, 163, 175, 0.1)'
                              }
                            },
                            x: {
                              grid: {
                                display: false
                              }
                            }
                          },
                          maintainAspectRatio: false,
                          animation: {
                            duration: 2000
                          },
                          elements: {
                            point: {
                              radius: 3,
                              hoverRadius: 6
                            }
                          }
                        }}
                      />
                    </div>
                  </div>
                  
                  {/* Summary Statistics */}
                  <div className="lg:col-span-2 mt-4">
                    <h3 className="text-lg font-medium text-gray-700 mb-4">Summary Statistics</h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-lg border border-indigo-200 shadow-md hover:shadow-lg transition-all duration-300 transform hover:scale-105">
                        <h4 className="font-medium text-indigo-800">Positive News</h4>
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

        <footer className="mt-10 text-center text-sm text-gray-500">
          <p>Last updated: {new Date().toLocaleDateString()}</p>
        </footer>
      </div>
    </main>
  );
}