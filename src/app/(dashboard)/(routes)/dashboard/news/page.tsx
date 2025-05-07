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
  ArcElement,
} from 'chart.js';

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

const LOAD_INCREMENT = 6;

export default function NewsDashboard() {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [displayCount, setDisplayCount] = useState(LOAD_INCREMENT);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState<'news' | 'analytics'>('news');
  const [cryptoFilter, setCryptoFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');

  const observer = useRef<IntersectionObserver | null>(null);
  const lastNewsElementRef = useCallback((node: HTMLDivElement | null) => {
    if (loading) return;
    if (observer.current) observer.current.disconnect();
    observer.current = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && filtered.length > displayCount) {
        setDisplayCount((prev) => prev + LOAD_INCREMENT);
      }
    });
    if (node) observer.current.observe(node);
  }, [loading, displayCount]);

  const toggleExpanded = (id: string) => {
    setExpanded((prev) => {
      const set = new Set(prev);
      set.has(id) ? set.delete(id) : set.add(id);
      return set;
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
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (label?: string) => {
    const l = label?.toLowerCase();
    if (l === 'positive') return { bg: 'bg-green-100', text: 'text-green-800' };
    if (l === 'negative') return { bg: 'bg-rose-100', text: 'text-rose-800' };
    return { bg: 'bg-amber-100', text: 'text-amber-800' };
  };

  const formatDate = (d: string) =>
    new Date(d).toLocaleDateString('en-GB', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });

  const cryptocurrencies = ['Bitcoin', 'Ethereum', 'XRP', 'BNB', 'SOL', 'DOGE', 'ADA', 'LINK', 'AVAX'];
  const newsSources = Array.from(new Set(news.map((n) => n.source))).sort();

  const filtered = news.filter((item) => {
    const text =
      item.title?.toLowerCase().includes(search.toLowerCase()) ||
      item.content?.toLowerCase().includes(search.toLowerCase());
    const sentiment = filter === 'all' || item.sentiment_label?.toLowerCase() === filter;
    const crypto =
      cryptoFilter === 'all' ||
      item.cryptocurrency?.toLowerCase() === cryptoFilter.toLowerCase() ||
      item.title?.toLowerCase().includes(cryptoFilter.toLowerCase()) ||
      item.content?.toLowerCase().includes(cryptoFilter.toLowerCase());
    const source = sourceFilter === 'all' || item.source === sourceFilter;
    return text && sentiment && crypto && source;
  });

  const sentimentCounts = {
    positive: filtered.filter(i => i.sentiment_label?.toLowerCase() === 'positive').length,
    neutral: filtered.filter(i => i.sentiment_label?.toLowerCase() === 'neutral').length,
    negative: filtered.filter(i => i.sentiment_label?.toLowerCase() === 'negative').length,
  };

  const sentimentByDate = filtered.reduce((acc, item) => {
    const date = item.date.split('T')[0];
    if (!acc[date]) acc[date] = { positive: 0, neutral: 0, negative: 0, total: 0 };
    const label = item.sentiment_label?.toLowerCase();
    if (label === 'positive') acc[date].positive++;
    else if (label === 'negative') acc[date].negative++;
    else acc[date].neutral++;
    acc[date].total++;
    return acc;
  }, {} as Record<string, { positive: number, neutral: number, negative: number, total: number }>);

  const sortedDates = Object.keys(sentimentByDate).sort();

  const pieChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [{
      data: [sentimentCounts.positive, sentimentCounts.neutral, sentimentCounts.negative],
      backgroundColor: ['#34D399', '#FBBF24', '#EF4444'],
    }],
  };

  const trendChartData = {
    labels: sortedDates,
    datasets: [
      {
        label: 'Positive',
        data: sortedDates.map(date => sentimentByDate[date].positive),
        borderColor: '#10B981',
        backgroundColor: 'rgba(16, 185, 129, 0.5)',
        tension: 0.2,
      },
      {
        label: 'Neutral',
        data: sortedDates.map(date => sentimentByDate[date].neutral),
        borderColor: '#F59E0B',
        backgroundColor: 'rgba(245, 158, 11, 0.5)',
        tension: 0.2,
      },
      {
        label: 'Negative',
        data: sortedDates.map(date => sentimentByDate[date].negative),
        borderColor: '#DC2626',
        backgroundColor: 'rgba(239, 68, 68, 0.5)',
        tension: 0.2,
      },
    ],
  };
  return (
    <main className="bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto p-6">
        {<div className="flex justify-between items-center mb-6">
          <h1
    className="text-2xl sm:text-3xl font-extrabold text-gray-800 flex items-center gap-2
               transition duration-300 ease-in-out hover:text-blue-600 transform hover:scale-[1.02]"
  >
     Crypto News Sentiment Dashboard
  </h1>

  {/* Refresh Button with animated icon and toast trigger */}
  <div className="flex items-center gap-3">
    <span className="text-sm text-gray-500 italic">
      Last updated: {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
    </span>
    <button
      onClick={async () => {
        setRefreshing(true);
        await fetchNews();
        setRefreshing(false);

        const toast = document.getElementById('refresh-toast');
        if (toast) {
          toast.classList.remove('translate-y-16', 'opacity-0');
          toast.classList.add('translate-y-0', 'opacity-100');
          setTimeout(() => {
            toast.classList.add('translate-y-16', 'opacity-0');
            toast.classList.remove('translate-y-0', 'opacity-100');
          }, 3000);
        }
      }}
      disabled={refreshing}
      className={`
        relative inline-flex items-center justify-center
        bg-gradient-to-r from-blue-500 to-blue-600
        text-white font-medium rounded-lg px-4 py-2
        shadow transition-all duration-300
        hover:from-blue-600 hover:to-blue-700
        active:scale-95
        ${refreshing ? 'opacity-60 cursor-not-allowed' : ''}
      `}
    >
      <span
        className={`inline-block mr-2 transition-transform duration-500 ${
          refreshing ? 'animate-spin' : ''
        }`}
      >
        ↻
      </span>
      {refreshing ? 'Refreshing...' : 'Refresh'}
    </button>
  </div>
</div>
}{/* 🔍 Search and Filters */}
<div className="bg-white p-4 rounded-lg shadow-sm mb-6 flex flex-wrap items-center justify-between gap-4">

  {/* 🔍 Search Bar */}
  <input
    type="text"
    placeholder="🔍 Search title or content..."
    value={search}
    onChange={(e) => {
      setSearch(e.target.value);
      setDisplayCount(LOAD_INCREMENT);
    }}
    className="flex-1 min-w-[200px] p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 text-sm"
  />

  {/* Sentiment Filter */}
  <div>
    <label className="text-sm text-gray-600 mr-2">Sentiment:</label>
    <select
      value={filter}
      onChange={(e) => {
        setFilter(e.target.value);
        setDisplayCount(LOAD_INCREMENT);
      }}
      className="text-sm px-3 py-1.5 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
    >
      <option value="all">All</option>
      <option value="positive">Positive</option>
      <option value="neutral">Neutral</option>
      <option value="negative">Negative</option>
    </select>
  </div>

  {/* Crypto Filter */}
  <div>
    <label className="text-sm text-gray-600 mr-2">Crypto:</label>
    <select
      value={cryptoFilter}
      onChange={(e) => {
        setCryptoFilter(e.target.value);
        setDisplayCount(LOAD_INCREMENT);
      }}
      className="text-sm px-3 py-1.5 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
    >
      <option value="all">All</option>
      {cryptocurrencies.map((c) => (
        <option key={c} value={c}>{c}</option>
      ))}
    </select>
  </div>

  {/* Source Filter */}
  <div>
    <label className="text-sm text-gray-600 mr-2">Source:</label>
    <select
      value={sourceFilter}
      onChange={(e) => {
        setSourceFilter(e.target.value);
        setDisplayCount(LOAD_INCREMENT);
      }}
      className="text-sm px-3 py-1.5 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
    >
      <option value="all">All</option>
      {newsSources.map((s) => (
        <option key={s} value={s}>{s}</option>
      ))}
    </select>
  </div>
</div>

        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setActiveTab('news')}
            className={`text-sm px-4 py-2 rounded-lg font-semibold ${
              activeTab === 'news'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            News Feed
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            className={`text-sm px-4 py-2 rounded-lg font-semibold ${
              activeTab === 'analytics'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Sentiment Analytics
          </button>
        </div>
        <div 
        id="refresh-toast"
        className="fixed bottom-6 right-6 bg-green-100 text-green-800 px-4 py-2 rounded shadow-lg
                   transform transition-all duration-500 translate-y-16 opacity-0 z-50 flex items-center gap-2"
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 
              7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
        <span>Data refreshed successfully!</span>
      </div>
        {/* Conditional Tabs */}
        {activeTab === 'news' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {filtered.slice(0, displayCount).map((item, idx) => {
              const { bg, text } = getSentimentColor(item.sentiment_label);
              const isLast = idx === displayCount - 1;
              return (
                <div
                  key={item.id || idx}
                  ref={isLast ? lastNewsElementRef : null}
                  className="group bg-white rounded-xl shadow-md border border-gray-100 hover:shadow-xl transition duration-300 transform hover:-translate-y-1"
                >
                  <div className="px-6 py-4">
                    <div className="flex justify-between items-start mb-2">
                      <h2 className="font-bold text-lg sm:text-xl text-gray-800 group-hover:text-blue-600 transition-colors duration-300">
                        {item.title}
                      </h2>
                      <span className="text-xs font-medium bg-gray-200 text-gray-700 px-2 py-1 rounded-full shadow-sm">
                        {item.source}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm mb-3 leading-snug line-clamp-3">
                      {item.content}
                    </p>
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
                    {item.explanation && (
                      <button
                        onClick={() => toggleExpanded(item.id || idx.toString())}
                        className="text-xs font-medium px-3 py-1 rounded-full bg-blue-100 text-blue-700 hover:bg-blue-600 hover:text-white transition duration-300 shadow hover:shadow-md"
                      >
                        {expanded.has(item.id || idx.toString()) ? 'Hide Explanation' : 'Show Explanation'}
                      </button>
                    )}
                    {expanded.has(item.id || idx.toString()) && (
                      <p className="text-sm text-gray-600 mt-2 italic leading-relaxed border-l-4 border-blue-200 pl-3">
                        {item.explanation}
                      </p>
                    )}
                  </div>
                  <div className="bg-gray-50 px-6 py-3 flex justify-between items-center border-t">
                    <span className="text-xs text-gray-400 italic">{formatDate(item.date)}</span>
                    {item.url && (
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 text-sm font-medium relative group inline-block transition-all duration-300 after:absolute after:bottom-0 after:left-0 after:w-full after:h-0.5 after:bg-blue-600 after:scale-x-0 after:origin-bottom-right group-hover:after:scale-x-100 group-hover:after:origin-bottom-left"
                      >
                        Read more →
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow p-6">
           <div className="bg-white rounded-2xl shadow-md p-6">
  <h2 className="text-3xl font-extrabold text-gray-800 mb-8 text-center tracking-tight">
     Sentiment Analysis Overview
  </h2>

  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
    {/* Pie Chart */}
    <div className="bg-white border rounded-xl shadow-sm p-6 transition hover:shadow-lg">
      <h3 className="text-lg font-semibold text-gray-700 mb-4 text-center">Sentiment Distribution</h3>
      <div className="h-64">
        <Pie data={pieChartData} options={{ maintainAspectRatio: false }} />
      </div>
    </div>

    {/* Line Chart */}
    <div className="bg-white border rounded-xl shadow-sm p-6 transition hover:shadow-lg">
      <h3 className="text-lg font-semibold text-gray-700 mb-4 text-center">Sentiment Trends Over Time</h3>
      <div className="h-64">
        <Line data={trendChartData} options={{ maintainAspectRatio: false }} />
      </div>
    </div>
  </div>

  {/* Summary Stats */}
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
    <div className="bg-green-50 border border-green-200 p-4 rounded-xl shadow-sm text-center">
      <h4 className="text-green-700 font-semibold text-sm uppercase tracking-wide">Positive News</h4>
      <p className="text-3xl font-bold text-green-600 mt-1">{sentimentCounts.positive}</p>
      <p className="text-xs text-green-600 mt-1">{Math.round((sentimentCounts.positive / filtered.length) * 100)}% of total</p>
    </div>
    <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-xl shadow-sm text-center">
      <h4 className="text-yellow-700 font-semibold text-sm uppercase tracking-wide">Neutral News</h4>
      <p className="text-3xl font-bold text-yellow-600 mt-1">{sentimentCounts.neutral}</p>
      <p className="text-xs text-yellow-600 mt-1">{Math.round((sentimentCounts.neutral / filtered.length) * 100)}% of total</p>
    </div>
    <div className="bg-rose-50 border border-rose-200 p-4 rounded-xl shadow-sm text-center">
      <h4 className="text-rose-700 font-semibold text-sm uppercase tracking-wide">Negative News</h4>
      <p className="text-3xl font-bold text-rose-600 mt-1">{sentimentCounts.negative}</p>
      <p className="text-xs text-rose-600 mt-1">{Math.round((sentimentCounts.negative / filtered.length) * 100)}% of total</p>
    </div>
  </div>
</div>

          </div>
        )}
      </div>
    </main>
  );
}
