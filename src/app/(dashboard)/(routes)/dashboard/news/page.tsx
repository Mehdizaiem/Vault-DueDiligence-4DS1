'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Line, Bar, Pie } from 'react-chartjs-2';
import EnhancedKeywords from '@/components/EnhancedKeywords';
import { Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
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
  date: string;               // e.g. "2025-05-09T14:32:10Z" or "2025-04-21 22:06:46.383532"
  content: string;
  sentiment_label: string;    // "Positive" | "Neutral" | "Negative"
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
  const lastNewsElementRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (loading) return;
      if (observer.current) observer.current.disconnect();
      observer.current = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting && filtered.length > displayCount) {
          setDisplayCount(prev => prev + LOAD_INCREMENT);
        }
      });
      if (node) observer.current.observe(node);
    },
    [loading, displayCount]
  );

  const toggleExpanded = (id: string) => {
    setExpanded(prev => {
      const s = new Set(prev);
      s.has(id) ? s.delete(id) : s.add(id);
      return s;
    });
  };

  // Parses a variety of timestamp formats into a valid Date
  function parseToDate(dateString: string): Date | null {
    // try native parse
    const d = new Date(dateString.replace(' ', 'T'));
    return isNaN(d.getTime()) ? null : d;
  }

  const fetchNews = async () => {
    setLoading(true);
    try {
      const until = new Date().toISOString();
      const sinceD = new Date();
      sinceD.setDate(sinceD.getDate() - 6);
      const since = sinceD.toISOString();
      const res = await fetch(`/api/news_sentiment?since=${since}&until=${until}`);
      const data = await res.json();
      if (Array.isArray(data.articles)) setNews(data.articles);
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  const getSentimentColor = (label?: string) => {
    const l = label?.toLowerCase();
    if (l === 'positive') return { bg: 'bg-green-100', text: 'text-green-800' };
    if (l === 'negative') return { bg: 'bg-rose-100', text: 'text-rose-800' };
    return { bg: 'bg-amber-100', text: 'text-amber-800' };
  };

  const formatDate = (d: string) =>
    new Date(d).toLocaleDateString('en-GB', {
      weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
    });

  const cryptocurrencies = ['Bitcoin','Ethereum','XRP','BNB','SOL','DOGE','ADA','LINK','AVAX'];
  const newsSources = Array.from(new Set(news.map(n => n.source))).sort();

  const filtered = news.filter(item => {
    const textMatch =
      item.title.toLowerCase().includes(search.toLowerCase()) ||
      item.content.toLowerCase().includes(search.toLowerCase());
    const sentimentMatch = filter === 'all' || item.sentiment_label.toLowerCase() === filter;
    const cryptoMatch = cryptoFilter === 'all' ||
      item.cryptocurrency?.toLowerCase() === cryptoFilter.toLowerCase() ||
      item.title.toLowerCase().includes(cryptoFilter.toLowerCase()) ||
      item.content.toLowerCase().includes(cryptoFilter.toLowerCase());
    const sourceMatch = sourceFilter === 'all' || item.source === sourceFilter;
    return textMatch && sentimentMatch && cryptoMatch && sourceMatch;
  });

  const totalFilteredCount = filtered.length || 1;
  const sentimentCounts = {
    positive: filtered.filter(i => i.sentiment_label.toLowerCase() === 'positive').length,
    neutral:  filtered.filter(i => i.sentiment_label.toLowerCase() === 'neutral').length,
    negative: filtered.filter(i => i.sentiment_label.toLowerCase() === 'negative').length,
  };

  const generateWordFrequencies = () => {
    const allText = filtered.map(i => `${i.title} ${i.content}`).join(' ').toLowerCase();
    const stopWords: Set<string> = new Set([/*...*/]);
    const words = allText.match(/\b[a-z]{4,}\b/g) || [];
    const freq = new Map<string, number>();
    words.forEach(w => { if (!stopWords.has(w)) freq.set(w, (freq.get(w)||0)+1); });
    return Array.from(freq.entries())
      .map(([text, value]) => ({ text, value }))
      .sort((a,b) => b.value - a.value).slice(0,50);
  };

  // 1) bucket counts by ISO date
  const sentimentByDate = filtered.reduce((acc, item) => {
    const d = parseToDate(item.date);
    if (!d) return acc;
    const day = d.toISOString().slice(0,10);
    if (!acc[day]) acc[day] = { positive:0, neutral:0, negative:0, total:0 };
    const lbl = item.sentiment_label.toLowerCase();
    if (lbl==='positive') acc[day].positive++;
    else if (lbl==='negative') acc[day].negative++;
    else acc[day].neutral++;
    acc[day].total++;
    return acc;
  }, {} as Record<string,{positive:number,neutral:number,negative:number,total:number}>);

  // 2) get the actual sorted dates and last 7
  const dateKeys = Object.keys(sentimentByDate).sort();
  const last7Dates = dateKeys.length>7 ? dateKeys.slice(-7) : dateKeys;

  // 3) chart data
  const pieChartData = {
    labels:['Positive','Neutral','Negative'],
    datasets:[{ data:[sentimentCounts.positive,sentimentCounts.neutral,sentimentCounts.negative], backgroundColor:['#238823','#FFBF00','#D2222D'] }]
  };

  const trendChartData = {
    labels:last7Dates,
    datasets:[
      { label:'Positive', data:last7Dates.map(d=>sentimentByDate[d]?.positive||0), borderColor:'#238823', backgroundColor:'rgba(35,136,35,0.3)', tension:0.2 },
      { label:'Neutral',  data:last7Dates.map(d=>sentimentByDate[d]?.neutral ||0), borderColor:'#FFBF00', backgroundColor:'rgba(255,191,0,0.3)', tension:0.2 },
      { label:'Negative', data:last7Dates.map(d=>sentimentByDate[d]?.negative||0), borderColor:'#D2222D', backgroundColor:'rgba(210,34,45,0.3)', tension:0.2 }
    ]
  };

  const barChartData = {
    labels:last7Dates,
    datasets:[
      { label:'Positive', data:last7Dates.map(d=>sentimentByDate[d]?.positive||0), backgroundColor:'#238823', stack:'sentiment' },
      { label:'Neutral',  data:last7Dates.map(d=>sentimentByDate[d]?.neutral||0), backgroundColor:'#FFBF00', stack:'sentiment' },
      { label:'Negative', data:last7Dates.map(d=>sentimentByDate[d]?.negative||0), backgroundColor:'#D2222D', stack:'sentiment' }
    ]
  };
  const barChartOptions = { responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{position:'top' as const}, tooltip:{callbacks:{label:(ctx:any)=>`${ctx.dataset.label}: ${ctx.raw}`}} },
    scales:{ x:{stacked:true}, y:{stacked:true, beginAtZero:true} }
  };

  const keywordFreqs = generateWordFrequencies();

  const keywordChartData = {
    labels: keywordFreqs.map(w => w.text).slice(0, 15),
    datasets: [{
      label: 'Frequency',
      data: keywordFreqs.map(w => w.value).slice(0, 15),
      backgroundColor: '#60A5FA', // Tailwind's blue-400
    }],
  };

  const keywordChartOptions = {
    indexAxis: 'y' as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: any) => `Frequency: ${ctx.raw}`,
        },
      },
    },
    scales: {
      x: { beginAtZero: true },
    },
  };

  // Loading state
  if (loading) {
    return (
      <main className="bg-gray-50 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="relative z-10 bg-white/50 backdrop-blur-lg rounded-2xl p-8 shadow-xl border border-blue-100">
            <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto" />
            <h3 className="text-xl font-medium text-muted-foreground mt-4">Loading dashboard</h3>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="bg-gray-50 min-h-[100vh]">

      <div className="max-w-7xl mx-auto p-6">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Crypto News Sentiment Dashboard
          </h1>
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
              className="relative inline-flex items-center justify-center bg-gradient-to-r from-blue-500 to-blue-600 text-white font-medium rounded-lg px-4 py-2 shadow transition-all duration-300 hover:from-blue-600 hover:to-blue-700 active:scale-95"
            >
              <span className={`inline-block mr-2 transition-transform duration-500 ${refreshing ? 'animate-spin' : ''}`}>↻</span>
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* 🔍 Search and Filters */}
        <div className="bg-white/50 backdrop-blur-lg p-6 rounded-xl shadow-md border border-gray-200 mb-6 flex flex-wrap items-center justify-between gap-4">
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

        {/* Tabs */}
        <div className="flex gap-4 mb-6">
        <button
            onClick={() => setActiveTab('news')}
            className={`text-sm font-medium rounded-full px-4 py-1.5 shadow transition-all
              ${activeTab === 'news' ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-lg' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
            `}
          >
            News Feed
          </button>

          <button
            onClick={() => setActiveTab('analytics')}
            className={`text-sm font-medium rounded-full px-4 py-1.5 shadow transition-all
              ${activeTab === 'analytics' ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-lg' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
            `}
          >
            Sentiment Analytics
          </button>
        </div>

        {/* Conditional Tabs Content */}
        {activeTab === 'news' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {filtered.length > 0 ? (
              filtered.slice(0, displayCount).map((item, idx) => {
                const { bg, text } = getSentimentColor(item.sentiment_label);
                const isLast = idx === displayCount - 1;
                return (
                  <motion.div
                    key={item.id || idx}
                    ref={isLast ? lastNewsElementRef : null}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                    className="relative bg-white/50 backdrop-blur-lg rounded-2xl border border-gray-200 shadow-lg hover:shadow-xl transform transition-transform duration-300 hover:scale-[1.02] overflow-hidden"
                  >
                    <div className="absolute -top-10 -right-10 h-24 w-24 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 opacity-10" />
                    <div className="p-6">
                      <div className="flex justify-between items-start mb-2">
                        <h2 className="font-bold text-lg sm:text-xl text-gray-800 group-hover:text-blue-600 transition-colors duration-300">
                          {item.title}
                        </h2>
                        <span className="text-xs font-medium bg-gray-100 text-gray-700 px-2 py-1 rounded-full shadow">
                          {item.source}
                        </span>
                      </div>
                      <p className="text-gray-600 text-sm mb-3 leading-snug line-clamp-3">
                        {item.content}
                      </p>
                      <div className="flex flex-wrap gap-2 mb-3 text-sm">
                        {item.sentiment_label && (
                          <span className={`px-3 py-1 text-xs font-medium rounded-full shadow-sm ${item.sentiment_label.toLowerCase() === 'positive' ? 'bg-green-500/10 text-green-700' : item.sentiment_label.toLowerCase() === 'negative' ? 'bg-red-500/10 text-red-700' : 'bg-yellow-500/10 text-yellow-700'}`}>
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
                          className="text-blue-600 text-sm font-medium hover:underline"
                        >
                          Read more →
                        </a>
                      )}
                    </div>
                  </motion.div>
                );
              })
            ) : (
              <div className="col-span-full text-center py-12">
                <div className="text-5xl mb-4">📰</div>
                <h3 className="text-xl font-semibold text-gray-600 mb-2">No news found</h3>
                <p className="text-gray-500">Try changing your search or filter criteria</p>
              </div>
            )}
          </div>
        ) : activeTab === 'analytics' ? (
          <div className="bg-white/50 backdrop-blur-xl border border-gray-200 rounded-2xl shadow-xl p-6 space-y-10">
                       <motion.h2
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
            >
              Sentiment Analysis Overview
            </motion.h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 auto-rows-min">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1 }}
                className="bg-white border rounded-xl shadow-md p-6 hover:shadow-lg transition"
              >
                <h3 className="text-lg font-semibold text-center text-gray-700">
                  Sentiment Bar Chart (By Date)
                </h3>
                <div className="h-64 mt-4">
                  <Bar data={barChartData} options={barChartOptions} />
                </div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2 }}
                className="bg-white border rounded-xl shadow-md p-6 hover:shadow-lg transition"
              >
                <h3 className="text-lg font-semibold text-center text-gray-700">
                  Sentiment Distribution
                </h3>
                <div className="h-64 mt-4">
                  <Pie data={pieChartData} options={{ maintainAspectRatio: false }} />
                </div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 }}
                className="lg:col-span-2 bg-white border rounded-xl shadow-md p-6 hover:shadow-lg transition"
              >
                <h3 className="text-lg font-semibold text-center text-gray-700">
                  Sentiment Trends Over Time
                </h3>
                <div className="h-80 mt-4">
                  <Line data={trendChartData} options={{ maintainAspectRatio: false }} />
                </div>
              </motion.div>
            </div>

            <motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.4, delay: 0.4 }}
  className="bg-gradient-to-r from-slate-50 to-white border border-gray-200 rounded-2xl shadow-md p-6"
>


  <EnhancedKeywords keywords={keywordFreqs.slice(0, 30)} />
</motion.div>


            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.5 }}
                className="bg-green-50 border border-green-200 p-4 rounded-xl shadow-sm text-center"
              >
                <h4 className="text-green-700 font-semibold text-sm uppercase tracking-wide">Positive News</h4>
                <p className="text-3xl font-bold text-green-600 mt-1">{sentimentCounts.positive}</p>
                <p className="text-xs text-green-600 mt-1">
                  {Math.round((sentimentCounts.positive / totalFilteredCount) * 100)}% of total
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.6 }}
                className="bg-yellow-50 border border-yellow-200 p-4 rounded-xl shadow-sm text-center"
              >
                <h4 className="text-yellow-700 font-semibold text-sm uppercase tracking-wide">Neutral News</h4>
                <p className="text-3xl font-bold text-yellow-600 mt-1">{sentimentCounts.neutral}</p>
                <p className="text-xs text-yellow-600 mt-1">
                  {Math.round((sentimentCounts.neutral / totalFilteredCount) * 100)}% of total
                </p>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.7 }}
                className="bg-rose-50 border border-rose-200 p-4 rounded-xl shadow-sm text-center"
              >
                <h4 className="text-rose-700 font-semibold text-sm uppercase tracking-wide">Negative News</h4>
                <p className="text-3xl font-bold text-rose-600 mt-1">{sentimentCounts.negative}</p>
                <p className="text-xs text-rose-600 mt-1">
                  {Math.round((sentimentCounts.negative / totalFilteredCount) * 100)}% of total
                </p>
              </motion.div>
            </div>
          </div>
        ): null}
      </div>
      {/* Toast notification */}
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
    </main>
  );
}
