import { motion } from 'framer-motion';
import { Loader2, Search, ChevronDown, ChevronUp, BarChart2, Cloud, Filter, Download, Sparkles } from 'lucide-react';
import { useState, useEffect, useMemo } from 'react';

interface Keyword {
  text: string;
  value: number;
}

interface Props {
  keywords: Keyword[];
}

function getContrastColor(bg: string): string {
  // Remove hash if present and parse R/G/B values
  const hex = bg.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);

  // YIQ formula to determine brightness
  const yiq = (r * 299 + g * 587 + b * 114) / 1000;

  return yiq >= 128 ? '#000000' : '#FFFFFF'; // black for light, white for dark
}

export default function EnhancedKeywords({ keywords }: Props) {
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [view, setView] = useState<'cloud' | 'bars'>('cloud');
  const [isLoading, setIsLoading] = useState(true);
  const [sortBy, setSortBy] = useState<'value' | 'alphabetical'>('value');
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  const baseLimit = 15;
  const maxValue = Math.max(...keywords.map(k => k.value));
  
  // Memoized filtered and sorted keywords
  const processedKeywords = useMemo(() => {
    const lowerSearch = searchTerm.toLowerCase();
    
    // Step 1: Filter by search term
    let result = keywords.filter(k => 
      k.text.toLowerCase().includes(lowerSearch)
    );
    
    // Step 2: Apply sentiment filter
    if (filter !== 'all') {
      const sorted = [...keywords].map(k => k.value).sort((a, b) => a - b);
      const p75 = sorted[Math.floor(sorted.length * 0.75)];
      const p50 = sorted[Math.floor(sorted.length * 0.5)];
      const p25 = sorted[Math.floor(sorted.length * 0.25)];
      
      result = result.filter(k => {
        if (filter === 'high') return k.value >= p75;
        if (filter === 'medium') return k.value >= p50 && k.value < p75;
        if (filter === 'low') return k.value < p50;
        return true;
      });
    }
    
    // Step 3: Sort
    return result.sort((a, b) => {
      if (sortBy === 'alphabetical') {
        return a.text.localeCompare(b.text);
      }
      return b.value - a.value; // Default to value-based sorting
    });
  }, [keywords, searchTerm, sortBy, filter]);
  
  const displayLimit = expanded ? processedKeywords.length : baseLimit;
  
  // Get sentiment color based on value and distribution
// Get sentiment color based on value and distribution
const getSentimentColor = (value: number) => {
  const normalized = value / maxValue;

  if (normalized >= 0.66) return '#10B981'; // Green (high)
  if (normalized >= 0.33) return '#FBBF24'; // Yellow (medium)
  return '#EF4444';                         // Red (low)
};

  
  // Download keywords as CSV
  const downloadCSV = () => {
    const csvContent = "data:text/csv;charset=utf-8," 
      + "Keyword,Value\n"
      + keywords.map(k => `"${k.text}",${k.value}`).join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "keyword_insights.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  useEffect(() => {
    // Simulate loading for demonstration
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 500);
    
    return () => clearTimeout(timer);
  }, [keywords]);

  if (isLoading) {
    return (
      <div className="flex-1 p-8 pt-6 flex items-center justify-center min-h-[50vh]">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="text-center space-y-4"
        >
          <div className="relative">
            <div className="absolute inset-0 rounded-full animate-ping bg-purple-400 opacity-20" />
            <div className="relative z-10 bg-white/80 backdrop-blur-lg rounded-2xl p-10 shadow-xl border border-purple-100">
              <Loader2 className="h-14 w-14 animate-spin text-purple-500 mx-auto" />
              <h3 className="text-xl font-medium text-gray-700 mt-4">Processing Keywords</h3>
              <p className="text-gray-500 mt-2">Analyzing frequency patterns...</p>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-indigo-50/50 via-purple-50/30 to-pink-50/30 min-h-screen">
      <div className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto py-8">
        <div className="space-y-8 pt-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="sticky top-0 z-10 bg-white/90 backdrop-blur-md border border-gray-200/50 shadow-lg rounded-2xl px-6 py-6"
          >
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                <div className="text-center sm:text-left">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-6 w-6 text-purple-500" />
                    <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                      Keyword Insights
                    </h2>
                  </div>
                  <p className="text-gray-500 mt-1">Discover patterns in your keyword data</p>
                </div>
                
                {/* Export Button */}
                <button
                  onClick={downloadCSV}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition duration-300 shadow-md hover:shadow-lg"
                >
                  <Download size={18} />
                  <span className="font-medium">Export Data</span>
                </button>
              </div>

              {/* Controls */}
              <div className="flex flex-wrap justify-center sm:justify-between gap-4">
                {/* Search */}
                <div className="relative flex-grow max-w-md">
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search for keywords..."
                    className="pl-10 pr-4 py-3 rounded-xl border border-gray-300 text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 w-full shadow-sm"
                  />
                  <Search size={18} className="absolute left-3.5 top-3.5 text-gray-400" />
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  {/* View Switch */}
                  <div className="flex items-center border border-gray-200 rounded-xl overflow-hidden shadow-sm">
                    <button
                      onClick={() => setView('bars')}
                      className={`flex items-center justify-center px-4 py-2.5 transition-all duration-300 text-sm font-medium ${
                        view === 'bars'
                          ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                          : 'bg-white text-gray-700 hover:bg-gray-50 hover:text-purple-600'
                      }`}
                      title="Bar View"
                    >
                      <BarChart2 size={18} className="mr-2" />
                      <span>Bars</span>
                    </button>
                    <button
                      onClick={() => setView('cloud')}
                      className={`flex items-center justify-center px-4 py-2.5 transition-all duration-300 text-sm font-medium ${
                        view === 'cloud'
                          ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
                          : 'bg-white text-gray-700 hover:bg-gray-50 hover:text-purple-600'
                      }`}
                      title="Cloud View"
                    >
                      <Cloud size={18} className="mr-2" />
                      <span>Cloud</span>
                    </button>
                  </div>

                  {/* Sort & Filter Controls Group */}
                  <div className="flex items-center gap-3">
                    {/* Sort */}
                    <div className="relative min-w-[160px]">
                      <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as 'value' | 'alphabetical')}
                        className="appearance-none w-full pl-10 pr-10 py-2.5 rounded-xl border border-gray-300 text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 shadow-sm bg-white"
                      >
                        <option value="value">Sort by Value</option>
                        <option value="alphabetical">Sort A-Z</option>
                      </select>
                      <div className="absolute left-3 top-2.5 text-gray-500">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                        </svg>
                      </div>
                      <ChevronDown size={16} className="absolute right-3 top-3 text-gray-400 pointer-events-none" />
                    </div>

                    {/* Filter */}
                    <div className="relative min-w-[180px]">
                      <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value as 'all' | 'high' | 'medium' | 'low')}
                        className="appearance-none w-full pl-10 pr-10 py-2.5 rounded-xl border border-gray-300 text-sm focus:ring-2 focus:ring-purple-500 focus:border-purple-500 shadow-sm bg-white"
                      >
                        <option value="all">All Relevance</option>
                        <option value="high">High Relevance</option>
                        <option value="medium">Medium Relevance</option>
                        <option value="low">Low Relevance</option>
                      </select>
                      <Filter size={16} className="absolute left-3.5 top-3 text-gray-500" />
                      <ChevronDown size={16} className="absolute right-3 top-3 text-gray-400 pointer-events-none" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
          
          {/* Main visualization area */}
          {view === 'bars' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-white border border-gray-200 rounded-2xl shadow-xl p-8 overflow-hidden"
            >
              <h3 className="text-2xl font-bold text-gray-800 mb-8 flex items-center">
                <BarChart2 className="h-6 w-6 mr-2 text-indigo-500" />
                Keyword Frequency Distribution
              </h3>
              <div className="space-y-4 max-h-[600px] overflow-y-auto pr-4 scrollbar-thin scrollbar-thumb-indigo-200 scrollbar-track-transparent">
                {processedKeywords.slice(0, displayLimit).map((keyword, index) => {
                  const percent = Math.round((keyword.value / maxValue) * 100);
                  const bg = getSentimentColor(keyword.value);
                  
                  return (
                    <motion.div
                      key={keyword.text}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.03 }}
                      className="group"
                    >
                      <div className="flex justify-between items-center mb-1.5 px-1">
                        <span className="font-semibold text-gray-800">{keyword.text}</span>
                        <span className="text-sm font-medium bg-gray-100 text-gray-700 px-2 py-1 rounded-md">{keyword.value}</span>
                      </div>
                      <div className="relative h-10 bg-gray-100 rounded-xl overflow-hidden group-hover:shadow-md transition-all duration-300">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percent}%` }}
                          transition={{ duration: 0.7, ease: "easeOut" }}
                          className="absolute inset-y-0 left-0 flex items-center justify-end px-4 rounded-xl"
                          style={{
                            background: `linear-gradient(90deg, ${bg}dd, ${bg})`,
                            color: getContrastColor(bg),
                          }}
                        >
                          {percent > 15 && (
                            <span className="font-semibold">{percent}%</span>
                          )}
                        </motion.div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}
          
          {view === 'cloud' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="bg-white border border-gray-200 rounded-2xl shadow-xl p-8 overflow-hidden"
            >
              <h3 className="text-2xl font-bold text-gray-800 mb-8 flex items-center">
                <Cloud className="h-6 w-6 mr-2 text-purple-500" />
                Keyword Cloud Visualization
              </h3>
              <div className="flex flex-wrap justify-center items-center gap-4 max-h-[600px] overflow-y-auto px-4 py-8 scrollbar-thin scrollbar-thumb-purple-200 scrollbar-track-transparent">
                {processedKeywords.slice(0, displayLimit).map((keyword, index) => {
                  const normalized = keyword.value / maxValue;
                  const bg = getSentimentColor(keyword.value);
                  const textColor = getContrastColor(bg);

                  return (
                    <motion.div
                      key={keyword.text}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{
                        duration: 0.4,
                        delay: index * 0.02,
                        type: 'spring',
                        stiffness: 200,
                      }}
                      whileHover={{ 
                        scale: 1.1,
                        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)'
                      }}
                      className="relative group"
                    >
                      <div
                        className="flex items-center justify-center font-bold transition-all duration-300 cursor-pointer shadow-md"
                        style={{
                          background: `linear-gradient(135deg, ${bg}dd, ${bg})`,
                          color: textColor,
                          fontSize: `${16 + normalized * 10}px`,
                          borderRadius: '1rem',
                          padding: '0.75rem 1.25rem',
                          minHeight: '2.5rem',
                          lineHeight: 1.25,
                        }}
                        title={`${keyword.text}: ${keyword.value}`}
                      >
                        {keyword.text}
                        <span 
                          className="ml-2 flex items-center justify-center text-xs opacity-90 bg-white/25 rounded-full h-6 px-2 backdrop-blur-sm" 
                          style={{ color: textColor }}
                        >
                          {keyword.value}
                        </span>
                      </div>
                      
                      {/* Tooltip on hover */}
                      <motion.div 
                        initial={{ opacity: 0 }}
                        whileHover={{ opacity: 1 }}
                        className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 whitespace-nowrap"
                      >
                        Frequency: {keyword.value} ({Math.round((keyword.value / maxValue) * 100)}%)
                      </motion.div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}
          
          {/* Analytics Insights Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6"
          >
            {/* Top Keywords */}
            <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow">
              <h4 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-indigo-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                </svg>
                Top Keywords
              </h4>
              <div className="space-y-3">
                {processedKeywords.slice(0, 5).map((keyword, idx) => (
                  <motion.div 
                    key={keyword.text} 
                    className="flex justify-between items-center p-3 rounded-xl hover:bg-indigo-50 transition-colors"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: idx * 0.1 }}
                  >
                    <div className="flex items-center">
                      <span className="w-7 h-7 flex items-center justify-center bg-gradient-to-br from-indigo-500 to-purple-600 text-white rounded-lg text-xs font-bold mr-3 shadow-sm">
                        {idx + 1}
                      </span>
                      <span className="font-medium text-gray-800">{keyword.text}</span>
                    </div>
                    <span className="text-sm font-semibold px-3 py-1 bg-indigo-100 text-indigo-700 rounded-lg">{keyword.value}</span>
                  </motion.div>
                ))}
              </div>
            </div>
            
            {/* Summary Stats */}
            <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-6 hover:shadow-xl transition-shadow">
              <h4 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Summary Statistics
              </h4>
              <div className="grid grid-cols-2 gap-4">
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-xl"
                >
                  <p className="text-indigo-600 text-sm font-medium">Total Keywords</p>
                  <p className="text-2xl font-bold text-indigo-900">{keywords.length}</p>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.1 }}
                  className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-xl"
                >
                  <p className="text-purple-600 text-sm font-medium">Highest Frequency</p>
                  <p className="text-2xl font-bold text-purple-900">{maxValue}</p>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.2 }}
                  className="bg-gradient-to-br from-pink-50 to-pink-100 p-4 rounded-xl"
                >
                  <p className="text-pink-600 text-sm font-medium">Average Frequency</p>
                  <p className="text-2xl font-bold text-pink-900">
                    {Math.round(keywords.reduce((sum, k) => sum + k.value, 0) / keywords.length)}
                  </p>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.3 }}
                  className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-xl"
                >
                  <p className="text-blue-600 text-sm font-medium">Filtered Keywords</p>
                  <p className="text-2xl font-bold text-blue-900">{processedKeywords.length}</p>
                </motion.div>
              </div>
            </div>
          </motion.div>
          
          {/* Expand/Collapse */}
          {processedKeywords.length > baseLimit && (
            <div className="text-center py-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setExpanded(!expanded)}
                className="inline-flex items-center gap-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium px-6 py-3 rounded-xl shadow-md hover:shadow-lg transition-all"
              >
                {expanded ? 'Show less' : `Show all ${processedKeywords.length} keywords`}
                {expanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </motion.button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}