import { motion } from 'framer-motion';
import { Loader2, Search, ChevronDown, ChevronUp, BarChart2, Cloud, List, Filter, Download } from 'lucide-react';
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
  const getSentimentColor = (value: number) => {
    const normalized = value / maxValue;
  
    if (normalized >= 0.66) return '#238823'; // Green - High freq
    if (normalized >= 0.33) return '#FFBF00'; // Yellow - Medium freq
    return '#D2222D';                         // Red - Low freq
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
            <div className="absolute inset-0 rounded-full animate-ping bg-blue-400 opacity-20" />
            <div className="relative z-10 bg-white/50 backdrop-blur-lg rounded-2xl p-8 shadow-xl border border-blue-100">
              <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto" />
              <h3 className="text-xl font-medium text-muted-foreground mt-4">Processing Keywords</h3>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-50 via-blue-50/20 to-purple-50/20">
      <div className="px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto py-8">
        <div className="space-y-6 pt-6">
        <motion.div
  initial={{ opacity: 0, y: -20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.5 }}
  className="sticky top-0 z-10 bg-white/90 backdrop-blur-md border-b border-slate-200 px-4 py-6"
>
  <div className="max-w-5xl mx-auto space-y-4 text-center">
    <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent text-center">
      Keyword Insights
    </h2>

    {/* Controls */}
    <div className="flex flex-wrap justify-center gap-3">
      {/* Search */}
      <div className="relative">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search for keywords..."
          className="pl-9 pr-4 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 w-52"
        />
        <Search size={16} className="absolute left-3 top-3 text-slate-400" />
      </div>

      {/* View Switch */}
<div className="flex items-center border border-slate-300 rounded-lg overflow-hidden shadow-sm">
  <button
    onClick={() => setView('bars')}
    className={`flex items-center justify-center px-4 py-2 transition-all duration-200 text-sm font-medium ${
      view === 'bars'
        ? 'bg-blue-600 text-white'
        : 'bg-white text-slate-700 hover:bg-slate-100 hover:text-blue-600'
    }`}
    title="Bar View"
  >
    <BarChart2 size={18} />
  </button>
  <button
    onClick={() => setView('cloud')}
    className={`flex items-center justify-center px-4 py-2 transition-all duration-200 text-sm font-medium ${
      view === 'cloud'
        ? 'bg-blue-600 text-white'
        : 'bg-white text-slate-700 hover:bg-slate-100 hover:text-blue-600'
    }`}
    title="Cloud View"
  >
    <Cloud size={18} />
  </button>
</div>



      {/* Sort */}
      <div className="relative w-[180px]">
  <select
    value={sortBy}
    onChange={(e) => setSortBy(e.target.value as 'value' | 'alphabetical')}
    className="appearance-none w-full pl-9 pr-8 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
  >
    <option value="value">Sort by Value</option>
    <option value="alphabetical">Sort A-Z</option>
  </select>
  <Filter size={16} className="absolute left-2 top-2.5 text-slate-400" />
  <ChevronDown size={16} className="absolute right-2 top-2.5 text-slate-400 pointer-events-none" />
</div>


      {/* Filter */}
      <div className="relative">
  <select
    value={filter}
    onChange={(e) => setFilter(e.target.value as 'all' | 'high' | 'medium' | 'low')}
    className="appearance-none pl-8 pr-8 py-2 rounded-lg border border-slate-300 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
  >
    <option value="all">All Relevance</option>
    <option value="high">High Relevance</option>
    <option value="medium">Medium Relevance</option>
    <option value="low">Low Relevance</option>
  </select>
  <Filter size={16} className="absolute left-2 top-2.5 text-slate-400" />
  <ChevronDown size={16} className="absolute right-2 top-2.5 text-slate-400 pointer-events-none" />
</div>


      {/* Export */}
      <button
  onClick={downloadCSV}
  className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition duration-200"
>
  <Download size={18} />
  <span className="hidden sm:inline text-sm font-medium">Export</span>
</button>

    </div>
  </div>
</motion.div>


          
          {/* Main visualization area */}
          {view === 'bars' && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
              className="bg-white border border-gray-200 rounded-2xl shadow-md p-6 overflow-hidden"
            >
              <h3 className="text-xl font-semibold text-gray-700 mb-6">Keyword Frequency</h3>
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                {processedKeywords.slice(0, displayLimit).map((keyword, index) => {
                  const percent = Math.round((keyword.value / maxValue) * 100);
                  const bg = getSentimentColor(keyword.value);
                  
                  return (
                    <motion.div
                      key={keyword.text}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: index * 0.03 }}
                      className="group"
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-medium text-gray-800">{keyword.text}</span>
                        <span className="text-sm text-gray-500">{keyword.value}</span>
                      </div>
                      <div className="relative h-8 bg-gray-100 rounded-lg overflow-hidden group-hover:shadow-md transition-shadow">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${percent}%` }}
                          transition={{ duration: 0.5, ease: "easeOut" }}
                          className="absolute inset-y-0 left-0 flex items-center justify-end px-3 rounded-lg"
                          style={{
                            backgroundColor: bg,
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
          <div className="overflow-y-auto max-h-[500px] pr-2"></div>
          {view === 'cloud' && (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.4 }}
    className="bg-slate-50 border border-slate-200 rounded-xl shadow-sm p-5"
  >
    <h3 className="text-xl font-semibold text-slate-700 mb-4"></h3>
    <div className="flex flex-wrap justify-center items-start gap-3 max-h-[600px] overflow-y-auto px-2">
      {processedKeywords.slice(0, displayLimit).map((keyword, index) => {
        const normalized = keyword.value / maxValue;
        const size = 0.8 + normalized * 0.8; // scale factor from 0.8x to 1.6x
        const bg =
          normalized > 0.8 ? '#1D4ED8' : // Strong blue
          normalized > 0.6 ? '#3B82F6' :
          normalized > 0.4 ? '#FACC15' :
          normalized > 0.25 ? '#F97316' :
          '#DC2626';

        return (
<motion.div
  key={keyword.text}
  initial={{ opacity: 0, scale: 0.95 }}
  animate={{ opacity: 1, scale: 1 }}
  transition={{
    duration: 0.3,
    delay: index * 0.02,
    type: 'spring',
    stiffness: 200,
  }}
  whileHover={{ scale: 1.08 }}
  className="inline-flex items-center justify-center font-semibold transition-transform cursor-pointer shadow-sm hover:shadow-md"
  style={{
    backgroundColor: getSentimentColor(keyword.value),
    color: getContrastColor(getSentimentColor(keyword.value)),
    fontSize: `${14 + normalized * 8}px`,
    borderRadius: '0.75rem',
    padding: '0.5rem 1rem',
    minHeight: '2.25rem',
    lineHeight: 1.25,
  }}
  title={`${keyword.text}: ${keyword.value}`}
>
  {keyword.text}
  <span className="ml-2 font-semibold text-xs opacity-80">{keyword.value}</span>
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
            <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-4">
              <h4 className="text-lg font-semibold text-gray-700 mb-3">Top Keywords</h4>
              <div className="space-y-2">
                {processedKeywords.slice(0, 5).map((keyword, idx) => (
                  <div key={keyword.text} className="flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="w-5 h-5 flex items-center justify-center bg-blue-100 text-blue-800 rounded-full text-xs font-bold mr-2">
                        {idx + 1}
                      </span>
                      <span className="font-medium">{keyword.text}</span>
                    </div>
                    <span className="text-sm font-semibold">{keyword.value}</span>
                  </div>
                ))}
              </div>
            </div>
            
            
            {/* Summary Stats */}
            <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-4">
              <h4 className="text-lg font-semibold text-gray-700 mb-3">Summary</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Keywords:</span>
                  <span className="font-semibold">{keywords.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Highest Frequency:</span>
                  <span className="font-semibold">{maxValue}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Frequency:</span>
                  <span className="font-semibold">
                    {Math.round(keywords.reduce((sum, k) => sum + k.value, 0) / keywords.length)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Filtered Keywords:</span>
                  <span className="font-semibold">{processedKeywords.length}</span>
                </div>
              </div>
            </div>
          </motion.div>
          
          {/* Expand/Collapse */}
          {processedKeywords.length > baseLimit && (
            <div className="text-center">
              <button
                onClick={() => setExpanded(!expanded)}
                className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium transition-colors px-4 py-2 rounded-full hover:bg-blue-50"
              >
                {expanded ? 'Show less' : `Show all ${processedKeywords.length} keywords`}
                {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}