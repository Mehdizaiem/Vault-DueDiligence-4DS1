'use client';

import { useState, useEffect } from 'react';
import { Search, Filter, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

type Keyword = {
  text: string;
  value: number;
};

type Props = {
  keywords: Keyword[];
};

const getKeywordColor = (index: number, value: number, maxValue: number) => {
  const colors = [
    'bg-blue-500',
    'bg-indigo-600',
    'bg-purple-600',
    'bg-green-500',
    'bg-yellow-500',
    'bg-red-500',
  ];
  return colors[index % colors.length];
};

const getTrendIcon = (index: number) => {
  const trends = [
    { icon: '↗', color: 'text-green-500' },
    { icon: '↘', color: 'text-red-500' },
    { icon: '→', color: 'text-amber-500' },
  ];
  return trends[index % trends.length];
};

export default function EnhancedKeywords({ keywords }: Props) {
  const [view, setView] = useState<'cloud' | 'bars' | 'list'>('bars');
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState(false);
  const [sortBy, setSortBy] = useState<'value' | 'name'>('value');
  const [isLoading, setIsLoading] = useState(true);

  const baseLimit = 15;
  const displayLimit = expanded ? keywords.length : baseLimit;

  const maxValue = Math.max(...keywords.map((k) => k.value), 1);

  const filteredKeywords = keywords
    .filter((k) => k.text.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) =>
      sortBy === 'value' ? b.value - a.value : a.text.localeCompare(b.text)
    )
    .slice(0, displayLimit);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 600);
    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return (
      <div className="flex-1 p-8 pt-6 flex items-center justify-center min-h-[100vh]">

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
              <h3 className="text-xl font-medium text-muted-foreground mt-4">Loading Keywords</h3>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-50 via-blue-50/20 to-purple-50/20">

      <div className="p-8 pt-6">
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
                  Trending Keywords
                </h2>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className="relative">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search for keywords..."
                  className="pl-9 pr-4 py-2 rounded-lg border text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <Search size={16} className="absolute left-3 top-3 text-gray-400" />
              </div>

              <div className="flex items-center gap-2 border rounded-lg overflow-hidden">
                <button
                  onClick={() => setView('cloud')}
                  className={`px-3 py-2 text-sm font-medium ${view === 'cloud' ? 'bg-blue-500 text-white' : 'bg-gray-50 text-gray-700 hover:bg-gray-100'}`}
                >
                  Cloud
                </button>
                <button
                  onClick={() => setView('bars')}
                  className={`px-3 py-2 text-sm font-medium ${view === 'bars' ? 'bg-blue-500 text-white' : 'bg-gray-50 text-gray-700 hover:bg-gray-100'}`}
                >
                  Bars
                </button>
                <button
                  onClick={() => setView('list')}
                  className={`px-3 py-2 text-sm font-medium ${view === 'list' ? 'bg-blue-500 text-white' : 'bg-gray-50 text-gray-700 hover:bg-gray-100'}`}
                >
                  List
                </button>
              </div>

              <button
                onClick={() => setSortBy(sortBy === 'value' ? 'name' : 'value')}
                className="flex items-center gap-1 text-sm bg-gray-50 hover:bg-gray-100 px-3 py-2 rounded-lg border"
              >
                <Filter size={16} />
                Sort: {sortBy === 'value' ? 'Frequency' : 'Name'}
              </button>
            </div>
          </motion.div>

          {/* Cloud View */}
          {view === 'cloud' && (
            <div className="flex flex-wrap gap-2 mb-4 justify-start items-start max-w-full">

              {filteredKeywords.map((keyword, index) => {
                const fontSize = Math.max(0.8, (keyword.value / maxValue) * 1.5);
                return (
                  <motion.div
                    key={keyword.text}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    className={`${getKeywordColor(index, keyword.value, maxValue)} text-white px-3 py-1.5 rounded-lg shadow-sm flex items-center gap-1.5 transition-all hover:scale-105 cursor-pointer`}
                    style={{ fontSize: `${fontSize}rem` }}
                  >
                    <span>{keyword.text}</span>
                    <span className="opacity-70 text-xs font-medium">{keyword.value}</span>
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* Bars View */}
          {view === 'bars' && (
            <div className="space-y-3 mb-4">
              {filteredKeywords.map((keyword, index) => {
                const width = Math.max(10, (keyword.value / maxValue) * 100);
                const trend = getTrendIcon(index);
                return (
                  <motion.div
                    key={keyword.text}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    className="relative"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-gray-700 w-24 truncate">{keyword.text}</span>
                      <div
                        className={`${getKeywordColor(index, keyword.value, maxValue)} h-8 rounded-r-lg relative`}
                        style={{ width: `${width}%` }}
                      >
                        <span className="absolute right-2 top-1.5 text-white font-medium">{keyword.value}</span>
                      </div>
                      <span className={`${trend.color} font-bold ml-1`}>{trend.icon}</span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* List View */}
          {view === 'list' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4 auto-rows-min">

              {filteredKeywords.map((keyword, index) => {
                const trend = getTrendIcon(index);
                return (
                  <motion.div
                    key={keyword.text}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                    className="flex items-center justify-between border rounded-lg p-3 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center gap-2">
                      <div className={`${getKeywordColor(index, keyword.value, maxValue)} w-3 h-3 rounded-full`}></div>
                      <span className="font-medium">{keyword.text}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="font-bold">{keyword.value}</span>
                      <span className={`${trend.color} font-bold`}>{trend.icon}</span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* Expand/Collapse */}
          {keywords.length > baseLimit && !expanded && (
  <div className="text-center mb-6">
    <button
      onClick={() => setExpanded(true)}
      className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium"
    >
      Show all {keywords.length} keywords
      <ChevronDown size={16} />
    </button>
  </div>
)}


          {expanded && (
            <div className="text-center">
              <button
                onClick={() => setExpanded(false)}
                className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium"
              >
                Show less
                <ChevronUp size={16} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
