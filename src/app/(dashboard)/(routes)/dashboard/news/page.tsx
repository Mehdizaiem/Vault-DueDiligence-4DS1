"use client";

import { useState } from 'react';
import { 
  Newspaper, 
  Search, 
  Bell, 
  Star, 
  MessageSquare, 
  Share2, 
  BookmarkPlus,
  TrendingUp,
  Filter,
  ChevronDown,
  ExternalLink,
  AlertCircle,
  BarChart
} from "lucide-react";

// Mock data for news articles
const newsArticles = [
  {
    id: 1,
    title: 'SEC Approves Multiple Bitcoin ETFs, Opening Doors for Institutional Investment',
    source: 'CryptoNews',
    category: 'Regulation',
    date: '2025-04-18',
    snippet: 'The Securities and Exchange Commission has approved several Bitcoin ETF applications, signaling a major shift in regulatory stance...',
    image: '/crypto-news-1.jpg',
    impact: 'high',
    bookmarked: false,
    commentCount: 24,
    shareCount: 18
  },
  {
    id: 2,
    title: 'Ethereum Layer 2 Solutions See Record Growth as Gas Fees Decrease',
    source: 'DeFi Insider',
    category: 'Technology',
    date: '2025-04-17',
    snippet: 'Ethereum scaling solutions like Optimism and Arbitrum are experiencing unprecedented growth as users migrate to Layer 2 networks...',
    image: '/crypto-news-2.jpg',
    impact: 'medium',
    bookmarked: true,
    commentCount: 14,
    shareCount: 9
  },
  {
    id: 3,
    title: 'Central Banks Worldwide Accelerate CBDC Development Efforts',
    source: 'Global Finance',
    category: 'CBDC',
    date: '2025-04-16',
    snippet: 'Central banks across the globe are intensifying their research and development of Central Bank Digital Currencies in response to...',
    image: '/crypto-news-3.jpg',
    impact: 'high',
    bookmarked: false,
    commentCount: 32,
    shareCount: 27
  },
  {
    id: 4,
    title: 'Major DeFi Protocol Announces Governance Token to Decentralize Platform',
    source: 'DeFi Daily',
    category: 'DeFi',
    date: '2025-04-15',
    snippet: 'A leading decentralized finance protocol has announced the launch of its governance token, enabling community members to participate in decision-making...',
    image: '/crypto-news-4.jpg',
    impact: 'medium',
    bookmarked: false,
    commentCount: 19,
    shareCount: 12
  },
  {
    id: 5,
    title: 'Bitcoin Mining Companies Shift to Renewable Energy Sources',
    source: 'Crypto Environmental',
    category: 'Mining',
    date: '2025-04-14',
    snippet: 'Major Bitcoin mining operations are rapidly transitioning to renewable energy sources amid growing environmental concerns and regulatory pressure...',
    image: '/crypto-news-5.jpg',
    impact: 'medium',
    bookmarked: false,
    commentCount: 27,
    shareCount: 31
  },
  {
    id: 6,
    title: 'New Regulatory Framework for Crypto Assets Proposed by G20 Nations',
    source: 'Global Policy',
    category: 'Regulation',
    date: '2025-04-13',
    snippet: 'The G20 nations have jointly proposed a comprehensive regulatory framework for cryptocurrency assets, aiming to address issues of taxation, consumer protection...',
    image: '/crypto-news-6.jpg',
    impact: 'high',
    bookmarked: true,
    commentCount: 42,
    shareCount: 38
  }
];

// Mock data for trending topics
const trendingTopics = [
  'Bitcoin ETF', 'Layer 2 Scaling', 'CBDCs', 'DeFi Regulation', 
  'Green Mining', 'Stablecoin Oversight', 'NFT Market Recovery'
];

// Mock data for impact metrics
const impactMetrics = [
  {
    label: 'Regulatory Updates',
    count: 24,
    change: '+8',
    trend: 'up'
  },
  {
    label: 'Market Events',
    count: 37,
    change: '+12',
    trend: 'up'
  },
  {
    label: 'Technology Updates',
    count: 19,
    change: '+5',
    trend: 'up'
  },
  {
    label: 'Security Incidents',
    count: 7,
    change: '-2',
    trend: 'down'
  }
];

export default function NewsPage() {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const categories = ['All', 'Regulation', 'Technology', 'CBDC', 'DeFi', 'Mining', 'Security'];
  const [searchQuery, setSearchQuery] = useState('');
  
  // Filter articles based on selected category and search query
  const filteredArticles = newsArticles.filter(article => {
    const matchesCategory = selectedCategory === 'All' || article.category === selectedCategory;
    const matchesSearch = searchQuery === '' || 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.snippet.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesCategory && matchesSearch;
  });
  
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto News</h2>
            <p className="text-muted-foreground">
              Latest news, regulatory updates, and market events
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="relative w-full md:w-auto">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
              <input 
                type="text" 
                placeholder="Search news..." 
                className="pl-9 pr-4 py-2 border rounded-lg text-sm w-full md:w-64"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            
            <button className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
              <Bell size={16} />
              <span className="hidden md:inline">Alerts</span>
            </button>
            
            <button className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
              <Filter size={16} />
              <span className="hidden md:inline">Filter</span>
              <ChevronDown size={16} className="md:ml-1" />
            </button>
          </div>
        </div>
        
        {/* Impact Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {impactMetrics.map((metric) => (
            <div key={metric.label} className="bg-white p-4 rounded-xl border shadow-sm">
              <div className="flex justify-between items-start">
                <p className="text-sm text-gray-500">{metric.label}</p>
                <div className={`rounded-full p-1 ${
                  metric.trend === 'up' ? 'bg-blue-100' : 'bg-gray-100'
                }`}>
                  {metric.trend === 'up' ? (
                    <TrendingUp className="h-4 w-4 text-blue-600" />
                  ) : (
                    <TrendingUp className="h-4 w-4 text-gray-500 rotate-180" />
                  )}
                </div>
              </div>
              <div className="flex items-end mt-2">
                <span className="text-2xl font-bold">{metric.count}</span>
                <span className={`text-xs ml-2 ${
                  metric.trend === 'up' 
                    ? 'text-green-600' 
                    : 'text-red-600'
                }`}>
                  {metric.change}
                </span>
              </div>
            </div>
          ))}
        </div>
        
        {/* Category Pills */}
        <div className="flex flex-wrap gap-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedCategory === category
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border hover:bg-gray-50 text-gray-700'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
        
        {/* News Feed Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-7 gap-6">
          {/* Main News Feed */}
          <div className="lg:col-span-5 space-y-6">
            {filteredArticles.length > 0 ? (
              filteredArticles.map((article) => (
                <article key={article.id} className="bg-white rounded-xl border shadow-sm overflow-hidden">
                  <div className="p-6">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-1 rounded-full font-medium ${
                          article.impact === 'high' 
                            ? 'bg-red-100 text-red-700' 
                            : article.impact === 'medium'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-green-100 text-green-700'
                        }`}>
                          {article.impact.charAt(0).toUpperCase() + article.impact.slice(1)} Impact
                        </span>
                        <span className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-700">
                          {article.category}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">{article.date}</span>
                    </div>
                    
                    <h3 className="text-lg font-semibold mb-2">{article.title}</h3>
                    <p className="text-gray-600 text-sm mb-4">{article.snippet}</p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1">
                        <span className="text-sm font-medium">{article.source}</span>
                        <ExternalLink size={14} className="text-gray-400" />
                      </div>
                      
                      <div className="flex items-center gap-3">
                        <button className="text-gray-400 hover:text-gray-600">
                          <MessageSquare size={18} />
                        </button>
                        <button className="text-gray-400 hover:text-gray-600">
                          <Share2 size={18} />
                        </button>
                        <button className={`${
                          article.bookmarked ? 'text-blue-500' : 'text-gray-400 hover:text-gray-600'
                        }`}>
                          <BookmarkPlus size={18} />
                        </button>
                      </div>
                    </div>
                  </div>
                </article>
              ))
            ) : (
              <div className="bg-white rounded-xl border shadow-sm p-8 text-center">
                <Newspaper className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                <h3 className="text-lg font-medium text-gray-700 mb-1">No news articles found</h3>
                <p className="text-gray-500 text-sm">
                  Try adjusting your filters or search criteria
                </p>
              </div>
            )}
          </div>
          
          {/* Sidebar */}
          <div className="lg:col-span-2 space-y-6">
            {/* Trending Topics */}
            <div className="bg-white rounded-xl border shadow-sm p-5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="h-5 w-5 text-blue-500" />
                <h3 className="font-semibold">Trending Topics</h3>
              </div>
              <div className="space-y-2">
                {trendingTopics.map((topic, index) => (
                  <div 
                    key={index} 
                    className="px-3 py-2 rounded-lg text-sm hover:bg-gray-50 cursor-pointer transition-colors flex items-center justify-between"
                  >
                    <span>{topic}</span>
                    {index < 3 && <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />}
                  </div>
                ))}
              </div>
            </div>
            
            {/* News Impact Chart */}
            <div className="bg-white rounded-xl border shadow-sm p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <BarChart className="h-5 w-5 text-indigo-500" />
                  <h3 className="font-semibold">Market Impact</h3>
                </div>
                <button className="text-xs text-gray-500 hover:text-gray-700">
                  Last 7 days
                </button>
              </div>
              
              {/* Chart placeholder */}
              <div className="h-48 bg-gray-50 rounded-lg border border-dashed flex items-center justify-center mb-3">
                <div className="text-center p-4">
                  <p className="text-gray-500 text-sm">Market impact visualization</p>
                  <p className="text-gray-400 text-xs mt-1">Connect to data source</p>
                </div>
              </div>
            </div>
            
            {/* Important Alert */}
            <div className="bg-amber-50 rounded-xl border border-amber-200 p-5">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-amber-500 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-amber-800 mb-1">Regulatory Alert</h3>
                  <p className="text-amber-700 text-sm">
                    New compliance requirements for crypto exchanges take effect next month.
                  </p>
                  <button className="mt-3 text-sm text-amber-800 font-medium hover:text-amber-900">
                    View Details
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}