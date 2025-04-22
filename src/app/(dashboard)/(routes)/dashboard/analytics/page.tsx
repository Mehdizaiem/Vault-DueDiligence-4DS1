"use client";

import { useState } from 'react';
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
  DollarSign,
  Users,
  Activity
} from "lucide-react";

// Mock data for asset distribution
const assetDistribution = [
  { name: 'Bitcoin', value: 45, color: 'bg-orange-500' },
  { name: 'Ethereum', value: 30, color: 'bg-indigo-500' },
  { name: 'Stablecoins', value: 15, color: 'bg-green-500' },
  { name: 'Other Altcoins', value: 10, color: 'bg-gray-500' },
];

// Mock data for portfolios
const portfolios = [
  {
    id: 1,
    name: 'Alpha Capital Fund',
    allocation: '$850M',
    performance: '+12.4%',
    risk: 'Medium',
    holdings: 12,
    trend: 'up'
  },
  {
    id: 2,
    name: 'Blockchain Ventures',
    allocation: '$550M',
    performance: '+8.7%',
    risk: 'Medium-High',
    holdings: 18,
    trend: 'up'
  },
  {
    id: 3,
    name: 'Digital Assets Growth',
    allocation: '$425M',
    performance: '+15.3%',
    risk: 'High',
    holdings: 24,
    trend: 'up'
  },
  {
    id: 4,
    name: 'Stablecoin Reserve',
    allocation: '$320M',
    performance: '+2.8%',
    risk: 'Low',
    holdings: 6,
    trend: 'up'
  },
  {
    id: 5,
    name: 'DeFi Opportunities',
    allocation: '$240M',
    performance: '-3.2%',
    risk: 'High',
    holdings: 15,
    trend: 'down'
  }
];

// Mock data for key performance indicators
const kpis = [
  {
    label: 'Total AUM',
    value: '$2.4B',
    change: '+8.5%',
    trend: 'up',
    icon: DollarSign,
    color: 'text-green-500',
    bgColor: 'bg-green-100'
  },
  {
    label: 'Asset Count',
    value: '34',
    change: '+3',
    trend: 'up',
    icon: FileText,
    color: 'text-blue-500',
    bgColor: 'bg-blue-100'
  },
  {
    label: 'Active Investors',
    value: '842',
    change: '+24',
    trend: 'up',
    icon: Users,
    color: 'text-violet-500',
    bgColor: 'bg-violet-100'
  },
  {
    label: 'Monthly Returns',
    value: '9.2%',
    change: '+2.1%',
    trend: 'up',
    icon: Activity,
    color: 'text-pink-500',
    bgColor: 'bg-pink-100'
  }
];

export default function AnalyticsPage() {
  const [dateRange, setDateRange] = useState('30d');
  
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Analytics</h2>
            <p className="text-muted-foreground">
              Comprehensive analytics and insights for crypto portfolios
            </p>
          </div>
          
          <div className="flex flex-wrap items-center gap-3">
            {/* Date range selector */}
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
        
        {/* Key Performance Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
          {kpis.map((kpi) => (
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
        
        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
          {/* Portfolio Performance Chart */}
          <Card className="p-6 lg:col-span-2">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-lg font-semibold">Portfolio Performance</h3>
                <p className="text-sm text-gray-500">Performance over time across all portfolios</p>
              </div>
              <div className="flex gap-2">
                <button className="bg-gray-100 text-gray-600 px-3 py-1 text-xs rounded-lg">
                  All
                </button>
                <button className="bg-white text-gray-600 px-3 py-1 text-xs rounded-lg border">
                  BTC
                </button>
                <button className="bg-white text-gray-600 px-3 py-1 text-xs rounded-lg border">
                  ETH
                </button>
                <button className="bg-white text-gray-600 px-3 py-1 text-xs rounded-lg border">
                  More
                </button>
              </div>
            </div>
            
            {/* Chart Placeholder */}
            <div className="bg-gray-50 border border-dashed rounded-xl h-64 flex items-center justify-center">
              <div className="text-center">
                <BarChart size={48} className="mx-auto text-gray-300 mb-3" />
                <p className="text-gray-500">Portfolio performance chart will appear here</p>
                <p className="text-gray-400 text-sm mt-1">Connect to API data source for live analytics</p>
              </div>
            </div>
          </Card>
          
          {/* Asset Distribution Chart */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-lg font-semibold">Asset Distribution</h3>
                <p className="text-sm text-gray-500">Allocation across asset classes</p>
              </div>
              <button className="text-gray-400 hover:text-gray-500">
                <ChevronDown size={18} />
              </button>
            </div>
            
            {/* Pie Chart Placeholder */}
            <div className="flex items-center justify-center h-48 mb-4">
              <div className="text-center">
                <PieChart size={48} className="mx-auto text-gray-300 mb-3" />
                <p className="text-gray-500">Distribution chart will appear here</p>
              </div>
            </div>
            
            {/* Legend */}
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
        
        {/* Portfolios Table */}
        <div className="mt-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Portfolio Analysis</h3>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
              <input 
                type="text" 
                placeholder="Search portfolios..." 
                className="pl-9 pr-4 py-2 border rounded-lg text-sm"
              />
            </div>
          </div>
          
          <div className="bg-white rounded-xl border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    <th className="text-left text-sm font-medium text-gray-500 px-6 py-3">Portfolio</th>
                    <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Allocation</th>
                    <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Performance</th>
                    <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Risk Level</th>
                    <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Holdings</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {portfolios.map((portfolio) => (
                    <tr key={portfolio.id} className="hover:bg-gray-50">
                      <td className="whitespace-nowrap px-6 py-4">
                        <div className="font-medium">{portfolio.name}</div>
                      </td>
                      <td className="whitespace-nowrap text-right px-6 py-4 font-medium">
                        {portfolio.allocation}
                      </td>
                      <td className={`whitespace-nowrap text-right px-6 py-4 font-medium ${
                        portfolio.trend === 'up' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        <div className="flex items-center justify-end gap-1">
                          {portfolio.trend === 'up' ? (
                            <TrendingUp size={16} />
                          ) : (
                            <TrendingUp className="rotate-180" size={16} />
                          )}
                          {portfolio.performance}
                        </div>
                      </td>
                      <td className="whitespace-nowrap text-right px-6 py-4">
                        <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                          portfolio.risk === 'Low' ? 'bg-green-100 text-green-700' :
                          portfolio.risk === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                          portfolio.risk === 'Medium-High' ? 'bg-orange-100 text-orange-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {portfolio.risk}
                        </span>
                      </td>
                      <td className="whitespace-nowrap text-right px-6 py-4 text-gray-500">
                        {portfolio.holdings} assets
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}