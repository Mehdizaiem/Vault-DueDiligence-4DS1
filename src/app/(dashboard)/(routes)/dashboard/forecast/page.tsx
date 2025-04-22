"use client";

import { useState } from 'react';
import { Card } from "@/components/ui/card";
import { 
  TrendingUp, 
  TrendingDown,
  AlertCircle,
  Calendar,
  BarChart,
  LineChart as LineChartIcon,
  RefreshCcw
} from "lucide-react";

// Mock data for crypto price forecasts
const cryptoForecasts = [
  {
    name: 'Bitcoin',
    symbol: 'BTC',
    currentPrice: 67432.18,
    predictedPrice: 72500.00,
    change: '+7.52%',
    trend: 'up',
    confidence: 87,
    timeframe: '30 days'
  },
  {
    name: 'Ethereum',
    symbol: 'ETH',
    currentPrice: 3421.50,
    predictedPrice: 3850.25,
    change: '+12.53%',
    trend: 'up',
    confidence: 82,
    timeframe: '30 days'
  },
  {
    name: 'Solana',
    symbol: 'SOL',
    currentPrice: 143.25,
    predictedPrice: 168.75,
    change: '+17.80%',
    trend: 'up',
    confidence: 79,
    timeframe: '30 days'
  },
  {
    name: 'Cardano',
    symbol: 'ADA',
    currentPrice: 0.57,
    predictedPrice: 0.63,
    change: '+10.53%',
    trend: 'up',
    confidence: 71,
    timeframe: '30 days'
  },
  {
    name: 'XRP',
    symbol: 'XRP',
    currentPrice: 0.64,
    predictedPrice: 0.59,
    change: '-7.81%',
    trend: 'down',
    confidence: 68,
    timeframe: '30 days'
  },
  {
    name: 'Polkadot',
    symbol: 'DOT',
    currentPrice: 7.85,
    predictedPrice: 8.67,
    change: '+10.45%',
    trend: 'up',
    confidence: 74,
    timeframe: '30 days'
  }
];

// Mock data for market indicators
const marketIndicators = [
  { name: 'Market Sentiment', value: 'Bullish', status: 'positive' },
  { name: 'Volatility Index', value: 'Moderate', status: 'neutral' },
  { name: 'BTC Dominance', value: '51.3%', status: 'neutral' },
  { name: 'Total Market Cap', value: '$2.47T', status: 'positive' }
];

export default function ForecastPage() {
  const [timeframe, setTimeframe] = useState('30d');
  
  return (
    <div className="flex-1 p-8 pt-6">
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Crypto Forecast</h2>
            <p className="text-muted-foreground">
              Price predictions and market analysis for major cryptocurrencies
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Timeframe selector */}
            <div className="flex border rounded-lg overflow-hidden">
              {['7d', '30d', '90d'].map((time) => (
                <button
                  key={time}
                  onClick={() => setTimeframe(time)}
                  className={`px-4 py-2 text-sm font-medium ${
                    timeframe === time
                      ? 'bg-black text-white'
                      : 'bg-white text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  {time}
                </button>
              ))}
            </div>
            
            <button className="flex items-center gap-2 bg-white border px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
              <RefreshCcw size={16} />
              <span>Refresh</span>
            </button>
          </div>
        </div>
        
        {/* Market Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {marketIndicators.map((indicator) => (
            <Card key={indicator.name} className="p-6">
              <div className="flex justify-between items-start">
                <div>
                  <p className="text-sm text-gray-500">{indicator.name}</p>
                  <h3 className="text-2xl font-bold mt-1">{indicator.value}</h3>
                </div>
                <div className={`rounded-full p-2 ${
                  indicator.status === 'positive' ? 'bg-green-100' : 
                  indicator.status === 'negative' ? 'bg-red-100' : 'bg-gray-100'
                }`}>
                  {indicator.status === 'positive' ? (
                    <TrendingUp className="h-5 w-5 text-green-600" />
                  ) : indicator.status === 'negative' ? (
                    <TrendingDown className="h-5 w-5 text-red-600" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-gray-600" />
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
        
        {/* Price Forecast Chart Placeholder */}
        <Card className="p-6 mb-8">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-lg font-semibold">Price Forecast</h3>
              <p className="text-sm text-gray-500">Predicted price movements based on ML models</p>
            </div>
            <div className="flex gap-3">
              <button className="flex items-center gap-2 px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded-lg text-sm hover:bg-indigo-100">
                <LineChartIcon size={16} />
                <span>Line</span>
              </button>
              <button className="flex items-center gap-2 px-3 py-1.5 bg-white border text-gray-600 rounded-lg text-sm hover:bg-gray-50">
                <BarChart size={16} />
                <span>Bar</span>
              </button>
              <button className="flex items-center gap-2 px-3 py-1.5 bg-white border text-gray-600 rounded-lg text-sm hover:bg-gray-50">
                <Calendar size={16} />
                <span>Calendar</span>
              </button>
            </div>
          </div>
          
          {/* Chart Placeholder */}
          <div className="bg-gray-50 border border-dashed rounded-xl h-80 flex items-center justify-center">
            <div className="text-center">
              <LineChartIcon size={48} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Price forecast chart visualization will appear here</p>
              <p className="text-gray-400 text-sm mt-1">Connect to API data source for live forecasts</p>
            </div>
          </div>
        </Card>
        
        {/* Crypto Forecast Table */}
        <h3 className="text-lg font-semibold mb-4">Cryptocurrency Forecasts</h3>
        <div className="bg-white rounded-xl border overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="text-left text-sm font-medium text-gray-500 px-6 py-3">Asset</th>
                  <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Current Price</th>
                  <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Predicted Price</th>
                  <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Change</th>
                  <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Confidence</th>
                  <th className="text-right text-sm font-medium text-gray-500 px-6 py-3">Timeframe</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {cryptoForecasts.map((crypto) => (
                  <tr key={crypto.symbol} className="hover:bg-gray-50">
                    <td className="whitespace-nowrap px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-full bg-gray-100 flex items-center justify-center">
                          <span className="font-semibold text-xs">{crypto.symbol}</span>
                        </div>
                        <div>
                          <div className="font-medium">{crypto.name}</div>
                          <div className="text-gray-500 text-sm">{crypto.symbol}</div>
                        </div>
                      </div>
                    </td>
                    <td className="whitespace-nowrap text-right px-6 py-4 font-medium">
                      ${crypto.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="whitespace-nowrap text-right px-6 py-4 font-medium">
                      ${crypto.predictedPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className={`whitespace-nowrap text-right px-6 py-4 font-medium ${
                      crypto.trend === 'up' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      <div className="flex items-center justify-end gap-1">
                        {crypto.trend === 'up' ? (
                          <TrendingUp size={16} />
                        ) : (
                          <TrendingDown size={16} />
                        )}
                        {crypto.change}
                      </div>
                    </td>
                    <td className="whitespace-nowrap text-right px-6 py-4">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              crypto.confidence >= 80 ? 'bg-green-500' : 
                              crypto.confidence >= 70 ? 'bg-yellow-500' : 'bg-orange-500'
                            }`}
                            style={{ width: `${crypto.confidence}%` }}
                          ></div>
                        </div>
                        <span className="text-sm">{crypto.confidence}%</span>
                      </div>
                    </td>
                    <td className="whitespace-nowrap text-right px-6 py-4 text-gray-500">
                      {crypto.timeframe}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}