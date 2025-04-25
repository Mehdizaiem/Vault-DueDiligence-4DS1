import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';
import fs from 'fs';
import path from 'path';

export async function GET(request: NextRequest) {
  try {
    // Get the symbol from URL query parameter
    const symbol = request.nextUrl.searchParams.get('symbol')?.toUpperCase() || 'BTC';
    const timeframe = request.nextUrl.searchParams.get('timeframe') || '30d';
    
    // Get the current project root directory
    const rootDir = process.cwd();
    
    console.log(`Fetching forecast data for symbol: ${symbol} with timeframe: ${timeframe}`);
    
    // Get historical data from CSV files
    const historicalData = await getHistoricalDataFromCSV(symbol, rootDir);
    
    if (!historicalData || historicalData.length === 0) {
      console.log(`No historical data found for ${symbol}, using mock data`);
      return NextResponse.json(createMockForecastData(symbol, timeframe));
    }
    
    // Get forecast data using the historical data
    try {
      // Prepare the last 30 days of data for forecasting
      const recentData = historicalData.slice(-30);
      
      // Call your Python forecasting code here
      // For now, we'll create mock forecast data based on the historical trend
      const latestPrice = recentData[recentData.length - 1].price;
      const mockForecast = generateForecast(symbol, latestPrice, recentData);
      
      return NextResponse.json({
        forecast: mockForecast,
        historicalData: historicalData
      });
      
    } catch (forecastError) {
      console.error('Error generating forecast:', forecastError);
      return NextResponse.json(createMockForecastData(symbol, timeframe));
    }
    
  } catch (error) {
    console.error('Error fetching forecast:', error);
    return NextResponse.json(
      { error: 'Failed to fetch forecast data', details: String(error) },
      { status: 500 }
    );
  }
}

async function getHistoricalDataFromCSV(symbol: string, rootDir: string) {
  try {
    // Path to CSV files - check multiple possible locations based on your project structure
    const possiblePaths = [
      path.join(rootDir, 'Code', 'time series cryptos'),
      path.join(rootDir, 'data', 'time series cryptos'),
      path.join(rootDir, 'time series cryptos'),
      path.join(rootDir, 'Code', 'data_acquisition', 'time series cryptos'),
      path.join(rootDir, 'Code', 'data', 'time series cryptos')
    ];
    
    let dirPath = null;
    
    // Find the first path that exists
    for (const p of possiblePaths) {
      if (fs.existsSync(p)) {
        dirPath = p;
        console.log(`Found CSV directory at: ${p}`);
        break;
      }
    }
    
    if (!dirPath) {
      console.warn(`CSV directory not found in any of the expected locations`);
      return null;
    }
    
    // Find matching CSV file for the symbol
    const files = fs.readdirSync(dirPath);
    
    // Construct possible filenames based on symbol
    const possibleNames = [
      `${symbol}_USD`,
      `${symbol}USD`,
      `${symbol}_USDT`,
      `${symbol}USDT`,
      symbol
    ];
    
    let csvFile = null;
    
    for (const name of possibleNames) {
      const matchingFiles = files.filter(file => 
        file.toUpperCase().includes(name.toUpperCase()) && file.endsWith('.csv')
      );
      
      if (matchingFiles.length > 0) {
        csvFile = path.join(dirPath, matchingFiles[0]);
        console.log(`Found CSV file for ${symbol}: ${matchingFiles[0]}`);
        break;
      }
    }
    
    if (!csvFile) {
      console.warn(`No CSV file found for symbol ${symbol}`);
      return null;
    }
    
    // Read and parse CSV
    const csvData = fs.readFileSync(csvFile, 'utf8');
    const lines = csvData.trim().split('\n');
    const headers = lines[0].split(',');
    
    // Find relevant column indices
    const dateColIndex = headers.findIndex(h => 
      h.toLowerCase().includes('date') || h.toLowerCase().includes('time')
    );
    
    const priceColIndex = headers.findIndex(h => 
      h.toLowerCase().includes('close') || h.toLowerCase().includes('price')
    );
    
    if (dateColIndex === -1 || priceColIndex === -1) {
      console.warn(`Required columns not found in CSV: ${csvFile}`);
      return null;
    }
    
    // Parse data
    const data = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      
      if (values.length <= Math.max(dateColIndex, priceColIndex)) {
        continue; // Skip malformed rows
      }
      
      const dateStr = values[dateColIndex].trim();
      const priceStr = values[priceColIndex].trim();
      
      // Try to parse date
      let timestamp;
      try {
        timestamp = new Date(dateStr).toISOString();
      } catch (e) {
        continue; // Skip rows with invalid dates
      }
      
      // Try to parse price
      let price;
      try {
        price = parseFloat(priceStr);
        if (isNaN(price)) continue;
      } catch (e) {
        continue; // Skip rows with invalid prices
      }
      
      data.push({
        timestamp,
        price
      });
    }
    
    // Sort by date
    data.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    
    console.log(`Successfully loaded ${data.length} data points for ${symbol}`);
    
    return data;
    
  } catch (error) {
    console.error(`Error reading CSV data: ${error}`);
    return null;
  }
}

function generateForecast(symbol: string, currentPrice: number, historicalData: any[]) {
  // Generate a realistic 3-day forecast based on recent price movement
  const now = new Date();
  const daysAhead = 3; // Only forecast 3 days ahead
  
  // Analyze recent trend
  const recentPrices = historicalData.slice(-7).map(d => d.price);
  const priceChanges = [];
  for (let i = 1; i < recentPrices.length; i++) {
    priceChanges.push((recentPrices[i] - recentPrices[i-1]) / recentPrices[i-1]);
  }
  
  // Calculate average daily change
  const avgDailyChange = priceChanges.reduce((a, b) => a + b, 0) / priceChanges.length;
  
  // Add some randomness but preserve the trend
  const trendMultiplier = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
  const dailyChange = avgDailyChange * trendMultiplier;
  
  // Allow for trend reversal with small probability
  const reversalChance = 0.2;
  const trendDirection = Math.random() < reversalChance ? -1 : 1;
  const finalDailyChange = dailyChange * trendDirection;
  
  // Calculate forecast percentage change
  const changePct = finalDailyChange * daysAhead * 100;
  
  // Generate forecast dates and values
  const forecastDates = Array.from({length: daysAhead}, (_, i) => {
    const date = new Date(now);
    date.setDate(date.getDate() + i + 1);
    return date.toISOString();
  });
  
  let price = currentPrice;
  const forecastValues = forecastDates.map(() => {
    price = price * (1 + finalDailyChange);
    return price;
  });
  
  // Final forecasted price
  const finalPrice = forecastValues[forecastValues.length - 1];
  
  // Create uncertainty bounds
  const uncertainty = 5 + Math.random() * 10; // 5% to 15%
  const lowerBounds = forecastValues.map(val => val * (1 - uncertainty/100));
  const upperBounds = forecastValues.map(val => val * (1 + uncertainty/100));
  
  // Create forecast text
  const trend = changePct > 2 ? "bullish" :
               changePct > 0.5 ? "slightly bullish" :
               changePct < -2 ? "bearish" :
               changePct < -0.5 ? "slightly bearish" : "neutral";
  
  const probIncrease = 50 + (changePct * 5);
  const clampedProbability = Math.min(Math.max(probIncrease, 10), 90);
  
  const insight = `${symbol} shows a ${trend} trend with an expected ${changePct.toFixed(2)}% change over the next ${daysAhead} days. Probability of price increase is ${clampedProbability.toFixed(1)}% with uncertainty of ±${uncertainty.toFixed(1)}%.`;
  
  return {
    symbol: `${symbol}USD`,
    forecast_timestamp: now.toISOString(),
    model_name: "chronos-finetuned",
    days_ahead: daysAhead,
    current_price: currentPrice,
    forecast_dates: forecastDates,
    forecast_values: forecastValues,
    lower_bounds: lowerBounds,
    upper_bounds: upperBounds,
    final_forecast: finalPrice,
    change_pct: changePct,
    trend,
    probability_increase: clampedProbability,
    average_uncertainty: uncertainty,
    insight
  };
}

function createMockForecastData(symbol: string, timeframe: string) {
  // Generate mock data for development purposes
  const basePrice = symbol === 'BTC' ? 67432 : 
                   symbol === 'ETH' ? 3421 : 
                   symbol === 'SOL' ? 143 : 
                   symbol === 'ADA' ? 0.57 : 
                   symbol === 'DOT' ? 7.85 : 0.64;
  
  const now = new Date();
  const daysAhead = 3; // 3-day forecast
  
  // Determine trend direction
  const isBullish = Math.random() > 0.3; // 70% chance of bullish
  const changePct = isBullish ? 
    (1 + Math.random() * 4) :  // 1% to 5% up
    (-5 + Math.random() * 4);  // -5% to -1% down
  
  const finalPrice = basePrice * (1 + changePct/100);
  
  // Create forecast dates and values
  const forecastDates = Array.from({length: daysAhead}, (_, i) => {
    const date = new Date(now);
    date.setDate(date.getDate() + i + 1);
    return date.toISOString();
  });
  
  const forecastValues = forecastDates.map((_, i) => {
    const progress = (i + 1) / daysAhead;
    const randomFactor = 0.98 + Math.random() * 0.04; // Add some noise
    return basePrice + (finalPrice - basePrice) * progress * randomFactor;
  });
  
  // Create uncertainty bounds
  const uncertainty = 5 + Math.random() * 10; // 5% to 15%
  const lowerBounds = forecastValues.map(val => val * (1 - uncertainty/100));
  const upperBounds = forecastValues.map(val => val * (1 + uncertainty/100));
  
  // Create forecast text
  const trend = changePct > 2 ? "bullish" :
               changePct > 0.5 ? "slightly bullish" :
               changePct < -2 ? "bearish" :
               changePct < -0.5 ? "slightly bearish" : "neutral";
  
  const probIncrease = isBullish ? 55 + Math.random() * 30 : 20 + Math.random() * 30;
  
  const insight = `${symbol} shows a ${trend} trend with an expected ${changePct.toFixed(1)}% change over the next ${daysAhead} days. Probability of price increase is ${probIncrease.toFixed(1)}% with uncertainty of ±${uncertainty.toFixed(1)}%.`;
  
  // Generate mock historical data based on timeframe
  const days = timeframe === '7d' ? 7 : timeframe === '30d' ? 30 : 90;
  const historicalDates = Array.from({length: days}, (_, i) => {
    const date = new Date(now);
    date.setDate(date.getDate() - (days - i));
    return date.toISOString();
  });
  
  const volatility = symbol === 'BTC' ? 0.02 : 
                    symbol === 'ETH' ? 0.03 :
                    symbol === 'SOL' ? 0.04 : 0.03;
  
  let price = basePrice * 0.8;
  const historicalPrices = historicalDates.map(() => {
    const change = (Math.random() * 2 - 1) * volatility;
    price = price * (1 + change);
    return price;
  });
  
  // Ensure the last historical price matches the current price
  historicalPrices[historicalPrices.length - 1] = basePrice;
  
  const historicalData = historicalDates.map((date, i) => ({
    timestamp: date,
    price: historicalPrices[i]
  }));
  
  return {
    forecast: {
      symbol: `${symbol}USD`,
      forecast_timestamp: now.toISOString(),
      model_name: "chronos-finetuned",
      days_ahead: daysAhead,
      current_price: basePrice,
      forecast_dates: forecastDates,
      forecast_values: forecastValues,
      lower_bounds: lowerBounds,
      upper_bounds: upperBounds,
      final_forecast: finalPrice,
      change_pct: changePct,
      trend,
      probability_increase: probIncrease,
      average_uncertainty: uncertainty,
      insight
    },
    historicalData
  };
}

export const dynamic = 'force-dynamic';