import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { parse } from 'papaparse';
import axios from 'axios';

interface HistoricalData {
  date: string;
  price: number;
}

// Helper function to convert symbol for API
function convertSymbolForApi(symbol: string) {
  const coinIdMap: Record<string, string> = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "DOT": "polkadot",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "LTC": "litecoin"
  };

  // Extract base and quote
  const parts = symbol.split("_");
  const baseSymbol = parts[0].toUpperCase();
  const quoteCurrency = parts.length > 1 ? parts[1].toLowerCase() : "usd";
  
  const coinId = coinIdMap[baseSymbol] || baseSymbol.toLowerCase();
  return { coinId, baseSymbol, quoteCurrency };
}

// Helper function to fetch data from CoinGecko API
async function fetchCoinGeckoData(symbol: string, days = 365) {
  const { coinId, quoteCurrency } = convertSymbolForApi(symbol);
  
  try {
    const url = `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart`;
    const params = {
      vs_currency: quoteCurrency,
      days: days.toString(),
      interval: 'daily'
    };
    
    const headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      'Accept': 'application/json'
    };

    const response = await axios.get(url, { 
      params, 
      headers,
      timeout: 15000
    });
    
    if (response.data?.prices) {
      return response.data.prices.map((item: [number, number]) => ({
        date: new Date(item[0]).toISOString().split('T')[0],
        price: item[1]
      }));
    }
    return null;
  } catch (error: unknown) {
    console.error('Error fetching API data:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

// Helper function to find CSV file
function findCsvFile(symbol: string) {
  const projectRoot = process.cwd();
  const possibleDirs = [
    path.join(projectRoot, 'data', 'time series cryptos'),
    path.join(projectRoot, 'data', 'time_series_cryptos'),
    path.join(projectRoot, 'Sample_Data', 'data', 'time_series'),
    path.join(projectRoot, 'data')
  ];

  for (const dir of possibleDirs) {
    if (!fs.existsSync(dir)) continue;

    const patterns = [
      `*${symbol}*.csv`,
      `*${symbol.replace('_', '')}*.csv`,
      `*${symbol.replace('_USD', '')}*.csv`,
      `*${symbol.replace('_USDT', '')}*.csv`,
      `*${symbol.split('_')[0]}*.csv`
    ];

    for (const pattern of patterns) {
      try {
        // Try to find files matching the pattern
        const files = fs.readdirSync(dir)
          .filter(file => 
            file.toLowerCase().includes(symbol.toLowerCase().replace('_', '')) && 
            file.endsWith('.csv')
          );

        if (files.length > 0) {
          return path.join(dir, files[0]);
        }
      } catch (err) {
        console.error(`Error searching for files with pattern ${pattern} in ${dir}:`, err);
      }
    }
  }

  return null;
}

// Helper function to load CSV data
function loadCsvData(filePath: string): HistoricalData[] | null {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    
    const parseResult = parse(fileContent, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true
    });
    
    if (parseResult.errors.length > 0) {
      console.warn('CSV parsing errors:', parseResult.errors);
    }
    
    const records = parseResult.data as Record<string, unknown>[];
    
    // Identify date and price columns
    const dateColumn = findColumn(records[0], ['Date', 'date', 'timestamp', 'Timestamp', 'Time', 'time']);
    const priceColumn = findColumn(records[0], ['price', 'Price', 'close', 'Close', 'last', 'Last', 'value', 'Value']);
    
    if (!dateColumn || !priceColumn) {
      console.error('Could not identify date or price columns in CSV');
      return null;
    }
    
    return records
      .map((record: Record<string, unknown>) => {
        const dateValue = record[dateColumn];
        let date;
        
        if (typeof dateValue === 'string') {
          // Try different date formats
          date = new Date(dateValue);
          if (isNaN(date.getTime())) {
            // Try alternative formats
            const formats = [
              /(\d{4})-(\d{2})-(\d{2})/, // YYYY-MM-DD
              /(\d{2})[/-](\d{2})[/-](\d{4})/, // DD/MM/YYYY or MM/DD/YYYY
              /(\d{2})[/-](\d{2})[/-](\d{2})/ // DD/MM/YY or MM/DD/YY
            ];
            
            for (const format of formats) {
              const match = dateValue.match(format);
              if (match) {
                if (format === formats[0]) {
                  date = new Date(`${match[1]}-${match[2]}-${match[3]}`);
                } else {
                  // Assume MM/DD/YYYY for simplicity
                  date = new Date(`${match[2]}/${match[1]}/${match[3]}`);
                }
                break;
              }
            }
          }
        } else if (typeof dateValue === 'number') {
          // Assume Unix timestamp (in seconds)
          date = new Date(dateValue * 1000);
        }
        
        if (!date || isNaN(date.getTime())) {
          return null;
        }
        
        let price = record[priceColumn];
        if (typeof price === 'string') {
          price = parseFloat(price.replace(/,/g, ''));
        }
        
        if (isNaN(price as number)) {
          return null;
        }
        
        return {
          date: date.toISOString().split('T')[0],
          price: price as number
        };
      })
      .filter((item): item is HistoricalData => item !== null)
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  } catch (error) {
    console.error('Error loading CSV data:', error);
    return null;
  }
}

// Helper function to find column with matching name
function findColumn(record: Record<string, unknown> | null, possibleNames: string[]): string | null {
  if (!record) return null;
  
  const recordKeys = Object.keys(record);
  for (const name of possibleNames) {
    const found = recordKeys.find(key => key.toLowerCase() === name.toLowerCase());
    if (found) return found;
  }
  return null;
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const symbol = searchParams.get('symbol') || 'BTC_USD';
    const daysParam = searchParams.get('days') || '365';
    const days = parseInt(daysParam, 10);
    
    // First try to get data from API for most recent data
    console.log(`Fetching data from API for ${symbol} (${days} days)`);
    const apiData = await fetchCoinGeckoData(symbol, days);
    
    if (apiData && apiData.length > 0) {
      console.log(`Successfully fetched ${apiData.length} days of data from API`);
      return NextResponse.json({
        success: true,
        data: apiData
      });
    }
    
    // If API fails, fall back to CSV data
    console.log('API fetch failed or returned no data, falling back to CSV');
    const csvFile = findCsvFile(symbol);
    
    if (!csvFile) {
      return NextResponse.json({
        success: false,
        error: `No data available for ${symbol}`
      });
    }
    
    // Load CSV data
    console.log(`Loading CSV data from ${csvFile}`);
    const csvData = loadCsvData(csvFile);
    
    if (!csvData || csvData.length === 0) {
      return NextResponse.json({
        success: false,
        error: `Failed to load data for ${symbol}`
      });
    }
    
    // Limit data to requested number of days
    const limitedData = csvData.slice(-days);
    
    return NextResponse.json({
      success: true,
      source: 'csv',
      data: limitedData
    });

  } catch (error) {
    console.error('Error in historical data route:', error);
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch historical data'
    });
  }
}

export const dynamic = 'force-dynamic';