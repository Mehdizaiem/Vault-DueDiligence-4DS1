import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';
import { parse } from 'csv-parse/sync';
import fs from 'fs';
import path from 'path';
import axios from 'axios';

interface HistoricalData {
  date: string;
  price: number;
  open: number;
  high: number;
  low: number;
  volume: number;
}

// Helper function to convert symbol for API
function convertSymbolForApi(symbol: string) {
  const coinIdMap: { [key: string]: string } = {
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

// Helper function to fetch API data
async function fetchApiData(symbol: string, days: number | string = 'max') {
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
      validateStatus: (status) => status === 200 // Only accept 200 status
    });
    
    if (response.data?.prices) {
      return response.data.prices.map((item: [number, number]) => ({
        date: new Date(item[0]),
        price: item[1]
      }));
    }
    return null;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Error fetching API data:', error.response?.status, error.message);
    } else {
      console.error('Error fetching API data:', (error as Error).message);
    }
    // Don't throw error, just return null to fallback to CSV
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
      `*${symbol.replace('USDT', '')}*.csv`,
      `*${symbol.replace('USD', '')}*.csv`
    ];

    for (const pattern of patterns) {
      const files = fs.readdirSync(dir).filter(file => 
        file.toLowerCase().includes(symbol.toLowerCase()) && file.endsWith('.csv')
      );

      if (files.length > 0) {
        return path.join(dir, files[0]);
      }
    }
  }

  return null;
}

// Helper function to load CSV data
function loadCsvData(filePath: string) {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
      relax_quotes: true,
      relax_column_count: true,
      bom: true
    });

    return records
      .map((record: any) => {
        const date = new Date(record.Date || record.date || record.timestamp);
        const price = parseFloat(String(record.Price || record.price || record.close || record.Close).replace(/,/g, ''));
        return { date, price };
      })
      .filter((record: any) => !isNaN(record.price) && record.date instanceof Date)
      .sort((a: any, b: any) => a.date.getTime() - b.date.getTime());
  } catch (error) {
    console.error('Error loading CSV data:', error);
    return null;
  }
}

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const symbol = searchParams.get('symbol') || 'BTC_USD';
    
    // Try to find CSV file first
    const csvFile = findCsvFile(symbol);
    if (!csvFile) {
      return NextResponse.json({
        success: false,
        error: `No data available for ${symbol}`
      });
    }

    // Load CSV data
    console.log('Loading CSV data from:', csvFile);
    const csvData = loadCsvData(csvFile);
    if (!csvData) {
      return NextResponse.json({
        success: false,
        error: `Failed to load CSV data for ${symbol}`
      });
    }

    // Try to get recent data from API (but don't fail if we can't)
    let apiData = null;
    try {
      apiData = await fetchApiData(symbol, 'max');
    } catch (error) {
      console.error('Failed to fetch API data:', error instanceof Error ? error.message : 'Unknown error');
    }

    if (apiData && apiData.length > 0) {
      // Find overlap to align data
      const apiFirstDate = new Date(apiData[0].date).getTime();
      const csvLastDate = new Date(csvData[csvData.length - 1].date).getTime();

      let combinedData;
      if (apiFirstDate <= csvLastDate) {
        // Remove overlapping days from CSV data
        const csvDataNoOverlap = csvData.filter(
          (d: any) => new Date(d.date).getTime() < apiFirstDate
        );
        combinedData = [...csvDataNoOverlap, ...apiData];
      } else {
        combinedData = [...csvData, ...apiData];
      }

      return NextResponse.json({
        success: true,
        data: combinedData
      });
    }

    // Return CSV data if API data is not available
    console.log('Using CSV data only');
    return NextResponse.json({
      success: true,
      data: csvData
    });

  } catch (error) {
    console.error('Error in historical data route:', error instanceof Error ? error.message : 'Unknown error');
    return NextResponse.json({
      success: false,
      error: 'Failed to fetch historical data'
    });
  }
}

export const dynamic = 'force-dynamic';