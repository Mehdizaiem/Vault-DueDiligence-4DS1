// app/api/analytics/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// Maximum time to wait for a response (in milliseconds)
const TIMEOUT = 90000; // 1.5 minutes - increase timeout for large data
const PYTHON_SCRIPT_PATH = path.join(process.cwd(), 'analytics', 'analytics_data.py');

// Interface for the expected JSON structure
interface AnalyticsData {
  kpis: {
    market_cap?: { value: string; change: string; trend: string };
    asset_count?: { value: string; change: string; trend: string };
    price_change?: { value: string; change: string; trend: string };
    market_sentiment?: { value: string; change: string; trend: string };
  };
  asset_distribution: Array<{ name: string; value: number; color: string }>;
  portfolio_performance: Array<{ symbol: string; timestamp: string; change_pct: number }>;
  recent_news: Array<{
    title: string;
    source: string;
    date: string;
    url: string;
    sentiment_score: number;
    sentiment_color: string;
    related_assets?: string[];
  }>;
  due_diligence: Array<{
    title: string;
    document_type: string;
    source: string;
    icon: string;
    keywords?: string[];
  }>;
}

export async function GET(request: NextRequest) {
  try {
    console.log('Fetching analytics data');
    
    // Extract filters from query parameters
    const searchParams = request.nextUrl.searchParams;
    const dateRange = searchParams.get('dateRange') || '30d';
    const symbols = searchParams.getAll('symbols');
    
    const filters = {
      dateRange,
      symbols,
    };

    console.log('Filters:', filters);

    // Run the Python script to fetch analytics data
    const rawOutput = await runPythonScript(filters);
    
    // Try to parse the output
    try {
      const data: AnalyticsData = parseOutputSafely(rawOutput);
      
      // Validate data structure
      validateDataStructure(data);
      
      return NextResponse.json(data);
    } catch (error) {
      console.error('Error parsing analytics data:', error);
      console.error('Raw output snippet:', rawOutput.substring(0, 500) + '...');
      
      return NextResponse.json(
        { 
          error: 'Failed to parse analytics data: ' + (error instanceof Error ? error.message : String(error)),
          kpis: createFallbackKpis(),
          asset_distribution: createFallbackAssetDistribution(),
          portfolio_performance: [],
          recent_news: [],
          due_diligence: []
        },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    return NextResponse.json(
      { 
        error: 'Failed to fetch analytics data: ' + String(error),
        kpis: createFallbackKpis(),
        asset_distribution: createFallbackAssetDistribution(),
        portfolio_performance: [],
        recent_news: [],
        due_diligence: []
      },
      { status: 500 }
    );
  }
}

function validateDataStructure(data: AnalyticsData): void {
  // Ensure all required properties exist
  if (!data.kpis) data.kpis = {};
  if (!data.asset_distribution) data.asset_distribution = [];
  if (!data.portfolio_performance) data.portfolio_performance = [];
  if (!data.recent_news) data.recent_news = [];
  if (!data.due_diligence) data.due_diligence = [];
  
  // Validate portfolio_performance data
  data.portfolio_performance = data.portfolio_performance.filter(item => {
    // Ensure valid symbol (not empty)
    if (!item.symbol) return false;
    
    // Ensure valid timestamp
    try {
      new Date(item.timestamp);
    } catch {
      return false;
    }
    
    // Ensure change_pct is a number
    if (typeof item.change_pct !== 'number') return false;
    
    return true;
  });
}

function runPythonScript(filters: any): Promise<string> {
  return new Promise((resolve, reject) => {
    let pythonProcess: ChildProcess | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    // Define cleanup function
    const cleanup = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      if (pythonProcess) {
        pythonProcess.kill('SIGTERM');
        pythonProcess = null;
      }
    };

    try {
      console.log(`Running Python script: ${PYTHON_SCRIPT_PATH}`);
      console.log(`With filters: ${JSON.stringify(filters)}`);

      pythonProcess = spawn('python', [
        PYTHON_SCRIPT_PATH,
        '--filters',
        JSON.stringify(filters)
      ], {
        stdio: ['ignore', 'pipe', 'pipe'],
        cwd: process.cwd(),
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout?.on('data', (data: Buffer) => {
        const chunk = data.toString();
        output += chunk;
        console.log(`Python stdout chunk: ${chunk.length} chars`);
      });

      pythonProcess.stderr?.on('data', (data: Buffer) => {
        const chunk = data.toString();
        errorOutput += chunk;
        console.error(`Python stderr: ${chunk}`);
      });

      pythonProcess.on('close', (code: number | null) => {
        cleanup();
        
        if (code !== 0) {
          reject(new Error(`Process exited with code ${code}: ${errorOutput || 'Unknown error'}`));
          return;
        }

        if (!output) {
          reject(new Error('No output received from Python script'));
          return;
        }

        resolve(output.trim());
      });

      pythonProcess.on('error', (err: Error) => {
        cleanup();
        console.error('Python process error:', err);
        reject(err);
      });

      timeoutId = setTimeout(() => {
        if (pythonProcess) {
          cleanup();
          reject(new Error(`Process timed out after ${TIMEOUT / 1000} seconds`));
        }
      }, TIMEOUT);
    } catch (err) {
      cleanup();
      reject(err);
    }
  });
}

function parseOutputSafely(output: string): AnalyticsData {
  // Try direct JSON parse first
  try {
    const data = JSON.parse(output);
    return normalizeKpiTrends(data);
  } catch (e) {
    console.log('Direct JSON parse failed, attempting to extract JSON');
  }

  // Try to extract JSON from mixed output
  const jsonStart = output.indexOf('{');
  const jsonEnd = output.lastIndexOf('}');

  if (jsonStart === -1 || jsonEnd === -1 || jsonEnd < jsonStart) {
    throw new Error('No valid JSON found in output');
  }

  const jsonString = output.slice(jsonStart, jsonEnd + 1);

  try {
    const data = JSON.parse(jsonString);
    return normalizeKpiTrends(data);
  } catch (error) {
    // Try to clean up common JSON issues
    try {
      // Replace single quotes with double quotes
      let cleanedJson = jsonString
        .replace(/'/g, '"')
        // Add quotes around property names
        .replace(/([{,]\s*)(\w+)(?=\s*:)/g, '$1"$2"')
        // Remove trailing commas in objects and arrays
        .replace(/,\s*}/g, '}')
        .replace(/,\s*\]/g, ']');
      
      // Handle special cases
      cleanedJson = cleanedJson
        // Fix invalid boolean values
        .replace(/":\s*True/g, '": true')
        .replace(/":\s*False/g, '": false')
        // Fix invalid None value
        .replace(/":\s*None/g, '": null');
      
      const data = JSON.parse(cleanedJson);
      return normalizeKpiTrends(data);
    } catch (innerError) {
      console.error('JSON parsing failed:', innerError);
      console.error('Cleaned JSON snippet:', jsonString.substring(0, 500) + '...');
      throw new Error('Failed to parse JSON output');
    }
  }
}

// Helper function to normalize KPI trend values
function normalizeKpiTrends(data: AnalyticsData): AnalyticsData {
  if (data.kpis) {
    // Normalize trend values for all KPIs
    if (data.kpis.market_cap) {
      data.kpis.market_cap.trend = normalizeTrend(data.kpis.market_cap.trend);
    }
    
    if (data.kpis.asset_count) {
      data.kpis.asset_count.trend = normalizeTrend(data.kpis.asset_count.trend);
    }
    
    if (data.kpis.price_change) {
      data.kpis.price_change.trend = normalizeTrend(data.kpis.price_change.trend);
    }
    
    if (data.kpis.market_sentiment) {
      data.kpis.market_sentiment.trend = normalizeTrend(data.kpis.market_sentiment.trend);
    }
  }
  
  return data;
}

// Helper to normalize a single trend value
function normalizeTrend(trend: any): 'up' | 'down' | 'neutral' {
  if (trend === 'up' || trend === 'down' || trend === 'neutral') {
    return trend;
  }
  // Default to neutral for invalid values
  return 'neutral';
}

function createFallbackKpis() {
  return {
    market_cap: {
      value: "$2,045,000M",
      change: "+2.3%",
      trend: "up"
    },
    asset_count: {
      value: "5",
      change: "+0",
      trend: "neutral"
    },
    price_change: {
      value: "1.2%",
      change: "+1.2%",
      trend: "up"
    },
    market_sentiment: {
      value: "65.4%",
      change: "+0.8%",
      trend: "up"
    }
  };
}

function createFallbackAssetDistribution() {
  return [
    { name: "BTC", value: 60.5, color: "bg-orange-500" },
    { name: "ETH", value: 15.2, color: "bg-indigo-500" },
    { name: "BNB", value: 3.8, color: "bg-green-500" },
    { name: "SOL", value: 2.5, color: "bg-purple-500" },
    { name: "ADA", value: 1.4, color: "bg-blue-500" },
    { name: "Other", value: 16.6, color: "bg-gray-500" }
  ];
}