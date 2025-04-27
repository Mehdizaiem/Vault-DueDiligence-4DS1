// File: app/api/analytics/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';

// Maximum time to wait for a response (in milliseconds)
const TIMEOUT = 60000; // 1 minute
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

    // Run the Python script to fetch analytics data
    const rawOutput = await runPythonScript();
    
    // Try to parse the output
    try {
      const data: AnalyticsData = parseOutputSafely(rawOutput);
      return NextResponse.json(data);
    } catch (error) {
      console.error('Error parsing analytics data:', error);
      console.error('Raw output was:', rawOutput);
      
      return NextResponse.json(
        { 
          error: 'Failed to parse analytics data',
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
        error: 'Failed to fetch analytics data', 
        details: String(error),
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

function runPythonScript(): Promise<string> {
  return new Promise((resolve, reject) => {
    let pythonProcess: ChildProcess | null = null;
    let timeoutId: NodeJS.Timeout | null = null;

    // Define cleanup function at the top of the function scope
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

      pythonProcess = spawn('python', [PYTHON_SCRIPT_PATH], {
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

/**
 * Parse the output from the Python script more robustly
 */
function parseOutputSafely(output: string): AnalyticsData {
  // Try direct JSON parse first
  try {
    return JSON.parse(output);
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
    return JSON.parse(jsonString);
  } catch (error) {
    // Try to clean up common JSON issues
    try {
      const cleanedJson = jsonString
        .replace(/'/g, '"')
        .replace(/([{,]\s*)(\w+)(?=\s*:)/g, '$1"$2"')
        .replace(/,\s*}/g, '}');
      
      return JSON.parse(cleanedJson);
    } catch (innerError) {
      console.error('JSON parsing failed:', innerError);
      throw new Error('Failed to parse JSON output');
    }
  }
}

/**
 * Creates fallback KPI data when real data can't be loaded
 */
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

/**
 * Creates fallback asset distribution data when real data can't be loaded
 */
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