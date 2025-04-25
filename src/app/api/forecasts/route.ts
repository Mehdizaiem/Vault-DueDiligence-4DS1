import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';

interface ForecastResult {
  id: string;
  symbol: string;
  forecast_timestamp: string;
  model_name: string;
  model_type: string;
  days_ahead: number;
  current_price: number;
  forecast_dates: string[];
  forecast_values: number[];
  lower_bounds: number[];
  upper_bounds: number[];
  final_forecast: number;
  change_pct: number;
  trend: string;
  probability_increase: number;
  average_uncertainty: number;
  insight: string;
  error?: string;
}

export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    const symbol = searchParams.get('symbol') || 'BTC_USD';

    return new Promise((resolve) => {
      const pythonProcess = spawn('python', [
        join(process.cwd(), 'Sample_Data/vector_store/forecast_manager.py'),
        '--action',
        'get_latest_forecast',
        '--symbol',
        symbol
      ]);

      let dataString = '';

      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });

      pythonProcess.on('close', () => {
        try {
          // Split by lines and find the last line that looks like JSON
          const lines = dataString.split('\n');
          const jsonLine = lines
            .map(line => line.trim())
            .filter(line => line.startsWith('{') && line.endsWith('}'))
            .pop();

          if (!jsonLine) {
            console.error('No valid JSON found in output:', dataString);
            resolve(NextResponse.json({
              success: false,
              error: 'No forecast data found'
            }));
            return;
          }

          const forecast = JSON.parse(jsonLine);
          
          if (forecast.error) {
            resolve(NextResponse.json({
              success: false,
              error: forecast.error
            }));
            return;
          }

          resolve(NextResponse.json({
            success: true,
            data: [forecast] // Wrap in array since frontend expects array
          }));
        } catch (error) {
          console.error('Error parsing forecast data:', error);
          console.error('Raw output:', dataString);
          resolve(NextResponse.json({
            success: false,
            error: 'Failed to parse forecast data'
          }, { status: 500 }));
        }
      });
    });
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error'
    }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';