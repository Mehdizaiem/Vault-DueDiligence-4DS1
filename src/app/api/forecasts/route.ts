import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';
import fs from 'fs';

export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    let symbol = searchParams.get('symbol') || 'BTC_USD';
    
    // Normalize symbol format
    symbol = symbol.toUpperCase()
      .replace(/[^A-Z]/g, '') // Remove all non-letters
      .replace(/^(BTC|ETH)/, '$1_') // Add underscore after BTC or ETH
      .replace(/USD$/, '_USD'); // Ensure _USD suffix
    
    if (!symbol.includes('_USD')) {
      symbol = `${symbol}_USD`;
    }

    console.log(`[API] Fetching forecast data for ${symbol}`);

    // Try different paths for the forecast_manager.py script
    const possiblePaths = [
      join(process.cwd(), 'Sample_Data', 'vector_store', 'forecast_manager.py'),
      join(process.cwd(), 'Sample_Data/vector_store/forecast_manager.py'),
      join(process.cwd(), 'sample_data', 'vector_store', 'forecast_manager.py'),
      join(process.cwd(), 'data', 'vector_store', 'forecast_manager.py')
    ];

    let scriptPath = '';
    for (const path of possiblePaths) {
      if (fs.existsSync(path)) {
        scriptPath = path;
        console.log(`[API] Found forecast manager script at: ${path}`);
        break;
      }
    }

    if (!scriptPath) {
      console.error('[API] Could not find forecast_manager.py script in any of the expected locations');
      return NextResponse.json({
        success: false,
        error: 'Forecast manager script not found',
        paths: possiblePaths
      }, { status: 500 });
    }

    return new Promise((resolve) => {
      const args = [
        scriptPath,
        '--action', 'get_latest_forecast',
        '--symbol', symbol
      ];

      console.log(`[API] Executing: python ${args.join(' ')}`);

      const pythonProcess = spawn('python', args);
      let dataString = '';
      let errorString = '';

      pythonProcess.stdout.on('data', (data) => {
        const stdout = data.toString();
        console.log('[Python stdout]:', stdout);
        dataString += stdout;
      });

      pythonProcess.stderr.on('data', (data) => {
        const stderr = data.toString();
        // Only log actual errors, not info messages
        if (!stderr.startsWith('INFO - ')) {
          errorString += stderr;
          console.error('[Python stderr]:', stderr);
        } else {
          console.log('[Python info]:', stderr);
        }
      });

      pythonProcess.on('close', (code) => {
        console.log(`[API] Python process exited with code ${code}`);
        
        if (code !== 0) {
          console.error('[API] Python script execution failed');
          resolve(NextResponse.json({
            success: false,
            error: 'Failed to execute forecast script',
            details: errorString || 'Unknown error',
            exitCode: code
          }, { status: 500 }));
          return;
        }

        try {
          // Try to find valid JSON in the output
          const jsonMatches = dataString.match(/\{[\s\S]*\}/g);
          let forecast = null;
          
          if (jsonMatches && jsonMatches.length > 0) {
            // Take the last matching JSON object
            const jsonStr = jsonMatches[jsonMatches.length - 1];
            forecast = JSON.parse(jsonStr);
            
            // Check if it's an error response from the Python script
            if (forecast.error) {
              resolve(NextResponse.json({
                success: false,
                error: forecast.error
              }, { status: 400 }));
              return;
            }
            
            // Convert forecast dates from null to actual dates if needed
            if (forecast.forecast_dates && forecast.forecast_dates.some((d: null | string) => d === null)) {
              const startDate = new Date(forecast.forecast_timestamp);
              forecast.forecast_dates = Array.from({ length: forecast.days_ahead }, (_, i) => {
                const date = new Date(startDate);
                date.setDate(date.getDate() + i);
                return date.toISOString();
              });
            }
            
            resolve(NextResponse.json({
              success: true,
              data: forecast
            }));
            return;
          }

          // If no JSON found in output
          console.error('[API] No valid JSON found in output');
          resolve(NextResponse.json({
            success: false,
            error: 'No valid forecast data found',
            rawOutput: dataString
          }, { status: 500 }));
        } catch (error) {
          console.error('[API] Error parsing forecast data:', error);
          console.error('[API] Raw output that failed to parse:', dataString);
          
          resolve(NextResponse.json({
            success: false,
            error: 'Failed to parse forecast data',
            details: error instanceof Error ? error.message : String(error)
          }, { status: 500 }));
        }
      });

      pythonProcess.on('error', (err) => {
        console.error('[API] Python process error:', err);
        resolve(NextResponse.json({
          success: false,
          error: 'Error executing forecast script',
          details: err.message
        }, { status: 500 }));
      });
    });
  } catch (error: any) {
    console.error('[API] API route error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error',
      details: error.message
    }, { status: 500 });
  }
}

export const config = {
  runtime: 'nodejs',
  unstable_allowDynamic: [
    '**/node_modules/**',
  ],
};

export const dynamic = 'force-dynamic';