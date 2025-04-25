import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { join } from 'path';

export async function GET(request: NextRequest) {
  try {
    // Get the symbol from URL query parameter
    const symbol = request.nextUrl.searchParams.get('symbol')?.toUpperCase() || 'BTC';
    const timeframe = request.nextUrl.searchParams.get('timeframe') || '30d';
    
    // Get the current project root directory
    const rootDir = process.cwd();
    
    console.log(`Fetching crypto data for symbol: ${symbol} with timeframe: ${timeframe}`);
    
    // Call Python script to get data from StorageManager
    try {
      const scriptResult = await runPythonScript(symbol, rootDir);
      
      if (scriptResult.success) {
        return NextResponse.json(scriptResult);
      } else {
        return NextResponse.json({
          success: false,
          error: scriptResult.error || 'No data available',
          prices: scriptResult.prices || []
        });
      }
    } catch (error) {
      console.error('Error retrieving crypto data:', error);
      return NextResponse.json({
        success: false, 
        error: 'Failed to retrieve cryptocurrency data'
      }, { status: 500 });
    }
    
  } catch (error) {
    console.error('Error in forecast API:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error', details: String(error) },
      { status: 500 }
    );
  }
}

function runPythonScript(symbol: string, cwd: string): Promise<any> {
  return new Promise((resolve, reject) => {
    // Path to the Python script that accesses StorageManager
    const scriptPath = join(cwd, 'Code/get_crypto_data.py');
    
    // Create the process with the symbol as an argument
    const pythonProcess = spawn('python', [scriptPath, '--symbol', symbol]);
    
    let output = '';
    let errorOutput = '';
    
    // Collect standard output
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    // Collect error output
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.log('Python stderr:', data.toString());
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}: ${errorOutput}`);
        // Even with an error, try to parse any output that might exist
        try {
          if (output.trim()) {
            const result = JSON.parse(output);
            resolve(result);
            return;
          }
        } catch (e) {
          console.error('Failed to parse partial output:', e);
        }
        
        resolve({ 
          success: false, 
          error: 'Failed to retrieve data',
          errorDetails: errorOutput
        });
        return;
      }
      
      try {
        // Parse the JSON output from the Python script
        const result = JSON.parse(output);
        resolve(result);
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        reject(error);
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

export const dynamic = 'force-dynamic';