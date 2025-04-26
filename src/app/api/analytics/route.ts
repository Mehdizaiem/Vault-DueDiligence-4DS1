// File: app/api/analytics/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

// Maximum time to wait for a response (in milliseconds)
const TIMEOUT = 60000; // 1 minute

export async function GET(request: NextRequest) {
  try {
    // Get the root directory of the project
    const rootDir = process.cwd();

    console.log('Fetching analytics data');

    // Run the Python script to fetch analytics data
    const data = await runPythonScript(rootDir);

    // Try to extract and parse the JSON from the potentially mixed output
    try {
      // First, try to parse as-is (in case it's clean)
      const parsedData = JSON.parse(data);
      return NextResponse.json(parsedData);
    } catch (error) {
      console.error('Error parsing raw JSON from Python script:', error);
      console.error('Python output was:', data);
      
      // Try to extract JSON from the mixed output
      const extractedJson = extractJsonFromOutput(data);
      if (extractedJson) {
        try {
          const parsedExtractedData = JSON.parse(extractedJson);
          console.log('Successfully extracted and parsed JSON from mixed output');
          return NextResponse.json(parsedExtractedData);
        } catch (jsonError) {
          console.error('Error parsing extracted JSON:', jsonError);
        }
      }
      
      // If all parsing attempts fail, return a fallback response
      return NextResponse.json(
        { 
          error: 'Failed to parse analytics data',
          kpis: {},
          asset_distribution: [],
          portfolio_performance: [],
          portfolios: []
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
        kpis: {},
        asset_distribution: [],
        portfolio_performance: [],
        portfolios: []
      },
      { status: 500 }
    );
  }
}

function runPythonScript(cwd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    // Update the path to the Python script
    const args = [path.join(cwd, 'analytics', 'analytics_data.py')];

    console.log(`Running Python command: python ${args.join(' ')}`);

    // Create the process
    const pythonProcess = spawn('python', args, { cwd });

    let output = '';
    let errorOutput = '';

    // Collect standard output
    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      console.log(`Python stdout chunk: ${chunk.length} chars`);
    });

    // Collect error output
    pythonProcess.stderr.on('data', (data) => {
      const chunk = data.toString();
      errorOutput += chunk;
      console.error(`Python stderr: ${chunk}`);
    });

    // Handle process completion
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);

      if (code !== 0 && !output) {
        reject(new Error(`Process exited with code ${code}: ${errorOutput}`));
        return;
      }

      // Return the raw output - we'll handle parsing/extraction later
      resolve(output.trim());
    });

    // Handle process errors
    pythonProcess.on('error', (err) => {
      console.error("Python process error:", err);
      reject(err);
    });

    // Set a timeout
    const timeout = setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Process timed out after ' + (TIMEOUT / 1000) + ' seconds'));
    }, TIMEOUT);

    // Clear the timeout when the process exits
    pythonProcess.on('close', () => {
      clearTimeout(timeout);
    });
  });
}

/**
 * Extracts JSON data from mixed output that might contain log messages
 * and other non-JSON content.
 */
function extractJsonFromOutput(output: string): string | null {
  // First look for complete JSON objects
  const jsonRegex = /\{[\s\S]*\}/;
  const match = output.match(jsonRegex);
  
  if (match && match[0]) {
    try {
      // Verify it's valid JSON
      JSON.parse(match[0]);
      return match[0];
    } catch (e) {
      // If not valid, try a more precise approach
    }
  }
  
  // Try to find the last occurrence of a proper JSON object
  // This handles cases where there might be multiple prints before the JSON
  const lastOpenBrace = output.lastIndexOf('{');
  const lastCloseBrace = output.lastIndexOf('}');
  
  if (lastOpenBrace !== -1 && lastCloseBrace !== -1 && lastCloseBrace > lastOpenBrace) {
    const potentialJson = output.substring(lastOpenBrace, lastCloseBrace + 1);
    try {
      // Verify it's valid JSON
      JSON.parse(potentialJson);
      return potentialJson;
    } catch (e) {
      // Not valid JSON
    }
  }
  
  return null;
}