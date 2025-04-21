import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { spawn } from 'child_process';
import { join } from 'path';

// Define interfaces for QA history data
interface QAFeedback {
  rating: number;
  comment?: string;
}

interface QAAnalysis {
  primary_category?: string;
  secondary_categories?: string[];
  crypto_entities?: string[];
  intent?: string;
  temporality?: string;
  complexity?: string;
  question_type?: string;
}

interface QAHistoryItem {
  id: string;
  user_id: string;
  question: string;
  answer: string;
  timestamp: string;
  document_ids?: string[];
  primary_category?: string;
  secondary_categories?: string[];
  crypto_entities?: string[];
  intent?: string;
  session_id?: string;
  user_feedback?: string;
  feedback_rating?: number;
  duration_ms?: number;
}

export async function POST(request: NextRequest) {
  try {
    // Check authentication
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Parse request body
    const {
      question,
      answer,
      analysis,
      documentIds,
      sessionId,
      feedback,
      durationMs
    } = await request.json();
    
    if (!question || !answer) {
      return NextResponse.json(
        { error: 'Question and answer are required' },
        { status: 400 }
      );
    }

    // Store in Weaviate using Python script
    const success = await storeQAInteraction({
      userId,
      question,
      answer,
      analysis,
      documentIds,
      sessionId,
      feedback,
      durationMs
    });

    return NextResponse.json({
      success,
      message: success ? 'QA interaction stored successfully' : 'Failed to store QA interaction'
    });
    
  } catch (error) {
    console.error('Error storing QA interaction:', error);
    return NextResponse.json(
      { error: 'Failed to store QA interaction', details: String(error) },
      { status: 500 }
    );
  }
}

// Helper function to run Python script
async function storeQAInteraction(data: {
  userId: string;
  question: string;
  answer: string;
  analysis?: QAAnalysis;
  documentIds?: string[];
  sessionId?: string;
  feedback?: QAFeedback;
  durationMs?: number;
}): Promise<boolean> {
  return new Promise((resolve, reject) => {
    // Create script path
    const scriptPath = join(process.cwd(), 'Code/qa/store_qa_interaction.py');
    
    // Build arguments
    const args = [
      scriptPath,
      '--user_id', data.userId,
      '--question', data.question,
      '--answer', data.answer
    ];
    
    // Add optional arguments if provided
    if (data.sessionId) {
      args.push('--session_id', data.sessionId);
    }
    
    if (data.durationMs) {
      args.push('--duration_ms', data.durationMs.toString());
    }
    
    // For complex objects, pass as JSON strings
    if (data.analysis) {
      args.push('--analysis', JSON.stringify(data.analysis));
    }
    
    if (data.documentIds && data.documentIds.length > 0) {
      args.push('--document_ids', JSON.stringify(data.documentIds));
    }
    
    if (data.feedback) {
      args.push('--feedback', JSON.stringify(data.feedback));
    }
    
    console.log(`Executing: python ${args.join(' ')}`);
    
    // Spawn Python process
    const pythonProcess = spawn('python', args);
    
    let outputData = '';
    let errorData = '';
    
    // Collect output
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    // Collect errors
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error output: ${errorData}`);
        reject(new Error(`Failed to store QA interaction: ${errorData}`));
        return;
      }
      
      // Parse output to determine success
      try {
        if (outputData.includes('success')) {
          resolve(true);
        } else {
          console.warn("Script did not indicate success:", outputData);
          resolve(false);
        }
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        resolve(false);
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

export async function GET(request: NextRequest) {
  try {
    // Check authentication
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Get query parameters
    const searchParams = request.nextUrl.searchParams;
    const days = parseInt(searchParams.get('days') || '30');
    const category = searchParams.get('category') || '';
    const entity = searchParams.get('entity') || '';
    const sessionId = searchParams.get('sessionId') || '';
    const limit = parseInt(searchParams.get('limit') || '100');
    
    // Get QA history using Python script
    const history = await getQAHistory({
      userId,
      days,
      category,
      entity,
      sessionId,
      limit
    });

    return NextResponse.json({
      success: true,
      history
    });
    
  } catch (error) {
    console.error('Error retrieving QA history:', error);
    return NextResponse.json(
      { error: 'Failed to retrieve QA history', details: String(error) },
      { status: 500 }
    );
  }
}

// Helper function to run Python script for history retrieval
async function getQAHistory(params: {
  userId: string;
  days?: number;
  startDate?: string;
  endDate?: string;
  category?: string;
  entity?: string;
  sessionId?: string;
  limit?: number;
}): Promise<QAHistoryItem[]> {
  return new Promise((resolve, reject) => {
    // Create script path
    const scriptPath = join(process.cwd(), 'Code/qa/get_qa_history.py');
    
    // Build arguments
    const args = [
      scriptPath,
      '--user_id', params.userId
    ];
    
    // Add optional arguments if provided
    if (params.days) {
      args.push('--days', params.days.toString());
    }
    
    if (params.startDate) {
      args.push('--start_date', params.startDate);
    }
    
    if (params.endDate) {
      args.push('--end_date', params.endDate);
    }
    
    if (params.category) {
      args.push('--category', params.category);
    }
    
    if (params.entity) {
      args.push('--entity', params.entity);
    }
    
    if (params.sessionId) {
      args.push('--session_id', params.sessionId);
    }
    
    if (params.limit) {
      args.push('--limit', params.limit.toString());
    }
    
    console.log(`Executing: python ${args.join(' ')}`);
    
    // Spawn Python process
    const pythonProcess = spawn('python', args);
    
    let outputData = '';
    let errorData = '';
    
    // Collect output
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    // Collect errors
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error output: ${errorData}`);
        reject(new Error(`Failed to retrieve QA history: ${errorData}`));
        return;
      }
      
      // Parse output
      try {
        // Clean the output - remove any lines that don't look like JSON
        const cleanedOutput = outputData.split('\n')
          .filter(line => line.trim().startsWith('{') || line.trim().startsWith('['))
          .join('');
          
        if (!cleanedOutput) {
          console.warn("No valid JSON found in output");
          resolve([]);
          return;
        }
        
        const result = JSON.parse(cleanedOutput);
        
        if (result.success) {
          resolve(result.interactions || []);
        } else {
          console.warn("Script reported failure:", result.error);
          resolve([]);
        }
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        resolve([]);
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';