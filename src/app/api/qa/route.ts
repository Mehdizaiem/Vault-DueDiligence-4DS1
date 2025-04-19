import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { auth } from '@clerk/nextjs';

// Maximum time to wait for a response (in milliseconds)
const TIMEOUT = 60000; // 1 minute

export async function POST(request: NextRequest) {
  try {
    const { question, documentId } = await request.json();
    
    if (!question) {
      return NextResponse.json(
        { error: 'Question is required' },
        { status: 400 }
      );
    }
    
    // Get the authenticated user ID from Clerk
    const { userId } = auth();
    
    // Get the root directory of the project
    const rootDir = process.cwd();
    
    console.log(`Processing question: ${question}`);
    
    const answer = await runPythonScript(question, rootDir, userId || undefined, documentId);
    
    return NextResponse.json({ answer });
  } catch (error) {
    console.error('Error processing Q&A request:', error);
    return NextResponse.json(
      { error: 'Failed to process question', details: String(error) },
      { status: 500 }
    );
  }
}

function runPythonScript(
  question: string, 
  cwd: string, 
  userId?: string, 
  documentId?: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    // Build command arguments
    const args = [
      path.join(cwd, 'crypto_qa.py'),
      '--question', 
      question
    ];
    
    // Add user ID if available
    if (userId) {
      args.push('--user_id', userId);
    }
    
    // Add document ID if available
    if (documentId) {
      args.push('--document_id', documentId);
    }
    
    // Create the process
    const pythonProcess = spawn('python', args, { cwd });
    
    let output = '';
    let errorOutput = '';
    
    // Collect standard output
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    // Collect error output
    pythonProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      // Don't reject here as some Python scripts use stderr for logging
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0 && !output) {
        reject(new Error(`Process exited with code ${code}: ${errorOutput}`));
        return;
      }
      
      // Extract the answer from the output
      try {
        const answerPattern = /Answer:\s*([\s\S]*?)(?:$)/i;
        const match = output.match(answerPattern);
        
        if (match && match[1]) {
          resolve(match[1].trim());
        } else {
          // If no clear answer format is found, return all output
          resolve(output.trim() || "Sorry, I couldn't find an answer to that question.");
        }
      } catch (err) {
        reject(err);
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
    
    // Set a timeout
    const timeout = setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Process timed out'));
    }, TIMEOUT);
    
    // Clear the timeout when the process exits
    pythonProcess.on('close', () => {
      clearTimeout(timeout);
    });
  });
}