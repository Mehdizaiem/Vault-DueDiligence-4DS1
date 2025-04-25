import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { auth } from '@clerk/nextjs';

// Maximum time to wait for a response (in milliseconds)
const TIMEOUT = 120000; // 2 minutes to allow for document retrieval

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
    console.log(`Document ID: ${documentId || 'no document ID provided'}`);
    console.log(`User ID: ${userId || 'no user ID'}`);
    
    // Run the Python script with enhanced logging
    const answer = await runPythonScript(question, rootDir, userId || undefined, documentId);
    
    // Check if the answer indicates an issue with document retrieval
    if (answer.includes("document not found") || answer.includes("no document information available")) {
      console.error("Document retrieval error detected in answer");
      // You could add additional error handling here
    }
    
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
      args.push('--user-id', userId);
    }
    
    // Add document ID if available
    if (documentId) {
      args.push('--document-id', documentId);
      console.log(`Including document ID in Python script call: ${documentId}`);
    }
    
    console.log(`Running Python command: python ${args.join(' ')}`);
    
    // Create the process
    const pythonProcess = spawn('python', args, { cwd });
    
    let output = '';
    let errorOutput = '';
    
    // Collect standard output
    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      // Log chunks to help debug
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
      
      // Extract the answer from the output
      try {
        const answerPattern = /Answer:\s*([\s\S]*?)(?:$)/i;
        const match = output.match(answerPattern);
        
        if (match && match[1]) {
          resolve(match[1].trim());
        } else {
          // If no clear answer format is found, return all output
          console.log("No answer pattern found, returning raw output");
          resolve(output.trim() || "Sorry, I couldn't find an answer to that question.");
        }
      } catch (err) {
        console.error("Error parsing Python output:", err);
        reject(err);
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      console.error("Python process error:", err);
      reject(err);
    });
    
    // Set a timeout
    const timeout = setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('Process timed out after ' + (TIMEOUT/1000) + ' seconds'));
    }, TIMEOUT);
    
    // Clear the timeout when the process exits
    pythonProcess.on('close', () => {
      clearTimeout(timeout);
    });
  });
}