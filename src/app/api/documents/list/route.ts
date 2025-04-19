import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { spawn } from 'child_process';
import { join } from 'path';

/**
 * Retrieves the list of documents for the authenticated user
 */
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
    const limit = parseInt(searchParams.get('limit') || '20');
    const offset = parseInt(searchParams.get('offset') || '0');
    const sortBy = searchParams.get('sortBy') || 'uploadDate';
    const sortOrder = searchParams.get('sortOrder') || 'desc';
    const status = searchParams.get('status') || '';

    // Call Python script to retrieve documents from Weaviate
    const documents = await getUserDocuments(userId, limit, offset, sortBy, sortOrder, status);

    return NextResponse.json({
      success: true,
      documents,
      pagination: {
        limit,
        offset,
        total: documents.length // In a real app, you should return the total count separate from the limited results
      }
    });
    
  } catch (error) {
    console.error('Error fetching user documents:', error);
    return NextResponse.json(
      { error: 'Failed to fetch documents' },
      { status: 500 }
    );
  }
}
interface UserDocument {
  id: string;
  title: string;
  filePath: string;
  uploadDate: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  isPublic: boolean;
  notes: string;
  userId: string;
  documentId: string;
}



/**
 * Helper function to call the Python script that retrieves documents from Weaviate
 */
async function getUserDocuments(userId: string, limit: number, offset: number, sortBy: string, sortOrder: string, status: string): Promise<UserDocument[]> {
  return new Promise((resolve, reject) => {
    // Path to the Python script
    const scriptPath = join(process.cwd(), 'Code/document_processing/get_user_documents.py');
    
    // Prepare arguments
    const args = [
      scriptPath,
      '--user_id', userId,
      '--limit', limit.toString(),
      '--offset', offset.toString(),
      '--sort_by', sortBy,
      '--sort_order', sortOrder
    ];
    
    // Add status filter if provided
    if (status) {
      args.push('--status', status);
    }
    
    // Spawn the Python process
    const pythonProcess = spawn('python', args);
    
    let outputData = '';
    let errorData = '';
    
    // Collect data from stdout
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    // Collect data from stderr
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        console.error(`Error output: ${errorData}`);
        reject(new Error(`Failed to retrieve documents: ${errorData}`));
        return;
      }
      
      try {
        // Parse the JSON output from the Python script
        const documents = JSON.parse(outputData);
        resolve(documents);
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        reject(new Error('Failed to parse document data'));
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}