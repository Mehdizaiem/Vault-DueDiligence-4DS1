import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { spawn } from 'child_process';
import { join } from 'path';

// Define a proper interface for the user document
interface UserDocument {
  id: string;
  title: string;
  source: string;
  document_type: string;
  user_id: string;
  upload_date: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  is_public: boolean;
  file_size?: number;
  file_type?: string;
  notes?: string;
  content_preview?: string;
  crypto_entities?: string[];
  risk_factors?: string[];
  word_count?: number;
  sentence_count?: number;
  org_entities?: string[];
  person_entities?: string[];
  location_entities?: string[];
  risk_score?: number;
  date?: string;
}

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
    const sortBy = searchParams.get('sortBy') || 'upload_date';
    const sortOrder = searchParams.get('sortOrder') || 'desc';
    const status = searchParams.get('status') || '';

    // First, ensure the schema exists by running the schema creation script
    try {
      await runSchemaSetup();
      console.log("Schema setup completed");
    } catch (schemaError) {
      console.error("Warning: Schema setup failed:", schemaError);
      // Continue anyway as the schema might already exist
    }

    try {
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
      console.error('Error retrieving documents:', error);
      return NextResponse.json({
        success: false,
        error: 'Failed to retrieve documents',
        documents: [] // Return empty array as fallback
      });
    }
    
  } catch (error) {
    console.error('Error fetching user documents:', error);
    return NextResponse.json(
      { error: 'Failed to fetch documents', details: String(error) },
      { status: 500 }
    );
  }
}

/**
 * Helper function to ensure the schema is set up
 */
async function runSchemaSetup(): Promise<void> {
  return new Promise((resolve, reject) => {
    // Path to the schema setup script
    const scriptPath = join(process.cwd(), 'Sample_Data/vector_store/create_schemas.py');
    
    // Spawn the Python process
    const pythonProcess = spawn('python', [scriptPath]);
    
    let errorData = '';
    
    // Collect data from stderr
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Schema setup process exited with code ${code}`);
        console.error(`Error output: ${errorData}`);
        reject(new Error(`Failed to set up schema: ${errorData}`));
        return;
      }
      
      resolve();
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
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
    
    console.log(`Executing: python ${args.join(' ')}`);
    
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
        if (!outputData.trim()) {
          console.warn("Warning: Empty output from document retrieval script");
          resolve([]);
          return;
        }
        
        // Log the output for debugging
        console.log(`Script output length: ${outputData.length} bytes`);
        
        // Clean the output - remove any lines that don't look like JSON
        const cleanedOutput = outputData.split('\n')
          .filter(line => line.trim().startsWith('[') || line.trim().startsWith('{'))
          .join('');
          
        if (!cleanedOutput) {
          console.warn("No valid JSON found in output");
          console.warn(`Raw output: ${outputData}`);
          resolve([]);
          return;
        }
        
        const documents = JSON.parse(cleanedOutput) as UserDocument[];
        resolve(documents);
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        console.error('Output that failed to parse:', outputData);
        reject(new Error('Failed to parse document data'));
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}