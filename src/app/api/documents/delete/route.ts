import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { spawn } from 'child_process';
import { join } from 'path';
import fs from 'fs/promises';
import path from 'path';

/**
 * Deletes a document for the authenticated user
 */
export async function DELETE(request: NextRequest) {
  try {
    // Check authentication
    const { userId } = auth();
    
    if (!userId) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // Get document ID from query parameters
    const searchParams = request.nextUrl.searchParams;
    const documentId = searchParams.get('id');
    
    if (!documentId) {
      return NextResponse.json(
        { error: 'Document ID is required' },
        { status: 400 }
      );
    }

    // Delete the document using Python script
    const success = await deleteUserDocument(userId, documentId);

    // Also try to delete the physical file from disk
    try {
      const uploadsDir = join(process.cwd(), 'uploads', userId);
      
      // Try to find files with this document ID pattern
      const files = await fs.readdir(uploadsDir);
      for (const file of files) {
        if (file.includes(documentId)) {
          console.log(`Found file to delete: ${file}`);
          await fs.unlink(path.join(uploadsDir, file));
          console.log(`Deleted file: ${file}`);
        }
      }
    } catch (fileError) {
      console.error("Error deleting physical file:", fileError);
      // Continue since the database entry is the most important part
    }

    return NextResponse.json({
      success,
      message: success ? 'Document deleted successfully' : 'Failed to delete document'
    });
    
  } catch (error) {
    console.error('Error deleting document:', error);
    return NextResponse.json(
      { error: 'Failed to delete document', details: String(error) },
      { status: 500 }
    );
  }
}

/**
 * Helper function to call the Python script that deletes a document from Weaviate
 */
async function deleteUserDocument(userId: string, documentId: string): Promise<boolean> {
  return new Promise((resolve, reject) => {
    // Path to the Python script
    const scriptPath = join(process.cwd(), 'Code/document_processing/delete_user_document.py');
    
    // Prepare arguments
    const args = [
      scriptPath,
      '--user_id', userId,
      '--document_id', documentId
    ];
    
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
        reject(new Error(`Failed to delete document: ${errorData}`));
        return;
      }
      
      try {
        // Parse the JSON output from the Python script
        if (!outputData.trim()) {
          console.warn("Warning: Empty output from document deletion script");
          resolve(false);
          return;
        }
        
        console.log("Raw output:", outputData);
        
        // Clean the output - remove any lines that don't look like JSON
        const cleanedOutput = outputData.split('\n')
          .filter(line => line.trim().startsWith('{') || line.trim().startsWith('['))
          .join('');
          
        if (!cleanedOutput) {
          console.warn("No valid JSON found in output");
          resolve(false);
          return;
        }
        
        const result = JSON.parse(cleanedOutput);
        resolve(result.success === true);
      } catch (error) {
        console.error('Error parsing Python script output:', error);
        console.error('Output that failed to parse:', outputData);
        resolve(false); // Assume failure if we can't parse the output
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}