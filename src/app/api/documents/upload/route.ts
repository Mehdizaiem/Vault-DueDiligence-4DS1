import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { join } from 'path';
import { mkdir, writeFile, stat } from 'fs/promises';
import { exec } from 'child_process';
import { promisify } from 'util';

const execPromise = promisify(exec);

// Set up temp upload directory
const UPLOAD_DIR = join(process.cwd(), 'uploads');

// Create a safe filename to prevent path traversal attacks
function createSafeFilename(originalFilename: string, userId: string): string {
  const timestamp = Date.now();
  // Replace potentially dangerous characters
  const cleanName = originalFilename.replace(/[^a-zA-Z0-9.-]/g, '_');
  // Add user ID and timestamp to ensure uniqueness
  return `${userId}_${timestamp}_${cleanName}`;
}

/**
 * API handler for document uploads
 */
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
    
    // Parse the form data
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }
    
    // Get additional metadata
    const notes = formData.get('notes') as string || '';
    const isPublic = formData.get('isPublic') === 'true';
    
    // Log file details for debugging
    console.log(`Received file upload: ${file.name}, size: ${file.size}, type: ${file.type}`);
    
    // Check file type
    const fileType = file.type;
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    
    if (!allowedTypes.includes(fileType)) {
      return NextResponse.json(
        { error: 'Invalid file type. Only PDF, DOCX, and TXT files are allowed.' },
        { status: 400 }
      );
    }
    
    // Check file size (limit to 10MB)
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: 'File size exceeds the 10MB limit.' },
        { status: 400 }
      );
    }
    
    // Check if file is empty
    if (file.size === 0) {
      return NextResponse.json(
        { error: 'File appears to be empty.' },
        { status: 400 }
      );
    }
    
    // Create the upload directory if it doesn't exist
    const userUploadDir = join(UPLOAD_DIR, userId);
    await mkdir(userUploadDir, { recursive: true });
    
    // Create a safe filename
    const safeFilename = createSafeFilename(file.name, userId);
    const filePath = join(userUploadDir, safeFilename);
    
    try {
      // Get file data as ArrayBuffer and convert to Uint8Array
      const arrayBuffer = await file.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      
      console.log(`Writing file to ${filePath} (${uint8Array.length} bytes)`);
      
      // Write file to disk
      await writeFile(filePath, uint8Array);
      
      // Check if file was written correctly
      const fileStats = await stat(filePath);
      if (fileStats.size === 0) {
        throw new Error("File was written but is empty");
      }
      
      console.log(`File successfully written: ${filePath}, size: ${fileStats.size} bytes`);
    } catch (fileError) {
      console.error("Error writing file to disk:", fileError);
      return NextResponse.json(
        { error: 'Failed to save file to disk' },
        { status: 500 }
      );
    }

    // Create a unique document ID
    const documentId = `${userId}_${Date.now()}`;
    
    // Process the document asynchronously
    try {
      // First run schema setup to ensure collections exist
      try {
        await execPromise(`python ${join(process.cwd(), 'Sample_Data/vector_store/create_schemas.py')}`);
        console.log("Schema creation script executed");
      } catch (schemaErr) {
        console.error("Warning: Schema creation failed:", schemaErr);
        // Continue anyway, schema might already exist
      }
      
      // Process the document
      const cmd = `python ${join(process.cwd(), 'Code/document_processing/process_user_document.py')} --file "${filePath}" --user_id "${userId}" --document_id "${documentId}" --is_public ${isPublic} --notes "${notes}"`;
      
      console.log(`Executing document processing: ${cmd}`);
      
      // Execute the command but don't wait for it to complete
      execPromise(cmd).catch(err => {
        console.error(`Error processing document: ${err.message}`);
      });
      
      console.log("Document processing started");
    } catch (processErr) {
      console.error("Failed to start document processing:", processErr);
      // Continue since the file upload succeeded
    }
    
    return NextResponse.json({
      success: true,
      message: 'File uploaded successfully. Processing started.',
      documentId,
      fileName: file.name,
      filePath: safeFilename,
      fileSize: file.size,
      processingStatus: 'pending',
      uploadDate: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error('Error uploading file:', error);
    return NextResponse.json(
      { error: 'Failed to upload file', details: String(error) },
      { status: 500 }
    );
  }
}

// Use modern route config
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';