import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs";
import { join } from 'path';
import { mkdir, writeFile } from 'fs/promises';
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
    
    // Create the upload directory if it doesn't exist
    const userUploadDir = join(UPLOAD_DIR, userId);
    await mkdir(userUploadDir, { recursive: true });
    
    // Create a safe filename
    const safeFilename = createSafeFilename(file.name, userId);
    const filePath = join(userUploadDir, safeFilename);
    
    const arrayBuffer = await file.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    await writeFile(filePath, uint8Array);

    // Record upload in database
    // In a production app, you would likely use Prisma or another ORM here

    // Trigger document processing asynchronously in the background
    // This is a simplified approach - in production you'd use a task queue
    const documentId = `${userId}_${Date.now()}`;
    
    try {
      // Use the Python script to process the document
      const cmd = `python ${join(process.cwd(), 'Code/document_processing/process_user_document.py')} --file "${filePath}" --user_id "${userId}" --document_id "${documentId}" --is_public ${isPublic} --notes "${notes}"`;
      
      // Run command asynchronously and don't wait for completion
      execPromise(cmd).catch(err => {
        console.error(`Error processing document: ${err.message}`);
      });
    } catch (err) {
      console.error('Failed to start document processing:', err);
      // We don't fail the request here, as the file upload succeeded
    }
    
    return NextResponse.json({
      success: true,
      message: 'File uploaded successfully',
      documentId,
      filePath: safeFilename,
      processingStatus: 'pending',
      uploadDate: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error('Error uploading file:', error);
    return NextResponse.json(
      { error: 'Failed to upload file' },
      { status: 500 }
    );
  }
}

/**
 * Configuration for file uploads (increased limit for large documents)
 */
export const config = {
  api: {
    bodyParser: false, // Disable the built-in parser
    responseLimit: '10mb',
  },
};