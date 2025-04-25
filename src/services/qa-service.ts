import { QAResponse } from '@/types/qa';
import { v4 as uuidv4 } from 'uuid';

// Track session ID for grouping related questions
let sessionId = uuidv4();

export async function askQuestion(question: string, documentId?: string): Promise<QAResponse> {
  const startTime = Date.now();
  
  try {
    const response = await fetch('/api/qa', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question, documentId }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get answer');
    }

    const result = await response.json();
    const durationMs = Date.now() - startTime;
    
    // Store the Q&A interaction for reporting
    try {
      await storeQAHistory(question, result.answer, documentId, durationMs);
    } catch (historyError) {
      console.error('Error storing QA history:', historyError);
      // Don't fail the main operation if history storage fails
    }

    return result;
  } catch (error) {
    console.error('Error asking question:', error);
    return {
      answer: '',
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

// Function to store Q&A history
async function storeQAHistory(
  question: string, 
  answer: string, 
  documentId?: string,
  durationMs?: number
): Promise<void> {
  try {
    // Prepare the document IDs array
    const documentIds = documentId ? [documentId] : undefined;
    
    // Make request to history API
    const response = await fetch('/api/qa/history', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        answer,
        documentIds,
        sessionId, // Use the session ID to group related questions
        durationMs
      }),
    });

    if (!response.ok) {
      console.error('Failed to store QA history:', await response.text());
    }
  } catch (error) {
    console.error('Error storing QA history:', error);
    // Don't throw to avoid affecting the main flow
  }
}

// Export function to create a new session
export function createNewQASession(): void {
  sessionId = uuidv4();
}