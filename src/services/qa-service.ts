import { QAResponse } from '@/types/qa';

export async function askQuestion(question: string, documentId?: string): Promise<QAResponse> {
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

    return await response.json();
  } catch (error) {
    console.error('Error asking question:', error);
    return {
      answer: '',
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}