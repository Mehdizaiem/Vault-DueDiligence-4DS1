export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    createdAt: Date;
  }
  
  export interface Conversation {
    messages: Message[];
    lastUpdated?: string;
  }
  
  export interface QAResponse {
    answer: string;
    error?: string;
  }
  
  export interface Feedback {
    messageId: string;
    rating: 'positive' | 'negative';
    comment?: string;
  }