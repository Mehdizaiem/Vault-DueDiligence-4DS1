export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    createdAt: Date;
  }
  
  export interface Conversation {
    messages: Message[];
  }
  
  export interface QAResponse {
    answer: string;
    error?: string;
  }