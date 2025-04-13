"use client";

import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Loader2, HelpCircle, CheckCircle, Clock } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import { askQuestion } from '@/services/qa-service';
import { Message } from '@/types/qa';
import ReactMarkdown from 'react-markdown';

export default function QAPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sample questions to help users get started
  const sampleQuestions = [
    "What is the risk score for Bitcoin?",
    "Compare Ethereum and Solana.",
    "What are the regulatory requirements for crypto funds?",
    "How does Tether maintain its peg to USD?",
    "What are the key risk factors for crypto investment funds?"
  ];

  useEffect(() => {
    // Scroll to bottom when messages change
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: inputValue,
      createdAt: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      const response = await askQuestion(userMessage.content);
      
      const assistantMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: response.error || response.answer,
        createdAt: new Date(),
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting answer:', error);
      
      const errorMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: 'Sorry, there was an error processing your question. Please try again.',
        createdAt: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleQuestionClick = (question: string) => {
    setInputValue(question);
  };

  // Function to format the message content with Markdown
  const formatMessage = (content: string) => {
    return (
      <ReactMarkdown
        components={{
          pre: ({ ...props }) => (
            <pre className="bg-gray-100 p-4 rounded-lg overflow-auto my-3 text-sm font-mono" {...props} />
          ),
          code: ({ ...props }) => (
            <code className="bg-gray-100 px-1.5 py-0.5 rounded-md font-mono text-sm" {...props} />
          ),
          h1: ({ ...props }) => (
            <h1 className="text-xl font-bold mt-4 mb-2" {...props} />
          ),
          h2: ({ ...props }) => (
            <h2 className="text-lg font-bold mt-3 mb-2" {...props} />
          ),
          h3: ({ ...props }) => (
            <h3 className="text-md font-bold mt-3 mb-1" {...props} />
          ),
          ul: ({ ...props }) => (
            <ul className="list-disc pl-6 my-2" {...props} />
          ),
          ol: ({ ...props }) => (
            <ol className="list-decimal pl-6 my-2" {...props} />
          ),
          li: ({ ...props }) => (
            <li className="my-1" {...props} />
          ),
          p: ({ ...props }) => (
            <p className="my-2" {...props} />
          ),
          a: ({ ...props }) => (
            <a className="text-blue-600 hover:underline" {...props} />
          ),
          blockquote: ({ ...props }) => (
            <blockquote className="border-l-4 border-gray-300 pl-4 italic my-3" {...props} />
          ),
          table: ({ ...props }) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full divide-y divide-gray-300 border border-gray-300 rounded-lg" {...props} />
            </div>
          ),
          thead: ({ ...props }) => (
            <thead className="bg-gray-100" {...props} />
          ),
          th: ({ ...props }) => (
            <th className="px-4 py-2 text-left text-sm font-medium text-gray-700" {...props} />
          ),
          td: ({ ...props }) => (
            <td className="px-4 py-2 text-sm" {...props} />
          ),
          tr: ({ ...props }) => (
            <tr className="border-b border-gray-300" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  // Format timestamp
  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  return (
    <div className="flex-1 p-8 pt-6 h-[calc(100vh-64px)] flex flex-col">
      <div className="space-y-4 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Q&A System</h2>
            <p className="text-muted-foreground">
              Get instant answers to due diligence queries using our AI-powered system
            </p>
          </div>
          <div className="flex space-x-2">
            <button className="bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-lg border shadow-sm flex items-center space-x-2 text-sm transition-colors">
              <HelpCircle size={16} />
              <span>Help</span>
            </button>
          </div>
        </div>
      </div>

      {messages.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="max-w-lg text-center">
            <div className="bg-blue-50 p-8 rounded-xl mb-6">
              <MessageSquare className="h-12 w-12 text-blue-600 mx-auto mb-4" />
              <h3 className="text-xl font-medium mb-2">Ask a Due Diligence Question</h3>
              <p className="text-gray-600 mb-4">
                Use the Q&A system to get instant answers about cryptocurrency funds,
                regulations, risks, and due diligence procedures.
              </p>
            </div>
            
            <div className="bg-white border shadow-sm rounded-xl p-6">
              <p className="text-sm font-medium text-gray-700 mb-3">Try asking about:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {sampleQuestions.map((question, i) => (
                  <button
                    key={i}
                    className="bg-blue-50 hover:bg-blue-100 text-blue-700 text-sm py-2 px-4 rounded-full transition-colors"
                    onClick={() => handleSampleQuestionClick(question)}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-auto pb-24 mb-4 rounded-xl border bg-gray-50">
          <div className="p-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-6 ${
                  message.role === 'user' ? 'flex justify-end' : 'flex justify-start'
                }`}
              >
                <div
                  className={`flex flex-col max-w-3xl rounded-xl shadow-sm ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white'
                  }`}
                >
                  <div className="px-5 py-3 flex items-center justify-between border-b border-opacity-10 border-gray-300">
                    <div className="flex items-center space-x-2">
                      {message.role === 'user' ? (
                        <div className="bg-blue-700 rounded-full p-1.5">
                          <MessageSquare size={14} />
                        </div>
                      ) : (
                        <div className="bg-blue-100 rounded-full p-1.5">
                          <CheckCircle size={14} className="text-blue-600" />
                        </div>
                      )}
                      <span className={`text-sm font-medium ${message.role === 'user' ? 'text-white' : 'text-gray-700'}`}>
                        {message.role === 'user' ? 'You' : 'AI Assistant'}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Clock size={12} className={message.role === 'user' ? 'text-blue-200' : 'text-gray-400'} />
                      <span className={`text-xs ${message.role === 'user' ? 'text-blue-200' : 'text-gray-400'}`}>
                        {formatTime(message.createdAt)}
                      </span>
                    </div>
                  </div>
                  <div 
                    className={`p-5 ${
                      message.role === 'user' ? 'text-white' : 'text-gray-800 prose prose-sm max-w-none'
                    }`}
                  >
                    {message.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                      <div>
                        {formatMessage(message.content)}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 w-[80%] max-w-4xl">
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-lg border p-4">
          <div className="flex gap-4">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question about crypto due diligence..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              disabled={isLoading}
            />
            <button
              type="submit"
              className={`${
                isLoading ? 'bg-gray-500' : 'bg-blue-600 hover:bg-blue-700'
              } text-white px-6 py-3 rounded-lg flex items-center gap-2 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 size={20} className="animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Send size={20} />
                  <span>Send</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}