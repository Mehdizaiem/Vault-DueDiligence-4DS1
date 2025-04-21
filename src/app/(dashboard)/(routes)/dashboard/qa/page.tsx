"use client";

import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Loader2, HelpCircle, CheckCircle, Clock, ThumbsUp, ThumbsDown, Trash2 } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import { askQuestion } from '@/services/qa-service';
import { Message, Conversation } from '@/types/qa';
import ReactMarkdown from 'react-markdown';

// Define an interface for feedback
interface Feedback {
  messageId: string;
  rating: 'positive' | 'negative';
  comment?: string;
}

export default function QAPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, Feedback>>({});
  const [conversationId, setConversationId] = useState<string>('');
  const [savedConversations, setSavedConversations] = useState<Record<string, Conversation>>({});
  const [showConversations, setShowConversations] = useState(false);
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
    // Load conversation history if we have a conversation ID
    const loadConversation = () => {
      try {
        const currentConversation = localStorage.getItem(`qa_conversation_${conversationId}`);
        if (currentConversation && messages.length === 0) {
          const parsed = JSON.parse(currentConversation);
          setMessages(parsed.messages.map((msg: Message) => ({
            ...msg,
            createdAt: new Date(msg.createdAt as unknown as string)
          })));
          
          // Load feedback
          const savedFeedback = localStorage.getItem(`qa_feedback_${conversationId}`);
          if (savedFeedback) {
            setFeedback(JSON.parse(savedFeedback));
          }
        }
      } catch (error) {
        console.error('Error loading conversation from localStorage:', error);
      }
    };
    
    loadConversation();
    
    // Check for document context in URL
    const searchParams = new URLSearchParams(window.location.search);
    const documentId = searchParams.get('documentId');
    
    if (documentId && messages.length === 0) {
      // Automatically add a hint about the document
      setMessages([{
        id: uuidv4(),
        role: 'assistant',
        content: "I'm ready to answer questions about the document you've selected. What would you like to know?",
        createdAt: new Date(),
      }]);
    }
  }, [conversationId, messages.length]);
  
  useEffect(() => {
    // Scroll to bottom when messages change only if we're close to the bottom already
    // or if it's a new message
    if (messagesEndRef.current) {
      const container = messagesEndRef.current.parentElement?.parentElement;
      if (container) {
        const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
        const isNewMessage = messages.length > 0 && messages[messages.length - 1].createdAt.getTime() > Date.now() - 1000;
        
        if (isNearBottom || isNewMessage) {
          messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }
    
    // Save conversation to local storage when messages change
    if (messages.length > 0 && conversationId) {
      try {
        // Save current conversation
        const conversation: Conversation = {
          messages: messages
        };
        
        localStorage.setItem(`qa_conversation_${conversationId}`, JSON.stringify(conversation));
        
        // Update the saved conversations list
        const updatedConversations = {
          ...savedConversations,
          [conversationId]: {
            messages: messages.slice(0, 1), // Just store the first message as preview
            lastUpdated: new Date().toISOString()
          }
        };
        
        localStorage.setItem('qa_conversations', JSON.stringify(updatedConversations));
        setSavedConversations(updatedConversations);
        
        // Save feedback
        if (Object.keys(feedback).length > 0) {
          localStorage.setItem(`qa_feedback_${conversationId}`, JSON.stringify(feedback));
        }
      } catch (error) {
        console.error('Error saving to localStorage:', error);
      }
    }
  }, [messages, conversationId, feedback, savedConversations]);

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
      // Get the document ID from URL if present
      const searchParams = new URLSearchParams(window.location.search);
      const documentId = searchParams.get('documentId');
      
      const response = await askQuestion(userMessage.content, documentId || undefined);
      
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
  
  const handleFeedback = (messageId: string, rating: 'positive' | 'negative') => {
    // First update the local state for immediate visual feedback
    setFeedback(prev => ({
      ...prev,
      [messageId]: {
        messageId,
        rating,
        comment: ''
      }
    }));
    
    // Then submit the feedback to be stored in the QA history
    submitFeedback(messageId, rating);
  };
  
  const startNewConversation = () => {
    const newId = uuidv4();
    setConversationId(newId);
    setMessages([]);
    setFeedback({});
  };
  
  const loadConversation = (id: string) => {
    try {
      const savedConversation = localStorage.getItem(`qa_conversation_${id}`);
      if (savedConversation) {
        const parsed = JSON.parse(savedConversation);
        setMessages(parsed.messages.map((msg: Message) => ({
          ...msg,
          createdAt: new Date(msg.createdAt as unknown as string)
        })));
        
        // Load feedback
        const savedFeedback = localStorage.getItem(`qa_feedback_${id}`);
        if (savedFeedback) {
          setFeedback(JSON.parse(savedFeedback));
        } else {
          setFeedback({});
        }
        
        setConversationId(id);
        setShowConversations(false);
      }
    } catch (error) {
      console.error('Error loading conversation:', error);
    }
  };
  
  const deleteConversation = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      // Remove from localStorage
      localStorage.removeItem(`qa_conversation_${id}`);
      localStorage.removeItem(`qa_feedback_${id}`);
      
      // Update savedConversations state
      const updatedConversations = { ...savedConversations };
      delete updatedConversations[id];
      
      localStorage.setItem('qa_conversations', JSON.stringify(updatedConversations));
      setSavedConversations(updatedConversations);
      
      // If we're deleting the current conversation, start a new one
      if (id === conversationId) {
        startNewConversation();
      }
    } catch (error) {
      console.error('Error deleting conversation:', error);
    }
  };
  // Function to submit feedback
const submitFeedback = async (messageId: string, rating: 'positive' | 'negative', comment?: string) => {
  try {
    // Find the message
    const message = messages.find(m => m.id === messageId);
    if (!message || message.role !== 'assistant') return;
    
    // Find the corresponding user question
    const questionIndex = messages.findIndex(m => m.id === messageId);
    if (questionIndex <= 0) return; // Make sure there's a question before this answer
    
    const userQuestion = messages[questionIndex - 1];
    if (userQuestion.role !== 'user') return;
    
    // Store feedback in QA history
    const response = await fetch('/api/qa/history', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: userQuestion.content,
        answer: message.content,
        sessionId: conversationId,
        feedback: {
          rating: rating === 'positive' ? 5 : 1, // Convert to numerical rating
          comment
        }
      }),
    });

    if (!response.ok) {
      console.error('Failed to store feedback:', await response.text());
    }
    
    // Update local feedback state
    setFeedback(prev => ({
      ...prev,
      [messageId]: {
        messageId,
        rating,
        comment
      }
    }));
    
  } catch (error) {
    console.error('Error submitting feedback:', error);
  }
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
      <div className="space-y-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Q&A System</h2>
            <p className="text-muted-foreground">
              Get instant answers to due diligence queries using our AI-powered system
            </p>
          </div>
          <div className="flex space-x-2">
            <button 
              onClick={() => setShowConversations(prev => !prev)}
              className="bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-lg border shadow-sm flex items-center space-x-2 text-sm transition-colors"
            >
              <MessageSquare size={16} />
              <span>History</span>
            </button>
            <button 
              onClick={startNewConversation}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg shadow-sm flex items-center space-x-2 text-sm transition-colors"
            >
              <MessageSquare size={16} />
              <span>New Chat</span>
            </button>
            <button className="bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-lg border shadow-sm flex items-center space-x-2 text-sm transition-colors">
              <HelpCircle size={16} />
              <span>Help</span>
            </button>
          </div>
        </div>
        
        {/* Conversation History Dropdown */}
        {showConversations && (
          <div className="absolute right-8 mt-2 z-10 bg-white rounded-xl shadow-lg border p-4 w-80">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-medium text-gray-700">Conversation History</h3>
              <button 
                onClick={() => setShowConversations(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                &times;
              </button>
            </div>
            {Object.keys(savedConversations).length > 0 ? (
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {Object.entries(savedConversations).map(([id, conversation]) => (
                  <div 
                    key={id}
                    onClick={() => loadConversation(id)}
                    className={`p-3 rounded-lg cursor-pointer flex justify-between items-start hover:bg-gray-50 ${
                      id === conversationId ? 'bg-blue-50 border border-blue-200' : 'border'
                    }`}
                  >
                    <div className="truncate flex-1">
                      <p className="font-medium text-sm truncate">
                        {conversation.messages[0]?.content.substring(0, 40)}...
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {typeof conversation.lastUpdated === 'string' ? new Date(conversation.lastUpdated).toLocaleString() : 'Unknown date'}
                      </p>
                    </div>
                    <button
                      onClick={(e) => deleteConversation(id, e)}
                      className="text-gray-400 hover:text-red-500 ml-2"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm text-center py-4">No saved conversations</p>
            )}
          </div>
        )}
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
        <div className="flex-1 overflow-y-auto pb-24 mb-4 rounded-xl border bg-gray-50" style={{ maxHeight: 'calc(100vh - 230px)' }}>
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
                        
                        {/* Feedback mechanism */}
                        <div className="mt-4 pt-3 border-t border-gray-200 flex items-center justify-between">
                          <div className="text-sm text-gray-500">
                            Was this answer helpful?
                          </div>
                          <div className="flex space-x-2">
                            <button 
                              onClick={() => handleFeedback(message.id, 'positive')}
                              className={`p-1.5 rounded-full ${
                                feedback[message.id]?.rating === 'positive' 
                                  ? 'bg-green-100 text-green-600' 
                                  : 'text-gray-400 hover:text-green-600 hover:bg-green-50'
                              }`}
                              aria-label="Thumbs up"
                            >
                              <ThumbsUp size={16} />
                            </button>
                            <button
                              onClick={() => handleFeedback(message.id, 'negative')}
                              className={`p-1.5 rounded-full ${
                                feedback[message.id]?.rating === 'negative' 
                                  ? 'bg-red-100 text-red-600' 
                                  : 'text-gray-400 hover:text-red-600 hover:bg-red-50'
                              }`}
                              aria-label="Thumbs down"
                            >
                              <ThumbsDown size={16} />
                            </button>
                          </div>
                        </div>
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