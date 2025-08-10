import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { FiSend, FiMessageSquare, FiChevronDown } from 'react-icons/fi';
import { IoSend } from 'react-icons/io5';

const ChatbotWidget = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    {
      text: "Hi there! I'm your FAQ assistant. Ask me anything about our services.",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Auto-scroll to bottom with smooth animation
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    setShowScrollButton(false);
  };

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      setShowScrollButton(scrollTop < scrollHeight - clientHeight - 50);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage = { 
      text: input, 
      sender: 'user',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await axios.post(
        process.env.REACT_APP_API_URL || 'http://localhost:8000/query', 
        { text: input }
      );
      
      setMessages(prev => [...prev, {
        text: response.data.answer,
        sender: 'bot',
        timestamp: new Date()
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        text: "Sorry, I'm having trouble responding. Please try again later.",
        sender: 'bot',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 w-96 bg-white shadow-xl rounded-xl overflow-hidden flex flex-col"
         style={{ 
           height: '70vh', 
           maxHeight: '600px', 
           zIndex: 1000, 
           fontFamily: 'system-ui, -apple-system, sans-serif',
           transform: 'translate3d(0,0,0)' /* Hardware acceleration */
         }}>
      
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 p-4 text-white flex items-center shrink-0">
        <div className="bg-white/20 p-2 rounded-full mr-3">
          <FiMessageSquare size={20} />
        </div>
        <div>
          <h2 className="font-semibold text-lg">Apex Financial Support</h2>
          <p className="text-xs opacity-90">Typically replies instantly</p>
        </div>
      </div>
      
      {/* Messages Container with scroll */}
      <div 
        ref={chatContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-gray-50 relative"
        style={{
          scrollBehavior: 'smooth',
          overscrollBehavior: 'contain' /* Prevent scroll chaining */
        }}
      >
        <div className="p-4 space-y-2">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div 
                className={`max-w-[80%] px-4 py-2 rounded-2xl ${msg.sender === 'user' 
                  ? 'bg-blue-500 text-white rounded-tr-none' 
                  : 'bg-gray-100 text-gray-800 rounded-tl-none'}`}
                style={{
                  boxShadow: msg.sender === 'user' 
                    ? '0 2px 4px rgba(29, 78, 216, 0.2)'
                    : '0 2px 4px rgba(0, 0, 0, 0.05)'
                }}
              >
                <div className="text-sm">{msg.text}</div>
                <div className={`text-xs mt-1 text-right ${msg.sender === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 px-4 py-2 rounded-2xl rounded-tl-none">
                <div className="flex space-x-1.5">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Scroll to bottom button */}
        {showScrollButton && (
          <button
            onClick={scrollToBottom}
            className="absolute bottom-4 right-4 bg-white p-2 rounded-full shadow-md border border-gray-200 hover:bg-gray-50 transition-colors"
            aria-label="Scroll to bottom"
          >
            <FiChevronDown className="text-gray-600" />
          </button>
        )}
      </div>
      
      {/* Input Area */}
      <div className="border-t border-gray-200 p-3 bg-white shrink-0">
        <div className="flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSend()}
            placeholder="Type a message..."
            className="flex-1 border border-gray-300 rounded-full py-2 px-4 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className={`ml-2 p-2 rounded-full ${isLoading || !input.trim() ? 'text-gray-400' : 'text-blue-500 hover:text-blue-600'}`}
          >
            <IoSend size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatbotWidget;