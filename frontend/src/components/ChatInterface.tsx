import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Paperclip, FileText, X } from 'lucide-react';
import axios from 'axios';
import clsx from 'clsx';
import mammoth from 'mammoth';
import * as pdfjsLib from 'pdfjs-dist';

// Set worker source for PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface UploadedFile {
  name: string;
  content: string;
}

export const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files;
    if (!fileList) return;

    const newFiles: UploadedFile[] = [];
    
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      let content = "";

      try {
        if (file.type === "application/pdf") {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullText = "";
            for (let j = 1; j <= pdf.numPages; j++) {
                const page = await pdf.getPage(j);
                const textContent = await page.getTextContent();
                // Improved text extraction: join items with space, but respect newlines if items are far apart vertically?
                // For now, just join with space to avoid merging words.
                const pageText = textContent.items.map((item: any) => item.str).join(" ");
                fullText += pageText + "\n\n"; // Double newline between pages
            }
            console.log(`PDF Parsed: ${fullText.length} characters.`);
            content = fullText;
        } else if (file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
            const arrayBuffer = await file.arrayBuffer();
            const result = await mammoth.extractRawText({ arrayBuffer });
            content = result.value;
        } else {
            // Fallback for text files
            content = await file.text();
        }
      } catch (e) {
          console.error("Error parsing file", file.name, e);
          content = "Error parsing file content. Please ensure it is a valid text, PDF, or DOCX file.";
      }

      newFiles.push({ name: file.name, content: content });
    }

    setFiles(prev => [...prev, ...newFiles]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const sendMessage = async () => {
    if (!input.trim() && files.length === 0) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Prepare context from files
    const context: Record<string, any> = {};
    if (files.length > 0) {
      context.documents = {};
      files.forEach(f => {
        context.documents[f.name] = f.content;
      });
    }

    try {
      // Format history for backend
      const chat_history = messages.map(m => ({ role: m.role, content: m.content }));

      const response = await axios.post('http://localhost:8002/agent/run', {
        query: userMessage.content || "Analyze the attached documents.",
        chat_history: chat_history,
        context: context
      });

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.answer,
      };

      setMessages((prev) => [...prev, botMessage]);
      // Optional: Clear files after sending if they are "consumed"
      // setFiles([]); 
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[80vh] w-full max-w-4xl bg-white rounded-2xl shadow-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-white p-4 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-gray-800">LexiBot Assistant</h2>
            <p className="text-xs text-gray-500">AI-Powered Legal Analysis</p>
          </div>
        </div>
        <div className="text-xs text-gray-400">
          {files.length > 0 ? `${files.length} file(s) attached` : 'No files attached'}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/50">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 space-y-4">
            <div className="bg-blue-50 p-4 rounded-full">
              <Bot className="w-12 h-12 text-blue-600" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-700">Welcome to LexiBot</h3>
              <p className="max-w-xs mx-auto mt-2">Upload a contract or ask a legal question to get started.</p>
            </div>
          </div>
        )}
        
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={clsx(
              "flex gap-4 max-w-[85%]",
              msg.role === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
            )}
          >
            <div className={clsx(
              "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm",
              msg.role === 'user' ? "bg-blue-600 text-white" : "bg-white border border-gray-200 text-blue-600"
            )}>
              {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className={clsx(
              "p-4 rounded-2xl text-sm leading-relaxed shadow-sm",
              msg.role === 'user' 
                ? "bg-blue-600 text-white rounded-tr-none" 
                : "bg-white border border-gray-200 text-gray-800 rounded-tl-none"
            )}>
              {msg.content}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex gap-4 mr-auto">
             <div className="w-10 h-10 rounded-full bg-white border border-gray-200 text-blue-600 flex items-center justify-center flex-shrink-0 shadow-sm">
              <Bot size={20} />
            </div>
            <div className="bg-white border border-gray-200 p-4 rounded-2xl rounded-tl-none flex items-center shadow-sm">
              <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
              <span className="ml-2 text-gray-500 text-sm">Analyzing...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-gray-100">
        {/* Attached Files Preview */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3 px-2">
            {files.map((file, idx) => (
              <div key={idx} className="flex items-center gap-2 bg-blue-50 text-blue-700 px-3 py-1.5 rounded-lg text-sm border border-blue-100">
                <FileText size={14} />
                <span className="truncate max-w-[150px]">{file.name}</span>
                <button 
                  onClick={() => removeFile(idx)}
                  className="hover:bg-blue-100 rounded-full p-0.5 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 bg-gray-50 p-2 rounded-xl border border-gray-200 focus-within:border-blue-400 focus-within:ring-4 focus-within:ring-blue-50 transition-all">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
            multiple
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
            title="Attach files"
          >
            <Paperclip size={20} />
          </button>
          
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Type your message..."
            className="flex-1 bg-transparent border-none focus:ring-0 resize-none max-h-32 py-2 text-gray-700 placeholder-gray-400"
            rows={1}
            disabled={isLoading}
            style={{ minHeight: '44px' }}
          />
          
          <button
            onClick={sendMessage}
            disabled={isLoading || (!input.trim() && files.length === 0)}
            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            <Send size={20} />
          </button>
        </div>
        <div className="text-center mt-2">
            <p className="text-xs text-gray-400">AI can make mistakes. Please review generated legal advice.</p>
        </div>
      </div>
    </div>
  );
};
