import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Paperclip,
  Bot,
  User,
  Copy,
  Check,
  RotateCcw,
  Sparkles,
  Code,
  FileText,
  Cpu,
  Terminal,
  AlertCircle
} from 'lucide-react';
import { Message } from '../types';

interface ChatWorkspaceProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  isLoading: boolean;
  error: string | null;
  selectedModel: string;
  thinkingLevel: string;
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  messages,
  onSendMessage,
  isLoading,
  error,
  selectedModel,
  thinkingLevel,
}) => {
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const suggestedPrompts = [
    { title: 'Design LLM Architecture', prompt: 'Design a distributed Transformer LLM architecture with multi-head attention and pipeline parallelism.' },
    { title: 'Python RAG Vector Search', prompt: 'Write a production-ready Python script for vector similarity search using embeddings and cosine distance.' },
    { title: 'Quantum Physics Study Notes', prompt: 'Generate comprehensive study notes on quantum entanglement, Bell states, and quantum key distribution.' },
    { title: 'Executive Project Report', prompt: 'Draft a professional executive project report outline for a global AI platform deployment.' },
  ];

  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-hidden relative">
      {/* Chat Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 md:px-12 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-2xl mx-auto space-y-8 animate-fade-in">
            <div className="w-20 h-20 rounded-3xl bg-gradient-to-tr from-cyan-500/20 via-blue-500/20 to-violet-500/30 border border-cyan-500/40 flex items-center justify-center shadow-[0_0_50px_rgba(0,240,255,0.25)]">
              <Sparkles className="w-10 h-10 text-cyan-400 animate-pulse" />
            </div>
            <div className="space-y-3">
              <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-white uppercase font-['Plus_Jakarta_Sans',sans-serif]">
                WHAT CAN I HELP YOU <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-violet-500">CREATE TODAY?</span>
              </h2>
              <p className="text-slate-400 text-sm md:text-base max-w-lg mx-auto">
                OMNIBRAIN AI is your universal intelligence platform. Harness advanced reasoning, code generation, and scientific problem-solving.
              </p>
            </div>

            {/* Suggested Prompts Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full pt-4">
              {suggestedPrompts.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(item.prompt)}
                  className="p-4 rounded-xl bg-[#12151c]/80 hover:bg-[#1a1f2c] border border-white/10 hover:border-cyan-500/40 text-left transition-all group flex flex-col justify-between"
                >
                  <span className="text-sm font-semibold text-white group-hover:text-cyan-400 transition-colors">
                    {item.title}
                  </span>
                  <span className="text-xs text-slate-400 mt-2 line-clamp-2">
                    {item.prompt}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-6 w-full">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-4 p-5 rounded-2xl border transition-all ${
                  msg.role === 'user'
                    ? 'bg-[#12151c]/90 border-white/10 ml-auto max-w-2xl'
                    : 'bg-[#0f1218]/90 border-cyan-500/20 shadow-[0_0_30px_rgba(0,240,255,0.05)] w-full'
                }`}
              >
                <div className="shrink-0 pt-1">
                  {msg.role === 'user' ? (
                    <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                      <User className="w-4 h-4" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded-lg bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-violet-400">
                      <Bot className="w-4 h-4" />
                    </div>
                  )}
                </div>

                <div className="flex-1 space-y-2 overflow-hidden">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                      {msg.role === 'user' ? 'You' : 'OMNIBRAIN AI'}
                    </span>
                    <span className="text-[10px] text-slate-400 font-mono">
                      {msg.timestamp}
                    </span>
                  </div>

                  <div className="text-slate-200 text-sm leading-relaxed whitespace-pre-wrap font-['Plus_Jakarta_Sans',sans-serif]">
                    {msg.content}
                  </div>

                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-3 pt-3 border-t border-white/10 mt-3 text-xs text-slate-400">
                      <button
                        onClick={() => handleCopy(msg.content, msg.id)}
                        className="flex items-center gap-1.5 hover:text-cyan-400 transition-colors"
                      >
                        {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-4 p-5 rounded-2xl bg-[#0f1218]/90 border border-cyan-500/20 w-full animate-pulse">
                <div className="w-8 h-8 rounded-lg bg-violet-500/20 border border-violet-500/40 flex items-center justify-center text-violet-400 shrink-0">
                  <Bot className="w-4 h-4 animate-spin" />
                </div>
                <div className="flex-1 space-y-3 py-1">
                  <div className="h-4 bg-white/10 rounded w-1/4"></div>
                  <div className="h-3 bg-white/10 rounded w-3/4"></div>
                  <div className="h-3 bg-white/10 rounded w-1/2"></div>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 md:mx-12 mb-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 flex items-center gap-3 text-rose-300 text-xs shrink-0">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Intelligent Chat Composer */}
      <div className="p-4 md:p-6 bg-[#0a0c10]/90 backdrop-blur-xl border-t border-white/10 shrink-0">
        <form onSubmit={handleSend} className="max-w-4xl mx-auto relative">
          <div className="flex items-center gap-2 bg-[#12151c] border border-white/10 focus-within:border-cyan-500/60 rounded-2xl p-2.5 shadow-2xl transition-all">
            <button
              type="button"
              onClick={() => alert('File attachment simulation: Upload documents, PDFs, or code files for neural analysis.')}
              className="p-2 text-slate-400 hover:text-cyan-400 hover:bg-white/5 rounded-xl transition-colors"
              title="Attach File"
            >
              <Paperclip className="w-5 h-5" />
            </button>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(e);
                }
              }}
              placeholder="Ask OMNIBRAIN anything (Press Enter to send)..."
              rows={1}
              className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 text-sm focus:outline-none resize-none px-2 max-h-32"
            />

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={`p-3 rounded-xl font-bold transition-all duration-200 flex items-center justify-center ${
                input.trim() && !isLoading
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-black shadow-[0_0_20px_rgba(0,240,255,0.3)] cursor-pointer hover:scale-105'
                  : 'bg-white/5 text-slate-600 cursor-not-allowed'
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center justify-between px-3 pt-2 text-[11px] text-slate-500 font-mono">
            <span>Model: {selectedModel} ({thinkingLevel})</span>
            <span>OMNIBRAIN AI v2.5 Universal Intelligence</span>
          </div>
        </form>
      </div>
    </div>
  );
};
