import React, { useState } from 'react';
import {
  MessageSquarePlus,
  Search,
  MessageSquare,
  Sparkles,
  GraduationCap,
  Code2,
  Database,
  Cpu,
  Settings as SettingsIcon,
  Trash2,
  Brain,
  ChevronRight
} from 'lucide-react';
import { NavigationTab, Conversation } from '../types';

interface SidebarProps {
  currentTab: NavigationTab;
  setTab: (tab: NavigationTab) => void;
  conversations: Conversation[];
  currentConversationId: string;
  selectConversation: (id: string) => void;
  newChat: () => void;
  deleteConversation: (id: string, e: React.MouseEvent) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  setTab,
  conversations,
  currentConversationId,
  selectConversation,
  newChat,
  deleteConversation,
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredConversations = conversations.filter(c =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <aside className="w-72 bg-[#0a0c10] border-r border-white/10 flex flex-col h-screen select-none shrink-0 z-20">
      {/* Brand Header */}
      <div className="p-5 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500/20 via-blue-500/20 to-violet-500/30 border border-cyan-500/40 flex items-center justify-center shadow-[0_0_20px_rgba(0,240,255,0.2)]">
            <Brain className="w-6 h-6 text-cyan-400 animate-pulse" />
          </div>
          <div>
            <h1 className="font-extrabold text-lg tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-100 to-cyan-400">
              OMNIBRAIN
            </h1>
            <span className="text-[10px] tracking-widest uppercase text-cyan-400 font-semibold">
              Neural Intelligence
            </span>
          </div>
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={newChat}
          className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold text-sm shadow-[0_0_25px_rgba(0,240,255,0.3)] transition-all duration-200 transform active:scale-[0.98]"
        >
          <MessageSquarePlus className="w-4 h-4 text-black" />
          <span>New Neural Chat</span>
        </button>
      </div>

      {/* Search Conversations */}
      <div className="px-4 pb-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#12151c] border border-white/10 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition-colors"
          />
        </div>
      </div>

      {/* Navigation Studios & Libraries */}
      <div className="px-3 py-2 space-y-1 border-b border-white/10">
        <div className="px-3 py-1 text-[10px] font-semibold tracking-wider uppercase text-slate-400">
          Workspaces
        </div>

        <button
          onClick={() => setTab('chat')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'chat'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <MessageSquare className="w-4 h-4 text-cyan-400" />
            <span>Universal Chat</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 opacity-40" />
        </button>

        <button
          onClick={() => setTab('create')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'create'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Sparkles className="w-4 h-4 text-violet-400" />
            <span>AI Create Studio</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 opacity-40" />
        </button>

        <button
          onClick={() => setTab('learning')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'learning'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <GraduationCap className="w-4 h-4 text-emerald-400" />
            <span>Learning Studio</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 opacity-40" />
        </button>

        <button
          onClick={() => setTab('code')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'code'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Code2 className="w-4 h-4 text-blue-400" />
            <span>Code Intelligence</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 opacity-40" />
        </button>

        <button
          onClick={() => setTab('knowledge')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'knowledge'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Database className="w-4 h-4 text-amber-400" />
            <span>Knowledge Library</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 opacity-40" />
        </button>

        <button
          onClick={() => setTab('ml')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'ml'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <Cpu className="w-4 h-4 text-rose-400" />
            <span>ML Architecture</span>
          </div>
          <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded font-mono">Core</span>
        </button>
      </div>

      {/* Recent Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        <div className="px-3 py-1 text-[10px] font-semibold tracking-wider uppercase text-slate-400">
          Recent Conversations
        </div>
        {filteredConversations.length === 0 ? (
          <div className="px-3 py-4 text-xs text-slate-500 text-center">
            No conversations found.
          </div>
        ) : (
          filteredConversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => {
                setTab('chat');
                selectConversation(conv.id);
              }}
              className={`group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer text-xs transition-all ${
                currentTab === 'chat' && currentConversationId === conv.id
                  ? 'bg-white/10 text-white font-medium'
                  : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
              }`}
            >
              <div className="flex items-center gap-2.5 truncate">
                <MessageSquare className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span className="truncate">{conv.title}</span>
              </div>
              <button
                onClick={(e) => deleteConversation(conv.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-400 transition-opacity"
                title="Delete conversation"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Footer / Settings Link */}
      <div className="p-3 border-t border-white/10">
        <button
          onClick={() => setTab('settings')}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
            currentTab === 'settings'
              ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30'
              : 'text-slate-300 hover:bg-white/5 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <SettingsIcon className="w-4 h-4 text-slate-400" />
            <span>Platform Settings</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">v2.5</span>
        </button>
      </div>
    </aside>
  );
};
