import React from 'react';
import { Sparkles, Cpu, ShieldCheck, User } from 'lucide-react';

interface TopBarProps {
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  thinkingLevel: string;
  setThinkingLevel: (level: string) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  selectedModel,
  setSelectedModel,
  thinkingLevel,
  setThinkingLevel,
}) => {
  return (
    <header className="h-16 bg-[#0a0c10]/80 backdrop-blur-xl border-b border-white/10 px-6 flex items-center justify-between gap-8 shrink-0 z-10">
      {/* Zone 1: Brand / Context Title */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold tracking-tight text-white uppercase font-['Plus_Jakarta_Sans',sans-serif]">
            OMNIBRAIN AI
          </span>
          <span className="text-xs text-cyan-400 font-mono px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
            Zero-API Local Core
          </span>
        </div>
      </div>

      {/* Zone 2: Model & Engine Controls */}
      <div className="hidden md:flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2 bg-[#12151c] border border-white/10 rounded-lg px-3 py-1.5">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">Local Engine:</span>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="bg-transparent text-white font-medium focus:outline-none cursor-pointer"
          >
            <option value="omnibrain-transformer-v1" className="bg-[#12151c] text-white">OMNIBRAIN Transformer v1 (Local)</option>
          </select>
        </div>

        <div className="flex items-center gap-2 bg-[#12151c] border border-white/10 rounded-lg px-3 py-1.5">
          <Sparkles className="w-3.5 h-3.5 text-violet-400" />
          <span className="text-slate-400">Mode:</span>
          <span className="text-white font-medium">CPU Local Inference</span>
        </div>
      </div>

      {/* Zone 3: Primary Action / User Profile */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="hidden sm:flex items-center gap-2 text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-lg">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>100% Offline / Zero API Keys</span>
        </div>
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-600 p-0.5 flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.3)]">
          <div className="w-full h-full bg-[#0a0c10] rounded-[10px] flex items-center justify-center">
            <User className="w-4 h-4 text-cyan-400" />
          </div>
        </div>
      </div>
    </header>
  );
};
