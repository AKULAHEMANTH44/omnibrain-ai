import React from 'react';
import { Settings as SettingsIcon, ShieldCheck, Cpu, Key, Globe, Info } from 'lucide-react';

export const SettingsModal: React.FC = () => {
  return (
    <div className="flex-1 flex flex-col h-screen bg-[#07080a] overflow-y-auto p-6 md:p-10">
      <div className="max-w-4xl mx-auto w-full space-y-8">
        {/* Header */}
        <div className="border-b border-white/10 pb-6">
          <div className="flex items-center gap-2 text-slate-400 text-xs font-mono uppercase tracking-widest mb-1">
            <SettingsIcon className="w-4 h-4" />
            <span>OMNIBRAIN Configuration</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Platform Settings & Zero-API Status
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            OMNIBRAIN AI operates 100% locally with zero external API dependencies or external AI provider keys.
          </p>
        </div>

        {/* Settings Sections */}
        <div className="space-y-6">
          {/* Zero API Status */}
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">Zero External AI Guarantee</h3>
                <p className="text-xs text-slate-400">No external APIs, no OpenAI/Anthropic/Gemini keys required</p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#0a0c10] border border-white/10 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300">External API Dependency Status:</span>
              <span className="text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded border border-emerald-500/20">
                Completely Removed (100% Local)
              </span>
            </div>
          </div>

          {/* Platform Info */}
          <div className="bg-[#12151c]/90 border border-white/10 rounded-2xl p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-base">OMNIBRAIN Core Engine</h3>
                <p className="text-xs text-slate-400">Local Transformer v1 Engine</p>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/10 flex justify-between">
                <span className="text-slate-400">Inference Mode:</span>
                <span className="text-white">Local CPU / Node.js</span>
              </div>
              <div className="p-3 rounded-xl bg-[#0a0c10] border border-white/10 flex justify-between">
                <span className="text-slate-400">Tokenizer:</span>
                <span className="text-white">Original ASCII/BPE</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
